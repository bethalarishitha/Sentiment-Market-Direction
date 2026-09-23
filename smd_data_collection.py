"""Sentiment-Driven Market Direction Prediction  - Sprint 1, Section 1: Data Collection.

Collects data to classify the NEXT-PERIOD DIRECTION (up/down) of a market index
from behavioural-finance signals:

  1. Index OHLCV (^GSPC, QQQ)   - daily (5y) + hourly (60d); source of the target
  2. Large-cap basket           - daily; for breadth / herding signals
  3. VIX (^VIX)                  - fear / greed sentiment
  4. Put/Call ratio (SPY chain)  - options-based sentiment snapshot
  5. News headlines + VADER      - text sentiment (best-effort)

Outputs CSVs to smd_data/raw_data/ + manifest.json + docs/data_catalogue (separately).

Run:  python src/smd_data_collection.py
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import smd_config as C


def _make_session():
    """Browser-impersonating session via curl_cffi to dodge Yahoo rate-limits.
    Falls back to None (yfinance default) if curl_cffi is unavailable."""
    try:
        from curl_cffi import requests as cr
        return cr.Session(impersonate="chrome")
    except Exception:  # noqa: BLE001
        return None


SESSION = _make_session()


def _retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001
            last_err = err
            wait = C.RETRY_SLEEP_SEC * (2 ** (attempt - 1))  # exponential backoff
            print(f"   ! attempt {attempt}/{C.MAX_RETRIES} failed: {err} (wait {wait}s)")
            time.sleep(wait)
    print(f"   x giving up: {last_err}")
    return None


def _stooq_daily(ticker: str) -> pd.DataFrame | None:
    sym = C.STOOQ_SYMBOLS.get(ticker)
    if not sym:
        return None
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        if not text or text.strip().lower().startswith("<"):
            return None
        df = pd.read_csv(io.StringIO(text))
        if "Date" not in df.columns or len(df) == 0:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index()
    except Exception as err:  # noqa: BLE001
        print(f"   ! stooq fallback failed for {ticker}: {err}")
        return None


def _download_daily(ticker: str) -> tuple[pd.DataFrame | None, str]:
    df = _retry(yf.download, ticker, period=C.DAILY_PERIOD, interval="1d",
                progress=False, auto_adjust=True, session=SESSION)
    if df is not None and len(df) > 0:
        return df, "yahoo"
    print(f"   ~ Yahoo empty for {ticker}; trying Stooq fallback")
    return _stooq_daily(ticker), "stooq"


def _save(df: pd.DataFrame | None, name: str, manifest: list, source: str = "yahoo") -> None:
    if df is None or len(df) == 0:
        print(f"   - {name}: no data")
        manifest.append({"dataset": name, "rows": 0, "status": "empty"})
        return
    path = C.RAW_DIR / f"{name}.csv"
    df.to_csv(path)
    print(f"   OK {name}: {len(df)} rows (source={source}) -> {path.name}")
    manifest.append(
        {
            "dataset": name,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns))[:25],
            "file": path.name,
            "source": source,
            "status": "ok",
        }
    )


def collect_index(manifest: list) -> None:
    for tk in (C.INDEX_TICKER, C.SECONDARY_INDEX):
        print(f" - index {tk}")
        daily, src = _download_daily(tk)
        _save(daily, f"{tk.strip('^')}_daily", manifest, source=src)
        time.sleep(C.INTER_CALL_SLEEP)
        hourly = _retry(yf.download, tk, period=C.HOURLY_PERIOD, interval="1h",
                        progress=False, auto_adjust=True, session=SESSION)
        _save(hourly, f"{tk.strip('^')}_hourly", manifest, source="yahoo")
        time.sleep(C.INTER_CALL_SLEEP)


def collect_basket(manifest: list) -> None:
    """Daily closes for the whole basket in ONE multi-ticker call (fewer requests
    => far less likely to be rate-limited). Wide frame for breadth/herding."""
    print(" - large-cap basket (breadth / herding) [single batched call]")
    data = _retry(yf.download, C.BASKET, period=C.DAILY_PERIOD, interval="1d",
                  progress=False, auto_adjust=True, group_by="column", session=SESSION)
    basket = None
    if data is not None and len(data) > 0:
        if isinstance(data.columns, pd.MultiIndex):
            basket = data["Close"].copy()
        else:  # single ticker edge case
            basket = data[["Close"]].copy()
        print(f"   . basket: {basket.shape[1]} tickers x {len(basket)} rows")
    _save(basket, "basket_closes_daily", manifest, source="yahoo")


def collect_context(manifest: list) -> None:
    print(" - VIX (fear/greed)")
    vix, src = _download_daily(C.VIX_TICKER)
    _save(vix, "VIX_daily", manifest, source=src)


def collect_putcall(manifest: list) -> None:
    """Put/Call snapshot from SPY option chain plus daily CBOE ratio history (PCCE)."""
    print(f" - put/call ratio from {C.PUTCALL_UNDERLYING} options and {C.PUTCALL_DAILY_TICKER}")
    tk = yf.Ticker(C.PUTCALL_UNDERLYING, session=SESSION)
    expiries = _retry(lambda: tk.options) or []
    rows = []
    for exp in expiries[:4]:
        chain = _retry(lambda e=exp: tk.option_chain(e))
        if chain is None:
            continue
        call_oi, put_oi = chain.calls["openInterest"].sum(), chain.puts["openInterest"].sum()
        call_vol, put_vol = chain.calls["volume"].sum(), chain.puts["volume"].sum()
        rows.append(
            {
                "expiry": exp,
                "call_oi": float(call_oi),
                "put_oi": float(put_oi),
                "putcall_oi_ratio": float(put_oi) / float(call_oi) if call_oi else None,
                "call_vol": float(call_vol),
                "put_vol": float(put_vol),
                "putcall_vol_ratio": float(put_vol) / float(call_vol) if call_vol else None,
            }
        )
        print(f"   . {exp}: P/C OI ratio = {rows[-1]['putcall_oi_ratio']}")
    df = pd.DataFrame(rows)
    _save(df, "putcall_ratio_snapshot", manifest, source="yahoo")

    daily, src = _download_daily(C.PUTCALL_DAILY_TICKER)
    if daily is not None and len(daily) > 0:
        close_col = "Close" if "Close" in daily.columns else daily.columns[0]
        pc = daily[[close_col]].copy()
        pc.columns = ["putcall_ratio"]
        pc.index = pd.to_datetime(pc.index).normalize()
        pc = pc.sort_index().dropna()
        path = C.PUTCALL_DAILY_FILE
        pc.to_csv(path)
        print(f"   OK putcall_ratio_daily: {len(pc)} rows (source={src}) -> {path.name}")
        manifest.append(
            {
                "dataset": "putcall_ratio_daily",
                "rows": int(len(pc)),
                "file": path.name,
                "source": src,
                "status": "ok",
            }
        )
    else:
        print("   - putcall_ratio_daily: no PCCE history from Yahoo")
        manifest.append({"dataset": "putcall_ratio_daily", "rows": 0, "status": "empty"})


def _news_published_at(item: dict) -> pd.Timestamp | None:
    """Best effort publish time from a Yahoo news item."""
    content = item.get("content", item)
    for key in ("pubDate", "displayTime", "providerPublishTime"):
        raw = content.get(key) if isinstance(content, dict) else None
        if raw is None:
            raw = item.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return pd.to_datetime(raw, unit="s", utc=True, errors="coerce")
        return pd.to_datetime(raw, utc=True, errors="coerce")
    return None


def _append_headline_archive(headlines: pd.DataFrame) -> pd.DataFrame:
    """Keep one growing file so news history builds up across collection runs."""
    if headlines is None or len(headlines) == 0:
        if C.NEWS_HEADLINES_ARCHIVE.exists():
            return pd.read_csv(C.NEWS_HEADLINES_ARCHIVE)
        return headlines
    new = headlines.copy()
    if C.NEWS_HEADLINES_ARCHIVE.exists():
        old = pd.read_csv(C.NEWS_HEADLINES_ARCHIVE)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["title"] = combined["title"].astype(str).str.strip()
    combined = combined[combined["title"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["title"], keep="last")
    combined.to_csv(C.NEWS_HEADLINES_ARCHIVE, index=False)
    print(f"   . headline archive: {len(combined)} unique titles")
    return combined


def _daily_from_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    hl = headlines.copy()
    hl["published_utc"] = pd.to_datetime(hl["published_utc"], utc=True, errors="coerce")
    hl = hl[hl["published_utc"].notna()]
    if len(hl) == 0:
        return pd.DataFrame(columns=["news_sentiment", "news_count"])
    hl["date"] = hl["published_utc"].dt.tz_convert("America/New_York").dt.normalize()
    daily = (
        hl.groupby("date")
        .agg(news_sentiment=("vader_compound", "mean"), news_count=("title", "count"))
        .sort_index()
    )
    daily.index.name = "Date"
    return daily


def collect_news_rss(manifest: list, sia) -> pd.DataFrame:
    """Extra headlines from public RSS feeds (helps fill more calendar days)."""
    try:
        import feedparser  # noqa: PLC0415
    except Exception as err:  # noqa: BLE001
        print(f"   - RSS skipped ({err})")
        return pd.DataFrame()
    rows: list[dict] = []
    for url in C.NEWS_RSS_FEEDS:
        print(f"   . RSS {url[:60]}...")
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:80]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                pub_ts = pd.Timestamp(
                    year=pub[0], month=pub[1], day=pub[2], tz="UTC",
                )
            else:
                pub_ts = pd.NaT
            rows.append(
                {
                    "title": title,
                    "ticker": "RSS",
                    "published_utc": pub_ts.isoformat() if pd.notna(pub_ts) else "",
                    "vader_compound": sia.polarity_scores(title)["compound"],
                    "source": "rss",
                }
            )
        time.sleep(1)
    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"   . RSS headlines fetched: {len(df)}")
    manifest.append({"dataset": "news_rss", "rows": int(len(df)), "status": "ok" if len(df) else "empty"})
    return df


def collect_news_sentiment(manifest: list) -> None:
    """Collect many headlines from SPY and the basket, score with VADER, aggregate by day.

    Feedback was that news had too few rows. Yahoo only exposes recent headlines, but
    pulling several tickers gives a much larger pool and a proper daily series.
    """
    print(" - news sentiment (SPY plus basket, many headlines)")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as err:  # noqa: BLE001
        print(f"   - VADER unavailable ({err}); skipping")
        manifest.append({"dataset": "news_sentiment", "rows": 0, "status": "skipped"})
        return
    sia = SentimentIntensityAnalyzer()
    seen_titles: set[str] = set()
    rows: list[dict] = []
    for tk in C.NEWS_TICKERS:
        print(f"   . fetching news for {tk}")
        news = _retry(lambda t=tk: yf.Ticker(t, session=SESSION).news) or []
        for item in news:
            content = item.get("content", item)
            title = (content.get("title") if isinstance(content, dict) else None) or item.get("title") or ""
            title = str(title).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            pub = _news_published_at(item)
            rows.append(
                {
                    "title": title,
                    "ticker": tk,
                    "published_utc": pub.isoformat() if pub is not None and pd.notna(pub) else "",
                    "vader_compound": sia.polarity_scores(title)["compound"],
                    "source": "yahoo",
                }
            )
        time.sleep(C.INTER_CALL_SLEEP)

    rss_rows = collect_news_rss(manifest, sia)
    if len(rss_rows) > 0:
        rows.extend(rss_rows.to_dict("records"))

    headlines = pd.DataFrame(rows)
    archive = _append_headline_archive(headlines)

    if archive is None or len(archive) == 0:
        manifest.append({"dataset": "news_sentiment_daily_merged", "rows": 0, "status": "empty"})
        return

    daily = _daily_from_headlines(archive)
    daily.to_csv(C.NEWS_DAILY_FILE)
    print(f"   OK news daily (from archive): {len(daily)} days -> {C.NEWS_DAILY_FILE.name}")
    manifest.append(
        {
            "dataset": "news_sentiment_daily_merged",
            "rows": int(len(daily)),
            "file": C.NEWS_DAILY_FILE.name,
            "source": "yahoo+rss",
            "status": "ok",
        }
    )
    manifest.append(
        {
            "dataset": "news_sentiment_headlines_archive",
            "rows": int(len(archive)),
            "file": C.NEWS_HEADLINES_ARCHIVE.name,
            "source": "yahoo+rss",
            "status": "ok",
        }
    )


def main() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Sentiment-Driven Market Direction Prediction  - Sprint 1.1 Data Collection")
    print("=" * 70)

    manifest: list = []
    for step in (collect_index, collect_basket, collect_context, collect_putcall, collect_news_sentiment):
        step(manifest)
        time.sleep(C.INTER_CALL_SLEEP)

    meta = {
        "project": "Sentiment-Driven Market Direction Prediction ",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "index": C.INDEX_TICKER,
        "datasets": manifest,
    }
    (C.RAW_DIR / "manifest.json").write_text(json.dumps(meta, indent=2))
    print("\nManifest written to smd_data/raw_data/manifest.json")

    ok = sum(1 for m in manifest if m.get("status") == "ok")
    print(f"Done. {ok}/{len(manifest)} datasets collected with data.")


if __name__ == "__main__":
    main()
