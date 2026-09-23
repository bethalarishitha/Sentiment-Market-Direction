"""Dynamic stock direction: download → features → train → next-day / next-week signal.

Used by the dashboard for:
  - Answer tab: all dataset stocks (GSPC, QQQ, basket)
  - Dynamic tab: user-entered ticker

Run (batch all dataset stocks):
    python src/smd_dynamic_predict.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

import smd_config as C
from smd_features import MOOD_FEATURES, PRICE_FEATURES, TARGET
from smd_model_comparison import _try_smote, maybe_smote, time_split
from smd_preprocessing import (
    load_basket,
    load_news_daily,
    load_ohlcv,
    load_putcall_daily,
    macd_histogram,
    merge_news_onto_index,
    rsi,
)

OUT_DIR = C.VALIDATION_DIR
UNIVERSE_JSON = OUT_DIR / "universe_predictions.json"
DYNAMIC_CACHE_DIR = OUT_DIR / "dynamic_cache"

# Stocks shown on the Answer tab (project dataset universe).
DATASET_UNIVERSE = [
    {"ticker": "^GSPC", "label": "S&P 500", "file_key": "GSPC"},
    {"ticker": "QQQ", "label": "QQQ (Nasdaq-100)", "file_key": "QQQ"},
] + [
    {"ticker": t, "label": t, "file_key": t} for t in C.BASKET
]


def _session():
    try:
        from curl_cffi import requests as cr  # noqa: PLC0415
        return cr.Session(impersonate="chrome")
    except Exception:  # noqa: BLE001
        return None


def _rf(n_estimators: int = 200):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _cols(df: pd.DataFrame) -> list[str]:
    return [c for c in PRICE_FEATURES + MOOD_FEATURES if c in df.columns]


def normalize_ticker(raw: str) -> str:
    t = (raw or "").strip().upper()
    t = t.replace(" ", "")
    if t in ("SPX", "SP500", "S&P500", "S&P", "GSPC"):
        return "^GSPC"
    if t in ("NASDAQ100", "NDX"):
        return "QQQ"
    if not re.match(r"^[A-Z0-9.\-^]+$", t):
        raise ValueError(f"Invalid ticker: {raw!r}")
    return t


def _as_ns_index(idx) -> pd.DatetimeIndex:
    """Normalize to timezone-naive datetime64[ns] for safe merges."""
    di = pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert(None).normalize()
    return pd.DatetimeIndex(di.astype("datetime64[ns]"))


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Prefer first level (Open/High/Low/Close/Volume)
        out.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in out.columns]
    # Deduplicate column names keeping first
    out = out.loc[:, ~pd.Index(out.columns).duplicated()]
    rename = {}
    for c in out.columns:
        cl = str(c).lower()
        if cl == "adj close" and "Close" not in out.columns:
            rename[c] = "Close"
        elif cl in ("open", "high", "low", "close", "volume"):
            rename[c] = cl.title() if cl != "volume" else "Volume"
    out = out.rename(columns=rename)
    need = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    out = out[out.index.notna()].sort_index()
    out = out[need].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Close"])
    out.index = _as_ns_index(out.index)
    return out


def download_ticker_daily(ticker: str, period: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for any Yahoo ticker."""
    period = period or C.DAILY_PERIOD
    session = _session()
    df = yf.download(
        ticker, period=period, interval="1d",
        progress=False, auto_adjust=True, session=session,
    )
    out = _flatten_ohlcv(df)
    if len(out) < 80:
        # Local cache fallback for known project files
        key = ticker.strip("^")
        local = C.RAW_DIR / f"{key}_daily.csv"
        if local.exists():
            try:
                out = load_ohlcv(f"{key}_daily")
                out = _flatten_ohlcv(out)
            except Exception:  # noqa: BLE001
                pass
    if len(out) < 80:
        raise ValueError(
            f"Not enough history for {ticker} ({len(out)} rows). "
            "Try a liquid US ticker like AAPL, MSFT, TSLA."
        )
    return out


def _ref_close(prefer: str = "GSPC") -> pd.Series:
    """Reference index close for relative momentum."""
    try:
        if prefer == "QQQ":
            return load_ohlcv("QQQ_daily")["Close"]
        return load_ohlcv("GSPC_daily")["Close"]
    except Exception:  # noqa: BLE001
        try:
            return load_ohlcv("QQQ_daily")["Close"]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Need GSPC_daily or QQQ_daily in raw_data for relative features.") from exc


def build_features_for_ohlcv(index: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build price+mood features; target = next-day up for this ticker."""
    index = index.copy()
    index.index = _as_ns_index(index.index)

    vix = load_ohlcv("VIX_daily")
    vix = vix.copy()
    vix.index = _as_ns_index(vix.index)
    basket = load_basket()
    basket = basket.copy()
    basket.index = _as_ns_index(basket.index)
    putcall_daily = load_putcall_daily()
    if putcall_daily is not None:
        putcall_daily = putcall_daily.copy()
        putcall_daily.index = _as_ns_index(putcall_daily.index)
    news_daily = load_news_daily()
    # Relative strength vs the other liquid market (avoid self-leak)
    ref = _ref_close("QQQ" if ticker.upper().replace("^", "") == "GSPC" else "GSPC")
    ref = ref.copy()
    ref.index = _as_ns_index(ref.index)

    close = index["Close"]
    volume = index["Volume"]
    feats = pd.DataFrame(index=index.index)
    feats["ret_1"] = close.pct_change(fill_method=None)
    feats["mom_5"] = close.pct_change(5, fill_method=None)
    feats["mom_10"] = close.pct_change(10, fill_method=None)
    feats["mom_20"] = close.pct_change(20, fill_method=None)
    feats["rsi_14"] = rsi(close, 14)
    feats["macd_hist"] = macd_histogram(close)
    feats["dist_ma20"] = close / close.rolling(20).mean() - 1.0
    feats["vol_change"] = volume.pct_change(fill_method=None)
    feats["realized_vol_20"] = feats["ret_1"].rolling(20).std()
    feats["intraday_range"] = (index["High"] - index["Low"]) / close
    prev_close = close.shift(1)
    feats["gap_ret"] = (index["Open"] - prev_close) / prev_close.replace(0, np.nan)
    feats["ret_lag2"] = feats["ret_1"].shift(1)
    other = ref.reindex(feats.index).ffill()
    feats["qqq_rel_mom5"] = other.pct_change(5, fill_method=None) - feats["mom_5"]

    feats["vix_level"] = vix["Close"].reindex(feats.index).ffill()
    feats["vix_change"] = feats["vix_level"].pct_change(fill_method=None)
    vix_ma20 = feats["vix_level"].rolling(20).mean()
    feats["vix_dist_ma20"] = feats["vix_level"] / vix_ma20 - 1.0

    basket_ret = basket.pct_change(fill_method=None)
    breadth = (basket_ret > 0).sum(axis=1) / basket_ret.notna().sum(axis=1)
    dispersion = basket_ret.std(axis=1)
    feats["breadth"] = breadth.reindex(feats.index).ffill()
    feats["dispersion"] = dispersion.reindex(feats.index).ffill()
    feats["breadth_change"] = feats["breadth"].diff()

    if putcall_daily is not None and len(putcall_daily) > 0:
        pc = putcall_daily.reindex(feats.index).ffill().bfill()
        feats["putcall_ratio"] = pc
        feats["putcall_change"] = pc.pct_change(fill_method=None)
        pc_ma20 = pc.rolling(20, min_periods=5).mean()
        feats["putcall_dist_ma20"] = pc / pc_ma20 - 1.0
    else:
        feats["putcall_ratio"] = np.nan
        feats["putcall_change"] = np.nan
        feats["putcall_dist_ma20"] = np.nan

    if news_daily is not None and len(news_daily) > 0:
        merged = merge_news_onto_index(news_daily, feats.index)
        feats["news_sentiment"] = merged["news_sentiment"].reindex(feats.index).fillna(0.0)
        feats["news_count"] = merged["news_count"].reindex(feats.index).fillna(0.0)
        feats["news_sent_roll5"] = feats["news_sentiment"].rolling(5, min_periods=1).mean()
    else:
        feats["news_sentiment"] = 0.0
        feats["news_count"] = 0.0
        feats["news_sent_roll5"] = 0.0

    next_ret = close.pct_change(fill_method=None).shift(-1)
    feats[TARGET] = (next_ret > 0).astype(float)
    feats.loc[feats.index.max(), TARGET] = np.nan

    feats = feats.replace([np.inf, -np.inf], np.nan)
    required = ["ret_1", "mom_20", "rsi_14", "vix_level", "realized_vol_20"]
    feats = feats.dropna(subset=required)
    feat_cols = [c for c in feats.columns if c != TARGET]
    feats = feats.ffill().dropna(subset=feat_cols)
    return feats


def train_and_predict(
    ticker: str,
    *,
    ohlcv: pd.DataFrame | None = None,
    n_estimators: int = 200,
    label: str | None = None,
) -> dict:
    """Full pipeline for one ticker. Returns JSON-serialisable result dict."""
    ticker = normalize_ticker(ticker)
    label = label or ticker
    if ohlcv is None:
        ohlcv = download_ticker_daily(ticker)
    else:
        ohlcv = _flatten_ohlcv(ohlcv)

    feats = build_features_for_ohlcv(ohlcv, ticker)
    cols = _cols(feats)
    labeled = feats.dropna(subset=[TARGET])
    if len(labeled) < 120:
        raise ValueError(f"Too few labeled days for {ticker} ({len(labeled)}).")

    latest = feats.iloc[[-1]]
    as_of = pd.Timestamp(latest.index[0])
    smote_cls = _try_smote()

    # Honest holdout metrics
    train, test = time_split(labeled, 0.8)
    model = _rf(n_estimators)
    x_tr, y_tr = maybe_smote(train[cols].values, train[TARGET].astype(int).values, True, smote_cls)
    model.fit(x_tr, y_tr)
    test_prob = model.predict_proba(test[cols].values)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    y_te = test[TARGET].astype(int).values
    test_acc = float((test_pred == y_te).mean())
    from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: PLC0415
    test_bal = float(balanced_accuracy_score(y_te, test_pred))
    test_f1 = float(f1_score(y_te, test_pred, zero_division=0))

    # Live next-day: refit on all labeled
    live_model = _rf(n_estimators)
    x_all, y_all = maybe_smote(labeled[cols].values, labeled[TARGET].astype(int).values, True, smote_cls)
    live_model.fit(x_all, y_all)
    day_prob = float(live_model.predict_proba(latest[cols].values)[0, 1])
    day_dir = "UP" if day_prob >= 0.5 else "DOWN"

    # Weekly (~5 day) target
    close = ohlcv["Close"].reindex(labeled.index).ffill()
    fwd = close.shift(-5) / close - 1.0
    week = labeled.copy()
    week["target_week_up"] = (fwd > 0).astype(float)
    week_lab = week.dropna(subset=["target_week_up"])
    w_train, w_test = time_split(week_lab, 0.8)
    w_model = _rf(n_estimators)
    wx, wy = maybe_smote(
        w_train[cols].values, w_train["target_week_up"].astype(int).values, True, smote_cls,
    )
    w_model.fit(wx, wy)
    w_prob_te = w_model.predict_proba(w_test[cols].values)[:, 1]
    w_pred_te = (w_prob_te >= 0.5).astype(int)
    w_acc = float((w_pred_te == w_test["target_week_up"].astype(int).values).mean())
    wx_all, wy_all = maybe_smote(
        week_lab[cols].values, week_lab["target_week_up"].astype(int).values, True, smote_cls,
    )
    w_model.fit(wx_all, wy_all)
    week_prob = float(w_model.predict_proba(latest[cols].values)[0, 1])
    week_dir = "UP" if week_prob >= 0.5 else "DOWN"

    # Compact history for chart (test window day model)
    hist = []
    for dt, row in test.iterrows():
        p = float(model.predict_proba(row[cols].values.reshape(1, -1))[0, 1])
        pred = int(p >= 0.5)
        actual = int(row[TARGET])
        hist.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "year": int(pd.Timestamp(dt).year),
            "prob_up": p,
            "pred_label": "UP" if pred else "DOWN",
            "actual_label": "UP" if actual else "DOWN",
            "correct": int(pred == actual),
        })

    result = {
        "ticker": ticker,
        "label": label,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": "Random Forest (price + mood + SMOTE)",
        "n_rows": int(len(labeled)),
        "n_test": int(len(test)),
        "data_start": str(pd.Timestamp(labeled.index.min()).date()),
        "data_end": str(pd.Timestamp(labeled.index.max()).date()),
        "next_day": {
            "direction": day_dir,
            "probability": day_prob,
            "confidence_pct": abs(day_prob - 0.5) * 200,
            "as_of": str(as_of.date()),
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bal,
            "test_f1": test_f1,
        },
        "next_week": {
            "direction": week_dir,
            "probability": week_prob,
            "confidence_pct": abs(week_prob - 0.5) * 200,
            "as_of": str(as_of.date()),
            "horizon": "next_5_trading_days",
            "test_accuracy": w_acc,
        },
        "quick_answer": (
            f"{label}: next day {day_dir} ({day_prob:.0%} P(Up)); "
            f"next week {week_dir} ({week_prob:.0%} P(Up))."
        ),
        "history": hist[-60:],
        "status": "ok",
        "error": None,
    }
    return result


def predict_dataset_universe(n_estimators: int = 180) -> dict:
    """Train/predict every stock in the project dataset; save JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in DATASET_UNIVERSE:
        ticker, label = item["ticker"], item["label"]
        print(f" - {label} ({ticker})")
        try:
            # Prefer local raw file for indexes; basket names download full OHLCV
            ohlcv = None
            key = item["file_key"]
            if key not in C.BASKET:
                local = C.RAW_DIR / f"{key}_daily.csv"
                if local.exists():
                    try:
                        ohlcv = load_ohlcv(f"{key}_daily")
                    except Exception:  # noqa: BLE001
                        ohlcv = None
            res = train_and_predict(ticker, ohlcv=ohlcv, n_estimators=n_estimators, label=label)
            rows.append(res)
            try:
                print(f"   -> {res['quick_answer']}")
            except Exception:  # noqa: BLE001
                print(f"   -> ok {ticker}")
        except Exception as exc:  # noqa: BLE001
            print(f"   x {ticker}: {exc}")
            rows.append({
                "ticker": ticker,
                "label": label,
                "status": "error",
                "error": str(exc),
                "quick_answer": f"{label}: unavailable ({exc})",
                "next_day": {"direction": "-", "probability": None, "test_accuracy": None},
                "next_week": {"direction": "-", "probability": None, "test_accuracy": None},
            })

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_stocks": len(rows),
        "stocks": rows,
    }
    UNIVERSE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f" - saved {UNIVERSE_JSON}")
    return payload


def cache_dynamic_result(result: dict) -> Path:
    DYNAMIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Z0-9]+", "_", result["ticker"])
    path = DYNAMIC_CACHE_DIR / f"{safe}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def load_universe() -> dict | None:
    if not UNIVERSE_JSON.exists():
        return None
    return json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 70)
    print("Dynamic predictions - dataset universe")
    print("=" * 70)
    predict_dataset_universe()
    print("=" * 70)


if __name__ == "__main__":
    main()
