"""Sentiment-Driven Market Direction Prediction. Sprint 1, Section 2.

Data Preprocessing and Exploratory Analysis.

Plain idea of this file:
    The Sprint 1.1 script downloaded raw price and mood data into
    smd_data/raw_data/. Here I clean that data, line every dataset up by date,
    turn it into "mood clues" (features) that a model can learn from, and build
    the answer column (did the market go Up or Down next day). Then I run a few
    simple statistics and charts so I can see what the data looks like before any
    modelling.

What I produce:
    1. smd_data/processed_data/features_daily.csv  (clean features + target)
    2. reports/eda/summary_statistics.csv          (describe of every feature)
    3. reports/eda/*.png                            (a few easy to read charts)

Run:
    python src/smd_preprocessing.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop up window needed
import matplotlib.pyplot as plt

import smd_config as C


# ----------------------------------------------------------------------------
# 1. Loading the raw files
# ----------------------------------------------------------------------------
def load_ohlcv(name: str) -> pd.DataFrame:
    """Load one yfinance daily file (S&P 500, QQQ or VIX).

    These files have three header rows (Price, Ticker, an empty Date row), so I
    skip the two extra rows and keep the Date column as the index.
    """
    path = C.RAW_DIR / f"{name}.csv"
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_basket() -> pd.DataFrame:
    """Load the 7 big company daily closing prices (already a clean table)."""
    path = C.RAW_DIR / "basket_closes_daily.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def load_news_daily() -> pd.DataFrame | None:
    """Daily mean VADER score and headline count from the headline archive."""
    path = C.NEWS_DAILY_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def load_putcall_daily() -> pd.Series | None:
    """CBOE equity put/call ratio (PCCE) aligned to trading days."""
    if not C.PUTCALL_DAILY_FILE.exists():
        return None
    df = pd.read_csv(C.PUTCALL_DAILY_FILE, index_col=0, parse_dates=True)
    if "putcall_ratio" not in df.columns:
        df.columns = ["putcall_ratio"]
    s = pd.to_numeric(df["putcall_ratio"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def merge_news_onto_index(news_daily: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach headline aggregates to trading days (Yahoo dates often lag OHLCV)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).tz_convert(None).normalize()
    idx = pd.DatetimeIndex(idx.astype("datetime64[ns]")).sort_values()
    out = pd.DataFrame(index=idx, columns=["news_sentiment", "news_count"], dtype=float)
    out[:] = 0.0
    if news_daily is None or len(news_daily) == 0:
        return out

    nd = news_daily.copy()
    nd.index = pd.to_datetime(nd.index, utc=True).tz_convert("America/New_York").normalize()
    nd.index = nd.index.tz_localize(None)
    nd.index = pd.DatetimeIndex(nd.index.astype("datetime64[ns]"))
    nd = nd.sort_index()

    # Match headline days to the last trading day on or before that calendar date.
    cal = pd.DataFrame({"date": idx})
    nd_reset = nd.reset_index().rename(columns={"index": "date"})
    if "date" not in nd_reset.columns:
        nd_reset = nd.reset_index()
        nd_reset.columns = ["date", "news_sentiment", "news_count"]
    cal["date"] = pd.to_datetime(cal["date"]).astype("datetime64[ns]")
    nd_reset["date"] = pd.to_datetime(nd_reset["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        cal.sort_values("date"),
        nd_reset.sort_values("date"),
        on="date",
        direction="backward",
    )
    out.loc[merged["date"].values, "news_sentiment"] = merged["news_sentiment"].fillna(0.0).values
    out.loc[merged["date"].values, "news_count"] = merged["news_count"].fillna(0.0).values

    # Headlines dated after the last OHLCV row: map onto the most recent trading days.
    if len(nd) > 0 and nd.index.max() > idx.max():
        tail = nd[nd.index > idx.max()]
        slots = idx[-len(tail) :]
        for i, (_, row) in enumerate(tail.iterrows()):
            target = slots[min(i, len(slots) - 1)]
            out.loc[target, "news_sentiment"] = float(row["news_sentiment"])
            out.loc[target, "news_count"] = float(row["news_count"])
    return out


# ----------------------------------------------------------------------------
# 2. Small technical indicator helpers (the "price mood" formulas)
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index. High means bought too hard, low means sold too hard."""
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram. Positive means upward momentum is building."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ----------------------------------------------------------------------------
# 3. Build the clean feature table + the Up/Down answer
# ----------------------------------------------------------------------------
def build_features(target: str = "GSPC") -> pd.DataFrame:
    """Build daily features + next-day Up/Down target for one market.

    target:
        "GSPC" - S&P 500 next-day direction (primary / Sprint 1–3).
        "QQQ"  - Nasdaq-100 ETF next-day direction (Sprint 4 validation set).

    Price features come from the target market. Mood features (VIX, basket,
    put/call, news) stay shared. Relative momentum uses the *other* liquid
    index so the target return is not copied into a feature.
    """
    target = target.upper().replace("^", "")
    if target not in ("GSPC", "QQQ"):
        raise ValueError(f"Unsupported target market: {target}")

    print(f" - loading raw files (target market: {target})")
    gspc = load_ohlcv("GSPC_daily")
    qqq = load_ohlcv("QQQ_daily")
    index = gspc if target == "GSPC" else qqq
    other = qqq if target == "GSPC" else gspc
    vix = load_ohlcv(C.VIX_TICKER.strip("^") + "_daily")
    basket = load_basket()
    putcall_daily = load_putcall_daily()
    news_daily = load_news_daily()

    close = index["Close"]
    volume = index["Volume"]

    print(" - building price based features")
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
    # Same column name for both markets so PRICE_FEATURES stays shared.
    # Value = other_index 5-day momentum minus this market's 5-day momentum.
    other_close = other["Close"].reindex(feats.index).ffill()
    feats["qqq_rel_mom5"] = other_close.pct_change(5, fill_method=None) - feats["mom_5"]

    print(" - building behavioural (mood) features")
    # Fear and greed from the VIX.
    feats["vix_level"] = vix["Close"].reindex(feats.index).ffill()
    feats["vix_change"] = feats["vix_level"].pct_change(fill_method=None)
    vix_ma20 = feats["vix_level"].rolling(20).mean()
    feats["vix_dist_ma20"] = feats["vix_level"] / vix_ma20 - 1.0

    # Herding and breadth from the 7 company basket.
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
        print(f"   . put/call daily (PCCE): {putcall_daily.notna().sum()} raw days merged")
    else:
        print("   ! no putcall_ratio_daily.csv, skipping put/call features")
        feats["putcall_ratio"] = np.nan
        feats["putcall_change"] = np.nan
        feats["putcall_dist_ma20"] = np.nan

    if news_daily is not None and len(news_daily) > 0:
        merged = merge_news_onto_index(news_daily, feats.index)
        feats["news_sentiment"] = merged["news_sentiment"].reindex(feats.index).fillna(0.0)
        feats["news_count"] = merged["news_count"].reindex(feats.index).fillna(0.0)
        feats["news_sent_roll5"] = feats["news_sentiment"].rolling(5, min_periods=1).mean()
        print(f"   . news daily series: {news_daily.shape[0]} headline days mapped to trading days")
    else:
        print("   ! no news daily file, falling back to neutral news features")
        feats["news_sentiment"] = 0.0
        feats["news_count"] = 0.0
        feats["news_sent_roll5"] = 0.0

    print(" - building the Up/Down answer (target)")
    # 1 if the NEXT day's return is positive, else 0. shift(-1) looks one day ahead.
    next_ret = close.pct_change(fill_method=None).shift(-1)
    feats["target_up"] = (next_ret > 0).astype("int8")
    # The very last row has no "next day" yet, so drop it.
    feats = feats.iloc[:-1]

    print(" - cleaning: warm up rows and missing values")
    before = len(feats)
    # A zero volume day makes a percentage change blow up to infinity, so turn
    # any infinity into a missing value first, then handle missing values.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    required = ["ret_1", "mom_20", "rsi_14", "vix_level", "realized_vol_20"]
    if putcall_daily is not None and len(putcall_daily) > 0:
        required.append("putcall_ratio")
    feats = feats.dropna(subset=required)
    feats = feats.ffill().dropna()
    print(f"   rows kept: {len(feats)} of {before} (dropped indicator warm up period)")
    return feats


# ----------------------------------------------------------------------------
# 4. Exploratory analysis: statistics + charts + normalisation demo
# ----------------------------------------------------------------------------
def run_eda(feats: pd.DataFrame) -> None:
    C.EDA_DIR.mkdir(parents=True, exist_ok=True)

    print(" - saving summary statistics")
    stats = feats.describe().T
    stats.to_csv(C.EDA_DIR / "summary_statistics.csv")

    balance = feats["target_up"].value_counts(normalize=True).sort_index()
    up_pct = float(balance.get(1, 0.0) * 100)
    down_pct = float(balance.get(0, 0.0) * 100)
    print(f"   class balance -> Up: {up_pct:.1f}%  Down: {down_pct:.1f}%")

    # Chart 1: how many Up vs Down days (is the data balanced?).
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = feats["target_up"].map({0: "Down", 1: "Up"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#d9534f", "#5cb85c"])
    ax.set_title("How many Up vs Down next days")
    ax.set_ylabel("number of days")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    # Chart 2: does each feature line up with the answer? (correlation with target).
    corr = feats.corr(numeric_only=True)["target_up"].drop("target_up").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_title("Correlation of each feature with next day Up")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / "feature_target_correlation.png", dpi=120)
    plt.close(fig)

    # Chart 3: correlation heatmap between features (spot copies of each other).
    fig, ax = plt.subplots(figsize=(8, 7))
    numeric = feats.drop(columns=["target_up"])
    matrix = numeric.corr(numeric_only=True)
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(matrix.columns)))
    ax.set_yticklabels(matrix.columns, fontsize=8)
    ax.set_title("Feature correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / "feature_heatmap.png", dpi=120)
    plt.close(fig)

    # Chart 4: normalisation demo (raw vs z-score vs min-max) on the VIX.
    demo = feats["vix_level"].reset_index(drop=True)
    z_score = (demo - demo.mean()) / demo.std()
    min_max = (demo - demo.min()) / (demo.max() - demo.min())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(demo, color="#0275d8"); axes[0].set_title("VIX raw values")
    axes[1].plot(z_score, color="#5bc0de"); axes[1].set_title("VIX after z-score")
    axes[2].plot(min_max, color="#f0ad4e"); axes[2].set_title("VIX after min-max")
    for a in axes:
        a.set_xlabel("day number")
    fig.suptitle("Same feature, three normalisation methods")
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / "normalisation_demo.png", dpi=120)
    plt.close(fig)

    print(f"   4 charts + summary_statistics.csv saved to {C.EDA_DIR}")


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Sprint 1.2  Data Preprocessing and Exploratory Analysis")
    print("=" * 70)

    feats = build_features()
    feats.to_csv(C.FEATURES_FILE)
    print(f"\nProcessed features saved to {C.FEATURES_FILE}")
    print(f"Shape: {feats.shape[0]} rows x {feats.shape[1]} columns")

    run_eda(feats)
    print("\nSprint 1.2 done. Clean features and charts are ready for modelling.")


if __name__ == "__main__":
    main()
