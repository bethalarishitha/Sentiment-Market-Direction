"""Sprint 4+ live direction signals for the dashboard.

Produces:
  - Next trading day UP / DOWN (S&P 500)
  - Next week (~5 trading days) UP / DOWN
  - Walk-forward prediction history (for year-axis charts)

Run:
    python src/smd_live_predict.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import smd_config as C
from smd_features import MOOD_FEATURES, PRICE_FEATURES, TARGET
from smd_model_comparison import _try_smote, maybe_smote, time_split
from smd_preprocessing import load_ohlcv

OUT_DIR = C.VALIDATION_DIR
LIVE_JSON = OUT_DIR / "live_predictions.json"
HISTORY_CSV = OUT_DIR / "prediction_history.csv"
WEEKLY_CSV = OUT_DIR / "weekly_predictions.csv"


def _cols(df: pd.DataFrame) -> list[str]:
    return [c for c in PRICE_FEATURES + MOOD_FEATURES if c in df.columns]


def _rf():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def build_features_live() -> pd.DataFrame:
    """Same as build_features but keeps the final day (target may be NaN)."""
    from smd_preprocessing import (  # local import to reuse helpers
        load_basket,
        load_news_daily,
        load_putcall_daily,
        merge_news_onto_index,
        rsi,
        macd_histogram,
    )

    gspc = load_ohlcv("GSPC_daily")
    qqq = load_ohlcv("QQQ_daily")
    vix = load_ohlcv("VIX_daily")
    basket = load_basket()
    putcall_daily = load_putcall_daily()
    news_daily = load_news_daily()

    close = gspc["Close"]
    volume = gspc["Volume"]
    feats = pd.DataFrame(index=gspc.index)
    feats["ret_1"] = close.pct_change(fill_method=None)
    feats["mom_5"] = close.pct_change(5, fill_method=None)
    feats["mom_10"] = close.pct_change(10, fill_method=None)
    feats["mom_20"] = close.pct_change(20, fill_method=None)
    feats["rsi_14"] = rsi(close, 14)
    feats["macd_hist"] = macd_histogram(close)
    feats["dist_ma20"] = close / close.rolling(20).mean() - 1.0
    feats["vol_change"] = volume.pct_change(fill_method=None)
    feats["realized_vol_20"] = feats["ret_1"].rolling(20).std()
    feats["intraday_range"] = (gspc["High"] - gspc["Low"]) / close
    prev_close = close.shift(1)
    feats["gap_ret"] = (gspc["Open"] - prev_close) / prev_close.replace(0, np.nan)
    feats["ret_lag2"] = feats["ret_1"].shift(1)
    other_close = qqq["Close"].reindex(feats.index).ffill()
    feats["qqq_rel_mom5"] = other_close.pct_change(5, fill_method=None) - feats["mom_5"]

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
    # Keep last day: target is NaN there
    feats.loc[feats.index.max(), TARGET] = np.nan

    feats = feats.replace([np.inf, -np.inf], np.nan)
    required = ["ret_1", "mom_20", "rsi_14", "vix_level", "realized_vol_20"]
    feats = feats.dropna(subset=required)
    feats = feats.ffill()
    # Drop rows still missing feature values (except target on last day)
    feat_cols = [c for c in feats.columns if c != TARGET]
    feats = feats.dropna(subset=feat_cols)
    return feats


def walk_forward_history(df: pd.DataFrame, cols: list[str], smote_cls) -> pd.DataFrame:
    """Expanding-window predictions on the held-out 20% test window (honest history)."""
    train, test = time_split(df.dropna(subset=[TARGET]), 0.8)
    rows = []
    x_train = train[cols].values
    y_train = train[TARGET].astype(int).values
    x_tr, y_tr = maybe_smote(x_train, y_train, True, smote_cls)
    model = _rf()
    model.fit(x_tr, y_tr)
    thr = 0.5
    for dt, row in test.iterrows():
        x = row[cols].values.reshape(1, -1)
        prob = float(model.predict_proba(x)[0, 1])
        pred = int(prob >= thr)
        actual = int(row[TARGET])
        rows.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "year": int(pd.Timestamp(dt).year),
            "prob_up": prob,
            "pred": pred,
            "pred_label": "UP" if pred == 1 else "DOWN",
            "actual": actual,
            "actual_label": "UP" if actual == 1 else "DOWN",
            "correct": int(pred == actual),
        })
    return pd.DataFrame(rows)


def weekly_from_close(
    df: pd.DataFrame,
    cols: list[str],
    smote_cls,
    live_row: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Predict whether the next ~5 trading days move up (using GSPC close)."""
    close = load_ohlcv("GSPC_daily")["Close"].reindex(df.index).ffill()
    fwd = close.shift(-5) / close - 1.0
    work = df.copy()
    work["target_week_up"] = (fwd > 0).astype(float)
    labeled = work.dropna(subset=["target_week_up"] + cols)
    train, test = time_split(labeled, 0.8)
    model = _rf()
    x_tr, y_tr = maybe_smote(
        train[cols].values, train["target_week_up"].astype(int).values, True, smote_cls,
    )
    model.fit(x_tr, y_tr)
    hist = []
    for dt, row in test.iterrows():
        prob = float(model.predict_proba(row[cols].values.reshape(1, -1))[0, 1])
        pred = int(prob >= 0.5)
        actual = int(row["target_week_up"])
        hist.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "year": int(pd.Timestamp(dt).year),
            "prob_up": prob,
            "pred": pred,
            "pred_label": "UP" if pred == 1 else "DOWN",
            "actual": actual,
            "actual_label": "UP" if actual == 1 else "DOWN",
            "correct": int(pred == actual),
        })
    hist_df = pd.DataFrame(hist)

    # Refit on all labeled week rows, then score the true latest live feature row
    x_all, y_all = maybe_smote(
        labeled[cols].values, labeled["target_week_up"].astype(int).values, True, smote_cls,
    )
    model.fit(x_all, y_all)
    live_prob = float(model.predict_proba(live_row[cols].values)[0, 1])
    live = {
        "direction": "UP" if live_prob >= 0.5 else "DOWN",
        "probability": live_prob,
        "confidence_pct": abs(live_prob - 0.5) * 200,
        "as_of": str(pd.Timestamp(live_row.index[0]).date()),
        "horizon": "next_5_trading_days",
        "test_accuracy": float(hist_df["correct"].mean()) if len(hist_df) else None,
    }
    return hist_df, live


def main() -> None:
    print("=" * 70)
    print("Live direction signals (next day / next week)")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smote_cls = _try_smote()

    print(" - building live feature table (keeps latest day)")
    live_feats = build_features_live()
    cols = _cols(live_feats)
    labeled = live_feats.dropna(subset=[TARGET])
    latest = live_feats.iloc[[-1]]
    as_of = pd.Timestamp(latest.index[0])

    print(f" - labeled rows: {len(labeled)} | live as-of: {as_of.date()}")
    model = _rf()
    x_all, y_all = maybe_smote(
        labeled[cols].values, labeled[TARGET].astype(int).values, True, smote_cls,
    )
    model.fit(x_all, y_all)
    day_prob = float(model.predict_proba(latest[cols].values)[0, 1])
    day_dir = "UP" if day_prob >= 0.5 else "DOWN"

    print(" - walk-forward next-day history (test window)")
    hist = walk_forward_history(labeled, cols, smote_cls)
    hist.to_csv(HISTORY_CSV, index=False)

    print(" - weekly (5-day) model")
    week_hist, week_live = weekly_from_close(labeled, cols, smote_cls, latest)
    week_hist.to_csv(WEEKLY_CSV, index=False)

    # Honest test accuracy for the day model
    train, test = time_split(labeled, 0.8)
    m2 = _rf()
    x_tr, y_tr = maybe_smote(train[cols].values, train[TARGET].astype(int).values, True, smote_cls)
    m2.fit(x_tr, y_tr)
    test_prob = m2.predict_proba(test[cols].values)[:, 1]
    test_acc = float(((test_prob >= 0.5).astype(int) == test[TARGET].astype(int).values).mean())

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "market": "S&P 500 (GSPC)",
        "model": "Random Forest (price + mood + SMOTE)",
        "next_day": {
            "direction": day_dir,
            "probability": day_prob,
            "confidence_pct": abs(day_prob - 0.5) * 200,
            "as_of": str(as_of.date()),
            "predicts_for": "next trading day after as_of",
            "test_accuracy": test_acc,
        },
        "next_week": week_live,
        "data_range": {
            "start": str(pd.Timestamp(labeled.index.min()).date()),
            "end": str(pd.Timestamp(labeled.index.max()).date()),
            "n_days": int(len(labeled)),
        },
        "quick_answer": (
            f"Next day: {day_dir} ({day_prob:.0%} prob Up). "
            f"Next week (~5 days): {week_live['direction']} "
            f"({week_live['probability']:.0%} prob Up)."
        ),
    }
    LIVE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f" - saved {LIVE_JSON}")
    print(f" - saved {HISTORY_CSV} ({len(hist)} rows)")
    print(f" - saved {WEEKLY_CSV} ({len(week_hist)} rows)")
    print(f"\nQUICK ANSWER: {payload['quick_answer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
