"""Feature column names shared by preprocessing and modelling scripts."""

PRICE_FEATURES = [
    "ret_1", "mom_5", "mom_10", "mom_20", "rsi_14", "macd_hist", "dist_ma20",
    "vol_change", "realized_vol_20", "intraday_range", "gap_ret", "ret_lag2",
    "qqq_rel_mom5",
]

MOOD_FEATURES = [
    "vix_level", "vix_change", "vix_dist_ma20", "breadth", "breadth_change",
    "dispersion", "putcall_ratio", "putcall_change", "putcall_dist_ma20",
    "news_sentiment", "news_count", "news_sent_roll5",
]

TARGET = "target_up"
