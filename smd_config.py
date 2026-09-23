"""Central configuration for the Sentiment-Driven Market Direction project.

Shared paths and settings used by data collection, preprocessing and modelling.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "smd_data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "smd_data" / "processed_data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Sprint 1.2 (preprocessing + EDA) and Sprint 2.1 (initial models) write here.
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MODELS_DIR = REPORTS_DIR / "models"
COMPARISON_DIR = REPORTS_DIR / "comparison"
# Sprint 4: external validation (secondary market) + side-by-side charts.
VALIDATION_DIR = REPORTS_DIR / "validation"

# Sprint 3.1 dashboard (Plotly Dash).
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050

# The final tidy table of features + target used by the models.
FEATURES_FILE = PROCESSED_DIR / "features_daily.csv"
# Sprint 4: same feature recipe, but next-day Up/Down target is QQQ.
FEATURES_QQQ_FILE = PROCESSED_DIR / "features_qqq_daily.csv"

# Index whose next-period DIRECTION we classify.
INDEX_TICKER = "^GSPC"        # S&P 500
SECONDARY_INDEX = "QQQ"       # Nasdaq-100 ETF (cross-check / validation, Sprint 4)

# Large-cap basket for BREADTH / HERDING signals (% advancing, dispersion).
BASKET = ["TSLA", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM"]

# News: pull from SPY plus the basket so we get many headlines, not one tiny snapshot.
NEWS_TICKERS = ["SPY"] + BASKET

# CBOE equity put/call ratio index (daily history on Yahoo as ticker PCCE).
PUTCALL_DAILY_TICKER = "PCCE"

# Free RSS feeds for extra headline history (no API key).
NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=S%26P+500+stock+market&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Wall+Street+markets&hl=en-US&gl=US&ceid=US:en",
]

NEWS_HEADLINES_ARCHIVE = RAW_DIR / "news_sentiment_headlines_archive.csv"
# Single daily news table (built from the archive after each collection run).
NEWS_DAILY_FILE = RAW_DIR / "news_sentiment_daily_merged.csv"
PUTCALL_DAILY_FILE = RAW_DIR / "putcall_ratio_daily.csv"

# Behavioural context.
VIX_TICKER = "^VIX"           # fear / greed
PUTCALL_UNDERLYING = "SPY"    # derive put/call ratio from SPY option chain

# Frequencies (the brief asks for "varying frequencies").
DAILY_PERIOD = "5y"
HOURLY_PERIOD = "60d"

MAX_RETRIES = 4
RETRY_SLEEP_SEC = 5      # base for exponential backoff (5, 10, 20, 40s)
INTER_CALL_SLEEP = 2     # polite gap between sequential Yahoo calls

# Stooq fallback (free, no key) for DAILY OHLCV when Yahoo rate-limits.
STOOQ_SYMBOLS = {
    "^GSPC": "^spx",
    "QQQ": "qqq.us",
    "^VIX": "^vix",
    "TSLA": "tsla.us",
    "MSFT": "msft.us",
    "AMZN": "amzn.us",
    "GOOGL": "googl.us",
    "META": "meta.us",
    "NVDA": "nvda.us",
    "JPM": "jpm.us",
}
