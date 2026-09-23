"""Market Direction Analytics Dashboard (tabbed professional UI).

Tabs:
  Stock signals | Models & validation | Market & years | Data tables

Run:
    python src/smd_model_comparison.py
    python src/smd_validation.py
    python src/smd_dynamic_predict.py
    python src/smd_dashboard.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, callback, clientside_callback, dash_table, dcc, html, no_update, ctx

import smd_config as C
from smd_preprocessing import load_basket, load_ohlcv

_SRC_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _SRC_DIR / "assets"
_DASH_APP_NAME = "smd_dashboard"
DASH_UI_VERSION = "6.2"

THEME = {
    "bg": "#0b1220",
    "card": "#121a2b",
    "grid": "#243044",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "teal": "#2dd4bf",
    "yellow": "#fbbf24",
    "blue": "#60a5fa",
    "green": "#34d399",
    "red": "#f87171",
    "series": ["#2dd4bf", "#fbbf24", "#60a5fa", "#34d399", "#f87171", "#a3e635"],
}

CHART_FONT = "DM Sans, Segoe UI, system-ui, sans-serif"
GRAPH_CFG = {"displayModeBar": False}


def _load_css() -> str:
    p = _ASSETS_DIR / "dashboard.css"
    return p.read_text(encoding="utf-8") if p.exists() else ""


CUSTOM_CSS = _load_css()

pio.templates["smd_pro"] = go.layout.Template(
    layout=go.Layout(
        font={"family": CHART_FONT, "size": 12, "color": THEME["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=THEME["card"],
        colorway=THEME["series"],
        hoverlabel={"bgcolor": "#070d18", "font_size": 12, "font_color": "#f8fafc"},
        xaxis={"gridcolor": THEME["grid"], "linecolor": THEME["grid"], "tickfont": {"color": THEME["muted"]}},
        yaxis={"gridcolor": THEME["grid"], "linecolor": THEME["grid"], "tickfont": {"color": THEME["muted"]}},
        margin={"l": 48, "r": 20, "t": 36, "b": 48},
    )
)
pio.templates.default = "smd_pro"


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_price_map() -> dict[str, pd.Series]:
    """Close prices for every dataset ticker (indexes + basket)."""
    out: dict[str, pd.Series] = {}
    for key, ticker in (("GSPC_daily", "^GSPC"), ("QQQ_daily", "QQQ")):
        try:
            df = load_ohlcv(key)
            if df is not None and "Close" in df.columns:
                out[ticker] = df["Close"].dropna()
        except Exception:  # noqa: BLE001
            pass
    try:
        basket = load_basket()
        for col in basket.columns:
            out[str(col)] = basket[col].dropna()
    except Exception:  # noqa: BLE001
        pass
    return out


def load_data() -> dict:
    feats = pd.read_csv(C.FEATURES_FILE, index_col=0, parse_dates=True) if C.FEATURES_FILE.exists() else None
    gspc = None
    try:
        gspc = load_ohlcv("GSPC_daily")
    except Exception:  # noqa: BLE001
        gspc = None
    vix = None
    try:
        vix = load_ohlcv("VIX_daily")
    except Exception:  # noqa: BLE001
        vix = None
    return {
        "runs": _read_csv(C.COMPARISON_DIR / "all_model_runs.csv"),
        "best": _read_csv(C.COMPARISON_DIR / "best_per_model.csv"),
        "baseline": _read_csv(C.COMPARISON_DIR / "baseline_vs_sprint22.csv"),
        "features": feats,
        "validation": _read_csv(C.VALIDATION_DIR / "validation_side_by_side.csv"),
        "validation_compare": _read_csv(C.VALIDATION_DIR / "previous_vs_validation.csv"),
        "live": _read_json(C.VALIDATION_DIR / "live_predictions.json"),
        "universe": _read_json(C.VALIDATION_DIR / "universe_predictions.json"),
        "pred_hist": _read_csv(C.VALIDATION_DIR / "prediction_history.csv"),
        "week_hist": _read_csv(C.VALIDATION_DIR / "weekly_predictions.csv"),
        "gspc": gspc,
        "vix": vix,
        "prices": load_price_map(),
    }


def to_store(df: pd.DataFrame | None) -> list:
    return json.loads(df.to_json(orient="records")) if df is not None and len(df) else []


def from_store(records: list | None) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def apply_filters(df: pd.DataFrame, model: str, feature_set: str, smote: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    out = df.copy()
    if model and model != "all" and "model" in out.columns:
        out = out[out["model"] == model]
    if feature_set and feature_set != "all" and "features" in out.columns:
        out = out[out["features"] == feature_set]
    if smote and smote != "all" and "smote" in out.columns:
        out = out[out["smote"].astype(bool) == (smote == "yes")]
    return out


def _empty(msg: str, h: int = 240) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font={"size": 13, "color": THEME["muted"]},
    )
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=h, template="smd_pro")
    return fig


def universe_stocks(universe: dict | None) -> list[dict]:
    return list((universe or {}).get("stocks") or [])


def ticker_options(universe: dict | None) -> list[dict]:
    opts = []
    for s in universe_stocks(universe):
        if s.get("status") != "ok" and not s.get("history"):
            continue
        label = s.get("label") or s.get("ticker")
        ticker = s.get("ticker")
        opts.append({"label": f"{label} ({ticker})", "value": ticker})
    if not opts:
        opts = [{"label": "S&P 500 (^GSPC)", "value": "^GSPC"}]
    return opts


def find_stock(universe: dict | None, ticker: str | None) -> dict | None:
    if not ticker:
        return None
    for s in universe_stocks(universe):
        if s.get("ticker") == ticker:
            return s
    return None


def default_ticker(universe: dict | None) -> str:
    opts = ticker_options(universe)
    return opts[0]["value"] if opts else "^GSPC"


def year_options_from_prices(prices: dict[str, pd.Series]) -> list[dict]:
    years: set[int] = set()
    for s in prices.values():
        if s is None or len(s) == 0:
            continue
        years.update(int(y) for y in pd.DatetimeIndex(s.index).year.unique())
    opts = [{"label": "All years", "value": "all"}]
    for y in sorted(years, reverse=True):
        opts.append({"label": str(y), "value": str(y)})
    return opts if len(opts) > 1 else [{"label": "All years", "value": "all"}]


def filter_series_by_year(series: pd.Series | None, year: str | int | None) -> pd.Series | None:
    if series is None or len(series) == 0:
        return series
    if year is None or str(year) == "all":
        return series
    return series[series.index.year == int(year)]


def filter_hist_by_year(df: pd.DataFrame | None, year: str | int | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "year" not in out.columns and "date" in out.columns:
        out["year"] = pd.to_datetime(out["date"]).dt.year
    if year is None or str(year) == "all" or "year" not in out.columns:
        return out
    return out[out["year"].astype(int) == int(year)]


def stock_history_df(stock: dict | None) -> pd.DataFrame | None:
    if not stock:
        return None
    hist = stock.get("history") or []
    if not hist:
        return None
    df = pd.DataFrame(hist)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def universe_summary_df(universe: dict | None) -> pd.DataFrame:
    rows = []
    for s in universe_stocks(universe):
        day = s.get("next_day") or {}
        week = s.get("next_week") or {}
        rows.append({
            "ticker": s.get("ticker"),
            "label": s.get("label"),
            "status": s.get("status"),
            "next_day": day.get("direction"),
            "day_p_up": day.get("probability"),
            "day_test_acc": day.get("test_accuracy"),
            "day_f1": day.get("test_f1"),
            "next_week": week.get("direction"),
            "week_p_up": week.get("probability"),
            "week_test_acc": week.get("test_accuracy"),
            "n_rows": s.get("n_rows"),
            "data_end": s.get("data_end") or day.get("as_of"),
            "quick_answer": s.get("quick_answer"),
        })
    return pd.DataFrame(rows)


def universe_history_df(universe: dict | None) -> pd.DataFrame:
    rows = []
    for s in universe_stocks(universe):
        ticker = s.get("ticker")
        label = s.get("label")
        for h in s.get("history") or []:
            rows.append({
                "ticker": ticker,
                "label": label,
                "date": h.get("date"),
                "year": h.get("year"),
                "prob_up": h.get("prob_up"),
                "pred_label": h.get("pred_label"),
                "actual_label": h.get("actual_label"),
                "correct": h.get("correct"),
            })
    return pd.DataFrame(rows)


def fig_sparkline(df: pd.DataFrame, col: str, color: str) -> go.Figure:
    if len(df) == 0 or col not in df.columns:
        return _empty("", 90)
    s = df[col].sort_values().reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        x=s.index, y=s, mode="lines", line={"color": color, "width": 2},
        fill="tozeroy", fillcolor="rgba(45, 212, 191, 0.12)",
    ))
    fig.update_layout(
        showlegend=False, xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 4, "r": 4, "t": 2, "b": 4}, height=90, template="smd_pro",
    )
    return fig


def fig_metric_by_model(df: pd.DataFrame, metric: str) -> go.Figure:
    if len(df) == 0 or "model" not in df.columns or metric not in df.columns:
        return _empty("No data for current filters")
    agg = df.groupby("model", as_index=False)[metric].max().sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(
        x=agg[metric], y=agg["model"], orientation="h",
        marker={"color": agg[metric], "colorscale": [[0, THEME["blue"]], [1, THEME["teal"]]]},
        text=agg[metric].map(lambda v: f"{v:.1%}" if metric != "f1" else f"{v:.3f}"),
        textposition="outside", textfont={"color": THEME["text"], "size": 11},
    ))
    fmt = ".0%" if metric != "f1" else ".3f"
    xmax = float(agg[metric].max()) if len(agg) else 1.0
    fig.update_layout(
        xaxis={"tickformat": fmt, "range": [0, xmax * 1.18]},
        yaxis={"automargin": True}, showlegend=False, height=320,
        margin={"l": 150, "r": 72, "t": 24, "b": 40}, template="smd_pro",
    )
    return fig


def fig_metrics_heatmap(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0 or "model" not in df.columns:
        return _empty("No model metrics")
    metrics = [c for c in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall") if c in df.columns]
    if not metrics:
        return _empty("No metric columns")
    agg = df.groupby("model")[metrics].max()
    fig = go.Figure(go.Heatmap(
        z=agg.values, x=[m.replace("_", " ").title() for m in metrics], y=agg.index.tolist(),
        colorscale=[[0, "#0b1220"], [0.5, "#1e3a5f"], [1, THEME["teal"]]],
        text=[[f"{v:.2f}" for v in row] for row in agg.values],
        texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=300, margin={"l": 140, "r": 24, "t": 20, "b": 40}, template="smd_pro")
    return fig


def fig_feature_compare(df: pd.DataFrame, metric: str) -> go.Figure:
    if len(df) == 0 or "features" not in df.columns or metric not in df.columns:
        return _empty("No feature comparison data")
    agg = df.groupby("features", as_index=False)[metric].mean()
    fig = go.Figure(go.Bar(
        x=agg["features"], y=agg[metric],
        marker={"color": [THEME["blue"], THEME["teal"]][: len(agg)]},
        text=agg[metric].map(lambda v: f"{v:.1%}" if metric != "f1" else f"{v:.3f}"),
        textposition="outside",
    ))
    fig.update_layout(
        yaxis={"tickformat": ".0%" if metric != "f1" else ".2f", "range": [0, float(agg[metric].max()) * 1.2]},
        showlegend=False, height=280, template="smd_pro",
    )
    return fig


def fig_scatter(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return _empty("Adjust filters or upload data")
    fig = px.scatter(
        df, x="accuracy", y="f1", color="model" if "model" in df.columns else None,
        color_discrete_sequence=THEME["series"],
        labels={"accuracy": "Accuracy", "f1": "F1 score", "model": "Model"},
    )
    fig.update_traces(marker={"size": 12, "line": {"width": 1, "color": THEME["bg"]}})
    fig.update_layout(
        xaxis={"tickformat": ".0%"}, yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.14}, height=300, template="smd_pro",
    )
    return fig


def fig_class_balance(features: pd.DataFrame | None) -> go.Figure:
    if features is None or "target_up" not in features.columns:
        return _empty("No market data")
    up = int(features["target_up"].sum())
    down = len(features) - up
    fig = go.Figure(go.Pie(
        labels=["Up days", "Down days"], values=[up, down], hole=0.58,
        marker={"colors": [THEME["teal"], THEME["muted"]]},
    ))
    fig.update_layout(
        showlegend=True, legend={"orientation": "h", "y": -0.05},
        annotations=[{"text": f"{len(features):,}", "showarrow": False, "font": {"size": 17, "color": THEME["text"]}}],
        height=280, template="smd_pro",
    )
    return fig


def fig_validation_side_by_side(df: pd.DataFrame | None, metric: str = "accuracy") -> go.Figure:
    if df is None or len(df) == 0 or metric not in df.columns:
        return _empty("Run: python src/smd_validation.py")
    models = list(dict.fromkeys(df["model"]))
    fig = go.Figure()
    colors = {"GSPC": THEME["teal"], "QQQ": THEME["yellow"]}
    for dataset in ("GSPC", "QQQ"):
        sub = df[df["dataset"] == dataset]
        vals = [float(sub[sub["model"] == m][metric].iloc[0]) if len(sub[sub["model"] == m]) else 0 for m in models]
        label = "S&P 500 (GSPC)" if dataset == "GSPC" else "QQQ (different dataset)"
        text = [f"{v:.1%}" if metric != "f1" else f"{v:.3f}" for v in vals]
        fig.add_trace(go.Bar(name=label, x=models, y=vals, marker_color=colors[dataset], text=text, textposition="outside"))
    ymax = float(df[metric].max())
    fig.update_layout(
        barmode="group",
        yaxis={"tickformat": ".0%" if metric != "f1" else ".3f", "range": [0, ymax * 1.22]},
        legend={"orientation": "h", "y": 1.14}, height=340, template="smd_pro",
        margin={"l": 48, "r": 24, "t": 48, "b": 60},
    )
    return fig


def fig_validation_delta(compare: pd.DataFrame | None) -> go.Figure:
    if compare is None or len(compare) == 0 or "delta_accuracy" not in compare.columns:
        return _empty("No GSPC→QQQ delta table")
    df = compare.sort_values("delta_accuracy")
    colors = [THEME["red"] if v < 0 else THEME["teal"] for v in df["delta_accuracy"]]
    fig = go.Figure(go.Bar(
        x=df["delta_accuracy"], y=df["model"], orientation="h",
        marker_color=colors,
        text=df["delta_accuracy"].map(lambda v: f"{v:+.1%}"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=THEME["muted"])
    fig.update_layout(
        xaxis={"title": "QQQ accuracy − GSPC accuracy", "tickformat": "+.0%"},
        yaxis={"automargin": True}, height=320, template="smd_pro",
        margin={"l": 150, "r": 72, "t": 24, "b": 48},
    )
    return fig


def fig_validation_metric_matrix(df: pd.DataFrame | None) -> go.Figure:
    if df is None or len(df) == 0:
        return _empty("No validation metrics")
    metrics = [c for c in ("accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc") if c in df.columns]
    if not metrics:
        return _empty("No metric columns")
    rows, z, y = [], [], []
    for _, r in df.iterrows():
        y.append(f"{r['dataset']} · {r['model']}")
        z.append([float(r[m]) for m in metrics])
    fig = go.Figure(go.Heatmap(
        z=z, x=[m.replace("_", " ").title() for m in metrics], y=y,
        colorscale=[[0, "#0b1220"], [0.45, "#1e3a5f"], [1, THEME["teal"]]],
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.update_layout(height=360, template="smd_pro", margin={"l": 200, "r": 24, "t": 20, "b": 48})
    return fig


def fig_universe_accuracy(universe: dict | None) -> go.Figure:
    summary = universe_summary_df(universe)
    if len(summary) == 0 or summary["day_test_acc"].isna().all():
        return _empty("Refresh Live signals to train all tickers")
    df = summary.dropna(subset=["day_test_acc"]).sort_values("day_test_acc")
    fig = go.Figure(go.Bar(
        x=df["day_test_acc"], y=df["label"].fillna(df["ticker"]), orientation="h",
        marker_color=THEME["blue"],
        text=df["day_test_acc"].map(lambda v: f"{float(v):.0%}"),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis={"tickformat": ".0%", "range": [0, max(0.7, float(df["day_test_acc"].max()) * 1.15)]},
        yaxis={"automargin": True}, height=360, template="smd_pro",
        margin={"l": 120, "r": 56, "t": 20, "b": 40},
    )
    return fig


def fig_price_years(series: pd.Series | None, title: str = "Close", year: str | int | None = "all") -> go.Figure:
    series = filter_series_by_year(series, year)
    if series is None or len(series) == 0:
        return _empty("Price history missing for this selection")
    s = series.dropna()
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", line={"color": THEME["teal"], "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.2f}<extra></extra>",
    ))
    x_title = f"Year {year}" if year and str(year) != "all" else "Year"
    fig.update_layout(
        height=320, template="smd_pro",
        xaxis={"title": x_title, "hoverformat": "%Y-%m-%d"},
        yaxis={"title": title, "tickformat": ",.2f"},
        margin={"l": 56, "r": 20, "t": 16, "b": 48},
    )
    return fig


def fig_vix_years(vix: pd.DataFrame | None, year: str | int | None = "all") -> go.Figure:
    if vix is None or "Close" not in vix.columns:
        return _empty("VIX history missing")
    s = filter_series_by_year(vix["Close"].dropna(), year)
    if s is None or len(s) == 0:
        return _empty("No VIX data for selected year")
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", line={"color": THEME["yellow"], "width": 1.8},
        fill="tozeroy", fillcolor="rgba(251, 191, 36, 0.12)",
        hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=320, template="smd_pro",
        xaxis={"title": f"Year {year}" if year and str(year) != "all" else "Year"},
        yaxis={"title": "VIX (fear index)"},
        margin={"l": 56, "r": 20, "t": 16, "b": 48},
    )
    return fig


def fig_updown_by_year_from_price(series: pd.Series | None, year: str | int | None = "all") -> go.Figure:
    series = filter_series_by_year(series, year)
    if series is None or len(series) < 10:
        return _empty("Not enough price history for this selection")
    rets = series.pct_change().dropna()
    tmp = pd.DataFrame({"up": (rets > 0).astype(int)}, index=rets.index)
    if year and str(year) != "all":
        tmp["period"] = tmp.index.to_period("M").astype(str)
        x_title = f"Month in {year}"
    else:
        tmp["period"] = tmp.index.year.astype(str)
        x_title = "Year"
    agg = tmp.groupby("period")["up"].agg(up="sum", n="count")
    agg["down"] = agg["n"] - agg["up"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Up days", x=agg.index.astype(str), y=agg["up"], marker_color=THEME["teal"]))
    fig.add_trace(go.Bar(name="Down days", x=agg.index.astype(str), y=agg["down"], marker_color=THEME["red"]))
    fig.update_layout(
        barmode="stack", height=300, template="smd_pro",
        xaxis={"title": x_title}, yaxis={"title": "Trading days"},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_updown_by_year(features: pd.DataFrame | None) -> go.Figure:
    if features is None or "target_up" not in features.columns:
        return _empty("Feature table missing")
    tmp = features.copy()
    tmp["year"] = tmp.index.year
    agg = tmp.groupby("year")["target_up"].agg(up="sum", n="count")
    agg["down"] = agg["n"] - agg["up"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Up days", x=agg.index.astype(str), y=agg["up"], marker_color=THEME["teal"]))
    fig.add_trace(go.Bar(name="Down days", x=agg.index.astype(str), y=agg["down"], marker_color=THEME["red"]))
    fig.update_layout(
        barmode="stack", height=300, template="smd_pro",
        xaxis={"title": "Year"}, yaxis={"title": "Trading days"},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_accuracy_by_year(pred_hist: pd.DataFrame | None, year: str | int | None = "all") -> go.Figure:
    pred_hist = filter_hist_by_year(pred_hist, year)
    if pred_hist is None or len(pred_hist) == 0:
        return _empty("No prediction history for this selection")
    df = pred_hist.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    if "correct" not in df.columns:
        return _empty("No correctness column")
    if year and str(year) != "all":
        # Single year: monthly hit rate
        df["date"] = pd.to_datetime(df["date"])
        df["period"] = df["date"].dt.to_period("M").astype(str)
        agg = df.groupby("period", as_index=False)["correct"].mean()
        x_title = f"Month in {year}"
    else:
        agg = df.groupby("year", as_index=False)["correct"].mean()
        agg = agg.rename(columns={"year": "period"})
        agg["period"] = agg["period"].astype(str)
        x_title = "Year (test-window predictions)"
    fig = go.Figure(go.Bar(
        x=agg["period"].astype(str), y=agg["correct"],
        marker_color=THEME["blue"],
        text=agg["correct"].map(lambda v: f"{v:.0%}"), textposition="outside",
    ))
    fig.update_layout(
        height=300, template="smd_pro",
        xaxis={"title": x_title},
        yaxis={"title": "Hit rate", "tickformat": ".0%", "range": [0, 1]},
        margin={"l": 48, "r": 20, "t": 24, "b": 48},
    )
    return fig


def fig_pred_timeline(pred_hist: pd.DataFrame | None, year: str | int | None = "all") -> go.Figure:
    pred_hist = filter_hist_by_year(pred_hist, year)
    if pred_hist is None or len(pred_hist) == 0:
        return _empty("No prediction history yet")
    df = pred_hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["prob_up"], mode="lines", name="P(Up)",
        line={"color": THEME["teal"], "width": 1.6},
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=THEME["muted"])
    fig.update_layout(
        height=300, template="smd_pro",
        xaxis={"title": f"Date ({year})" if year and str(year) != "all" else "Date", "tickformat": "%b %Y"},
        yaxis={"title": "Probability market goes UP", "tickformat": ".0%", "range": [0, 1]},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_hit_donut(pred_hist: pd.DataFrame | None, year: str | int | None = "all") -> go.Figure:
    pred_hist = filter_hist_by_year(pred_hist, year)
    if pred_hist is None or len(pred_hist) == 0 or "correct" not in pred_hist.columns:
        return _empty("No hit/miss data")
    ok = int(pred_hist["correct"].sum())
    miss = int(len(pred_hist) - ok)
    fig = go.Figure(go.Pie(
        labels=["Correct", "Miss"], values=[ok, miss], hole=0.62,
        marker={"colors": [THEME["teal"], THEME["red"]]},
    ))
    hit = ok / max(len(pred_hist), 1)
    fig.update_layout(
        height=280, template="smd_pro",
        annotations=[{"text": f"{hit:.0%}", "showarrow": False, "font": {"size": 20, "color": THEME["text"]}}],
        legend={"orientation": "h", "y": -0.05},
    )
    return fig


def _stat(label: str, val: str, css: str = "") -> html.Div:
    return html.Div(
        [html.Span(label, className="stat-label"), html.Span(val, className=f"stat-value {css}".strip())],
        className="stat-row",
    )


def build_stats_panel(row: dict | None, filtered_n: int) -> html.Div:
    if not row:
        return html.Div([
            html.H4("Selected run", className="stats-panel-title"),
            html.P(f"{filtered_n} runs match filters. Click a scatter point for details.", className="stat-desc"),
        ], className="stats-panel")
    return html.Div([
        html.H4("Selected run details", className="stats-panel-title"),
        _stat("Model", str(row.get("model", "-"))),
        _stat("Features", str(row.get("features", "-"))),
        _stat("SMOTE", "Yes" if row.get("smote") else "No"),
        _stat("Accuracy", f"{float(row.get('accuracy', 0)):.1%}", "good"),
        _stat("Balanced acc", f"{float(row.get('balanced_accuracy', 0)):.1%}"),
        _stat("F1", f"{float(row.get('f1', 0)):.3f}", "good"),
        _stat("ROC AUC", f"{float(row.get('roc_auc', 0)):.3f}"),
        html.P("Use filters above to narrow the comparison table and charts.", className="stat-desc"),
    ], className="stats-panel")


def kpi_block(title: str, value: str, sub: str, color: str, fig: go.Figure | None = None) -> html.Div:
    body = []
    if fig is not None:
        body.append(html.Div(dcc.Graph(figure=fig, config=GRAPH_CFG), className="widget-body widget-body-chart"))
    return html.Div([
        html.Div([
            html.P(title, className="widget-title"),
            html.P(value, className=f"widget-kpi {color}"),
            html.P(sub, className="widget-sub"),
        ], className="widget-header"),
        *body,
    ], className="widget")


def _table_style():
    return {
        "style_table": {"overflowX": "auto", "backgroundColor": THEME["card"]},
        "style_header": {
            "backgroundColor": "#182235", "color": THEME["text"], "fontWeight": "600",
            "fontSize": "11px", "border": f"1px solid {THEME['grid']}", "textTransform": "uppercase",
        },
        "style_cell": {
            "backgroundColor": THEME["card"], "color": THEME["text"], "fontSize": "12px",
            "border": f"1px solid {THEME['grid']}", "padding": "8px 10px", "minWidth": "80px",
        },
        "style_data_conditional": [{"if": {"row_index": "odd"}, "backgroundColor": "#182235"}],
    }


def parse_upload(contents: str, filename: str) -> tuple[pd.DataFrame | None, str]:
    if not contents:
        return None, "No file."
    try:
        _, raw = contents.split(",", 1)
        df = pd.read_csv(io.StringIO(base64.b64decode(raw).decode("utf-8")))
        df.columns = [c.strip() for c in df.columns]
    except Exception as exc:  # noqa: BLE001
        return None, f"Read error: {exc}"
    lower = {c.lower(): c for c in df.columns}
    if not {"accuracy", "f1"}.issubset(lower):
        return None, "CSV needs accuracy and f1 columns."
    rename = {lower[k]: k for k in lower if lower[k] != k}
    if rename:
        df = df.rename(columns=rename)
    return df, f"Loaded {filename} ({len(df)} rows)."


def filter_options(df: pd.DataFrame) -> dict:
    models = sorted(df["model"].unique()) if len(df) and "model" in df.columns else []
    feats = sorted(df["features"].unique()) if len(df) and "features" in df.columns else []
    return {
        "model": [{"label": "All models", "value": "all"}] + [{"label": m, "value": m} for m in models],
        "features": [{"label": "All feature sets", "value": "all"}] + [{"label": f, "value": f} for f in feats],
        "smote": [
            {"label": "All", "value": "all"},
            {"label": "SMOTE on", "value": "yes"},
            {"label": "SMOTE off", "value": "no"},
        ],
        "metric": [
            {"label": "Accuracy", "value": "accuracy"},
            {"label": "F1 score", "value": "f1"},
            {"label": "Balanced accuracy", "value": "balanced_accuracy"},
            {"label": "ROC AUC", "value": "roc_auc"},
        ],
    }


def tab_intro(title: str, desc: str) -> html.Div:
    return html.Div(className="tab-intro", children=[
        html.H2(title, className="section-title"),
        html.P(desc, className="section-desc"),
    ])


def _stock_card(stock: dict) -> html.Div:
    day = stock.get("next_day") or {}
    week = stock.get("next_week") or {}
    d_dir = str(day.get("direction") or "-")
    w_dir = str(week.get("direction") or "-")
    status = stock.get("status", "ok")
    d_cls = "up" if d_dir == "UP" else ("down" if d_dir == "DOWN" else "")
    w_cls = "up" if w_dir == "UP" else ("down" if w_dir == "DOWN" else "")
    day_p = day.get("probability")
    week_p = week.get("probability")
    acc = day.get("test_accuracy")
    return html.Div(className=f"stock-card {'stock-card-error' if status != 'ok' else ''}", children=[
        html.P(stock.get("label", stock.get("ticker", "?")), className="stock-card-title"),
        html.P(stock.get("ticker", ""), className="stock-card-ticker"),
        html.Div(className="stock-card-row", children=[
            html.Div([
                html.Span("Next day", className="horizon"),
                html.Span(d_dir, className=f"call-sm {d_cls}"),
                html.Span("-" if day_p is None else f"{float(day_p):.0%} P(Up)", className="meta"),
            ]),
            html.Div([
                html.Span("Next week", className="horizon"),
                html.Span(w_dir, className=f"call-sm {w_cls}"),
                html.Span("-" if week_p is None else f"{float(week_p):.0%} P(Up)", className="meta"),
            ]),
        ]),
        html.P(
            stock.get("error") if status != "ok"
            else f"Test acc {float(acc):.0%} · {stock.get('n_rows', '-')} days · as of {day.get('as_of', '-')}",
            className="stock-card-foot",
        ),
    ])


def universe_answer_block(universe: dict | None, live: dict | None) -> html.Div:
    stocks = universe_stocks(universe)
    headline = None
    if live and live.get("quick_answer"):
        headline = live["quick_answer"]
    elif stocks:
        ok = [s for s in stocks if s.get("status") == "ok"]
        if ok:
            headline = ok[0].get("quick_answer")

    children = [
        html.P("Live signals - every stock in the dataset", className="quick-answer-label"),
        html.P(
            headline or "No stock signals yet. Click “Refresh all dataset stocks” below "
            "(or run: python src/smd_dynamic_predict.py).",
            className="quick-answer-text",
        ),
    ]
    if stocks:
        children.append(html.Div(className="stock-grid", children=[_stock_card(s) for s in stocks]))
    else:
        children.append(html.P(
            "Dataset universe: S&P 500, QQQ, TSLA, MSFT, AMZN, GOOGL, META, NVDA, JPM.",
            className="note-text",
        ))
    children.append(html.P(
        "Each card: fetch/train Random Forest on that ticker’s price + shared mood features, "
        "then show next-day and next-week UP/DOWN. Research signals only - not financial advice.",
        className="note-text mt-3 mb-0",
    ))
    return html.Div(className="quick-answer", children=children)


def universe_loading_spinner() -> html.Div:
    return html.Div(
        className="dyn-spinner-box",
        children=[
            html.Div(className="dyn-spinner", **{"aria-hidden": "true"}),
            html.P("Refreshing all dataset stocks…", className="dyn-spinner-title"),
            html.P("Training each ticker one by one (often 2–5 minutes)", className="dyn-spinner-sub"),
            html.Ol(
                className="dyn-spinner-steps",
                children=[
                    html.Li("S&P 500 + QQQ"),
                    html.Li("Basket: TSLA, MSFT, AMZN, GOOGL…"),
                    html.Li("Score next-day / next-week for each"),
                    html.Li("Update Live signals cards"),
                ],
            ),
        ],
    )


def dynamic_loading_spinner() -> html.Div:
    return html.Div(
        className="dyn-spinner-box",
        children=[
            html.Div(className="dyn-spinner", **{"aria-hidden": "true"}),
            html.P("Working on your ticker…", className="dyn-spinner-title"),
            html.P("Usually 20–60 seconds", className="dyn-spinner-sub"),
            html.Ol(
                className="dyn-spinner-steps",
                children=[
                    html.Li("Download price history"),
                    html.Li("Build features + mood"),
                    html.Li("Train Random Forest"),
                    html.Li("Score next-day / next-week"),
                ],
            ),
        ],
    )


def dynamic_result_block(result: dict | None) -> html.Div:
    if not result:
        return html.P("Enter a ticker (e.g. AAPL) and click Fetch, train & predict.", className="note-text")
    if result.get("status") != "ok":
        return html.Div(className="quick-answer", children=[
            html.P("Dynamic result", className="quick-answer-label"),
            html.P(result.get("error") or result.get("quick_answer") or "Failed.", className="quick-answer-text"),
        ])
    day, week = result.get("next_day", {}), result.get("next_week", {})
    d_dir, w_dir = str(day.get("direction")), str(week.get("direction"))
    hist = pd.DataFrame(result.get("history") or [])
    fig = fig_pred_timeline(hist) if len(hist) else _empty("No history")
    return html.Div([
        html.Div(className="quick-answer mb-3", children=[
            html.P(f"Dynamic result · {result.get('label', result.get('ticker'))}", className="quick-answer-label"),
            html.P(result.get("quick_answer", ""), className="quick-answer-text"),
            dbc.Row([
                dbc.Col(html.Div(className="direction-card", children=[
                    html.P("Next trading day", className="horizon"),
                    html.P(d_dir, className=f"call {'up' if d_dir == 'UP' else 'down'}"),
                    html.P(
                        f"P(Up) {float(day.get('probability', 0)):.0%} · "
                        f"test acc {float(day.get('test_accuracy') or 0):.0%} · "
                        f"F1 {float(day.get('test_f1') or 0):.2f}",
                        className="meta",
                    ),
                ]), md=6, className="mb-3 mb-md-0"),
                dbc.Col(html.Div(className="direction-card", children=[
                    html.P("Next week (~5 days)", className="horizon"),
                    html.P(w_dir, className=f"call {'up' if w_dir == 'UP' else 'down'}"),
                    html.P(
                        f"P(Up) {float(week.get('probability', 0)):.0%} · "
                        f"test acc {float(week.get('test_accuracy') or 0):.0%} · "
                        f"{result.get('n_rows')} training days",
                        className="meta",
                    ),
                ]), md=6),
            ]),
            html.P(
                f"Data {result.get('data_start')} → {result.get('data_end')} · "
                f"Model: {result.get('model')}",
                className="note-text mt-3 mb-0",
            ),
        ]),
        html.Div(className="chart-card", children=[
            html.P("Test-window P(Up) for this ticker", className="chart-card-title"),
            dcc.Graph(figure=fig, config=GRAPH_CFG),
        ]),
    ])


def create_app(data: dict | None = None) -> Dash:
    data = data or load_data()
    runs = data["runs"]
    features = data["features"]
    validation = data.get("validation")
    validation_compare = data.get("validation_compare")
    live = data.get("live")
    universe = data.get("universe")
    pred_hist = data.get("pred_hist")
    vix = data.get("vix")
    best = data.get("best")
    prices = data.get("prices") or {}
    opts = filter_options(runs if runs is not None else pd.DataFrame())
    tbl = _table_style()
    run_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in (runs.columns if runs is not None else [])]
    val_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in (validation.columns if validation is not None else [])]
    best_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in (best.columns if best is not None else [])]
    compare_cols = [
        {"name": c.replace("_", " ").title(), "id": c}
        for c in (validation_compare.columns if validation_compare is not None else [])
    ]
    uni_summary = universe_summary_df(universe)
    uni_hist = universe_history_df(universe)
    uni_sum_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in uni_summary.columns]
    uni_hist_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in uni_hist.columns]
    t_opts = ticker_options(universe)
    t_default = default_ticker(universe)
    price_opts = [{"label": f"{k}", "value": k} for k in prices.keys()] or t_opts
    price_default = t_default if t_default in prices else (price_opts[0]["value"] if price_opts else "^GSPC")
    table_ticker_opts = [{"label": "All tickers", "value": "all"}] + t_opts
    y_opts = year_options_from_prices(prices)
    y_default = "all"

    app = Dash(
        _DASH_APP_NAME,
        assets_folder=str(_ASSETS_DIR),
        external_stylesheets=[
            dbc.themes.DARKLY,
            "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="Market Direction Analytics",
    )
    app.index_string = f"""<!DOCTYPE html><html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<style>{CUSTOM_CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

    app.layout = html.Div(className="dashboard-root", children=[
        dcc.Store(id="runs-store", data=to_store(runs)),
        dcc.Store(id="universe-store", data=universe),
        dcc.Store(id="dynamic-result-store", data=None),
        dcc.Store(id="prices-store", data={k: {"x": [str(i.date()) for i in s.index], "y": [float(v) for v in s.values]} for k, s in prices.items()}),
        dcc.Store(
            id="vix-store",
            data=(
                {"x": [str(i.date()) for i in vix["Close"].dropna().index],
                 "y": [float(v) for v in vix["Close"].dropna().values]}
                if vix is not None and "Close" in vix.columns else None
            ),
        ),

        html.Div(className="dash-header", children=dbc.Container(fluid=True, className="px-4", children=dbc.Row([
            dbc.Col([
                html.H1("Market Direction Analytics", className="brand-title"),
                html.P("Signals · models · market years · tables", className="brand-sub"),
            ], md=8),
            dbc.Col(html.Span(f"UI {DASH_UI_VERSION}", className="header-badge d-inline-block mt-2"), md=4, className="text-md-end"),
        ]))),

        dbc.Container(fluid=True, className="px-4 py-3", children=[
            dbc.Tabs(id="main-tabs", active_tab="tab-signals", className="dashboard-tabs mb-3", children=[
                dbc.Tab(label="Stock signals", tab_id="tab-signals", children=html.Div(className="tab-pane-body", children=[
                    tab_intro(
                        "Stock signals",
                        "See next-day / next-week UP or DOWN for every dataset stock, or type any Yahoo ticker to fetch, train, and predict live.",
                    ),
                    html.Div(className="filter-bar mb-3", children=[
                        html.P("Dataset universe", className="filter-bar-title"),
                        dbc.Row([
                            dbc.Col(dbc.Button(
                                "Refresh all dataset stocks", id="btn-refresh-universe",
                                color="info", outline=True,
                            ), md=4),
                            dbc.Col(html.Div(id="universe-refresh-status", className="note-text pt-2",
                                             children="Ready — click refresh to retrain all tickers."), md=8),
                        ]),
                    ]),
                    dcc.Loading(
                        id="universe-loading",
                        type="default",
                        color=THEME["teal"],
                        className="dyn-loading",
                        custom_spinner=universe_loading_spinner(),
                        children=html.Div(id="universe-answer-panel", children=universe_answer_block(universe, live)),
                    ),
                    html.Hr(className="my-4", style={"borderColor": THEME["grid"]}),
                    html.H3("Any ticker (dynamic)", className="section-title"),
                    html.P(
                        "Download price history, build features with shared mood data, train, and show UP/DOWN.",
                        className="section-desc",
                    ),
                    html.Div(className="filter-bar", children=[
                        html.P("Fetch · train · predict", className="filter-bar-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Stock ticker", className="filter-label"),
                                dbc.Input(
                                    id="dyn-ticker", type="text", placeholder="e.g. AAPL, MSFT, TSLA, NVDA",
                                    value="AAPL", className="dashboard-input", debounce=True,
                                ),
                            ], md=5),
                            dbc.Col([
                                html.Label("Action", className="filter-label"),
                                dbc.Button(
                                    "Fetch, train & predict", id="btn-dynamic-run",
                                    color="success", className="w-100",
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Status", className="filter-label"),
                                html.Div(id="dyn-status", className="note-text pt-2",
                                         children="Ready — enter a ticker and run."),
                            ], md=3),
                        ]),
                        html.P(
                            "Takes ~20–60 seconds. Needs internet for new tickers.",
                            className="note-text mt-2 mb-0",
                        ),
                    ]),
                    dcc.Loading(
                        id="dyn-loading",
                        type="default",
                        color=THEME["teal"],
                        className="dyn-loading",
                        custom_spinner=dynamic_loading_spinner(),
                        children=html.Div(
                            id="dyn-result-panel",
                            className="dyn-result-panel",
                            children=dynamic_result_block(None),
                        ),
                    ),
                ])),

                dbc.Tab(label="Models & validation", tab_id="tab-models", children=html.Div(className="tab-pane-body", children=[
                    tab_intro(
                        "Models & validation",
                        "Pick a trained ticker for test metrics. Then review QQQ external validation side by side with S&P 500.",
                    ),
                    html.Div(className="filter-bar", children=[
                        html.P("Trained ticker", className="filter-bar-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Ticker", className="filter-label"),
                                dbc.Select(id="model-ticker", options=t_opts, value=t_default, className="dashboard-select"),
                            ], md=4),
                            dbc.Col(html.Div(id="model-ticker-hint", className="note-text pt-4"), md=8),
                        ]),
                    ]),
                    html.Div(id="ticker-model-panel"),
                    html.Hr(className="my-4", style={"borderColor": THEME["grid"]}),
                    html.H3("External validation · GSPC vs QQQ", className="section-title"),
                    html.P(
                        "Same frozen models on a different market. Transfer gaps show what generalises.",
                        className="section-desc",
                    ),
                    dbc.Row([
                        dbc.Col(kpi_block(
                            "GSPC best accuracy",
                            f"{float(validation[validation['dataset']=='GSPC']['accuracy'].max()):.1%}"
                            if validation is not None and len(validation) else "—",
                            "Side-by-side table", "teal",
                        ), lg=3, className="mb-3"),
                        dbc.Col(kpi_block(
                            "QQQ best accuracy",
                            f"{float(validation[validation['dataset']=='QQQ']['accuracy'].max()):.1%}"
                            if validation is not None and len(validation) else "—",
                            "Different market", "yellow",
                        ), lg=3, className="mb-3"),
                        dbc.Col(kpi_block(
                            "Worst transfer gap",
                            f"{float(validation_compare['delta_accuracy'].min()):+.1%}"
                            if validation_compare is not None and len(validation_compare) else "—",
                            "QQQ − GSPC accuracy", "red",
                        ), lg=3, className="mb-3"),
                        dbc.Col(kpi_block(
                            "Universe tickers",
                            str(len(universe_stocks(universe))),
                            "Trained live signals", "green",
                        ), lg=3, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([html.P("Accuracy · GSPC vs QQQ", className="chart-card-title"),
                                          dcc.Graph(figure=fig_validation_side_by_side(validation, "accuracy"), config=GRAPH_CFG)],
                                         className="chart-card"), lg=6, className="mb-3"),
                        dbc.Col(html.Div([html.P("F1 · GSPC vs QQQ", className="chart-card-title"),
                                          dcc.Graph(figure=fig_validation_side_by_side(validation, "f1"), config=GRAPH_CFG)],
                                         className="chart-card"), lg=6, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([html.P("Transfer gap (QQQ − GSPC accuracy)", className="chart-card-title"),
                                          dcc.Graph(figure=fig_validation_delta(validation_compare), config=GRAPH_CFG)],
                                         className="chart-card"), lg=6, className="mb-3"),
                        dbc.Col(html.Div([html.P("Per-ticker next-day test accuracy", className="chart-card-title"),
                                          dcc.Graph(figure=fig_universe_accuracy(universe), config=GRAPH_CFG)],
                                         className="chart-card"), lg=6, className="mb-3"),
                    ]),
                    html.Div(id="gspc-compare-wrap", children=[
                        html.Hr(className="my-4", style={"borderColor": THEME["grid"]}),
                        html.H3("S&P 500 multi-model comparison", className="section-title"),
                        html.P(
                            "Benchmark bake-off on GSPC (shown when S&P 500 is selected above).",
                            className="section-desc",
                        ),
                        html.Div(className="filter-bar", children=[
                            html.P("Filters for model comparison", className="filter-bar-title"),
                            dbc.Row([
                                dbc.Col([html.Label("Model", className="filter-label"),
                                         dbc.Select(id="f-model", options=opts["model"], value="all", className="dashboard-select")], md=3),
                                dbc.Col([html.Label("Feature set", className="filter-label"),
                                         dbc.Select(id="f-features", options=opts["features"], value="all", className="dashboard-select")], md=3),
                                dbc.Col([html.Label("SMOTE", className="filter-label"),
                                         dbc.Select(id="f-smote", options=opts["smote"], value="all", className="dashboard-select")], md=3),
                                dbc.Col([html.Label("Highlight metric", className="filter-label"),
                                         dbc.Select(id="f-metric", options=opts["metric"], value="accuracy", className="dashboard-select")], md=3),
                            ]),
                            html.Div(id="filter-hint", className="filter-active-hint"),
                        ]),
                        html.Div(id="kpi-row"),
                        dbc.Row([
                            dbc.Col(html.Div([html.P(id="chart-title-metric", className="chart-card-title"),
                                              dcc.Graph(id="chart-metric", config=GRAPH_CFG)], className="chart-card"), lg=7, className="mb-3"),
                            dbc.Col(html.Div(id="stats-panel"), lg=5, className="mb-3"),
                        ]),
                        dbc.Row([
                            dbc.Col(html.Div([html.P("Accuracy vs F1 (all filtered runs)", className="chart-card-title"),
                                              dcc.Graph(id="chart-scatter", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                            dbc.Col(html.Div([html.P("Average score by feature set", className="chart-card-title"),
                                              dcc.Graph(id="chart-features", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                        ]),
                        dbc.Row([
                            dbc.Col(html.Div([html.P("Model × metric heatmap (best per model)", className="chart-card-title"),
                                              dcc.Graph(id="chart-heatmap", config=GRAPH_CFG)], className="chart-card"), lg=7, className="mb-3"),
                            dbc.Col(html.Div([html.P("Up vs down days in training data", className="chart-card-title"),
                                              dcc.Graph(figure=fig_class_balance(features), config=GRAPH_CFG)], className="chart-card"), lg=5, className="mb-3"),
                        ]),
                    ]),
                ])),

                dbc.Tab(label="Market & years", tab_id="tab-market", children=html.Div(className="tab-pane-body", children=[
                    tab_intro(
                        "Market & years",
                        "Choose a ticker and a year. Charts update like ticker/model filters — price, VIX, up/down days, and hit-rate.",
                    ),
                    html.Div(className="filter-bar", children=[
                        html.P("Ticker & year selection", className="filter-bar-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Ticker", className="filter-label"),
                                dbc.Select(
                                    id="market-ticker", options=price_opts, value=price_default,
                                    className="dashboard-select",
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Year", className="filter-label"),
                                dbc.Select(
                                    id="market-year", options=y_opts, value=y_default,
                                    className="dashboard-select",
                                ),
                            ], md=3),
                            dbc.Col(html.Div(id="market-ticker-hint", className="note-text pt-4"), md=5),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([html.P(id="market-price-title", className="chart-card-title"),
                                          dcc.Graph(id="market-price-chart", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                        dbc.Col(html.Div([html.P(id="market-vix-title", className="chart-card-title", children="VIX (fear)"),
                                          dcc.Graph(id="market-vix-chart", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([html.P(id="market-updown-title", className="chart-card-title", children="Up vs down days"),
                                          dcc.Graph(id="market-updown-chart", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                        dbc.Col(html.Div([html.P(id="market-acc-title", className="chart-card-title", children="Model hit rate"),
                                          dcc.Graph(id="market-acc-year-chart", config=GRAPH_CFG)], className="chart-card"), lg=6, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([html.P(id="market-prob-title", className="chart-card-title", children="Test-window P(Up)"),
                                          dcc.Graph(id="market-prob-chart", config=GRAPH_CFG)], className="chart-card"), lg=12, className="mb-3"),
                    ]),
                ])),

                dbc.Tab(label="Data tables", tab_id="tab-tables", children=html.Div(className="tab-pane-body", children=[
                    tab_intro(
                        "Data tables",
                        "Universe metrics, prediction history, validation proof, and comparison grids. Filter history by ticker.",
                    ),
                    html.Div(className="filter-bar", children=[
                        html.P("History filter", className="filter-bar-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Ticker", className="filter-label"),
                                dbc.Select(
                                    id="table-ticker", options=table_ticker_opts, value="all",
                                    className="dashboard-select",
                                ),
                            ], md=4),
                        ]),
                    ]),
                    html.P("Live signals — all dataset tickers", className="chart-card-title mb-2"),
                    dash_table.DataTable(
                        columns=uni_sum_cols, data=to_store(uni_summary),
                        sort_action="native", page_size=12, **tbl,
                    ) if len(uni_summary) else html.P("Refresh Stock signals first.", className="note-text"),
                    html.P("Prediction history", className="chart-card-title mt-4 mb-2"),
                    dash_table.DataTable(
                        id="universe-hist-table",
                        columns=uni_hist_cols or [
                            {"name": c, "id": c}
                            for c in ("ticker", "label", "date", "year", "prob_up", "pred_label", "actual_label", "correct")
                        ],
                        data=to_store(uni_hist),
                        sort_action="native", filter_action="native", page_size=12, **tbl,
                    ),
                    html.P("Best model per family (GSPC bake-off)", className="chart-card-title mt-4 mb-2"),
                    dash_table.DataTable(
                        columns=best_cols, data=to_store(best), sort_action="native", page_size=6, **tbl,
                    ) if best is not None else html.P("Run model comparison first.", className="note-text"),
                    html.P("Validation side-by-side (GSPC vs QQQ)", className="chart-card-title mt-4 mb-2"),
                    dash_table.DataTable(
                        columns=val_cols, data=to_store(validation), sort_action="native", page_size=8, **tbl,
                    ) if validation is not None else html.P("Run python src/smd_validation.py", className="note-text"),
                    html.P("Transfer deltas (previous vs validation)", className="chart-card-title mt-4 mb-2"),
                    dash_table.DataTable(
                        columns=compare_cols, data=to_store(validation_compare),
                        sort_action="native", page_size=8, **tbl,
                    ) if validation_compare is not None else html.P("No compare table.", className="note-text"),
                    html.P("Filtered comparison grid (GSPC)", className="chart-card-title mt-4 mb-2"),
                    dash_table.DataTable(
                        id="results-table", columns=run_cols, data=to_store(runs),
                        sort_action="native", filter_action="native", page_size=10, **tbl,
                    ),
                    html.Div(className="mt-4", children=[
                        html.P("Import optional results CSV", className="section-title"),
                        dcc.Upload(id="csv-upload", className="upload-zone", accept=".csv", children=html.Div([
                            html.Div("Upload results CSV", className="upload-zone-title"),
                            html.P("Needs accuracy & f1 columns", className="upload-zone-hint"),
                        ])),
                        html.Div(id="upload-msg", className="upload-status"),
                        html.Div(id="upload-preview"),
                    ]),
                ])),
            ]),

            html.Div(
                "Tabs: Stock signals · Models & validation · Market & years · Data tables",
                className="dash-footer",
            ),
        ]),
    ])

    @callback(
        Output("runs-store", "data"),
        Output("upload-msg", "children"),
        Output("f-model", "options"),
        Output("f-features", "options"),
        Input("csv-upload", "contents"),
        State("csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def on_upload(contents, filename):
        df, msg = parse_upload(contents, filename or "upload.csv")
        if df is None:
            return no_update, msg, no_update, no_update
        o = filter_options(df)
        return to_store(df), msg, o["model"], o["features"]

    clientside_callback(
        """
        function(nClicks) {
            if (!nClicks) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            return [
                "Running… training every dataset ticker (often 2–5 minutes)",
                true,
                "Working…"
            ];
        }
        """,
        Output("universe-refresh-status", "children", allow_duplicate=True),
        Output("btn-refresh-universe", "disabled", allow_duplicate=True),
        Output("btn-refresh-universe", "children", allow_duplicate=True),
        Input("btn-refresh-universe", "n_clicks"),
        prevent_initial_call=True,
    )

    @callback(
        Output("universe-store", "data"),
        Output("universe-refresh-status", "children"),
        Output("universe-answer-panel", "children"),
        Output("btn-refresh-universe", "disabled"),
        Output("btn-refresh-universe", "children"),
        Output("model-ticker", "options"),
        Output("table-ticker", "options"),
        Input("btn-refresh-universe", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_universe(n_clicks):
        btn = "Refresh all dataset stocks"
        if not n_clicks:
            return (no_update,) * 7
        try:
            from smd_dynamic_predict import predict_dataset_universe  # noqa: PLC0415
            payload = predict_dataset_universe(n_estimators=160)
            status = f"Done · updated {payload.get('n_stocks', 0)} stocks."
            t_opts_new = ticker_options(payload)
            table_opts = [{"label": "All tickers", "value": "all"}] + t_opts_new
            return (
                payload, status, universe_answer_block(payload, live), False, btn,
                t_opts_new, table_opts,
            )
        except Exception as exc:  # noqa: BLE001
            return no_update, f"Refresh failed: {exc}", no_update, False, btn, no_update, no_update

    clientside_callback(
        """
        function(nClicks) {
            if (!nClicks) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            return [
                "Running… download → features → train → predict (often 20–60s)",
                true,
                "Working…"
            ];
        }
        """,
        Output("dyn-status", "children", allow_duplicate=True),
        Output("btn-dynamic-run", "disabled", allow_duplicate=True),
        Output("btn-dynamic-run", "children", allow_duplicate=True),
        Input("btn-dynamic-run", "n_clicks"),
        prevent_initial_call=True,
    )

    @callback(
        Output("dynamic-result-store", "data"),
        Output("dyn-status", "children"),
        Output("dyn-result-panel", "children"),
        Output("btn-dynamic-run", "disabled"),
        Output("btn-dynamic-run", "children"),
        Input("btn-dynamic-run", "n_clicks"),
        State("dyn-ticker", "value"),
        prevent_initial_call=True,
    )
    def run_dynamic(n_clicks, ticker):
        btn_label = "Fetch, train & predict"
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update
        if not ticker or not str(ticker).strip():
            err = {"status": "error", "error": "Please enter a ticker."}
            return err, "Missing ticker", dynamic_result_block(err), False, btn_label
        try:
            from smd_dynamic_predict import cache_dynamic_result, train_and_predict  # noqa: PLC0415
            result = train_and_predict(str(ticker).strip(), n_estimators=180)
            cache_dynamic_result(result)
            status = f"Done · {result.get('ticker')} · {result.get('quick_answer')}"
            return result, status, dynamic_result_block(result), False, btn_label
        except Exception as exc:  # noqa: BLE001
            err = {"status": "error", "error": str(exc), "quick_answer": str(exc)}
            return err, f"Error: {exc}", dynamic_result_block(err), False, btn_label

    @callback(
        Output("ticker-model-panel", "children"),
        Output("model-ticker-hint", "children"),
        Output("gspc-compare-wrap", "style"),
        Input("model-ticker", "value"),
        Input("universe-store", "data"),
    )
    def render_ticker_model(ticker, universe_data):
        stock = find_stock(universe_data, ticker)
        show_compare = {"display": "block"} if ticker in ("^GSPC", "GSPC") else {"display": "none"}
        if not stock or stock.get("status") != "ok":
            msg = f"No trained result for {ticker}. Refresh Stock signals first."
            return html.P(msg, className="note-text"), msg, show_compare
        day = stock.get("next_day") or {}
        week = stock.get("next_week") or {}
        hist = stock_history_df(stock)
        hint = (
            f"{stock.get('label')} · {stock.get('model')} · "
            f"{stock.get('data_start')} → {stock.get('data_end')} · {stock.get('n_rows')} days"
        )
        acc = float(day.get("test_accuracy") or 0)
        f1 = float(day.get("test_f1") or 0)
        bal = float(day.get("test_balanced_accuracy") or 0)
        panel = html.Div([
            dbc.Row([
                dbc.Col(kpi_block("Next day", str(day.get("direction", "—")),
                                  f"P(Up) {float(day.get('probability') or 0):.0%}", "teal"), lg=3, className="mb-3"),
                dbc.Col(kpi_block("Next week", str(week.get("direction", "—")),
                                  f"P(Up) {float(week.get('probability') or 0):.0%}", "yellow"), lg=3, className="mb-3"),
                dbc.Col(kpi_block("Test accuracy", f"{acc:.1%}", "Holdout window", "blue"), lg=3, className="mb-3"),
                dbc.Col(kpi_block("Test F1", f"{f1:.3f}", f"Balanced acc {bal:.1%}", "green"), lg=3, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(html.Div([html.P("P(Up) over test window", className="chart-card-title"),
                                  dcc.Graph(figure=fig_pred_timeline(hist), config=GRAPH_CFG)], className="chart-card"),
                        lg=7, className="mb-3"),
                dbc.Col(html.Div([html.P("Hit vs miss", className="chart-card-title"),
                                  dcc.Graph(figure=fig_hit_donut(hist), config=GRAPH_CFG)], className="chart-card"),
                        lg=5, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(html.Div([html.P("Hit rate by year", className="chart-card-title"),
                                  dcc.Graph(figure=fig_accuracy_by_year(hist), config=GRAPH_CFG)], className="chart-card"),
                        lg=12, className="mb-3"),
            ]),
            html.P(stock.get("quick_answer", ""), className="note-text"),
        ])
        return panel, hint, show_compare

    @callback(
        Output("market-price-title", "children"),
        Output("market-price-chart", "figure"),
        Output("market-vix-title", "children"),
        Output("market-vix-chart", "figure"),
        Output("market-updown-title", "children"),
        Output("market-updown-chart", "figure"),
        Output("market-acc-title", "children"),
        Output("market-acc-year-chart", "figure"),
        Output("market-prob-title", "children"),
        Output("market-prob-chart", "figure"),
        Output("market-ticker-hint", "children"),
        Input("market-ticker", "value"),
        Input("market-year", "value"),
        Input("prices-store", "data"),
        Input("vix-store", "data"),
        Input("universe-store", "data"),
    )
    def refresh_market(ticker, year, price_data, vix_data, universe_data):
        year = year or "all"
        year_label = "all years" if str(year) == "all" else str(year)
        series = None
        if price_data and ticker in price_data:
            payload = price_data[ticker]
            series = pd.Series(payload["y"], index=pd.to_datetime(payload["x"]))
        vix_series = None
        if vix_data:
            vix_series = pd.Series(vix_data["y"], index=pd.to_datetime(vix_data["x"]))
        stock = find_stock(universe_data, ticker)
        hist = stock_history_df(stock)
        hint = f"Showing {ticker} · {year_label}."
        if stock and stock.get("status") == "ok":
            hint += f" Model test acc {float((stock.get('next_day') or {}).get('test_accuracy') or 0):.0%}."
        vix_df = None
        if vix_series is not None:
            vix_df = pd.DataFrame({"Close": vix_series})
        return (
            f"{ticker} close · {year_label}",
            fig_price_years(series, title=f"{ticker} close", year=year),
            f"VIX (fear) · {year_label}",
            fig_vix_years(vix_df, year=year),
            f"Up vs down · {year_label}",
            fig_updown_by_year_from_price(series, year=year),
            f"Model hit rate · {year_label}",
            fig_accuracy_by_year(hist, year=year),
            f"Test-window P(Up) · {year_label}",
            fig_pred_timeline(hist, year=year),
            hint,
        )

    @callback(
        Output("universe-hist-table", "data"),
        Input("table-ticker", "value"),
        Input("universe-store", "data"),
    )
    def filter_hist_table(ticker, universe_data):
        df = universe_history_df(universe_data)
        if len(df) == 0:
            return []
        if ticker and ticker != "all":
            df = df[df["ticker"] == ticker]
        return to_store(df)

    @callback(
        Output("kpi-row", "children"),
        Output("chart-metric", "figure"),
        Output("chart-title-metric", "children"),
        Output("chart-scatter", "figure"),
        Output("chart-features", "figure"),
        Output("chart-heatmap", "figure"),
        Output("stats-panel", "children"),
        Output("results-table", "data"),
        Output("filter-hint", "children"),
        Input("runs-store", "data"),
        Input("f-model", "value"),
        Input("f-features", "value"),
        Input("f-smote", "value"),
        Input("f-metric", "value"),
        Input("chart-scatter", "clickData"),
    )
    def refresh_all(records, f_model, f_features, f_smote, f_metric, click_data):
        df = from_store(records)
        filt = apply_filters(df, f_model, f_features, f_smote)
        metric = f_metric or "accuracy"
        hint = f"Showing {len(filt)} of {len(df)} runs · focus metric: {metric.replace('_', ' ')}"
        if len(filt) == 0:
            empty = _empty("No runs match filters")
            kpis = dbc.Row([dbc.Col(kpi_block("-", "-", "Adjust filters", "teal", empty), lg=3)] * 4)
            return kpis, empty, "No data", empty, empty, empty, build_stats_panel(None, 0), [], hint

        top = filt.sort_values(metric, ascending=False).iloc[0].to_dict()
        if ctx.triggered_id == "chart-scatter" and click_data:
            pt = click_data["points"][0]
            acc, f1v = pt.get("x"), pt.get("y")
            match = filt[(filt["accuracy"].round(6) == round(acc, 6)) & (filt["f1"].round(6) == round(f1v, 6))]
            if len(match):
                top = match.iloc[0].to_dict()

        acc = float(filt["accuracy"].max())
        f1 = float(filt["f1"].max())
        bal = float(filt["balanced_accuracy"].max()) if "balanced_accuracy" in filt.columns else 0
        kpis = dbc.Row([
            dbc.Col(kpi_block("Best accuracy", f"{acc:.1%}", "In filtered set", "teal",
                              fig_sparkline(filt, "accuracy", THEME["teal"])), lg=3, className="mb-3"),
            dbc.Col(kpi_block("Best F1", f"{f1:.3f}", "In filtered set", "yellow",
                              fig_sparkline(filt, "f1", THEME["yellow"])), lg=3, className="mb-3"),
            dbc.Col(kpi_block("Best balanced acc", f"{bal:.1%}", "Fair Up/Down score", "blue",
                              fig_sparkline(filt, "balanced_accuracy", THEME["blue"])), lg=3, className="mb-3"),
            dbc.Col(kpi_block("Configurations", str(len(filt)), f"of {len(df)} total", "green",
                              fig_sparkline(filt, metric, THEME["green"])), lg=3, className="mb-3"),
        ])
        return (
            kpis,
            fig_metric_by_model(filt, metric),
            f"Best {metric.replace('_', ' ')} by model",
            fig_scatter(filt),
            fig_feature_compare(filt, metric),
            fig_metrics_heatmap(filt),
            build_stats_panel(top, len(filt)),
            filt.to_dict("records"),
            hint,
        )

    @callback(Output("upload-preview", "children"), Input("runs-store", "data"))
    def preview(records):
        df = from_store(records)
        if len(df) == 0:
            return html.P("Upload a CSV to preview.", className="note-text")
        prev = df.head(5)
        c = [{"name": x, "id": x} for x in prev.columns[:8]]
        return html.Div([
            html.P(f"Preview · {len(df)} rows", className="note-text mb-2"),
            dash_table.DataTable(columns=c, data=prev.iloc[:, :8].to_dict("records"), page_size=5, **_table_style()),
        ])

    return app


def main() -> None:
    print("=" * 70)
    print("Market Direction Analytics Dashboard")
    print("=" * 70)
    data = load_data()
    if data["runs"] is None:
        print(" ! run: python src/smd_model_comparison.py")
    if data.get("universe") is None:
        print(" ! run: python src/smd_dynamic_predict.py   (Live signals for all dataset stocks)")
    if data.get("live") is None:
        print(" ! optional: python src/smd_live_predict.py")
    app = create_app(data)
    url = f"http://{C.DASHBOARD_HOST}:{C.DASHBOARD_PORT}"
    print(f" UI {DASH_UI_VERSION} · {url} · Ctrl+F5 if cached")
    app.run(host=C.DASHBOARD_HOST, port=C.DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()
