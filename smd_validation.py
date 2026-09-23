"""Sentiment-Driven Market Direction Prediction. Sprint 4.

Validation and Visualisation.

What this script does:
  1. Completes the improved Sprint 2.2 model configs (freeze best settings).
  2. Validates at least two best models on a DIFFERENT dataset (QQQ next-day
     Up/Down) using the same chronological train/test protocol.
  3. Compares QQQ results with previous S&P 500 (GSPC) performance indicators.
  4. Writes side-by-side CSVs and charts for the dashboard / report.

Run (after Sprint 1–2 data exists):
    python src/smd_preprocessing.py
    python src/smd_model_comparison.py   # optional: refresh GSPC grid
    python src/smd_validation.py
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import smd_config as C
from smd_features import MOOD_FEATURES, PRICE_FEATURES, TARGET
from smd_model_comparison import (
    evaluate_ensemble_soft_vote,
    evaluate_tuned_tree,
    time_split,
    _try_smote,
)
from smd_preprocessing import build_features

# Frozen Sprint 2.2 winners (honest pick + strong second + ensemble check).
# These are the "completed improvements" we validate externally.
VALIDATED_MODELS = (
    {
        "model": "Random Forest",
        "features": "price plus mood",
        "smote": True,
        "factory": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "kind": "tree",
    },
    {
        "model": "HistGradientBoosting",
        "features": "price plus mood",
        "smote": False,
        "factory": lambda: HistGradientBoostingClassifier(
            max_depth=7,
            learning_rate=0.05,
            max_iter=500,
            min_samples_leaf=16,
            random_state=42,
        ),
        "kind": "tree",
    },
    {
        "model": "Ensemble soft vote",
        "features": "price plus mood",
        "smote": True,
        "factory": None,
        "kind": "ensemble",
    },
)


def _available_cols(df: pd.DataFrame) -> list[str]:
    price = [c for c in PRICE_FEATURES if c in df.columns]
    mood = [c for c in MOOD_FEATURES if c in df.columns]
    return price + mood


def _load_or_build_features(target: str, path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 100 and TARGET in df.columns:
            print(f" - using existing {path.name} ({len(df)} rows)")
            return df
    print(f" - building features for {target} → {path.name}")
    df = build_features(target=target)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def _drop_constants(df: pd.DataFrame) -> pd.DataFrame:
    constant = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    if constant:
        print(f"   dropping constant columns: {constant}")
        df = df.drop(columns=constant)
    return df


def evaluate_on_dataset(
    df: pd.DataFrame,
    dataset: str,
    smote_cls,
) -> list[dict]:
    """Train/test the frozen models on one market feature table."""
    df = _drop_constants(df)
    cols = _available_cols(df)
    if not cols:
        raise SystemExit(f"No usable feature columns for {dataset}")
    train, test = time_split(df, train_frac=0.8)
    rows: list[dict] = []
    print(f" - {dataset}: train={len(train)}  test={len(test)}  features={len(cols)}")

    for spec in VALIDATED_MODELS:
        use_smote = bool(spec["smote"]) and smote_cls is not None
        if spec["kind"] == "ensemble":
            metrics = evaluate_ensemble_soft_vote(
                train, test, cols, use_smote=use_smote, smote_cls=smote_cls,
            )
        else:
            metrics = evaluate_tuned_tree(
                spec["factory"](),
                train,
                test,
                cols,
                use_smote=use_smote,
                smote_cls=smote_cls,
            )
        rows.append({
            "dataset": dataset,
            "model": spec["model"],
            "features": spec["features"],
            "smote": use_smote,
            "n_train": len(train),
            "n_test": len(test),
            "date_train_start": str(train.index.min().date()),
            "date_train_end": str(train.index.max().date()),
            "date_test_start": str(test.index.min().date()),
            "date_test_end": str(test.index.max().date()),
            **metrics,
        })
        print(
            f"   {spec['model']:22s}  acc={metrics['accuracy']:.3f}  "
            f"bal={metrics['balanced_accuracy']:.3f}  f1={metrics['f1']:.3f}"
        )
    return rows


def load_previous_gspc_indicators() -> pd.DataFrame:
    """Previous performance indicators from Sprint 2.2 best_per_model.csv."""
    path = C.COMPARISON_DIR / "best_per_model.csv"
    if not path.exists():
        return pd.DataFrame()
    prev = pd.read_csv(path)
    names = {s["model"] for s in VALIDATED_MODELS}
    prev = prev[prev["model"].isin(names)].copy()
    prev.insert(0, "source", "Sprint 2.2 GSPC (previous)")
    prev.insert(1, "dataset", "GSPC")
    return prev


def build_comparison_table(side: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Wide table: each model with GSPC previous vs GSPC recheck vs QQQ validation."""
    metric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
    ]
    frames = []

    gspc = side[side["dataset"] == "GSPC"].set_index("model")
    qqq = side[side["dataset"] == "QQQ"].set_index("model")

    for model in gspc.index.intersection(qqq.index):
        row = {"model": model}
        for m in metric_cols:
            row[f"gspc_{m}"] = float(gspc.loc[model, m])
            row[f"qqq_{m}"] = float(qqq.loc[model, m])
            row[f"delta_{m}"] = row[f"qqq_{m}"] - row[f"gspc_{m}"]
        if len(previous) and model in set(previous["model"]):
            p = previous[previous["model"] == model].iloc[0]
            row["previous_gspc_accuracy"] = float(p["accuracy"])
            row["previous_gspc_f1"] = float(p["f1"])
            row["previous_gspc_balanced_accuracy"] = float(p.get("balanced_accuracy", np.nan))
        frames.append(row)
    return pd.DataFrame(frames)


def plot_side_by_side(side: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar charts: GSPC vs QQQ for accuracy, F1, balanced accuracy."""
    metrics = [
        ("accuracy", "Accuracy", ".0%"),
        ("f1", "F1 score", ".2f"),
        ("balanced_accuracy", "Balanced accuracy", ".0%"),
    ]
    models = list(dict.fromkeys(side["model"]))
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, title, _fmt) in zip(axes, metrics):
        gspc_vals = [
            float(side[(side["model"] == m) & (side["dataset"] == "GSPC")][col].iloc[0])
            for m in models
        ]
        qqq_vals = [
            float(side[(side["model"] == m) & (side["dataset"] == "QQQ")][col].iloc[0])
            for m in models
        ]
        bars1 = ax.bar(x - width / 2, gspc_vals, width, label="GSPC (S&P 500)", color="#2dd4bf")
        bars2 = ax.bar(x + width / 2, qqq_vals, width, label="QQQ (validation)", color="#fbbf24")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_ylim(0, max(gspc_vals + qqq_vals + [0.6]) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                label = f"{h:.1%}" if col != "f1" else f"{h:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, h, label, ha="center", va="bottom", fontsize=7)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Sprint 4 · Side-by-side validation (GSPC vs QQQ)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_side_by_side.png", dpi=140)
    plt.close(fig)

    # One focused accuracy comparison for slides / dashboard fallback.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    gspc_vals = [
        float(side[(side["model"] == m) & (side["dataset"] == "GSPC")]["accuracy"].iloc[0])
        for m in models
    ]
    qqq_vals = [
        float(side[(side["model"] == m) & (side["dataset"] == "QQQ")]["accuracy"].iloc[0])
        for m in models
    ]
    ax.bar(x - width / 2, gspc_vals, width, label="GSPC previous market", color="#60a5fa")
    ax.bar(x + width / 2, qqq_vals, width, label="QQQ different dataset", color="#a78bfa")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_title("Best models: previous (GSPC) vs validation (QQQ)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_accuracy_compare.png", dpi=140)
    plt.close(fig)


def write_markdown(side: pd.DataFrame, compare: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Sprint 4 - Validation and visualisation",
        "",
        "## What we did",
        "",
        "1. **Completed model improvements** - froze the best Sprint 2.2 configs "
        "(Random Forest + HistGradientBoosting + Ensemble soft vote) instead of "
        "re-searching the full 24-run grid.",
        "2. **Different dataset** - rebuilt the same feature recipe with **QQQ** "
        "next-day Up/Down as the target (Nasdaq-100 ETF). Mood features stay shared; "
        "price features come from QQQ.",
        "3. **Same protocol** - chronological 80/20 train/test, calibration threshold "
        "tuning, SMOTE only where the frozen config used it.",
        "4. **Side-by-side** - GSPC vs QQQ metrics in CSV + PNG charts.",
        "",
        "## Results (accuracy)",
        "",
        "| Model | GSPC (recheck) | QQQ (validation) | Δ accuracy |",
        "|-------|----------------|------------------|------------|",
    ]
    for _, r in compare.iterrows():
        lines.append(
            f"| {r['model']} | {r['gspc_accuracy']:.1%} | {r['qqq_accuracy']:.1%} | "
            f"{r['delta_accuracy']:+.1%} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- If QQQ scores stay close to GSPC, the model generalises beyond one index.",
        "- If QQQ drops a lot, the GSPC result was more market-specific.",
        "- Modest accuracy (~50–60%) is expected for next-day direction; look at "
        "balanced accuracy and F1 too.",
        "",
        "## Files",
        "",
        "- `validation_side_by_side.csv` - long table (dataset × model × metrics)",
        "- `previous_vs_validation.csv` - wide GSPC vs QQQ deltas",
        "- `validation_side_by_side.png` - grouped bars",
        "- `validation_accuracy_compare.png` - accuracy focus chart",
        "",
    ]
    (out_dir / "sprint4_validation.md").write_text("\n".join(lines), encoding="utf-8")


def write_html_overview(side: pd.DataFrame, compare: pd.DataFrame, out_dir: Path) -> None:
    rows_html = []
    for _, r in compare.iterrows():
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(r['model']))}</td>"
            f"<td>{r['gspc_accuracy']:.1%}</td>"
            f"<td>{r['qqq_accuracy']:.1%}</td>"
            f"<td>{r['delta_accuracy']:+.1%}</td>"
            f"<td>{r['gspc_f1']:.3f}</td>"
            f"<td>{r['qqq_f1']:.3f}</td>"
            "</tr>"
        )
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sprint 4 Validation</title>
<style>
body {{ font-family: Segoe UI, sans-serif; background:#0c0e14; color:#e2e8f0; padding:24px; }}
table {{ border-collapse: collapse; width:100%; max-width:900px; }}
th, td {{ border:1px solid #2a3042; padding:8px 12px; text-align:left; }}
th {{ background:#161922; }}
img {{ max-width:100%; margin-top:16px; border:1px solid #2a3042; }}
h1,h2 {{ color:#2dd4bf; }}
</style></head><body>
<h1>Sprint 4 - Side-by-side validation</h1>
<p>Best improved models on GSPC vs a different dataset (QQQ).</p>
<table>
<tr><th>Model</th><th>GSPC acc</th><th>QQQ acc</th><th>Δ acc</th><th>GSPC F1</th><th>QQQ F1</th></tr>
{''.join(rows_html)}
</table>
<h2>Charts</h2>
<img src="validation_side_by_side.png" alt="side by side metrics"/>
<img src="validation_accuracy_compare.png" alt="accuracy compare"/>
</body></html>"""
    (out_dir / "validation_overview.html").write_text(body, encoding="utf-8")


def main() -> None:
    print("=" * 70)
    print("Sprint 4  Validation and Visualisation")
    print("=" * 70)

    C.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Always refresh QQQ features so validation uses the latest raw data.
    print("\n[1] Feature tables")
    if not C.FEATURES_FILE.exists():
        gspc_df = build_features(target="GSPC")
        gspc_df.to_csv(C.FEATURES_FILE)
        print(f" - wrote {C.FEATURES_FILE.name}")
    else:
        gspc_df = pd.read_csv(C.FEATURES_FILE, index_col=0, parse_dates=True)
        print(f" - using existing {C.FEATURES_FILE.name} ({len(gspc_df)} rows)")

    qqq_df = build_features(target="QQQ")
    qqq_df.to_csv(C.FEATURES_QQQ_FILE)
    print(f" - wrote {C.FEATURES_QQQ_FILE.name} ({len(qqq_df)} rows)")

    smote_cls = _try_smote()
    if smote_cls is None:
        print(" ! SMOTE not installed - RF/Ensemble will run without oversampling")

    print("\n[2] Validate frozen improved models")
    rows = []
    rows.extend(evaluate_on_dataset(gspc_df, "GSPC", smote_cls))
    rows.extend(evaluate_on_dataset(qqq_df, "QQQ", smote_cls))
    side = pd.DataFrame(rows)
    side_path = C.VALIDATION_DIR / "validation_side_by_side.csv"
    side.to_csv(side_path, index=False)
    print(f" - saved {side_path}")

    print("\n[3] Compare with previous GSPC indicators")
    previous = load_previous_gspc_indicators()
    if len(previous):
        previous.to_csv(C.VALIDATION_DIR / "previous_gspc_indicators.csv", index=False)
    compare = build_comparison_table(side, previous)
    compare_path = C.VALIDATION_DIR / "previous_vs_validation.csv"
    compare.to_csv(compare_path, index=False)
    print(f" - saved {compare_path}")

    print("\n[4] Side-by-side charts")
    plot_side_by_side(side, C.VALIDATION_DIR)
    write_markdown(side, compare, C.VALIDATION_DIR)
    write_html_overview(side, compare, C.VALIDATION_DIR)
    print(f" - charts + markdown + HTML in {C.VALIDATION_DIR}")

    print("\nDone. Open the dashboard (python src/smd_dashboard.py) for interactive side-by-side.")
    print("=" * 70)


if __name__ == "__main__":
    main()
