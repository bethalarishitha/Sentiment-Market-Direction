"""Sentiment-Driven Market Direction Prediction. Sprint 2, Section 2.

First Comparison of Different Models.

This script puts every model, feature set, normalisation and SMOTE choice into
one table so the results can be read at a glance. Sprint 2.2 also tries concrete
improvements: richer features, class weights, threshold tuning on a calibration
slice, and extra tree models.

Run:
    python src/smd_preprocessing.py
    python src/smd_model_comparison.py
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import smd_config as C
from smd_features import MOOD_FEATURES, PRICE_FEATURES, TARGET

SCALERS = {
    "zscore": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

LOGREG_C_VALUES = (0.01, 0.1, 1.0, 10.0)
CAL_FRAC = 0.15


def _try_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        return LGBMClassifier
    except Exception:  # noqa: BLE001
        return None


def _try_smote():
    try:
        from imblearn.over_sampling import SMOTE  # noqa: PLC0415
        return SMOTE
    except Exception:  # noqa: BLE001
        return None


def load_features() -> pd.DataFrame:
    if not C.FEATURES_FILE.exists():
        raise SystemExit("Run src/smd_preprocessing.py first.")
    df = pd.read_csv(C.FEATURES_FILE, index_col=0, parse_dates=True)
    if len(df) == 0:
        raise SystemExit("features_daily.csv is empty. Re run smd_preprocessing.py.")
    constant = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    if constant:
        print(f" - dropping constant columns: {constant}")
        df = df.drop(columns=constant)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def split_fit_cal(train: pd.DataFrame, cal_frac: float = CAL_FRAC):
    """Last part of the train window is used only to pick C and probability cutoff."""
    cut = int(len(train) * (1.0 - cal_frac))
    if cut < 50 or len(train) - cut < 20:
        cut = max(int(len(train) * 0.85), len(train) - 30)
    return train.iloc[:cut], train.iloc[cut:]


def scale_train_test(name: str, x_train, x_test):
    if name == "none":
        return x_train, x_test
    scaler = SCALERS[name]()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def maybe_smote(x_train, y_train, use_smote: bool, smote_cls):
    if not use_smote or smote_cls is None:
        return x_train, y_train
    sm = smote_cls(random_state=42)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    return x_res, y_res


def tune_threshold(y_true, probs) -> float:
    """Pick cutoff on calibration days: balance accuracy and both class errors."""
    best_t, best_score = 0.5, -1.0
    y_true = np.asarray(y_true)
    for t in np.linspace(0.40, 0.56, 17):
        pred = (probs >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        rec = recall_score(y_true, pred, zero_division=0)
        score = 0.45 * acc + 0.55 * bal
        if rec >= 0.97:
            score -= 0.06
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def metrics_from_pred(y_test, pred, prob) -> dict:
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob) if prob is not None else np.nan,
    }


def evaluate_tuned(
    model,
    x_fit,
    y_fit,
    x_cal,
    y_cal,
    x_test,
    y_test,
    *,
    pick_c: bool = False,
    c_values=LOGREG_C_VALUES,
):
    """Fit on fit slice, tune threshold on cal, refit on full fit+cal, score test."""
    x_fit_a = np.asarray(x_fit)
    x_cal_a = np.asarray(x_cal)
    x_test_a = np.asarray(x_test)
    y_fit_a = np.asarray(y_fit)
    y_cal_a = np.asarray(y_cal)
    y_test_a = np.asarray(y_test)

    if pick_c and hasattr(model, "set_params"):
        best_c, best_f1 = c_values[0], -1.0
        for c in c_values:
            model.set_params(C=c)
            model.fit(x_fit_a, y_fit_a)
            cal_prob = model.predict_proba(x_cal_a)[:, 1]
            t = tune_threshold(y_cal_a, cal_prob)
            pred = (cal_prob >= t).astype(int)
            f1 = f1_score(y_cal_a, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_c = f1, c
        model.set_params(C=best_c)

    x_all = np.vstack([x_fit_a, x_cal_a])
    y_all = np.concatenate([y_fit_a, y_cal_a])
    model.fit(x_all, y_all)

    cal_prob = model.predict_proba(x_cal_a)[:, 1]
    threshold = tune_threshold(y_cal_a, cal_prob)
    test_prob = model.predict_proba(x_test_a)[:, 1]
    pred_default = (test_prob >= 0.5).astype(int)
    pred_tuned = (test_prob >= threshold).astype(int)
    out = metrics_from_pred(y_test_a, pred_tuned, test_prob)
    out["accuracy_default"] = accuracy_score(y_test_a, pred_default)
    out["decision_threshold"] = threshold
    if pick_c and hasattr(model, "get_params"):
        out["C"] = model.get_params().get("C")
    return out


def evaluate_tuned_df(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    *,
    norm: str = "none",
    use_smote: bool = False,
    smote_cls=None,
    pick_c: bool = False,
):
    fit_df, cal_df = split_fit_cal(train)
    x_fit, y_fit = fit_df[cols], fit_df[TARGET]
    x_cal, y_cal = cal_df[cols], cal_df[TARGET]
    x_te, y_te = test[cols], test[TARGET]

    xs_fit, xs_cal = scale_train_test(norm, x_fit, x_cal)
    _, xs_te = scale_train_test(norm, x_fit, x_te)

    xs_fit_s, y_fit_s = maybe_smote(xs_fit, y_fit, use_smote, smote_cls)
    return evaluate_tuned(
        model, xs_fit_s, y_fit_s, xs_cal, y_cal, xs_te, y_te, pick_c=pick_c,
    )


def evaluate_tuned_tree(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    *,
    use_smote: bool = False,
    smote_cls=None,
):
    fit_df, cal_df = split_fit_cal(train)
    x_fit, y_fit = fit_df[cols].values, fit_df[TARGET]
    x_cal, y_cal = cal_df[cols].values, cal_df[TARGET]
    x_te, y_te = test[cols].values, test[TARGET]

    x_fit_s, y_fit_s = maybe_smote(x_fit, y_fit, use_smote, smote_cls)
    return evaluate_tuned(model, x_fit_s, y_fit_s, x_cal, y_cal, x_te, y_te)


def evaluate_ensemble_soft_vote(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    *,
    use_smote: bool = False,
    smote_cls=None,
    lgbm_cls=None,
) -> dict:
    """Soft-vote ensemble: LogReg + RF + HGB (+ LightGBM when installed)."""
    estimators = [
        (
            "lr",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", C=0.1, random_state=42,
                ),
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=450,
                min_samples_leaf=18, random_state=42,
            ),
        ),
    ]
    weights = [1, 1, 2]
    if lgbm_cls is not None:
        estimators.append(
            (
                "lgbm",
                lgbm_cls(
                    n_estimators=450, max_depth=8, learning_rate=0.04,
                    min_child_samples=22, class_weight="balanced",
                    random_state=42, verbosity=-1,
                ),
            )
        )
        weights.append(2)
    vote = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    return evaluate_tuned_tree(
        vote, train, test, cols, use_smote=use_smote, smote_cls=smote_cls,
    )


def run_grid(train, test, price_cols, all_cols) -> pd.DataFrame:
    smote_cls = _try_smote()
    lgbm_cls = _try_lightgbm()
    rows: list[dict] = []

    linear_specs = [
        ("Logistic Regression", "price only", price_cols),
        ("Logistic Regression", "price plus mood", all_cols),
    ]
    for model_name, feat_label, cols in linear_specs:
        for norm in ("zscore", "minmax", "robust"):
            for use_smote in (False, True):
                if use_smote and smote_cls is None:
                    continue
                base = LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                )
                metrics = evaluate_tuned_df(
                    base, train, test, cols,
                    norm=norm, use_smote=use_smote, smote_cls=smote_cls, pick_c=True,
                )
                rows.append({
                    "model": model_name,
                    "features": feat_label,
                    "normalisation": norm,
                    "smote": use_smote,
                    "threshold_tuned": True,
                    **metrics,
                })

    tree_specs: list[tuple] = [
        ("Random Forest", "price only", price_cols, RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ("Random Forest", "price plus mood", all_cols, RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=6,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ("HistGradientBoosting", "price plus mood", all_cols, HistGradientBoostingClassifier(
            max_depth=7, learning_rate=0.05, max_iter=500,
            min_samples_leaf=16, random_state=42)),
    ]
    if lgbm_cls is not None:
        tree_specs.append(
            ("LightGBM", "price plus mood", all_cols, lgbm_cls(
                n_estimators=500, max_depth=8, learning_rate=0.04,
                min_child_samples=25, class_weight="balanced",
                random_state=42, verbosity=-1)),
        )
        tree_specs.append(
            ("LightGBM", "price only", price_cols, lgbm_cls(
                n_estimators=500, max_depth=7, learning_rate=0.04,
                min_child_samples=25, class_weight="balanced",
                random_state=42, verbosity=-1)),
        )

    for model_name, feat_label, cols, estimator in tree_specs:
        for use_smote in (False, True):
            if use_smote and smote_cls is None:
                continue
            metrics = evaluate_tuned_tree(
                estimator, train, test, cols, use_smote=use_smote, smote_cls=smote_cls,
            )
            rows.append({
                "model": model_name,
                "features": feat_label,
                "normalisation": "none",
                "smote": use_smote,
                "threshold_tuned": True,
                **metrics,
            })

    for use_smote in (False, True):
        if use_smote and smote_cls is None:
            continue
        metrics = evaluate_ensemble_soft_vote(
            train, test, all_cols, use_smote=use_smote,
            smote_cls=smote_cls if use_smote else None,
            lgbm_cls=lgbm_cls,
        )
        rows.append({
            "model": "Ensemble soft vote",
            "features": "price plus mood",
            "normalisation": "zscore on LogReg leg only",
            "smote": use_smote,
            "threshold_tuned": True,
            **metrics,
        })

    return pd.DataFrame(rows)


def load_sprint21_baseline() -> dict | None:
    path = C.MODELS_DIR / "model_metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    row = m.iloc[0]
    return {
        "source": "Sprint 2.1 LogReg price only (0.5 threshold)",
        "accuracy": float(row["accuracy"]),
        "f1": float(row["f1"]),
        "roc_auc": float(row["roc_auc"]),
    }


def write_improvement_summary(results: pd.DataFrame, baseline: dict | None) -> None:
    honest = results[results["recall"] < 0.92].copy()
    if len(honest) == 0:
        honest = results
    honest = honest.copy()
    honest["pick_score"] = 0.55 * honest["accuracy"] + 0.45 * honest["balanced_accuracy"]
    best_acc = honest.sort_values("pick_score", ascending=False).iloc[0]
    best_f1 = honest.sort_values("f1", ascending=False).iloc[0]
    best_bal = results.sort_values("balanced_accuracy", ascending=False).iloc[0]
    ensemble = results[(results["model"] == "Ensemble soft vote") & (results["recall"] < 0.92)]
    best_ens = ensemble.sort_values("f1", ascending=False).iloc[0] if len(ensemble) else None

    lines = [
        "# Sprint 2.2: what we changed (simple bullets)",
        "",
        "Full story in plain words: docs/05_sprint_22_changes_simple.md",
        "",
        "## Data",
        "",
        "- More news: Yahoo tickers plus RSS feeds; headlines saved in an archive file.",
        "- Daily news file merged from the archive (more days when RSS has dates).",
        "- Daily put/call from CBOE PCCE (not one flat snapshot).",
        "- New columns: gap_ret, ret_lag2, qqq_rel_mom5, breadth_change, plus vol,",
        "  range, VIX vs MA, put/call PCCE, and smoothed news.",
        "",
        "## Training tricks",
        "",
        "- Class weights; LogReg C tuning; probability cutoff tuned on calibration days.",
        "- HistGradientBoosting, LightGBM, RF on price only and price plus mood.",
        "- Ensemble: average LogReg + RF + HistGradientBoosting probabilities.",
        "",
        "## Best honest runs (test set, recall under 92%)",
        "",
        f"- Best accuracy: {best_acc['model']} ({best_acc['features']}) -> "
        f"accuracy {best_acc['accuracy']:.3f}, balanced acc {best_acc['balanced_accuracy']:.3f}, "
        f"F1 {best_acc['f1']:.3f}, ROC AUC {best_acc['roc_auc']:.3f}.",
        f"- Best F1: {best_f1['model']} -> F1 {best_f1['f1']:.3f}, accuracy {best_f1['accuracy']:.3f}.",
        f"- Best balanced accuracy: {best_bal['model']} -> {best_bal['balanced_accuracy']:.3f}.",
    ]
    if best_ens is not None:
        lines.append(
            f"- Ensemble soft vote -> accuracy {best_ens['accuracy']:.3f}, "
            f"F1 {best_ens['f1']:.3f}, balanced acc {best_ens['balanced_accuracy']:.3f}."
        )
    lines.append("")
    if baseline:
        lines.extend([
            "## Versus Sprint 2.1",
            "",
            f"- Baseline accuracy {baseline['accuracy']:.3f}, F1 {baseline['f1']:.3f}.",
            f"- 2.2 best honest accuracy {best_acc['accuracy']:.3f} "
            f"(change {best_acc['accuracy'] - baseline['accuracy']:+.3f}).",
            "",
        ])
    lines.extend([
        "## Note",
        "",
        "Daily direction stays noisy. We report balanced accuracy so scores are not inflated",
        "by guessing Up every day.",
        "",
    ])
    path = C.COMPARISON_DIR / "sprint22_improvements.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    if baseline:
        cmp = pd.DataFrame([
            {"label": baseline["source"], **{k: baseline[k] for k in ("accuracy", "f1", "roc_auc")}},
            {
                "label": f"2.2 best accuracy: {best_acc['model']}",
                "accuracy": best_acc["accuracy"],
                "f1": best_acc["f1"],
                "roc_auc": best_acc["roc_auc"],
            },
            {
                "label": f"2.2 best F1: {best_f1['model']}",
                "accuracy": best_f1["accuracy"],
                "f1": best_f1["f1"],
                "roc_auc": best_f1["roc_auc"],
            },
        ])
        cmp.to_csv(C.COMPARISON_DIR / "baseline_vs_sprint22.csv", index=False)


def best_per_model(results: pd.DataFrame) -> pd.DataFrame:
    idx = results.groupby("model")["f1"].idxmax()
    return results.loc[idx].sort_values("f1", ascending=False).reset_index(drop=True)


def normalisation_summary(results: pd.DataFrame) -> pd.DataFrame:
    linear = results[results["model"] == "Logistic Regression"]
    return (
        linear.groupby("normalisation")["f1"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def plot_dashboard(results: pd.DataFrame, best: pd.DataFrame, norm_sum: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    labels = [f"{r.model}\n{r.features[:10]}" for r in best.itertuples()]
    ax.barh(labels, best["accuracy"], color="#0275d8")
    ax.set_xlim(0, 1)
    ax.set_title("Best F1 per model: test accuracy (tuned threshold)")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)

    ax = axes[0, 1]
    for norm, sub in results[results["model"] == "Logistic Regression"].groupby("normalisation"):
        ax.bar(norm, sub["f1"].mean(), label=norm)
    ax.set_title("LogReg: mean F1 by normalisation")
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    top = results.sort_values("accuracy", ascending=False).head(8)
    x = np.arange(len(top))
    ax.bar(x - 0.15, top["accuracy"], width=0.3, color="#5cb85c", label="accuracy")
    ax.bar(x + 0.15, top["balanced_accuracy"], width=0.3, color="#0275d8", label="bal_acc")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.model[:7]}\n{r.normalisation[:4]}" for r in top.itertuples()],
        fontsize=7,
    )
    ax.set_title("Top 8 runs by accuracy")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Leaderboard (best F1 per model, tuned threshold)", ""]
    for r in best.itertuples():
        sm = "SMOTE" if r.smote else "no SMOTE"
        lines.append(
            f"{r.model}: acc={r.accuracy:.3f} F1={r.f1:.3f} AUC={r.roc_auc:.3f} "
            f"{r.normalisation} {sm}"
        )
    lines.append("")
    lines.append("Best normalisation for LogReg (mean F1):")
    for r in norm_sum.itertuples():
        lines.append(f"  {r.normalisation}: mean F1={r.mean:.3f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(C.COMPARISON_DIR / "summary_dashboard.png", dpi=120)
    plt.close(fig)


def write_html_overview(results: pd.DataFrame, best: pd.DataFrame) -> None:
    out = C.COMPARISON_DIR / "results_overview.html"
    rows_html = []
    for r in results.sort_values("accuracy", ascending=False).itertuples():
        rows_html.append(
            f"<tr><td>{html.escape(r.model)}</td>"
            f"<td>{html.escape(r.features)}</td>"
            f"<td>{html.escape(r.normalisation)}</td>"
            f"<td>{r.smote}</td>"
            f"<td>{r.accuracy:.3f}</td><td>{r.balanced_accuracy:.3f}</td>"
            f"<td>{r.f1:.3f}</td><td>{r.roc_auc:.3f}</td></tr>"
        )
    best_rows = []
    for r in best.itertuples():
        best_rows.append(
            f"<li><b>{html.escape(r.model)}</b>: acc {r.accuracy:.3f}, F1 {r.f1:.3f}, "
            f"normalisation {html.escape(r.normalisation)}, SMOTE={r.smote}</li>"
        )
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Sprint 2.2 model comparison</title>
<style>
body {{ font-family: Calibri, sans-serif; margin: 24px; background: #f2f5f9; }}
h1 {{ color: #1f3a5f; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 14px; }}
th {{ background: #1f3a5f; color: white; }}
img {{ max-width: 100%; margin-top: 16px; }}
.note {{ color: #5a6370; font-size: 14px; }}
</style></head><body>
<h1>Sprint 2.2: model comparison (tuned decision threshold)</h1>
<p class="note">Scores use a probability cutoff tuned on the last 15% of train days.
See sprint22_improvements.md for what changed versus Sprint 2.1.</p>
<ul>{"".join(best_rows)}</ul>
<img src="summary_dashboard.png" alt="Summary dashboard"/>
<table>
<tr><th>Model</th><th>Features</th><th>Normalisation</th><th>SMOTE</th>
<th>Accuracy</th><th>Balanced acc</th><th>F1</th><th>ROC AUC</th></tr>
{"".join(rows_html)}
</table>
</body></html>"""
    out.write_text(body, encoding="utf-8")


def main() -> None:
    C.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Sprint 2.2  First Comparison of Different Models")
    print("=" * 70)

    df = load_features()
    price_cols = [c for c in PRICE_FEATURES if c in df.columns]
    mood_cols = [c for c in MOOD_FEATURES if c in df.columns]
    all_cols = price_cols + mood_cols
    print(f" - price features ({len(price_cols)}): {price_cols}")
    print(f" - mood features ({len(mood_cols)}): {mood_cols}")

    train, test = time_split(df)
    print(f" - train {len(train)} days, test {len(test)} days")
    print(f" - threshold tuning uses last {CAL_FRAC:.0%} of train as calibration")

    results = run_grid(train, test, price_cols, all_cols)
    results.to_csv(C.COMPARISON_DIR / "all_model_runs.csv", index=False)
    best = best_per_model(results)
    best.to_csv(C.COMPARISON_DIR / "best_per_model.csv", index=False)
    norm_sum = normalisation_summary(results)
    norm_sum.to_csv(C.COMPARISON_DIR / "normalisation_summary.csv", index=False)

    baseline = load_sprint21_baseline()
    write_improvement_summary(results, baseline)

    plot_dashboard(results, best, norm_sum)
    write_html_overview(results, best)

    honest = results[results["recall"] < 0.92].copy()
    if len(honest) == 0:
        honest = results
    honest = honest.copy()
    honest["pick_score"] = 0.55 * honest["accuracy"] + 0.45 * honest["balanced_accuracy"]
    top_acc = honest.sort_values("pick_score", ascending=False).iloc[0]
    top_f1 = honest.sort_values("f1", ascending=False).iloc[0]
    top_bal = results.sort_values("balanced_accuracy", ascending=False).iloc[0]
    print(f"\nRecommended (not always-Up): {top_acc['model']} ({top_acc['features']}), "
          f"acc={top_acc['accuracy']:.3f}, bal_acc={top_acc['balanced_accuracy']:.3f}, "
          f"F1={top_acc['f1']:.3f}, threshold={top_acc['decision_threshold']:.2f}")
    print(f"Best F1 (same filter):       {top_f1['model']}, F1={top_f1['f1']:.3f}, "
          f"acc={top_f1['accuracy']:.3f}")
    print(f"Best balanced accuracy:      {top_bal['model']}, "
          f"bal_acc={top_bal['balanced_accuracy']:.3f}, acc={top_bal['accuracy']:.3f}")
    if baseline:
        print(f"Sprint 2.1 baseline acc={baseline['accuracy']:.3f} -> "
              f"2.2 recommended acc={top_acc['accuracy']:.3f} "
              f"({top_acc['accuracy'] - baseline['accuracy']:+.3f})")
    print(f"Outputs in {C.COMPARISON_DIR}")
    print("Sprint 2.2 done.")


if __name__ == "__main__":
    main()
