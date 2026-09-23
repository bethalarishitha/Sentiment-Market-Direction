"""Sentiment-Driven Market Direction Prediction. Sprint 2, Section 1.

Initial Model Building.

Plain idea of this file:
    Sprint 1.2 gave me a clean table of mood clues (features) and an Up/Down
    answer column. Here I train my first models to guess Up or Down for the next
    day, read off their score cards, tweak a parameter to see what changes, and
    compare a second algorithm against the first one. This is the first look at
    "does the crowd mood help", so I keep it simple and honest.

Three questions I answer here:
    1. Baseline. How well does a simple model do using price clues only.
    2. Do mood clues help. Same simple model, now with the mood features added.
    3. Second algorithm. Does a Random Forest beat the simple model.

What I produce:
    1. reports/models/model_metrics.csv     (score card for every model)
    2. reports/models/*.png                 (comparison, confusion matrix, ROC)

Run:
    python src/smd_modeling.py
    (run src/smd_preprocessing.py first so the features file exists)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

import smd_config as C
from smd_features import MOOD_FEATURES, PRICE_FEATURES, TARGET


def load_features() -> pd.DataFrame:
    if not C.FEATURES_FILE.exists():
        raise SystemExit("Run src/smd_preprocessing.py first (features file missing).")
    df = pd.read_csv(C.FEATURES_FILE, index_col=0, parse_dates=True)
    if len(df) == 0:
        raise SystemExit("features_daily.csv is empty. Re run smd_preprocessing.py.")
    constant = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    if constant:
        print(f" - dropping constant columns (snapshots): {constant}")
        df = df.drop(columns=constant)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split by time, oldest days for training and newest days for testing.

    I never shuffle the dates. A real forecast only ever knows the past, so the
    test set must be the future relative to the training set.
    """
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f" - train: {len(train)} days, test: {len(test)} days (no shuffling)")
    return train, test


def score_model(name: str, y_true, y_pred, y_prob) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
    }


def fit_and_score(name, model, x_train, y_train, x_test, y_test, scale=True):
    """Fit a model, scaling the inputs when asked (fit the scaler on train only)."""
    if scale:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    row = score_model(name, y_test, y_pred, y_prob)
    print(f"   {name:34s} acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  "
          f"auc={row['roc_auc']:.3f}")
    return row, y_pred, y_prob


def plot_comparison(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(results))
    width = 0.25
    colors = ["#0275d8", "#5cb85c", "#f0ad4e"]
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results[m], width, label=m, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(results["model"], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="coin flip")
    ax.set_title("Model comparison (higher is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.MODELS_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title: str, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Down", "Up"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Down", "Up"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(C.MODELS_DIR / filename, dpi=120)
    plt.close(fig)


def plot_roc(curves: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, (y_true, y_prob) in curves.items():
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (auc={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="coin flip")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.MODELS_DIR / filename, dpi=120)
    plt.close(fig)


def parameter_tweak(x_train, y_train, x_test, y_test) -> pd.DataFrame:
    """Sprint 2.1 asks me to tweak a parameter and watch the outcome change.

    Here I move the Logistic Regression strength setting C across a range and
    record the F1 score, so I can see which setting works best.
    """
    print(" - tweaking Logistic Regression strength (C)")
    scaler = StandardScaler().fit(x_train)
    xs_train, xs_test = scaler.transform(x_train), scaler.transform(x_test)
    rows = []
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(xs_train, y_train)
        pred = model.predict(xs_test)
        rows.append({"C": c, "f1": f1_score(y_test, pred, zero_division=0),
                     "accuracy": accuracy_score(y_test, pred)})
        print(f"   C={c:<6} f1={rows[-1]['f1']:.3f}  acc={rows[-1]['accuracy']:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Sprint 2.1  Initial Model Building")
    print("=" * 70)

    df = load_features()
    price_cols = [c for c in PRICE_FEATURES if c in df.columns]
    mood_cols = [c for c in MOOD_FEATURES if c in df.columns]
    all_cols = price_cols + mood_cols
    print(f" - price features: {price_cols}")
    print(f" - mood features : {mood_cols}")

    train, test = time_split(df)
    y_train, y_test = train[TARGET], test[TARGET]

    print("\nTraining models")
    results = []
    curves = {}

    # 1. Baseline: simple model, price clues only.
    row, _, prob_base = fit_and_score(
        "LogReg price only (baseline)", LogisticRegression(max_iter=1000),
        train[price_cols], y_train, test[price_cols], y_test)
    results.append(row)
    curves["LogReg price only"] = (y_test, prob_base)

    # 2. Same model, now with the mood clues added.
    row, pred_mood, prob_mood = fit_and_score(
        "LogReg price + mood", LogisticRegression(max_iter=1000),
        train[all_cols], y_train, test[all_cols], y_test)
    results.append(row)
    curves["LogReg price + mood"] = (y_test, prob_mood)

    # 3. Second algorithm: Random Forest with the mood clues (trees need no scaling).
    row, pred_rf, prob_rf = fit_and_score(
        "Random Forest price + mood",
        RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
        train[all_cols], y_train, test[all_cols], y_test, scale=False)
    results.append(row)
    curves["Random Forest"] = (y_test, prob_rf)

    results_df = pd.DataFrame(results)
    results_df.to_csv(C.MODELS_DIR / "model_metrics.csv", index=False)
    print(f"\nScore cards saved to {C.MODELS_DIR / 'model_metrics.csv'}")

    tweak_df = parameter_tweak(train[all_cols], y_train, test[all_cols], y_test)
    tweak_df.to_csv(C.MODELS_DIR / "logreg_C_tweak.csv", index=False)

    print("\nSaving charts")
    plot_comparison(results_df)
    plot_confusion(y_test, pred_mood, "Logistic Regression (price + mood)",
                   "confusion_logreg.png")
    plot_confusion(y_test, pred_rf, "Random Forest (price + mood)",
                   "confusion_random_forest.png")
    plot_roc(curves, "roc_curves.png")
    print(f"   charts saved to {C.MODELS_DIR}")

    best = results_df.loc[results_df["f1"].idxmax()]
    print("\nQuick read of the results")
    print(f"   best F1 so far: {best['model']} (f1={best['f1']:.3f})")
    print("Sprint 2.1 done. First models trained, scored and compared.")


if __name__ == "__main__":
    main()
