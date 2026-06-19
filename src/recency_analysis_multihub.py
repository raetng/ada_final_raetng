"""Multi-hub recency analysis: does training-window size still matter?

Mirrors recency_analysis.py. The fixed test window is the last 12 months;
three training windows (all / 3y / 1y) are compared on the multi-hub
dataset, using the best RF configuration from
train_random_forest_multihub.py (depth=10, leaf=5).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_classifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data_multihub.parquet"
SESSION_DIR = PROJECT_ROOT / "outputs" / "final"
RESULTS_PATH = SESSION_DIR / "recency_results.json"
BAR_PLOT_PATH = SESSION_DIR / "recency_comparison.png"
PROBA_PLOT_PATH = SESSION_DIR / "probability_distributions.png"

TARGET = "is_spike"
TEST_MONTHS = 12
NON_FEATURE_COLS = ("datetime", "Location", TARGET)

# Best RF variant from train_random_forest_multihub.py: depth=10_leaf=5.
RF_PARAMS = dict(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",
    max_depth=10,
    min_samples_leaf=5,
)

WINDOWS = [
    ("all_history", None),
    ("3_years", 3),
    ("1_year", 1),
]


def split_test_set(df, test_months):
    test_start = df["datetime"].max() - pd.DateOffset(months=test_months)
    is_test = df["datetime"] > test_start
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True), test_start


def select_training_window(df_train, test_start, years_back):
    if years_back is None:
        return df_train
    cutoff = test_start - pd.DateOffset(years=years_back)
    return df_train.loc[df_train["datetime"] > cutoff].reset_index(drop=True)


def fit_and_score(df_train_window, df_test, feature_cols):
    X_train = df_train_window[feature_cols]
    y_train = df_train_window[TARGET].astype(int)
    X_test = df_test[feature_cols]
    y_test = df_test[TARGET].astype(int).to_numpy()

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, proba)
    return proba, metrics


def plot_pr_auc_bars(window_results, output_path):
    names = [w[0] for w in WINDOWS]
    pr_aucs = [window_results[name]["pr_auc"] for name in names]
    train_sizes = [window_results[name]["n_train"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, pr_aucs, color=["#4c78a8", "#f58518", "#54a24b"], edgecolor="white")
    for bar, n in zip(bars, train_sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"PR-AUC={bar.get_height():.4f}\nn_train={n:,}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("PR-AUC on fixed test set (last 12 months)")
    ax.set_title("Random Forest (multi-hub): training-window recency vs PR-AUC")
    ax.set_ylim(0, max(pr_aucs) * 1.25)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distributions(window_probas, y_test, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    bins = np.linspace(0, 1, 41)
    colors = {"all_history": "#4c78a8", "3_years": "#f58518", "1_year": "#54a24b"}

    for cls_idx, (cls_label, mask) in enumerate([("non-spike", y_test == 0), ("spike", y_test == 1)]):
        ax = axes[cls_idx]
        for name, proba in window_probas.items():
            ax.hist(
                proba[mask], bins=bins, alpha=0.5, label=name,
                color=colors[name], density=True, edgecolor="white", linewidth=0.5,
            )
        ax.set_title(f"Predicted P(spike)  on  {cls_label}  rows  (n={int(mask.sum()):,})")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.4)
        ax.legend()

    fig.suptitle("Multi-hub test-set probability distributions by training window", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 80)
    print("LOAD DATA AND DEFINE FIXED TEST SET (LAST 12 MONTHS)")
    print("=" * 80)
    df = pd.read_parquet(DATA_PATH).sort_values(["datetime", "Location"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    df_pretest, df_test, test_start = split_test_set(df, TEST_MONTHS)
    print(f"  total rows: {len(df):,}    features: {len(feature_cols)}")
    print(f"  test set:   {df_test['datetime'].min()} -> {df_test['datetime'].max()}  "
          f"(n={len(df_test):,}, spike_rate={df_test[TARGET].mean():.4f})")
    print(f"  pre-test:   {df_pretest['datetime'].min()} -> {df_pretest['datetime'].max()}  (n={len(df_pretest):,})")

    print("\n" + "=" * 80)
    print(f"TRAIN ON {len(WINDOWS)} WINDOWS, EVALUATE ON FIXED TEST SET")
    print("=" * 80)
    window_results = {}
    window_probas = {}
    for name, years_back in WINDOWS:
        df_train_w = select_training_window(df_pretest, test_start, years_back)
        train_start = df_train_w["datetime"].min()
        train_end = df_train_w["datetime"].max()
        train_spike = df_train_w[TARGET].mean()
        print(
            f"\n  window={name:11s}  train: {train_start} -> {train_end}  "
            f"n={len(df_train_w):,}  spike_rate={train_spike:.4f}"
        )

        proba, metrics = fit_and_score(df_train_w, df_test, feature_cols)
        window_probas[name] = proba

        window_results[name] = {
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
            "threshold": metrics["threshold"],
            "n_train": int(len(df_train_w)),
            "train_start": str(train_start),
            "train_end": str(train_end),
            "train_spike_rate": float(train_spike),
        }
        print(
            f"    PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
        )

    out = {
        "test_set": {
            "start": str(df_test["datetime"].min()),
            "end": str(df_test["datetime"].max()),
            "n": int(len(df_test)),
            "spike_rate": float(df_test[TARGET].mean()),
        },
        "rf_params": {k: (None if v is None else v) for k, v in RF_PARAMS.items()},
        "windows": window_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n  saved results -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    plot_pr_auc_bars(window_results, BAR_PLOT_PATH)
    print(f"  saved bar chart -> {BAR_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    plot_probability_distributions(window_probas, df_test[TARGET].astype(int).to_numpy(), PROBA_PLOT_PATH)
    print(f"  saved proba plot -> {PROBA_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    pr = {name: window_results[name]["pr_auc"] for name, _ in WINDOWS}
    best_name = max(pr, key=pr.get)
    worst_name = min(pr, key=pr.get)
    spread = pr[best_name] - pr[worst_name]
    all_h = pr["all_history"]
    one_y = pr["1_year"]
    delta_recent = one_y - all_h

    if abs(delta_recent) < 0.01:
        verdict = "training-window choice barely moves PR-AUC"
        narrative = "the relationships are stable enough that more data wins"
    elif delta_recent > 0:
        verdict = f"the 1-year window beats all-history by {delta_recent:+.4f} PR-AUC"
        narrative = "the grid has changed and recent data captures that"
    else:
        verdict = f"the 1-year window underperforms all-history by {delta_recent:+.4f} PR-AUC"
        narrative = "the relationships are stable enough that more data wins"

    print(
        f"  Best window: {best_name} (PR-AUC={pr[best_name]:.4f}); worst: {worst_name} "
        f"(PR-AUC={pr[worst_name]:.4f}); spread={spread:.4f}."
    )
    print(
        f"  1-year vs all-history delta is {delta_recent:+.4f} PR-AUC, so {verdict}."
    )
    print(
        f"  Project narrative: {narrative}."
    )


if __name__ == "__main__":
    main()
