"""Reusable evaluation utilities for spike-prediction classifiers.

Primary metric is PR-AUC (average precision) because the positive class is
rare (~4% spike rate), which makes ROC-AUC optimistic. ROC-AUC is reported
secondarily so we can sanity-check ranking ability against PR-AUC.

All public functions assume binary classification with `y_proba` giving the
predicted probability of the positive (spike) class.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


TARGET_RECALL = 0.80


def make_time_series_splits(df, n_splits=5):
    """Expanding-window chronological CV splits.

    Wraps `sklearn.model_selection.TimeSeriesSplit`, which never shuffles and
    always trains on a strict prefix of the data. Returns a list of
    `(train_idx, test_idx)` pairs of integer positions into `df`.
    """
    if "datetime" in df.columns and not df["datetime"].is_monotonic_increasing:
        raise ValueError(
            "df must be sorted ascending by 'datetime' before time-series splitting."
        )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(len(df))))


def evaluate_classifier(y_true, y_proba, threshold=None):
    """Score a classifier and return a dict of metrics.

    If `threshold` is None, the operating threshold is chosen to maximize F1
    on the supplied data — a convenience for headline numbers, but note the
    optimism: in production the threshold should be picked on a held-out set.

    Also reports precision and recall at the highest threshold whose recall is
    still >= 0.80 — a fixed operating point that lets us compare models at the
    same recall regardless of where each model's F1-optimal threshold falls.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_true, y_proba)

    chose_f1 = threshold is None
    if chose_f1:
        prec = precision_arr[:-1]
        rec = recall_arr[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_curve = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        best_idx = int(np.argmax(f1_curve))
        threshold = float(thresholds_pr[best_idx])

    y_pred = (y_proba >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # recall_arr is monotonically nonincreasing in threshold; pick the highest
    # threshold whose recall still meets TARGET_RECALL.
    eligible = np.where(recall_arr[:-1] >= TARGET_RECALL)[0]
    if len(eligible) > 0:
        idx = eligible[-1]
        precision_at_target_recall = float(precision_arr[idx])
        recall_at_target_recall = float(recall_arr[idx])
    else:
        precision_at_target_recall = float("nan")
        recall_at_target_recall = float("nan")

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
        "threshold_source": "f1_max" if chose_f1 else "user_supplied",
        "precision_at_recall_0.80": precision_at_target_recall,
        "recall_at_recall_0.80": recall_at_target_recall,
        "y_true": y_true,
        "y_proba": y_proba,
    }


def plot_pr_curve(results_dict, output_path):
    """Overlay PR curves for multiple models, with the no-skill baseline."""
    fig, ax = plt.subplots(figsize=(9, 6))

    first = next(iter(results_dict.values()))
    baseline = float(np.mean(first["y_true"]))

    for name, res in results_dict.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ax.plot(rec, prec, linewidth=2, label=f"{name}  (PR-AUC = {res['pr_auc']:.3f})")

    ax.axhline(
        baseline,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"No-skill baseline ({baseline:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(results_dict, output_path):
    """Overlay ROC curves for multiple models, with the chance diagonal."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC-ROC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0,
            label="No-skill baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_results(results_dict, output_path):
    """Write a markdown comparison table sorted by PR-AUC descending.

    Reports PR-AUC (primary) plus ROC-AUC and a fixed-recall operating point
    (Precision@Recall=0.80) as secondary diagnostics. Precision/Recall at the
    F1-maximizing threshold are kept as operating-point info; F1 and Brier are
    intentionally excluded — F1 is fully determined by Precision/Recall, and
    Brier mixes calibration with discrimination in a way that confounds
    rank-based metrics like PR-AUC."""
    rows = []
    for name, res in results_dict.items():
        rows.append({
            "Model": name,
            "PR-AUC": res["pr_auc"],
            "ROC-AUC": res["roc_auc"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "Threshold": res["threshold"],
            "Precision@Recall=0.80": res["precision_at_recall_0.80"],
            "Recall@Recall=0.80": res["recall_at_recall_0.80"],
        })
    summary = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    columns = list(summary.columns)
    lines = [
        "# Model comparison",
        "",
        f"Sorted by PR-AUC descending. Operating threshold per model is the F1-maximizing "
        f"threshold unless the caller supplied one explicitly. The `*@Recall=0.80` columns "
        f"report a fixed-recall operating point for like-for-like comparison.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in summary.iterrows():
        cells = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                cells.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return summary
