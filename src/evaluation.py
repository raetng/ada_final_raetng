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
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
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


def make_time_series_splits_grouped(df, group_col="datetime", n_splits=5):
    """Time-series CV where fold boundaries respect timestamp groups.

    Needed for the multi-hub modeling matrix: each timestamp contributes one
    row per hub, and HB_* hubs co-spike with correlation >0.998. A naive
    row-index split would put hub A at time T in train and hub B at time T
    in test — same weather, same gas, near-identical target — which inflates
    test performance. This splitter assigns whole timestamps to train or test.
    """
    if not df[group_col].is_monotonic_increasing:
        raise ValueError(f"df must be sorted ascending by '{group_col}'.")
    groups = df.groupby(group_col, sort=False).indices
    timestamps = list(groups.keys())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_ts_idx, test_ts_idx in tscv.split(timestamps):
        train_rows = np.concatenate([groups[timestamps[i]] for i in train_ts_idx])
        test_rows = np.concatenate([groups[timestamps[i]] for i in test_ts_idx])
        splits.append((train_rows, test_rows))
    return splits


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


def evaluate_regressor(y_true_log, y_pred_log):
    """Score a log-target regressor for stage-2 magnitude prediction.

    Inputs are on the log scale (natural log of SPP). Reports the standard log-
    scale fit metrics (RMSE, MAE, R^2) plus back-transformed dollar metrics so
    we can sanity-check what the log-scale errors mean for an operator: median
    absolute dollar error, and median absolute percentage error in dollars.
    Median is preferred over mean for the dollar-scale metrics because spike
    magnitudes have an extreme right tail that would let a few hours dominate
    any mean-based summary.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    y_true_d = np.exp(y_true_log)
    y_pred_d = np.exp(y_pred_log)
    abs_err_d = np.abs(y_pred_d - y_true_d)
    pct_err_d = abs_err_d / y_true_d

    # MAE on the top decile of *actual* spike magnitudes — where the operational
    # cost of being wrong is highest. Computed in log-space (consistent with
    # the headline metric) using the 90th percentile of y_true_log as the cut.
    cutoff = float(np.quantile(y_true_log, 0.90))
    top = y_true_log >= cutoff
    if top.any():
        mae_log_top_decile = float(mean_absolute_error(y_true_log[top], y_pred_log[top]))
        n_top = int(top.sum())
    else:
        mae_log_top_decile = float("nan")
        n_top = 0

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "median_abs_err_dollars": float(np.median(abs_err_d)),
        "median_abs_pct_err": float(np.median(pct_err_d)),
        "mae_log_top_decile": mae_log_top_decile,
        "top_decile_cutoff_log": cutoff,
        "n_top_decile": n_top,
        "n": int(len(y_true_log)),
        "y_true_log": y_true_log,
        "y_pred_log": y_pred_log,
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
