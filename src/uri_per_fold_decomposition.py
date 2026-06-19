"""Uri per-fold decomposition for Stage 1.

Loads the saved out-of-fold predictions from outputs/session7/*_best_predictions.npz
and splits each fold's PR-AUC into Uri-window vs non-Uri components. Tells
us which folds' headline scores are driven by Uri specifically.

No re-training: uses the existing concatenated npz arrays and reconstructs
fold boundaries from the modeling_data row order. Since the training scripts
concatenate test predictions in fold order, we can map array positions
1-to-1 back to dataframe rows via `make_time_series_splits`.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S1_DATA = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"
SESSION_DIR = PROJECT_ROOT / "outputs" / "session7"
OUT_DIR = PROJECT_ROOT / "outputs" / "uri_sensitivity"
RESULTS_PATH = OUT_DIR / "per_fold_decomposition.json"
PLOT_PATH = OUT_DIR / "per_fold_decomposition.png"

URI_START = pd.Timestamp("2021-02-10 00:00:00")
URI_END = pd.Timestamp("2021-02-21 23:00:00")
N_SPLITS = 5

MODELS = {
    "logistic": SESSION_DIR / "logistic_best_predictions.npz",
    "random_forest": SESSION_DIR / "rf_best_predictions.npz",
    "knn": SESSION_DIR / "knn_best_predictions.npz",
}


def safe_pr_auc(y_true, y_proba):
    """average_precision_score is undefined if only one class is present.
    Return NaN in that case so the table is honest about coverage."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(S1_DATA)
    splits = make_time_series_splits(df, n_splits=N_SPLITS)
    test_indices_per_fold = [test_idx for _, test_idx in splits]
    # Concatenate in fold order — matches the npz concatenation order.
    all_test_idx = np.concatenate(test_indices_per_fold)
    fold_assignment = np.repeat(
        np.arange(1, N_SPLITS + 1),
        [len(t) for t in test_indices_per_fold],
    )

    test_dt = df["datetime"].iloc[all_test_idx].to_numpy()
    uri_mask = (test_dt >= np.datetime64(URI_START)) & (test_dt <= np.datetime64(URI_END))

    print(f"Total test predictions across folds: {len(all_test_idx):,}")
    print(f"Uri-window test rows: {int(uri_mask.sum())}  (in fold(s): "
          f"{sorted(set(fold_assignment[uri_mask].tolist()))})")

    out = {
        "uri_window": {"start": str(URI_START), "end": str(URI_END)},
        "models": {},
    }

    rows_for_table = []
    for cls, path in MODELS.items():
        data = np.load(path)
        y_true = data["y_true"]
        y_proba = data["y_proba"]
        assert len(y_true) == len(all_test_idx), (
            f"{cls}: npz length {len(y_true)} != reconstructed fold-test length {len(all_test_idx)}"
        )

        per_fold = []
        for fold in range(1, N_SPLITS + 1):
            in_fold = fold_assignment == fold
            uri_in_fold = uri_mask & in_fold
            non_uri_in_fold = (~uri_mask) & in_fold

            fold_n = int(in_fold.sum())
            uri_n = int(uri_in_fold.sum())
            non_uri_n = int(non_uri_in_fold.sum())

            entry = {
                "fold": fold,
                "n_test": fold_n,
                "n_uri": uri_n,
                "n_non_uri": non_uri_n,
                "uri_spike_rate": float(y_true[uri_in_fold].mean()) if uri_n else float("nan"),
                "non_uri_spike_rate": float(y_true[non_uri_in_fold].mean()) if non_uri_n else float("nan"),
                "pr_auc_all": safe_pr_auc(y_true[in_fold], y_proba[in_fold]),
                "pr_auc_uri": safe_pr_auc(y_true[uri_in_fold], y_proba[uri_in_fold]),
                "pr_auc_non_uri": safe_pr_auc(y_true[non_uri_in_fold], y_proba[non_uri_in_fold]),
            }
            per_fold.append(entry)

            rows_for_table.append({
                "model": cls,
                "fold": fold,
                "n_test": fold_n,
                "n_uri": uri_n,
                "PR-AUC (all)": entry["pr_auc_all"],
                "PR-AUC (Uri)": entry["pr_auc_uri"],
                "PR-AUC (non-Uri)": entry["pr_auc_non_uri"],
            })

        out["models"][cls] = {"per_fold": per_fold}

    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nsaved JSON -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    # Pretty print
    tbl = pd.DataFrame(rows_for_table)
    print("\nPer-fold decomposition:")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))

    # Plot: focus on fold 1 since it's the only fold with Uri test rows
    fold1 = tbl[tbl["fold"] == 1].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = fold1["model"].tolist()
    x = np.arange(len(classes))
    width = 0.27
    all_vals = fold1["PR-AUC (all)"].to_numpy()
    uri_vals = fold1["PR-AUC (Uri)"].to_numpy()
    non_uri_vals = fold1["PR-AUC (non-Uri)"].to_numpy()
    ax.bar(x - width, all_vals, width, label="All", color="#5b6e7f", edgecolor="white")
    ax.bar(x,         uri_vals, width, label=f"Uri only (n={fold1['n_uri'].iloc[0]})", color="#e45756", edgecolor="white")
    ax.bar(x + width, non_uri_vals, width, label="Non-Uri", color="#4c78a8", edgecolor="white")
    for xi, (a, u, n) in enumerate(zip(all_vals, uri_vals, non_uri_vals)):
        ax.text(xi - width, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi,         u + 0.01, f"{u:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width, n + 0.01, f"{n:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Fold 1 PR-AUC decomposed by Uri-window vs non-Uri-window test rows\n"
                 "(folds 2-5 contain no Uri rows)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, float(np.nanmax(np.concatenate([all_vals, uri_vals, non_uri_vals]))) * 1.2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
