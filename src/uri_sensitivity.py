"""Uri window-exclusion ablation for Stage 1 (and Stage 2) best models.

Refits each best-of-class model in two conditions:
  - with Uri:    Winter Storm Uri window (2021-02-10 -> 2021-02-21) included.
                 This replicates the saved results in outputs/session7/.
  - without Uri: Uri window rows removed from the dataset before CV.

Same 5-fold expanding-window time-series CV, same hyperparameters (loaded
from the saved best-model pickles), so the only thing changing is the
presence or absence of the Uri rows.

Stage 1 models: logistic (l1_balanced), RF (depth=None_leaf=20), KNN (k=25).
Stage 2 model:  RF (depth=10_leaf=5) on log(SPP) | SPP > $100 (HB_HOUSTON).

Writes outputs/uri_sensitivity/ablation_results.json and a comparison plot.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_classifier, evaluate_regressor, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S1_DATA = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"
S2_DATA = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_houston.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "uri_sensitivity"
RESULTS_PATH = OUT_DIR / "ablation_results.json"
PLOT_PATH = OUT_DIR / "ablation_comparison.png"

URI_START = pd.Timestamp("2021-02-10 00:00:00")
URI_END = pd.Timestamp("2021-02-21 23:00:00")

N_SPLITS = 5
RANDOM_STATE = 42


def filter_uri(df, exclude):
    if not exclude:
        return df
    mask = (df["datetime"] >= URI_START) & (df["datetime"] <= URI_END)
    return df.loc[~mask].reset_index(drop=True)


def stage1_cv(df, factory, scale, target_col="is_spike"):
    """Run 5-fold time-series CV on a Stage-1 classifier and return per-fold
    + aggregate metrics. `factory()` returns a fresh estimator each call.
    `scale=True` fits a StandardScaler per fold; `scale=False` passes raw
    features (used by RF)."""
    feature_cols = [c for c in df.columns if c not in ("datetime", target_col)]
    X = df[feature_cols]
    y = df[target_col].astype(int)
    splits = make_time_series_splits(df, n_splits=N_SPLITS)

    fold_metrics = []
    proba_all, y_all = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if scale:
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train)
            X_test_use = scaler.transform(X_test)
        else:
            X_train_use, X_test_use = X_train, X_test
        model = factory()
        t0 = time.time()
        model.fit(X_train_use, y_train)
        proba = model.predict_proba(X_test_use)[:, 1]
        elapsed = time.time() - t0
        res = evaluate_classifier(y_test.to_numpy(), proba)
        fold_metrics.append({
            "fold": fold_idx,
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "f1": res["f1"],
            "fit_predict_seconds": float(elapsed),
            "n_test": int(len(test_idx)),
            "test_spike_rate": float(y_test.mean()),
            "test_start": str(df["datetime"].iloc[test_idx[0]]),
            "test_end": str(df["datetime"].iloc[test_idx[-1]]),
        })
        proba_all.append(proba)
        y_all.append(y_test.to_numpy())

    y_true = np.concatenate(y_all)
    y_proba = np.concatenate(proba_all)
    agg = evaluate_classifier(y_true, y_proba)
    return {
        "per_fold": fold_metrics,
        "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "std_pr_auc": float(np.std([m["pr_auc"] for m in fold_metrics])),
        "concat_pr_auc": agg["pr_auc"],
        "concat_roc_auc": agg["roc_auc"],
        "concat_p_at_r_080": agg["precision_at_recall_0.80"],
        "n_total": int(len(df)),
        "n_positives": int(y.sum()),
    }


def stage2_cv(df, factory, target_col="log_spp"):
    """Run 5-fold time-series CV on the Stage-2 RF regressor."""
    non_feature = ("datetime", "SPP", target_col)
    feature_cols = [c for c in df.columns if c not in non_feature]
    X = df[feature_cols]
    y = df[target_col].astype(float)
    splits = make_time_series_splits(df, n_splits=N_SPLITS)

    fold_metrics = []
    pred_all, y_all = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        res = evaluate_regressor(y_test.to_numpy(), pred)
        fold_metrics.append({
            "fold": fold_idx,
            "rmse_log": res["rmse_log"],
            "mae_log": res["mae_log"],
            "r2_log": res["r2_log"],
            "mae_log_top_decile": res["mae_log_top_decile"],
            "median_abs_err_dollars": res["median_abs_err_dollars"],
            "n_test": int(len(test_idx)),
        })
        pred_all.append(pred)
        y_all.append(y_test.to_numpy())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(pred_all)
    agg = evaluate_regressor(y_true, y_pred)
    return {
        "per_fold": fold_metrics,
        "mean_rmse_log": float(np.mean([m["rmse_log"] for m in fold_metrics])),
        "std_rmse_log": float(np.std([m["rmse_log"] for m in fold_metrics])),
        "concat_rmse_log": agg["rmse_log"],
        "concat_mae_log": agg["mae_log"],
        "concat_r2_log": agg["r2_log"],
        "concat_mae_top_decile": agg["mae_log_top_decile"],
        "concat_median_err_dollars": agg["median_abs_err_dollars"],
        "n_total": int(len(df)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1 — URI ABLATION (HB_HOUSTON)")
    print("=" * 70)
    s1 = pd.read_parquet(S1_DATA)
    uri_rows_s1 = int(((s1["datetime"] >= URI_START) & (s1["datetime"] <= URI_END)).sum())
    print(f"  Stage 1 rows: {len(s1):,}  Uri-window rows: {uri_rows_s1}")
    s1_with = s1
    s1_without = filter_uri(s1, exclude=True)
    print(f"  without-Uri row count: {len(s1_without):,}  (delta: -{len(s1) - len(s1_without)})")

    # Load best-variant hyperparameters from saved pickles
    log_saved = load(PROJECT_ROOT / "models" / "logistic_best.pkl")
    rf_saved = load(PROJECT_ROOT / "models" / "rf_best.pkl")
    knn_saved = load(PROJECT_ROOT / "models" / "knn_best.pkl")
    print(f"\n  saved variants: logistic={log_saved['variant']}  rf={rf_saved['variant']}  knn={knn_saved['variant']}")

    def make_log():
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **log_saved["params"])

    def make_rf():
        return RandomForestClassifier(**rf_saved["params"])

    def make_knn():
        return KNeighborsClassifier(**knn_saved["params"], n_jobs=-1)

    s1_results = {}
    for label, factory, scale in [
        (f"logistic ({log_saved['variant']})", make_log, True),
        (f"random_forest ({rf_saved['variant']})", make_rf, False),
        (f"knn ({knn_saved['variant']})", make_knn, True),
    ]:
        print(f"\n  >>> {label}")
        with_res = stage1_cv(s1_with, factory, scale)
        without_res = stage1_cv(s1_without, factory, scale)
        s1_results[label] = {"with_uri": with_res, "without_uri": without_res}
        print(f"    with Uri:    mean PR-AUC = {with_res['mean_pr_auc']:.4f}  +/-  {with_res['std_pr_auc']:.4f}")
        print(f"    without Uri: mean PR-AUC = {without_res['mean_pr_auc']:.4f}  +/-  {without_res['std_pr_auc']:.4f}")
        print(f"    delta (without - with):    {without_res['mean_pr_auc'] - with_res['mean_pr_auc']:+.4f}")

    print("\n" + "=" * 70)
    print("STAGE 2 — URI ABLATION (HB_HOUSTON, RF regressor)")
    print("=" * 70)
    s2 = pd.read_parquet(S2_DATA)
    uri_rows_s2 = int(((s2["datetime"] >= URI_START) & (s2["datetime"] <= URI_END)).sum())
    uri_share_s2 = uri_rows_s2 / len(s2)
    print(f"  Stage 2 rows: {len(s2):,}  Uri-window spike rows: {uri_rows_s2} ({uri_share_s2*100:.1f}%)")
    s2_with = s2
    s2_without = filter_uri(s2, exclude=True)
    print(f"  without-Uri row count: {len(s2_without):,}")

    rf2_saved = load(PROJECT_ROOT / "models" / "stage2_rf_best_houston.pkl")
    print(f"  saved variant: {rf2_saved['variant']}")

    def make_rf2():
        return RandomForestRegressor(**rf2_saved["params"])

    s2_with_res = stage2_cv(s2_with, make_rf2)
    s2_without_res = stage2_cv(s2_without, make_rf2)
    print(f"  with Uri:    mean RMSE_log = {s2_with_res['mean_rmse_log']:.4f}  "
          f"top-10 MAE_log = {s2_with_res['concat_mae_top_decile']:.4f}  "
          f"med $err = ${s2_with_res['concat_median_err_dollars']:.2f}")
    print(f"  without Uri: mean RMSE_log = {s2_without_res['mean_rmse_log']:.4f}  "
          f"top-10 MAE_log = {s2_without_res['concat_mae_top_decile']:.4f}  "
          f"med $err = ${s2_without_res['concat_median_err_dollars']:.2f}")

    s2_results = {
        f"stage2_rf ({rf2_saved['variant']})": {
            "with_uri": s2_with_res, "without_uri": s2_without_res,
        }
    }

    # Save combined JSON
    payload = {
        "uri_window": {"start": str(URI_START), "end": str(URI_END)},
        "stage1": s1_results,
        "stage2": s2_results,
        "counts": {
            "stage1_total": len(s1),
            "stage1_uri_rows": uri_rows_s1,
            "stage1_uri_spikes": int(s1.loc[
                (s1["datetime"] >= URI_START) & (s1["datetime"] <= URI_END), "is_spike"
            ].sum()),
            "stage2_total": len(s2),
            "stage2_uri_rows": uri_rows_s2,
            "stage2_uri_share": uri_share_s2,
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    # Comparison plot — grouped bars: with/without per model class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    labels_s1 = list(s1_results.keys())
    x = np.arange(len(labels_s1))
    width = 0.35
    with_vals = [s1_results[l]["with_uri"]["mean_pr_auc"] for l in labels_s1]
    without_vals = [s1_results[l]["without_uri"]["mean_pr_auc"] for l in labels_s1]
    ax.bar(x - width/2, with_vals, width, label="with Uri", color="#e45756", edgecolor="white")
    ax.bar(x + width/2, without_vals, width, label="without Uri", color="#4c78a8", edgecolor="white")
    for xi, (v, w) in enumerate(zip(with_vals, without_vals)):
        ax.text(xi - width/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + width/2, w + 0.005, f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(" (")[0] for l in labels_s1], rotation=0)
    ax.set_ylabel("Mean per-fold PR-AUC")
    ax.set_title("Stage 1: Uri ablation\n(higher is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    ax = axes[1]
    labels_s2 = list(s2_results.keys())
    s2_with_vals = [s2_results[l]["with_uri"]["mean_rmse_log"] for l in labels_s2]
    s2_without_vals = [s2_results[l]["without_uri"]["mean_rmse_log"] for l in labels_s2]
    x2 = np.arange(len(labels_s2))
    ax.bar(x2 - width/2, s2_with_vals, width, label="with Uri", color="#e45756", edgecolor="white")
    ax.bar(x2 + width/2, s2_without_vals, width, label="without Uri", color="#4c78a8", edgecolor="white")
    for xi, (v, w) in enumerate(zip(s2_with_vals, s2_without_vals)):
        ax.text(xi - width/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + width/2, w + 0.005, f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x2)
    ax.set_xticklabels([l.split(" (")[0] for l in labels_s2])
    ax.set_ylabel("Mean per-fold RMSE_log")
    ax.set_title("Stage 2: Uri ablation\n(lower is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    fig.suptitle(
        f"Winter Storm Uri ablation  ({URI_START.date()} → {URI_END.date()})  •  HB_HOUSTON",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
