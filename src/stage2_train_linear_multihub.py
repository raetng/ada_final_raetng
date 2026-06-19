"""STAGE 2 baseline: OLS linear regression of log(SPP) | SPP > $100 (multi-hub).

Multi-hub analog of stage2_train_linear.py. Pools spike rows across the four
real HB_* hubs (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_PAN) and uses the
timestamp-grouped time-series splitter so fold boundaries respect multi-hub
timestamps (same reasoning as the stage-1 multi-hub trainers).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_regressor, make_time_series_splits_grouped

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_multihub.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "linear_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "linear_predictions.npz"
COEF_PLOT_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "linear_coefficients.png"
SCATTER_PLOT_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "predicted_vs_actual.png"
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_linear_baseline_multihub.pkl"

TARGET = "log_spp"
N_SPLITS = 5
NON_FEATURE_COLS = ("datetime", "Location", "SPP", TARGET)


def run_cv(X, y, df, splits):
    fold_pred = []
    fold_true = []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        y_test_arr = y_test.to_numpy()

        test_start = str(df["datetime"].iloc[test_idx[0]])
        test_end = str(df["datetime"].iloc[test_idx[-1]])
        res = evaluate_regressor(y_test_arr, pred)
        print(
            f"  fold {fold_idx}: train={len(X_train):5d}  test={len(X_test):5d}  "
            f"({test_start[:10]} -> {test_end[:10]})  "
            f"RMSE_log={res['rmse_log']:.4f}  MAE_log={res['mae_log']:.4f}  "
            f"R2={res['r2_log']:+.4f}  MdAPE={res['median_abs_pct_err']*100:5.1f}%"
        )

        fold_pred.append(pred)
        fold_true.append(y_test_arr)
        fold_metrics.append({
            "fold": fold_idx,
            "rmse_log": res["rmse_log"],
            "mae_log": res["mae_log"],
            "r2_log": res["r2_log"],
            "median_abs_err_dollars": res["median_abs_err_dollars"],
            "median_abs_pct_err": res["median_abs_pct_err"],
            "test_start": test_start,
            "test_end": test_end,
            "n_test": int(len(test_idx)),
        })

    y_true_all = np.concatenate(fold_true)
    y_pred_all = np.concatenate(fold_pred)
    agg = evaluate_regressor(y_true_all, y_pred_all)
    agg["per_fold"] = fold_metrics
    agg["mean_rmse_log"] = float(np.mean([fm["rmse_log"] for fm in fold_metrics]))
    agg["std_rmse_log"] = float(np.std([fm["rmse_log"] for fm in fold_metrics]))
    agg["mean_r2_log"] = float(np.mean([fm["r2_log"] for fm in fold_metrics]))
    return agg


def save_results_json(result, path):
    serializable = {k: v for k, v in result.items() if k not in ("y_true_log", "y_pred_log")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def refit_full(X, y):
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_full, y)
    return model, scaler


def plot_coefficients(model, feature_names, output_path):
    coefs = model.coef_
    coef_df = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        })
        .sort_values("abs_coefficient", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4c78a8" if c > 0 else "#e45756" for c in coef_df["coefficient"]]
    ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized features, target = log SPP)")
    ax.set_title("Stage 2 OLS (multi-hub) — coefficients on standardized features")
    ax.grid(True, axis="x", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def plot_predicted_vs_actual(y_true_log, y_pred_log, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_log, y_pred_log, alpha=0.3, s=12, color="#4c78a8")
    lo = float(min(y_true_log.min(), y_pred_log.min()))
    hi = float(max(y_true_log.max(), y_pred_log.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual log(SPP)")
    ax.set_ylabel("Predicted log(SPP)")
    ax.set_title("Stage 2 OLS (multi-hub) — out-of-fold predictions")
    ax.legend()
    ax.grid(True, alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("STAGE 2: LOAD MAGNITUDE-REGRESSION DATA (MULTI-HUB)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET].astype(float)
    print(
        f"  shape: {df.shape}  features: {len(feature_cols)}  "
        f"target=log(SPP) on SPP > $100"
    )
    print(f"  unique timestamps: {df['datetime'].nunique():,}")
    print(f"  hubs: {sorted(df['Location'].unique())}")
    print(f"  log_spp mean={y.mean():.3f}  std={y.std():.3f}  range=[{y.min():.3f}, {y.max():.3f}]")

    splits = make_time_series_splits_grouped(df, group_col="datetime", n_splits=N_SPLITS)

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV (TIMESTAMP-GROUPED)")
    print("=" * 70)
    result = run_cv(X, y, df, splits)

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (out-of-fold)")
    print("=" * 70)
    print(f"  mean per-fold RMSE_log: {result['mean_rmse_log']:.4f}  +/-  {result['std_rmse_log']:.4f}")
    print(f"  mean per-fold R^2_log:  {result['mean_r2_log']:+.4f}")
    print(f"  concat RMSE_log:        {result['rmse_log']:.4f}")
    print(f"  concat MAE_log:         {result['mae_log']:.4f}")
    print(f"  concat R^2_log:         {result['r2_log']:+.4f}")
    print(f"  median |err| in $:      ${result['median_abs_err_dollars']:.2f}")
    print(f"  median |pct err|:       {result['median_abs_pct_err']*100:.1f}%")

    save_results_json(result, RESULTS_PATH)
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_PATH, y_true_log=result["y_true_log"], y_pred_log=result["y_pred_log"])
    print(f"  saved predictions -> {PRED_PATH.relative_to(PROJECT_ROOT)}")

    model, scaler = refit_full(X, y)
    coef_df = plot_coefficients(model, feature_cols, COEF_PLOT_PATH)
    print(f"  saved coefficient plot -> {COEF_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    plot_predicted_vs_actual(result["y_true_log"], result["y_pred_log"], SCATTER_PLOT_PATH)
    print(f"  saved predicted-vs-actual plot -> {SCATTER_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    print("\n  coefficients (standardized, sorted by |coef|):")
    print(coef_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    payload = {
        "model": model,
        "scaler": scaler,
        "features": feature_cols,
        "target": TARGET,
        "stage": 2,
        "scope": "multi-hub (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_PAN)",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
