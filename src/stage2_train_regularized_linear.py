"""STAGE 2 regularized-linear: Ridge and Lasso on log(SPP) | SPP > $100 (HB_HOUSTON).

Adds shrinkage to the stage-2 OLS baseline. The OLS baseline (stage2_train_linear.py)
had a clear collinearity tell — apparent_temperature ~+3.9 against dewpoint ~-3.2 —
plus negative out-of-fold R^2. Ridge controls that by pulling all coefficients
toward zero proportionally; Lasso does the same and also zeros out features it
deems uninformative, which is useful given how many calendar features the OLS
baseline pushed near zero.

Grid:
  - Ridge alpha in {0.1, 1.0, 10.0}
  - Lasso alpha in {0.001, 0.01, 0.1}
Winner selected by mean per-fold RMSE_log (lower is better — note this is the
opposite sign convention from stage-1's PR-AUC selection).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_regressor, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_houston.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "regularized_linear_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "regularized_linear_best_predictions.npz"
COEF_PLOT_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "regularized_linear_coefficients.png"
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_regularized_linear_best_houston.pkl"

TARGET = "log_spp"
N_SPLITS = 5
RANDOM_STATE = 42
NON_FEATURE_COLS = ("datetime", "SPP", TARGET)

VARIANTS = {
    "ridge_0.1":   ("ridge", 0.1),
    "ridge_1.0":   ("ridge", 1.0),
    "ridge_10.0":  ("ridge", 10.0),
    "lasso_0.001": ("lasso", 0.001),
    "lasso_0.01":  ("lasso", 0.01),
    "lasso_0.1":   ("lasso", 0.1),
}


def make_model(kind, alpha):
    if kind == "ridge":
        return Ridge(alpha=alpha, random_state=RANDOM_STATE)
    return Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=20000)


def run_cv(X, y, df, splits):
    fold_pred = {name: [] for name in VARIANTS}
    fold_true = {name: [] for name in VARIANTS}
    fold_metrics = {name: [] for name in VARIANTS}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_test_arr = y_test.to_numpy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        test_start = str(df["datetime"].iloc[test_idx[0]])
        test_end = str(df["datetime"].iloc[test_idx[-1]])
        print(
            f"  fold {fold_idx}: train={len(X_train):5d}  test={len(X_test):5d}  "
            f"({test_start[:10]} -> {test_end[:10]})"
        )

        for name, (kind, alpha) in VARIANTS.items():
            model = make_model(kind, alpha)
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            res = evaluate_regressor(y_test_arr, pred)
            fold_pred[name].append(pred)
            fold_true[name].append(y_test_arr)
            fold_metrics[name].append({
                "fold": fold_idx,
                "rmse_log": res["rmse_log"],
                "mae_log": res["mae_log"],
                "r2_log": res["r2_log"],
                "mae_log_top_decile": res["mae_log_top_decile"],
                "median_abs_err_dollars": res["median_abs_err_dollars"],
                "median_abs_pct_err": res["median_abs_pct_err"],
                "test_start": test_start,
                "test_end": test_end,
                "n_test": int(len(test_idx)),
            })
            print(
                f"    {name:14s} RMSE_log={res['rmse_log']:.4f}  MAE_log={res['mae_log']:.4f}  "
                f"MAE_top10={res['mae_log_top_decile']:.4f}"
            )

    results = {}
    for name in VARIANTS:
        y_true_all = np.concatenate(fold_true[name])
        y_pred_all = np.concatenate(fold_pred[name])
        agg = evaluate_regressor(y_true_all, y_pred_all)
        agg["per_fold"] = fold_metrics[name]
        agg["mean_rmse_log"] = float(np.mean([fm["rmse_log"] for fm in fold_metrics[name]]))
        agg["std_rmse_log"] = float(np.std([fm["rmse_log"] for fm in fold_metrics[name]]))
        agg["mean_mae_log"] = float(np.mean([fm["mae_log"] for fm in fold_metrics[name]]))
        agg["mean_r2_log"] = float(np.mean([fm["r2_log"] for fm in fold_metrics[name]]))
        agg["mean_mae_log_top_decile"] = float(
            np.mean([fm["mae_log_top_decile"] for fm in fold_metrics[name] if not np.isnan(fm["mae_log_top_decile"])])
        )
        results[name] = agg
    return results


def save_results_json(results, path):
    serializable = {}
    for name, res in results.items():
        serializable[name] = {k: v for k, v in res.items() if k not in ("y_true_log", "y_pred_log")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def refit_best(X, y, best_name):
    kind, alpha = VARIANTS[best_name]
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    model = make_model(kind, alpha)
    model.fit(X_full, y)
    return model, scaler


def plot_coefficients(model, feature_names, output_path, title):
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
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def main():
    print("=" * 70)
    print("STAGE 2: REGULARIZED LINEAR (HB_HOUSTON)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET].astype(float)
    print(f"  shape: {df.shape}  features: {len(feature_cols)}")

    splits = make_time_series_splits(df, n_splits=N_SPLITS)

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV")
    print("=" * 70)
    results = run_cv(X, y, df, splits)

    print("\n" + "=" * 70)
    print("VARIANT METRICS")
    print("=" * 70)
    print(
        f"  {'variant':14s} {'mean RMSE_log':>13s} {'std':>7s} {'concat RMSE_log':>15s} "
        f"{'mean R2_log':>11s} {'mean MAE_top10':>14s}"
    )
    rows = []
    for name, res in results.items():
        rows.append((name, res["mean_rmse_log"], res["std_rmse_log"], res["rmse_log"],
                     res["mean_r2_log"], res["mean_mae_log_top_decile"]))
    rows.sort(key=lambda r: r[1])  # lower RMSE_log is better
    for name, mean_rmse, std_rmse, concat_rmse, mean_r2, mae_top in rows:
        print(
            f"  {name:14s} {mean_rmse:>13.4f} {std_rmse:>7.4f} {concat_rmse:>15.4f} "
            f"{mean_r2:>+11.4f} {mae_top:>14.4f}"
        )

    save_results_json(results, RESULTS_PATH)
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    best_name = min(results, key=lambda k: results[k]["mean_rmse_log"])
    print("\n" + "=" * 70)
    print(f"BEST VARIANT (by mean per-fold RMSE_log): {best_name}")
    print("=" * 70)
    print(f"  mean RMSE_log: {results[best_name]['mean_rmse_log']:.4f}  +/-  {results[best_name]['std_rmse_log']:.4f}")
    print(f"  mean R2_log:   {results[best_name]['mean_r2_log']:+.4f}")
    print(f"  mean MAE_log top decile: {results[best_name]['mean_mae_log_top_decile']:.4f}")
    print(f"  concat median |err| in $: ${results[best_name]['median_abs_err_dollars']:.2f}")

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_PATH, y_true_log=results[best_name]["y_true_log"], y_pred_log=results[best_name]["y_pred_log"])
    print(f"  saved predictions -> {PRED_PATH.relative_to(PROJECT_ROOT)}")

    best_model, best_scaler = refit_best(X, y, best_name)
    coef_df = plot_coefficients(
        best_model, feature_cols, COEF_PLOT_PATH,
        f"Stage 2 {best_name} (HB_HOUSTON) — coefficients"
    )
    print(f"  saved coef plot -> {COEF_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print("\n  coefficients (sorted by |coef|):")
    print(coef_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    payload = {
        "model": best_model,
        "scaler": best_scaler,
        "features": feature_cols,
        "target": TARGET,
        "stage": 2,
        "scope": "HB_HOUSTON",
        "variant": best_name,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
