"""STAGE 2 HistGradientBoostingRegressor on log(SPP) | SPP > $100 (HB_HOUSTON).

Gradient boosting on log-space spike magnitudes. Fits sequential shallow
trees on residuals, which typically outperforms RF on tabular regression
when the signal is smooth in feature space. The grid sweeps learning rate
and number of iterations — the two parameters that most strongly trade off
under/over-fitting on a small (~2.8k row) dataset.

Grid: learning_rate in {0.05, 0.1}  x  max_iter in {200, 500}  = 4 variants.
Winner by mean per-fold RMSE_log.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_regressor, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_houston.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "hgb_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "hgb_best_predictions.npz"
IMP_PLOT_PATH = PROJECT_ROOT / "outputs" / "stage2" / "houston" / "hgb_feature_importance.png"
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_hgb_best_houston.pkl"

TARGET = "log_spp"
N_SPLITS = 5
RANDOM_STATE = 42
NON_FEATURE_COLS = ("datetime", "SPP", TARGET)

HGB_BASE = dict(random_state=RANDOM_STATE, early_stopping=False)
GRID_LR = [0.05, 0.1]
GRID_MAX_ITER = [200, 500]


def variant_name(lr, mi):
    return f"lr={lr}_iter={mi}"


def build_variants():
    return {variant_name(lr, mi): {"learning_rate": lr, "max_iter": mi}
            for lr in GRID_LR for mi in GRID_MAX_ITER}


def make_model(params):
    return HistGradientBoostingRegressor(**HGB_BASE, **params)


def permutation_importance_simple(model, X, y, n_repeats=3, seed=42):
    """Mean drop in negative-RMSE on shuffling each column. Cheaper than
    sklearn.inspection.permutation_importance and sufficient for a ranking."""
    rng = np.random.default_rng(seed)
    base_rmse = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
    importances = np.zeros(X.shape[1])
    for j, col in enumerate(X.columns):
        scores = []
        for _ in range(n_repeats):
            X_shuf = X.copy()
            X_shuf[col] = rng.permutation(X_shuf[col].values)
            rmse = float(np.sqrt(np.mean((model.predict(X_shuf) - y) ** 2)))
            scores.append(rmse - base_rmse)
        importances[j] = float(np.mean(scores))
    return importances


def run_cv(X, y, df, splits, variants):
    fold_pred = {name: [] for name in variants}
    fold_true = {name: [] for name in variants}
    fold_metrics = {name: [] for name in variants}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_test_arr = y_test.to_numpy()

        test_start = str(df["datetime"].iloc[test_idx[0]])
        test_end = str(df["datetime"].iloc[test_idx[-1]])
        print(
            f"  fold {fold_idx}: train={len(X_train):5d}  test={len(X_test):5d}  "
            f"({test_start[:10]} -> {test_end[:10]})"
        )

        for name, params in variants.items():
            model = make_model(params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
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
                f"    {name:18s} RMSE_log={res['rmse_log']:.4f}  MAE_log={res['mae_log']:.4f}  "
                f"MAE_top10={res['mae_log_top_decile']:.4f}  R2={res['r2_log']:+.3f}"
            )

    results = {}
    for name in variants:
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


def plot_perm_importance(importances, feature_names, output_path, title):
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Permutation importance (RMSE_log increase when shuffled)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    print("=" * 70)
    print("STAGE 2: HIST GRADIENT BOOSTING (HB_HOUSTON)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET].astype(float)
    print(f"  shape: {df.shape}  features: {len(feature_cols)}")

    splits = make_time_series_splits(df, n_splits=N_SPLITS)
    variants = build_variants()
    print(f"\n  grid: {len(variants)} combos over learning_rate={GRID_LR} x max_iter={GRID_MAX_ITER}")

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV")
    print("=" * 70)
    results = run_cv(X, y, df, splits, variants)

    print("\n" + "=" * 70)
    print("VARIANT METRICS")
    print("=" * 70)
    print(
        f"  {'variant':18s} {'mean RMSE_log':>13s} {'std':>7s} {'concat RMSE_log':>15s} "
        f"{'mean R2_log':>11s} {'mean MAE_top10':>14s}"
    )
    rows = []
    for name, res in results.items():
        rows.append((name, res["mean_rmse_log"], res["std_rmse_log"], res["rmse_log"],
                     res["mean_r2_log"], res["mean_mae_log_top_decile"]))
    rows.sort(key=lambda r: r[1])
    for name, mean_rmse, std_rmse, concat_rmse, mean_r2, mae_top in rows:
        print(
            f"  {name:18s} {mean_rmse:>13.4f} {std_rmse:>7.4f} {concat_rmse:>15.4f} "
            f"{mean_r2:>+11.4f} {mae_top:>14.4f}"
        )

    save_results_json(results, RESULTS_PATH)
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    best_name = min(results, key=lambda k: results[k]["mean_rmse_log"])
    print("\n" + "=" * 70)
    print(f"BEST VARIANT: {best_name}")
    print("=" * 70)
    print(f"  mean RMSE_log: {results[best_name]['mean_rmse_log']:.4f}  +/-  {results[best_name]['std_rmse_log']:.4f}")
    print(f"  mean R2_log:   {results[best_name]['mean_r2_log']:+.4f}")
    print(f"  mean MAE_log top decile: {results[best_name]['mean_mae_log_top_decile']:.4f}")
    print(f"  concat median |err| in $: ${results[best_name]['median_abs_err_dollars']:.2f}")

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_PATH, y_true_log=results[best_name]["y_true_log"], y_pred_log=results[best_name]["y_pred_log"])
    print(f"  saved predictions -> {PRED_PATH.relative_to(PROJECT_ROOT)}")

    best_params = variants[best_name]
    best_model = make_model(best_params)
    best_model.fit(X, y)

    print("\n  computing permutation importance on full data...")
    perm = permutation_importance_simple(best_model, X, y, n_repeats=3, seed=RANDOM_STATE)
    imp_df = plot_perm_importance(
        perm, feature_cols, IMP_PLOT_PATH,
        f"Stage 2 HGB {best_name} (HB_HOUSTON) — permutation importance"
    )
    print(f"  saved importance plot -> {IMP_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print("\n  permutation importances (descending):")
    print(imp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    payload = {
        "model": best_model,
        "scaler": None,
        "features": feature_cols,
        "target": TARGET,
        "stage": 2,
        "scope": "HB_HOUSTON",
        "variant": best_name,
        "params": {**HGB_BASE, **best_params},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
