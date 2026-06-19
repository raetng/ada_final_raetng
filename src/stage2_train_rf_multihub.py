"""STAGE 2 Random Forest regressor on log(SPP) | SPP > $100 (multi-hub).

Multi-hub analog of stage2_train_rf.py. Same 2x2 grid, but uses the
timestamp-grouped splitter (avoids cross-hub leakage at fold boundaries).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_regressor, make_time_series_splits_grouped

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_multihub.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "rf_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "rf_best_predictions.npz"
IMP_PLOT_PATH = PROJECT_ROOT / "outputs" / "stage2" / "multihub" / "rf_feature_importance.png"
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_rf_best_multihub.pkl"

TARGET = "log_spp"
N_SPLITS = 5
RANDOM_STATE = 42
NON_FEATURE_COLS = ("datetime", "Location", "SPP", TARGET)

RF_BASE = dict(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)
GRID_DEPTH = [10, None]
GRID_LEAF = [5, 20]


def variant_name(d, l):
    return f"depth={d}_leaf={l}"


def build_variants():
    return {variant_name(d, l): {"max_depth": d, "min_samples_leaf": l}
            for d in GRID_DEPTH for l in GRID_LEAF}


def make_model(params):
    return RandomForestRegressor(**RF_BASE, **params)


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


def plot_feature_importance(model, feature_names, output_path, title):
    imp = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    print("=" * 70)
    print("STAGE 2: RANDOM FOREST REGRESSOR (MULTI-HUB)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET].astype(float)
    print(f"  shape: {df.shape}  features: {len(feature_cols)}")

    splits = make_time_series_splits_grouped(df, group_col="datetime", n_splits=N_SPLITS)
    variants = build_variants()
    print(f"\n  grid: {len(variants)} combos over max_depth={GRID_DEPTH} x min_samples_leaf={GRID_LEAF}")

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV (TIMESTAMP-GROUPED)")
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
    imp_df = plot_feature_importance(
        best_model, feature_cols, IMP_PLOT_PATH,
        f"Stage 2 RF {best_name} (multi-hub) — feature importance"
    )
    print(f"  saved importance plot -> {IMP_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print("\n  feature importances (descending):")
    print(imp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    payload = {
        "model": best_model,
        "scaler": None,
        "features": feature_cols,
        "target": TARGET,
        "stage": 2,
        "scope": "multi-hub (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_PAN)",
        "variant": best_name,
        "params": {**RF_BASE, **best_params},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
