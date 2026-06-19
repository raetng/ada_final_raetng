"""Train Random Forest models for spike prediction.

The grid is intentionally small (3x3 = 9 combinations) — chronological CV is
expensive and we don't want to claim hyperparameter tuning results that are
swamped by fold-to-fold variance. The combination (max_depth=None,
min_samples_leaf=5) is the recommended baseline; the grid surrounds it.

Per the user's brief, the winning combination is selected by **mean PR-AUC
across folds**, not by PR-AUC computed on concatenated predictions — these
are subtly different aggregations and the mean variant treats each fold equally.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_classifier, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "session7" / "rf_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "session7" / "rf_best_predictions.npz"
IMP_PLOT_PATH = PROJECT_ROOT / "outputs" / "session7" / "rf_feature_importance.png"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_best.pkl"

TARGET = "is_spike"
N_SPLITS = 5
RANDOM_STATE = 42

RF_BASE = dict(
    n_estimators=300,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
)
GRID_DEPTH = [10, 20, None]
GRID_LEAF = [1, 5, 20]


def variant_name(max_depth, min_samples_leaf):
    return f"depth={max_depth}_leaf={min_samples_leaf}"


def build_variants():
    variants = {}
    for d in GRID_DEPTH:
        for l in GRID_LEAF:
            variants[variant_name(d, l)] = {"max_depth": d, "min_samples_leaf": l}
    return variants


def make_model(grid_params):
    return RandomForestClassifier(**RF_BASE, **grid_params)


def run_cv(X, y, df, splits, variants):
    fold_proba = {name: [] for name in variants}
    fold_true = {name: [] for name in variants}
    fold_metrics = {name: [] for name in variants}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_test_arr = y_test.to_numpy()

        test_start = str(df["datetime"].iloc[test_idx[0]])
        test_end = str(df["datetime"].iloc[test_idx[-1]])
        print(
            f"  fold {fold_idx}: train={len(X_train):,}  test={len(X_test):,}  "
            f"({test_start} -> {test_end})  test_spike_rate={y_test.mean():.4f}"
        )

        for name, params in variants.items():
            model = make_model(params)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            fold_proba[name].append(proba)
            fold_true[name].append(y_test_arr)
            res = evaluate_classifier(y_test_arr, proba)
            fold_metrics[name].append({
                "fold": fold_idx,
                "pr_auc": res["pr_auc"],
                "roc_auc": res["roc_auc"],
                "f1": res["f1"],
                "brier_score": res["brier_score"],
                "test_start": test_start,
                "test_end": test_end,
                "n_test": int(len(test_idx)),
                "test_spike_rate": float(y_test.mean()),
            })
            print(f"    {name:22s} PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    results = {}
    for name in variants:
        y_true = np.concatenate(fold_true[name])
        y_proba = np.concatenate(fold_proba[name])
        agg = evaluate_classifier(y_true, y_proba)
        agg["per_fold"] = fold_metrics[name]
        agg["mean_pr_auc"] = float(np.mean([fm["pr_auc"] for fm in fold_metrics[name]]))
        agg["std_pr_auc"] = float(np.std([fm["pr_auc"] for fm in fold_metrics[name]]))
        results[name] = agg
    return results


def save_results_json(results, path):
    serializable = {}
    for name, res in results.items():
        serializable[name] = {k: v for k, v in res.items() if k not in ("y_true", "y_proba")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest feature importance (best variant, refit on full data)")
    ax.grid(True, axis="x", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    print("=" * 70)
    print("LOAD MODELING DATA")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("datetime", TARGET)]
    X = df[feature_cols]
    y = df[TARGET].astype(int)
    print(f"  shape: {df.shape}  features: {len(feature_cols)}  positives: {int(y.sum()):,} ({y.mean()*100:.2f}%)")

    splits = make_time_series_splits(df, n_splits=N_SPLITS)
    variants = build_variants()
    print(f"\n  grid: {len(variants)} combinations over max_depth={GRID_DEPTH} x min_samples_leaf={GRID_LEAF}")

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV")
    print("=" * 70)
    results = run_cv(X, y, df, splits, variants)

    print("\n" + "=" * 70)
    print("VARIANT METRICS")
    print("=" * 70)
    print(f"  {'variant':22s} {'mean PR-AUC':>11s} {'std':>7s} {'concat PR-AUC':>13s} {'ROC-AUC':>8s} {'P@R=.80':>8s}")
    rows = []
    for name, res in results.items():
        rows.append((name, res["mean_pr_auc"], res["std_pr_auc"], res["pr_auc"], res["roc_auc"], res["precision_at_recall_0.80"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, mean_pr, std_pr, pr, roc, par80 in rows:
        print(f"  {name:22s} {mean_pr:>11.4f} {std_pr:>7.4f} {pr:>13.4f} {roc:>8.4f} {par80:>8.4f}")

    save_results_json(results, RESULTS_PATH)
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    best_name = max(results, key=lambda k: results[k]["mean_pr_auc"])
    print("\n" + "=" * 70)
    print(f"BEST VARIANT (by mean PR-AUC across folds): {best_name}")
    print("=" * 70)
    print(f"  mean PR-AUC: {results[best_name]['mean_pr_auc']:.4f}  +/-  {results[best_name]['std_pr_auc']:.4f}")
    print("  per-fold PR-AUC:")
    for fm in results[best_name]["per_fold"]:
        print(
            f"    fold {fm['fold']}  PR-AUC={fm['pr_auc']:.4f}  "
            f"({fm['test_start'][:10]} -> {fm['test_end'][:10]})"
        )

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_PATH, y_true=results[best_name]["y_true"], y_proba=results[best_name]["y_proba"])
    print(f"\n  saved predictions -> {PRED_PATH.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("REFIT BEST ON FULL DATA")
    print("=" * 70)
    best_params = {k: v for k, v in variants[best_name].items()}
    best_model = make_model(best_params)
    best_model.fit(X, y)

    imp_df = plot_feature_importance(best_model, feature_cols, IMP_PLOT_PATH)
    print("  feature importances (descending):")
    print(imp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  saved importance plot -> {IMP_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    payload = {
        "model": best_model,
        "scaler": None,
        "features": feature_cols,
        "variant": best_name,
        "params": {**RF_BASE, **best_params},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved model -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
