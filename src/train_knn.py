"""Train KNN classifiers for spike prediction.

KNN serves as a non-parametric contrast to the linear and tree-based models.
Two design choices matter for an imbalanced ~4% positive class:

  - weights='distance' so nearer neighbors carry more weight, which avoids the
    pathological case where all `k` neighbors are negative simply because
    negatives outnumber positives 20:1 in any local neighborhood.
  - k swept across [5, 25, 75, 200]: small k overfits, large k washes out the
    rare-class signal. The sweep lets us see where the bias-variance tradeoff
    lands for this dataset.

KNN scales poorly because every prediction is a distance computation against
the entire training set. Wall-clock fit and predict time are recorded per fold
so we can flag this against the linear and tree-based baselines.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_classifier, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "session7" / "knn_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "session7" / "knn_best_predictions.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "knn_best.pkl"

TARGET = "is_spike"
N_SPLITS = 5
K_VALUES = [5, 25, 75, 200]


def make_model(k):
    return KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)


def run_cv(X, y, df, splits):
    variants = {f"k={k}": k for k in K_VALUES}
    fold_proba = {name: [] for name in variants}
    fold_true = {name: [] for name in variants}
    fold_metrics = {name: [] for name in variants}

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
            f"  fold {fold_idx}: train={len(X_train):,}  test={len(X_test):,}  "
            f"({test_start} -> {test_end})  test_spike_rate={y_test.mean():.4f}"
        )

        for name, k in variants.items():
            model = make_model(k)
            t0 = time.time()
            model.fit(X_train_s, y_train)
            fit_time = time.time() - t0
            t0 = time.time()
            proba = model.predict_proba(X_test_s)[:, 1]
            predict_time = time.time() - t0

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
                "fit_seconds": float(fit_time),
                "predict_seconds": float(predict_time),
            })
            print(
                f"    {name:8s} PR-AUC={res['pr_auc']:.4f}  "
                f"fit={fit_time:5.2f}s  predict={predict_time:6.2f}s"
            )

    results = {}
    for name in variants:
        y_true = np.concatenate(fold_true[name])
        y_proba = np.concatenate(fold_proba[name])
        agg = evaluate_classifier(y_true, y_proba)
        agg["per_fold"] = fold_metrics[name]
        agg["mean_pr_auc"] = float(np.mean([fm["pr_auc"] for fm in fold_metrics[name]]))
        agg["std_pr_auc"] = float(np.std([fm["pr_auc"] for fm in fold_metrics[name]]))
        agg["mean_fit_seconds"] = float(np.mean([fm["fit_seconds"] for fm in fold_metrics[name]]))
        agg["mean_predict_seconds"] = float(np.mean([fm["predict_seconds"] for fm in fold_metrics[name]]))
        results[name] = agg
    return results


def save_results_json(results, path):
    serializable = {}
    for name, res in results.items():
        serializable[name] = {k: v for k, v in res.items() if k not in ("y_true", "y_proba")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def refit_best(X, y, k):
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    model = make_model(k)
    model.fit(X_full, y)
    return model, scaler


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

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV  (k = {K_VALUES})")
    print("=" * 70)
    results = run_cv(X, y, df, splits)

    print("\n" + "=" * 70)
    print("VARIANT METRICS")
    print("=" * 70)
    print(
        f"  {'variant':8s} {'mean PR-AUC':>11s} {'std':>7s} {'concat PR-AUC':>13s} "
        f"{'ROC-AUC':>8s} {'fit (s)':>8s} {'predict (s)':>11s}"
    )
    rows = []
    for name, res in results.items():
        rows.append((name, res["mean_pr_auc"], res["std_pr_auc"], res["pr_auc"], res["roc_auc"],
                     res["mean_fit_seconds"], res["mean_predict_seconds"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, mean_pr, std_pr, pr, roc, fit_s, pred_s in rows:
        print(
            f"  {name:8s} {mean_pr:>11.4f} {std_pr:>7.4f} {pr:>13.4f} {roc:>8.4f} "
            f"{fit_s:>8.2f} {pred_s:>11.2f}"
        )

    save_results_json(results, RESULTS_PATH)
    print(f"\n  saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    best_name = max(results, key=lambda k_: results[k_]["mean_pr_auc"])
    best_k = int(best_name.split("=")[1])
    print("\n" + "=" * 70)
    print(f"BEST VARIANT (by mean PR-AUC across folds): {best_name}")
    print("=" * 70)
    print(f"  mean PR-AUC: {results[best_name]['mean_pr_auc']:.4f}  +/-  {results[best_name]['std_pr_auc']:.4f}")
    print(f"  mean fit time:     {results[best_name]['mean_fit_seconds']:.2f} s/fold")
    print(f"  mean predict time: {results[best_name]['mean_predict_seconds']:.2f} s/fold")
    print("  per-fold PR-AUC:")
    for fm in results[best_name]["per_fold"]:
        print(
            f"    fold {fm['fold']}  PR-AUC={fm['pr_auc']:.4f}  "
            f"({fm['test_start'][:10]} -> {fm['test_end'][:10]})"
        )

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_PATH, y_true=results[best_name]["y_true"], y_proba=results[best_name]["y_proba"])
    print(f"\n  saved predictions -> {PRED_PATH.relative_to(PROJECT_ROOT)}")

    best_model, best_scaler = refit_best(X, y, best_k)
    payload = {
        "model": best_model,
        "scaler": best_scaler,
        "features": feature_cols,
        "variant": best_name,
        "params": {"n_neighbors": best_k, "weights": "distance"},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved model -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
