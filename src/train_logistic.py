"""Train logistic-regression baselines for spike prediction.

Baseline that more complex models must beat. The grid is fully crossed across
two axes — regularization choice (none / L2 / L1) and class weighting (none /
balanced) — for 6 variants total under 5-fold expanding-window time-series CV.
Crossing both axes lets us tell whether `class_weight='balanced'` helps
regardless of regularization choice, or only at L2.

The scaler is fit only on each fold's training data to avoid leakage. Per-fold
predictions are concatenated before computing aggregate metrics so the headline
PR-AUC is computed on every row exactly once. Per-fold metrics are also saved
so `compare_models.py` can answer "which fold does each model class win on?".

Winner is selected by **mean per-fold PR-AUC** (each fold weighted equally),
matching the selection criterion used in train_random_forest.py and train_knn.py
so all three model classes are picked on the same basis.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_classifier, make_time_series_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "session7" / "logistic_results.json"
PRED_PATH = PROJECT_ROOT / "outputs" / "session7" / "logistic_best_predictions.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_best.pkl"

TARGET = "is_spike"
N_SPLITS = 5
RANDOM_STATE = 42

VARIANTS = {
    "no_regularization":          dict(penalty="l2", C=1e10, solver="lbfgs"),
    "no_regularization_balanced": dict(penalty="l2", C=1e10, solver="lbfgs",     class_weight="balanced"),
    "l2":                         dict(penalty="l2", C=1.0,  solver="lbfgs"),
    "l2_balanced":                dict(penalty="l2", C=1.0,  solver="lbfgs",     class_weight="balanced"),
    "l1":                         dict(penalty="l1", C=1.0,  solver="liblinear"),
    "l1_balanced":                dict(penalty="l1", C=1.0,  solver="liblinear", class_weight="balanced"),
}


def make_model(params):
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)


def run_cv(X, y, df, splits):
    fold_proba = {name: [] for name in VARIANTS}
    fold_true = {name: [] for name in VARIANTS}
    fold_metrics = {name: [] for name in VARIANTS}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        test_start = str(df["datetime"].iloc[test_idx[0]])
        test_end = str(df["datetime"].iloc[test_idx[-1]])
        print(
            f"  fold {fold_idx}: train={len(X_train):,}  test={len(X_test):,}  "
            f"({test_start} -> {test_end})  test_spike_rate={y_test.mean():.4f}"
        )

        for name, params in VARIANTS.items():
            model = make_model(params)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            y_test_arr = y_test.to_numpy()
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

    results = {}
    for name in VARIANTS:
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


def refit_best(X, y, best_name):
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    model = make_model(VARIANTS[best_name])
    model.fit(X_full, y)
    return model, scaler


def print_coefficients(model, feature_names):
    coefs = model.coef_[0]
    coef_df = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    print(coef_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    return coef_df


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
    print(f"{N_SPLITS}-FOLD EXPANDING-WINDOW CV")
    print("=" * 70)
    results = run_cv(X, y, df, splits)

    print("\n" + "=" * 70)
    print("VARIANT METRICS")
    print("=" * 70)
    print(
        f"  {'variant':28s} {'mean PR-AUC':>11s} {'std':>7s} {'concat PR-AUC':>13s} "
        f"{'ROC-AUC':>8s} {'P@R=.80':>8s}"
    )
    rows = []
    for name, res in results.items():
        rows.append((name, res["mean_pr_auc"], res["std_pr_auc"], res["pr_auc"], res["roc_auc"],
                     res["precision_at_recall_0.80"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, mean_pr, std_pr, pr, roc, par80 in rows:
        print(
            f"  {name:28s} {mean_pr:>11.4f} {std_pr:>7.4f} {pr:>13.4f} {roc:>8.4f} "
            f"{par80:>8.4f}"
        )

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

    best_model, best_scaler = refit_best(X, y, best_name)

    print("\n  coefficients (on standardized features, sorted by |coef|):")
    print_coefficients(best_model, feature_cols)

    payload = {
        "model": best_model,
        "scaler": best_scaler,
        "features": feature_cols,
        "variant": best_name,
        "params": VARIANTS[best_name],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pr = {k: results[k]["mean_pr_auc"] for k in VARIANTS}
    print(f"  Winner on mean per-fold PR-AUC: {best_name}  (mean PR-AUC = {pr[best_name]:.4f})")
    print("\n  Effect of regularization (vs no_regularization, holding weight fixed):")
    print(f"    L2 unbalanced:   {pr['l2'] - pr['no_regularization']:+.4f}")
    print(f"    L1 unbalanced:   {pr['l1'] - pr['no_regularization']:+.4f}")
    print(f"    L2 balanced:     {pr['l2_balanced'] - pr['no_regularization_balanced']:+.4f}")
    print(f"    L1 balanced:     {pr['l1_balanced'] - pr['no_regularization_balanced']:+.4f}")
    print("\n  Effect of class_weight='balanced' (vs unbalanced, holding regularization fixed):")
    print(f"    no_regularization:  {pr['no_regularization_balanced'] - pr['no_regularization']:+.4f}")
    print(f"    L2:                 {pr['l2_balanced'] - pr['l2']:+.4f}")
    print(f"    L1:                 {pr['l1_balanced'] - pr['l1']:+.4f}")


if __name__ == "__main__":
    main()
