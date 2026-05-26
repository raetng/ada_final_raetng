"""Compare logistic, RF, and KNN on the multi-hub dataset.

Multi-hub analog of compare_models.py. Pulls each class's per-variant JSON
from outputs/final/, picks the best variant per class by mean per-fold PR-AUC,
emits the markdown comparison table + PR/ROC curve overlays, and prints the
per-fold winner table.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import plot_pr_curve, plot_roc_curve, summarize_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = PROJECT_ROOT / "outputs" / "final"

MODEL_FILES = {
    "logistic": {
        "results": SESSION_DIR / "logistic_results.json",
        "predictions": SESSION_DIR / "logistic_best_predictions.npz",
    },
    "random_forest": {
        "results": SESSION_DIR / "rf_results.json",
        "predictions": SESSION_DIR / "rf_best_predictions.npz",
    },
    "knn": {
        "results": SESSION_DIR / "knn_results.json",
        "predictions": SESSION_DIR / "knn_best_predictions.npz",
    },
}

SUMMARY_PATH = SESSION_DIR / "model_comparison.md"
PR_PLOT_PATH = SESSION_DIR / "pr_curves_comparison.png"
ROC_PLOT_PATH = SESSION_DIR / "roc_curves_comparison.png"


def mean_per_fold_pr_auc(variant_result):
    return float(np.mean([fm["pr_auc"] for fm in variant_result["per_fold"]]))


def load_best_per_class():
    best = {}
    for cls, paths in MODEL_FILES.items():
        all_variants = json.loads(paths["results"].read_text())
        best_name = max(all_variants, key=lambda v: mean_per_fold_pr_auc(all_variants[v]))
        res = dict(all_variants[best_name])

        with np.load(paths["predictions"]) as data:
            res["y_true"] = data["y_true"]
            res["y_proba"] = data["y_proba"]

        res["model_class"] = cls
        res["variant"] = best_name
        res["mean_pr_auc_computed"] = mean_per_fold_pr_auc(res)
        best[f"{cls} ({best_name})"] = res
    return best


def per_fold_winners(best):
    fold_table = {}
    for label, res in best.items():
        for fm in res["per_fold"]:
            fold_table.setdefault(fm["fold"], {})[label] = {
                "pr_auc": fm["pr_auc"],
                "test_start": fm["test_start"],
                "test_end": fm["test_end"],
                "test_spike_rate": fm["test_spike_rate"],
            }
    return dict(sorted(fold_table.items()))


def print_per_fold_table(fold_table, model_labels):
    print("=" * 90)
    print("PER-FOLD PR-AUC (MULTI-HUB)")
    print("=" * 90)
    header = f"  {'fold':4s} {'period':28s} {'spike':>6s}"
    for label in model_labels:
        header += f"  {label[:22]:>22s}"
    header += f"  {'winner':>16s}"
    print(header)

    win_count = {label: 0 for label in model_labels}
    for fold, entries in fold_table.items():
        any_label = next(iter(entries.values()))
        period = f"{any_label['test_start'][:10]} -> {any_label['test_end'][:10]}"
        spike = any_label["test_spike_rate"]

        scores = {label: entries[label]["pr_auc"] for label in model_labels if label in entries}
        winner = max(scores, key=scores.get)
        win_count[winner] += 1

        line = f"  {fold:>4d} {period:28s} {spike:>6.3f}"
        for label in model_labels:
            v = scores.get(label, float("nan"))
            mark = " *" if label == winner else "  "
            line += f"  {v:>20.4f}{mark}"
        line += f"  {winner.split(' (')[0]:>16s}"
        print(line)

    print("\n  fold-win counts:")
    for label, n in win_count.items():
        print(f"    {label:40s} wins {n}/{len(fold_table)} folds")
    return win_count


def print_interpretation(best, win_count):
    print("\n" + "=" * 90)
    print("INTERPRETATION (MULTI-HUB)")
    print("=" * 90)
    ranked = sorted(best.items(), key=lambda kv: kv[1]["mean_pr_auc_computed"], reverse=True)
    leader_label, leader = ranked[0]
    runner_label, runner = ranked[1]
    third_label, third = ranked[2]

    leader_fold_pr = np.array([fm["pr_auc"] for fm in leader["per_fold"]])
    runner_fold_pr = np.array([fm["pr_auc"] for fm in runner["per_fold"]])

    gap = leader["mean_pr_auc_computed"] - runner["mean_pr_auc_computed"]
    leader_std = float(np.std(leader_fold_pr))
    n_folds = len(leader["per_fold"])
    leader_wins = win_count[leader_label]

    decisive = abs(gap) > leader_std
    fold_dominance = leader_wins / n_folds
    spread = "concentrated on the leader" if fold_dominance >= 0.6 else "split across model classes"

    print(
        f"  Leader on mean per-fold PR-AUC: {leader_label} at {leader['mean_pr_auc_computed']:.4f}, "
        f"ahead of {runner_label} ({runner['mean_pr_auc_computed']:.4f}) by {gap:+.4f}, "
        f"with {third_label} third ({third['mean_pr_auc_computed']:.4f})."
    )
    print(
        f"  The leader's cross-fold std is {leader_std:.4f}, "
        f"{'larger' if not decisive else 'smaller'} than the {abs(gap):.4f} gap to the runner-up, so "
        f"the headline lead is {'within' if not decisive else 'beyond'} the natural fold-to-fold variance "
        f"of any single model."
    )
    print(
        f"  Per-fold winners are {spread}: {leader_wins}/{n_folds} folds to the leader."
    )


def main():
    print("=" * 90)
    print("LOAD BEST VARIANT FROM EACH MODEL CLASS (MULTI-HUB)")
    print("=" * 90)
    best = load_best_per_class()
    for label, res in best.items():
        print(
            f"  {label:40s} PR-AUC={res['pr_auc']:.4f}  "
            f"ROC-AUC={res['roc_auc']:.4f}  F1={res['f1']:.4f}"
        )

    summary_df = summarize_results(best, SUMMARY_PATH)
    print(f"\n  saved comparison table -> {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")

    plot_pr_curve(best, PR_PLOT_PATH)
    plot_roc_curve(best, ROC_PLOT_PATH)
    print(f"  saved PR curves       -> {PR_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  saved ROC curves      -> {ROC_PLOT_PATH.relative_to(PROJECT_ROOT)}")

    print()
    fold_table = per_fold_winners(best)
    win_count = print_per_fold_table(fold_table, list(best.keys()))

    print_interpretation(best, win_count)


if __name__ == "__main__":
    main()
