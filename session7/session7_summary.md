# Session 7 — Modeling Summary

ERCOT day-ahead Settlement Point Price spikes at HB_HOUSTON (>= $100/MWh).
16 features (weather + Henry Hub gas + calendar), 63,791 hourly rows
(2019-01-01 → 2026-04-11), positive class rate 4.46%. All cross-validation
uses 5-fold expanding-window time-series splits with the scaler refit per
fold (no leakage).

## 1. Evaluation framework

We chose **PR-AUC (average precision) as the primary metric before training
any models**, because the positive class is rare (~4%) and ROC-AUC is
optimistic on imbalanced data — most of the ROC space is dominated by
true-negative dominance, so a model can have an impressive ROC-AUC while still
producing a mostly-useless precision-recall curve. PR-AUC reflects how well
a model ranks actual spikes near the top of its predicted-probability list,
which is what an electricity-trading or grid-operations downstream user would
care about. We report **ROC-AUC** and **precision at a fixed Recall = 0.80
operating point** as secondary diagnostics — the fixed-recall metric in
particular lets us compare models like-for-like at an operationally meaningful
sensitivity. Precision and recall at each model's F1-maximizing threshold are
also reported as operating-point context.

## 2. Simple model — Logistic regression

Six variants in a fully-crossed grid (3 regularization choices × 2 class-
weighting choices), 5-fold CV, all 16 features standardized per fold. Winner
picked by **mean per-fold PR-AUC** (each fold weighted equally) — same
selection criterion as the RF and KNN scripts, so all three model classes are
compared on a consistent basis.

| variant | mean PR-AUC | std | concat PR-AUC | ROC-AUC | P@R=0.80 |
|---|---|---|---|---|---|
| **l1_balanced** | **0.346** | 0.158 | **0.371** | **0.855** | 0.134 |
| no_regularization_balanced | 0.345 | 0.157 | 0.372 | 0.856 | 0.135 |
| l2_balanced | 0.341 | 0.159 | 0.367 | 0.853 | 0.130 |
| no_regularization | 0.273 | 0.189 | 0.328 | 0.805 | 0.124 |
| l1 (C=1.0) | 0.270 | 0.189 | 0.326 | 0.802 | 0.122 |
| l2 (C=1.0) | 0.268 | 0.192 | 0.327 | 0.799 | 0.117 |

**Winner: `l1_balanced` at mean PR-AUC = 0.346 (concat 0.371)** — but with the
caveat that the three balanced variants are statistically tied (within 0.005
mean PR-AUC of each other, against a cross-fold std of ~0.16). The crossed
design lets us answer the two design questions cleanly:

- **Regularization barely matters.** Holding the weight choice fixed, the
  effect of swapping no-reg → L1 → L2 is within ±0.005 mean PR-AUC in every
  comparison (e.g. L2 unbalanced = -0.005, L1 unbalanced = -0.002 vs no-reg).
  At C=1.0 with 16 features and ~63k rows, the regularizer has nothing to do.
- **`class_weight='balanced'` is the dominant lever and helps regardless of
  regularization.** It adds **+0.072 to +0.075 mean PR-AUC** in every
  regularization regime (no-reg: +0.072, L2: +0.074, L1: +0.075). This is
  what we'd want to see — the balanced effect doesn't depend on a particular
  regularization choice, it's a property of the loss function.

The balanced variants distort the absolute probabilities (the model implicitly
assumes a 50/50 prior instead of the true 4/96), which improves the *ranking*
of positives among negatives — exactly what PR-AUC measures — at the cost of
calibration. If a downstream user needs meaningful absolute probabilities
rather than a ranking, post-hoc isotonic calibration on a held-out set is the
standard fix.

**Top coefficients (on standardized features, l1_balanced):**

| feature | coefficient |
|---|---:|
| apparent_temperature | +11.46 |
| temperature_2m | -6.84 |
| dewpoint_2m | -4.30 |
| wind_speed_10m | +3.89 |
| wind_speed_100m | -2.82 |
| hour | +1.02 |
| shortwave_radiation | -0.92 |
| gas_price_henry_hub | +0.88 |

The very large opposite-sign coefficients on `apparent_temperature` /
`temperature_2m` and on `wind_speed_10m` / `wind_speed_100m` are a textbook
collinearity signal — the model is differencing two near-duplicate inputs to
extract a small residual. The interpretable story is still that **heat
extremity, gas price, and time-of-day drive linear spike separability**, but
we should not read individual signs literally for the collinear pairs.

## 3. Advanced models — Random Forest and KNN

Both selected on **mean per-fold PR-AUC** (each fold weighted equally).

| model | concat PR-AUC | ROC-AUC | Precision | Recall | P@R=0.80 |
|---|---|---|---|---|---|
| logistic (l1_balanced) | 0.371 | 0.855 | 0.451 | 0.330 | 0.134 |
| **random_forest** (depth=None, leaf=20) | 0.272 | **0.900** | 0.265 | 0.605 | **0.216** |
| knn (k=25, weights=distance) | 0.218 | 0.785 | 0.217 | 0.587 | 0.049 |

The **headline ranking flips between metrics**: logistic wins on PR-AUC, RF
wins on ROC-AUC, and RF is dramatically better at the operationally-useful
fixed-recall point (Precision@Recall=0.80 of 0.216 vs 0.134 — at the same 80%
recall, RF's positive predictions are nearly twice as precise). KNN is the
weakest on every metric and is largely a sanity check.

**Per-fold winners (PR-AUC, best variant of each class) — the models genuinely
disagree across regimes:**

| fold | period | test spike rate | logistic | RF | KNN | winner |
|---|---|---|---|---|---|---|
| 1 | 2020-03 → 2021-06 | 2.7% | **0.370** | 0.198 | 0.109 | logistic |
| 2 | 2021-06 → 2022-08 (Uri-adjacent) | 10.5% | **0.588** | 0.334 | 0.232 | logistic |
| 3 | 2022-08 → 2023-11 | 7.6% | 0.334 | **0.550** | 0.413 | RF |
| 4 | 2023-11 → 2025-01 | 1.7% | 0.092 | **0.102** | 0.091 | RF |
| 5 | 2025-01 → 2026-04 | 2.0% | 0.344 | 0.132 | **0.372** | KNN |

A 2-2-1 split across model classes — no single model dominates. Logistic wins
the **high-volatility** windows (folds 1-2), RF wins the **transitional**
windows (folds 3-4), and KNN wins the **most recent low-volatility** window.
Cross-fold std is large for every model (~0.16 for the logistic leader),
*larger than* the 0.08 mean-PR-AUC gap between logistic and RF — the headline
lead is well within fold-to-fold noise. An ensemble or regime-aware switch is
a natural follow-up.

## 4. Recency analysis

Held out the most recent 12 months (2025-04-12 → 2026-04-11, n=8,760, spike
rate 2.10%) as a fixed test set, then retrained the best RF on three windows
ending at the test boundary:

| training window | n_train | train spike rate | PR-AUC | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| all_history (2019-01 → 2025-04) | 55,031 | 4.84% | 0.129 | 0.916 | 0.170 | 0.446 |
| 3_years (2022-04 → 2025-04) | 26,304 | 7.27% | 0.119 | 0.908 | 0.142 | 0.478 |
| **1_year (2024-04 → 2025-04)** | **8,760** | **1.70%** | **0.180** | 0.897 | **0.235** | 0.353 |

**Recent-only training helped substantially.** The 1-year model beat
all-history by **+0.05 PR-AUC** despite using ~6× less data. The 3-year
window is the *worst* of the three because it overweights the 2022-2023
high-volatility regime — a 7.3% train spike rate against a 2.1% test spike
rate is a regime mismatch, and the 1-year window's 1.7% spike rate matches
the test distribution much more closely. This supports the **"the grid has
changed and recent data captures that"** narrative: post-Uri ERCOT
volatility has subsided, more transmission and renewables are online, and
hourly spike dynamics in 2024-2026 do not look like 2021-2022. Worth noting
that ROC-AUC barely moves across windows (0.90 → 0.90 → 0.90) — the ranking
ability is preserved, what changes is the precision-vs-recall tradeoff, which
is exactly what a probability-distribution shift looks like.

## 5. Open questions for the final checkpoint

1. **Spike magnitude prediction (regression task).** All Session 7 work is
   binary classification. The final deliverable asks us to predict how
   *severe* the spike is, conditional on a spike occurring. Plan: fit a
   regression model (likely log-price or quantile regression) on the
   spike-only subset, using the same features. Two-stage architecture:
   classifier gates whether to spike, regressor predicts magnitude.

2. **Strongest predictors and articulation.** Logistic and RF agree on the
   *headline four* — temperature, apparent temperature, gas price, hour —
   but disagree on the rest. We need a more principled feature-importance
   reconciliation: permutation importance on a single held-out window
   (avoiding the impurity-based bias that favors high-cardinality features
   in RF), and partial-dependence plots for the top features so the writeup
   can describe the *shape* of the relationships, not just their rank.

3. **Are there multiple "types" of spikes?** The fold-by-fold disagreements
   (heat-volatility regimes vs calmer recent periods) and the recency
   finding both suggest yes. A clustering pass over spike rows (k-means or
   GMM on the feature matrix at spike timestamps) would let us label spikes
   as e.g. "summer heat", "winter cold-snap", "shoulder-season anomaly" and
   re-evaluate model performance per cluster. This addresses both
   "predictor strength" (does it differ by spike type?) and the structural-
   change story directly.

4. **Calibration follow-up.** `class_weight='balanced'` improves the
   *ranking* of positives among negatives (good for PR-AUC) but distorts
   the absolute probabilities, because it makes the model implicitly
   assume a 50/50 prior instead of the true 4/96. For any downstream use
   that needs meaningful probabilities rather than just a ranking, isotonic
   or sigmoid calibration (`CalibratedClassifierCV`) on a held-out set is
   a quick win that recovers calibrated outputs without giving up the
   ranking improvement.

5. **Ensembling and regime-aware models.** Given the per-fold split, a
   simple averaging ensemble of logistic + RF, or a switch keyed on
   trailing 30-day spike rate, deserves at least a sanity check.
