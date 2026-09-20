# EARLYEVAL_PHASE0D - CROSS_AGENT_CALIBRATION_TRANSFER_AUDIT - PROTOCOL

Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, new LightGBM folds = 0, threshold sweep = 0, TerminalBench = NOT RUN, Toolathlon = NOT RUN.

## Research question

Does a globally calibrated EarlyEval-style stopping policy provide uniform decision reliability across unseen agent models, or does reliability vary systematically with the held-out agent's outcome distribution?

Offline association audit only; no causal claim.

## Reused frozen inputs (byte-identical Phase 0B artifacts)

- `predictions/trajectory_policy_decisions.csv` (0.95/0.95 dual policy)
- `analysis/trajectory_outcomes.csv` (trajectory universe, resolved labels)
- `analysis/per_model_summary.csv`
- `folds/fold_manifest.csv` (fold manifests)
- `predictions/heldout_prefix_predictions_all.parquet` (held-out calibrated prefix predictions)

Predictions are never rebuilt.

## Method

1. Held-out outcome prevalence per model: TEST_SUCCESS_RATE_m, TRAIN_SUCCESS_RATE_m over the other 9 model labels of that fold, PRIOR_SHIFT_m = TEST - TRAIN. No smoothing.
2. Decision-class reliability per model: precision and error rate within SUCCESS decisions and within FAILURE decisions; NA when the denominator is 0, never imputed.
3. Binomial uncertainty: 95% Wilson score intervals (no normal approximation), denominator reported explicitly.
4. Cross-agent heterogeneity: max/min/range/std of per-model precision, with task-cluster (instance_id) bootstrap CIs for the ranges (2000 replicates, seed 42).
5. Prior-shift association: Spearman rho of PRIOR_SHIFT against the per-model success error rate (expected negative) and failure error rate (expected positive), plus task-cluster bootstrap CIs.
6. Decision-score calibration at the stop point: the calibrated head probability at each trajectory's actual stopping prefix, compared with empirical precision; CALIBRATION_GAP = mean score - precision. Probabilities are not altered.
7. Degenerate-fold sensitivity excluding gemini-3-pro (declared, non-primary).

## Pre-registered gate (section 10)

HETEROGENEITY_SIGNAL: either head has precision range >= 0.15 AND the task-cluster bootstrap 95% CI lower bound for that range > 0.05.

PRIOR_SHIFT_SIGNAL: SUCCESS rho <= -0.50 OR FAILURE rho >= +0.50, AND the same sign retained after excluding gemini-3-pro.

CALIBRATION_TRANSFER_SIGNAL: at least 3 held-out models with denominator >= 20 decisions have absolute calibration gap >= 0.10 in at least one head.

OVERALL: GO_CANDIDATE iff at least two of the three are TRUE, otherwise NO_STRONG_SIGNAL. Thresholds were not altered after seeing results.

## Interpretation boundary

Even if GO_CANDIDATE, no claim that prior shift causes miscalibration, that Platt scaling is invalid, or that EarlyEval is unsafe. Only agent-conditional reliability heterogeneity and association with the held-out outcome distribution within this frozen setting are supported.
