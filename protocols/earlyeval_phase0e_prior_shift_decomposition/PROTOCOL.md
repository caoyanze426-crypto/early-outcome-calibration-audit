# EARLYEVAL_PHASE0E - PRIOR_SHIFT_DECOMPOSITION - PROTOCOL

Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, new folds = 0, threshold sweep = 0, TerminalBench = NOT RUN, Toolathlon = NOT RUN.

## Question

Does agent-conditional miscalibration remain after an oracle label-prior (base-rate) correction, or do the calibration gaps largely collapse so that the Phase 0D finding is better described as prior-shift adaptation?

Decomposition only. No causal claim.

## Inputs

- the exact frozen Phase 0B stop-point predictions, rebuilt by the same `(traj_id, decision_step)` join to the held-out prefix predictions that Phase 0D used and verified
- the frozen Phase 0D fold priors (`prevalence_shift.csv`) and the frozen Phase 0D gap tables, used as the identity reference

Every consumed input is checked against the producing phase's own `artifact_sha256sums.txt`. Predictions are never rebuilt and no probability is re-fit.

## Method

1. TRAIN_SUCCESS_PRIOR_m and TEST_SUCCESS_PRIOR_m per held-out model, recomputed from the frozen trajectory universe and cross-checked against Phase 0D.
2. Oracle base-rate correction. Each stop-point head probability is shifted in log-odds by ln r_m, where r_m = [pi_test/(1-pi_test)] / [pi_train/(1-pi_train)] and p = sigmoid(logit(p) + ln r_m). The success head's positive class is a resolved success; the failure head's positive class is the complement, so its odds ratio is the exact reciprocal and its log shift is exactly -ln r_m. Empirical precision is unchanged by the correction because the correction moves probabilities, not decisions.
3. Recomputed per-model per-head calibration gap = mean corrected decision score - empirical precision, kept beside the raw gap.
4. Residual gap range per head across non-degenerate models with a defined denominator, beside the raw range, with task-cluster bootstrap CIs (2000 replicates, seed 42, instance_id resampled with all model trajectories of a selected task entering together).
5. Spearman association of PRIOR_SHIFT with the raw and the residual calibration error, with bootstrap CIs.
6. Degenerate prior: a held-out success prior of exactly 0 or 1 makes the oracle odds ratio 0 or +inf and puts the corrected probabilities on the boundary. Such models are flagged `prior_degenerate`, are reported in every table, and are excluded from the gate and from the residual range, because a boundary-valued correction is not a valid oracle correction in either direction.

## Pre-registered gate

RESIDUAL_LARGE_GAP_COUNT = held-out models that are not prior-degenerate, have >= 20 decisions in at least one head and |corrected calibration gap| >= 0.10 in at least one head.

RESIDUAL_HETEROGENEITY_SUBSTANTIAL = either head has a corrected calibration-gap range >= 0.10 across non-degenerate models with a defined denominator AND a task-cluster bootstrap 95% CI lower bound > 0.05 for that range.

OVERALL = GO iff RESIDUAL_LARGE_GAP_COUNT >= 3 AND RESIDUAL_HETEROGENEITY_SUBSTANTIAL. PIVOT_TO_PRIOR_SHIFT_ADAPTATION iff RESIDUAL_LARGE_GAP_COUNT <= 1 AND NOT RESIDUAL_HETEROGENEITY_SUBSTANTIAL. Otherwise PARTIAL_RESIDUAL_MISCALIBRATION, a mechanically defined middle category that the two-way rule does not cover. Thresholds were not altered after seeing results.

## Interpretation boundary

Even if GO, no claim that prior shift causes miscalibration, that the Oracle correction is how a deployed system should be adjusted, that Platt scaling is invalid, or that EarlyEval is unsafe. The correction is an analysis transform on frozen probabilities, not a new predictor.
