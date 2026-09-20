# EARLYEVAL_PHASE1B - TARGET_SPECIFIC_THRESHOLD_ROBUSTNESS

Frozen scientific state: Phase 1A = PARTIAL_PASS, Phase 1A-D = STRONG_TARGET_SPECIFIC_SIGNAL. Only the two frozen targets are tested:
TARGET_1 = gpt-5-mini / SUCCESS, TARGET_2 = claude-opus-4.6 / FAILURE.
No target is added, replaced or selected post hoc, and broad pairwise heterogeneity is not re-opened (Phase 1A: SAME_PREDICTOR_HETEROGENEITY = FALSE).

## Policies

Exactly four symmetric policies, in declared order:
T1 = 0.900, T2 = 0.925, T3 = 0.950, T4 = 0.975 with success_thr = failure_thr = t, policy_mode = dual, min_step = 0, consecutive = 1.
0.950 must reproduce the frozen Phase 1A decision tables exactly.

## What is reused and what is not

All 45 frozen leave-two-agent-out predictors are reused. Only the stopping policy is re-applied to the frozen held-out prefix probabilities in predictions/per_pair_target_predictions/. No training, no feature rebuilding, no recalibration refitting, no prediction regeneration, no task additions or removals, and the Phase 1A common-task-support rule is reused unchanged.

## Prior correction (frozen Phase 0E / Phase 1A definition)

ln r = log_odds(pi_target) - log_odds(pi_train); corrected stop score = sigmoid(logit(stop score) + sign * ln r) with sign = +1 for the success head and -1 for the failure head. Empirical precision is the fraction of correct decisions for that head. The signed corrected calibration gap is the mean corrected stop score minus empirical precision. Boundaries (pi exactly 0 or 1) are preserved without smoothing.

## Eligibility (section 6)

An occurrence is eligible iff its decision denominator >= 20 and the target prior is non-degenerate. Denominators are recorded and never imputed.

## Bootstrap (section 8)

Cluster = instance_id, 2000 replicates, seed = 42100 + threshold_index*10 + target_index with threshold_index = 0..3 in the declared threshold order and target_index = 0 for TARGET_1, 1 for TARGET_2. Within every replicate the empirical precision, corrected mean stop score and corrected gap are recomputed, and the target-level median signed corrected gap is reported with a percentile 95% CI. No predictor is retrained inside the bootstrap.

## Gates (frozen before any result was inspected)

THRESHOLD_POINT_ROBUST = TRUE iff eligible occurrences >= 7 AND median absolute corrected gap >= 0.06 AND same-sign fraction >= 0.75 AND the bootstrap 95% CI for the median signed gap excludes 0 and retains the baseline Phase 1A-D sign.
TARGET_THRESHOLD_ROBUST = TRUE iff at least 3 of the 4 thresholds are THRESHOLD_POINT_ROBUST and threshold 0.950 itself is TRUE.
PHASE1B_RESULT = STRONG_PASS (both targets), PARTIAL_PASS (exactly one), NO_PASS (neither).

## Hard cost rule

API calls = 0, LLM calls = 0, predictor retraining = 0, new LightGBM training = 0, new trajectories = 0, new datasets = 0.
