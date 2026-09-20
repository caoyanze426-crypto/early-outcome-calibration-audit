# EARLYEVAL_PHASE1A_D - TARGET_SPECIFIC_PERSISTENCE_VALIDATION

Frozen primary targets (section 0):
- TARGET_1 = gpt-5-mini / success
- TARGET_2 = claude-opus-4.6 / failure
- sensitivity only = minimax-m2.5-high / failure (never enters the primary gate)

## Inputs

Phase 1A per_pair_target_predictions, pair_policy_decisions, pairwise_metrics.csv, target_persistence.csv, pair_fold_manifest.csv, plus the Phase 0B trajectory_outcomes.csv. Every hash is verified against the Phase 1A artifact_sha256sums.txt. No prediction is regenerated and no predictor is trained.

## Frozen constants

- pair-fold eligibility = 20 early decisions (Phase 1A section 14, both held-out sides)
- task-half eligibility = 10 early decisions in that half (section 7)
- task-half minimum eligible folds = 5
- bootstrap replicates = 5000, seed = 42031, cluster = instance_id
- gate A: eligible occurrences >= 7
- gate B: baseline median absolute corrected gap >= 0.08
- gate C: baseline same-sign fraction >= 0.8
- gate D: bootstrap 95% CI for the median signed gap excludes 0 and retains the baseline sign
- gate E: every leave-one-partner-out median retains the baseline sign
- gate F: jackknife minimum absolute median >= 0.06
- gates G/H: each task half retains the baseline sign with absolute magnitude >= 0.06 and >= 5 eligible pair folds

TARGET_ROBUST = TRUE iff A and B and C and D and E and F and G and H. PHASE1A_D_RESULT = STRONG_TARGET_SPECIFIC_SIGNAL (both targets robust), SINGLE_TARGET_SIGNAL (exactly one), NO_ROBUST_TARGET_SIGNAL (neither).

## Prior correction (frozen Phase 0E definition)

ln r = log_odds(pi_target) - log_odds(pi_train); corrected score = sigmoid(logit(raw stop score) + sign * ln r) with sign = +1 for the success head and -1 for the failure head. Boundaries (pi exactly 0 or 1) are preserved exactly and never smoothed. Every weighting (baseline, bootstrap replicate, task half) recomputes pi_target and pi_train from the trajectories of the weighted task set.

## Task halves (section 6)

task_half = A if the lowest bit of SHA256(instance_id) is 0 else B. The assignment for the full task universe is written to analysis/task_half_assignment.csv and hashed before any target statistic is produced; halves are never rebalanced or imputed.

## Offline guarantees

predictor training = 0, new LightGBM heads = 0, API calls = 0, LLM calls = 0, new trajectories = 0, new pair folds = 0, threshold sweep = 0, method design = 0.

## Interpretation boundary (section 11)

Results support only that specific unseen agent/head combinations show persistent calibration-transfer error across multiple shared-predictor training cohorts and task subsets. They do not license claims that all agents are miscalibrated, that pairwise heterogeneity is broad, or that agent identity causes calibration failure.
