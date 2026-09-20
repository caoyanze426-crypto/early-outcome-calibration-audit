# EARLYEVAL_PHASE1A - SAME_PREDICTOR_MULTI_AGENT_TRANSFER

PREDICTOR = P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE
POLICY = current_safe_stop (dual head, success_thr=0.95, failure_thr=0.95, min_step=0, consecutive=1)
PAIR FOLDS = 45 (all unordered label pairs of the frozen 10-label universe)
TRAINING UNIVERSE = 8 remaining labels; VALID = deterministic Phase 0B split of those 8 labels; TEST = both held-out labels.
PRIMARY SUPPORT = COMMON_TASK_SUPPORT_AB (instance_id with an adapter-PASS trajectory for both held-out labels).

## Frozen constants

- PAIR_MIN_DECISIONS = 20
- PAIR_MIN_ELIGIBLE_FOLDS = 20
- PAIR_MEDIAN_DIFF_MIN = 0.05
- PAIR_CI_LOWER_MIN = 0.03
- PERSIST_MIN_OCCURRENCES = 7
- PERSIST_MEDIAN_ABS_GAP_MIN = 0.08
- PERSIST_SAME_SIGN_FRACTION = 7/9
- BOOTSTRAP_REPLICATES = 2000, seeded per declared scope

## Oracle prior correction

p_corrected = sigmoid(logit(p_raw) + sign * ln r) with
ln r = log_odds(pi_target) - log_odds(pi_train), sign = +1 for the success head and -1 for the failure head (complementary prior).
Predictor outputs and original decisions are never modified; the correction is analysis-only. pi_train and pi_target are measured on COMMON_TASK_SUPPORT_AB. A prior of exactly 0 or 1 is flagged PRIOR_DEGENERATE and preserved without smoothing.

## Gates (frozen before any result was inspected)

HEAD_SAME_PREDICTOR_HETEROGENEITY = TRUE iff eligible pair folds >= 20 AND median pairwise residual gap difference >= 0.05 AND task-cluster bootstrap 95% CI lower bound > 0.03.
SAME_PREDICTOR_HETEROGENEITY = TRUE iff either head passes.
TARGET_HEAD_PERSISTENT_LARGE_GAP = TRUE iff eligible occurrences >= 7 AND median absolute corrected gap >= 0.08 AND same-sign fraction >= 7/9 (or >= 80% when exactly 7 or 8 occurrences are eligible).
TARGET_SPECIFIC_PERSISTENCE = TRUE iff >= 2 distinct agent labels are persistent in >= 1 head.
PHASE1A_RESULT = STRONG_PASS (both), PARTIAL_PASS (exactly one), NO_PASS (neither).

## Offline guarantees

API calls = 0, LLM calls = 0, new trajectories = 0, threshold sweep = 0, method modification = 0. Thresholds, hyperparameters, the feature lineage and the calibration procedure are reused from the Phase 0B freeze; the only intended design change is 9 train + 1 held-out -> 8 train + 2 held-out.

## Deviations

- pi_train and pi_target are measured on COMMON_TASK_SUPPORT_AB (the primary support required by section 9); full-universe priors are not used for the gate.
- The task-cluster bootstrap draws independently seeded replicates per declared scope and head (seed = 42*100 + scope offset + head) so that every reported CI is reproducible from bootstrap_seed alone.
- GEMINI-3-PRO has target success prior 1.0 on some folds, so PRIOR_DEGENERATE = TRUE is flagged and the boundary behaviour is preserved exactly (no smoothing); those occurrences are excluded from target persistence exactly as section 17 prescribes.
- The mirror-label sensitivity removes gpt-5.2-high from the pair scope without retraining, because section 5 forbids retuning and section 22 asks only to recalculate the principal gate metrics on a declared secondary scope.
- Secondary correlation (section 20) uses the corrected calibration gap, matching the Phase 0E convention; the raw decision error rate is kept as the separate *_all_task_* descriptive columns.
- artifact_sha256sums.txt excludes itself; manifest.json excludes itself and artifact_sha256sums.txt.
