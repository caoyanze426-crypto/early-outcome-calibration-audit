# EARLYEVAL_PHASE2B_D - TERMINALBENCH_THRESHOLD_DENOMINATOR_DIAGNOSTIC

Pre-registered protocol, written before any metric was computed.

This phase does not modify the Phase 2B primary result.

    PHASE2B_RESULT = NO_CROSS_BENCHMARK_SIGNAL   (unchanged)

## 0. Cost and scope

API calls = 0. LLM calls = 0. Predictor retraining = 0.
New LightGBM training = 0. New trajectories = 0. Cloud compute = 0.

Everything below is a deterministic re-scan of the frozen Phase 2B held-out
prefix predictions. No fold is retrained. No feature is rebuilt. No calibrator
is refit. The only thing that changes across the grid is the two decision
thresholds handed to the frozen vendor policy function.

## 1. Frozen inputs

Reused verbatim from Phase 2B (all hashes re-verified against the frozen
Phase 2B `artifact_sha256sums.txt` before any computation):

    predictions/heldout_prefix_predictions/fold-NN.parquet   (29 files)
    predictions/policy_decisions/fold-NN.csv                 (29 files, 0.950)
    analysis/per_target_metrics.csv                          (0.950 reference)
    analysis/target_bootstrap.json                           (0.950 reference)
    analysis/phase2b_gate.json                               (0.950 reference)
    work/lre_phase2b/preflight.json                          (29-target universe)
    work/lre_phase2b/trajectory_index.csv                    (priors, labels)
    work/lre_phase2b/fold_manifest.csv                       (fold -> model map)

The 29-target universe is frozen. No new model may enter it.

## 2. Threshold grid

Evaluated exactly: 0.900, 0.925, 0.950, with

    success_thr = failure_thr = t
    policy_mode = dual
    score_mode  = calibrated
    min_step    = 0
    consecutive = 1

Applied through the unmodified vendored `earlyeval.policies.safe_stop.apply_policy`.

0.950 is a reproduction anchor: re-derived decisions must match the frozen
Phase 2B `policy_decisions/fold-NN.csv` on every trajectory (same decided flag,
same decision, same decision_step, decision_score within 1e-12). If the anchor
fails, the diagnostic stops and reports failure.

## 3. Target eligibility

Per threshold and head, a calibration target is eligible iff it has

    >= 20 early decisions   AND   non-degenerate target prior

The non-degenerate flag is the frozen Phase 2B per-model prior flag (held-out
success prior and leave-one-out train success prior both strictly inside (0,1));
it does not depend on the threshold.

## 4. Prior correction

Reused verbatim from Phase 0E / Phase 2B: for a stop row with raw head
probability p, head h in {success, failure}, and the model's Phase 0E log-odds
ratio r = logit(test success prior) - logit(LOO train success prior),

    sign = +1 for the success head, -1 for the failure head
    p_corrected = sigmoid(logit(p) + sign * r)

Early decisions themselves are not changed; the correction is applied only to
the stop-point scores when computing the calibration gap. The coefficient is
held at its frozen point value inside the bootstrap; only precision, corrected
mean score and the gap are recomputed.

Per target/head/threshold the diagnostic reports: decisions, precision,
mean stop score (raw), corrected mean score, signed corrected gap,
absolute corrected gap.

    signed corrected gap = corrected mean score - empirical precision

## 5. Robust large gap

    TARGET_HEAD_LARGE_GAP = TRUE  iff
        decisions >= 20
        AND |point corrected gap| >= 0.08
        AND task-cluster bootstrap 95% CI excludes 0
        AND the CI retains the point-estimate sign

Bootstrap: 2000 task_name-cluster replicates, deterministic seeds, fixed rule

    seed = 43000 + fold_index * 10 + head_code     (success = 0, failure = 1)

identical to the Phase 2B rule, so the 0.950 bootstrap reproduces Phase 2B
exactly when the 0.950 decision set reproduces exactly. No retraining happens
inside the bootstrap. The resampling unit is the task cluster; each replicate
draws n_tasks task IDs with replacement and keeps every decision row of the
drawn tasks.

## 6. Denominator adequacy

    HEAD_DENOMINATOR_ADEQUATE = TRUE  iff  eligible targets >= 8

This is a feasibility requirement only, not a scientific signal.

## 7. Diagnostic cross-benchmark gate

Per threshold independently:

    THRESHOLD_SIGNAL = TRUE  iff
        >= 3 distinct exact models have TARGET_HEAD_LARGE_GAP in >= 1 head
        AND >= 2 of those models have |corrected gap| >= 0.10
        AND the large-gap models span >= 2 provider families
        AND the relevant head has HEAD_DENOMINATOR_ADEQUATE = TRUE

"Provider family" is derived from the exact dataset model label by the frozen
Phase 2B `provider_family` mapping.

Interpretation of "the relevant head", fixed here before results:

    PRIMARY  reading: every head that contributes at least one robust large gap
                      must itself have HEAD_DENOMINATOR_ADEQUATE = TRUE.
    VARIANT  reading: at least one contributing head is adequate.

Both are computed and reported. The final gate uses the PRIMARY reading.

## 8. Final diagnostic gate

    PHASE2B_D_RESULT = DENOMINATOR_RESOLVED_SIGNAL
        iff THRESHOLD_SIGNAL = TRUE at >= 2 of the 3 thresholds.
    Otherwise
    PHASE2B_D_RESULT = NO_CROSS_BENCHMARK_REPLICATION

The frozen Phase 2B primary result remains unchanged either way.

## 9-10. Required head reports

Per threshold, for the success head and for the failure head separately:
eligible targets, total decisions, median decisions per eligible target,
min/max decisions per eligible target, and robust target count.

The Phase 2B negative evidence for the failure head is not weakened post hoc;
the failure-head numbers at 0.950 are reproduced, not replaced.

## 11. Artifacts

    PROTOCOL.md, manifest.json, input_hashes.json
    analysis/threshold_target_metrics.csv
    analysis/bootstrap_results.json
    analysis/denominator_summary.csv
    analysis/robust_targets.csv
    analysis/reproduction_check.json
    analysis/diagnostic_gate.json
    PHASE2B_D_REPORT.md
    integrity_report.json
    artifact_sha256sums.txt

## 12. Stopping rule

Stop after the diagnostic. Do not run other scaffolds, Toolathlon, new models,
new trajectories, threshold values outside the grid, mitigation design, or any
new scientific call.
