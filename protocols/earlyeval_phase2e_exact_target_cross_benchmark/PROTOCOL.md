# EARLYEVAL_PHASE2E - EXACT_TARGET_CROSS_BENCHMARK_AUDIT

Pre-registered protocol, written before any Phase 2E number was computed.

Reuses only frozen artifacts: SWE Phase 1A / 1A-D / 1B, TerminalBench Phase
2B / 2B-D. No predictor is retrained, no fold is recalculated, no threshold
value is changed, no new scaffold, no Toolathlon, no new trajectories, no
API or LLM call.

    PHASE2B_RESULT = NO_CROSS_BENCHMARK_SIGNAL   (frozen, unchanged)

This phase cannot overturn the Phase 2B gate. It is descriptive.

## 1. Frozen SWE targets

    TARGET_1: model = gpt-5-mini,        head = SUCCESS
    TARGET_2: model = claude-opus-4.6,   head = FAILURE

Both are the frozen Phase 1B scope labels.

## 2. Model identity qualification

Evidence is taken from direct dataset/model metadata only, never from name
similarity:

    SWE side  : the frozen dataset snapshot
                swebench_verified_raw/<model_label>/<instance>/<instance>.traj.json,
                field info.config.model.model_name (the literal model the
                harness was configured with) and info.docent.model_label.
    TB side   : the frozen dataset's own `model` column, format
                <model-slug>@<provider> as documented in the dataset card.

    IDENTITY = CONFIRMED   iff the SWE canonical model_name provider/slug equal
                           the TB provider/slug for the exact frozen label.
    IDENTITY = AMBIGUOUS   iff more than one canonical model_name is observed
                           for one SWE label, or the provider cannot be pinned.
    IDENTITY = MISMATCH    iff the canonical model_name contradicts the TB label.

A target whose identity is AMBIGUOUS or MISMATCH is excluded from the primary
comparison and reported as excluded.

## 3. Input freeze

Every frozen manifest listed below is re-verified before any comparison:

    outputs/earlyeval_phase1a_same_predictor_transfer/artifact_sha256sums.txt
    outputs/earlyeval_phase1a_d_target_persistence/artifact_sha256sums.txt
    outputs/earlyeval_phase1b_threshold_robustness/artifact_sha256sums.txt
    outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/artifact_sha256sums.txt
    outputs/earlyeval_phase2b_d_threshold_diagnostic/artifact_sha256sums.txt

SWE metrics and bootstrap results are read from the frozen Phase 1B artifacts
(threshold_occurrences.csv, target_threshold_summary.csv,
bootstrap_results.json) and Phase 1A / 1A-D where needed.

TerminalBench metrics and bootstrap results are read from the frozen Phase 2B
(per_target_metrics.csv, target_bootstrap.json) and Phase 2B-D
(threshold_target_metrics.csv, bootstrap_results.json) artifacts.

No metric is recomputed except where needed to verify a frozen value. The
single derived quantity in this phase is the SWE pooled precision and pooled
corrected mean score, computed by re-aggregating the frozen Phase 1B
per-occurrence rows with weights equal to their frozen n_decisions; this is a
re-weighting of frozen numbers, not a new model run.

## 4. Primary threshold

Primary comparison threshold = 0.950, the original frozen operating point of
both benchmarks.

SWE side, per target/head:

    headline signed corrected gap = frozen Phase 1B median_signed_gap
                                    (median across eligible frozen occurrences)
    bootstrap CI                  = frozen Phase 1B median_signed_gap CI
    decisions                     = sum of n_decisions over eligible frozen
                                    occurrences, plus the per-occurrence range
    precision                     = pooled (sum correct / sum decisions) and
                                    median across occurrences
    corrected mean score          = pooled and median across occurrences

TerminalBench side, per target/head, the frozen Phase 2B values:
decisions, empirical_precision, mean_corrected_score, corrected_gap and the
frozen task-cluster bootstrap CI.

## 5. Secondary frozen thresholds

0.900, 0.925, 0.950 are reported descriptively for the same model/head, from
the frozen Phase 1B (SWE) and frozen Phase 2B-D (TerminalBench) artifacts.
No threshold is selected as best, and no new significance gate is created from
whichever threshold looks strongest.

## 6. TerminalBench status classification

Evaluated at the primary threshold 0.950, with the SWE sign taken from the
frozen SWE headline gap:

    PERSISTENT_CROSS_BENCHMARK  iff TB decisions >= 20
                                    AND |TB corrected gap| >= 0.08
                                    AND TB bootstrap CI excludes 0
                                    AND TB gap sign matches SWE.
    COLLAPSED_ON_TERMINALBENCH  iff TB decisions >= 20
                                    AND |TB corrected gap| < 0.04
                                    AND TB bootstrap CI includes 0.
    SIGN_CHANGED                iff TB decisions >= 20
                                    AND the robust corrected gap sign is
                                    opposite the SWE sign.
    INDETERMINATE               iff TB decisions < 20
                                    or identity not confirmed
                                    or bootstrap inference unavailable.

Pre-registered handling of the residual case: the four clauses above do not
exhaust the space (for example 0.04 <= |gap| < 0.08 with a CI spanning 0, and
neither signs disagreeing nor collapsing). If a qualified target matches none
of the four, its status is recorded literally as UNCLASSIFIED_BY_SPEC together
with the four component clause flags, it is treated as not persistent, and the
occasion is reported as an explicit deviation. No label is invented in place
of the four allowed names.

## 7. Overall descriptive result

    BOTH_PERSIST                          iff both qualified targets are
                                              PERSISTENT_CROSS_BENCHMARK.
    ONE_PERSISTS                          iff exactly one is PERSISTENT.
    BOTH_COLLAPSE                         iff both are
                                              COLLAPSED_ON_TERMINALBENCH.
    INSUFFICIENT_CROSS_BENCHMARK_SUPPORT  otherwise.

Descriptive only. It does not replace PHASE2B_RESULT = NO_CROSS_BENCHMARK_SIGNAL.

## 8. Interpretation boundary

If an exact SWE target does not persist on TerminalBench, the record states
only that the target-specific calibration phenotype did not transfer unchanged
across the two frozen benchmark settings. No claim is made that the benchmark
causes the difference.

## 9. Artifacts

    PROTOCOL.md, manifest.json, input_hashes.json
    analysis/model_identity.json
    analysis/target_cross_benchmark.csv
    analysis/threshold_descriptives.csv
    analysis/target_status.json
    PHASE2E_REPORT.md
    integrity_report.json
    artifact_sha256sums.txt

## 10. Stopping rule

Stop after returning the audit. No new scientific call.
