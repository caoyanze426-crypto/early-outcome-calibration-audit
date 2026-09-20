# EARLYEVAL_PHASE2B - TERMINALBENCH_FIXED_SCAFFOLD_CROSS_MODEL_REPLICATION

## Scientific aim

Cross-benchmark replication of the SWE-bench finding that specific unseen
targets may exhibit substantial early-outcome calibration-transfer error.
TerminalBench primary control: SCAFFOLD = `terminus-2` fixed; only the exact
underlying model varies. Each exact model under `terminus-2` is one TARGET.

## Cost / scope

API calls = 0; LLM calls = 0; new agent generation = 0; cloud compute = 0.
Local predictor training is the only compute performed. No Toolathlon, no
other TerminalBench scaffolds, no threshold sweep, no mitigation design, no
paid services.

## Frozen inputs

- TerminalBench dataset revision `04e8940f5b6736a7ce8d22224fe2f2af74163ed2`
- Phase 2A deterministic adapter `adapter_terminalbench.py` (byte-frozen)
- EarlyEval commit `7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0` (unmodified, clean working tree)

## Pipeline

1. Missingness preflight on ALL `terminus-2` rows (including `steps = null`).
2. Section 5 primary eligibility (usable >= 100, usable successes >= 20,
   usable failures >= 20, usable fraction >= 0.60, MISSINGNESS_GAP <= 0.20).
3. Section 6 minimum-model gate (>= 12 primary eligible models).
4. Section 7/11 prefix table via the frozen adapter + unmodified vendored
   `step_builder` / `prefix_builder`; all usable trajectories retained.
5. Leave-one-model-out folds: TEST = held-out target's usable trajectories;
   TRAIN = other N-1 targets; deterministic train/valid split following the
   Phase 0B lineage (`select_valid_model_pairs_per_instance`, seed 42+3571,
   3 valid models/instance, trainval trajectories with < 5 steps dropped).
6. Fold-local reference-free features: FeatureEngineer parameters are fitted
   on TRAIN only (`name.startswith('gold_')` family removed, tfidf level
   `action_feedback`, TF-IDF -> SVD 64 dims per block, StandardScaler and
   LabelEncoders fit on TRAIN). The held-out model never contributes a
   fitted parameter, a training row or a calibration row.
7. Dual-head LightGBM (`config.LGBM_PARAMS`, CPU fallback) + validation-only
   sigmoid (Platt) calibration, exactly as the frozen Phase 0B lineage.
8. Frozen dual policy: success_thr = 0.95, failure_thr = 0.95, min_step = 0,
   consecutive = 1. No threshold sweep.
9. Section 14 oracle label-prior correction (exact Phase 0E odds-shift form),
   analysis-only: early decisions are never changed.
10. Section 16 task-cluster bootstrap (2000 replicates, cluster = task_name,
    seed = 43000 + model_index*10 + head_index, predictor never retrained).
11. Section 17/18 pre-registered robust-gap and cross-benchmark gates.

## Frozen thresholds (not relaxed after inspecting results)

- primary eligibility: 100 / 20 / 20 / 0.60 / 0.20
- minimum model count: 12
- target-specific signal: decisions >= 20, non-degenerate target prior
- robust large gap: |corrected gap| >= 0.08 with a bootstrap 95% CI excluding
  0 and retaining the point-estimate sign
- gate: >= 3 models robust, >= 2 of them with |gap| >= 0.10, not all from a
  single provider family

## Deviations

Recorded in `PHASE2B_REPORT.md` section J and `integrity_report.json`.
