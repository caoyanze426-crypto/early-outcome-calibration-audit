# LATE_REVERSAL_EARLY_EVAL_PHASE0B - PROTOCOL

Strictly offline. SCIENTIFIC_API_CALLS = 0, LLM_CALLS = 0.

## 1. Dataset freeze
tarsur385/swebench-verified-trajectories @ 773748a7c1222e8a642a7059821498e14293562a
- raw trajectories downloaded: 5000
- adapter PASS: 4989
- adapter FAIL: 11
- step rows: 230979
- prefix rows: 235968

## 2. Code freeze
earlyeval @ 7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0, git status clean, clone not modified. All Phase 0B code lives outside the clone.

## 3. Adapter
Phase 0A deterministic mini-SWE-agent adapter (unchanged, imported by path). Field/tool-call normalisation only; no semantic rewriting, no per-trajectory repair.

## 4. Predictor design
EarlyEval-style dual-head LightGBM, repository hyperparameters, reference-free feature configuration (whole gold_/answer family removed).

## 5. Fold protocol
Leave-one-model-out over the 10 model labels. TEST = one held-out model; TRAIN/VALID = the other 9 models. Validation uses the repository default `per_instance_model` strategy (up to 3 validation models per instance, seed 42).
The fold-local FeatureEngineer is fitted on the TRAIN split only (`--fit-feature-engineer-on-train`), so the held-out model contributes to no fitted parameter: not the TF-IDF vocabulary, not the idf, not the SVD basis, not the numeric scaler, not the label encoders, not calibration, not early stopping, not threshold selection.
Pre-existing repository contract exclusion: trajectories with fewer than MIN_TRAJECTORY_STEPS=5 steps are dropped from TRAIN/VALID only (logged before any predictor result is inspected).

## 6. Locked stopping policy
policy_mode=dual, success_thr=0.95, failure_thr=0.95, min_step=0, consecutive=1. Implemented by `earlyeval.policies.safe_stop.apply_policy` / `decide_dual`. No threshold sweep was run.
Head targets use the repository default `--safe-label-min-step 10`.

## 7. Detectors
Deterministic text/tool patterns frozen from Phase 0A before any error-class result was inspected; scan only steps after the decision step.

## 8. Pre-registered gate
Mechanical and pre-registered; thresholds were not altered after seeing results.
