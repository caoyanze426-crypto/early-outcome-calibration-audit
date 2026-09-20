# LATE_REVERSAL_EARLY_EVAL_PHASE0C - PROTOCOL

Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, threshold sweep = 0, TerminalBench = NOT RUN, Toolathlon = NOT RUN.

## Reused frozen inputs

Phase 0C does not retrain the predictor and does not re-run the threshold policy. It reuses, byte-identical:

- the Phase 0B trajectory universe (`trajectory_outcomes.csv`)
- the frozen LOMO held-out policy decisions (`trajectory_policy_decisions.csv`)
- the frozen 0.95 / 0.95 policy decisions and outcome classes
- the frozen deterministic post-stop detectors (`detector_spec.json`)
- the frozen per-model summary and late-reversal signature table

## Questions

Q1. Is the apparent error-direction asymmetry still present after conditioning on the number of SUCCESS vs FAILURE early decisions?

Q2. Are deterministic late-reversal signatures enriched among incorrect early decisions relative to correct early decisions of the same decision class?

## Method

1. Decision-class confusion: conditional error rate within each decision class, kept distinct from error count share.
2. Per-model conditional metrics; denominator-zero strata recorded as NA, never imputed, and class-degenerate held-out models retained as factual records.
3. Signature controls: the exact frozen Phase 0B detectors, comparing FALSE_SUCCESS against CORRECT_SUCCESS and FALSE_FAILURE against CORRECT_FAILURE. No detector was redefined.
4. Stratified control analysis on held-out model x decision-fraction bin ([0,0.25) [0.25,0.50) [0.50,0.75) [0.75,1.00]); empty strata are kept and reported as NA.
5. Standardized control rate by direct reweighting of controls to the error group's stratum distribution. No propensity model.
6. Task-cluster bootstrap by instance_id, 2000 replicates, seed 42; all model trajectories of a selected task enter together and prefix rows are never resampled.
7. Detector re-check: the frozen detector spec is re-applied to the frozen step table and compared against Phase 0B's own flags.

## Pre-registered gate

FS_ENRICHED = FS late-collapse signature risk difference >= +0.20 AND bootstrap 95% CI lower bound > 0.

FF_ENRICHED = FF late-recovery signature risk difference >= +0.20 AND bootstrap 95% CI lower bound > 0.

A. FS_ENRICHED AND FF_ENRICHED -> ROBUST_BIDIRECTIONAL_SIGNAL

B. one direction with risk difference >= +0.30, CI lower bound > 0, errors >= 30 and representation in >= 3 models -> ROBUST_ONE_DIRECTION_SIGNAL

Otherwise -> SIGNAL_NOT_YET_ROBUST. Thresholds were not altered after seeing results.

## Interpretation boundary

Only association / enrichment within this frozen EarlyEval-style setting on this SWE-bench trajectory corpus is supported. No causal claim about late trajectory activity, no general claim that EarlyEval is biased, and no claim that all early evaluators share the effect.
