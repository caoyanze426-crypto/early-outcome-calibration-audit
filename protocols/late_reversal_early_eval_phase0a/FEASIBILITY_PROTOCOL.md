# FEASIBILITY_PROTOCOL - LATE_REVERSAL_EARLY_EVAL_PHASE0A

Local / public-data engineering feasibility audit. Not a scientific experiment.
SCIENTIFIC_API_CALLS = 0 and LLM_CALLS = 0 for the whole phase.

1. Clone https://github.com/inphotoo/earlyeval unmodified; record commit SHA, clone
   timestamp and license.
2. Run the official smoke workflow:
     python -m earlyeval.cli pipeline current-safe-stop --mode smoke
   capture stdout, stderr, exit code, and policy_decisions.csv, policy_summary.csv,
   policy_per_agent.csv, run_metadata.json.
3. Mechanically recover the 9-run smoke forensic: trajectory count, prefix rows, per
   trajectory final label, halt status and halt step, then count FALSE_FAILURE,
   FALSE_SUCCESS, CORRECT_FAILURE, CORRECT_SUCCESS and NO_STOP.
4. Verify which prediction/policy fields actually exist in the produced tables.
5. Inspect the public dataset tarsur385/swebench-verified-trajectories metadata, then
   download a small inspection sample only (>= 20 trajectories, >= 2 model labels, both
   resolved and unresolved) without pulling the full 821 MB corpus.
6. Record the public trajectory schema mechanically (only fields that actually exist).
7. Audit EarlyEval compatibility and implement a minimal proof-of-mapping, then run the
   unmodified vendored step builder over the reserialized sample.
8. Determine whether late-reversal signals are mechanically recoverable after a
   hypothetical early stop at step k.

Not performed: full-corpus download, predictor training, LightGBM runs, new trajectory
generation, any model call, and any LLM-based trajectory semantic labeling.

Boundary: this phase establishes data/pipeline feasibility only. It makes no claim that
late-reversal bias exists, that a false-failure asymmetry exists, that EarlyEval is
biased, or that recovery trajectories are harmed.