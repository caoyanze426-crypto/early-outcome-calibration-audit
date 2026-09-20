# RUN INSTRUCTIONS

## 1. What the released code is

`code/` contains the analysis code that produced the frozen artifacts of the
manuscript, one directory per frozen phase (`lre_phase0a`, `lre_phase0b`,
`lre_phase0c`, `lre_phase0d`, `lre_phase0e`, `lre_phase1a`, `lre_phase1a_d`,
`lre_phase1b`, `lre_phase2a`, `lre_phase2b`, `lre_phase2b_d`, `lre_phase2e`).
Each directory holds the phase's run scripts, its shared helper module and its
verification scripts, as executed in the frozen workspace. Per-phase scratch
helpers and the vendored upstream EarlyEval clone are not included.

The authoritative definition of what each phase does is its frozen protocol in
`protocols/<namespace>/`; the frozen report of each phase records the numbers
the phase produced.

## 2. Environment

See `ENVIRONMENT.md`. The phases are offline: their frozen reports record API
calls = 0, LLM calls = 0 for the study's analyses. The upstream EarlyEval
release is imported by path; clone it at the frozen commit
`7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0` (MIT) and make it importable, and
provide the two corpora of `DATASET_REVISIONS.md` at the recorded revisions.

## 3. Workspace layout expected by the code

The code was executed inside a project workspace and addresses it through the
environment variable `PAPER2_ROOT`, e.g.:

    set PAPER2_ROOT=<your reconstruction of the frozen workspace>

The released code expects the frozen workspace layout:

    $PAPER2_ROOT/work/<phase dir>/...            phase code and intermediates
    $PAPER2_ROOT/outputs/<namespace>/analysis/   frozen result files
    $PAPER2_ROOT/outputs/<namespace>/reports     frozen phase reports

The aggregate result files shipped here map onto that layout as
`derived_data/data/<file>` and `derived_data/metadata/<file>`, with the exact
per-file source mapping listed in `derived_data/PROVENANCE.md`.

## 4. Phase contents and execution order

Phase 0B is numbered and therefore ordered: `step1_download.py`,
`step2_adapter_and_prefixes.py`, `step3_encode.py`, `step3_features.py`,
`step3a_fit_vocab.py`, `step3b_project_blocks.py`, `step3c_assemble.py`,
`step4_fold_features.py`, `step5_train_fold.py`,
`step6_policy_and_analysis.py`, `step7_analysis.py`, `step8_reports.py`, with
`verify_tfidf.py`, `verify_fold_features.py` and the `run_all.py` driver.

The later phases follow the same pattern - per-phase preparation, execution,
statistics/analysis and verification scripts - and their protocols name the
steps. In outline:

    0A  feasibility:        smoke_forensic.py, mapping_adapter.py,
                            schema_audit.py, observability_audit.py,
                            hf_probe.py, hf_sample.py, finalize_lre_phase0a.py
    0B  predictor build:    the numbered steps above
    0C  decomposition:      p0c_analyze.py -> p0c_report.py
    0D  cross-agent audit:  p0d_analyze.py -> p0d_report.py
    0E  prior correction:   p0e_analyze.py -> p0e_report.py
    1A  pair folds:         p1a_inputs.py, p1a_pair_fold.py, p1a_driver.py,
                            p1a_analyze.py, p1a_manifest.py (+ parity checks)
    1A-D persistence:       p1d_prep.py, p1d_run.py, p1d_stats.py, verify.py
    1B  thresholds:         b1_policy.py, b1_records.py, b1_run.py, verify.py
    2A  coverage:           fetch_dataset.py, scan_coverage.py,
                            build_analysis.py, build_artifacts.py,
                            write_docs.py
    2B  fixed scaffold:     b2_preflight.py, b2_prefix_build.py,
                            step3_encode.py, step4_fold_features.py,
                            step5_train_fold.py, b2_fold_driver.py,
                            b2_analysis.py, b2_artifacts.py, b2_verify.py
    2B-D diagnostics:       d1_policy_grid.py, d2_analysis.py,
                            d3_artifacts.py, d4_verify.py
    2E  identity audit:     e1_identity.py, e2_targets.py, e3_artifacts.py,
                            e4_verify.py

## 5. Scope of reproduction

The frozen phases are the record behind every number in the manuscript: each
namespace carries `manifest.json`, `integrity_report.json` and
`artifact_sha256sums.txt` (see `FROZEN_LINEAGE.md`), and each phase's
verification scripts re-derive its recorded checks. Running the phases again
from the public corpora reproduces the frozen result files; the released
`code/` is that code, with only two mechanical edits - the workspace root is
read from `PAPER2_ROOT` instead of a local absolute path, and the frozen local
interpreter path is replaced by the running interpreter - and with line
endings normalized to LF.
