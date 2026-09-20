# Paper 2 - reliability audit of trajectory-based early outcome prediction (release package)

This repository accompanies the manuscript *Testing-Driven Reliability Audit
of Trajectory-Based Early Outcome Prediction for LLM Agents: Target-Specific
Calibration Transfer Persists Within a Single Benchmark* (submission target:
Expert Systems with Applications).

The study re-analyzes two third-party public trajectory corpora and reports a
frozen reliability audit: a leave-one-agent-out calibration audit, a
shared-predictor leave-two-agents-out control, oracle label-prior correction,
a target-persistence validation with a threshold-robustness sweep, and a
fixed-scaffold TerminalBench boundary test. The supported claim is explicitly
single-benchmark, and the cross-benchmark boundary result is negative.

Repository status: the release package is published; the public URL is
`https://github.com/caoyanze426-crypto/early-outcome-calibration-audit`.

## Layout

| path | content | license |
|---|---|---|
| `code/` | author-owned Paper 2 analysis code, one directory per frozen phase | MIT (`LICENSE`) |
| `derived_data/data/` | aggregate frozen results and plot/table source data of the analysis phases | CC BY 4.0 (`LICENSE-DATA`) |
| `derived_data/metadata/` | claim-to-evidence matrix and figure/table source maps that trace every rendered value | CC BY 4.0 |
| `derived_data/audits/` | figure render audits of the main and supplement figures (0 mismatches) | CC BY 4.0 |
| `derived_data/README.md`, `PROVENANCE.md`, `REPRODUCIBILITY.md` | inclusion policy, per-file provenance and SHA-256, environment notes | CC BY 4.0 |
| `figures/`, `tables/` | the four main figures (PDF + SVG + 300-dpi PNG) and the three main tables (markdown + CSV) | CC BY 4.0 |
| `protocols/` | the frozen per-phase scientific protocols | CC BY 4.0 |
| `reproducibility/` | environment, dataset revisions, run instructions, frozen lineage | CC BY 4.0 |

## What this release deliberately does not contain

- The two third-party trajectory corpora (SWE-bench Verified agent
  trajectories; community-compiled TerminalBench 2.0 trajectories). They were
  consumed at the frozen revisions recorded in
  `reproducibility/DATASET_REVISIONS.md` and must be retrieved from their
  original sources.
- The upstream EarlyEval code (MIT, `inphotoo/earlyeval`). The study imports
  its stopping-policy module unmodified at a frozen commit; clone it from the
  upstream repository (see `THIRD_PARTY_NOTICES.md`).
- Trajectory-level prediction tables and other row-level intermediates derived
  from the third-party corpora.
- No credentials or API keys exist in this release: the frozen phases were
  offline, and their reports record API calls = 0 and LLM calls = 0 for the
  study's analyses.

## Datasets

| corpus | repo id | revision | license |
|---|---|---|---|
| SWE-bench Verified mini-SWE-agent trajectories (primary) | `tarsur385/swebench-verified-trajectories` | `773748a7c1222e8a642a7059821498e14293562a` | MIT |
| TerminalBench 2.0 community trajectories (boundary test) | `yoonholee/terminalbench-trajectories` | `04e8940f5b6736a7ce8d22224fe2f2af74163ed2` | Apache-2.0 |

## Verifying this copy

    sha256sum -c CHECKSUMS.sha256

`CHECKSUMS.sha256` lists the SHA-256 of every file in this release. The frozen
per-phase namespaces behind the analysis carry their own manifests and
checksums; their checksum-file hashes are recorded in
`reproducibility/FROZEN_LINEAGE.md`.

## Reproducing the analysis

See `reproducibility/ENVIRONMENT.md` for the recorded runtimes,
`reproducibility/DATASET_REVISIONS.md` for the corpus revisions,
`reproducibility/RUN_INSTRUCTIONS.md` for how the released code addresses the
workspace and what each phase contains, and
`reproducibility/FROZEN_LINEAGE.md` for the frozen namespace inventory.

## Licensing summary

- Code: MIT - see `LICENSE`.
- Author-produced derived artifacts and documentation: CC BY 4.0 - see
  `LICENSE-DATA`.
- Third-party datasets and software: their own licenses; not redistributed -
  see `THIRD_PARTY_NOTICES.md`.

## Citation

    Cao, YanZe (2026). Testing-Driven Reliability Audit of Trajectory-Based
    Early Outcome Prediction for LLM Agents: Target-Specific Calibration
    Transfer Persists Within a Single Benchmark. Manuscript under submission;
    derived artifacts deposited at
    https://github.com/caoyanze426-crypto/early-outcome-calibration-audit.
