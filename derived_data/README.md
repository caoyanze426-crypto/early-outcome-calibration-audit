> **Release note (PAPER2_W7B).** In this repository these derived files
> live under `derived_data/` (the pre-finalization package used the
> directory name `repository_candidate/`). File contents are unchanged
> except for the single corrected literal documented in
> `derived_data/PROVENANCE.md`, and the references to the author-decision
> files of the pre-finalization workspace, which this release replaces
> with the release level `README.md` / `LICENSE` / `LICENSE-DATA`.

# Paper 2 - derived artifacts

This directory is the deposited package of the shareable derived material
behind the manuscript "Testing-Driven Reliability Audit of Trajectory-Based
Early Outcome Prediction for LLM Agents: Target-Specific Calibration Transfer
Persists Within a Single Benchmark". It is deposited in the public repository
at `https://github.com/caoyanze426-crypto/early-outcome-calibration-audit`.

## What is included

    data/       aggregate results of the frozen analysis phases (0D, 0E, 1A,
                1A-D, 1B, 2A, 2B, 2B-D, 2E) and the plot-ready figure/table
                source data used to render every main figure and table
    metadata/   the frozen claim-to-evidence matrix (W1) and the figure/table
                source maps (W2) that trace every rendered value to a file
    PROVENANCE.md       per-file source path and SHA-256 of each copy
    REPRODUCIBILITY.md  environment and step-by-step regeneration notes

## What is deliberately NOT included

* The third-party trajectory corpora themselves (SWE-bench Verified agent
  trajectories - `tarsur385/swebench-verified-trajectories` @
  `773748a7c1222e8a642a7059821498e14293562a` (MIT); community-compiled
  TerminalBench trajectories - `yoonholee/terminalbench-trajectories` @
  `04e8940f5b6736a7ce8d22224fe2f2af74163ed2` (Apache-2.0). They are consumed at
  those revisions and are not redistributed. Retrieve them from their original
  sources.
* Analysis code. The author-owned analysis code is released in `code/`
  under the MIT License (see the repository `README.md` and `LICENSE`).
* Trajectory-level prediction tables and other row-level intermediates
  derived from the third-party corpora; they are not included in this
  release.

## Licensing and ownership

Every file in `data/` and `metadata/` is an author-produced derived table of
aggregate statistics or metadata from this study's frozen analysis. No file
contains trajectory text, task text, prompts, completions or any other
third-party content; no file contains personal data. These derived files are licensed
under CC BY 4.0 (`LICENSE-DATA`); that license is compatible with the
sources' licenses (the TerminalBench source is Apache-2.0 and the SWE-bench
Verified trajectory source is MIT).

## How to verify a copy

    sha256sum -c <(grep -v '^#' PROVENANCE.md)   # or compare against PROVENANCE.md by hand

`PROVENANCE.md` lists the SHA-256 of every file in this package exactly as
copied from the frozen workspace.
