# THIRD_PARTY_NOTICES

This repository contains only author-produced material: the Paper 2 analysis
code (`code/`, MIT), the author's derived aggregate results (`derived_data/`,
CC BY 4.0), the submission artwork and tables, the frozen phase protocols,
and the reproduction documentation. No third-party data and no third-party
code is redistributed here.

## Third-party datasets used by the study (not redistributed)

| dataset | revision | license (as recorded in the frozen metadata) | role |
|---|---|---|---|
| `tarsur385/swebench-verified-trajectories` | `773748a7c1222e8a642a7059821498e14293562a` | MIT (dataset metadata tag `license:mit`, recorded in the Phase 0A public-dataset metadata) | primary corpus: SWE-bench Verified mini-SWE-agent trajectories, 10 model labels |
| `yoonholee/terminalbench-trajectories` | `04e8940f5b6736a7ce8d22224fe2f2af74163ed2` | Apache-2.0 (recorded in the Phase 2A dataset metadata) | second-benchmark boundary corpus: community-compiled TerminalBench 2.0 trajectories |

Both corpora must be obtained from their original sources at the revisions
above. This repository represents them only through aggregate derived
results; it ships no trajectory text, task text, prompt or completion.

## Third-party software (not redistributed)

| software | source | commit | license | use |
|---|---|---|---|---|
| EarlyEval | https://github.com/inphotoo/earlyeval | `7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0` | MIT | public early-outcome-prediction release; its stopping-policy module is used unmodified at the frozen commit (clone not modified; recorded license-file SHA-256 `0343cd8e47ffa9bf2ed77fe835188b3e36073a72e5a4dbe4e5f9329670082387`). The upstream release ships no trained predictor, and no upstream code is included in this repository. |

Runtime libraries used for the frozen analysis and for figure rendering
(Python 3.14.3 with its bundled numpy/pandas, LightGBM, scikit-learn and
matplotlib 3.11.2) are standard open-source packages obtained from their own
distributions and carry their own licenses; they are listed as environment
information only (see `reproducibility/ENVIRONMENT.md`).

## Attribution

If you use the derived artifacts of this repository, please cite the
manuscript and this deposit (see `README.md`). The upstream EarlyEval release
must be credited to its own authors; nothing in this repository presents
upstream code as author-created.
