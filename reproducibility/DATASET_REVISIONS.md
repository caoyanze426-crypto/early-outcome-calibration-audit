# DATASET REVISIONS

Both corpora are third-party public datasets, consumed at exactly these
revisions and not redistributed by this repository.

| corpus | repo id | revision | license | recorded facts |
|---|---|---|---|---|
| SWE-bench Verified mini-SWE-agent trajectories (primary) | `tarsur385/swebench-verified-trajectories` | `773748a7c1222e8a642a7059821498e14293562a` | MIT (dataset metadata tag `license:mit`) | 5002 listed files, ~0.821 GB; 10 model labels x 500 SWE-bench Verified instances |
| TerminalBench 2.0 community trajectories (boundary test) | `yoonholee/terminalbench-trajectories` | `04e8940f5b6736a7ce8d22224fe2f2af74163ed2` | apache-2.0 | 52104 trajectories, 89 tasks, 109 agent/model combos; resolved revision requested as `main` |

Consumed counts recorded by the frozen phases:

    SWE side   5000 raw trajectories downloaded; 4989 adapter PASS, 11 adapter
               FAIL; 500 shared tasks across all 10 model labels
    TB side    52104 public rows;
               34462 rows with usable
               trajectory; 15889 trajectories entering the frozen Phase 2B
               analysis universe after step-builder drops

The SWE-bench Verified corpus revision is also the snapshot recorded in the
Phase 0B dataset manifest and re-verified in Phase 2E (`500/500` file identity
checks). No other snapshot of either corpus was used.
