# Table 3. TerminalBench boundary results (source: `data/table3_terminal_boundary.csv`)

## Section A. Fixed-scaffold (terminus-2) replication summary

| Field | Description (frozen) | Value | Frozen record |
|---|---|---|---|
| `fixed_scaffold_models` | terminus-2 eligible models | 29 | Phase 2B report; eligible-model list |
| `lomo_folds` | leave-one-model-out folds completed | 29/29 | Phase 2B report |
| `usable_trajectories` | trajectories in the frozen Phase 2B analysis | 15889 | Phase 2B report |
| `decided_trajectories_at_0.950` | decided rows of 15889 | 3784 | Phase 2B-D report |
| `decision_coverage_at_0.950` | 3784 / 15889 | 0.2382 | Phase 2B-D report |
| `success_head_eligible_targets_at_0.950` | median \|gap\| = 0.0154, max = 0.0154 | 1 | Phase 2B target-signal record |
| `failure_head_eligible_targets` | median \|gap\| = 0.0102, max = 0.1062, range = 0.1831 | 27 | Phase 2B target-signal record |
| `robust_large_gap_rows` | gemini-3-pro-preview@gemini failure gap -0.1062 CI [-0.1144, -0.0986] | 1 | Phase 2B robust-target table |
| `phase2b_result` | TERMINAL_TARGET_SPECIFIC_SIGNAL = False | NO_CROSS_BENCHMARK_SIGNAL | Phase 2B gate record |
| `phase2bd_result` | THRESHOLD_SIGNAL 1 of 3 (0.900 True) | NO_CROSS_BENCHMARK_REPLICATION | Phase 2B-D diagnostic gate record |

## Section B. Exact-target boundary at 0.950

| Target | Benchmark | Model label (identity) | Decisions | Gap at 0.950 | Status |
|---|---|---|---|---|---|
| TARGET_1 | SWE-bench | gpt-5-mini | 1860 | 0.1377462284504496 | PERSISTENT_ON_SWE_BENCH |
| TARGET_1 | TerminalBench | gpt-5-mini@openai | 0 | NA | INDETERMINATE |
| TARGET_2 | SWE-bench | claude-opus-4.6 | 289 | 0.1107374012181559 | PERSISTENT_ON_SWE_BENCH |
| TARGET_2 | TerminalBench | claude-opus-4-6@anthropic | 45 | 0.0619669131120315 | NOT_PERSISTENT;UNCLASSIFIED_BY_SPEC_AT_0.950 |

Footnotes. (1) Section B gaps are oracle-prior-corrected signed gaps at the frozen 0.950 threshold. (2) gpt-5-mini / SUCCESS produced 0 TerminalBench decisions; its gap is NA (absence of measurement), never 0, and its status is INDETERMINATE rather than any form of collapse. (3) claude-opus-4.6 / FAILURE produced 45 TerminalBench decisions; its interval includes zero, so it is recorded as not persistent under the frozen criterion, with the 0.950 point lying in an intermediate region not exhausted by the preregistered classification categories. (4) Mandatory statement: failure to replicate does not establish that the benchmark/environment causes the discrepancy.

**Caption (for typesetting).** Table 3. TerminalBench boundary results. Section A summarizes the fixed-scaffold replication: 29 eligible terminus-2 models, 29/29 leave-one-model-out folds completed, 15889 trajectories in the frozen analysis universe, 3784 decided at threshold 0.950 (coverage 0.2382), 1 eligible SUCCESS-head target and 27 eligible FAILURE-head targets, and a single robust large-gap row (gemini-3-pro-preview@gemini, FAILURE head, gap -0.1062, CI [-0.1144, -0.0986]) against a 20-decision eligibility rule; Phase 2B reports NO_CROSS_BENCHMARK_SIGNAL and Phase 2B-D reports NO_CROSS_BENCHMARK_REPLICATION. Section B gives the exact-identity comparison at 0.950. gpt-5-mini/SUCCESS is persistent on SWE-bench Verified but produced 0 TerminalBench decisions, so its TerminalBench status is INDETERMINATE and its gap is reported as NA, not as zero. claude-opus-4.6/FAILURE is persistent on SWE-bench Verified with a 0.950 gap of +0.110737...; on TerminalBench it produced 45 decisions over 14 tasks with a gap of +0.061967 and a bootstrap interval including zero, and is recorded as not persistent under the frozen criterion, with the 0.950 point lying in an intermediate region not exhausted by the preregistered classification categories. Unit of analysis: held-out model within the fixed scaffold, and identity-matched target model for section B. All TerminalBench gaps are oracle-prior-corrected. The table supports the statement that the SWE target-specific pattern is not established as benchmark-general under this scaffold and criterion. Failure to replicate does not establish that the benchmark or the environment causes the discrepancy, and the table does not show that the SWE effect disappears.
