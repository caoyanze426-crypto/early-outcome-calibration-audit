# Table 2. Primary SWE-bench Verified target results (source: `data/table2_primary_targets.csv`)

| Field | gpt-5-mini / SUCCESS | claude-opus-4.6 / FAILURE |
|---|---|---|
| Agent (model label) | gpt-5-mini | claude-opus-4.6 |
| Prediction head | success | failure |
| Eligible same-predictor occurrences | 9 | 8 |
| Partner folds in scope | 9 | 9 |
| Excluded below decision minimum | 0 | 1 |
| Excluded (prior-degenerate) | 0 | 0 |
| Median corrected gap | 0.1377462284504496 | 0.11073740121815578 |
| Same-sign fraction | 1.0 (9/9) | 1.0 (8/8) |
| Bootstrap 95% CI (task-cluster) | [0.11681530181196477, 0.15944975771607114] | [0.05100413519648092, 0.14856491377866127] |
| Jackknife min abs median | 0.13617040153017057 | 0.11039703482593621 |
| Task-half A median | 0.11767970623031776 | 0.08688594138784889 |
| Task-half B median | 0.14894173870245364 | 0.11354561216502507 |
| Threshold robustness | 4/4 evaluable thresholds robust (0.9/0.925/0.95/0.975) | 3/3 evaluable thresholds robust (0.9/0.925/0.95); 0.975 NOT_EVALUABLE (denominator) |
| Final target status | True (ROBUST) | True (ROBUST) |

Footnotes. (1) All gap values are oracle-prior-corrected signed gaps (predicted confidence minus realized outcome rate at the early stop point). (2) Interval-source rule: the persistence columns (bootstrap CI, jackknife, task halves) come from Phase 1A-D; the threshold column comes from Phase 1B, whose bootstrap execution is distinct. Where the two disagree it is in the last printed digits only (for example the Phase 1A-D median 0.11073740121815578 versus the Phase 1B 0.950 median 0.1107374012181559). (3) 0.975 is NOT EVALUABLE for claude-opus-4.6 / FAILURE because no occurrence clears the frozen decision-count minimum at that threshold. (4) Unit of analysis: one eligible partner-fold occurrence of the target cell.

**Caption (for typesetting).** Table 2. Primary SWE-bench Verified target results. Two rows, one per frozen primary target cell. Columns report the number of eligible same-predictor occurrences, the median signed oracle-prior-corrected calibration gap, the same-sign fraction, the task-cluster bootstrap 95% confidence interval, the partner-fold jackknife minimum absolute median, the two task-half medians, threshold robustness across the 0.900/0.925/0.950/0.975 grid, and the final frozen target status. Unit of analysis: one eligible partner-fold occurrence of the target cell (9 for gpt-5-mini/SUCCESS; 8 for claude-opus-4.6/FAILURE, of 9 folds in scope with one occurrence failing the frozen decision-count minimum). All gap values are oracle-prior-corrected. The persistence columns (confidence interval, jackknife, task halves) come from Phase 1A-D; the threshold column comes from Phase 1B, and the two bootstrap executions are kept separate because their intervals are not identical. The table supports the statement that both target cells' errors persist across changes in the source-agent cohort, task resampling, and stopping thresholds within this frozen setting. It does not establish that calibration failure is broad across agents, and it does not generalize beyond SWE-bench Verified.
