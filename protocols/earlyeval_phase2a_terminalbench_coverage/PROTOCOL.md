# EARLYEVAL_PHASE2A - PROTOCOL

TERMINALBENCH_COVERAGE_AND_PROVENANCE_AUDIT

## 1. Hard cost rule

API calls = 0. LLM calls = 0. Cloud cost = 0. New agent trajectories = 0.
Predictor training = 0. LightGBM training = 0.

Only public dataset metadata retrieval, a bounded public trajectory download,
deterministic local Python, schema parsing, coverage analysis, provenance
comparison, local hashing, and a bounded deterministic adapter were used.

## 2. Dataset identity

* repo: `yoonholee/terminalbench-trajectories`
* resolved revision: `04e8940f5b6736a7ce8d22224fe2f2af74163ed2`
* license: `apache-2.0`
* shards: `data/train-00000-of-00002.parquet`, `data/train-00001-of-00002.parquet`
* downloaded once, hashed locally, never modified.

## 3. Row classification (Section 2)

Every row is classified mechanically. `TRAJECTORY_USABLE` requires all four
Section-2 conditions: final outcome present, reward interpretable as binary,
step sequence present and non-empty, and trajectory schema parsing
deterministically.

Rejection reasons are recorded independently as condition flags
(`condition_flags`) and as one exclusive primary label using this fixed
priority: `MISSING_REWARD`, `NON_BINARY_REWARD`, `SCHEMA_FAIL`, `NO_STEPS`,
`OTHER`. No manual repair is performed and no row is dropped silently.

Observed step states: `{"JSON_NULL": 17642, "OK": 34462}`. A literal JSON
`null` in the `steps` column means "no trajectory recorded" and is classified
`NO_STEPS`, not `SCHEMA_FAIL`.

## 4. Outcome label (Section 3)

`reward` semantics are taken from the dataset README Schema table verbatim:
"1 if the agent solved the task, 0 otherwise". The audit additionally confirms
the raw value set is exactly `{0, 1}` with no missing and no non-binary values.

## 5. Target unit and eligibility (Sections 4/5)

Primary target = exact underlying `model` x exact scaffold `agent`. Model
families are not collapsed. Eligibility is frozen before outcomes are
inspected: usable trajectories >= 100, successes >= 20, failures >= 20,
unique tasks >= 30. All combos are reported, including ineligible ones, and
the thresholds are not relaxed after seeing counts.

## 6. Coverage (Section 6)

Usable fraction is computed per combo. `LOW_TRAJECTORY_COVERAGE` is a flag only;
no combo is excluded solely for a low fraction.

## 7. Connectivity (Sections 7/8/9)

`MODEL_CROSSED_ELIGIBLE` = exact model occurs in >= 3 eligible scaffolds.
`SCAFFOLD_CROSSED_ELIGIBLE` = exact scaffold occurs with >= 3 eligible models.
`PAIR_COMMON_TASK_ELIGIBLE` = >= 30 task ids shared by both combos with usable
trajectories. Strictly descriptive counts for >= 40 / >= 50 / >= 70 are also
reported. `common_usable_trials` = sum over shared tasks of min(usable trials in
each combo), i.e. the number of usable trials that can be matched pairwise on
common tasks.

## 8. Trial multiplicity (Section 10)

Trials per `task x model x scaffold` cell are audited. Repeated trials are not
treated as independent tasks. Frozen recommendation: the future bootstrap /
inference unit is the `task_name` cluster, retaining all repeated trials for a
selected task. This is design metadata only.

## 9. Adapter (Sections 11/12)

The public dataset is mapped against the frozen EarlyEval commit `7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0`. The
benchmark ingestion contract is matched exactly; the feature/prefix layer needs
a deterministic field remap, so the mapping is classified `SIMPLE_ADAPTER`.
The adapter is exercised on a deterministic bounded sample through the frozen
upstream functions (`normalize_record`, `rebuild_steps_for_trajectory`,
`build_prefix_samples_for_trajectory`) with the vendor runtime root redirected
into this phase's work directory. The upstream clone is never written to and no
predictor is trained.

## 10. Provenance (Section 13)

Only verifiable identifiers are compared. Trial-level identity cannot be
established from the EarlyEval release, so no exact overlap percentage is
inferred and the community dataset is not described as independent.

## 11. Gate (Section 15)

`FEASIBLE` only if all of A-G hold; `PARTIALLY_FEASIBLE` if A-C hold but the
crossed design is weaker; `BLOCKED` if the core outcome/trajectory mapping
fails.

## 12. Reproduction

```bash
python work/lre_phase2a/fetch_dataset.py meta
python work/lre_phase2a/fetch_dataset.py download --revision 04e8940f5b6736a7ce8d22224fe2f2af74163ed2
python work/lre_phase2a/scan_coverage.py
python work/lre_phase2a/build_analysis.py
python work/lre_phase2a/adapter_probe.py
python work/lre_phase2a/write_docs.py
python work/lre_phase2a/build_artifacts.py
```
