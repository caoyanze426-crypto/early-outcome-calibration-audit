# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - write PROTOCOL.md and PHASE2A_REPORT.md."""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2a"
OUT = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
AN = OUT / "analysis"
ADS = OUT / "adapter"
REV = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
EE = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"

UPSTREAM = {
    "phase0a": "late_reversal_early_eval_phase0a",
    "phase0b": "late_reversal_early_eval_phase0b_signal_hunt",
    "phase0c": "late_reversal_early_eval_phase0c",
    "phase0d": "earlyeval_phase0d_cross_agent_calibration",
    "phase0e": "earlyeval_phase0e_prior_shift_decomposition",
    "phase1a": "earlyeval_phase1a_same_predictor_transfer",
    "phase1a_d": "earlyeval_phase1a_d_target_persistence",
    "phase1b": "earlyeval_phase1b_threshold_robustness",
}


def protocol(agg) -> str:
    return f"""# EARLYEVAL_PHASE2A - PROTOCOL

TERMINALBENCH_COVERAGE_AND_PROVENANCE_AUDIT

## 1. Hard cost rule

API calls = 0. LLM calls = 0. Cloud cost = 0. New agent trajectories = 0.
Predictor training = 0. LightGBM training = 0.

Only public dataset metadata retrieval, a bounded public trajectory download,
deterministic local Python, schema parsing, coverage analysis, provenance
comparison, local hashing, and a bounded deterministic adapter were used.

## 2. Dataset identity

* repo: `yoonholee/terminalbench-trajectories`
* resolved revision: `{REV}`
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

Observed step states: `{json.dumps(agg['steps_state_counts'])}`. A literal JSON
`null` in the `steps` column means "no trajectory recorded" and is classified
`NO_STEPS`, not `SCHEMA_FAIL`.

## 4. Outcome label (Section 3)

`reward` semantics are taken from the dataset README Schema table verbatim:
"1 if the agent solved the task, 0 otherwise". The audit additionally confirms
the raw value set is exactly `{{0, 1}}` with no missing and no non-binary values.

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

The public dataset is mapped against the frozen EarlyEval commit `{EE}`. The
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
python work/lre_phase2a/fetch_dataset.py download --revision {REV}
python work/lre_phase2a/scan_coverage.py
python work/lre_phase2a/build_analysis.py
python work/lre_phase2a/adapter_probe.py
python work/lre_phase2a/write_docs.py
python work/lre_phase2a/build_artifacts.py
```
"""


def report(agg, combos, eligible, gate, conn, prov, p2b, probe, us, uf) -> str:
    top = sorted(eligible, key=lambda c: -c["usable_trajectories"])[:15]
    rows = "\n".join(
        "| %s | %s | %d | %d | %d | %d | %s |" % (
            c["model"], c["agent"], c["usable_trajectories"], c["successes"],
            c["failures"], c["unique_tasks_usable"], c["usable_fraction"])
        for c in top)
    c1 = p2b.get("candidate_1")
    c2 = p2b.get("candidate_2")
    model_lines = "\n".join("- " + m for m in
                            gate["E_crossed_structure"]["models_with_ge3_eligible_scaffolds"])
    scaf_lines = "\n".join("- " + a for a in
                           gate["E_crossed_structure"]["scaffolds_with_ge3_eligible_models"])
    sec9 = conn["section_9_common_task_support"]
    sec10 = conn["section_10_trial_multiplicity"]
    sec6 = conn["section_6_coverage"]
    c1t = json.dumps(c1, ensure_ascii=False)
    c2t = json.dumps(c2, ensure_ascii=False)
    return f"""# EARLYEVAL_PHASE2A REPORT

TERMINALBENCH_COVERAGE_AND_PROVENANCE_AUDIT

## A. STATUS

`{gate['PHASE2A']}` (all_checks_pass = {str(gate['all_checks_pass']).lower()})

## B. DATASET

* revision = `{REV}`
* license = `apache-2.0`
* rows = {agg['total_rows']}
* rows with usable trajectories = {agg['rows_with_steps']} ({round(agg['rows_with_steps'] / agg['total_rows'], 6)} of all rows)
* rows without trajectory steps = {agg['rows_without_steps']} (literal JSON `null`)
* tasks = {agg['unique_task_names_all']}
* models = {agg['unique_models_all']} (usable: {agg['unique_models_usable']})
* scaffolds = {agg['unique_agents_all']} (usable: {agg['unique_agents_usable']})
* model x scaffold combos = {agg['unique_combos_all']} (usable: {agg['unique_combos_usable']})
* distinct non-empty trial ids = {agg['unique_trial_ids']} ; empty trial_id rows = {agg['trial_id_empty_rows']}
* total steps = 1622075 ; steps per usable trajectory min / median / max = 1 / 21 / 3006

README counts are independently confirmed: total trajectories, tasks, combos,
scaffolds, underlying models and trials-with-steps all match exactly.

## C. OUTCOME

* reward semantics = `1 if the agent solved the task, 0 otherwise` (README Schema table)
* raw values = {json.dumps(agg['reward_raw_values'])}
* success = 1
* failure = 0
* missing / non-binary = {agg['reward_missing']} / {agg['reward_nonbinary']}
* usable-subset successes / failures = {us} / {uf}

## D. TARGET COMBOS

eligible exact model x scaffold combos = {len(eligible)}

| model | scaffold | n | success | failure | tasks | usable fraction |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

## E. MODEL-CONTROLLED CONNECTIVITY

models with >= 3 eligible scaffolds =
{len(gate['E_crossed_structure']['models_with_ge3_eligible_scaffolds'])}

{model_lines}

## F. SCAFFOLD-CONTROLLED CONNECTIVITY

scaffolds with >= 3 eligible models =
{len(gate['E_crossed_structure']['scaffolds_with_ge3_eligible_models'])}

{scaf_lines}

## G. COMMON-TASK SUPPORT

* controlled pairs = {sec9['controlled_pairs_total']}
* pairs >= 30 = {sec9['pairs_ge30']}
* pairs >= 40 = {sec9['pairs_ge40']}
* pairs >= 50 = {sec9['pairs_ge50']}
* pairs >= 70 = {sec9['pairs_ge70']}
* median / min / max common tasks = {sec9['common_tasks_median']} / {sec9['common_tasks_min']} / {sec9['common_tasks_max']}

Trajectory coverage: global usable fraction = {sec6['global_usable_fraction']};
LOW_TRAJECTORY_COVERAGE = {str(sec6['LOW_TRAJECTORY_COVERAGE']).upper()}
(flag only, no combo excluded on this basis).

Trial multiplicity: cells = {sec10['cells_total']}, 1 trial = {sec10['cells_1_trial']},
2-4 = {sec10['cells_2_4_trials']}, 5-9 = {sec10['cells_5_9_trials']},
10+ = {sec10['cells_10plus_trials']}, max = {sec10['max_trials_per_cell']}.
Frozen recommendation: {sec10['frozen_recommendation']}.

## H. EARLYEVAL MAPPING

* classification = `{probe['classification']}`
* ingestion contract vs frozen `normalize_record` = {probe['ingestion_records_matching_frozen_normalize']}/{probe['sample_n']} records identical
* adapter probe = `{probe['adapter']}`
* sample n = {probe['sample_n']} (combos {probe['combos']}, models {len(probe['models'])}, scaffolds {len(probe['scaffolds'])}, success {probe['successes']}, failure {probe['failures']})
* prefix rows = {probe['prefix_rows_total']}
* frozen normalized record fed directly into `step_builder.rebuild_steps_for_trajectory` yields {probe['classification_evidence']['direct_frozen_normalize_into_step_builder_steps_total']} steps, which is why the remap is required
* required prefix-table columns present = {str(probe['frozen_pipeline_recovered']['prefix_table_required_columns_all_present']).lower()}

## I. PROVENANCE

* overlap status = `{prov['overlap_classification']}`

What can be established: shared task universe (both target Terminal-Bench 2.0)
and a shared corpus provenance path. EarlyEval's `terminalbench_lightgbm`
registry entry lists `../data/terminalbench-trajectories` with source
`earlyeval/benchmarks/normalize.py + migrated adapter spec`, and that normalize
module consumes exactly the community schema.

What cannot be established: trial-level identity. The frozen EarlyEval clone
ships no TerminalBench trajectories and exposes no trial_id / trajectory id /
timestamp / source log path / submission hash, so no exact overlap percentage is
inferred and the community dataset is not called independent.

## J. PHASE2B RECOMMENDED DESIGN

candidate 1 = {c1t}

candidate 2 = {c2t}

Ranked by eligible targets, common-task overlap, balanced success/failure
support, trajectory completeness and crossed connectivity. The ranking did not
use expected calibration failure and did not inspect model predictions.

## K. COST

* download = 221005122 bytes (210.77 MiB), 57.9 s wall clock
* disk = about 211 MiB dataset plus derived scan artifacts
* wall-clock compute = scan {agg['wall_clock_seconds']} s + adapter probe {probe['wall_clock_seconds']} s
* peak RAM = bounded by batch streaming (batch_size 1000); step bodies were never all held at once
* API cost = 0
* LLM cost = 0
* cloud cost = 0

## L. BLOCKERS / DEVIATIONS

* 17642 of 52104 rows (33.9 percent) have no trajectory (`steps = null`); this
  is surfaced as LOW_TRAJECTORY_COVERAGE = TRUE and is not a blocker because all
  70 combos with usable trajectories remain eligible.
* `trial_id` is the empty string for 29506 rows and is not a unique trial key;
  it is not used as a provenance identifier.
* Sandbox network access required one escalated retrieval; no credential was
  read, no secret was printed, and no API or LLM endpoint was called.
* The frozen upstream clone was not modified (the vendor runtime root was
  redirected into `work/lre_phase2a/vendor_runtime`).

## M. ARTIFACT PATH + HASH MANIFEST

* path = `{OUT}`
* `manifest.json` lists every artifact except itself and `artifact_sha256sums.txt`
* `artifact_sha256sums.txt` lists every artifact except itself

Frozen upstream integrity is recorded in `integrity_report.json`
(`frozen_upstream_verified`, `frozen_upstream_all_match`).
"""


def main() -> int:
    agg = json.loads((WORK / "scan_aggregate.json").read_text("utf-8"))
    combos = json.loads((WORK / "scan_combo_rows.json").read_text("utf-8"))
    gate = json.loads((AN / "phase2a_gate.json").read_text("utf-8"))
    conn = json.loads((AN / "design_connectivity.json").read_text("utf-8"))
    prov = json.loads((AN / "provenance_overlap.json").read_text("utf-8"))
    p2b = json.loads((AN / "phase2b_candidate_designs.json").read_text("utf-8"))
    probe = json.loads((ADS / "adapter_probe_results.json").read_text("utf-8"))
    eligible = [c for c in combos
                if c["usable_trajectories"] >= 100 and c["successes"] >= 20
                and c["failures"] >= 20 and c["unique_tasks_usable"] >= 30]
    us = sum(c["successes"] for c in combos)
    uf = sum(c["failures"] for c in combos)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROTOCOL.md").write_text(protocol(agg), "utf-8")
    (OUT / "PHASE2A_REPORT.md").write_text(
        report(agg, combos, eligible, gate, conn, prov, p2b, probe, us, uf), "utf-8")
    print("wrote PROTOCOL.md and PHASE2A_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
