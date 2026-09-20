# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - coverage / connectivity / provenance analysis.

Consumes work/lre_phase2a/scan_*.json|jsonl and writes the Section-18
analysis artifacts under outputs/earlyeval_phase2a_terminalbench_coverage/.

Deterministic, offline, zero API/LLM cost.
"""
from __future__ import annotations
import os

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2a"
OUT = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
AN = OUT / "analysis"

ELIG_USABLE = 100
ELIG_SUCC = 20
ELIG_FAIL = 20
ELIG_TASKS = 30
CROSS_MIN = 3
PAIR_MIN_TASKS = 30


def wcsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def wjson(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def med(vals):
    return round(statistics.median(vals), 6) if vals else None


def mean(vals):
    return round(statistics.fmean(vals), 6) if vals else None


def build_provenance(agg, tasks):
    return {
        "section": "13 PROVENANCE OVERLAP AUDIT",
        "community_dataset": {
            "repo_id": "yoonholee/terminalbench-trajectories",
            "revision": "04e8940f5b6736a7ce8d22224fe2f2af74163ed2",
            "license": "apache-2.0",
            "benchmark": "Terminal-Bench 2.0",
            "task_names": len(tasks),
        },
        "earlyeval_side": {
            "repo_url": "https://github.com/inphotoo/earlyeval",
            "commit_sha": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
            "terminalbench_registry_entry": "configs/experiment_registry.yaml:terminalbench_lightgbm",
            "registry_source_field": "earlyeval/benchmarks/normalize.py + migrated adapter spec",
            "registry_data_path": "../data/terminalbench-trajectories",
            "terminalbench_prefix_tables": [
                "../data/other_bench_prefix_tables/terminalbench/holdout15_p99_v3_audited_20260512/prefix_table_terminalbench.parquet",
                "../data/harness_debug_cross_agent_20260625/prefix_tables/prefix_table_terminalbench_harness_core16.parquet",
                "../data/harness_debug_exclusion_20260626/prefix_tables/prefix_table_terminalbench_slot4x4.parquet",
            ],
            "prefix_table_contract_columns": [
                "traj_id", "instance_id", "model_id", "label",
                "prefix_step_idx", "n_steps_total_for_weighting"],
            "local_corpus_present": False,
            "local_corpus_note": (
                "work/lre_phase0a/third_party/data does not exist; the frozen clone "
                "ships no TerminalBench trajectories"),
        },
        "identifiers_comparable": {
            "task_name": {
                "available_both_sides": "partial",
                "community": f"{len(tasks)} distinct Terminal-Bench 2.0 task names",
                "earlyeval": "task names referenced only through derived prefix tables; raw corpus absent",
            },
            "model_label": {"available_both_sides": "labels only (community); EarlyEval raw labels absent"},
            "scaffold_label": {"available_both_sides": "labels only (community); EarlyEval raw labels absent"},
            "trial_id": {"available_both_sides": "community only (and empty for 29506/52104 rows)"},
            "trajectory_id": {"available_both_sides": "not present in EarlyEval release"},
            "timestamp": {"available_both_sides": "community started_at/ended_at; EarlyEval none"},
            "source_log_path": {"available_both_sides": "not present on either side"},
            "submission_hash": {"available_both_sides": "not present on either side"},
        },
        "shared_task_universe": {
            "established": True,
            "basis": (
                "both sides target Terminal-Bench 2.0; EarlyEval config references a "
                "corpus directory named `terminalbench-trajectories` and normalizes it "
                "with earlyeval/benchmarks/normalize.py, whose terminalbench schema "
                "(task_name / model / reward / steps) is exactly the community schema"),
        },
        "shared_model_scaffold_labels": {
            "established": False,
            "basis": "EarlyEval release ships no raw TerminalBench label table; only derived prefix parquet paths are visible",
        },
        "trial_identity_established": False,
        "overlap_classification": "PARTIAL_OVERLAP_VERIFIABLE",
        "classification_reason": (
            "shared task universe and shared corpus provenance path are verifiable, "
            "but trial-level identity cannot be established and an exact overlap "
            "percentage must not be inferred"),
        "independent_dataset_claim_permitted": False,
        "independent_dataset_reason": "trial-level non-overlap was not demonstrated",
    }


def build_phase2b(design_a, design_b, combo_index, task_usable, status):
    if status != "FEASIBLE":
        return {"section": "16 PHASE 2B DESIGN RECOMMENDATION",
                "status": status, "candidates": [],
                "note": "no recommendation issued because the gate is not FEASIBLE"}

    def common(k1, k2):
        t1, t2 = set(task_usable[k1]), set(task_usable[k2])
        return len(t1 & t2)

    def score(key, members, same_model):
        pairs_ct = []
        combos_here = []
        for x, y in combinations(members, 2):
            k1 = (key, x) if same_model else (x, key)
            k2 = (key, y) if same_model else (y, key)
            pairs_ct.append(common(k1, k2))
        for x in members:
            combos_here.append(combo_index[(key, x) if same_model else (x, key)])
        balance = min(min(c["success_rate"], 1 - c["success_rate"]) for c in combos_here)
        return {
            "group": key, "members": members,
            "n_targets": len(members), "n_pairs": len(pairs_ct),
            "common_tasks_median": med(pairs_ct),
            "common_tasks_min": min(pairs_ct) if pairs_ct else None,
            "common_tasks_max": max(pairs_ct) if pairs_ct else None,
            "min_balance": round(balance, 6),
            "mean_usable_fraction": round(statistics.fmean(
                [c["usable_fraction"] for c in combos_here]), 6),
            "mean_usable_trajectories": round(statistics.fmean(
                [c["usable_trajectories"] for c in combos_here]), 3),
        }

    cand_a = [score(g, v["members"], True) for g, v in design_a.items()
              if v["crossed_eligible"]]
    cand_b = [score(g, v["members"], False) for g, v in design_b.items()
              if v["crossed_eligible"]]

    def rank(cands):
        return sorted(cands, key=lambda c: (-c["n_targets"],
                                            -(c["common_tasks_median"] or 0),
                                            -c["min_balance"],
                                            -c["mean_usable_fraction"]))

    cands = []
    if cand_a:
        c = rank(cand_a)[0]
        d = dict(c)
        d["design"] = "MODEL_CONTROLLED"
        d["same_exact_model"] = c["group"]
        d["scaffolds"] = c["members"]
        d.pop("group", None)
        d.pop("members", None)
        cands.append(d)
    if cand_b:
        c = rank(cand_b)[0]
        d = dict(c)
        d["design"] = "SCAFFOLD_CONTROLLED"
        d["same_exact_scaffold"] = c["group"]
        d["models"] = c["members"]
        d.pop("group", None)
        d.pop("members", None)
        cands.append(d)
    return {
        "section": "16 PHASE 2B DESIGN RECOMMENDATION",
        "status": "FEASIBLE",
        "ranking_policy": ["eligible targets", "common-task overlap",
                           "balanced success/failure support",
                           "trajectory completeness",
                           "crossed model/scaffold connectivity"],
        "explicit_exclusions": ["did NOT rank by expected calibration failure",
                                "did NOT inspect model prediction outcomes"],
        "candidate_1": cands[0] if len(cands) > 0 else None,
        "candidate_2": cands[1] if len(cands) > 1 else None,
    }


def main() -> int:
    agg = json.loads((WORK / "scan_aggregate.json").read_text("utf-8"))
    combos = json.loads((WORK / "scan_combo_rows.json").read_text("utf-8"))
    aux = json.loads((WORK / "scan_aux_sets.json").read_text("utf-8"))

    task_usable = defaultdict(Counter)
    with (WORK / "scan_rows.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["usable"]:
                task_usable[(r["model"], r["agent"])][r["task_name"]] += 1

    combo_index = {(c["model"], c["agent"]): c for c in combos}
    for c in combos:
        c["eligible"] = bool(
            c["usable_trajectories"] >= ELIG_USABLE
            and c["successes"] >= ELIG_SUCC
            and c["failures"] >= ELIG_FAIL
            and c["unique_tasks_usable"] >= ELIG_TASKS)

    eligible = [c for c in combos if c["eligible"]]
    elig_keys = [(c["model"], c["agent"]) for c in eligible]

    def common_tasks(k1, k2):
        t1, t2 = set(task_usable[k1]), set(task_usable[k2])
        shared = t1 & t2
        matched = sum(min(task_usable[k1][t], task_usable[k2][t]) for t in shared)
        return len(shared), matched

    # ---- Section 4/5/6 ------------------------------------------------------
    combo_summary_rows = []
    coverage_rows = []
    for c in sorted(combos, key=lambda x: (-x["usable_trajectories"], x["model"], x["agent"])):
        counts = sorted(task_usable[(c["model"], c["agent"])].values())
        combo_summary_rows.append({
            "model": c["model"], "agent": c["agent"],
            "usable_trajectories": c["usable_trajectories"],
            "unique_tasks": c["unique_tasks_usable"],
            "successes": c["successes"], "failures": c["failures"],
            "success_rate": c["success_rate"],
            "trials_per_task_mean": mean(counts),
            "trials_per_task_median": med(counts),
            "trials_per_task_min": min(counts) if counts else None,
            "trials_per_task_max": max(counts) if counts else None,
            "eligible": c["eligible"],
            "ineligible_reasons": "|".join(
                r for r, ok in (
                    ("USABLE_LT_100", c["usable_trajectories"] >= ELIG_USABLE),
                    ("SUCCESS_LT_20", c["successes"] >= ELIG_SUCC),
                    ("FAILURE_LT_20", c["failures"] >= ELIG_FAIL),
                    ("TASKS_LT_30", c["unique_tasks_usable"] >= ELIG_TASKS),
                ) if not ok),
        })
        frac = c["usable_fraction"]
        coverage_rows.append({
            "model": c["model"], "agent": c["agent"],
            "all_public_trials": c["all_trials"],
            "usable_full_trajectories": c["usable_trajectories"],
            "usable_fraction": frac,
            "low_trajectory_coverage": bool(frac is not None and frac < 0.70),
            "unique_tasks_usable": c["unique_tasks_usable"],
            "successes": c["successes"], "failures": c["failures"],
        })

    wcsv(AN / "target_combo_summary.csv",
         ["model", "agent", "usable_trajectories", "unique_tasks", "successes",
          "failures", "success_rate", "trials_per_task_mean",
          "trials_per_task_median", "trials_per_task_min",
          "trials_per_task_max", "eligible", "ineligible_reasons"],
         combo_summary_rows)
    wcsv(AN / "trajectory_usability.csv",
         ["model", "agent", "all_public_trials", "usable_full_trajectories",
          "usable_fraction", "low_trajectory_coverage", "unique_tasks_usable",
          "successes", "failures"],
         coverage_rows)

    # ---- Section 7/8 --------------------------------------------------------
    model_to_scaffolds = defaultdict(list)
    scaffold_to_models = defaultdict(list)
    for m, a in elig_keys:
        model_to_scaffolds[m].append(a)
        scaffold_to_models[a].append(m)
    for m in model_to_scaffolds:
        model_to_scaffolds[m] = sorted(model_to_scaffolds[m])
    for a in scaffold_to_models:
        scaffold_to_models[a] = sorted(scaffold_to_models[a])

    model_crossed = {m: len(s) >= CROSS_MIN for m, s in model_to_scaffolds.items()}
    scaffold_crossed = {a: len(ms) >= CROSS_MIN for a, ms in scaffold_to_models.items()}

    sm_rows = []
    for m in sorted(model_to_scaffolds):
        sc = model_to_scaffolds[m]
        sm_rows.append({
            "row_type": "model_summary", "model": m,
            "eligible_scaffolds_count": len(sc),
            "eligible_scaffolds": "|".join(sc),
            "model_crossed_eligible": model_crossed[m],
            "scaffold_a": "", "scaffold_b": "",
            "common_tasks": "", "common_usable_trials": "",
        })
        for a, b in combinations(sc, 2):
            ct, cm = common_tasks((m, a), (m, b))
            sm_rows.append({
                "row_type": "scaffold_pair", "model": m,
                "eligible_scaffolds_count": len(sc), "eligible_scaffolds": "",
                "model_crossed_eligible": model_crossed[m],
                "scaffold_a": a, "scaffold_b": b,
                "common_tasks": ct, "common_usable_trials": cm,
            })
    wcsv(AN / "same_model_cross_scaffold.csv",
         ["row_type", "model", "eligible_scaffolds_count", "eligible_scaffolds",
          "model_crossed_eligible", "scaffold_a", "scaffold_b",
          "common_tasks", "common_usable_trials"], sm_rows)

    ss_rows = []
    for a in sorted(scaffold_to_models):
        ms = scaffold_to_models[a]
        ss_rows.append({
            "row_type": "scaffold_summary", "agent": a,
            "eligible_models_count": len(ms), "eligible_models": "|".join(ms),
            "scaffold_crossed_eligible": scaffold_crossed[a],
            "model_a": "", "model_b": "",
            "common_tasks": "", "common_usable_trials": "",
        })
        for m1, m2 in combinations(ms, 2):
            ct, cm = common_tasks((m1, a), (m2, a))
            ss_rows.append({
                "row_type": "model_pair", "agent": a,
                "eligible_models_count": len(ms), "eligible_models": "",
                "scaffold_crossed_eligible": scaffold_crossed[a],
                "model_a": m1, "model_b": m2,
                "common_tasks": ct, "common_usable_trials": cm,
            })
    wcsv(AN / "same_scaffold_cross_model.csv",
         ["row_type", "agent", "eligible_models_count", "eligible_models",
          "scaffold_crossed_eligible", "model_a", "model_b",
          "common_tasks", "common_usable_trials"], ss_rows)

    # ---- Section 9 ----------------------------------------------------------
    seen = set()
    pair_rows = []
    for (m1, a1), (m2, a2) in combinations(sorted(elig_keys), 2):
        rel = []
        if m1 == m2:
            rel.append("SAME_MODEL")
        if a1 == a2:
            rel.append("SAME_SCAFFOLD")
        if not rel:
            continue
        ct, cm = common_tasks((m1, a1), (m2, a2))
        pair_rows.append({
            "model_a": m1, "scaffold_a": a1, "model_b": m2, "scaffold_b": a2,
            "relation": "+".join(rel),
            "common_tasks": ct, "common_usable_trials": cm,
            "ge30": ct >= 30, "ge40": ct >= 40, "ge50": ct >= 50, "ge70": ct >= 70,
            "pair_common_task_eligible": ct >= PAIR_MIN_TASKS,
        })
    pair_rows.sort(key=lambda r: (-r["common_tasks"], r["model_a"], r["scaffold_a"],
                                  r["model_b"], r["scaffold_b"]))
    wcsv(AN / "pair_common_tasks.csv",
         ["model_a", "scaffold_a", "model_b", "scaffold_b", "relation",
          "common_tasks", "common_usable_trials", "ge30", "ge40", "ge50",
          "ge70", "pair_common_task_eligible"], pair_rows)

    # ---- Section 10 ---------------------------------------------------------
    tm_rows = []
    cell_hist = Counter()
    for c in sorted(combos, key=lambda x: (x["agent"], x["model"])):
        counts = sorted(task_usable[(c["model"], c["agent"])].values())
        for v in counts:
            cell_hist[v] += 1
        tm_rows.append({
            "model": c["model"], "agent": c["agent"],
            "n_task_cells": len(counts),
            "usable_trials": c["usable_trajectories"],
            "trials_per_cell_mean": mean(counts),
            "trials_per_cell_median": med(counts),
            "trials_per_cell_min": min(counts) if counts else None,
            "trials_per_cell_max": max(counts) if counts else None,
            "cells_1_trial": sum(1 for v in counts if v == 1),
            "cells_2_4_trials": sum(1 for v in counts if 2 <= v <= 4),
            "cells_5_9_trials": sum(1 for v in counts if 5 <= v <= 9),
            "cells_10plus_trials": sum(1 for v in counts if v >= 10),
        })
    wcsv(AN / "trial_multiplicity.csv",
         ["model", "agent", "n_task_cells", "usable_trials",
          "trials_per_cell_mean", "trials_per_cell_median",
          "trials_per_cell_min", "trials_per_cell_max",
          "cells_1_trial", "cells_2_4_trials", "cells_5_9_trials",
          "cells_10plus_trials"], tm_rows)

    # ---- Section 14 ---------------------------------------------------------
    def design_summary(same_model):
        groups = {}
        if same_model:
            groups = {m: s for m, s in model_to_scaffolds.items() if len(s) >= 2}
        else:
            groups = {a: ms for a, ms in scaffold_to_models.items() if len(ms) >= 2}
        out = {}
        for g, members in sorted(groups.items()):
            if same_model:
                pairs = [common_tasks((g, x), (g, y))[0] for x, y in combinations(members, 2)]
            else:
                pairs = [common_tasks((x, g), (y, g))[0] for x, y in combinations(members, 2)]
            out[g] = {"eligible_combos": len(members), "members": members,
                      "eligible_pairs": len(pairs),
                      "common_tasks_median": med(pairs),
                      "common_tasks_min": min(pairs) if pairs else None,
                      "common_tasks_max": max(pairs) if pairs else None,
                      "crossed_eligible": len(members) >= CROSS_MIN}
        return out

    design_a = design_summary(True)
    design_b = design_summary(False)
    global_frac = agg["rows_with_steps"] / agg["total_rows"]

    connectivity = {
        "section": "14 DESIGN CONNECTIVITY",
        "DESIGN_A": {
            "definition": "same exact model across multiple eligible scaffolds",
            "groups_with_ge2_eligible_scaffolds": len(design_a),
            "groups_with_ge3_eligible_scaffolds": sum(
                1 for v in design_a.values() if v["crossed_eligible"]),
            "per_model": design_a,
        },
        "DESIGN_B": {
            "definition": "same exact scaffold across multiple eligible models",
            "groups_with_ge2_eligible_models": len(design_b),
            "groups_with_ge3_eligible_models": sum(
                1 for v in design_b.values() if v["crossed_eligible"]),
            "per_scaffold": design_b,
        },
        "section_6_coverage": {
            "rows": agg["total_rows"],
            "usable_full_trajectories": agg["rows_with_steps"],
            "global_usable_fraction": round(global_frac, 6),
            "LOW_TRAJECTORY_COVERAGE": bool(global_frac < 0.70),
        },
        "section_9_common_task_support": {
            "controlled_pairs_total": len(pair_rows),
            "pairs_ge30": sum(1 for r in pair_rows if r["ge30"]),
            "pairs_ge40": sum(1 for r in pair_rows if r["ge40"]),
            "pairs_ge50": sum(1 for r in pair_rows if r["ge50"]),
            "pairs_ge70": sum(1 for r in pair_rows if r["ge70"]),
            "common_tasks_median": med([r["common_tasks"] for r in pair_rows]),
            "common_tasks_min": min((r["common_tasks"] for r in pair_rows), default=None),
            "common_tasks_max": max((r["common_tasks"] for r in pair_rows), default=None),
        },
        "section_10_trial_multiplicity": {
            "cells_total": sum(cell_hist.values()),
            "cells_1_trial": cell_hist.get(1, 0),
            "cells_2_4_trials": sum(v for k, v in cell_hist.items() if 2 <= k <= 4),
            "cells_5_9_trials": sum(v for k, v in cell_hist.items() if 5 <= k <= 9),
            "cells_10plus_trials": sum(v for k, v in cell_hist.items() if k >= 10),
            "max_trials_per_cell": max(cell_hist) if cell_hist else None,
            "frozen_recommendation": (
                "future bootstrap / inference unit = task_name cluster, retaining "
                "all repeated trials for a selected task; repeated trials must not "
                "be treated as independent tasks"),
        },
    }
    wjson(AN / "design_connectivity.json", connectivity)
    wjson(AN / "provenance_overlap.json",
          build_provenance(agg, sorted(set(aux["tasks_usable"]))))

    # ---- Section 15 ---------------------------------------------------------
    models_ge3 = sorted(m for m, ok in model_crossed.items() if ok)
    scaffolds_ge3 = sorted(a for a, ok in scaffold_crossed.items() if ok)
    pairs_ge30 = sum(1 for r in pair_rows if r["ge30"])
    gate = {
        "section": "15 PHASE 2A FEASIBILITY GATE",
        "A_binary_final_outcome_recoverable": {
            "pass": (agg["reward_missing"] == 0 and agg["reward_nonbinary"] == 0
                     and set(agg["reward_raw_values"]) == {"0", "1"}),
            "evidence": {
                "reward_raw_values": agg["reward_raw_values"],
                "missing": agg["reward_missing"], "nonbinary": agg["reward_nonbinary"],
                "documented_semantics": "reward = 1 if the agent solved the task, 0 otherwise (dataset README, Schema table)"},
        },
        "B_full_step_trajectory_recoverable": {
            "pass": agg["rows_with_steps"] >= 0.5 * agg["total_rows"],
            "evidence": {
                "rows_with_steps": agg["rows_with_steps"],
                "total_rows": agg["total_rows"],
                "fraction": round(global_frac, 6),
                "schema_fail_rows": agg["schema_fail_rows"]},
        },
        "C_mapping_direct_or_simple_adapter": {
            "pass": True, "classification": "SIMPLE_ADAPTER",
            "evidence": "ingestion contract is DIRECT (normalize.py terminalbench reads this schema); feature/prefix layer needs a deterministic field-remap adapter",
        },
        "D_at_least_8_eligible_combos": {
            "pass": len(eligible) >= 8, "eligible_combos": len(eligible)},
        "E_crossed_structure": {
            "pass": (len(models_ge3) >= 2) or (len(scaffolds_ge3) >= 2),
            "models_with_ge3_eligible_scaffolds": models_ge3,
            "scaffolds_with_ge3_eligible_models": scaffolds_ge3},
        "F_at_least_10_pairs_ge30_common_tasks": {
            "pass": pairs_ge30 >= 10, "pairs_ge30": pairs_ge30},
        "G_no_paid_api": {"pass": True, "api_calls": 0, "llm_calls": 0,
                          "cloud_cost_cny": 0, "new_trajectories": 0},
    }
    checks = [gate[k]["pass"] for k in (
        "A_binary_final_outcome_recoverable", "B_full_step_trajectory_recoverable",
        "C_mapping_direct_or_simple_adapter", "D_at_least_8_eligible_combos",
        "E_crossed_structure", "F_at_least_10_pairs_ge30_common_tasks",
        "G_no_paid_api")]
    if all(checks):
        status = "FEASIBLE"
    elif all(checks[:3]):
        status = "PARTIALLY_FEASIBLE"
    else:
        status = "BLOCKED"
    gate["PHASE2A"] = status
    gate["all_checks_pass"] = all(checks)
    wjson(AN / "phase2a_gate.json", gate)

    wjson(AN / "phase2b_candidate_designs.json",
          build_phase2b(design_a, design_b, combo_index, task_usable, status))

    print(json.dumps({
        "total_rows": agg["total_rows"],
        "usable": agg["rows_with_steps"],
        "global_frac": round(global_frac, 6),
        "eligible_combos": len(eligible),
        "models_ge3": models_ge3,
        "scaffolds_ge3": scaffolds_ge3,
        "controlled_pairs": len(pair_rows),
        "pairs_ge30": pairs_ge30,
        "pairs_ge50": sum(1 for r in pair_rows if r["ge50"]),
        "pairs_ge70": sum(1 for r in pair_rows if r["ge70"]),
        "status": status,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
