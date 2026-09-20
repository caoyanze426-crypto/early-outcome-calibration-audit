# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - Section 11/12 adapter probe.

Runs the *frozen* EarlyEval functions
  earlyeval.benchmarks.normalize.normalize_record
  vendor/prefix_predict_model_holdout_answer/step_builder.rebuild_steps_for_trajectory
  vendor/prefix_predict_model_holdout_answer/prefix_builder.build_prefix_samples_for_trajectory
on a deterministic bounded sample of the community dataset, through the
deterministic adapter shipped alongside the artifact.

The frozen upstream clone is never written to: the vendor runtime root is
redirected into this phase's work directory.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2a"
DATA = WORK / "data"
OUT = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
ADAPTER_DIR = OUT / "adapter"
CLONE = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
VENDOR = CLONE / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"

SAMPLE_TARGET = 150
PER_COMBO = 25
COLLECT_CAP = 300


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pick_combos(combos):
    """Deterministically choose >=5 combos spanning >=3 models and >=3 scaffolds."""
    elig = [c for c in combos
            if c["usable_trajectories"] >= 100 and c["successes"] >= 20
            and c["failures"] >= 20 and c["unique_tasks_usable"] >= 30]
    by_size = sorted(elig, key=lambda c: (-c["usable_trajectories"], c["model"], c["agent"]))
    chosen = [(c["model"], c["agent"]) for c in by_size[:6]]
    pool = [(c["model"], c["agent"]) for c in
            sorted(elig, key=lambda c: (c["model"], c["agent"]))]

    def div(sel):
        return len({m for m, _ in sel}), len({a for _, a in sel})

    guard = 0
    while guard < 50:
        guard += 1
        nm, na = div(chosen)
        if nm >= 3 and na >= 3:
            break
        added = False
        for k in pool:
            if k in chosen:
                continue
            nm2, na2 = div(chosen + [k])
            if nm2 > nm or na2 > na:
                chosen.append(k)
                added = True
                break
        if not added:
            break
    return chosen


def interleave(a, b):
    out = []
    i = j = 0
    while i < len(a) or j < len(b):
        if i < len(a):
            out.append(a[i])
            i += 1
        if j < len(b):
            out.append(b[j])
            j += 1
    return out


def main() -> int:
    t0 = time.time()
    combos = json.loads((WORK / "scan_combo_rows.json").read_text("utf-8"))
    chosen = pick_combos(combos)
    chosen_set = set(chosen)

    # ---- collect a bounded deterministic sample ----------------------------
    buckets = defaultdict(list)
    for shard in sorted(DATA.rglob("data/*.parquet")):
        pf = pq.ParquetFile(shard)
        cols = ["task_name", "agent", "model", "reward", "trial_id",
                "trial_name", "started_at", "steps"]
        for batch in pf.iter_batches(batch_size=1000, columns=cols):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                key = (d["model"][i], d["agent"][i])
                if key not in chosen_set:
                    continue
                if d["steps"][i] in (None, "", "null"):
                    continue
                if len(buckets[key]) >= COLLECT_CAP:
                    continue
                buckets[key].append({
                    "task_name": d["task_name"][i], "agent": d["agent"][i],
                    "model": d["model"][i], "reward": d["reward"][i],
                    "trial_id": d["trial_id"][i], "trial_name": d["trial_name"][i],
                    "started_at": d["started_at"][i], "steps": d["steps"][i],
                })

    sample = []
    per_combo_taken = {}
    for key in chosen:
        rows = buckets.get(key, [])
        rows.sort(key=lambda r: (r["task_name"], r["started_at"] or "",
                                 r["trial_name"] or "", r["trial_id"] or ""))
        succ = [r for r in rows if r["reward"] == 1]
        fail = [r for r in rows if r["reward"] == 0]
        take = interleave(succ, fail)[:PER_COMBO]
        per_combo_taken[f"{key[0]}\u241f{key[1]}"] = len(take)
        sample.extend(take)

    # ---- frozen upstream + adapter -----------------------------------------
    adapter = load_module("p2a_adapter", ADAPTER_DIR / "adapter_terminalbench.py")

    os.environ["EARLYEVAL_VENDOR_RUNTIME_ROOT"] = str(WORK / "vendor_runtime")
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(CLONE))
    from earlyeval.benchmarks import normalize as frozen_normalize  # noqa: E402
    import step_builder  # noqa: E402
    import prefix_builder  # noqa: E402

    ingestion_matches = 0
    ingestion_mismatch_examples = []
    direct_steps_total = 0
    adapter_steps_total = 0
    prefix_rows_total = 0
    steps_ok = 0
    prefix_ok = 0
    failures = []
    prefix_schema = set()
    example_mapping = None

    for idx, row in enumerate(sample):
        ref = frozen_normalize.normalize_record(row, "terminalbench", idx)
        ours = adapter.to_ingestion_record(row, idx)
        if ref == ours:
            ingestion_matches += 1
        elif len(ingestion_mismatch_examples) < 5:
            ingestion_mismatch_examples.append({"idx": idx, "ref": ref, "ours": ours})

        vendor = adapter.to_vendor_record(row, idx)

        # evidence that the frozen ingestion contract alone is NOT sufficient
        direct = step_builder.rebuild_steps_for_trajectory(ref)
        direct_steps_total += len(direct)

        try:
            steps = step_builder.rebuild_steps_for_trajectory(vendor)
        except Exception as exc:  # noqa: BLE001
            failures.append({"idx": idx, "stage": "step_builder",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        adapter_steps_total += len(steps)
        if len(steps) > 0:
            steps_ok += 1
        try:
            prefixes = prefix_builder.build_prefix_samples_for_trajectory(vendor)
        except Exception as exc:  # noqa: BLE001
            failures.append({"idx": idx, "stage": "prefix_builder",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        prefix_rows_total += len(prefixes)
        if prefixes and len(prefixes) == len(steps) + 1:
            prefix_ok += 1
            prefix_schema = set(prefixes[0].keys())

        if example_mapping is None:
            example_mapping = {
                "traj_id": vendor["traj_id"],
                "instance_id": vendor["instance_id"],
                "model": vendor["model"],
                "resolved": vendor["resolved"],
                "n_community_steps": len(adapter.parse_steps(row["steps"])),
                "n_vendor_messages": len(vendor["messages"]),
                "n_rebuilt_steps": len(steps),
                "n_prefix_rows": len(prefixes),
                "first_community_step": json.loads(row["steps"])[0],
                "first_vendor_message": vendor["messages"][0] if vendor["messages"] else None,
            }

    models = sorted({m for m, _ in chosen})
    scaffolds = sorted({a for _, a in chosen})
    rewards = Counter(r["reward"] for r in sample)
    required_cols = ["traj_id", "instance_id", "model_id", "label",
                     "prefix_step_idx", "n_steps_total_for_weighting"]
    constraints = {
        "sample_n_ge_100": len(sample) >= 100,
        "combos_ge_5": len(chosen) >= 5,
        "models_ge_3": len(models) >= 3,
        "scaffolds_ge_3": len(scaffolds) >= 3,
        "has_success_and_failure": rewards.get(1, 0) > 0 and rewards.get(0, 0) > 0,
    }
    adapter_pass = (all(constraints.values()) and ingestion_matches == len(sample)
                    and steps_ok == len(sample)
                    and prefix_ok == len(sample) and not failures)

    results = {
        "section": "11/12 EARLYEVAL TERMINALBENCH SUPPORT + ADAPTER PROBE",
        "classification": "SIMPLE_ADAPTER",
        "classification_evidence": {
            "ingestion_contract_is_direct": (
                "community schema columns task_name/model/reward/trial_id/trial_name/"
                "steps are exactly what earlyeval/benchmarks/normalize.py:"
                "normalize_record(...,'terminalbench') consumes"),
            "feature_layer_requires_adapter": (
                "feeding the frozen normalized record straight into "
                "vendor step_builder.rebuild_steps_for_trajectory yields zero steps "
                "because the frozen ingestion messages lack message_type=='action' / "
                "'action' / 'tool_calls'"),
            "direct_frozen_normalize_into_step_builder_steps_total": direct_steps_total,
            "adapted_steps_total": adapter_steps_total,
        },
        "frozen_pipeline_recovered": {
            "normalize_module": str(CLONE / "earlyeval" / "benchmarks" / "normalize.py"),
            "normalize_terminalbench_fields": ["benchmark", "instance_id", "traj_id",
                                               "model_id", "resolved", "messages", "patch"],
            "step_builder_module": str(VENDOR / "step_builder.py"),
            "prefix_builder_module": str(VENDOR / "prefix_builder.py"),
            "vendor_action_message_requirement": "role=assistant AND message_type=='action' AND 'action' in msg",
            "controlled_by_env": "EARLYEVAL_VENDOR_RUNTIME_ROOT",
            "prefix_table_required_columns": required_cols,
            "prefix_table_columns_present": sorted(prefix_schema),
            "prefix_table_required_columns_all_present": all(
                c in prefix_schema for c in required_cols),
        },
        "sample_selection": {
            "policy": "deterministic: 6 largest eligible combos, extended greedily until >=3 models and >=3 scaffolds; per combo sorted by (task_name, started_at, trial_name, trial_id), success/failure interleaved, first 25 taken",
            "target_trajectories": SAMPLE_TARGET,
            "chosen_combos": [{"model": m, "agent": a} for m, a in chosen],
            "per_combo_taken": per_combo_taken,
        },
        "constraints": constraints,
        "sample_n": len(sample),
        "combos": len(chosen),
        "models": models,
        "scaffolds": scaffolds,
        "successes": rewards.get(1, 0),
        "failures": rewards.get(0, 0),
        "ingestion_records_matching_frozen_normalize": ingestion_matches,
        "ingestion_mismatch_examples": ingestion_mismatch_examples,
        "trajectories_with_steps_rebuilt": steps_ok,
        "trajectories_with_prefix_rows": prefix_ok,
        "prefix_rows_total": prefix_rows_total,
        "adapter_errors": failures,
        "adapter": "PASS" if adapter_pass else "FAIL",
        "example_mapping": example_mapping,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "upstream_clone_modified": False,
        "api_calls": 0, "llm_calls": 0, "trainer_calls": 0,
    }
    (ADAPTER_DIR / "adapter_probe_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("example_mapping", "sample_selection",
                                   "ingestion_mismatch_examples")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
