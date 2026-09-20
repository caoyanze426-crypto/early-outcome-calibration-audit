# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - full-corpus coverage scan (streaming, deterministic).

Reads both pinned Parquet shards batch-by-batch (never materialising the
939 MB of step text at once) and produces:

  work/lre_phase2a/scan_aggregate.json   - complete aggregate evidence
  work/lre_phase2a/scan_rows.jsonl       - one compact record per ROW
                                           (no step bodies, only derived flags)

No API / LLM / cloud calls. Pure local deterministic Python.
"""
from __future__ import annotations
import os

import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
DATA = WS / "work" / "lre_phase2a" / "data"
OUT = WS / "work" / "lre_phase2a"

COLS = ["task_name", "agent", "model", "reward", "trial_id", "trial_name",
        "duration_seconds", "input_tokens", "output_tokens", "cache_tokens",
        "cost_cents", "started_at", "ended_at", "steps"]

PRESENT = object()  # sentinel meaning "column physically present in file"


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pct(values, q):
    """Deterministic nearest-rank percentile (no interpolation surprises)."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def fsum(v):
    return round(v, 6) if isinstance(v, float) else v


def main() -> int:
    t0 = time.time()
    shards = sorted(DATA.rglob("data/*.parquet"))
    if not shards:
        print("no parquet shards found", file=sys.stderr)
        return 2

    # ---- file-level identity -------------------------------------------------
    file_manifest = []
    for p in sorted(DATA.rglob("*")):
        if not p.is_file():
            continue
        file_manifest.append({
            "path": p.relative_to(DATA).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    shard_info = []
    for p in shards:
        f = pq.ParquetFile(p)
        shard_info.append({
            "path": p.relative_to(DATA).as_posix(),
            "rows": f.metadata.num_rows,
            "row_groups": f.metadata.num_row_groups,
            "created_by": f.metadata.created_by,
            "columns": [f.schema_arrow.field(i).name
                        for i in range(len(f.schema_arrow))],
        })

    # ---- accumulators --------------------------------------------------------
    total = 0
    reasons = Counter()
    flag_counts = Counter()          # raw condition flags, not exclusive
    reward_raw = Counter()
    reward_missing = 0
    reward_nonbinary = 0
    rows_with_steps = 0
    rows_without_steps = 0
    empty_step_lists = 0
    schema_fail_rows = 0
    warmup_first = 0
    empty_first_msg = 0
    null_msg = 0
    null_obs = 0
    null_tools = 0

    step_keys = Counter()
    src_values = Counter()
    step_len_hist = Counter()
    steps_len_usable = []
    tool_fn_values = Counter()

    tasks_all, tasks_usable = set(), set()
    models_all, models_usable = set(), set()
    agents_all, agents_usable = set(), set()
    combos_all, combos_usable = set(), set()
    trial_ids = set()
    trial_id_rows = 0
    trial_id_empty_rows = 0
    trial_id_nonempty_rows = 0
    usable_with_nonempty_trial_id = 0
    trial_name_ids = set()
    steps_state_counts = Counter()

    # combo -> aggregates
    combo_agg = defaultdict(lambda: {
        "all": 0, "usable": 0, "success": 0, "failure": 0,
        "task_usable": Counter(), "task_all": Counter(),
        "task_success": Counter(), "task_failure": Counter(),
        "steps_total": 0, "steps_n": 0,
    })
    # (model, task) and (agent, task) usable task sets for provenance/connectivity
    model_task_usable = defaultdict(set)
    agent_task_usable = defaultdict(set)
    combo_task_usable_set = defaultdict(set)

    rows_out = (OUT / "scan_rows.jsonl").open("w", encoding="utf-8")

    for shard in shards:
        pf = pq.ParquetFile(shard)
        cols = [c for c in COLS if c in set(pf.schema_arrow.names)]
        for batch in pf.iter_batches(batch_size=1000, columns=cols):
            d = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                total += 1
                task = d["task_name"][i]
                agent = d["agent"][i]
                model = d["model"][i]
                reward = d["reward"][i]
                tid = d["trial_id"][i]
                tname = d["trial_name"][i]
                raw_steps = d["steps"][i]

                tasks_all.add(task)
                models_all.add(model)
                agents_all.add(agent)
                combo = (model, agent)
                combos_all.add(combo)

                if tid is not None:
                    trial_id_rows += 1
                    if tid == "":
                        trial_id_empty_rows += 1
                    else:
                        trial_id_nonempty_rows += 1
                        trial_ids.add(tid)
                if tname is not None:
                    trial_name_ids.add(tname)

                reward_raw[reward if reward is not None else "<null>"] += 1

                # ---- reward conditions ----
                if reward is None:
                    reward_missing += 1
                    flag_counts["MISSING_REWARD"] += 1
                    reward_ok = False
                    reward_binary = False
                elif reward in (0, 1):
                    flag_counts["BINARY_REWARD"] += 1
                    reward_ok = True
                    reward_binary = True
                else:
                    reward_nonbinary += 1
                    flag_counts["NON_BINARY_REWARD"] += 1
                    reward_ok = True
                    reward_binary = False

                # ---- step conditions ------------------------------------
                parsed = None
                parse_fail = False
                not_list = False
                if raw_steps is None or raw_steps == "":
                    flag_counts["NO_STEPS"] += 1
                    steps_present = False
                    steps_state = "NULL_FIELD" if raw_steps is None else "EMPTY_STRING"
                else:
                    try:
                        parsed = json.loads(raw_steps)
                    except Exception:  # noqa: BLE001
                        parse_fail = True
                    if parse_fail:
                        schema_fail_rows += 1
                        flag_counts["SCHEMA_FAIL"] += 1
                        steps_present = False
                        steps_state = "PARSE_FAIL"
                    elif parsed is None:
                        # literal JSON `null` == no trajectory recorded
                        flag_counts["NO_STEPS"] += 1
                        steps_present = False
                        steps_state = "JSON_NULL"
                    elif not isinstance(parsed, list):
                        not_list = True
                        schema_fail_rows += 1
                        flag_counts["SCHEMA_FAIL"] += 1
                        steps_present = False
                        steps_state = "NOT_LIST"
                    elif len(parsed) == 0:
                        empty_step_lists += 1
                        flag_counts["EMPTY_STEPS"] += 1
                        steps_present = False
                        steps_state = "EMPTY_LIST"
                    else:
                        steps_present = True
                        flag_counts["HAS_STEPS"] += 1
                        steps_state = "OK"
                steps_state_counts[steps_state] += 1

                if steps_present:
                    rows_with_steps += 1
                else:
                    rows_without_steps += 1

                usable = bool(reward_binary and steps_present)

                # ---- step-schema inventory (usable rows only) ----
                slen = None
                if usable:
                    slen = len(parsed)
                    if tid:
                        usable_with_nonempty_trial_id += 1
                    step_len_hist[slen] += 1
                    steps_len_usable.append(slen)
                    combo_agg[combo]["steps_total"] += slen
                    combo_agg[combo]["steps_n"] += 1
                    first = parsed[0]
                    if isinstance(first, dict):
                        if (first.get("src") == "user"
                                and isinstance(first.get("msg"), str)
                                and "warmup" in first["msg"].lower()):
                            warmup_first += 1
                        if "msg" in first and not first.get("msg"):
                            empty_first_msg += 1
                    for st in parsed:
                        if not isinstance(st, dict):
                            continue
                        for k in st.keys():
                            step_keys[k] += 1
                        src_values[str(st.get("src"))] += 1
                        if st.get("msg") is None:
                            null_msg += 1
                        if st.get("obs") is None:
                            null_obs += 1
                        tl = st.get("tools")
                        if tl is None:
                            null_tools += 1
                        elif isinstance(tl, list):
                            for t in tl:
                                if isinstance(t, dict) and t.get("fn") is not None:
                                    tool_fn_values[str(t["fn"])] += 1
                # ---- classify exclusive primary reason --------------------
                # Deterministic priority (documented in PROTOCOL.md), following
                # the Section-2 condition order outcome -> binary -> steps ->
                # schema. Independent per-condition tallies are kept verbatim in
                # `condition_flags`, so no failure mode is hidden by this label.
                if usable:
                    reason = "TRAJECTORY_USABLE"
                elif reward is None:
                    reason = "MISSING_REWARD"
                elif not reward_binary:
                    reason = "NON_BINARY_REWARD"
                elif parse_fail or not_list:
                    reason = "SCHEMA_FAIL"
                elif not steps_present:
                    reason = "NO_STEPS"
                else:
                    reason = "OTHER"
                reasons[reason] += 1

                # ---- accumulate ----
                ca = combo_agg[combo]
                ca["all"] += 1
                ca["task_all"][task] += 1
                if usable:
                    combos_usable.add(combo)
                    tasks_usable.add(task)
                    models_usable.add(model)
                    agents_usable.add(agent)
                    ca["usable"] += 1
                    ca["task_usable"][task] += 1
                    combo_task_usable_set[combo].add(task)
                    model_task_usable[model].add(task)
                    agent_task_usable[agent].add(task)
                    if reward == 1:
                        ca["success"] += 1
                        ca["task_success"][task] += 1
                    else:
                        ca["failure"] += 1
                        ca["task_failure"][task] += 1

                rows_out.write(json.dumps({
                    "shard": shard.name, "row": total,
                    "task_name": task, "agent": agent, "model": model,
                    "reward": reward, "trial_id": tid,
                    "usable": usable, "reason": reason,
                    "reward_binary": reward_binary,
                    "steps_present": steps_present,
                    "n_steps": slen,
                    "started_at": d["started_at"][i],
                    "ended_at": d["ended_at"][i],
                }, ensure_ascii=False) + "\n")

    rows_out.close()

    # ---- per-combo summary ---------------------------------------------------
    combo_rows = []
    for (model, agent), ca in sorted(combo_agg.items(), key=lambda kv: kv[0]):
        t_usable = sorted(ca["task_usable"].values())
        n_task = len(ca["task_usable"])
        combo_rows.append({
            "model": model, "agent": agent,
            "all_trials": ca["all"],
            "usable_trajectories": ca["usable"],
            "usable_fraction": round(ca["usable"] / ca["all"], 6) if ca["all"] else None,
            "unique_tasks_usable": n_task,
            "unique_tasks_all": len(ca["task_all"]),
            "successes": ca["success"],
            "failures": ca["failure"],
            "success_rate": round(ca["success"] / ca["usable"], 6) if ca["usable"] else None,
            "trials_per_task_mean": round(statistics.fmean(t_usable), 6) if t_usable else None,
            "trials_per_task_median": round(statistics.median(t_usable), 6) if t_usable else None,
            "trials_per_task_min": min(t_usable) if t_usable else None,
            "trials_per_task_max": max(t_usable) if t_usable else None,
            "mean_steps": round(ca["steps_total"] / ca["steps_n"], 6) if ca["steps_n"] else None,
        })

    agg = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shards": shard_info,
        "file_manifest": file_manifest,
        "total_rows": total,
        "rows_with_steps": rows_with_steps,
        "rows_without_steps": rows_without_steps,
        "empty_step_lists": empty_step_lists,
        "schema_fail_rows": schema_fail_rows,
        "classification_reasons": dict(reasons),
        "condition_flags": dict(flag_counts),
        "steps_state_counts": dict(steps_state_counts),
        "reward_raw_values": {str(k): v for k, v in reward_raw.items()},
        "reward_missing": reward_missing,
        "reward_nonbinary": reward_nonbinary,
        "unique_task_names_all": len(tasks_all),
        "unique_task_names_usable": len(tasks_usable),
        "unique_models_all": len(models_all),
        "unique_models_usable": len(models_usable),
        "unique_agents_all": len(agents_all),
        "unique_agents_usable": len(agents_usable),
        "unique_combos_all": len(combos_all),
        "unique_combos_usable": len(combos_usable),
        "unique_trial_ids": len(trial_ids),
        "trial_id_rows": trial_id_rows,
        "trial_id_empty_rows": trial_id_empty_rows,
        "trial_id_nonempty_rows": trial_id_nonempty_rows,
        "usable_with_nonempty_trial_id": usable_with_nonempty_trial_id,
        "unique_trial_names": len(trial_name_ids),
        "warmup_first_step_rows": warmup_first,
        "null_msg_steps": null_msg,
        "null_obs_steps": null_obs,
        "null_tools_steps": null_tools,
        "step_keys": {str(k): v for k, v in step_keys.most_common()},
        "src_values": {str(k): v for k, v in src_values.most_common()},
        "tool_fn_values_top": {str(k): v for k, v in tool_fn_values.most_common(50)},
        "steps_per_trajectory_usable": {
            "n": len(steps_len_usable),
            "min": min(steps_len_usable) if steps_len_usable else None,
            "p25": fsum(pct(steps_len_usable, 0.25)),
            "median": fsum(pct(steps_len_usable, 0.50)),
            "p75": fsum(pct(steps_len_usable, 0.75)),
            "p90": fsum(pct(steps_len_usable, 0.90)),
            "max": max(steps_len_usable) if steps_len_usable else None,
            "mean": fsum(statistics.fmean(steps_len_usable)) if steps_len_usable else None,
        },
        "step_len_histogram": {str(k): v for k, v in sorted(step_len_hist.items())},
        "wall_clock_seconds": None,
        "combo_count": len(combo_rows),
    }
    agg["wall_clock_seconds"] = round(time.time() - t0, 1)

    (OUT / "scan_aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), "utf-8")
    (OUT / "scan_combo_rows.json").write_text(
        json.dumps(combo_rows, ensure_ascii=False, indent=2), "utf-8")

    # aux sets for downstream connectivity/provenance
    aux = {
        "model_task_usable": {k: sorted(v) for k, v in model_task_usable.items()},
        "agent_task_usable": {k: sorted(v) for k, v in agent_task_usable.items()},
        "combo_task_usable": {
            f"{m}\u241f{a}": sorted(t)
            for (m, a), t in sorted(combo_task_usable_set.items())},
        "tasks_all": sorted(tasks_all),
        "tasks_usable": sorted(tasks_usable),
        "models_all": sorted(models_all),
        "agents_all": sorted(agents_all),
    }
    (OUT / "scan_aux_sets.json").write_text(
        json.dumps(aux, ensure_ascii=False, indent=2), "utf-8")

    print(json.dumps({k: v for k, v in agg.items()
                      if k not in ("step_keys", "src_values",
                                   "step_len_histogram", "file_manifest",
                                   "tool_fn_values_top", "shards")}, indent=2))
    print("rows:", total, "elapsed:", agg["wall_clock_seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
