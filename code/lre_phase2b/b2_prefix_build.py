# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - Section 7/11: frozen adapter -> step table -> prefix table.

Every usable ``terminus-2`` trajectory of the 29 missingness-eligible exact
models is pushed through the frozen Phase 2A deterministic adapter and the
unmodified vendored ``step_builder`` / ``prefix_builder``. No manual repair, no
downsampling, no model trained here. Offline only.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

from common import (ADAPTER_DIR, WORK, ensure_dirs, read_json, set_vendor_env,
                    write_json)

set_vendor_env()

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from adapter_terminalbench import to_vendor_record  # noqa: E402
from prefix_builder import build_prefix_samples_for_trajectory  # noqa: E402
from step_builder import rebuild_steps_for_trajectory  # noqa: E402

DATA = WORK.parent / "lre_phase2a" / "data"
PREFLIGHT = WORK / "preflight.json"
PREFIX_DIR = WORK / "prefix_table"
PART_ROWS = 20000
ROW_COLUMNS = ["task_name", "agent", "model", "reward", "trial_id",
               "trial_name", "steps"]


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> int:
    t0 = time.time()
    ensure_dirs(PREFIX_DIR)
    for old in PREFIX_DIR.glob("prefix_table.part-*.parquet"):
        old.unlink()

    pre = read_json(PREFLIGHT)
    eligible = set(pre["eligible_models"])
    print(f"eligible models = {len(eligible)}", flush=True)

    shard_files = sorted(DATA.rglob("data/*.parquet"))
    print(f"shards = {[p.name for p in shard_files]}", flush=True)

    buffer: list[dict] = []
    part_idx = 0
    traj_rows = []
    failures = []
    per_model: dict[str, dict] = {}
    step_lengths = []
    ordinal = 0
    candidate_rows = 0

    def flush():
        nonlocal buffer, part_idx
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        df.to_parquet(PREFIX_DIR / f"prefix_table.part-{part_idx:04d}.parquet",
                      index=False, compression="zstd")
        part_idx += 1
        print(f"  flushed part {part_idx - 1} rows={len(df)} "
              f"trajectories={len(traj_rows)} elapsed={time.time() - t0:.0f}s",
              flush=True)
        buffer = []

    prefix_rows = 0
    for shard in shard_files:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2000, columns=ROW_COLUMNS):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                model = d["model"][i]
                if d["agent"][i] != "terminus-2" or model not in eligible:
                    continue
                candidate_rows += 1
                idx = ordinal
                ordinal += 1
                raw_steps = d["steps"][i]
                if raw_steps in (None, "", "null"):
                    continue
                try:
                    parsed = json.loads(raw_steps)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(parsed, list) or not parsed:
                    continue
                row = {
                    "task_name": d["task_name"][i], "model": model,
                    "reward": d["reward"][i], "trial_id": d["trial_id"][i],
                    "trial_name": d["trial_name"][i], "steps": parsed,
                }
                rec = to_vendor_record(row, idx)
                vrow = pd.Series({
                    "traj_id": rec["traj_id"], "instance_id": rec["instance_id"],
                    "resolved": rec["resolved"], "model": rec["model"],
                    "messages": json.dumps(rec["messages"], ensure_ascii=False),
                })
                try:
                    steps = rebuild_steps_for_trajectory(vrow)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"traj_id": rec["traj_id"],
                                     "stage": "step_builder",
                                     "reason": _reason(exc)})
                    continue
                if not steps:
                    failures.append({"traj_id": rec["traj_id"],
                                     "stage": "step_builder",
                                     "reason": "NO_ACTION_MESSAGES_ZERO_STEPS"})
                    continue
                try:
                    samples = build_prefix_samples_for_trajectory(
                        vrow, pd.DataFrame(steps))
                except Exception as exc:  # noqa: BLE001
                    failures.append({"traj_id": rec["traj_id"],
                                     "stage": "prefix_builder",
                                     "reason": _reason(exc)})
                    continue
                buffer.extend(samples)
                prefix_rows += len(samples)
                step_lengths.append(len(steps))
                st = per_model.setdefault(
                    model, {"trajectories": 0, "success": 0, "failure": 0,
                            "prefix_rows": 0, "steps": 0})
                st["trajectories"] += 1
                st["success" if rec["resolved"] else "failure"] += 1
                st["prefix_rows"] += len(samples)
                st["steps"] += len(steps)
                traj_rows.append({
                    "traj_id": rec["traj_id"], "instance_id": rec["instance_id"],
                    "model": model, "resolved": rec["resolved"],
                    "n_steps": len(steps), "n_prefix_rows": len(samples),
                    "trial_id": d["trial_id"][i]})
                if len(buffer) >= PART_ROWS:
                    flush()
        del pf
    flush()

    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(WORK / "trajectory_index.csv", index=False, encoding="utf-8")
    if failures:
        pd.DataFrame(failures).to_csv(WORK / "adapter_failures.csv", index=False,
                                      encoding="utf-8")
    else:
        (WORK / "adapter_failures.csv").write_text(
            "traj_id,stage,reason\n", "utf-8")

    lengths = sorted(int(v) for v in step_lengths)
    def _pct(p):
        if not lengths:
            return None
        k = min(len(lengths) - 1, int(round(p / 100.0 * (len(lengths) - 1))))
        return lengths[k]

    unique = traj_df["traj_id"].nunique()
    report = {
        "section": "7/11 TASK SUPPORT + PREFIX CONSTRUCTION",
        "scaffold": "terminus-2",
        "eligible_models": sorted(eligible),
        "eligible_model_count": len(eligible),
        "candidate_rows_scanned": int(candidate_rows),
        "adapter_pass_trajectories": int(len(traj_df)),
        "adapter_fail": int(len(failures)),
        "traj_ids_unique": int(unique),
        "traj_ids_duplicate": int(len(traj_df) - unique),
        "prefix_rows": int(prefix_rows),
        "success_trajectories": int((traj_df["resolved"]).sum()),
        "failure_trajectories": int((~traj_df["resolved"]).sum()),
        "usable_unique_tasks": int(traj_df["instance_id"].nunique()),
        "trajectory_length": {
            "min": lengths[0] if lengths else None,
            "p25": _pct(25), "median": _pct(50), "p75": _pct(75),
            "p90": _pct(90), "max": lengths[-1] if lengths else None,
            "mean": round(statistics.fmean(lengths), 6) if lengths else None,
        },
        "trials_per_task": {
            "mean": round(statistics.fmean(
                traj_df.groupby("instance_id").size().tolist()), 6),
            "median": statistics.median(
                traj_df.groupby("instance_id").size().tolist()),
            "max": int(traj_df.groupby("instance_id").size().max()),
        },
        "per_model": {m: per_model[m] for m in sorted(per_model)},
        "adapter": "phase2a deterministic adapter + unmodified vendored "
                   "step_builder/prefix_builder (imported from Phase 2A artifact)",
        "adapter_dir": str(ADAPTER_DIR),
        "no_manual_repair": True,
        "no_downsampling": True,
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "new_trajectories_generated": 0,
        "predictor_training": 0,
    }
    write_json(WORK / "prefix_build_report.json", report)
    print(json.dumps({k: report[k] for k in (
        "adapter_pass_trajectories", "adapter_fail", "prefix_rows",
        "usable_unique_tasks", "seconds")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
