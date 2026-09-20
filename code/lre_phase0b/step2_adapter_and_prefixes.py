# -*- coding: utf-8 -*-
"""Phase 0B step 2: deterministic adapter, frozen step table, frozen prefix table. Offline only."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from common import OUT, WORK, ensure_dirs, read_json, set_vendor_env, write_json

set_vendor_env()

import pandas as pd  # noqa: E402

from mapping_adapter import to_contract  # noqa: E402
from prefix_builder import build_prefix_samples_for_trajectory  # noqa: E402
from step_builder import rebuild_steps_for_trajectory  # noqa: E402

STEPS_DIR = WORK / "step_table"
PREFIX_DIR = WORK / "prefix_table"
ADAPTER_DIR = OUT / "adapter"
TRAJ_CHUNK = 100


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _validate_fields(doc) -> str | None:
    if not isinstance(doc, dict):
        return "NOT_A_JSON_OBJECT"
    if not doc.get("instance_id"):
        return "MISSING_INSTANCE_ID"
    if not isinstance(doc.get("messages"), list) or not doc.get("messages"):
        return "MISSING_OR_EMPTY_MESSAGES"
    info = doc.get("info")
    if not isinstance(info, dict):
        return "MISSING_INFO"
    if not isinstance(info.get("resolved"), bool):
        return "MISSING_RESOLVED_BOOL"
    if not isinstance(info.get("docent"), dict) or not info["docent"].get("model_label"):
        return "MISSING_MODEL_LABEL"
    return None


def main() -> int:
    ensure_dirs(STEPS_DIR, PREFIX_DIR, ADAPTER_DIR, OUT / "features")
    dataset = read_json(OUT / "dataset" / "dataset_manifest.json")
    root = Path(dataset["snapshot_subdir"])
    files = sorted(root.rglob("*.traj.json"))
    failures = []
    counts = {"raw_trajectories": len(files)}
    traj_rows = []
    n_pass = 0
    t0 = time.time()
    state = {"step_part": 0, "prefix_part": 0, "steps": 0, "prefix_rows": 0}
    per_model = {}

    def flush(chunk):
        if not chunk:
            return
        step_frames, prefix_frames = [], []
        for rec, steps in chunk:
            step_frames.append(pd.DataFrame(steps))
            row = pd.Series({
                "traj_id": rec["traj_id"], "instance_id": rec["instance_id"],
                "model": rec["model_id"], "resolved": rec["resolved"],
                "messages": json.dumps(rec["messages"], ensure_ascii=False)})
            prefix_frames.extend(build_prefix_samples_for_trajectory(
                row, step_frames[-1]))
        sd = pd.concat(step_frames, ignore_index=True)
        sd.to_parquet(STEPS_DIR / f"step_table.part-{state['step_part']:04d}.parquet",
                      index=False, compression="zstd")
        state["steps"] += int(len(sd))
        state["step_part"] += 1
        del sd, step_frames
        pd.DataFrame(prefix_frames).to_parquet(
            PREFIX_DIR / f"prefix_table.part-{state['prefix_part']:04d}.parquet",
            index=False, compression="zstd")
        state["prefix_rows"] += len(prefix_frames)
        state["prefix_part"] += 1
        del prefix_frames

    chunk = []
    for i, f in enumerate(files, 1):
        rel = f.relative_to(root).as_posix()
        model = rel.split("/")[0]
        try:
            doc = json.loads(f.read_text("utf-8"))
        except Exception as exc:
            failures.append({"file": rel, "status": "FAIL", "stage": "read_json",
                             "reason": _reason(exc)})
            continue
        bad = _validate_fields(doc)
        if bad:
            failures.append({"file": rel, "status": "FAIL", "stage": "field_validation",
                             "reason": bad})
            continue
        info = doc["info"]
        rec = to_contract(doc, info["docent"]["model_label"])
        row = pd.Series({
            "traj_id": rec["traj_id"], "instance_id": rec["instance_id"],
            "resolved": rec["resolved"], "model": rec["model_id"],
            "messages": json.dumps(rec["messages"], ensure_ascii=False),
        })
        try:
            steps = rebuild_steps_for_trajectory(row)
        except Exception as exc:
            failures.append({"file": rel, "status": "FAIL", "stage": "step_builder",
                             "reason": _reason(exc),
                             "traceback": traceback.format_exc()[:2000]})
            continue
        if not steps:
            failures.append({"file": rel, "status": "FAIL", "stage": "step_builder",
                             "reason": "NO_ACTION_MESSAGES_ZERO_STEPS"})
            continue
        n_pass += 1
        st = per_model.setdefault(model, {"n": 0, "success": 0, "failure": 0, "steps": 0})
        st["n"] += 1
        st["success" if rec["resolved"] else "failure"] += 1
        st["steps"] += len(steps)
        traj_rows.append({"traj_id": rec["traj_id"], "instance_id": rec["instance_id"],
                          "model": model, "resolved": rec["resolved"],
                          "n_steps": len(steps), "file": rel})
        chunk.append((rec, steps))
        if len(chunk) >= TRAJ_CHUNK:
            flush(chunk)
            chunk = []
            print(f"  [{i}/{len(files)}] trajs={n_pass} step_rows={state['steps']} "
                  f"prefix_rows={state['prefix_rows']} elapsed={time.time() - t0:.0f}s",
                  flush=True)
    flush(chunk)
    counts["adapter_pass"] = n_pass
    counts["adapter_fail"] = len(failures)
    counts["step_rows"] = state["steps"]
    counts["prefix_rows"] = state["prefix_rows"]
    counts["elapsed_sec"] = round(time.time() - t0, 1)
    counts["per_model"] = per_model
    fail_df = pd.DataFrame(failures, columns=["file", "status", "stage", "reason"])
    fail_df.to_csv(ADAPTER_DIR / "adapter_failures.csv", index=False, encoding="utf-8")
    pd.DataFrame(traj_rows).to_csv(WORK / "trajectory_index.csv", index=False,
                                   encoding="utf-8")
    write_json(ADAPTER_DIR / "adapter_spec.json", {
        "adapter": "phase0a deterministic field adapter + unmodified vendored step_builder",
        "source": "work/lre_phase0a/mapping_adapter.py (unchanged, imported by path)",
        "normalization_required": ("mini-SWE-agent tool_calls[].function is a string with a "
                                   "sibling 'arguments' key; rewritten to OpenAI-style "
                                   "{name, arguments} for the vendored step builder"),
        "no_semantic_rewriting": True,
        "no_per_trajectory_repair": True,
        "validation_rules": ["json_parse_ok", "instance_id_non_empty",
                             "messages_non_empty_list", "info.resolved_is_bool",
                             "info.docent.model_label_present",
                             "step_builder_yields_at_least_one_step"],
        "failure_stages": ["read_json", "field_validation", "step_builder"],
    })
    write_json(WORK / "adapter_consort_counts.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
