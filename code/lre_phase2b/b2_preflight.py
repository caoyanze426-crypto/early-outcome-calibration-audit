# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - Sections 2/4/5/6: input provenance + missingness preflight.

Reads ALL TerminalBench terminus-2 rows (including `steps = null`) straight from
the frozen Phase 2A dataset, recomputes the Section-4 missingness statistics,
applies the Section-5 primary eligibility gate and evaluates the Section-6
minimum-model-count gate. No predictor is trained here.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from b2_common import (ANA, DATA, ELIG_FAILURE, ELIG_MISSINGNESS_GAP,
                       ELIG_SUCCESS, ELIG_USABLE, ELIG_USABLE_FRACTION,
                       MIN_MODELS, OUT, OUT_2A, SCAFFOLD, WORK, as_builtin,
                       ensure_dirs, read_manifest, sha256_file, write_json)

INPUT_FILES = [
    "analysis/phase2a_gate.json",
    "analysis/design_connectivity.json",
    "analysis/target_combo_summary.csv",
    "analysis/trajectory_usability.csv",
    "adapter/adapter_terminalbench.py",
    "adapter/adapter_probe_results.json",
    "dataset/dataset_manifest.json",
]
ROW_COLUMNS = ["task_name", "agent", "model", "reward", "trial_id",
               "trial_name", "steps"]


def verify_inputs() -> dict:
    rec = read_manifest(OUT_2A / "artifact_sha256sums.txt")
    checks, ok = {}, True
    for rel in INPUT_FILES:
        p = OUT_2A / rel
        obs = sha256_file(p) if p.exists() else None
        exp = rec.get(rel)
        match = bool(obs is not None and exp is not None and obs == exp)
        checks[rel] = {"sha256": obs, "manifest_sha256": exp, "match": match}
        ok = ok and match
    shards = {}
    for shard in sorted(DATA.rglob("data/*.parquet")):
        shards[f"data/{shard.name}"] = {
            "sha256": sha256_file(shard), "bytes": shard.stat().st_size}
    return {
        "phase2a_artifact_root": str(OUT_2A),
        "phase2a_manifest": str(OUT_2A / "artifact_sha256sums.txt"),
        "phase2a_manifest_entries": len(rec),
        "artifact_checks": checks,
        "ALL_PHASE2A_INPUT_HASHES_MATCH": bool(ok),
        "dataset_shards": shards,
        "dataset_revision": "04e8940f5b6736a7ce8d22224fe2f2af74163ed2",
    }


def scan_terminus2():
    """Stream every row; retain full per-row detail for the terminus-2 scaffold."""
    rows = []
    seen = 0
    for shard in sorted(DATA.rglob("data/*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2000, columns=ROW_COLUMNS):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                seen += 1
                if d["agent"][i] != SCAFFOLD:
                    continue
                raw = d["steps"][i]
                parsed = None
                if raw not in (None, "", "null"):
                    try:
                        parsed = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        parsed = None
                usable = bool(isinstance(parsed, list) and len(parsed) > 0)
                rows.append({
                    "model": d["model"][i], "agent": d["agent"][i],
                    "task_name": d["task_name"][i], "reward": d["reward"][i],
                    "trial_id": d["trial_id"][i], "trial_name": d["trial_name"][i],
                    "usable": usable,
                    "n_steps": len(parsed) if usable else 0,
                })
    return rows, seen


def missingness(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    audit = []
    for model in sorted(by_model):
        rs = by_model[model]
        usable = [r for r in rs if r["usable"]]
        succ_all = [r for r in rs if r["reward"] == 1]
        fail_all = [r for r in rs if r["reward"] == 0]
        succ_us = [r for r in succ_all if r["usable"]]
        fail_us = [r for r in fail_all if r["usable"]]
        us_frac = len(usable) / len(rs) if rs else None
        succ_frac = len(succ_us) / len(succ_all) if succ_all else None
        fail_frac = len(fail_us) / len(fail_all) if fail_all else None
        gap = (abs(succ_frac - fail_frac)
               if (succ_frac is not None and fail_frac is not None) else None)
        tasks = sorted({r["task_name"] for r in usable})
        counts = [sum(1 for r in usable if r["task_name"] == t) for t in tasks]
        audit.append({
            "model": model, "scaffold": SCAFFOLD,
            "all_trials": len(rs),
            "usable_trajectories": len(usable),
            "usable_fraction": us_frac,
            "successes_all": len(succ_all), "successes_usable": len(succ_us),
            "success_usable_fraction": succ_frac,
            "failures_all": len(fail_all), "failures_usable": len(fail_us),
            "failure_usable_fraction": fail_frac,
            "MISSINGNESS_GAP": gap,
            "usable_unique_tasks": len(tasks),
            "trials_per_task_mean": (round(statistics.fmean(counts), 6)
                                     if counts else None),
            "trials_per_task_median": (round(statistics.median(counts), 6)
                                       if counts else None),
            "trials_per_task_min": min(counts) if counts else None,
            "trials_per_task_max": max(counts) if counts else None,
        })
    return audit


def classify(audit):
    out = []
    for a in audit:
        reasons = []
        if a["usable_trajectories"] < ELIG_USABLE:
            reasons.append("USABLE_LT_100")
        if a["successes_usable"] < ELIG_SUCCESS:
            reasons.append("SUCCESS_USABLE_LT_20")
        if a["failures_usable"] < ELIG_FAILURE:
            reasons.append("FAILURE_USABLE_LT_20")
        if a["usable_fraction"] is None or a["usable_fraction"] < ELIG_USABLE_FRACTION:
            reasons.append("USABLE_FRACTION_LT_0.60")
        if (a["MISSINGNESS_GAP"] is None
                or a["MISSINGNESS_GAP"] > ELIG_MISSINGNESS_GAP):
            reasons.append("MISSINGNESS_GAP_GT_0.20")
        row = dict(a)
        row["primary_eligible"] = not reasons
        row["ineligible_reasons"] = "|".join(reasons)
        out.append(row)
    return out


def main() -> int:
    t0 = time.time()
    ensure_dirs(OUT, ANA, WORK)
    inputs = verify_inputs()
    print("phase2a input hashes match = "
          f"{inputs['ALL_PHASE2A_INPUT_HASHES_MATCH']}", flush=True)
    write_json(OUT / "input_hashes.json", {
        **inputs,
        "predictor_retraining": 0, "new_lightgbm_training": 0,
        "new_trajectories": 0, "new_datasets": 0,
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
    })

    rows, seen = scan_terminus2()
    audit = classify(missingness(rows))
    eligible = [a for a in audit if a["primary_eligible"]]
    ineligible = [a for a in audit if not a["primary_eligible"]]

    fields = list(audit[0].keys())
    with (ANA / "missingness_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for a in audit:
            w.writerow(a)

    # cross-check usable counts against the frozen Phase 2A combo summary
    import pandas as pd

    tcs = pd.read_csv(OUT_2A / "analysis" / "target_combo_summary.csv")
    tcs = tcs[tcs["agent"] == SCAFFOLD]
    frozen = {str(r.model): int(r.usable_trajectories) for r in tcs.itertuples()}
    cross = {a["model"]: {
        "phase2b_usable": a["usable_trajectories"],
        "phase2a_usable": frozen.get(a["model"]),
        "match": a["usable_trajectories"] == frozen.get(a["model"]),
    } for a in audit}
    cross_ok = (all(v["match"] for v in cross.values())
                and len(cross) == len(frozen))

    stopped = len(eligible) < MIN_MODELS
    payload = {
        "section": "2/4/5/6 INPUT PROVENANCE + MISSINGNESS PREFLIGHT",
        "scaffold": SCAFFOLD,
        "dataset_revision": "04e8940f5b6736a7ce8d22224fe2f2af74163ed2",
        "earlyeval_commit": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
        "all_rows_seen_in_dataset": seen,
        "terminus2_rows": len(rows),
        "terminus2_raw_target_models": len(audit),
        "terminus2_usable_trajectories": sum(a["usable_trajectories"] for a in audit),
        "terminus2_usable_unique_tasks": len({r["task_name"] for r in rows if r["usable"]}),
        "primary_eligible_models": len(eligible),
        "missingness_ineligible_models": len(ineligible),
        "minimum_model_gate": MIN_MODELS,
        "stopped_before_training": stopped,
        "STATUS": ("INSUFFICIENT_FIXED_SCAFFOLD_MODEL_SUPPORT" if stopped
                   else "PROCEED_TO_TRAINING"),
        "eligibility_rule": {
            "usable_trajectories>=": ELIG_USABLE,
            "successes_usable>=": ELIG_SUCCESS,
            "failures_usable>=": ELIG_FAILURE,
            "usable_fraction>=": ELIG_USABLE_FRACTION,
            "MISSINGNESS_GAP<=": ELIG_MISSINGNESS_GAP,
            "note": "thresholds frozen before training; not relaxed after "
                    "inspecting calibration results",
        },
        "phase2a_usable_count_cross_check": cross,
        "phase2a_usable_count_cross_check_all_match": bool(cross_ok),
        "eligible_models": [a["model"] for a in eligible],
        "ineligible_detail": [
            {"model": a["model"], "reasons": a["ineligible_reasons"],
             "usable": a["usable_trajectories"],
             "usable_fraction": a["usable_fraction"],
             "MISSINGNESS_GAP": a["MISSINGNESS_GAP"]}
            for a in ineligible],
        "wall_clock_seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "predictor_training": 0,
    }
    write_json(ANA / "eligible_models.json", payload)
    write_json(WORK / "preflight.json", payload)

    print(json.dumps({
        "terminus2_raw_target_models": len(audit),
        "primary_eligible_models": len(eligible),
        "usable_trajectories": payload["terminus2_usable_trajectories"],
        "stopped_before_training": stopped,
        "cross_check_all_match": cross_ok,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
