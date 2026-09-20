# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - Sections 10/12/13/22: leave-one-model-out fold driver.

For N eligible exact models under the fixed scaffold ``terminus-2`` this runs N
folds. Fold m holds out target model m entirely: it never enters the fold-local
FeatureEngineer fit, LightGBM training or calibrator fit. Each fold runs in a
fresh subprocess so peak memory is released between folds; a failing fold is
recorded as FOLD_FAIL and the remaining frozen folds still run. No hyperparameter
rescue. Offline only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from common import (ANA, FOLDS, OUT, PRED, WORK, ensure_dirs, read_json,
                    set_vendor_env, write_json)

set_vendor_env()

from earlyeval.core.contracts import PolicySpec  # noqa: E402
from earlyeval.policies.safe_stop import apply_policy  # noqa: E402

THREADS = 6
PY = sys.executable  # use the interpreter running this script
PREDICTOR = "P2B_TerminalBench_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
POLICY = {
    "name": "current_safe_stop",
    "predictor": PREDICTOR,
    "score_mode": "calibrated",
    "policy_mode": "dual",
    "success_thr": 0.95,
    "failure_thr": 0.95,
    "min_step": 0,
    "consecutive": 1,
}

PRED_DIR = PRED / "heldout_prefix_predictions"
DEC_DIR = PRED / "policy_decisions"
WORK_FOLDS = WORK / "folds"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text: str, n: int = 500) -> str:
    return (text or "")[-n:]


def run_step(script: str, args: list) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run([PY, script] + [str(a) for a in args],
                          cwd=str(WORK), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return {
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - t0, 1),
        "stdout_tail": _tail(proc.stdout, 4000),
        "stderr_tail": _tail(proc.stderr, 4000),
    }


def policy_spec() -> PolicySpec:
    return PolicySpec(**POLICY)


def _first(shape, key, idx):
    try:
        return shape[key][idx]
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    t_all = time.time()
    ensure_dirs(OUT, ANA, FOLDS, PRED, PRED_DIR, DEC_DIR, WORK_FOLDS)
    pre = read_json(WORK / "preflight.json")
    models = sorted(pre["eligible_models"])
    print("eligible models = %d -> %d LOMO folds" % (len(models), len(models)),
          flush=True)

    manifest_rows, meta_rows = [], []
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        print("[%d/%d] %s holdout=%s" % (i + 1, len(models), tag, model),
              flush=True)
        started = _utc()
        t0 = time.time()
        f4 = run_step("step4_fold_features.py", [model, tag])
        status, note = "OK", ""
        if f4["returncode"] != 0:
            status = "FOLD_FAIL"
            note = "step4 rc=%d: %s" % (f4["returncode"], _tail(f4["stderr_tail"]))
        f5 = {"returncode": 0, "seconds": 0.0}
        if status == "OK":
            f5 = run_step("step5_train_fold.py", [tag, THREADS])
            if f5["returncode"] != 0:
                status = "FOLD_FAIL"
                note = "step5 rc=%d: %s" % (
                    f5["returncode"], _tail(f5["stderr_tail"]))

        fold_dir = WORK_FOLDS / tag
        feat_meta = calib_meta = None
        if (fold_dir / "feature_meta.json").exists():
            feat_meta = read_json(fold_dir / "feature_meta.json")
        if (fold_dir / "calibration_meta.json").exists():
            calib_meta = read_json(fold_dir / "calibration_meta.json")

        decisions = None
        if status == "OK":
            try:
                test = pd.read_parquet(
                    fold_dir / "test_prefix_predictions.parquet")
                test.to_parquet(PRED_DIR / (tag + ".parquet"), index=False,
                                compression="zstd")
                decisions, _summary, per_agent = apply_policy(
                    test, policy_spec())
                decisions.insert(0, "fold_tag", tag)
                decisions.insert(1, "holdout_model", model)
                decisions.to_csv(DEC_DIR / (tag + ".csv"), index=False,
                                 encoding="utf-8")
                per_agent.insert(0, "fold_tag", tag)
                per_agent.insert(1, "holdout_model", model)
                per_agent.to_csv(DEC_DIR / (tag + ".per_agent.csv"), index=False,
                                 encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                status = "FOLD_FAIL"
                note = "policy step: %s: %s" % (type(exc).__name__, exc)

        best_s = best_f = None
        for row in (calib_meta or {}).get("calibration", []):
            if row["head"] == "safe_success":
                best_s = row["best_iteration"]
            elif row["head"] == "safe_failure":
                best_f = row["best_iteration"]
        shape = (feat_meta or {}).get("design_matrix_shape", {})

        manifest_rows.append({
            "fold_tag": tag, "fold_index": i, "holdout_model": model,
            "scaffold": "terminus-2", "train_models": len(models) - 1,
        })
        meta_rows.append({
            "fold_tag": tag, "fold_index": i, "holdout_model": model,
            "started_utc": started, "ended_utc": _utc(),
            "wall_clock_seconds": round(time.time() - t0, 1),
            "feature_seconds": (feat_meta or {}).get("seconds"),
            "train_seconds": (calib_meta or {}).get("seconds"),
            "train_rows": _first(shape, "train", 0),
            "valid_rows": _first(shape, "valid", 0),
            "test_rows": _first(shape, "test", 0),
            "selected_features": _first(shape, "train", 1),
            "short_trajectories_dropped_from_trainval": (feat_meta or {}).get(
                "short_trajectories_dropped_from_trainval"),
            "best_iteration_success": best_s,
            "best_iteration_failure": best_f,
            "safe_success_head": "OK" if best_s is not None else "n/a",
            "safe_failure_head": "OK" if best_f is not None else "n/a",
            "calibration_status": (
                "OK" if (best_s is not None and best_f is not None) else "n/a"),
            "fold_status": status,
            "decisions_total": int(len(decisions)) if decisions is not None else 0,
            "decisions_decided": (
                int(decisions["decided"].astype(bool).sum())
                if decisions is not None else 0),
            "notes": note,
        })
        print("    %s seconds=%s train=%s best_iter=(%s,%s)" % (
            status, meta_rows[-1]["wall_clock_seconds"],
            meta_rows[-1]["train_rows"], best_s, best_f), flush=True)

    pd.DataFrame(manifest_rows).to_csv(FOLDS / "fold_manifest.csv", index=False,
                                       encoding="utf-8")
    pd.DataFrame(meta_rows).to_csv(FOLDS / "fold_training_metadata.csv",
                                   index=False, encoding="utf-8")
    write_json(FOLDS / "fold_manifest.json", manifest_rows)
    summary = {
        "section": "10/12/13/22 LOMO EXECUTION",
        "scaffold": "terminus-2",
        "folds_planned": len(models),
        "folds_completed": int(sum(1 for r in meta_rows
                                   if r["fold_status"] == "OK")),
        "folds_failed": int(sum(1 for r in meta_rows
                                if r["fold_status"] != "OK")),
        "policy": POLICY,
        "threads": THREADS,
        "wall_clock_seconds": round(time.time() - t_all, 1),
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "new_agent_generation": 0,
        "held_out_model_never_trained_on": True,
    }
    write_json(WORK / "fold_run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
