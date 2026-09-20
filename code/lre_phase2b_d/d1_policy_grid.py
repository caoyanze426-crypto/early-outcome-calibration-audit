# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B_D - Section 2: threshold grid policy re-scan.

Re-applies the frozen vendored safe-stop policy to the frozen Phase 2B
held-out prefix predictions for success_thr = failure_thr in {0.900, 0.925,
0.950}. The 0.950 run must reproduce the frozen Phase 2B policy decisions.
No predictor is retrained; no prediction is recomputed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from d_common import (ANCHOR, B2, DEC_2B, DECS, DEC_TOL, OUT_2B, PRED_2B,
                      PREDICTOR, THRESHOLDS, WORK, anchor_tag, ensure_dirs,
                      read_json, set_vendor_env, thr_tag, write_json)

set_vendor_env()

from earlyeval.core.contracts import PolicySpec  # noqa: E402
from earlyeval.policies.safe_stop import apply_policy  # noqa: E402


def policy_spec(thr: float) -> PolicySpec:
    return PolicySpec(
        name="current_safe_stop", predictor=PREDICTOR,
        score_mode="calibrated", policy_mode="dual",
        success_thr=float(thr), failure_thr=float(thr),
        min_step=0, consecutive=1)


def run_threshold(thr: float, models: list) -> dict:
    tdir = DECS / thr_tag(thr)
    ensure_dirs(tdir)
    decided = total = n_success = n_failure = 0
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        frame = pd.read_parquet(PRED_2B / (tag + ".parquet"))
        decisions, _summary, _per_agent = apply_policy(
            frame, policy_spec(thr))
        decisions.insert(0, "fold_tag", tag)
        decisions.insert(1, "holdout_model", model)
        decisions.to_csv(tdir / (tag + ".csv"), index=False,
                         encoding="utf-8")
        dec = decisions["decided"].astype(bool)
        decided += int(dec.sum())
        total += int(len(decisions))
        n_success += int((decisions.loc[dec, "decision"] == "success").sum())
        n_failure += int((decisions.loc[dec, "decision"] == "failure").sum())
    return {
        "threshold": float(thr), "trajectories": int(total),
        "decided": int(decided), "undecided": int(total - decided),
        "decided_success": int(n_success),
        "decided_failure": int(n_failure),
        "coverage": (decided / total) if total else None,
        "decisions_dir": str(tdir),
    }


def reproduction(models: list) -> dict:
    """Decision-level reproduction of the frozen Phase 2B 0.950 decisions."""
    tdir = DECS / anchor_tag()
    per_fold = []
    n_traj = n_dec_field = n_dec_flag = n_step = n_decision = 0
    max_score_diff = 0.0
    order_identical = True
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        old = pd.read_csv(DEC_2B / (tag + ".csv"))
        new = pd.read_csv(tdir / (tag + ".csv"))
        row = {"fold_tag": tag, "holdout_model": model,
               "frozen_rows": int(len(old)), "rederived_rows": int(len(new))}
        if list(old["traj_id"]) != list(new["traj_id"]):
            order_identical = False
        old = old.sort_values("traj_id").reset_index(drop=True)
        new = new.sort_values("traj_id").reset_index(drop=True)
        row["traj_index_match"] = bool(list(old["traj_id"]) == list(new["traj_id"]))
        neq = int((old["decided"].astype(bool)
                   != new["decided"].astype(bool)).sum())
        row["decided_flag_mismatches"] = neq
        both = old["decided"].astype(bool) & new["decided"].astype(bool)
        row["decided_rows"] = int(both.sum())
        row["decision_mismatches"] = int(
            (old.loc[both, "decision"] != new.loc[both, "decision"]).sum())
        row["decision_step_mismatches"] = int(
            (old.loc[both, "decision_step"].astype(np.int64)
             != new.loc[both, "decision_step"].astype(np.int64)).sum())
        if int(both.sum()):
            diff = np.abs(old.loc[both, "decision_score"].to_numpy(float)
                          - new.loc[both, "decision_score"].to_numpy(float))
            row["max_abs_decision_score_diff"] = float(np.nanmax(diff))
        else:
            row["max_abs_decision_score_diff"] = None
        row["saved_steps_mismatches"] = int(
            (old["saved_steps"].astype(np.int64)
             != new["saved_steps"].astype(np.int64)).sum())
        row["reproduced"] = bool(
            row["traj_index_match"] and neq == 0
            and row["decision_mismatches"] == 0
            and row["decision_step_mismatches"] == 0
            and row["saved_steps_mismatches"] == 0
            and (row["max_abs_decision_score_diff"] or 0.0) <= DEC_TOL)
        per_fold.append(row)
        n_traj += 0 if row["traj_index_match"] else 1
        n_dec_flag += neq
        n_decision += row["decision_mismatches"]
        n_step += row["decision_step_mismatches"]
        n_dec_field += row["saved_steps_mismatches"]
        max_score_diff = max(max_score_diff,
                             row["max_abs_decision_score_diff"] or 0.0)
    total_rows = sum(r["frozen_rows"] for r in per_fold)
    decided_rows = sum(r["decided_rows"] for r in per_fold)
    passed = all(r["reproduced"] for r in per_fold)
    return {
        "anchor_threshold": ANCHOR,
        "comparison": "re-derived 0.950 decisions vs frozen Phase 2B "
                      "predictions/policy_decisions/fold-NN.csv",
        "tolerance_decision_score": DEC_TOL,
        "folds": len(per_fold),
        "total_rows": int(total_rows),
        "decided_rows": int(decided_rows),
        "row_sequence_identical_to_frozen_csv": bool(order_identical),
        "traj_id_mismatched_folds": int(n_traj),
        "decided_flag_mismatches": int(n_dec_flag),
        "decision_mismatches": int(n_decision),
        "decision_step_mismatches": int(n_step),
        "saved_steps_mismatches": int(n_dec_field),
        "max_abs_decision_score_diff": float(max_score_diff),
        "REPRODUCTION_PASS": bool(passed),
        "per_fold": per_fold,
    }


def main() -> int:
    t0 = time.time()
    ensure_dirs(WORK, DECS)
    pre = read_json(B2 / "preflight.json")
    models = sorted(pre["eligible_models"])
    print("eligible models = %d" % len(models), flush=True)
    summary = {}
    for thr in THRESHOLDS:
        info = run_threshold(thr, models)
        summary["%.3f" % thr] = info
        print("%.3f decided=%d/%d (success=%d failure=%d)"
              % (thr, info["decided"], info["trajectories"],
                 info["decided_success"], info["decided_failure"]), flush=True)
    repro = reproduction(models)
    write_json(WORK / "policy_grid_summary.json", {
        "section": "2 THRESHOLD GRID",
        "thresholds": list(THRESHOLDS),
        "policy": {"success_thr": "t", "failure_thr": "t",
                   "policy_mode": "dual", "score_mode": "calibrated",
                   "min_step": 0, "consecutive": 1,
                   "predictor": PREDICTOR},
        "source": str(PRED_2B),
        "decisions_written_to": str(DECS),
        "per_threshold": summary,
        "reproduction": repro,
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
    })
    print("REPRODUCTION_PASS = %s" % repro["REPRODUCTION_PASS"], flush=True)
    return 0 if repro["REPRODUCTION_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
