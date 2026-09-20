# -*- coding: utf-8 -*-
"""Phase 0B step 7: late-reversal signatures, concentration, pre-registered gate,
and secondary descriptives. Strictly offline; no LLM / model API calls."""
from __future__ import annotations

import json
import sys

from common import OUT, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ANA = OUT / "analysis"


def pct(num, den):
    return (100.0 * float(num) / float(den)) if den else None


def main() -> int:
    ensure_dirs(ANA)
    traj = pd.read_csv(ANA / "trajectory_outcomes.csv")
    ff = traj[traj["outcome_class"] == "FALSE_FAILURE"].copy()
    fs = traj[traj["outcome_class"] == "FALSE_SUCCESS"].copy()

    ff_cols = ["traj_id", "model_id", "instance_id", "n_steps_total",
               "decision_step", "decision_fraction", "saved_steps",
               "POST_STOP_ACTIVITY", "POST_STOP_EDIT", "POST_STOP_TEST",
               "POST_STOP_TEST_PASS", "POST_STOP_TEST_FAIL",
               "POST_STOP_ERROR_OR_TRACEBACK", "POST_STOP_SUBMISSION",
               "POST_STOP_EDIT_COUNT", "POST_STOP_TEST_COUNT", "POST_STOP_STEPS",
               "FIRST_TEST_PASS_AFTER_STOP", "FF_LATE_RECOVERY_SIGNATURE"]
    fs_cols = ["traj_id", "model_id", "instance_id", "n_steps_total",
               "decision_step", "decision_fraction", "saved_steps",
               "POST_STOP_ACTIVITY", "POST_STOP_EDIT", "POST_STOP_TEST",
               "POST_STOP_TEST_PASS", "POST_STOP_TEST_FAIL",
               "POST_STOP_ERROR_OR_TRACEBACK", "POST_STOP_SUBMISSION",
               "POST_STOP_STEPS", "FS_LATE_COLLAPSE_SIGNATURE"]
    ff[ff_cols].to_csv(ANA / "false_failures.csv", index=False, encoding="utf-8")
    fs[fs_cols].to_csv(ANA / "false_successes.csv", index=False, encoding="utf-8")
    late = pd.concat([ff[["traj_id", "model_id", "decision_step",
                          "FF_LATE_RECOVERY_SIGNATURE"]].assign(direction="FALSE_FAILURE"),
                      fs[["traj_id", "model_id", "decision_step",
                          "FS_LATE_COLLAPSE_SIGNATURE"]]
                      .rename(columns={"FS_LATE_COLLAPSE_SIGNATURE":
                                       "FF_LATE_RECOVERY_SIGNATURE"})
                      .assign(direction="FALSE_SUCCESS")], ignore_index=True)
    late.to_csv(ANA / "late_reversal_signatures.csv", index=False, encoding="utf-8")

    # ---- section 15: error-direction concentration
    def concentration(frame, direction):
        if frame.empty:
            return {"direction": direction, "count": 0, "by_model": {},
                    "n_models_with_error": 0, "largest_model_share": None,
                    "largest_model": None, "unique_tasks": 0,
                    "by_instance_top10": {}}
        by_model = frame["model_id"].value_counts().to_dict()
        largest_model = max(by_model, key=by_model.get)
        by_instance = frame["instance_id"].value_counts().to_dict()
        return {
            "direction": direction,
            "count": int(len(frame)),
            "by_model": {k: int(v) for k, v in sorted(by_model.items())},
            "n_models_with_error": int(len(by_model)),
            "largest_model": largest_model,
            "largest_model_share": float(by_model[largest_model]) / float(len(frame)),
            "unique_tasks": int(frame["instance_id"].nunique()),
            "by_instance_top10": {k: int(v) for k, v in
                                  sorted(by_instance.items(),
                                         key=lambda kv: -kv[1])[:10]},
        }

    conc_ff = concentration(ff, "FALSE_FAILURE")
    conc_fs = concentration(fs, "FALSE_SUCCESS")
    n_ff, n_fs = len(ff), len(fs)
    n_err = n_ff + n_fs
    if n_err == 0:
        dominant, dominant_frame, conc = None, None, None
    elif n_ff >= n_fs:
        dominant, dominant_frame, conc = "FALSE_FAILURE", ff, conc_ff
    else:
        dominant, dominant_frame, conc = "FALSE_SUCCESS", fs, conc_fs

    # ---- section 16: pre-registered gate
    gate = {"A_N_ERRORS_GE_10": bool(n_err >= 10),
            "B_ONE_DIRECTION_GE_70PCT": None, "B_dominant_share": None,
            "C_DOMINANT_IN_GE_3_MODELS": None,
            "D_LARGEST_MODEL_LE_60PCT": None,
            "E_SIGNATURE_GE_50PCT": None, "E_signature_rate": None,
            "dominant_direction": dominant}
    if dominant is not None:
        ds = (n_ff / n_err) if dominant == "FALSE_FAILURE" else (n_fs / n_err)
        gate["B_dominant_share"] = ds
        gate["B_ONE_DIRECTION_GE_70PCT"] = bool(ds >= 0.70)
        gate["C_DOMINANT_IN_GE_3_MODELS"] = bool(
            conc["n_models_with_error"] >= 3)
        gate["D_LARGEST_MODEL_LE_60PCT"] = bool(
            conc["largest_model_share"] is not None
            and conc["largest_model_share"] <= 0.60)
        col = ("FF_LATE_RECOVERY_SIGNATURE" if dominant == "FALSE_FAILURE"
               else "FS_LATE_COLLAPSE_SIGNATURE")
        rate = float(dominant_frame[col].astype(bool).mean())
        gate["E_signature_rate"] = rate
        gate["E_SIGNATURE_GE_50PCT"] = bool(rate >= 0.50)
    checks = [gate[k] for k in ("A_N_ERRORS_GE_10", "B_ONE_DIRECTION_GE_70PCT",
                                "C_DOMINANT_IN_GE_3_MODELS",
                                "D_LARGEST_MODEL_LE_60PCT",
                                "E_SIGNATURE_GE_50PCT")]
    gate["OVERALL"] = ("GO_CANDIDATE" if all(c is True for c in checks)
                       else "NO_STRONG_SIGNAL")
    gate["note"] = ("mechanical, pre-registered; thresholds were not altered after "
                    "seeing results")

    signature_summary = {
        "FALSE_FAILURE_count": int(n_ff),
        "FF_LATE_RECOVERY_SIGNATURE_count": int(
            ff["FF_LATE_RECOVERY_SIGNATURE"].astype(bool).sum()) if n_ff else 0,
        "FF_LATE_RECOVERY_SIGNATURE_rate": (float(
            ff["FF_LATE_RECOVERY_SIGNATURE"].astype(bool).mean()) if n_ff else None),
        "FALSE_SUCCESS_count": int(n_fs),
        "FS_LATE_COLLAPSE_SIGNATURE_count": int(
            fs["FS_LATE_COLLAPSE_SIGNATURE"].astype(bool).sum()) if n_fs else 0,
        "FS_LATE_COLLAPSE_SIGNATURE_rate": (float(
            fs["FS_LATE_COLLAPSE_SIGNATURE"].astype(bool).mean()) if n_fs else None),
        "FF_post_stop_component_rates": {
            c: (float(ff[c].astype(bool).mean()) if n_ff else None)
            for c in ("POST_STOP_ACTIVITY", "POST_STOP_EDIT", "POST_STOP_TEST",
                      "POST_STOP_TEST_PASS", "POST_STOP_TEST_FAIL",
                      "POST_STOP_ERROR_OR_TRACEBACK", "POST_STOP_SUBMISSION")},
        "FS_post_stop_component_rates": {
            c: (float(fs[c].astype(bool).mean()) if n_fs else None)
            for c in ("POST_STOP_ACTIVITY", "POST_STOP_EDIT", "POST_STOP_TEST",
                      "POST_STOP_TEST_PASS", "POST_STOP_TEST_FAIL",
                      "POST_STOP_ERROR_OR_TRACEBACK", "POST_STOP_SUBMISSION")},
        "FF_FIRST_TEST_PASS_AFTER_STOP": (
            ff["FIRST_TEST_PASS_AFTER_STOP"].value_counts().to_dict()
            if n_ff else {}),
    }
    write_json(ANA / "late_reversal_signature_summary.json", signature_summary)
    write_json(ANA / "concentration_summary.json",
               {"false_failure": conc_ff, "false_success": conc_fs})
    scope = {
        "total_trajectories": int(len(traj)),
        "total_prefix_rows": int(traj["n_prefix_rows"].sum()),
        "prefix_rows_by_model": {m: int(p["n_prefix_rows"].sum())
                                 for m, p in traj.groupby("model_id", sort=True)},
        "trajectories_by_model": {m: int(len(p))
                                  for m, p in traj.groupby("model_id", sort=True)},
        "trajectory_length_steps": {
            "mean": float(traj["n_steps_total"].mean()),
            "median": float(traj["n_steps_total"].median()),
            "min": int(traj["n_steps_total"].min()),
            "max": int(traj["n_steps_total"].max()),
        },
    }
    write_json(ANA / "dataset_scope_summary.json", scope)
    write_json(ANA / "signal_gate.json", gate)

    # ---- section 17: descriptives
    bins = [0.0, 0.25, 0.5, 0.75, 1.0000001]
    labels = ["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
    decided = traj[traj["decided"]].copy()
    decided["fraction_bin"] = pd.cut(decided["decision_fraction"], bins=bins,
                                     labels=labels, right=False, include_lowest=True)
    rows = []
    for b in labels:
        part = decided[decided["fraction_bin"] == b]
        errs = part[part["outcome_class"].isin(["FALSE_FAILURE", "FALSE_SUCCESS"])]
        rows.append({
            "decision_fraction_bin": b, "early_decisions": int(len(part)),
            "errors": int(len(errs)),
            "error_rate": (float(len(errs)) / len(part)) if len(part) else None,
            "false_failure": int((part["outcome_class"] == "FALSE_FAILURE").sum()),
            "false_success": int((part["outcome_class"] == "FALSE_SUCCESS").sum()),
        })
    frac = pd.DataFrame(rows)
    frac.to_csv(ANA / "decision_fraction_summary.csv", index=False, encoding="utf-8")

    lengths = []
    for cls, part in traj.groupby("outcome_class", sort=True):
        lengths.append({
            "outcome_class": cls, "n": int(len(part)),
            "traj_len_mean": float(part["n_steps_total"].mean()),
            "traj_len_median": float(part["n_steps_total"].median()),
            "traj_len_min": int(part["n_steps_total"].min()),
            "traj_len_max": int(part["n_steps_total"].max()),
            "saved_steps_mean": float(part["saved_steps"].mean()),
            "saved_steps_median": float(part["saved_steps"].median()) if len(part) else None,
            "decision_step_mean": (float(part["decision_step"].mean())
                                   if part["decided"].any() else None),
            "decision_fraction_mean": (float(part["decision_fraction"].mean())
                                       if part["decided"].any() else None),
        })
    pd.DataFrame(lengths).to_csv(ANA / "trajectory_length_summary.csv", index=False,
                                 encoding="utf-8")
    print(json.dumps({"signatures": signature_summary, "gate": gate},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
