# -*- coding: utf-8 -*-
"""Phase 0B step 6: locked stopping policy + deterministic post-stop analyses.

Strictly offline. SCIENTIFIC_API_CALLS = 0, LLM_CALLS = 0. The pre-registered
signal gate (section 16) is applied only after every fold and detector is complete.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

from common import OUT, WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earlyeval.core.contracts import PolicySpec  # noqa: E402
from earlyeval.policies.safe_stop import apply_policy  # noqa: E402

FOLD_DIR = WORK / "folds"
STEP_DIR = WORK / "step_table"
ANA = OUT / "analysis"
PRED = OUT / "predictions"
FEAT = OUT / "features"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TEST_RE = re.compile(r"(?i)(pytest|python\s+-m\s+unittest|py\.test|\btox\b|"
                     r"manage\.py\s+test|nosetests|run_tests)")
EDIT_RE = re.compile(r"(?i)(sed\s+-i|patch\s+-p\d|git\s+apply|apply_patch|"
                     r"str_replace|cat\s*>\s*\S+|\btee\s+\S+|<<\s*'?EOF|"
                     r">>\s*\S+\.(py|json|txt|md|toml|cfg))")
DETECTOR_LINES = [
    "POST_STOP_ACTIVITY = exists step with step_idx > k",
    "POST_STOP_EDIT = EDIT_RE.search(action_text + NL + tool_args_text)",
    "POST_STOP_TEST = ('test' in action_subtypes) or TEST_RE.search(action_text)",
    "POST_STOP_TEST_PASS = observation_parser test_pass_seen_this_step",
    "POST_STOP_TEST_FAIL = observation_parser test_fail_seen_this_step",
    "POST_STOP_ERROR_OR_TRACEBACK = tool_error_seen_this_step or traceback_seen_this_step",
    "POST_STOP_SUBMISSION = ('submit' in action_subtypes) or SUBMIT_MARKER in action_text",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detector_spec() -> dict:
    return {
        "source": "reused and frozen from LATE_REVERSAL_EARLY_EVAL_PHASE0A "
                  "(work/lre_phase0a/observability_audit.py) plus the submission "
                  "marker from work/lre_phase0a/mapping_adapter.py",
        "scope": "for a trajectory with an early decision at prefix step k, only "
                 "steps with step_idx > k are scanned",
        "TEST_RE": TEST_RE.pattern,
        "EDIT_RE": EDIT_RE.pattern,
        "SUBMIT_MARKER": SUBMIT_MARKER,
        "events": DETECTOR_LINES,
        "definitions": {
            "FF_LATE_RECOVERY_SIGNATURE": "POST_STOP_EDIT AND "
                                          "(POST_STOP_TEST_PASS OR POST_STOP_SUBMISSION)",
            "FS_LATE_COLLAPSE_SIGNATURE": "POST_STOP_TEST_FAIL OR "
                                          "POST_STOP_ERROR_OR_TRACEBACK",
        },
        "no_llm_semantic_labeling": True,
        "frozen_before_inspecting_error_classes": True,
        "spec_sha256": sha256_text(chr(10).join(DETECTOR_LINES)),
    }


def load_step_events() -> dict:
    import pyarrow.parquet as pq

    parts = sorted(STEP_DIR.glob("step_table.part-*.parquet"))
    cols = ["traj_id", "step_idx", "action_text", "tool_args_text",
            "action_subtypes", "test_pass_seen_this_step",
            "test_fail_seen_this_step", "traceback_seen_this_step",
            "tool_error_seen_this_step"]
    events: dict = {}
    for p in parts:
        pf = pq.ParquetFile(p)
        tbl = pf.read(columns=cols)
        traj = tbl.column("traj_id").to_pylist()
        step = tbl.column("step_idx").to_pylist()
        act = tbl.column("action_text").to_pylist()
        args = tbl.column("tool_args_text").to_pylist()
        sub = tbl.column("action_subtypes").to_pylist()
        tpass = tbl.column("test_pass_seen_this_step").to_pylist()
        tfail = tbl.column("test_fail_seen_this_step").to_pylist()
        tback = tbl.column("traceback_seen_this_step").to_pylist()
        terr = tbl.column("tool_error_seen_this_step").to_pylist()
        for i in range(tbl.num_rows):
            a = act[i] or ""
            g = args[i] or ""
            combined = a + chr(10) + g
            subtypes = [str(x) for x in (sub[i] or [])]
            row = (
                int(step[i]),
                1 if EDIT_RE.search(combined) else 0,
                1 if (("test" in subtypes) or TEST_RE.search(a)) else 0,
                1 if tpass[i] else 0,
                1 if tfail[i] else 0,
                1 if (tback[i] or terr[i]) else 0,
                1 if (("submit" in subtypes) or (SUBMIT_MARKER in a)) else 0,
            )
            events.setdefault(str(traj[i]), []).append(row)
        del pf, tbl
    for key in events:
        events[key].sort(key=lambda r: r[0])
    return events


NO_TAIL = {
    "POST_STOP_ACTIVITY": False, "POST_STOP_EDIT": False, "POST_STOP_TEST": False,
    "POST_STOP_TEST_PASS": False, "POST_STOP_TEST_FAIL": False,
    "POST_STOP_ERROR_OR_TRACEBACK": False, "POST_STOP_SUBMISSION": False,
    "POST_STOP_EDIT_COUNT": 0, "POST_STOP_TEST_COUNT": 0, "POST_STOP_STEPS": 0,
    "FIRST_TEST_PASS_AFTER_STOP": "NOT_OBSERVABLE",
}


def scan_post_stop(events, k: int) -> dict:
    tail = [r for r in events if r[0] > k]
    if not tail:
        return dict(NO_TAIL)
    pass_idx = [r[0] for r in tail if r[3]]
    return {
        "POST_STOP_ACTIVITY": True,
        "POST_STOP_EDIT": any(r[1] for r in tail),
        "POST_STOP_TEST": any(r[2] for r in tail),
        "POST_STOP_TEST_PASS": bool(pass_idx),
        "POST_STOP_TEST_FAIL": any(r[4] for r in tail),
        "POST_STOP_ERROR_OR_TRACEBACK": any(r[5] for r in tail),
        "POST_STOP_SUBMISSION": any(r[6] for r in tail),
        "POST_STOP_EDIT_COUNT": int(sum(r[1] for r in tail)),
        "POST_STOP_TEST_COUNT": int(sum(r[2] for r in tail)),
        "POST_STOP_STEPS": int(len(tail)),
        "FIRST_TEST_PASS_AFTER_STOP": "TRUE" if pass_idx else "FALSE",
    }


def classify(decided: bool, decision: str, label: int) -> str:
    if not decided:
        return "NO_STOP"
    if decision == "success":
        return "CORRECT_SUCCESS" if label == 1 else "FALSE_SUCCESS"
    return "CORRECT_FAILURE" if label == 0 else "FALSE_FAILURE"


def main() -> int:
    ensure_dirs(ANA, PRED, FEAT)
    write_json(FEAT / "detector_spec.json", detector_spec())
    frames = []
    for fold in sorted(FOLD_DIR.glob("fold-*")):
        f = fold / "test_prefix_predictions.parquet"
        if f.exists():
            frames.append(pd.read_parquet(
                f, columns=["prefix_id", "traj_id", "instance_id", "model_id",
                            "prefix_step_idx", "label",
                            "n_steps_total",
                            f"prob_cal_safe_success__{PREDICTOR}",
                            f"prob_cal_safe_failure__{PREDICTOR}"]))
    if not frames:
        raise SystemExit("no fold test predictions found")
    all_pred = pd.concat(frames, ignore_index=True)
    all_pred.to_parquet(PRED / "heldout_prefix_predictions_all.parquet",
                        index=False, compression="zstd")
    policy = PolicySpec(name="current_safe_stop", predictor=PREDICTOR,
                        score_mode="calibrated", policy_mode="dual",
                        success_thr=0.95, failure_thr=0.95, min_step=0,
                        consecutive=1)
    decisions, policy_summary, per_agent = apply_policy(all_pred, policy)
    decisions.to_csv(PRED / "trajectory_policy_decisions.csv", index=False,
                     encoding="utf-8")
    policy_summary.to_csv(PRED / "policy_summary.csv", index=False, encoding="utf-8")
    per_agent.to_csv(PRED / "policy_per_agent.csv", index=False, encoding="utf-8")
    event_map = load_step_events()
    n_steps_total = (all_pred.groupby("traj_id", sort=False)
                     ["n_steps_total"].first().to_dict())
    rows = []
    for r in decisions.itertuples(index=False):
        tid = str(r.traj_id)
        events = event_map.get(tid, [])
        total = int(n_steps_total.get(tid, 0))
        row = {
            "traj_id": tid, "model_id": r.agent_model,
            "instance_id": tid.split("::")[-1],
            "resolved": int(r.label), "n_steps_total": total,
            "n_prefix_rows": int(r.n_steps), "decided": bool(r.decided),
            "decision": str(r.decision), "decision_step": int(r.decision_step),
            "decision_score": float(r.decision_score),
            "saved_steps": int(r.saved_steps),
            "decision_fraction": (float(r.decision_step) / total) if total else np.nan,
            "outcome_class": classify(bool(r.decided), str(r.decision), int(r.label)),
            "observed_steps_in_step_table": len(events),
        }
        if r.decided:
            post = scan_post_stop(events, int(r.decision_step))
            row.update(post)
            row["FF_LATE_RECOVERY_SIGNATURE"] = bool(
                post["POST_STOP_EDIT"] and (post["POST_STOP_TEST_PASS"]
                                            or post["POST_STOP_SUBMISSION"]))
            row["FS_LATE_COLLAPSE_SIGNATURE"] = bool(
                post["POST_STOP_TEST_FAIL"] or post["POST_STOP_ERROR_OR_TRACEBACK"])
        rows.append(row)
    traj = pd.DataFrame(rows)
    traj.to_csv(ANA / "trajectory_outcomes.csv", index=False, encoding="utf-8")
    decided = traj[traj["decided"]]
    counts = traj["outcome_class"].value_counts().to_dict()
    n_ff = int(counts.get("FALSE_FAILURE", 0))
    n_fs = int(counts.get("FALSE_SUCCESS", 0))
    n_err = n_ff + n_fs
    error_summary = {
        "N_TRAJECTORIES": int(len(traj)),
        "N_EARLY_DECISIONS": int(len(decided)),
        "CORRECT_SUCCESS": int(counts.get("CORRECT_SUCCESS", 0)),
        "CORRECT_FAILURE": int(counts.get("CORRECT_FAILURE", 0)),
        "FALSE_SUCCESS": n_fs, "FALSE_FAILURE": n_ff,
        "NO_STOP": int(counts.get("NO_STOP", 0)), "N_ERRORS": n_err,
        "FALSE_FAILURE_FRACTION": (n_ff / n_err) if n_err else None,
        "FALSE_SUCCESS_FRACTION": (n_fs / n_err) if n_err else None,
        "COVERAGE": (float(len(decided)) / float(len(traj))) if len(traj) else None,
        "policy": {"predictor": PREDICTOR, "score_mode": "calibrated",
                   "policy_mode": "dual", "success_thr": 0.95,
                   "failure_thr": 0.95, "min_step": 0, "consecutive": 1},
    }
    write_json(ANA / "error_summary.json", error_summary)
    per_model = []
    for model, part in traj.groupby("model_id", sort=True):
        d = part[part["decided"]]
        c = part["outcome_class"].value_counts().to_dict()
        per_model.append({
            "model_id": model, "trajectories": int(len(part)),
            "resolved": int(part["resolved"].sum()),
            "resolve_rate": float(part["resolved"].mean()),
            "early_decisions": int(len(d)),
            "correct_success": int(c.get("CORRECT_SUCCESS", 0)),
            "correct_failure": int(c.get("CORRECT_FAILURE", 0)),
            "false_success": int(c.get("FALSE_SUCCESS", 0)),
            "false_failure": int(c.get("FALSE_FAILURE", 0)),
            "no_stop": int(c.get("NO_STOP", 0)),
            "decision_accuracy": ((float(c.get("CORRECT_SUCCESS", 0)
                                         + c.get("CORRECT_FAILURE", 0)) / len(d))
                                  if len(d) else None),
            "mean_decision_fraction": (float(d["decision_fraction"].mean())
                                       if len(d) else None),
        })
    pd.DataFrame(per_model).to_csv(ANA / "per_model_summary.csv", index=False,
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
