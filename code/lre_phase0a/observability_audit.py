# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - late-reversal observability audit.

Question only: can deterministic post-stop signals be mechanically recovered from the public
trajectories? No recovery classifier is defined and no LLM/API call is made.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
VENDOR = REPO / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
SAMPLES = WS / "outputs" / "late_reversal_early_eval_phase0a" / "public_dataset" / "samples"
OUT = WS / "outputs" / "late_reversal_early_eval_phase0a"
NL = chr(10)

os.environ["EARLYEVAL_VENDOR_RUNTIME_ROOT"] = str(
    WS / "work" / "lre_phase0a" / "vendor_runtime")
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import step_builder as SB  # noqa: E402

from mapping_adapter import SUBMIT_MARKER, adapt_messages  # noqa: E402

TEST_RE = re.compile(r"(?i)(pytest|python\s+-m\s+unittest|py\.test|\btox\b|"
                     r"manage\.py\s+test|nosetests|run_tests)")
EDIT_RE = re.compile(r"(?i)(sed\s+-i|patch\s+-p\d|git\s+apply|apply_patch|str_replace|"
                     r"cat\s*>\s*\S+|\btee\s+\S+|<<\s*'?EOF|>>\s*\S+\.(py|json|txt|md|toml|cfg))")


def main():
    files = sorted(SAMPLES.rglob("*.traj.json"))
    timelines, agg = {}, {"traj_k_pairs": 0, "late_activity": 0, "post_stop_edit": 0,
                          "post_stop_test": 0, "post_stop_test_pass": 0,
                          "post_stop_submission": 0}
    example = None
    per_traj_summary = []

    for f in files:
        d = json.loads(f.read_text("utf-8"))
        info = d.get("info") or {}
        model = (info.get("docent") or {}).get("model_label") or f.parent.name
        resolved = bool(info.get("resolved"))
        rec = pd.Series({
            "traj_id": str(model) + "::" + str(d.get("instance_id")),
            "instance_id": d.get("instance_id"), "resolved": resolved, "model": str(model),
            "messages": json.dumps(adapt_messages(d.get("messages") or []), ensure_ascii=False),
        })
        steps = SB.rebuild_steps_for_trajectory(rec)
        events = []
        for s in steps:
            action_text = (s.get("action_text") or "") + NL + (s.get("tool_args_text") or "")
            subtypes = [str(x) for x in (s.get("action_subtypes") or [])]
            events.append({
                "step_idx": s["step_idx"],
                "is_edit": bool(EDIT_RE.search(action_text)),
                "is_test": ("test" in subtypes) or bool(TEST_RE.search(action_text)),
                "test_pass_feedback": bool(s.get("test_pass_seen_this_step")),
                "is_submission": ("submit" in subtypes)
                or (SUBMIT_MARKER in (s.get("action_text") or "")),
                "action_type": s.get("action_major_type"),
                "action_text_head": (s.get("action_text") or "")[:160].replace(NL, " "),
            })
        timelines[str(f.relative_to(SAMPLES))] = {"model": str(model), "resolved": resolved,
                                                 "n_steps": len(steps), "events": events}

        n = len(steps)
        for k in range(1, n):
            tail = [e for e in events if e["step_idx"] > k]
            agg["traj_k_pairs"] += 1
            if tail:
                agg["late_activity"] += 1
            if any(e["is_edit"] for e in tail):
                agg["post_stop_edit"] += 1
            if any(e["is_test"] for e in tail):
                agg["post_stop_test"] += 1
            if any(e["test_pass_feedback"] for e in tail):
                agg["post_stop_test_pass"] += 1
            if any(e["is_submission"] for e in tail):
                agg["post_stop_submission"] += 1

        per_traj_summary.append({
            "file": str(f.relative_to(SAMPLES)), "model_label": str(model),
            "resolved": resolved, "n_steps": n,
            "edit_steps": [e["step_idx"] for e in events if e["is_edit"]],
            "test_steps": [e["step_idx"] for e in events if e["is_test"]],
            "test_pass_steps": [e["step_idx"] for e in events if e["test_pass_feedback"]],
            "submission_steps": [e["step_idx"] for e in events if e["is_submission"]],
        })

        if example is None and n >= 12 and any(e["is_edit"] for e in events) \
                and any(e["is_test"] for e in events):
            mid = n // 2
            tail = [e for e in events if e["step_idx"] > mid]
            example = {
                "file": str(f.relative_to(SAMPLES)), "model_label": str(model),
                "resolved": resolved, "n_steps": n, "hypothetical_early_stop_step_k": mid,
                "post_stop_events": [e for e in tail][:20],
                "post_stop_signal_presence": {
                    "LATE_ACTIVITY": bool(tail),
                    "POST_STOP_EDIT": any(e["is_edit"] for e in tail),
                    "POST_STOP_TEST": any(e["is_test"] for e in tail),
                    "POST_STOP_TEST_PASS": any(e["test_pass_feedback"] for e in tail),
                    "POST_STOP_SUBMISSION": any(e["is_submission"] for e in tail),
                },
            }

    n_traj = len(files)
    result = {
        "method": "deterministic step timeline via EarlyEval vendored step_builder after a "
                  "field-only adapter; signals detected by regex over action_text/tool args and "
                  "by observation_parser flags",
        "trajectories": n_traj,
        "step_indexing": "step_idx starts at 1; a hypothetical early stop at k leaves steps > k",
        "signal_definition": {
            "LATE_ACTIVITY": "any further agent step exists after k",
            "POST_STOP_EDIT": "a later action whose command matches a shell/file-edit pattern",
            "POST_STOP_TEST": "a later action classified as a test command",
            "POST_STOP_TEST_PASS": "a later observation carrying test-pass feedback",
            "POST_STOP_SUBMISSION": "a later submit action / COMPLETE_TASK_AND_SUBMIT marker",
            "FINAL_RESOLVED": "info.resolved for the trajectory",
        },
        "aggregate_over_all_trajectory_x_k_pairs": agg,
        "availability": {
            "LATE_ACTIVITY": "AVAILABLE",
            "POST_STOP_EDIT": "AVAILABLE",
            "POST_STOP_TEST": "AVAILABLE",
            "POST_STOP_TEST_PASS": "AVAILABLE",
            "POST_STOP_SUBMISSION": "AVAILABLE",
            "FINAL_RESOLVED": "AVAILABLE",
        },
        "notes": {
            "POST_STOP_EDIT": "Recoverable from the raw command text. The shipped EarlyEval "
                              "action taxonomy labels mini-SWE-agent shell edits as run_cli "
                              "(0 edit-labeled steps), so edit detection must use a deterministic "
                              "command-pattern detector rather than action_major_type.",
            "POST_STOP_SUBMISSION": "mini-SWE-agent submits via "
                                    "'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'; info.exit_status "
                                    "is 'Submitted'. Not present in every trajectory.",
            "FINAL_RESOLVED": "info.resolved present for all sampled trajectories; the sample "
                              "contains both resolved and unresolved runs.",
        },
        "per_trajectory": per_traj_summary,
        "worked_example": example,
        "no_llm_semantic_labeling": True,
    }
    (OUT / "late_reversal_observability.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    total = agg["traj_k_pairs"]
    print(json.dumps({
        "trajectories": n_traj,
        "traj_x_k_pairs": total,
        "availability_pct": {k: round(100.0 * agg[k] / total, 2) for k in
                             ("late_activity", "post_stop_edit", "post_stop_test",
                              "post_stop_test_pass", "post_stop_submission")},
        "availability": result["availability"],
        "worked_example": None if example is None else {
            "file": example["file"], "k": example["hypothetical_early_stop_step_k"],
            "n_steps": example["n_steps"],
            "signals": example["post_stop_signal_presence"]},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
