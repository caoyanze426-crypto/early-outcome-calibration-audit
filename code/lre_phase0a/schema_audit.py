# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - trajectory schema + earlyeval compatibility audit.

Deterministic only. No LLM/API calls, no semantic labeling, no LLM-based judgment.
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

from mapping_adapter import (SUBMIT_MARKER, adapt_messages, command_of,  # noqa: E402
                             split_tool_call, to_contract)

TEST_RE = re.compile(r"(?i)(pytest|python\s+-m\s+unittest|py\.test|\btox\b|"
                     r"manage\.py\s+test|nosetests)")
EDIT_RE = re.compile(r"(?i)(sed\s+-i|patch\s+-p|git\s+apply|apply_patch|str_replace|"
                     r"<\s*<\s*'?EOF|cat\s*>|tee\s+\S+)")


def load(p: Path):
    return json.loads(p.read_text("utf-8"))


def main():
    files = sorted(SAMPLES.rglob("*.traj.json"))
    schema_rows, per_traj, mapping_examples, contracts = [], [], [], []

    for f in files:
        d = load(f)
        info = d.get("info") or {}
        model = (info.get("docent") or {}).get("model_label") or f.parent.name
        msgs = d.get("messages") or []
        roles = [m.get("role") for m in msgs]
        keys = sorted({k for m in msgs for k in m})
        cmds = [command_of(m) for m in msgs
                if m.get("role") == "assistant" and m.get("tool_calls")]
        joined = NL.join(cmds)
        obs_all = NL.join(str(m.get("content") or "") for m in msgs if m.get("role") == "tool")
        schema_rows.append({
            "file": str(f.relative_to(SAMPLES)),
            "instance_id": d.get("instance_id"),
            "trajectory_format": d.get("trajectory_format"),
            "model_label": model,
            "resolved": bool(info.get("resolved")),
            "exit_status": info.get("exit_status"),
            "submission_present": info.get("submission") is not None,
            "submission_len": len(info.get("submission") or ""),
            "message_count": len(msgs),
            "message_keys_union": "|".join(keys),
            "roles_present": "|".join(sorted(set(roles))),
            "assistant_action_messages": sum(1 for m in msgs
                                             if m.get("role") == "assistant" and m.get("tool_calls")),
            "tool_observations": sum(1 for r in roles if r == "tool"),
            "test_commands": sum(1 for c in cmds if TEST_RE.search(c)),
            "edit_commands": sum(1 for c in cmds if EDIT_RE.search(c)),
            "submit_marker_occurrences": joined.count(SUBMIT_MARKER)
            + obs_all.count(SUBMIT_MARKER),
            "obs_pass_hits": len(re.findall(r"(?i)(\d+\s+passed|PASSED|\bOK\b)", obs_all)),
            "obs_fail_hits": len(re.findall(r"(?i)(\d+\s+failed|FAILED|FAILURES)", obs_all)),
            "tool_call_function_names": "|".join(sorted({
                split_tool_call(tc)[0]
                for m in msgs for tc in (m.get("tool_calls") or [])})),
            "tool_call_argument_keys": "|".join(sorted({
                "|".join(sorted(split_tool_call(tc)[1].keys()))
                if isinstance(split_tool_call(tc)[1], dict) else "NON_DICT_ARGUMENTS"
                for m in msgs for tc in (m.get("tool_calls") or [])})),
        })

        contracts.append(to_contract(d, model))

        rec = pd.Series({
            "traj_id": str(model) + "::" + str(d.get("instance_id")),
            "instance_id": d.get("instance_id"),
            "resolved": bool(info.get("resolved")),
            "model": str(model),
            "messages": json.dumps(adapt_messages(msgs), ensure_ascii=False),
        })
        try:
            steps = SB.rebuild_steps_for_trajectory(rec)
            err = None
        except Exception as exc:  # noqa: BLE001
            steps, err = [], type(exc).__name__ + ": " + str(exc)

        edit_idx = [s["step_idx"] for s in steps
                    if s.get("action_major_type") == "edit"
                    or any(str(x).startswith("edit_") for x in (s.get("action_subtypes") or []))]
        test_idx = [s["step_idx"] for s in steps
                    if "test" in (s.get("action_subtypes") or [])]
        pass_idx = [s["step_idx"] for s in steps if s.get("test_pass_seen_this_step")]
        fail_idx = [s["step_idx"] for s in steps if s.get("test_fail_seen_this_step")]
        sub_idx = [s["step_idx"] for s in steps if s.get("action_major_type") == "submit"
                   or SUBMIT_MARKER in (s.get("action_text") or "")]
        per_traj.append({
            "file": str(f.relative_to(SAMPLES)), "instance_id": d.get("instance_id"),
            "model_label": str(model), "resolved": bool(info.get("resolved")),
            "n_steps_built": len(steps), "step_build_error": err,
            "edit_steps": edit_idx, "test_steps": test_idx, "test_pass_steps": pass_idx,
            "test_fail_steps": fail_idx, "submit_steps": sub_idx,
        })
        if len(mapping_examples) < 4:
            mapping_examples.append({
                "file": str(f.relative_to(SAMPLES)), "model_label": str(model),
                "resolved": bool(info.get("resolved")),
                "src_instance_id": d.get("instance_id"), "dst_instance_id": d.get("instance_id"),
                "src_model": "info.docent.model_label", "dst_model_id": str(model),
                "src_resolved": "info.resolved", "dst_resolved": bool(info.get("resolved")),
                "src_action": "assistant.tool_calls[0].function.arguments.command",
                "dst_action": "message_type='action' + action=<command>",
                "src_observation": "role='tool'.content",
                "dst_observation": "message_type='observation'",
                "n_steps_built": len(steps),
            })

    (OUT / "schema").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(schema_rows).to_csv(OUT / "schema" / "source_schema_audit.csv", index=False)
    (OUT / "schema" / "source_schema.json").write_text(
        json.dumps({"per_file": schema_rows}, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    (OUT / "schema" / "earlyeval_schema.json").write_text(json.dumps({
        "trajectory_contract": {
            "module": "earlyeval/core/contracts.py :: TrajectoryRecord",
            "fields": ["benchmark", "instance_id", "traj_id", "model_id", "resolved",
                       "messages", "patch", "metadata"]},
        "prefix_contract": {
            "module": "earlyeval/core/contracts.py :: PrefixRecord",
            "fields": ["prefix_id", "traj_id", "instance_id", "model_id", "prefix_step_idx",
                       "label", "sample_weight", "metadata"]},
        "raw_trajectory_parquet_contract": {
            "module": "earlyeval/vendor/.../step_builder.py",
            "required_message_keys_for_action": ["role=assistant", "message_type=action",
                                                 "action"],
            "required_columns": ["messages", "traj_id", "instance_id", "resolved", "model"]},
        "prediction_table_contract": {
            "module": "examples/smoke_predictions.csv + configs/policy_presets.yaml",
            "required_columns": ["traj_id", "label", "prefix_step_idx",
                                 "prob_cal_safe_success__<predictor>",
                                 "prob_cal_safe_failure__<predictor>"]},
    }, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    (OUT / "schema" / "field_mapping.json").write_text(json.dumps({
        "direction": "public mini-SWE-agent .traj.json -> EarlyEval representations",
        "trajectory_level": [
            {"source": "instance_id", "target": "instance_id / traj_id"},
            {"source": "info.resolved", "target": "resolved"},
            {"source": "info.docent.model_label", "target": "model_id / model"},
            {"source": "messages", "target": "messages"},
            {"source": "info.submission", "target": "patch (optional; empty in sample)"}],
        "step_level_adapter": [
            {"source": "assistant.tool_calls[0].function.arguments.command",
             "target": "message_type='action' + action=<command string>"},
            {"source": "assistant.content", "target": "assistant content / thought text"},
            {"source": "role='tool'.content", "target": "message_type='observation'"},
            {"source": "traj_id", "target": "traj_id / group_id"}],
        "missing_for_earlyeval": [
            "message_type field (must be synthesized)",
            "action field (must be synthesized from the tool_calls command)",
            "prefix probability columns (require a trained EarlyEval predictor)"],
        "examples": mapping_examples,
    }, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    (OUT / "public_dataset" / "normalized_contract_sample.jsonl").write_text(
        NL.join(json.dumps(c, ensure_ascii=False) for c in contracts) + NL, "utf-8", newline=NL)
    (OUT / "public_dataset" / "step_rebuild_probe.json").write_text(
        json.dumps(per_traj, ensure_ascii=False, indent=2), "utf-8", newline=NL)

    print(json.dumps({
        "files": len(files),
        "models": sorted({r["model_label"] for r in schema_rows}),
        "resolved_true": sum(1 for r in schema_rows if r["resolved"]),
        "resolved_false": sum(1 for r in schema_rows if not r["resolved"]),
        "step_build_ok": sum(1 for r in per_traj if not r["step_build_error"]),
        "step_build_errors": [r["step_build_error"] for r in per_traj if r["step_build_error"]],
        "total_steps_built": sum(r["n_steps_built"] for r in per_traj),
        "traj_with_test_steps": sum(1 for r in per_traj if r["test_steps"]),
        "traj_with_edit_steps": sum(1 for r in per_traj if r["edit_steps"]),
        "traj_with_pass_feedback": sum(1 for r in per_traj if r["test_pass_steps"]),
        "traj_with_submit_steps": sum(1 for r in per_traj if r["submit_steps"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
