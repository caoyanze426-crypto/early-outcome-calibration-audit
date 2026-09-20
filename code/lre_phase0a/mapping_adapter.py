# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - deterministic mini-SWE-agent -> EarlyEval adapter.

Field renaming / synthesis only. No LLM calls, no semantic labeling.
"""
from __future__ import annotations

import json

NL = chr(10)
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def split_tool_call(tc):
    """Return (name, arguments) from either tool_call shape.

    OpenAI style:   {"function": {"name": ..., "arguments": {...}}}
    mini-swe style: {"function": "bash", "arguments": {...}}
    """
    fn = (tc or {}).get("function")
    if isinstance(fn, dict):
        return fn.get("name", "unknown"), fn.get("arguments", "")
    if isinstance(fn, str):
        return fn, (tc or {}).get("arguments", "")
    return "unknown", (tc or {}).get("arguments", "")


def command_of(msg) -> str:
    """Concatenate bash commands issued by one assistant action message."""
    out = []
    for tc in (msg.get("tool_calls") or []):
        _name, args = split_tool_call(tc)
        if isinstance(args, dict):
            out.append(str(args.get("command", json.dumps(args, ensure_ascii=False))))
        else:
            out.append(str(args))
    return NL.join(out)


def normalize_tool_calls(msg):
    """Rewrite tool_calls into OpenAI style so the vendored step_builder can read them."""
    tcs = msg.get("tool_calls")
    if not tcs:
        return None
    out = []
    for tc in tcs:
        name, args = split_tool_call(tc)
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        out.append({"id": (tc or {}).get("id"), "type": "function",
                    "function": {"name": name, "arguments": args}})
    return out


def adapt_messages(messages):
    """mini-SWE-agent message list -> SWE-smith-like shape expected by step_builder.

    assistant -> message_type='action' with 'action' = command string
    tool      -> message_type='observation'
    system/user -> message_type='task'
    """
    out = []
    for m in messages:
        role = m.get("role")
        nm = dict(m)
        if role == "assistant":
            nm["message_type"] = "action"
            nm["action"] = command_of(m)
            normalized = normalize_tool_calls(m)
            if normalized is not None:
                nm["tool_calls"] = normalized
        elif role == "tool":
            nm["message_type"] = "observation"
        else:
            nm["message_type"] = "task" if role in ("system", "user") else role
        out.append(nm)
    return out


def to_contract(doc, model_label):
    """mini-SWE-agent .traj.json -> EarlyEval TrajectoryRecord-compatible dict."""
    info = doc.get("info") or {}
    return {
        "benchmark": "swebench",
        "instance_id": doc.get("instance_id"),
        "traj_id": str(model_label) + "::" + str(doc.get("instance_id")),
        "model_id": str(model_label),
        "resolved": bool(info.get("resolved")),
        "messages": adapt_messages(doc.get("messages") or []),
        "patch": str(info.get("submission") or ""),
    }
