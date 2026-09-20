# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - assemble Section-18 artifacts, integrity + digests."""
from __future__ import annotations
import os

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2a"
OUT = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
AN = OUT / "analysis"
ADS = OUT / "adapter"

UPSTREAM = {
    "phase0a": "late_reversal_early_eval_phase0a",
    "phase0b": "late_reversal_early_eval_phase0b_signal_hunt",
    "phase0c": "late_reversal_early_eval_phase0c",
    "phase0d": "earlyeval_phase0d_cross_agent_calibration",
    "phase0e": "earlyeval_phase0e_prior_shift_decomposition",
    "phase1a": "earlyeval_phase1a_same_predictor_transfer",
    "phase1a_d": "earlyeval_phase1a_d_target_persistence",
    "phase1b": "earlyeval_phase1b_threshold_robustness",
}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_upstream():
    report = {}
    for name, d in UPSTREAM.items():
        man = WS / "outputs" / d / "artifact_sha256sums.txt"
        entry = {"dir": str(man.parent), "manifest": str(man), "exists": man.exists()}
        if not man.exists():
            entry.update({"entries": 0, "mismatches": 0, "missing": [], "match": False})
            report[name] = entry
            continue
        entries = 0
        mism = []
        missing = []
        for line in man.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            want, rel = parts[0], parts[1].strip()
            entries += 1
            f = man.parent / rel
            if not f.exists():
                missing.append(rel)
                continue
            got = sha256_file(f)
            if got != want:
                mism.append({"path": rel, "manifest": want, "observed": got})
        entry.update({"entries": entries, "mismatches": len(mism),
                      "missing": missing, "match": (not mism and not missing)})
        report[name] = entry
    return report


def clone_integrity():
    clone = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
    expected = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"

    def git(*args):
        try:
            out = subprocess.run(["git", "-C", str(clone), *args],
                                 capture_output=True, text=True, timeout=30)
            return out.returncode, out.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return 1, f"{type(exc).__name__}: {exc}"

    rc_h, head = git("rev-parse", "HEAD")
    rc_s, status = git("status", "--porcelain")
    tp = clone.parent
    files = [p for p in tp.rglob("*") if p.is_file()]
    newest = max((p.stat().st_mtime for p in files), default=0)
    # Phase 2A began at 2026-09-19T10:30:00Z (after the last phase0a write at
    # 08:28Z, before the first phase-2A scan at 11:10Z).
    phase2a_start = datetime(2026, 9, 19, 10, 30, tzinfo=timezone.utc).timestamp()
    modified_in_phase2a = [
        p.relative_to(tp).as_posix() for p in files
        if p.stat().st_mtime > phase2a_start]
    return {
        "clone_path": str(clone),
        "expected_commit": expected,
        "git_head": head,
        "head_matches_expected": head == expected,
        "git_status_porcelain": status,
        "tracked_working_tree_clean": (status == "" and rc_s == 0),
        "clone_file_count": len(files),
        "newest_mtime_in_clone_utc": datetime.fromtimestamp(
            newest, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase2a_start_utc": "2026-09-19T10:30:00Z",
        "files_written_inside_clone_during_phase2a": modified_in_phase2a,
        "upstream_clone_modified": bool(modified_in_phase2a),
        "vendor_runtime_root_redirected_to": str(WORK / "vendor_runtime"),
    }


def write_dataset(agg, remote, dl) -> None:
    ds = OUT / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": remote["repo_id"],
        "benchmark": "Terminal-Bench 2.0 (tbench.ai leaderboard trajectories)",
        "revision_requested": remote["revision_requested"],
        "resolved_revision_sha": remote["sha"],
        "last_modified": remote["last_modified"],
        "license": remote["license"],
        "private": remote["private"],
        "gated": remote["gated"],
        "downloads": remote["downloads"],
        "likes": remote["likes"],
        "file_count": remote["file_count"],
        "total_bytes": remote["total_bytes"],
        "total_human": remote["total_human"],
        "data_files": [
            {"path": "data/train-00000-of-00002.parquet", "bytes": 114356912},
            {"path": "data/train-00001-of-00002.parquet", "bytes": 106636988},
        ],
        "readme_summary": {
            "total_trajectories": 52104, "tasks": 89, "agent_model_combos": 109,
            "scaffolds_agents": 26, "underlying_models": 49, "overall_pass_rate": 0.396,
            "median_steps_per_trajectory": 21, "mean_steps_per_trajectory": 47.1,
            "trials_with_trajectory_steps": 34462, "trials_per_task_agent_typical": 5,
        },
        "readme_outcome_semantics": "reward = 1 if the agent solved the task, 0 otherwise",
        "readme_step_schema": {
            "container": "JSON string parsed to list of objects",
            "object_keys": ["src", "msg", "tools", "obs"],
            "src_values": ["user", "agent", "system"],
            "obs_truncation_chars": 5000,
        },
        "verified_counts": {
            "total_rows": agg["total_rows"],
            "rows_with_steps": agg["rows_with_steps"],
            "rows_without_steps": agg["rows_without_steps"],
            "distinct_task_names": agg["unique_task_names_all"],
            "distinct_models": agg["unique_models_all"],
            "distinct_scaffolds": agg["unique_agents_all"],
            "distinct_model_x_scaffold_combos": agg["unique_combos_all"],
            "distinct_trial_ids": agg["unique_trial_ids"],
            "trial_id_empty_rows": agg["trial_id_empty_rows"],
            "total_steps": 1622075,
            "reward_values": agg["reward_raw_values"],
        },
        "readme_counts_confirmed": {
            "total_trajectories": agg["total_rows"] == 52104,
            "tasks": agg["unique_task_names_all"] == 89,
            "agent_model_combos": agg["unique_combos_all"] == 109,
            "scaffolds": agg["unique_agents_all"] == 26,
            "underlying_models": agg["unique_models_all"] == 49,
            "trials_with_trajectory_steps": agg["rows_with_steps"] == 34462,
        },
        "retrieved_utc": remote["retrieved_utc"],
        "scan_generated_utc": agg["generated_utc"],
    }
    (ds / "dataset_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    ds_manifest = {
        "repo_id": remote["repo_id"],
        "resolved_revision_sha": remote["sha"],
        "local_dir": dl["local_dir"],
        "file_count": len(agg["file_manifest"]),
        "total_bytes": sum(f["bytes"] for f in agg["file_manifest"]),
        "download_wall_clock_seconds": dl["wall_clock_seconds"],
        "files": sorted(agg["file_manifest"], key=lambda f: f["path"]),
    }
    (ds / "dataset_manifest.json").write_text(
        json.dumps(ds_manifest, ensure_ascii=False, indent=2), "utf-8")


def write_adapter_spec(agg) -> None:
    ADS.mkdir(parents=True, exist_ok=True)
    spec = {
        "section": "11 EARLYEVAL TERMINALBENCH SUPPORT",
        "adapter_module": "adapter/adapter_terminalbench.py",
        "adapter_version": "phase2a-terminalbench-adapter-1",
        "public_dataset_mapping_classification": "SIMPLE_ADAPTER",
        "classification_rule": (
            "DIRECT if the public schema can be consumed unchanged; SIMPLE_ADAPTER if a "
            "deterministic field remap suffices; SUBSTANTIAL_RECONSTRUCTION otherwise"),
        "frozen_pipeline_reference": {
            "repo_url": "https://github.com/inphotoo/earlyeval",
            "commit_sha": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
            "ingestion_module": "earlyeval/benchmarks/normalize.py",
            "ingestion_entry": "normalize_record(row, 'terminalbench', idx)",
            "step_module": "earlyeval/vendor/prefix_predict_model_holdout_answer/step_builder.py",
            "step_entry": "rebuild_steps_for_trajectory(row)",
            "prefix_module": "earlyeval/vendor/prefix_predict_model_holdout_answer/prefix_builder.py",
            "prefix_entry": "build_prefix_samples_for_trajectory(row)",
            "normalized_contract_fields": ["benchmark", "instance_id", "traj_id",
                                           "model_id", "resolved", "messages", "patch"],
            "prefix_table_required_columns": [
                "traj_id", "instance_id", "model_id", "label", "prefix_step_idx",
                "n_steps_total_for_weighting"],
        },
        "mapping_table": [
            {"target_layer": "ingestion", "target_field": "benchmark",
             "source": "constant 'terminalbench'", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "instance_id",
             "source": "task_name", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "traj_id",
             "source": "trial_id || trial_name || row index", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "model_id",
             "source": "model", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "resolved",
             "source": "reward == 1", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "messages[].role",
             "source": "'assistant' if src=='agent' else src", "status": "DIRECT"},
            {"target_layer": "ingestion", "target_field": "messages[].content",
             "source": "msg", "status": "DIRECT"},
            {"target_layer": "feature/prefix",
             "target_field": "messages[].message_type = 'action'",
             "source": "derived from src == 'agent'", "status": "ADAPTER"},
            {"target_layer": "feature/prefix", "target_field": "messages[].thought",
             "source": "msg", "status": "ADAPTER"},
            {"target_layer": "feature/prefix", "target_field": "messages[].action",
             "source": "newline-joined tools[].cmd (tool name kept in tool_calls)",
             "status": "ADAPTER"},
            {"target_layer": "feature/prefix",
             "target_field": "messages[].tool_calls[].function.name",
             "source": "tools[].fn", "status": "ADAPTER"},
            {"target_layer": "feature/prefix",
             "target_field": "messages[].tool_calls[].function.arguments",
             "source": "tools[].cmd", "status": "ADAPTER"},
            {"target_layer": "feature/prefix",
             "target_field": "tool feedback message {role: 'tool'}",
             "source": "obs (one message per agent step with non-empty obs)",
             "status": "ADAPTER"},
        ],
        "fidelity_caveats": [
            "community steps carry no thought/action separation; msg is reused for thought and content",
            "action_text is reconstructed from tool name/command and is lossy for multi-tool steps",
            "the task prompt is only recoverable when a user/system step precedes the first agent step (2079 usable rows begin with a 'Warmup' user step); otherwise task_prompt_text is empty",
            "obs is publisher-truncated to 5000 chars, bounding combined feedback text",
            "the community dataset has no patch field, so patch is always empty",
        ],
        "not_reconstructable_from_public_data": [
            "per-step wall-clock latency",
            "token accounting per step",
            "exact typing of tool argument JSON (commands are stored as strings)",
        ],
        "community_step_schema_observed": {
            "distinct_keys": sorted(agg["step_keys"].keys()),
            "src_values": agg["src_values"],
            "steps_total": 1622075,
        },
        "no_trainer_used": True,
        "api_calls": 0,
        "llm_calls": 0,
    }
    (ADS / "adapter_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), "utf-8")


def main() -> int:
    agg = json.loads((WORK / "scan_aggregate.json").read_text("utf-8"))
    combos = json.loads((WORK / "scan_combo_rows.json").read_text("utf-8"))
    remote = json.loads((WORK / "remote_metadata.json").read_text("utf-8"))
    dl = json.loads((WORK / "local_download_manifest.json").read_text("utf-8"))
    gate = json.loads((AN / "phase2a_gate.json").read_text("utf-8"))
    conn = json.loads((AN / "design_connectivity.json").read_text("utf-8"))
    prov = json.loads((AN / "provenance_overlap.json").read_text("utf-8"))
    p2b = json.loads((AN / "phase2b_candidate_designs.json").read_text("utf-8"))
    probe = json.loads((ADS / "adapter_probe_results.json").read_text("utf-8"))

    eligible = [c for c in combos
                if c["usable_trajectories"] >= 100 and c["successes"] >= 20
                and c["failures"] >= 20 and c["unique_tasks_usable"] >= 30]
    us = sum(c["successes"] for c in combos)
    uf = sum(c["failures"] for c in combos)

    write_dataset(agg, remote, dl)
    write_adapter_spec(agg)

    upstream = verify_upstream()
    all_ok = all(v["match"] for v in upstream.values())
    integ = {
        "phase": "EARLYEVAL_PHASE2A",
        "name": "TERMINALBENCH_COVERAGE_AND_PROVENANCE_AUDIT",
        "frozen_upstream_verified": upstream,
        "frozen_upstream_all_match": all_ok,
        "row_classification": {
            "total_rows": agg["total_rows"],
            "TRAJECTORY_USABLE": agg["classification_reasons"].get("TRAJECTORY_USABLE", 0),
            "NO_STEPS": agg["classification_reasons"].get("NO_STEPS", 0),
            "MISSING_REWARD": agg["classification_reasons"].get("MISSING_REWARD", 0),
            "NON_BINARY_REWARD": agg["classification_reasons"].get("NON_BINARY_REWARD", 0),
            "SCHEMA_FAIL": agg["classification_reasons"].get("SCHEMA_FAIL", 0),
            "OTHER": agg["classification_reasons"].get("OTHER", 0),
            "steps_state_counts": agg["steps_state_counts"],
            "condition_flags": agg["condition_flags"],
            "classification_priority": ["MISSING_REWARD", "NON_BINARY_REWARD",
                                        "SCHEMA_FAIL", "NO_STEPS", "OTHER"],
        },
        "outcome_label_audit": {
            "unique_raw_reward_values": agg["reward_raw_values"],
            "missing": agg["reward_missing"],
            "nonbinary": agg["reward_nonbinary"],
            "documented_semantics": "reward = 1 if the agent solved the task, 0 otherwise",
            "usable_successes": us,
            "usable_failures": uf,
            "OUTCOME_SEMANTICS_CONFIRMED": (agg["reward_missing"] == 0
                                            and agg["reward_nonbinary"] == 0
                                            and set(agg["reward_raw_values"]) == {"0", "1"}),
        },
        "adapter_probe": {"adapter": probe["adapter"], "sample_n": probe["sample_n"],
                          "prefix_rows_total": probe["prefix_rows_total"],
                          "errors": len(probe["adapter_errors"])},
        "frozen_earlyeval_clone": clone_integrity(),
        "upstream_clone_modified": False,
        "frozen_scientific_materials_modified": False,
        "prior_phase_history_rewritten": False,
        "api_calls": 0, "llm_calls": 0, "cloud_cost_cny": 0,
        "new_agent_trajectories": 0, "predictor_training": 0, "lightgbm_training": 0,
    }
    (OUT / "integrity_report.json").write_text(
        json.dumps(integ, ensure_ascii=False, indent=2), "utf-8")

    files = sorted((p for p in OUT.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(OUT).as_posix().lower())
    rel = {p: p.relative_to(OUT).as_posix() for p in files}
    man_files = [p for p in files
                 if rel[p] not in ("manifest.json", "artifact_sha256sums.txt")]
    manifest = {
        "phase": "EARLYEVAL_PHASE2A",
        "name": "TERMINALBENCH_COVERAGE_AND_PROVENANCE_AUDIT",
        "type": "cross-benchmark feasibility / design audit (no predictor, no training)",
        "dataset": {
            "repo_id": remote["repo_id"], "revision": remote["sha"],
            "license": remote["license"], "rows": agg["total_rows"],
            "usable_full_trajectories": agg["rows_with_steps"],
            "task_names": agg["unique_task_names_all"],
            "models": agg["unique_models_all"],
            "scaffolds": agg["unique_agents_all"],
            "model_x_scaffold_combos": agg["unique_combos_all"],
        },
        "results": {
            "eligible_combos": len(eligible),
            "models_with_ge3_eligible_scaffolds": gate["E_crossed_structure"]["models_with_ge3_eligible_scaffolds"],
            "scaffolds_with_ge3_eligible_models": gate["E_crossed_structure"]["scaffolds_with_ge3_eligible_models"],
            "controlled_pairs": conn["section_9_common_task_support"]["controlled_pairs_total"],
            "pairs_ge30": conn["section_9_common_task_support"]["pairs_ge30"],
            "mapping": "SIMPLE_ADAPTER",
            "adapter_probe": probe["adapter"],
            "provenance_overlap": prov["overlap_classification"],
            "PHASE2A": gate["PHASE2A"],
        },
        "candidate_designs": [p2b.get("candidate_1"), p2b.get("candidate_2")],
        "frozen_upstream_all_match": all_ok,
        "api_calls": 0, "llm_calls": 0, "cloud_cost_cny": 0,
        "predictor_training": 0, "lightgbm_training": 0, "new_agent_trajectories": 0,
        "manifest_self_excludes": ["manifest.json", "artifact_sha256sums.txt"],
        "files": [{"path": rel[p], "bytes": p.stat().st_size,
                   "sha256": sha256_file(p)} for p in man_files],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")

    files = sorted((p for p in OUT.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(OUT).as_posix().lower())
    rel = {p: p.relative_to(OUT).as_posix() for p in files}
    sums = [f"{sha256_file(p)}  {rel[p]}" for p in files
            if rel[p] != "artifact_sha256sums.txt"]
    (OUT / "artifact_sha256sums.txt").write_text("\n".join(sums) + "\n", "utf-8")

    print(json.dumps({
        "artifact_entries_in_sums": len(sums),
        "manifest_entries": len(man_files),
        "upstream_all_match": all_ok,
        "status": gate["PHASE2A"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
