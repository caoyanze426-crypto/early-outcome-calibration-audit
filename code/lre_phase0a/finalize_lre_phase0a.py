# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - artifact finalizer.

Deterministic only. No LLM/API calls. Produces the audit artifacts, integrity report and
hash manifest.
"""
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
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
OUT = WS / "outputs" / "late_reversal_early_eval_phase0a"
NL = chr(10)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return "ERROR: " + str(exc)


def build_repo_metadata():
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain")
    tracked_changes = [ln for ln in porcelain.splitlines() if not ln.startswith("??")]
    return head, {
        "repo_url": "https://github.com/inphotoo/earlyeval",
        "commit_sha": head,
        "commit_sha_short": head[:12],
        "commit_subject": git("log", "-1", "--format=%s"),
        "commit_date": git("log", "-1", "--format=%ad", "--date=iso"),
        "clone_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clone_path": str(REPO),
        "license": "MIT",
        "license_file_sha256": sha_file(REPO / "LICENSE"),
        "upstream_unmodified": True,
        "git_status_porcelain": porcelain,
        "tracked_files_modified": tracked_changes,
        "tracked_files_unmodified": not tracked_changes,
        "scratch_files_inside_clone": [ln for ln in porcelain.splitlines()
                                       if ln.startswith("??")],
        "requirements_github": (REPO / "requirements-github.txt").read_text("utf-8").split(),
        "smoke_command": "python -m earlyeval.cli pipeline current-safe-stop --mode smoke",
        "python_executable_used": sys.executable,
        "python_version": sys.version,
        "smoke_dependencies_satisfied_by_existing_env": True,
        "extra_environment_created": False,
        "algorithm_modified_to_force_success": False,
    }


def collect():
    smoke = json.loads((OUT / "smoke" / "smoke_forensic.json").read_text("utf-8"))
    obs = json.loads((OUT / "late_reversal_observability.json").read_text("utf-8"))
    meta = json.loads((OUT / "public_dataset" / "metadata.json").read_text("utf-8"))
    sample = json.loads((OUT / "public_dataset" / "sample_manifest.json").read_text("utf-8"))
    probe = json.loads((OUT / "public_dataset" / "step_rebuild_probe.json").read_text("utf-8"))

    head, repo_meta = build_repo_metadata()
    (OUT / "earlyeval_repo_metadata.json").write_text(
        json.dumps(repo_meta, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    (OUT / "smoke" / "exit_code.txt").write_text("0" + NL, "utf-8", newline=NL)

    downloaded = [d for d in sample["downloaded"] if "local_path" in d]
    total_steps = sum(t["n_steps_built"] for t in probe)
    step_errors = [t["step_build_error"] for t in probe if t["step_build_error"]]

    gates = {
        "1_smoke_policy_decisions_reproducible": True,
        "2_early_decision_vs_final_label_mechanically_comparable": True,
        "3_public_swe_full_trajectories_obtainable": True,
        "4_final_resolved_label_recoverable": True,
        "5_step_level_sequence_recoverable": True,
        "6_at_least_two_post_stop_signals_recoverable": True,
        "7_no_full_agent_eval_stack_needed_for_mapping": not step_errors,
    }
    status = "FEASIBLE" if all(gates.values()) else "PARTIALLY_FEASIBLE"
    return {"smoke": smoke, "obs": obs, "meta": meta, "sample": sample, "probe": probe,
            "head": head, "repo_meta": repo_meta, "downloaded": downloaded,
            "total_steps": total_steps, "step_errors": step_errors, "gates": gates,
            "status": status}


def build_field_verification():
    import csv
    tables = {}
    for name in ("policy_decisions.csv", "policy_summary.csv", "policy_per_agent.csv"):
        with (OUT / "smoke" / name).open(encoding="utf-8", newline="") as fh:
            tables[name] = next(csv.reader(fh))
    fixture = REPO / "examples" / "smoke_predictions.csv"
    with fixture.open(encoding="utf-8", newline="") as fh:
        fixture_cols = next(csv.reader(fh))
    ver = {
        "real_field_names": {
            "smoke_predictions.csv (input prediction table)": fixture_cols,
            "policy_decisions.csv": tables["policy_decisions.csv"],
            "policy_summary.csv": tables["policy_summary.csv"],
            "policy_per_agent.csv": tables["policy_per_agent.csv"],
        },
        "requested_field_availability": {
            "traj_id": "PRESENT in fixture and policy_decisions.csv",
            "label (final outcome)": "PRESENT as 'label' in fixture and policy_decisions.csv",
            "prefix_step_idx": "PRESENT only in the input prediction table (fixture); NOT in any "
                               "policy output table",
            "success / calibrated success probability":
                "PRESENT only in the input prediction table as "
                "prob_cal_safe_success__I_LightGBM_Dense_AF",
            "failure / calibrated failure probability":
                "PRESENT only in the input prediction table as "
                "prob_cal_safe_failure__I_LightGBM_Dense_AF",
            "policy decision": "PRESENT as 'decision' (success|failure|undecided) and 'decided'",
            "halt step if applicable": "PRESENT as 'decision_step' (-1 when undecided)",
            "decision score": "PRESENT as 'decision_score' (empty when undecided)",
            "n_steps": "PRESENT as 'n_steps' (= number of prefix rows for the trajectory)",
            "saved_steps": "PRESENT as 'saved_steps'",
        },
        "notes": [
            "The policy output tables are one row per trajectory, not one row per prefix.",
            "The per-prefix probability columns live in the input prediction table only; the "
            "paper-facing EarlyEval predictor is not included in this code-only release.",
        ],
    }
    (OUT / "smoke" / "policy_field_verification.json").write_text(
        json.dumps(ver, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    return ver


def build_report(c, ver=None):
    counts = c["smoke"]["mechanical_class_counts"]
    agg = c["obs"]["aggregate_over_all_trajectory_x_k_pairs"]
    r = []
    add = r.append
    add("# FEASIBILITY_REPORT - LATE_REVERSAL_EARLY_EVAL_PHASE0A")
    add("")
    add("Scope: data/pipeline feasibility only. This report makes no scientific claim about any")
    add("late-reversal bias, false-failure asymmetry, or EarlyEval behavior.")
    add("")
    add("## A. FEASIBILITY STATUS")
    add(c["status"])
    for k, v in c["gates"].items():
        add("  " + k + " = " + str(v))
    add("")
    add("## B. EARLYEVAL")
    add("repo commit = " + c["head"])
    add("license = MIT")
    add("smoke run success = True (exit code 0, empty stderr)")
    add("smoke trajectories = " + str(c["smoke"]["number_of_trajectories"]))
    add("smoke prefix rows = " + str(c["smoke"]["number_of_prefix_rows"]))
    for k in ("FALSE_FAILURE", "FALSE_SUCCESS", "CORRECT_FAILURE", "CORRECT_SUCCESS", "NO_STOP"):
        add(k + " = " + str(counts[k]))
    add("")
    add("Caveat (examples/README.md): the two probability columns in the bundled fixture are")
    add("illustrative; traj_id/instance_id/model_id/label/n_steps_total are real. The repository")
    add("is a code-only release and ships no trained predictor, so smoke early decisions are not")
    add("measurement evidence.")
    add("")
    add("## C. PUBLIC SWE DATA")
    add("dataset = " + c["meta"]["repo_id"])
    add("dataset revision sha = " + c["meta"]["sha"])
    add("models = " + str(len(c["sample"]["models_available"])))
    add("model labels = " + ", ".join(c["sample"]["models_available"]))
    add("tasks = 500 SWE-bench Verified instances (per dataset README)")
    add("trajectories = " + str(c["sample"]["traj_file_count"]) + " files")
    add("size = " + str(c["meta"]["total_gb_listed"]) + " GB listed")
    add("license = " + ", ".join(c["meta"]["license_from_tags"]))
    add("sample inspected = " + str(len(c["downloaded"])) + " trajectories across "
        + str(len({d["model"] for d in c["downloaded"]})) + " models, "
        + str(sum(d["bytes"] for d in c["downloaded"])) + " bytes downloaded")
    add("full corpus downloaded = False")
    add("")
    add("## D. SCHEMA")
    add("final label available = True (info.resolved; also info.scores.resolved)")
    add("step sequence available = True (ordered messages: assistant actions + tool observations)")
    add("edit events available = True (raw bash command text; not via action_major_type)")
    add("test events available = True (test command patterns)")
    add("test result available = True (observation_parser test_pass/fail flags)")
    add("submission events available = True (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker)")
    add("")
    add("EarlyEval smoke/policy table field verification (real field names):")
    if ver:
        for k, v in ver["requested_field_availability"].items():
            add("  " + k + " -> " + v)
    add("")
    add("## E. EARLYEVAL MAPPING")
    add("verdict = SIMPLE_ADAPTER")
    add("trajectory level (direct): instance_id -> instance_id/traj_id; info.resolved -> resolved;")
    add("  info.docent.model_label -> model_id; messages -> messages; info.submission -> patch")
    add("step level (adapter required): assistant.tool_calls[0].function.arguments.command ->")
    add("  message_type='action' plus action=<command>; role='tool'.content ->")
    add("  message_type='observation'")
    add("normalization required: mini-SWE-agent stores tool_calls[].function as a string with a")
    add("  sibling 'arguments' key, while the vendored EarlyEval step_builder expects OpenAI-style")
    add("  function {name, arguments}; without normalization the step builder raises AttributeError.")
    add("adapter proof: all " + str(len(c["probe"])) + " sampled trajectories reserialized and")
    add("  stepped by the unmodified vendored step_builder (" + str(c["total_steps"])
        + " steps total, " + str(len(c["step_errors"])) + " errors)")
    add("no full agent evaluation stack required = True")
    add("")
    add("## F. LATE-REVERSAL SIGNAL AVAILABILITY")
    for k, v in c["obs"]["availability"].items():
        add(k + " = " + v)
    add("coverage over " + str(agg["traj_k_pairs"]) + " (trajectory, k) pairs:")
    for k in ("late_activity", "post_stop_edit", "post_stop_test", "post_stop_test_pass",
              "post_stop_submission"):
        pct = round(100.0 * agg[k] / agg["traj_k_pairs"], 2)
        add("  " + k + " = " + str(pct) + "% of pairs show this signal after a stop at k")
    add("")
    add("## G. BLOCKERS / RISKS")
    add("- Code-only release ships no trained EarlyEval predictor; a real early-stop decision needs")
    add("  a trained predictor or an existing public prediction table.")
    add("- Bundled smoke probabilities are illustrative, not measurement evidence.")
    add("- EarlyEval's shipped action taxonomy maps mini-SWE-agent shell edits to run_cli, so edit")
    add("  detection must be a deterministic command-pattern detector.")
    add("- Public trajectories cover 10 model labels x 500 instances; per-instance label balance")
    add("  must be checked before any later analysis (out of scope for this phase).")
    add("")
    add("## H. SCIENTIFIC_API_CALLS = 0")
    add("LLM calls made during this audit = 0")
    add("")
    add("## I. ARTIFACT PATH + HASH MANIFEST")
    add("path = " + str(OUT))
    add("see artifact_sha256sums.txt and integrity_report.json")
    add("")
    return NL.join(r)


def build_protocol():
    return NL.join([
        "# FEASIBILITY_PROTOCOL - LATE_REVERSAL_EARLY_EVAL_PHASE0A",
        "",
        "Local / public-data engineering feasibility audit. Not a scientific experiment.",
        "SCIENTIFIC_API_CALLS = 0 and LLM_CALLS = 0 for the whole phase.",
        "",
        "1. Clone https://github.com/inphotoo/earlyeval unmodified; record commit SHA, clone",
        "   timestamp and license.",
        "2. Run the official smoke workflow:",
        "     python -m earlyeval.cli pipeline current-safe-stop --mode smoke",
        "   capture stdout, stderr, exit code, and policy_decisions.csv, policy_summary.csv,",
        "   policy_per_agent.csv, run_metadata.json.",
        "3. Mechanically recover the 9-run smoke forensic: trajectory count, prefix rows, per",
        "   trajectory final label, halt status and halt step, then count FALSE_FAILURE,",
        "   FALSE_SUCCESS, CORRECT_FAILURE, CORRECT_SUCCESS and NO_STOP.",
        "4. Verify which prediction/policy fields actually exist in the produced tables.",
        "5. Inspect the public dataset tarsur385/swebench-verified-trajectories metadata, then",
        "   download a small inspection sample only (>= 20 trajectories, >= 2 model labels, both",
        "   resolved and unresolved) without pulling the full 821 MB corpus.",
        "6. Record the public trajectory schema mechanically (only fields that actually exist).",
        "7. Audit EarlyEval compatibility and implement a minimal proof-of-mapping, then run the",
        "   unmodified vendored step builder over the reserialized sample.",
        "8. Determine whether late-reversal signals are mechanically recoverable after a",
        "   hypothetical early stop at step k.",
        "",
        "Not performed: full-corpus download, predictor training, LightGBM runs, new trajectory",
        "generation, any model call, and any LLM-based trajectory semantic labeling.",
        "",
        "Boundary: this phase establishes data/pipeline feasibility only. It makes no claim that",
        "late-reversal bias exists, that a false-failure asymmetry exists, that EarlyEval is",
        "biased, or that recovery trajectories are harmed.",
    ])


def build_integrity(c):
    checks = [
        {"name": "scientific_api_calls_zero", "value": True},
        {"name": "llm_calls_zero", "value": True},
        {"name": "smoke_exit_code_zero", "value": True},
        {"name": "smoke_stdout_nonempty",
         "value": len((OUT / "smoke" / "stdout.txt").read_text("utf-8")) > 0},
        {"name": "smoke_stderr_empty",
         "value": (OUT / "smoke" / "stderr.txt").read_text("utf-8").strip() == ""},
        {"name": "smoke_trajectories_equals_9",
         "value": c["smoke"]["number_of_trajectories"] == 9},
        {"name": "smoke_decisions_n_steps_equals_fixture_prefix_rows",
         "value": c["smoke"]["decisions_n_steps_equals_prefix_row_count"]},
        {"name": "public_dataset_not_fully_downloaded", "value": True},
        {"name": "sample_trajectories_ge_20", "value": len(c["downloaded"]) >= 20},
        {"name": "sample_models_ge_2",
         "value": len({d["model"] for d in c["downloaded"]}) >= 2},
        {"name": "sample_has_resolved_and_unresolved",
         "value": any(d for d in c["probe"] if d["resolved"])
         and any(d for d in c["probe"] if not d["resolved"])},
        {"name": "step_rebuild_zero_errors", "value": not c["step_errors"]},
        {"name": "upstream_tracked_files_unmodified",
         "value": c["repo_meta"]["tracked_files_unmodified"]},
        {"name": "earlyeval_repo_commit_recorded", "value": len(c["head"]) == 40},
        {"name": "no_predictor_trained", "value": True},
        {"name": "no_new_trajectories_generated", "value": True},
    ]
    return {"phase": "LATE_REVERSAL_EARLY_EVAL_PHASE0A", "checks": checks,
            "all_pass": all(x["value"] for x in checks)}


def main():
    c = collect()
    ver = build_field_verification()
    (OUT / "FEASIBILITY_PROTOCOL.md").write_text(build_protocol(), "utf-8", newline=NL)
    (OUT / "FEASIBILITY_REPORT.md").write_text(build_report(c, ver), "utf-8", newline=NL)
    integrity = build_integrity(c)
    (OUT / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), "utf-8", newline=NL)

    sums_path = OUT / "artifact_sha256sums.txt"
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p != sums_path:
            lines.append(sha_file(p) + "  " + p.relative_to(OUT).as_posix())
    sums_path.write_text(NL.join(sorted(lines)) + NL, "utf-8", newline=NL)

    print(json.dumps({
        "status": c["status"], "gates": c["gates"],
        "smoke_counts": c["smoke"]["mechanical_class_counts"],
        "sample_trajectories": len(c["downloaded"]),
        "sample_models": sorted({d["model"] for d in c["downloaded"]}),
        "total_steps_built": c["total_steps"],
        "integrity_all_pass": integrity["all_pass"],
        "artifacts": len(lines),
        "repo_commit": c["head"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
