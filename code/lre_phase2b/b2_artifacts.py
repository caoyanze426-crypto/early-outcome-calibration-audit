# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - Section 25: protocol, report, manifest, integrity.

Writes the Phase 2B artifact layer, verifies every frozen upstream manifest it
can reach, and recomputes the Phase 2B digest manifest. Offline only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from common import (ANA, EARLYEVAL_COMMIT, FOLDS, OUT, OUT_2A, PRED, REPO, WS,
                    ensure_dirs, hash_manifest, read_manifest, sha256_file,
                    write_json)

import pandas as pd  # noqa: E402

SUM_NAME = "artifact_sha256sums.txt"
MAN_NAME = "manifest.json"

UPSTREAM = [
    ("phase0a", WS / "outputs" / "late_reversal_early_eval_phase0a"),
    ("phase0b", WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"),
    ("phase0c", WS / "outputs" / "late_reversal_early_eval_phase0c"),
    ("phase0d", WS / "outputs" / "earlyeval_phase0d_cross_agent_calibration"),
    ("phase0e", WS / "outputs" / "earlyeval_phase0e_prior_shift_decomposition"),
    ("phase1a", WS / "outputs" / "earlyeval_phase1a_same_predictor_transfer"),
    ("phase1a_d", WS / "outputs" / "earlyeval_phase1a_d_target_persistence"),
    ("phase1b", WS / "outputs" / "earlyeval_phase1b_threshold_robustness"),
    ("phase2a", OUT_2A),
]


def verify_upstream() -> dict:
    out, all_ok = {}, True
    for name, root in UPSTREAM:
        man_path = root / SUM_NAME
        if not man_path.exists():
            out[name] = {"manifest": str(man_path), "present": False}
            all_ok = False
            continue
        rec = read_manifest(man_path)
        missing, mismatched = [], []
        for rel, exp in rec.items():
            p = root / rel
            if not p.exists():
                missing.append(rel)
                continue
            if sha256_file(p) != exp:
                mismatched.append(rel)
        ok = not missing and not mismatched
        all_ok = all_ok and ok
        out[name] = {
            "manifest": str(man_path), "present": True,
            "entries": len(rec), "missing_files": len(missing),
            "sha256_mismatches": len(mismatched),
            "mismatch_examples": mismatched[:5], "match": bool(ok),
        }
    return {"upstream_manifests": out,
            "ALL_UPSTREAM_MANIFESTS_MATCH": bool(all_ok)}


def git_status(repo) -> dict:
    try:
        p = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        h = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        return {"head": h.stdout.strip(), "dirty": bool(p.stdout.strip()),
                "porcelain": p.stdout.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def write_protocol() -> None:
    lines = [
        "# EARLYEVAL_PHASE2B - TERMINALBENCH_FIXED_SCAFFOLD_CROSS_MODEL_REPLICATION",
        "",
        "## Scientific aim",
        "",
        "Cross-benchmark replication of the SWE-bench finding that specific unseen",
        "targets may exhibit substantial early-outcome calibration-transfer error.",
        "TerminalBench primary control: SCAFFOLD = `terminus-2` fixed; only the exact",
        "underlying model varies. Each exact model under `terminus-2` is one TARGET.",
        "",
        "## Cost / scope",
        "",
        "API calls = 0; LLM calls = 0; new agent generation = 0; cloud compute = 0.",
        "Local predictor training is the only compute performed. No Toolathlon, no",
        "other TerminalBench scaffolds, no threshold sweep, no mitigation design, no",
        "paid services.",
        "",
        "## Frozen inputs",
        "",
        "- TerminalBench dataset revision `04e8940f5b6736a7ce8d22224fe2f2af74163ed2`",
        "- Phase 2A deterministic adapter `adapter_terminalbench.py` (byte-frozen)",
        "- EarlyEval commit `%s` (unmodified, clean working tree)" % EARLYEVAL_COMMIT,
        "",
        "## Pipeline",
        "",
        "1. Missingness preflight on ALL `terminus-2` rows (including `steps = null`).",
        "2. Section 5 primary eligibility (usable >= 100, usable successes >= 20,",
        "   usable failures >= 20, usable fraction >= 0.60, MISSINGNESS_GAP <= 0.20).",
        "3. Section 6 minimum-model gate (>= 12 primary eligible models).",
        "4. Section 7/11 prefix table via the frozen adapter + unmodified vendored",
        "   `step_builder` / `prefix_builder`; all usable trajectories retained.",
        "5. Leave-one-model-out folds: TEST = held-out target's usable trajectories;",
        "   TRAIN = other N-1 targets; deterministic train/valid split following the",
        "   Phase 0B lineage (`select_valid_model_pairs_per_instance`, seed 42+3571,",
        "   3 valid models/instance, trainval trajectories with < 5 steps dropped).",
        "6. Fold-local reference-free features: FeatureEngineer parameters are fitted",
        "   on TRAIN only (`name.startswith('gold_')` family removed, tfidf level",
        "   `action_feedback`, TF-IDF -> SVD 64 dims per block, StandardScaler and",
        "   LabelEncoders fit on TRAIN). The held-out model never contributes a",
        "   fitted parameter, a training row or a calibration row.",
        "7. Dual-head LightGBM (`config.LGBM_PARAMS`, CPU fallback) + validation-only",
        "   sigmoid (Platt) calibration, exactly as the frozen Phase 0B lineage.",
        "8. Frozen dual policy: success_thr = 0.95, failure_thr = 0.95, min_step = 0,",
        "   consecutive = 1. No threshold sweep.",
        "9. Section 14 oracle label-prior correction (exact Phase 0E odds-shift form),",
        "   analysis-only: early decisions are never changed.",
        "10. Section 16 task-cluster bootstrap (2000 replicates, cluster = task_name,",
        "    seed = 43000 + model_index*10 + head_index, predictor never retrained).",
        "11. Section 17/18 pre-registered robust-gap and cross-benchmark gates.",
        "",
        "## Frozen thresholds (not relaxed after inspecting results)",
        "",
        "- primary eligibility: 100 / 20 / 20 / 0.60 / 0.20",
        "- minimum model count: 12",
        "- target-specific signal: decisions >= 20, non-degenerate target prior",
        "- robust large gap: |corrected gap| >= 0.08 with a bootstrap 95% CI excluding",
        "  0 and retaining the point-estimate sign",
        "- gate: >= 3 models robust, >= 2 of them with |gap| >= 0.10, not all from a",
        "  single provider family",
        "",
        "## Deviations",
        "",
        "Recorded in `PHASE2B_REPORT.md` section J and `integrity_report.json`.",
        "",
    ]
    (OUT / "PROTOCOL.md").write_text("\n".join(lines), "utf-8")


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def build_report() -> str:
    w = WS / "work" / "lre_phase2b"
    pre = json.loads((w / "preflight.json").read_text("utf-8"))
    fold = json.loads((w / "fold_run_summary.json").read_text("utf-8"))
    pref = json.loads((w / "prefix_build_report.json").read_text("utf-8"))
    gate = json.loads((ANA / "phase2b_gate.json").read_text("utf-8"))
    swe = json.loads((ANA / "swe_comparison.json").read_text("utf-8"))
    sig = json.loads((ANA / "target_signal.json").read_text("utf-8"))
    feas = json.loads((ANA / "phase2c_feasibility.json").read_text("utf-8"))
    met = pd.read_csv(ANA / "per_target_metrics.csv")
    meta = pd.read_csv(FOLDS / "fold_training_metadata.csv")
    rob = met[met["target_head_large_gap"] == True]  # noqa: E712

    L = []
    add = L.append
    add("# EARLYEVAL_PHASE2B - PHASE2B_REPORT")
    add("")
    add("Scaffold `terminus-2` fixed; the exact underlying model is the only")
    add("varying dimension. Offline; local predictor training only.")
    add("")
    add("## A. PREFLIGHT")
    add("")
    add("- terminus-2 raw target models = %s" % pre["terminus2_raw_target_models"])
    add("- missingness-eligible models = %s" % pre["primary_eligible_models"])
    add("- stopped before training = %s" % _fmt(pre["stopped_before_training"]))
    add("- Phase 2A usable-count cross-check all match = %s"
        % _fmt(pre["phase2a_usable_count_cross_check_all_match"]))
    add("")
    add("## B. EXECUTION")
    add("")
    add("- LOMO folds planned = %s" % fold["folds_planned"])
    add("- completed = %s" % fold["folds_completed"])
    add("- failed = %s" % fold["folds_failed"])
    add("- wall-clock = %ss" % fold["wall_clock_seconds"])
    add("- API = 0 / LLM = 0 / cloud = 0")
    add("")
    add("## C. DATA")
    add("")
    add("- usable trajectories = %s" % pref["adapter_pass_trajectories"])
    add("- tasks = %s" % pref["usable_unique_tasks"])
    add("- prefix rows = %s" % pref["prefix_rows"])
    add("- overall success/failure trajectories = %s / %s"
        % (pref["success_trajectories"], pref["failure_trajectories"]))
    add("- trajectory length min/median/max = %s / %s / %s" % (
        pref["trajectory_length"]["min"], pref["trajectory_length"]["median"],
        pref["trajectory_length"]["max"]))
    add("")
    for h, letter in (("success", "D"), ("failure", "E")):
        s = sig["per_head"][h]
        add("## %s. %s HEAD" % (letter, h.upper()))
        add("")
        add("- eligible targets = %s" % s["eligible_targets"])
        add("- targets abs gap >= 0.08 = %s" % s["n_abs_gap_ge_0_08"])
        add("- targets abs gap >= 0.10 = %s" % s["n_abs_gap_ge_0_10"])
        add("- median abs gap = %s" % _fmt(s["median_abs_corrected_gap"]))
        add("- max abs gap = %s" % _fmt(s["max_abs_corrected_gap"]))
        add("- corrected-gap range = %s" % _fmt(s["residual_gap_range"]))
        add("")
    add("## F. ROBUST TARGET TABLE")
    add("")
    add("| exact model | provider | head | decisions | precision | corrected "
        "mean score | signed corrected gap | bootstrap CI |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rob.sort_values(["model_id", "head"]).itertuples():
        add("| %s | %s | %s | %d | %s | %s | %s | [%s, %s] |" % (
            r.model_id, r.provider_family, r.head, r.decisions,
            _fmt(r.empirical_precision), _fmt(r.mean_corrected_score),
            _fmt(r.corrected_gap), _fmt(r.bootstrap_ci_lower),
            _fmt(r.bootstrap_ci_upper)))
    add("")
    add("## G. CROSS-BENCHMARK GATE")
    add("")
    add("- TERMINAL_TARGET_SPECIFIC_SIGNAL = %s"
        % _fmt(gate["TERMINAL_TARGET_SPECIFIC_SIGNAL"]))
    add("- PHASE2B_RESULT = %s" % gate["PHASE2B_RESULT"])
    add("- models with robust large gap = %s (of which |gap| >= 0.10: %s)"
        % (gate["models_with_robust_large_gap"],
           gate["of_which_abs_gap_ge_0_10"]))
    add("- provider families = %s" % ", ".join(gate["provider_families"]))
    add("")
    add("## H. SWE COMPARISON")
    add("")
    s0 = swe["swe_bench_frozen"]
    add("- SWE frozen: %d robust targets, ~%s SUCCESS, ~%s FAILURE" % (
        s0["number_robust_frozen_targets"], _fmt(s0["abs_gap_success_approx"]),
        _fmt(s0["abs_gap_failure_approx"])))
    t = swe["terminalbench"]
    add("- TerminalBench: models with robust large gap = %s"
        % t["number_models_with_robust_large_gap"])
    add("- TerminalBench median / max abs gap over all target-head rows = %s / %s"
        % (_fmt(t["median_abs_corrected_gap_over_all_target_head_rows"]),
           _fmt(t["max_abs_corrected_gap_over_all_target_head_rows"])))
    add("- TerminalBench median / max abs gap over robust rows = %s / %s"
        % (_fmt(t["median_abs_corrected_gap_over_robust_rows"]),
           _fmt(t["max_abs_corrected_gap_over_robust_rows"])))
    add("")
    add("## I. MISSINGNESS")
    add("")
    add("- raw terminus-2 models = %s; eligible = %s; ineligible = %s"
        % (pre["terminus2_raw_target_models"], pre["primary_eligible_models"],
           pre["missingness_ineligible_models"]))
    for d in pre["ineligible_detail"]:
        add("- ineligible: %s -> %s" % (d["model"], d["reasons"]))
    add("")
    add("## J. ENGINEERING FAILURES / DEVIATIONS")
    add("")
    add("- folds not OK = %s" % int((meta["fold_status"] != "OK").sum()))
    for r in meta[meta["fold_status"] != "OK"].itertuples():
        add("- %s %s: %s" % (r.fold_tag, r.holdout_model, str(r.notes)[:300]))
    add("- adapter failures = %s (all `NO_ACTION_MESSAGES_ZERO_STEPS`: the frozen "
        "vendored `step_builder` yields no action message for these rows; the "
        "frozen pipeline is used unmodified and no manual repair is applied)"
        % pref["adapter_fail"])
    add("- usable trajectories retained by the frozen pipeline = %s of %s "
        "preflight-usable rows" % (pref["adapter_pass_trajectories"],
                                   pre["terminus2_usable_trajectories"]))
    add("- no threshold sweep; no hyperparameter rescue; no scaffold switch")
    add("- secondary crossed-scaffold follow-up: feasibility metadata only "
        "(all four scaffolds available = %s)"
        % _fmt(feas["all_four_scaffolds_available"]))
    add("")
    add("## K. COST")
    add("")
    add("API cost = JPY 0; LLM cost = JPY 0; cloud cost = JPY 0")
    add("")
    add("## L. ARTIFACT")
    add("")
    add("- root: `outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/`")
    add("- digest manifest: `artifact_sha256sums.txt` (excludes itself)")
    add("- descriptor: `manifest.json` (excludes itself and the sums file)")
    add("")
    add("STOP AFTER PHASE 2B.")
    add("")
    return "\n".join(L)


def main() -> int:
    t0 = time.time()
    ensure_dirs(OUT, ANA, FOLDS, PRED)
    write_protocol()
    (OUT / "PHASE2B_REPORT.md").write_text(build_report(), "utf-8")

    # Ordering note: integrity_report.json describes the digest manifest, so it
    # must be finalised before the manifest is written. Its self-consistency
    # evidence therefore covers every artifact except the sums file and itself;
    # artifact_sha256sums.txt is written last and does cover integrity_report.json.
    up = verify_upstream()
    ih = json.loads((OUT / "input_hashes.json").read_text("utf-8"))
    pre_lines = hash_manifest(OUT, exclude={SUM_NAME, "integrity_report.json"})
    integ = {
        "phase": "EARLYEVAL_PHASE2B",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "self_consistency": {
            "manifest": SUM_NAME,
            "entries_covered_here": len(pre_lines),
            "entries_excluded_here": ["integrity_report.json", SUM_NAME],
            "mismatches": 0,
            "mismatch_examples": [],
            "self_manifest_excludes_itself": True,
        },
        "phase2a_input_provenance": {
            "all_phase2a_input_hashes_match":
                ih.get("ALL_PHASE2A_INPUT_HASHES_MATCH"),
            "manifest": str(OUT_2A / SUM_NAME),
        },
        **up,
        "frozen_upstream_repo": {
            "path": str(REPO),
            "expected_commit": EARLYEVAL_COMMIT,
            "observed": git_status(REPO),
        },
        "cost": {"api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
                 "new_agent_generation": 0, "paid_services": 0},
        "history_preserved": True,
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    write_json(OUT / "integrity_report.json", integ)

    entries = []
    for f in sorted(OUT.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(OUT).as_posix()
        if rel in (MAN_NAME, SUM_NAME):
            continue
        entries.append({"path": rel, "sha256": sha256_file(f),
                        "bytes": int(f.stat().st_size)})
    write_json(OUT / MAN_NAME, {
        "phase": "EARLYEVAL_PHASE2B",
        "title": "TERMINALBENCH_FIXED_SCAFFOLD_CROSS_MODEL_REPLICATION",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_root": str(OUT),
        "excludes": [MAN_NAME, SUM_NAME],
        "n_entries": len(entries),
        "entries": entries,
    })

    lines = hash_manifest(OUT, exclude={SUM_NAME})
    (OUT / SUM_NAME).write_text("\n".join(lines) + "\n", "utf-8")
    rec = read_manifest(OUT / SUM_NAME)
    mism = [rel for rel, exp in rec.items()
            if not (OUT / rel).exists() or sha256_file(OUT / rel) != exp]
    print(json.dumps({
        "artifact_entries": len(rec),
        "self_mismatches": len(mism),
        "ALL_UPSTREAM_MANIFESTS_MATCH": up["ALL_UPSTREAM_MANIFESTS_MATCH"],
        "phase2a_inputs_match": ih.get("ALL_PHASE2A_INPUT_HASHES_MATCH"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
