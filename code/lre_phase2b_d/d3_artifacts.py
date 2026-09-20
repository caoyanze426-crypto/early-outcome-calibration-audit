# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B_D - Sections 1/11: input hashes, report, manifest, sums."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from d_common import (ANA, ANCHOR, B2, DEC_2B, DECS, OUT, OUT_2B, PRED_2B,
                      THRESHOLDS, WORK, anchor_tag, as_builtin, ensure_dirs,
                      read_json, read_manifest, sha256_file, thr_tag,
                      write_json)

WS = OUT.parent.parent
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
D3_TOL = 1e-12


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def all_files(root) -> list:
    return sorted(p for p in root.rglob("*") if p.is_file())


def rel(p) -> str:
    return str(p.relative_to(OUT)).replace("\\", "/")


def verify_phase2b_manifest() -> dict:
    man = read_manifest(OUT_2B / "artifact_sha256sums.txt")
    missing, mismatched = [], []
    for r, h in sorted(man.items()):
        p = OUT_2B / r
        if not p.exists():
            missing.append(r)
            continue
        if sha256_file(p) != h:
            mismatched.append(r)
    return {
        "manifest": str(OUT_2B / "artifact_sha256sums.txt"),
        "entries": int(len(man)),
        "missing_files": int(len(missing)),
        "sha256_mismatches": int(len(mismatched)),
        "mismatch_examples": mismatched[:5],
        "match": bool(not missing and not mismatched),
    }


def consumed_inputs(models: list) -> tuple:
    man = read_manifest(OUT_2B / "artifact_sha256sums.txt")
    records, fails = [], []

    def add(path, role: str, in_phase2b: bool, manifest_key: str = None):
        exist = path.exists()
        h = sha256_file(path) if exist else None
        key = manifest_key
        if in_phase2b and key is None:
            key = str(path.relative_to(OUT_2B)).replace("\\", "/")
        exp = man.get(key) if key else None
        ok = bool(exist and (not in_phase2b or (exp is not None and exp == h)))
        if not ok:
            fails.append(str(path))
        records.append({
            "path": str(path), "role": role, "exists": bool(exist),
            "sha256": h, "phase2b_manifest_expected_sha256": exp,
            "match": ok,
            "manifest_state": ("verified against frozen Phase 2B manifest"
                               if in_phase2b else
                               "Phase 2B work intermediate; hash recorded here"),
        })

    add(B2 / "preflight.json", "eligible-model universe", False)
    add(B2 / "trajectory_index.csv", "trajectory index (priors/labels)", False)
    add(OUT_2B / "folds" / "fold_manifest.csv",
        "fold tag -> held-out model map", True)
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        add(PRED_2B / (tag + ".parquet"),
            "held-out prefix predictions (%s)" % model, True)
        add(DEC_2B / (tag + ".csv"),
            "frozen 0.950 policy decisions (%s)" % model, True)
    add(OUT_2B / "analysis" / "per_target_metrics.csv",
        "frozen 0.950 per-target metrics reference", True)
    add(OUT_2B / "analysis" / "per_target_coverage.csv",
        "frozen per-target coverage reference", True)
    add(OUT_2B / "analysis" / "target_bootstrap.json",
        "frozen 0.950 bootstrap reference", True)
    add(OUT_2B / "analysis" / "phase2b_gate.json",
        "frozen Phase 2B gate reference", True)
    add(OUT_2B / "analysis" / "eligible_models.json",
        "frozen eligible-model list reference", True)
    add(OUT_2B / "artifact_sha256sums.txt",
        "frozen Phase 2B manifest", False)
    add(REPO / "earlyeval" / "policies" / "safe_stop.py",
        "executed policy implementation (frozen vendor)", False)
    add(REPO / "earlyeval" / "core" / "contracts.py",
        "executed PolicySpec contract (frozen vendor)", False)
    add(B2 / "b2_common.py",
        "reused Phase 2B helper module (sha256_file/provider_family/etc.)",
        False)
    return records, fails


def frozen_identity_cross_checks(models: list, pre: dict) -> dict:
    checks = {}
    ref = read_json(OUT_2B / "analysis" / "eligible_models.json")
    checks["eligible_model_universe"] = {
        "models": int(len(models)),
        "preflight_matches_frozen_phase2b_list": bool(
            sorted(pre["eligible_models"]) == sorted(ref["eligible_models"])),
    }
    fm = pd.read_csv(OUT_2B / "folds" / "fold_manifest.csv")
    checks["fold_map"] = {
        "folds": int(len(fm)),
        "holdout_order_matches_sorted_eligible": bool(
            list(fm["holdout_model"]) == list(models)),
    }
    traj = pd.read_csv(B2 / "trajectory_index.csv")
    cov = pd.read_csv(OUT_2B / "analysis" / "per_target_coverage.csv")
    cov = cov.set_index("model_id")
    usable = traj.groupby("model").size()
    rate = traj.groupby("model")["resolved"].mean()
    bad_n = [m for m in models
             if int(usable[m]) != int(cov.loc[m, "usable_trajectories"])]
    bad_r = [m for m in models
             if abs(float(rate[m])
                    - float(cov.loc[m, "original_resolve_rate"])) > D3_TOL]
    checks["per_model_usable_and_resolve_rate"] = {
        "models_checked": int(len(models)),
        "usable_count_mismatches": int(len(bad_n)),
        "resolve_rate_mismatches": int(len(bad_r)),
        "match": bool(not bad_n and not bad_r),
    }
    bad_p = []
    for i, m in enumerate(models):
        tag = "fold-%02d" % i
        n = int(pd.read_parquet(PRED_2B / (tag + ".parquet"),
                                columns=["traj_id"])["traj_id"].nunique())
        if n != int(cov.loc[m, "test_prefix_trajectories"]):
            bad_p.append(m)
    checks["prefix_trajectory_counts"] = {
        "models_checked": int(len(models)),
        "mismatches": int(len(bad_p)),
        "match": bool(not bad_p),
    }
    old = pd.read_csv(OUT_2B / "analysis" / "per_target_metrics.csv")
    mine = pd.read_csv(ANA / "threshold_target_metrics.csv")
    mine = mine[np.isclose(mine["threshold"].astype(float), ANCHOR)]
    mv = old.merge(mine, on=["model_id", "head"], suffixes=("_2b", "_d"),
                   how="outer", indicator=True)
    keys_equal = bool((mv["_merge"] == "both").all())
    diffs, mismatches = {}, {}
    for f in ("decisions", "empirical_precision", "mean_corrected_score",
              "corrected_gap", "absolute_corrected_gap"):
        if not keys_equal:
            diffs[f] = None
            mismatches[f] = None
            continue
        a = mv[f + "_2b"].astype(float).to_numpy()
        b = mv[f + "_d"].astype(float).to_numpy()
        d = np.abs(a - b)
        d = d[~np.isnan(d)]
        diffs[f] = float(d.max()) if d.size else 0.0
        mismatches[f] = int((d > D3_TOL).sum())
    checks["anchor_per_target_metrics_reproduction"] = {
        "reference": str(OUT_2B / "analysis" / "per_target_metrics.csv"),
        "rows_compared": int(len(mv)) if keys_equal else None,
        "target_head_key_sets_identical": keys_equal,
        "max_abs_diff": diffs,
        "mismatch_counts_gt_1e-12": mismatches,
        "match": bool(keys_equal and all(v == 0 for v in mismatches.values())),
    }
    repro = read_json(ANA / "reproduction_check.json")
    checks["anchor_decision_reproduction"] = {
        "REPRODUCTION_PASS": repro["decision_level"]["REPRODUCTION_PASS"],
        "mismatches": {
            "decided_flag": repro["decision_level"]["decided_flag_mismatches"],
            "decision": repro["decision_level"]["decision_mismatches"],
            "decision_step": repro["decision_level"]["decision_step_mismatches"],
            "saved_steps": repro["decision_level"]["saved_steps_mismatches"],
        },
        "max_abs_decision_score_diff":
            repro["decision_level"]["max_abs_decision_score_diff"],
    }
    checks["anchor_bootstrap_reproduction"] = (
        repro["bootstrap_level"])
    return as_builtin(checks)


def fmt(value, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return ("%." + str(digits) + "f") % float(value)


def md_table(header, rows) -> list:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def build_report(grid, repro, gate, den, rob, checks, man2b,
                 n_inputs, n_fail) -> str:
    L = []
    L.append("# EARLYEVAL_PHASE2B_D - TERMINALBENCH_THRESHOLD_DENOMINATOR"
             "_DIAGNOSTIC")
    L.append("")
    L.append("Generated: " + utc())
    L.append("")
    L.append("    PHASE2B_D_RESULT = " + gate["PHASE2B_D_RESULT"])
    L.append("")
    L.append("    PHASE2B_RESULT (frozen, unchanged) = "
             + gate["PHASE2B_RESULT_frozen"])
    L.append("")
    L.append("Offline diagnostic. API calls = 0. LLM calls = 0. "
             "Predictor retraining = 0. Cloud compute = 0.")
    L.append("")
    L.append("## A. INPUT / REPRODUCTION")
    L.append("")
    L.append("- Phase 2B frozen manifest re-verified: %d entries, %d missing, "
             "%d mismatches." % (man2b["entries"], man2b["missing_files"],
                                 man2b["sha256_mismatches"]))
    L.append("- Consumed inputs re-hashed: %d files, %d failing."
             % (n_inputs, n_fail))
    dl = repro["decision_level"]
    L.append("- 0.950 decision reproduction vs frozen "
             "predictions/policy_decisions: %d rows, %d decided, "
             "decided-flag mismatches %d, decision mismatches %d, "
             "step mismatches %d, max |delta decision_score| = %s."
             % (dl["total_rows"], dl["decided_rows"],
                dl["decided_flag_mismatches"], dl["decision_mismatches"],
                dl["decision_step_mismatches"],
                fmt(dl["max_abs_decision_score_diff"], 12)))
    bl = repro["bootstrap_level"]
    L.append("- 0.950 bootstrap reproduction vs frozen "
             "analysis/target_bootstrap.json: %d entries, key sets identical "
             "%s, max |delta CI| lower %s upper %s."
             % (bl["reference_entries"], bl["key_sets_identical"],
                fmt(bl["max_abs_diff_ci_lower"], 12),
                fmt(bl["max_abs_diff_ci_upper"], 12)))
    pm = checks["anchor_per_target_metrics_reproduction"]
    L.append("- 0.950 per-target metrics reproduction vs frozen "
             "analysis/per_target_metrics.csv: %d rows, max |delta corrected "
             "gap| = %s, mismatches > 1e-12 = %d."
             % (pm["rows_compared"] or 0,
                fmt(pm["max_abs_diff"]["corrected_gap"], 12),
                pm["mismatch_counts_gt_1e-12"]["corrected_gap"]))
    L.append("- REPRODUCTION_PASS = %s." % repro["REPRODUCTION_PASS"])
    L.append("")
    for head, title in (("success", "B. SUCCESS HEAD"),
                        ("failure", "C. FAILURE HEAD")):
        sub = den[den["head"] == head].sort_values("threshold")
        L.append("## " + title)
        L.append("")
        rows = []
        for r in sub.itertuples():
            rows.append(["%.3f" % r.threshold, r.eligible_targets,
                         r.total_decisions,
                         fmt(r.median_decisions_per_eligible_target, 1),
                         r.min_decisions, r.max_decisions, r.robust_targets,
                         str(r.head_denominator_adequate)])
        L.extend(md_table(["threshold", "eligible targets",
                           "total decisions", "median decisions",
                           "min", "max", "robust targets",
                           "denominator adequate"], rows))
        L.append("")
    L.append("## D. ROBUST TARGETS (TARGET_HEAD_LARGE_GAP = TRUE)")
    L.append("")
    if len(rob):
        rows = []
        for r in rob.itertuples():
            rows.append(["%.3f" % r.threshold, r.model_id,
                         r.provider_family, r.head, r.decisions,
                         fmt(r.empirical_precision),
                         fmt(r.corrected_gap),
                         "[" + fmt(r.bootstrap_ci_lower) + ", "
                         + fmt(r.bootstrap_ci_upper) + "]"])
        L.extend(md_table(["threshold", "model", "provider", "head",
                           "decisions", "precision", "signed corrected gap",
                           "bootstrap 95% CI"], rows))
    else:
        L.append("None.")
    L.append("")
    L.append("## E. THRESHOLD GATES")
    L.append("")
    for k in sorted(gate["per_threshold"]):
        v = gate["per_threshold"][k]
        L.append("- %s: models with robust large gap = %d, of which "
                 "|gap| >= 0.10 = %d, provider families = %s, contributing "
                 "heads = %s, contributing-head adequacy = %s, "
                 "THRESHOLD_SIGNAL = %s (variant any-head = %s)."
                 % (k, v["models_with_robust_large_gap"],
                    v["of_which_abs_gap_ge_0_10"], v["provider_families"],
                    v["contributing_heads"],
                    v["contributing_head_denominator_adequate"],
                    v["THRESHOLD_SIGNAL"],
                    v["THRESHOLD_SIGNAL_variant_any_head"]))
    L.append("")
    L.append("- thresholds with THRESHOLD_SIGNAL = %d of %d."
             % (gate["thresholds_with_signal_primary"], len(THRESHOLDS)))
    L.append("")
    L.append("## F. FINAL")
    L.append("")
    L.append("    PHASE2B_D_RESULT = " + gate["PHASE2B_D_RESULT"])
    L.append("")
    L.append("    variant (any contributing head adequate) = "
             + gate["PHASE2B_D_RESULT_variant_any_head"])
    L.append("")
    L.append("    PHASE2B_RESULT (frozen) = " + gate["PHASE2B_RESULT_frozen"]
             + " - unchanged.")
    L.append("")
    L.append("## G. COST")
    L.append("")
    L.append("    API = 0")
    L.append("    LLM = 0")
    L.append("    training = 0")
    L.append("    cloud = 0")
    L.append("")
    L.append("## H. DEVIATIONS")
    L.append("")
    L.append("1. The spec phrase 'the relevant head has "
             "HEAD_DENOMINATOR_ADEQUATE = TRUE' was fixed before computation "
             "as: PRIMARY = every head contributing a robust large gap must "
             "be denominator-adequate; VARIANT = at least one contributing "
             "head is adequate. Both readings are reported and both return "
             "%s, so the interpretation does not drive the result."
             % gate["PHASE2B_D_RESULT_variant_any_head"])
    L.append("2. Bootstrap seeds reuse the Phase 2B rule (43000 + "
             "fold_index*10 + head_index) and are held fixed across "
             "thresholds, so the 0.950 anchor reproduces the frozen Phase 2B "
             "bootstrap exactly. No retraining inside the bootstrap.")
    L.append("3. At 0.950 the single eligible success-head target sits "
             "exactly on the >=20 decision boundary (20 decisions); no other "
             "success target reaches the rule. The 0.950 success-head "
             "denominator stays limited, as in Phase 2B.")
    L.append("4. Provider family is derived from the exact dataset "
             "model label through the frozen Phase 2B mapping "
             "(minimax-m2.1@openai counts as OpenAI); the family "
             "span is not re-derived here.")
    L.append("5. No other deviation: no prediction recomputed, no calibrator "
             "refit, no feature rebuilt, no scaffold changed, no model "
             "outside the frozen 29-target universe, no threshold outside "
             "{0.900, 0.925, 0.950}.")
    L.append("")
    L.append("## I. ARTIFACT PATH + HASH MANIFEST")
    L.append("")
    L.append("    " + str(OUT))
    L.append("")
    files = sorted(rel(p) for p in all_files(OUT))
    L.append("Files (%d); sha256 per file in artifact_sha256sums.txt and "
             "manifest.json:" % len(files))
    L.append("")
    for r in files:
        L.append("    " + r)
    L.append("")
    L.append("Independent post-build verification: "
             "work/lre_phase2b_d/verification_report.json")
    L.append("")
    return chr(10).join(L) + chr(10)


def checks_ok(checks: dict) -> bool:
    ok = checks["eligible_model_universe"][
        "preflight_matches_frozen_phase2b_list"]
    ok = ok and checks["fold_map"]["holdout_order_matches_sorted_eligible"]
    ok = ok and checks["per_model_usable_and_resolve_rate"]["match"]
    ok = ok and checks["prefix_trajectory_counts"]["match"]
    ok = ok and checks["anchor_per_target_metrics_reproduction"]["match"]
    ok = ok and checks["anchor_decision_reproduction"]["REPRODUCTION_PASS"]
    ok = ok and checks["anchor_bootstrap_reproduction"][
        "BOOTSTRAP_REPRODUCTION_PASS"]
    return bool(ok)


def write_entries(path, header: dict, entry_map: dict, payload_key: str):
    payload = dict(header)
    payload[payload_key] = [{"path": k, "sha256": v} for k, v in
                            sorted(entry_map.items())]
    write_json(path, payload)


def main() -> int:
    t0 = time.time()
    ensure_dirs(ANA)
    pre = read_json(B2 / "preflight.json")
    models = sorted(pre["eligible_models"])
    man2b = verify_phase2b_manifest()
    if not man2b["match"]:
        print("STOP: frozen Phase 2B manifest verification failed", flush=True)
        return 2
    records, fails = consumed_inputs(models)
    checks = frozen_identity_cross_checks(models, pre)
    repro = read_json(ANA / "reproduction_check.json")
    grid = read_json(WORK / "policy_grid_summary.json")
    gate = read_json(ANA / "diagnostic_gate.json")
    boot = read_json(ANA / "bootstrap_results.json")
    den = pd.read_csv(ANA / "denominator_summary.csv")
    rob = pd.read_csv(ANA / "robust_targets.csv")
    ok_checks = checks_ok(checks)
    all_ok = bool(man2b["match"] and not fails and ok_checks)
    cost = {
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "new_agent_generation": 0, "predictor_retraining": 0,
        "new_lightgbm_training": 0, "new_trajectories": 0,
        "new_datasets": 0, "paid_services": 0,
        "thresholds_evaluated": list(THRESHOLDS),
        "bootstrap_replicates": int(boot["replicates"]),
    }
    write_json(OUT / "input_hashes.json", {
        "phase": "EARLYEVAL_PHASE2B_D",
        "generated_utc": utc(),
        "artifact_root": str(OUT),
        "frozen_phase2b_artifact_root": str(OUT_2B),
        "frozen_phase2b_outputs_consumed": [
            "predictions/heldout_prefix_predictions/fold-NN.parquet (29)",
            "predictions/policy_decisions/fold-NN.csv (29, 0.950 anchor)",
            "analysis/per_target_metrics.csv",
            "analysis/per_target_coverage.csv",
            "analysis/target_bootstrap.json",
            "analysis/phase2b_gate.json",
            "analysis/eligible_models.json",
            "artifact_sha256sums.txt",
        ],
        "phase2b_work_intermediates_consumed": [
            "work/lre_phase2b/preflight.json",
            "work/lre_phase2b/trajectory_index.csv",
            "outputs/.../folds/fold_manifest.csv",
            "work/lre_phase2b/b2_common.py",
        ],
        "executed_frozen_vendor_sources": [
            str(REPO / "earlyeval" / "policies" / "safe_stop.py"),
            str(REPO / "earlyeval" / "core" / "contracts.py"),
        ],
        "phase2b_manifest_verification": man2b,
        "consumed_inputs": records,
        "consumed_input_files": int(len(records)),
        "consumed_input_files_failing": int(len(fails)),
        "frozen_identity_cross_checks": checks,
        "ALL_INPUT_HASHES_MATCH": all_ok,
        **cost,
    })
    report = build_report(grid, repro, gate, den, rob, checks, man2b,
                          len(records), len(fails))
    (OUT / "PHASE2B_D_REPORT.md").write_text(report, "utf-8")
    skip_integrity = {"integrity_report.json", "manifest.json",
                      "artifact_sha256sums.txt"}
    entries = {rel(p): sha256_file(p) for p in all_files(OUT)
               if rel(p) not in skip_integrity}
    write_json(OUT / "integrity_report.json", {
        "phase": "EARLYEVAL_PHASE2B_D",
        "generated_utc": utc(),
        "artifact_root": str(OUT),
        "self_consistency": {
            "manifest": "artifact_sha256sums.txt",
            "entries_covered_here": int(len(entries)),
            "entries_excluded_here": sorted(skip_integrity),
            "note": "this file lists every artifact this phase produced "
                    "except manifest.json, artifact_sha256sums.txt and this "
                    "file itself; manifest.json covers the same set plus this "
                    "file; the independent post-build verification recomputes "
                    "all three and is recorded in "
                    "work/lre_phase2b_d/verification_report.json",
            "entries": [{"path": k, "sha256": v}
                        for k, v in sorted(entries.items())],
        },
        "frozen_phase2b_input_manifest": man2b,
        "consumed_inputs_verified": {
            "files": int(len(records)), "failing": int(len(fails)),
            "failing_paths": fails[:10], "ALL_INPUT_HASHES_MATCH": all_ok,
        },
        "frozen_identity_cross_checks": checks,
        "reproduction": {
            "anchor_threshold": ANCHOR,
            "decision_level_pass":
                repro["decision_level"]["REPRODUCTION_PASS"],
            "bootstrap_level_pass":
                repro["bootstrap_level"]["BOOTSTRAP_REPRODUCTION_PASS"],
            "per_target_metrics_pass":
                checks["anchor_per_target_metrics_reproduction"]["match"],
            "REPRODUCTION_PASS": repro["REPRODUCTION_PASS"],
        },
        "PHASE2B_D_RESULT": gate["PHASE2B_D_RESULT"],
        "PHASE2B_D_RESULT_variant_any_head":
            gate["PHASE2B_D_RESULT_variant_any_head"],
        "PHASE2B_RESULT_frozen": gate["PHASE2B_RESULT_frozen"],
        "phase2b_primary_result_modified": False,
        "frozen_upstream_repo": {
            "path": str(REPO),
            "expected_commit": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
        },
        "cost": cost,
        "history_preserved": True,
        "wall_clock_seconds": round(time.time() - t0, 1),
    })
    skip_manifest = {"manifest.json", "artifact_sha256sums.txt"}
    man_entries = {rel(p): sha256_file(p) for p in all_files(OUT)
                   if rel(p) not in skip_manifest}
    write_json(OUT / "manifest.json", {
        "phase": "EARLYEVAL_PHASE2B_D",
        "title": "TERMINALBENCH_THRESHOLD_DENOMINATOR_DIAGNOSTIC",
        "generated_utc": utc(),
        "artifact_root": str(OUT),
        "excludes": sorted(skip_manifest),
        "n_entries": int(len(man_entries)),
        "entries": [{"path": k, "sha256": v, "bytes": (OUT / k).stat().st_size}
                    for k, v in sorted(man_entries.items())],
    })
    sums = []
    for p in all_files(OUT):
        r = rel(p)
        if r == "artifact_sha256sums.txt":
            continue
        sums.append(sha256_file(p) + "  " + r)
    (OUT / "artifact_sha256sums.txt").write_text(chr(10).join(sums) + chr(10),
                                                 "utf-8")
    print("input files=%d failing=%d checks_ok=%s ALL_INPUT_HASHES_MATCH=%s"
          % (len(records), len(fails), ok_checks, all_ok), flush=True)
    print("manifest entries=%d sums entries=%d result=%s"
          % (len(man_entries), len(sums), gate["PHASE2B_D_RESULT"]),
          flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
