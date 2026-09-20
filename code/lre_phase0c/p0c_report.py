# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0C - protocol, report, manifest, integrity."""
from __future__ import annotations

import sys

from common0c import (ANA, INPUT_FILES, NL, OUT, OUT_0B, as_builtin,
                      ensure_dirs, read_json, sha256_file, write_json)

import numpy as np
import pandas as pd


def verify_inputs_against_0b(inputs: dict) -> dict:
    """Confirm each reused input matches Phase 0B's own frozen hash manifest."""
    manifest_path = OUT_0B / "artifact_sha256sums.txt"
    recorded = {}
    if manifest_path.exists():
        for line in manifest_path.read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    checks = {}
    all_match = True
    for name, path in INPUT_FILES.items():
        rel = path.relative_to(OUT_0B).as_posix()
        expected = recorded.get(rel)
        observed = inputs["input_files"][name]["sha256"]
        match = bool(expected is not None and observed is not None
                     and expected == observed)
        checks[name] = {
            "relative_path_in_phase0b": rel,
            "phase0b_manifest_sha256": expected,
            "observed_sha256": observed,
            "match": match,
        }
        all_match = all_match and match
    return {
        "phase0b_manifest": str(manifest_path),
        "phase0b_manifest_entries": len(recorded),
        "inputs": checks,
        "ALL_INPUT_HASHES_MATCH": bool(all_match),
    }


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_protocol() -> str:
    L = ["# LATE_REVERSAL_EARLY_EVAL_PHASE0C - PROTOCOL", ""]
    L += ["Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, "
          "threshold sweep = 0, TerminalBench = NOT RUN, Toolathlon = NOT RUN.", ""]
    L += ["## Reused frozen inputs", "",
          "Phase 0C does not retrain the predictor and does not re-run the threshold "
          "policy. It reuses, byte-identical:", "",
          "- the Phase 0B trajectory universe (`trajectory_outcomes.csv`)",
          "- the frozen LOMO held-out policy decisions "
          "(`trajectory_policy_decisions.csv`)",
          "- the frozen 0.95 / 0.95 policy decisions and outcome classes",
          "- the frozen deterministic post-stop detectors "
          "(`detector_spec.json`)",
          "- the frozen per-model summary and late-reversal signature table",
          ""]
    L += ["## Questions", "",
          "Q1. Is the apparent error-direction asymmetry still present after "
          "conditioning on the number of SUCCESS vs FAILURE early decisions?", "",
          "Q2. Are deterministic late-reversal signatures enriched among incorrect "
          "early decisions relative to correct early decisions of the same decision "
          "class?", ""]
    L += ["## Method", "",
          "1. Decision-class confusion: conditional error rate within each decision "
          "class, kept distinct from error count share.",
          "2. Per-model conditional metrics; denominator-zero strata recorded as NA, "
          "never imputed, and class-degenerate held-out models retained as factual "
          "records.",
          "3. Signature controls: the exact frozen Phase 0B detectors, comparing "
          "FALSE_SUCCESS against CORRECT_SUCCESS and FALSE_FAILURE against "
          "CORRECT_FAILURE. No detector was redefined.",
          "4. Stratified control analysis on held-out model x decision-fraction bin "
          "([0,0.25) [0.25,0.50) [0.50,0.75) [0.75,1.00]); empty strata are kept "
          "and reported as NA.",
          "5. Standardized control rate by direct reweighting of controls to the "
          "error group's stratum distribution. No propensity model.",
          "6. Task-cluster bootstrap by instance_id, 2000 replicates, seed 42; all "
          "model trajectories of a selected task enter together and prefix rows are "
          "never resampled.",
          "7. Detector re-check: the frozen detector spec is re-applied to the frozen "
          "step table and compared against Phase 0B's own flags.",
          ""]
    L += ["## Pre-registered gate", "",
          "FS_ENRICHED = FS late-collapse signature risk difference >= +0.20 AND "
          "bootstrap 95% CI lower bound > 0.", "",
          "FF_ENRICHED = FF late-recovery signature risk difference >= +0.20 AND "
          "bootstrap 95% CI lower bound > 0.", "",
          "A. FS_ENRICHED AND FF_ENRICHED -> ROBUST_BIDIRECTIONAL_SIGNAL", "",
          "B. one direction with risk difference >= +0.30, CI lower bound > 0, "
          "errors >= 30 and representation in >= 3 models -> "
          "ROBUST_ONE_DIRECTION_SIGNAL", "",
          "Otherwise -> SIGNAL_NOT_YET_ROBUST. Thresholds were not altered after "
          "seeing results.", ""]
    L += ["## Interpretation boundary", "",
          "Only association / enrichment within this frozen EarlyEval-style setting "
          "on this SWE-bench trajectory corpus is supported. No causal claim about "
          "late trajectory activity, no general claim that EarlyEval is biased, and "
          "no claim that all early evaluators share the effect.", ""]
    return NL.join(L)


DEVIATIONS = """\
1. Detector execution: Phase 0C does not re-derive the post-stop events from the raw
   trajectories by default. It first re-runs the frozen Phase 0B detector spec against
   the frozen Phase 0B step table for every decided trajectory and confirms the Phase
   0B event flags exactly (`analysis/detector_recheck.json`); the control analysis then
   consumes the validated Phase 0B flags. Any residual disagreement would have been
   reported as a hard mismatch rather than smoothed over.
2. Standardization rule (common support): direct standardization can only compare a
   stratum that holds both error and control observations. The standardized error
   rate and the standardized control rate are therefore both computed on that shared
   stratum support, and their difference is the reported risk difference. Error mass
   in strata with no matched control cannot be compared and is excluded from the risk
   difference; how much mass that is, is reported as
   `standardization_covered_error_weight` and `n_error_in_common_support` for every
   comparison. For FALSE_FAILURE this coverage is only 0.50 because the held-out
   gemini-3-pro fold contains zero final failures and therefore supplies no
   CORRECT_FAILURE controls for its 25 false-failure trajectories.
   For transparency the mechanically composite quantity
   `risk_difference_composite_full_error_minus_covered_control` (full-group error rate
   minus the covered-stratum control rate) is also recorded. It mixes the full error
   group with a partially covered control group, so it overstates the contrast
   whenever coverage is incomplete; it is reported as a diagnostic and is not used
   for the pre-registered gate.
3. Decision-class conditioning denominator: conditional rates use decided
   trajectories only (SUCCESS or FAILURE decision). NO_STOP trajectories carry no
   decision and enter neither denominator.
4. Bootstrap undefined replicates: a replicate in which a decision class or a
   standardization target is empty yields an undefined statistic. Undefined
   replicates are dropped from the percentile calculation and the number of valid
   replicates is reported alongside every interval.
5. gemini-3-pro sensitivity: performed strictly as a declared sensitivity analysis
   because that held-out fold contains zero final failures. The primary universe for
   every headline number remains all 10 models.
6. Adapter failures: the 11 Phase 0B adapter failures are reported, not repaired. Their
   final labels are read mechanically from the frozen raw trajectory JSON
   (`info.resolved`) and are unavailable only if the raw file is missing.
"""


def build_report(inputs_ok, inputs, conf, boot, fs_block, ff_block, fs_controls,
                 ff_controls, pm, conc, gem, af_summary, gate) -> str:
    nl = NL
    L = ["# LATE_REVERSAL_EARLY_EVAL_PHASE0C - SIGNAL DECOMPOSITION AND CONTROLS", ""]
    L += ["## A. Input status", "",
          f"all Phase 0B input hashes = "
          f"{'MATCH' if inputs_ok['ALL_INPUT_HASHES_MATCH'] else 'FAIL'}",
          f"(checked against Phase 0B's own `artifact_sha256sums.txt`, "
          f"{inputs_ok['phase0b_manifest_entries']} recorded entries)", "",
          "| input | match | sha256 |", "|---|---|---|"]
    for name, rec in inputs_ok["inputs"].items():
        L.append(f"| {name} | {'MATCH' if rec['match'] else 'FAIL'} | "
                 f"`{rec['observed_sha256']}` |")
    L += ["", "## B. Decision-class metrics", "",
          f"early success total = {conf['EARLY_SUCCESS_TOTAL']}",
          f"false success = {conf['FALSE_SUCCESS']}",
          f"conditional success-decision error rate = "
          f"{fmt(conf['SUCCESS_DECISION_ERROR_RATE'])}",
          f"success-decision precision = {fmt(conf['SUCCESS_DECISION_PRECISION'])}",
          "",
          f"early failure total = {conf['EARLY_FAILURE_TOTAL']}",
          f"false failure = {conf['FALSE_FAILURE']}",
          f"conditional failure-decision error rate = "
          f"{fmt(conf['FAILURE_DECISION_ERROR_RATE'])}",
          f"failure-decision precision = {fmt(conf['FAILURE_DECISION_PRECISION'])}",
          "",
          f"difference (failure minus success conditional error rate) = "
          f"{fmt(conf['DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE'])}",
          f"95% task-cluster bootstrap CI = "
          f"[{fmt(boot['DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE']['ci_lower'])}, "
          f"{fmt(boot['DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE']['ci_upper'])}]",
          f"success error rate 95% CI = "
          f"[{fmt(boot['SUCCESS_DECISION_ERROR_RATE']['ci_lower'])}, "
          f"{fmt(boot['SUCCESS_DECISION_ERROR_RATE']['ci_upper'])}]",
          f"failure error rate 95% CI = "
          f"[{fmt(boot['FAILURE_DECISION_ERROR_RATE']['ci_lower'])}, "
          f"{fmt(boot['FAILURE_DECISION_ERROR_RATE']['ci_upper'])}]",
          "",
          "ERROR COUNT SHARE of the pooled error set (a different quantity from the "
          "conditional rates above):",
          f"- FALSE_SUCCESS share = {fmt(conf['FALSE_SUCCESS_SHARE_OF_ERRORS'])}",
          f"- FALSE_FAILURE share = {fmt(conf['FALSE_FAILURE_SHARE_OF_ERRORS'])}",
          ""]
    L += ["## C. Success decision control", "",
          f"FALSE_SUCCESS late-collapse signature = "
          f"{fmt(fs_block['signature_error_rate'])} "
          f"(n = {fs_block['n_error']})",
          f"CORRECT_SUCCESS raw control signature = "
          f"{fmt(fs_block['signature_raw_control_rate'])} "
          f"(n = {fs_block['n_control']})",
          f"standardized control signature = "
          f"{fmt(fs_block['signature_standardized_control_rate'])} "
          f"(standardized error signature = "
          f"{fmt(fs_block['signature_standardized_error_rate_common_support'])}, "
          f"covered error weight "
          f"{fmt(fs_block['standardization_covered_error_weight'])}, "
          f"{fs_block['standardization_n_error_in_common_support']} / "
          f"{fs_block['n_error']} errors on common support, "
          f"strata used {fs_block['standardization_strata_used']} / "
          f"{fs_block['standardization_strata_with_error']} strata with errors)",
          f"raw risk difference = {fmt(fs_block['risk_difference_raw'])}",
          f"standardized risk difference = "
          f"{fmt(fs_block['risk_difference_standardized'])}",
          f"composite diagnostic (full error minus covered control, not used for "
          f"the gate) = "
          f"{fmt(fs_block['risk_difference_composite_full_error_minus_covered_control'])}",
          f"95% bootstrap CI (standardized RD) = "
          f"[{fmt(boot['FS_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED']['ci_lower'])}, "
          f"{fmt(boot['FS_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED']['ci_upper'])}]",
          "",
          "| component | FALSE_SUCCESS rate | CORRECT_SUCCESS rate |",
          "|---|---|---|"]
    for c in ["POST_STOP_TEST_FAIL", "POST_STOP_ERROR_OR_TRACEBACK",
              "POST_STOP_EDIT", "POST_STOP_TEST", "POST_STOP_SUBMISSION",
              "POST_STOP_ACTIVITY", "POST_STOP_TEST_PASS",
              "FS_LATE_COLLAPSE_SIGNATURE"]:
        L.append(f"| {c} | {fmt(fs_controls['error_group'][c + '_rate'])} | "
                 f"{fmt(fs_controls['control_group'][c + '_rate'])} |")
    L += ["", "|| error group | control group |", "|---|---|---|"]
    for k, v in (("post-stop steps mean", "mean"),
                 ("post-stop steps median", "median"),
                 ("post-stop steps p25", "p25"),
                 ("post-stop steps p75", "p75"),
                 ("post-stop steps max", "max"),
                 ("post-stop zero-steps rate", "zero_steps_rate")):
        L.append(f"| {k} | {fmt(fs_controls['error_group']['post_stop_steps'][v])} | "
                 f"{fmt(fs_controls['control_group']['post_stop_steps'][v])} |")
    L += ["", "## D. Failure decision control", "",
          f"FALSE_FAILURE late-recovery signature = "
          f"{fmt(ff_block['signature_error_rate'])} (n = {ff_block['n_error']})",
          f"CORRECT_FAILURE raw control signature = "
          f"{fmt(ff_block['signature_raw_control_rate'])} "
          f"(n = {ff_block['n_control']})",
          f"standardized control signature = "
          f"{fmt(ff_block['signature_standardized_control_rate'])} "
          f"(standardized error signature = "
          f"{fmt(ff_block['signature_standardized_error_rate_common_support'])}, "
          f"covered error weight "
          f"{fmt(ff_block['standardization_covered_error_weight'])}, "
          f"{ff_block['standardization_n_error_in_common_support']} / "
          f"{ff_block['n_error']} errors on common support, "
          f"strata used {ff_block['standardization_strata_used']} / "
          f"{ff_block['standardization_strata_with_error']} strata with errors)",
          f"raw risk difference = {fmt(ff_block['risk_difference_raw'])}",
          f"standardized risk difference = "
          f"{fmt(ff_block['risk_difference_standardized'])}",
          f"composite diagnostic (full error minus covered control, not used for "
          f"the gate) = "
          f"{fmt(ff_block['risk_difference_composite_full_error_minus_covered_control'])}",
          f"95% bootstrap CI (standardized RD) = "
          f"[{fmt(boot['FF_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED']['ci_lower'])}, "
          f"{fmt(boot['FF_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED']['ci_upper'])}]",
          "",
          "| component | FALSE_FAILURE rate | CORRECT_FAILURE rate |",
          "|---|---|---|"]
    for c in ["POST_STOP_EDIT", "POST_STOP_TEST_PASS", "POST_STOP_SUBMISSION",
              "POST_STOP_TEST", "POST_STOP_ACTIVITY", "POST_STOP_TEST_FAIL",
              "POST_STOP_ERROR_OR_TRACEBACK", "FF_LATE_RECOVERY_SIGNATURE"]:
        L.append(f"| {c} | {fmt(ff_controls['error_group'][c + '_rate'])} | "
                 f"{fmt(ff_controls['control_group'][c + '_rate'])} |")
    L += ["", "|| error group | control group |", "|---|---|---|"]
    for k, v in (("post-stop steps mean", "mean"),
                 ("post-stop steps median", "median"),
                 ("post-stop steps p25", "p25"),
                 ("post-stop steps p75", "p75"),
                 ("post-stop steps max", "max"),
                 ("post-stop zero-steps rate", "zero_steps_rate")):
        L.append(f"| {k} | {fmt(ff_controls['error_group']['post_stop_steps'][v])} | "
                 f"{fmt(ff_controls['control_group']['post_stop_steps'][v])} |")
    L += ["", "## E. Per-model conditional table", "",
          "| model | early success | false success | succ err rate | early failure | "
          "false failure | fail err rate | no stop |", "|---|---|---|---|---|---|---|---|"]
    for r in pm.itertuples(index=False):
        L.append(f"| {r.model_id} | {r.early_success_decisions} | {r.false_success} | "
                 f"{fmt(r.success_decision_error_rate)} | "
                 f"{r.early_failure_decisions} | {r.false_failure} | "
                 f"{fmt(r.failure_decision_error_rate)} | {r.no_stop} |")
    L += ["", "## F. Concentration", ""]
    for label in ("FALSE_SUCCESS", "FALSE_FAILURE"):
        c = conc[label]
        L += [f"{label}: count = {c['count']}, models with error = "
              f"{c['n_models_with_error']}, unique tasks = {c['unique_tasks']}, "
              f"max trajectories from a single task = "
              f"{c['max_trajectories_single_task']} "
              f"(`{c['task_with_max_trajectories']}`)",
              "",
              f"top 5 models: "
              + ", ".join(f"{m['model_id']} {m['count']}"
                          for m in c["top5_models"]),
              "",
              f"top 10 tasks: "
              + ", ".join(f"{t['instance_id']} {t['count']}"
                          for t in c["top10_tasks"]),
              ""]
    L += ["## G. gemini-3-pro sensitivity (declared, not primary)", "",
          f"excluded model = {gem['excluded_model']} "
          f"({gem['trajectories_remaining']} trajectories remain)",
          f"rationale = {gem['rationale']}", "",
          "| metric | primary (all models) | excluding gemini-3-pro |",
          "|---|---|---|",
          f"| success decision error rate | "
          f"{fmt(conf['SUCCESS_DECISION_ERROR_RATE'])} | "
          f"{fmt(gem['decision_confusion']['SUCCESS_DECISION_ERROR_RATE'])} |",
          f"| failure decision error rate | "
          f"{fmt(conf['FAILURE_DECISION_ERROR_RATE'])} | "
          f"{fmt(gem['decision_confusion']['FAILURE_DECISION_ERROR_RATE'])} |",
          f"| FS signature error rate | {fmt(fs_block['signature_error_rate'])} | "
          f"{fmt(gem['FS_signature']['signature_error_rate'])} |",
          f"| FS raw control rate | "
          f"{fmt(fs_block['signature_raw_control_rate'])} | "
          f"{fmt(gem['FS_signature']['signature_raw_control_rate'])} |",
          f"| FS standardized RD | "
          f"{fmt(fs_block['risk_difference_standardized'])} | "
          f"{fmt(gem['FS_signature']['risk_difference_standardized'])} |",
          f"| FS common-support coverage | "
          f"{fmt(fs_block['standardization_covered_error_weight'])} | "
          f"{fmt(gem['FS_signature']['standardization_covered_error_weight'])} |",
          f"| FF signature error rate | {fmt(ff_block['signature_error_rate'])} | "
          f"{fmt(gem['FF_signature']['signature_error_rate'])} |",
          f"| FF raw control rate | "
          f"{fmt(ff_block['signature_raw_control_rate'])} | "
          f"{fmt(gem['FF_signature']['signature_raw_control_rate'])} |",
          f"| FF standardized RD | "
          f"{fmt(ff_block['risk_difference_standardized'])} | "
          f"{fmt(gem['FF_signature']['risk_difference_standardized'])} |",
          f"| FF common-support coverage | "
          f"{fmt(ff_block['standardization_covered_error_weight'])} | "
          f"{fmt(gem['FF_signature']['standardization_covered_error_weight'])} |",
          ""]
    L += ["## H. Adapter failures", "",
          f"n failures = {af_summary['n_failures']}, labels recovered = "
          f"{af_summary['labels_recovered']}, unique tasks = "
          f"{af_summary['unique_tasks']}",
          f"by model = {af_summary['by_model']}",
          f"by final label = {af_summary['by_final_label']}",
          f"by stage = {af_summary['by_stage']}",
          f"task IDs = {af_summary['task_ids']}", "",
          "No trajectory was repaired. See `analysis/adapter_failures.csv`.", ""]
    L += ["## I. Pre-registered gate", "",
          f"FS_ENRICHED = {gate['FS_ENRICHED']} "
          f"(RD {fmt(gate['FS_risk_difference_standardized'])}, "
          f"CI lower {fmt(gate['FS_bootstrap_ci_lower'])})",
          f"FF_ENRICHED = {gate['FF_ENRICHED']} "
          f"(RD {fmt(gate['FF_risk_difference_standardized'])}, "
          f"CI lower {fmt(gate['FF_bootstrap_ci_lower'])})", "",
          f"OVERALL = {gate['OVERALL']}", ""]
    L += ["## J. API / training", "",
          "API calls = 0", "LLM calls = 0", "predictor retraining = 0",
          "threshold sweep = 0", "TerminalBench = NOT RUN",
          "Toolathlon = NOT RUN", ""]
    L += ["## K. Deviations", "", DEVIATIONS, "",
          "## L. Artifact path + hash manifest", "",
          "`outputs/late_reversal_early_eval_phase0c/`", "",
          "See `manifest.json`, `integrity_report.json`, "
          "`artifact_sha256sums.txt`.", ""]
    return nl.join(L)


def main() -> int:
    ensure_dirs(ANA)
    inputs = read_json(OUT / "input_hashes.json")
    inputs_ok = verify_inputs_against_0b(inputs)
    write_json(OUT / "input_status.json", inputs_ok)
    conf = read_json(ANA / "decision_confusion.json")
    boot = read_json(ANA / "bootstrap_results.json")
    std = read_json(ANA / "standardized_rates.json")
    fs_block = std["FALSE_SUCCESS"]
    ff_block = std["FALSE_FAILURE"]
    profiles = read_json(ANA / "post_stop_profiles.json")
    fs_controls = profiles["success_decisions"]
    ff_controls = profiles["failure_decisions"]
    pm = pd.read_csv(ANA / "per_model_conditional.csv")
    conc = read_json(ANA / "concentration.json")
    gem = read_json(ANA / "gemini3pro_sensitivity.json")
    af_summary = read_json(ANA / "adapter_failure_summary.json")
    gate = read_json(ANA / "phase0c_gate.json")
    (OUT / "PROTOCOL.md").write_text(build_protocol(), "utf-8", newline=NL)
    (OUT / "PHASE0C_REPORT.md").write_text(
        build_report(inputs_ok, inputs, conf, boot, fs_block, ff_block,
                     fs_controls, ff_controls, pm, conc, gem, af_summary, gate),
        "utf-8", newline=NL)
    recheck = read_json(ANA / "detector_recheck.json")
    files = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json", "integrity_report.json",
                                          "artifact_sha256sums.txt"):
            files.append({"path": f.relative_to(OUT).as_posix(),
                          "bytes": int(f.stat().st_size)})
    write_json(OUT / "manifest.json", {
        "artifact": "late_reversal_early_eval_phase0c",
        "protocol": "LATE_REVERSAL_EARLY_EVAL_PHASE0C "
                    "SIGNAL_DECOMPOSITION_AND_CONTROL_ANALYSIS",
        "reuses_phase0b_artifacts": str(OUT_0B),
        "phase0b_artifacts_modified": False,
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "threshold_sweep": 0, "files": files,
    })
    write_json(OUT / "integrity_report.json", as_builtin({
        "api_calls": 0,
        "llm_calls": 0,
        "predictor_retraining": 0,
        "threshold_sweep": 0,
        "terminalbench_run": False,
        "toolathlon_run": False,
        "phase0b_artifacts_modified": False,
        "all_phase0b_input_hashes_match": inputs_ok["ALL_INPUT_HASHES_MATCH"],
        "detector_recheck_all_events_reproduced": recheck["all_events_reproduced"],
        "detector_recheck_decided_trajectories":
            recheck["decided_trajectories_checked"],
        "bootstrap_replicates": boot["replicates"],
        "bootstrap_seed": boot["seed"],
        "bootstrap_cluster_unit": "instance_id",
        "gate_applied_after_all_analysis": True,
        "gate_thresholds_altered_after_results": False,
        "gemini3pro_sensitivity_is_declared_not_primary": True,
        "class_degenerate_models_retained": True,
        "checks": {
            "conditional_error_rate_distinguished_from_error_count_share": True,
            "denominator_zero_recorded_as_na": True,
            "empty_strata_recorded": True,
            "no_propensity_modeling": True,
            "no_prefix_row_bootstrap": True,
        },
        "environment": {
            "python": "3.14.3",
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }))
    lines = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "artifact_sha256sums.txt":
            lines.append(f"{sha256_file(f)}  {f.relative_to(OUT).as_posix()}")
    (OUT / "artifact_sha256sums.txt").write_text(NL.join(lines) + NL, "utf-8",
                                                 newline=NL)
    print(f"report written: files={len(files)} gate={gate['OVERALL']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
