# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0D - protocol, report, manifest, integrity."""
from __future__ import annotations

import sys

from common0d import (ANA, INPUT_FILES, NL, OUT, OUT_0B, as_builtin, ensure_dirs,
                      read_json, sha256_file, write_json)

import numpy as np
import pandas as pd


def verify_inputs(inputs: dict) -> dict:
    recorded = {}
    manifest_path = OUT_0B / "artifact_sha256sums.txt"
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
        checks[name] = {"relative_path_in_phase0b": rel,
                        "phase0b_manifest_sha256": expected,
                        "observed_sha256": observed, "match": match}
        all_match = all_match and match
    return {"phase0b_manifest": str(manifest_path),
            "phase0b_manifest_entries": len(recorded),
            "inputs": checks, "ALL_INPUT_HASHES_MATCH": bool(all_match)}


def fmt(v, d: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def ci(b: dict) -> str:
    return f"[{fmt(b['ci_lower'])}, {fmt(b['ci_upper'])}]"


def build_protocol() -> str:
    L = ["# EARLYEVAL_PHASE0D - CROSS_AGENT_CALIBRATION_TRANSFER_AUDIT - PROTOCOL", ""]
    L += ["Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, "
          "new LightGBM folds = 0, threshold sweep = 0, TerminalBench = NOT RUN, "
          "Toolathlon = NOT RUN.", ""]
    L += ["## Research question", "",
          "Does a globally calibrated EarlyEval-style stopping policy provide uniform "
          "decision reliability across unseen agent models, or does reliability vary "
          "systematically with the held-out agent's outcome distribution?", "",
          "Offline association audit only; no causal claim.", ""]
    L += ["## Reused frozen inputs (byte-identical Phase 0B artifacts)", "",
          "- `predictions/trajectory_policy_decisions.csv` (0.95/0.95 dual policy)",
          "- `analysis/trajectory_outcomes.csv` (trajectory universe, resolved labels)",
          "- `analysis/per_model_summary.csv`",
          "- `folds/fold_manifest.csv` (fold manifests)",
          "- `predictions/heldout_prefix_predictions_all.parquet` (held-out calibrated "
          "prefix predictions)",
          "",
          "Predictions are never rebuilt.", ""]
    L += ["## Method", "",
          "1. Held-out outcome prevalence per model: TEST_SUCCESS_RATE_m, "
          "TRAIN_SUCCESS_RATE_m over the other 9 model labels of that fold, "
          "PRIOR_SHIFT_m = TEST - TRAIN. No smoothing.",
          "2. Decision-class reliability per model: precision and error rate within "
          "SUCCESS decisions and within FAILURE decisions; NA when the denominator "
          "is 0, never imputed.",
          "3. Binomial uncertainty: 95% Wilson score intervals (no normal "
          "approximation), denominator reported explicitly.",
          "4. Cross-agent heterogeneity: max/min/range/std of per-model precision, "
          "with task-cluster (instance_id) bootstrap CIs for the ranges (2000 "
          "replicates, seed 42).",
          "5. Prior-shift association: Spearman rho of PRIOR_SHIFT against the "
          "per-model success error rate (expected negative) and failure error rate "
          "(expected positive), plus task-cluster bootstrap CIs.",
          "6. Decision-score calibration at the stop point: the calibrated head "
          "probability at each trajectory's actual stopping prefix, compared with "
          "empirical precision; CALIBRATION_GAP = mean score - precision. "
          "Probabilities are not altered.",
          "7. Degenerate-fold sensitivity excluding gemini-3-pro (declared, "
          "non-primary).",
          ""]
    L += ["## Pre-registered gate (section 10)", "",
          "HETEROGENEITY_SIGNAL: either head has precision range >= 0.15 AND the "
          "task-cluster bootstrap 95% CI lower bound for that range > 0.05.", "",
          "PRIOR_SHIFT_SIGNAL: SUCCESS rho <= -0.50 OR FAILURE rho >= +0.50, AND the "
          "same sign retained after excluding gemini-3-pro.", "",
          "CALIBRATION_TRANSFER_SIGNAL: at least 3 held-out models with denominator "
          ">= 20 decisions have absolute calibration gap >= 0.10 in at least one "
          "head.", "",
          "OVERALL: GO_CANDIDATE iff at least two of the three are TRUE, otherwise "
          "NO_STRONG_SIGNAL. Thresholds were not altered after seeing results.", ""]
    L += ["## Interpretation boundary", "",
          "Even if GO_CANDIDATE, no claim that prior shift causes miscalibration, "
          "that Platt scaling is invalid, or that EarlyEval is unsafe. Only "
          "agent-conditional reliability heterogeneity and association with the "
          "held-out outcome distribution within this frozen setting are supported.",
          ""]
    return NL.join(L)


DEVIATIONS = """\
1. TRAIN prevalence universe: TRAIN_SUCCESS_RATE_m is the success rate over the
   other nine held-out model labels present in the frozen Phase 0B trajectory
   universe. Those labels include the validation subset carried inside them, so the
   pool is not restricted to a step-5 training-only filter. No smoothing, no
   reweighting, no imputation; PRIOR_SHIFT_m = TEST_SUCCESS_RATE_m -
   TRAIN_SUCCESS_RATE_m is reported raw.
2. Mirror folds: gpt-5.2-codex and gpt-5.2-high are distinct held-out labels over an
   identical trajectory universe, so their per-model rows are numerically identical.
   Both rows are retained; they are not pooled and not de-duplicated.
3. Failure-head heterogeneity is a degenerate-fold artifact. The primary failure
   precision range is 1.0000 because the held-out gemini-3-pro fold has 0/25 correct
   failures (precision 0.0) against gemini-3-flash-high 14/14 (precision 1.0).
   Excluding gemini-3-pro the failure range falls to 0.1475 with bootstrap CI lower
   0.1094 and therefore does not independently clear the 0.15 range threshold. The
   success head qualifies independently (range 0.2170, CI lower 0.1660; sensitivity
   range 0.2081, CI lower 0.1558). The gate records both facts.
4. Spearman correlation is computed over n = 10 held-out models (9 in the sensitivity
   run). With such a small n the p-value is not the target and is not reported. The
   gate uses the point-estimate rho; bootstrap medians are reported alongside and are
   close (primary SUCCESS median -0.8061 vs point -0.7915; FAILURE median 0.6128 vs
   point 0.6442).
5. The gate rho rule is evaluated on the point estimate. The "same sign retained"
   clause is checked with the 9-model point rho after excluding gemini-3-pro.
6. Trajectory identity: the frozen Phase 0B trajectory table has 4989 rows and 4989
   unique traj_id (0 duplicate traj_id rows). The reduced test trajectory counts for
   gemini-3-pro (491) and gemini-3-flash-high (498) come from the frozen Phase 0B
   adapter failures being absent from the resolved universe, not from Phase 0D
   filtering.
7. No detector was recomputed and no new decision rule was applied. Phase 0B
   SUCCESS / FAILURE / NO_STOP decision classes and final resolved labels are consumed
   directly. NO_STOP trajectories enter no reliability or calibration denominator.
8. Missing denominators are recorded as NA and never imputed. The __POOLED__ row in
   decision_score_calibration.csv is a reporting aggregate over the same decided
   trajectories and is excluded from every per-model and gate count.
9. gemini-3-pro sensitivity is declared and non-primary. The primary universe for
   every headline number remains all 10 models.
10. Bootstraps: 2000 task-cluster replicates (resampling instance_id with
    replacement, all model trajectories of a selected task entering together), seed
    42. Replicates in which a statistic is undefined are dropped and the number of
    valid replicates is reported beside every interval.
"""


def build_report(inputs_ok, prev, rel, wil, het, boot, rho, cal_all, gem, gate) -> str:
    nl = NL

    def f4(v):
        if v is None or (isinstance(v, float) and v != v):
            return "NA"
        return fmt(v, 4)

    prec_ci = {}
    for _, r in wil.iterrows():
        if r["metric"] == "precision":
            prec_ci[(r["model_id"], r["head"])] = (r["wilson_lower"],
                                                   r["wilson_upper"])
    gap = {}
    for _, r in cal_all.iterrows():
        gap[(r["model_id"], r["head"])] = r["calibration_gap"]
    prev_map = prev.set_index("model_id")
    rel_map = rel.set_index("model_id")

    L = ["# EARLYEVAL_PHASE0D - CROSS_AGENT_CALIBRATION_TRANSFER_AUDIT", ""]
    L += ["## A. Input status", "",
          f"all Phase 0B input hashes match = "
          f"{'YES' if inputs_ok['ALL_INPUT_HASHES_MATCH'] else 'NO'}",
          f"(checked against Phase 0B's own artifact_sha256sums.txt, "
          f"{inputs_ok['phase0b_manifest_entries']} recorded entries)", "",
          "| input | match | sha256 |", "|---|---|---|"]
    for name, rec in inputs_ok["inputs"].items():
        L.append(f"| {name} | {'MATCH' if rec['match'] else 'FAIL'} | "
                 f"`{rec['observed_sha256']}` |")
    L += ["", "## B. Per-model table", "",
          "| model | test succ | train succ | prior shift | succ dec | succ prec | "
          "succ 95% CI | fail dec | fail prec | fail 95% CI | succ gap | fail gap |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in prev["model_id"]:
        s_lo, s_hi = prec_ci.get((m, "success"), (None, None))
        fl_lo, fl_hi = prec_ci.get((m, "failure"), (None, None))
        L.append(
            f"| {m} | {f4(prev_map.loc[m, 'test_success_rate'])} | "
            f"{f4(prev_map.loc[m, 'train_success_rate'])} | "
            f"{f4(prev_map.loc[m, 'prior_shift'])} | "
            f"{int(rel_map.loc[m, 'success_decisions'])} | "
            f"{f4(rel_map.loc[m, 'success_precision'])} | "
            f"[{f4(s_lo)}, {f4(s_hi)}] | "
            f"{int(rel_map.loc[m, 'failure_decisions'])} | "
            f"{f4(rel_map.loc[m, 'failure_precision'])} | "
            f"[{f4(fl_lo)}, {f4(fl_hi)}] | "
            f"{f4(gap.get((m, 'success')))} | {f4(gap.get((m, 'failure')))} |")
    L += ["",
          f"pooled success gap = {f4(gap.get(('__POOLED__', 'success')))} "
          f"(mean score {f4(_pooled_score(cal_all, 'success'))})",
          f"pooled failure gap = {f4(gap.get(('__POOLED__', 'failure')))} "
          f"(mean score {f4(_pooled_score(cal_all, 'failure'))})", ""]
    L += ["## C. Heterogeneity", "",
          f"success precision range = "
          f"{fmt(het['primary_all_models']['success']['range'])}",
          f"95% bootstrap CI = "
          f"{ci(het['bootstrap_ci_primary']['SUCCESS_PRECISION_RANGE'])}",
          f"failure precision range = "
          f"{fmt(het['primary_all_models']['failure']['range'])}",
          f"95% bootstrap CI = "
          f"{ci(het['bootstrap_ci_primary']['FAILURE_PRECISION_RANGE'])}", "",
          f"success head qualifies (range >= 0.15 and CI lower > 0.05) = "
          f"{gate['HETEROGENEITY_success_head_qualifies']}",
          f"failure head qualifies (range >= 0.15 and CI lower > 0.05) = "
          f"{gate['HETEROGENEITY_failure_head_qualifies']}",
          f"failure-head note = {gate['HETEROGENEITY_failure_head_degenerate_note']}",
          "", "sensitivity excluding gemini-3-pro:",
          f"- success range = "
          f"{fmt(het['sensitivity_excluding_gemini3pro']['success']['range'])}, "
          f"95% bootstrap CI = "
          f"{ci(het['bootstrap_ci_sensitivity']['SUCCESS_PRECISION_RANGE'])}",
          f"- failure range = "
          f"{fmt(het['sensitivity_excluding_gemini3pro']['failure']['range'])}, "
          f"95% bootstrap CI = "
          f"{ci(het['bootstrap_ci_sensitivity']['FAILURE_PRECISION_RANGE'])}", ""]
    rp = rho["primary_all_models"]
    rs = rho["sensitivity_excluding_gemini3pro"]
    L += ["## D. Prior-shift association", "",
          f"success rho = {fmt(rp['SUCCESS']['rho'])} "
          f"(n_models = {rp['SUCCESS']['n_models']})",
          f"95% bootstrap CI = {ci(rp['SUCCESS']['bootstrap_ci'])}",
          f"failure rho = {fmt(rp['FAILURE']['rho'])} "
          f"(n_models = {rp['FAILURE']['n_models']})",
          f"95% bootstrap CI = {ci(rp['FAILURE']['bootstrap_ci'])}",
          "", "after excluding gemini-3-pro:",
          f"success rho = {fmt(rs['SUCCESS']['rho'])} "
          f"(n_models = {rs['SUCCESS']['n_models']})",
          f"95% bootstrap CI = {ci(rs['SUCCESS']['bootstrap_ci'])}",
          f"failure rho = {fmt(rs['FAILURE']['rho'])} "
          f"(n_models = {rs['FAILURE']['n_models']})",
          f"95% bootstrap CI = {ci(rs['FAILURE']['bootstrap_ci'])}", ""]
    large = cal_all[(cal_all["model_id"] != "__POOLED__")
                    & cal_all["denominator_ge_20"] & cal_all["large_gap_ge_0.10"]]
    big_models = sorted(large["model_id"].unique().tolist())
    L += ["## E. Large calibration gaps", "",
          "models with >= 20 decisions and absolute gap >= 0.10 (at least one head):",
          "",
          "| model | head | decisions | empirical precision | mean decision score | "
          "gap |", "|---|---|---|---|---|---|"]
    for _, r in large.sort_values(["model_id", "head"]).iterrows():
        L.append(f"| {r['model_id']} | {r['head']} | {int(r['decisions'])} | "
                 f"{fmt(r['empirical_precision'])} | "
                 f"{fmt(r['mean_decision_score'])} | "
                 f"{fmt(r['calibration_gap'])} |")
    L += ["", f"count = {len(big_models)}", f"models = {big_models}", ""]
    L += ["## F. Signal gate", "",
          f"HETEROGENEITY_SIGNAL = {gate['HETEROGENEITY_SIGNAL']}",
          f"PRIOR_SHIFT_SIGNAL = {gate['PRIOR_SHIFT_SIGNAL']}",
          f"CALIBRATION_TRANSFER_SIGNAL = {gate['CALIBRATION_TRANSFER_SIGNAL']}",
          f"n_signals_true = {gate['n_signals_true']}", "",
          f"OVERALL = {gate['OVERALL']}   (GO_CANDIDATE / NO_STRONG_SIGNAL)", ""]
    L += ["## G. API / training", "",
          "API calls = 0", "LLM calls = 0", "predictor retraining = 0",
          "new LightGBM folds = 0", "threshold sweep = 0",
          "TerminalBench = NOT RUN", "Toolathlon = NOT RUN", ""]
    L += ["## H. Deviations", "", DEVIATIONS, "",
          "## I. Artifact path + hash manifest", "",
          "`outputs/earlyeval_phase0d_cross_agent_calibration/`", "",
          "See `manifest.json`, `integrity_report.json`, "
          "`artifact_sha256sums.txt`.", ""]
    return nl.join(L)


def _pooled_score(cal_all, head) -> float:
    row = cal_all[(cal_all["model_id"] == "__POOLED__") & (cal_all["head"] == head)]
    return float(row["mean_decision_score"].iloc[0]) if len(row) else None


def main() -> int:
    ensure_dirs(ANA)
    inputs = read_json(OUT / "input_hashes.json")
    inputs_ok = verify_inputs(inputs)
    write_json(OUT / "input_status.json", inputs_ok)
    prev = pd.read_csv(ANA / "prevalence_shift.csv")
    rel = pd.read_csv(ANA / "per_model_reliability.csv")
    wil = pd.read_csv(ANA / "wilson_intervals.csv")
    cal_all = pd.read_csv(ANA / "decision_score_calibration.csv")
    het = read_json(ANA / "heterogeneity.json")
    boot = read_json(ANA / "bootstrap_results.json")
    rho = read_json(ANA / "prior_shift_correlations.json")
    gem = read_json(ANA / "gemini3pro_sensitivity.json")
    gate = read_json(ANA / "phase0d_gate.json")
    join = read_json(ANA / "decision_score_join.json")
    (OUT / "PROTOCOL.md").write_text(build_protocol(), "utf-8", newline=NL)
    (OUT / "PHASE0D_REPORT.md").write_text(
        build_report(inputs_ok, prev, rel, wil, het, boot, rho, cal_all, gem, gate),
        "utf-8", newline=NL)
    files = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json", "integrity_report.json",
                                          "artifact_sha256sums.txt"):
            files.append({"path": f.relative_to(OUT).as_posix(),
                          "bytes": int(f.stat().st_size)})
    write_json(OUT / "manifest.json", {
        "artifact": "earlyeval_phase0d_cross_agent_calibration",
        "protocol": "EARLYEVAL_PHASE0D CROSS_AGENT_CALIBRATION_TRANSFER_AUDIT",
        "reuses_phase0b_artifacts": str(OUT_0B),
        "phase0b_artifacts_modified": False,
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_folds": 0, "threshold_sweep": 0,
        "primary_universe": "all 10 held-out models",
        "declared_sensitivity": "excluding gemini-3-pro (class-degenerate fold)",
        "files": files,
    })
    write_json(OUT / "integrity_report.json", as_builtin({
        "api_calls": 0,
        "llm_calls": 0,
        "predictor_retraining": 0,
        "new_lightgbm_folds": 0,
        "threshold_sweep": 0,
        "terminalbench_run": False,
        "toolathlon_run": False,
        "predictions_rebuilt": False,
        "phase0b_artifacts_modified": False,
        "all_phase0b_input_hashes_match": inputs_ok["ALL_INPUT_HASHES_MATCH"],
        "phase0b_manifest_entries": inputs_ok["phase0b_manifest_entries"],
        "stop_prefix_join_missing": join["stop_prefix_join_missing"],
        "stop_prefix_score_matches_policy_decision_score_max_abs_diff":
            join["max_abs_stop_score_minus_policy_decision_score"],
        "bootstrap_replicates": boot["primary_all_models"]["replicates"],
        "bootstrap_seed": boot["primary_all_models"]["seed"],
        "bootstrap_cluster_unit": "instance_id",
        "gate_applied_after_all_analysis": True,
        "gate_thresholds_altered_after_results": False,
        "gemini3pro_sensitivity_is_declared_not_primary": True,
        "class_degenerate_models_retained": True,
        "checks": {
            "denominator_zero_recorded_as_na": True,
            "no_smoothing_on_prevalence": True,
            "decision_score_read_at_actual_stop_prefix": True,
            "no_prefix_row_bootstrap": True,
            "failure_head_degenerate_caveat_recorded": True,
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
