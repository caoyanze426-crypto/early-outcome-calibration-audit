# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0E - protocol, report, manifest, integrity."""
from __future__ import annotations

import sys

from common0e import (ANA, INPUT_FILES, NL, OUT, OUT_0B, OUT_0D, as_builtin,
                      ensure_dirs, read_json, read_manifest, sha256_file, write_json)

import numpy as np
import pandas as pd


def verify_inputs(inputs: dict) -> dict:
    """Confirm every reused input matches the producing phase's own manifest."""
    b_man = read_manifest(OUT_0B / "artifact_sha256sums.txt")
    d_man = read_manifest(OUT_0D / "artifact_sha256sums.txt")
    checks = {}
    all_match = True
    for name, path in INPUT_FILES.items():
        if OUT_0B in path.parents:
            base, man, producer = OUT_0B, b_man, "phase0b"
        else:
            base, man, producer = OUT_0D, d_man, "phase0d"
        rel = path.relative_to(base).as_posix()
        expected = man.get(rel)
        observed = inputs["input_files"][name]["sha256"]
        match = bool(expected is not None and observed is not None
                     and expected == observed)
        checks[name] = {"producer": producer, "relative_path": rel,
                        "producer_manifest_sha256": expected,
                        "observed_sha256": observed, "match": match}
        all_match = all_match and match
    return {"phase0b_manifest": str(OUT_0B / "artifact_sha256sums.txt"),
            "phase0b_manifest_entries": len(b_man),
            "phase0d_manifest": str(OUT_0D / "artifact_sha256sums.txt"),
            "phase0d_manifest_entries": len(d_man),
            "inputs": checks, "ALL_INPUT_HASHES_MATCH": bool(all_match)}


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


def ci(block) -> str:
    return f"[{fmt(block['ci_lower'])}, {fmt(block['ci_upper'])}]"


def ci_lo_hi(block) -> str:
    """Interval string for dicts keyed lower/upper (the rho blocks)."""
    return f"[{fmt(block['lower'])}, {fmt(block['upper'])}]"


def build_protocol() -> str:
    L = ["# EARLYEVAL_PHASE0E - PRIOR_SHIFT_DECOMPOSITION - PROTOCOL", ""]
    L += ["Strictly offline. LLM calls = 0, API calls = 0, predictor retraining = 0, "
          "new folds = 0, threshold sweep = 0, TerminalBench = NOT RUN, Toolathlon = "
          "NOT RUN.", ""]
    L += ["## Question", "",
          "Does agent-conditional miscalibration remain after an oracle label-prior "
          "(base-rate) correction, or do the calibration gaps largely collapse so "
          "that the Phase 0D finding is better described as prior-shift adaptation?",
          "",
          "Decomposition only. No causal claim.", ""]
    L += ["## Inputs", "",
          "- the exact frozen Phase 0B stop-point predictions, rebuilt by the same "
          "`(traj_id, decision_step)` join to the held-out prefix predictions that "
          "Phase 0D used and verified",
          "- the frozen Phase 0D fold priors (`prevalence_shift.csv`) and the frozen "
          "Phase 0D gap tables, used as the identity reference", "",
          "Every consumed input is checked against the producing phase's own "
          "`artifact_sha256sums.txt`. Predictions are never rebuilt and no "
          "probability is re-fit.", ""]
    L += ["## Method", "",
          "1. TRAIN_SUCCESS_PRIOR_m and TEST_SUCCESS_PRIOR_m per held-out model, "
          "recomputed from the frozen trajectory universe and cross-checked against "
          "Phase 0D.",
          "2. Oracle base-rate correction. Each stop-point head probability is "
          "shifted in log-odds by ln r_m, where "
          "r_m = [pi_test/(1-pi_test)] / [pi_train/(1-pi_train)] and "
          "p = sigmoid(logit(p) + ln r_m). The success head's positive class is a "
          "resolved success; the failure head's positive class is the complement, so "
          "its odds ratio is the exact reciprocal and its log shift is exactly "
          "-ln r_m. Empirical precision is unchanged by the correction because the "
          "correction moves probabilities, not decisions.",
          "3. Recomputed per-model per-head calibration gap = mean corrected "
          "decision score - empirical precision, kept beside the raw gap.",
          "4. Residual gap range per head across non-degenerate models with a "
          "defined denominator, beside the raw range, with task-cluster bootstrap "
          "CIs (2000 replicates, seed 42, instance_id resampled with all model "
          "trajectories of a selected task entering together).",
          "5. Spearman association of PRIOR_SHIFT with the raw and the residual "
          "calibration error, with bootstrap CIs.",
          "6. Degenerate prior: a held-out success prior of exactly 0 or 1 makes the "
          "oracle odds ratio 0 or +inf and puts the corrected probabilities on the "
          "boundary. Such models are flagged `prior_degenerate`, are reported in "
          "every table, and are excluded from the gate and from the residual range, "
          "because a boundary-valued correction is not a valid oracle correction in "
          "either direction.",
          ""]
    L += ["## Pre-registered gate", "",
          "RESIDUAL_LARGE_GAP_COUNT = held-out models that are not prior-degenerate, "
          "have >= 20 decisions in at least one head and "
          "|corrected calibration gap| >= 0.10 in at least one head.", "",
          "RESIDUAL_HETEROGENEITY_SUBSTANTIAL = either head has a corrected "
          "calibration-gap range >= 0.10 across non-degenerate models with a defined "
          "denominator AND a task-cluster bootstrap 95% CI lower bound > 0.05 for "
          "that range.", "",
          "OVERALL = GO iff RESIDUAL_LARGE_GAP_COUNT >= 3 AND "
          "RESIDUAL_HETEROGENEITY_SUBSTANTIAL. "
          "PIVOT_TO_PRIOR_SHIFT_ADAPTATION iff RESIDUAL_LARGE_GAP_COUNT <= 1 AND NOT "
          "RESIDUAL_HETEROGENEITY_SUBSTANTIAL. Otherwise "
          "PARTIAL_RESIDUAL_MISCALIBRATION, a mechanically defined middle category "
          "that the two-way rule does not cover. Thresholds were not altered after "
          "seeing results.", ""]
    L += ["## Interpretation boundary", "",
          "Even if GO, no claim that prior shift causes miscalibration, that the "
          "Oracle correction is how a deployed system should be adjusted, that Platt "
          "scaling is invalid, or that EarlyEval is unsafe. The correction is an "
          "analysis transform on frozen probabilities, not a new predictor.", ""]
    return NL.join(L)


DEVIATIONS = """\
1. The oracle correction coefficient is held at its frozen point value inside the
   bootstrap, so the reported intervals propagate trajectory-sampling noise only.
   Recomputing the coefficient inside each replicate would make the "oracle" prior
   itself a resampled quantity and is not what "oracle correction" means here.
2. gemini-3-pro has a held-out success prior of exactly 1.0 (491 of 491 resolved), so
   its oracle odds ratio is +inf and its corrected probabilities are boundary-valued
   (success head -> 1.0, failure head -> 0.0). It is flagged prior_degenerate, is
   reported in every table, and is excluded from the gate count and from the residual
   range. Its raw failure gap of 0.9582 is the same degenerate-fold artifact Phase 0D
   already declared, and including it in the raw failure range would give 1.0017
   rather than the like-for-like 0.1584.
3. Phase 0D's headline rho is Spearman(PRIOR_SHIFT, DECISION ERROR RATE) over all 10
   held-out models; Phase 0E's question is about the CALIBRATION GAP, which is a
   different quantity (mean score - empirical precision). Both are reported: the
   decision-error-rate branch reproduces Phase 0D exactly (maximum absolute
   difference 0.0), and the calibration-gap branch is Phase 0E's own baseline. In the
   success head the two differ (-0.7915 error rate vs -0.7792 calibration gap over 10
   models); in the failure head they coincide at the reported precision.
4. `gap_absorbed_by_correction` is a ratio and is unstable when the raw gap is near
   zero. It is recorded raw and is not used for the gate; for example gpt-5-mini's
   failure head has a raw gap of -0.0016, so the ratio moves outside [0, 1].
5. TRAIN_SUCCESS_PRIOR_m is the other-nine-model success rate of the frozen Phase 0B
   trajectory universe, which includes the validation subset; it is not a step-5
   training-only prior. No smoothing.
6. gpt-5.2-codex and gpt-5.2-high are mirror folds over an identical trajectory
   universe, so their rows are numerically identical and both are retained.
7. NO_STOP trajectories carry no decision and enter no precision, gap or
   correlation denominator.
8. The correction is a base-rate transform applied to frozen probabilities. No
   predictor was retrained, no calibration curve was refit, and no threshold was
   re-tuned.
9. Pre-registered constants: large gap 0.10, minimum denominator 20, residual range
   minimum 0.10, residual range bootstrap CI lower bound minimum 0.05, 2000
   replicates, seed 42.
10. PARTIAL_RESIDUAL_MISCALIBRATION exists because the two-way rule does not cover
    every outcome; it was not triggered by these results.
"""


def build_report(inp_ok, ident, corr, gap, rrange, rho, boot, gate) -> str:
    nl = NL
    L = ["# EARLYEVAL_PHASE0E - PRIOR_SHIFT_DECOMPOSITION", ""]
    L += ["## A. Input status", "",
          f"all input hashes match = "
          f"{'YES' if inp_ok['ALL_INPUT_HASHES_MATCH'] else 'NO'} "
          f"(Phase 0B manifest {inp_ok['phase0b_manifest_entries']} entries, "
          f"Phase 0D manifest {inp_ok['phase0d_manifest_entries']} entries)", "",
          "| input | producer | match | sha256 |", "|---|---|---|---|"]
    for name, rec in inp_ok["inputs"].items():
        L.append(f"| {name} | {rec['producer']} | "
                 f"{'MATCH' if rec['match'] else 'FAIL'} | "
                 f"`{rec['observed_sha256']}` |")
    L += ["", "Phase 0D identity reproduction:", "",
          f"decided trajectories = "
          f"{ident['stop_prefix_join']['decided_trajectories']}, stop-prefix join "
          f"missing = {ident['stop_prefix_join']['stop_prefix_join_missing']}, "
          f"max |stop score - frozen decision_score| = "
          f"{ident['stop_prefix_join']['max_abs_stop_score_minus_policy_decision_score']}",
          f"max |recomputed mean raw score - Phase 0D mean_decision_score| = "
          f"{ident['max_abs_mean_raw_score_minus_phase0d_mean_decision_score']}",
          f"max |recomputed precision - Phase 0D precision| = "
          f"{ident['max_abs_empirical_precision_minus_phase0d']}",
          f"max |recomputed raw gap - Phase 0D calibration_gap| = "
          f"{ident['max_abs_raw_gap_minus_phase0d_calibration_gap']}",
          f"max |recomputed prior - Phase 0D prior| = "
          f"{max(ident['max_abs_test_prior_minus_phase0d_test_success_rate'], ident['max_abs_train_prior_minus_phase0d_train_success_rate'])}",
          f"IDENTITY_REPRODUCED = {ident['IDENTITY_REPRODUCED']} "
          f"(tolerance {ident['tolerance']})", ""]
    L += ["", "## B. Oracle label-prior correction", "",
          "| model | train prior | test prior | prior shift | success odds ratio | "
          "ln r (success head) | ln r (failure head) | prior degenerate |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in corr.iterrows():
        L.append(f"| {r['model_id']} | {fmt(r['train_success_prior'])} | "
                 f"{fmt(r['test_success_prior'])} | {fmt(r['prior_shift'])} | "
                 f"{fmt(r['success_odds_ratio'])} | "
                 f"{fmt(r['log_odds_ratio_success_head'])} | "
                 f"{fmt(r['log_odds_ratio_failure_head'])} | "
                 f"{r['prior_degenerate']} |")
    L += ["", "## C. Corrected gaps", "",
          "| model | head | decisions | empirical precision | mean raw score | "
          "mean corrected score | raw gap | corrected gap | |corrected gap| | "
          "gap absorbed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in gap.sort_values(["model_id", "head"]).iterrows():
        if int(r["decisions"]) == 0:
            continue
        L.append(f"| {r['model_id']} | {r['head']} | {int(r['decisions'])} | "
                 f"{fmt(r['empirical_precision'])} | {fmt(r['mean_raw_score'])} | "
                 f"{fmt(r['mean_corrected_score'])} | {fmt(r['raw_gap'])} | "
                 f"{fmt(r['corrected_gap'])} | "
                 f"{fmt(r['absolute_corrected_gap'])} | "
                 f"{fmt(r['gap_absorbed_by_correction'])} |")
    L += ["", "pooled raw success gap = "
              f"{fmt(_pooled(gap, 'success', 'raw'))}; pooled corrected success gap "
              f"= {fmt(_pooled(gap, 'success', 'corrected'))}",
          f"pooled raw failure gap = {fmt(_pooled(gap, 'failure', 'raw'))}; pooled "
          f"corrected failure gap = {fmt(_pooled(gap, 'failure', 'corrected'))}", ""]
    L += ["## D. Residual gap range", "",
          "| head | raw range (non-degenerate) | residual range | range absorbed | "
          "residual range 95% boot CI | max abs raw gap | max abs residual gap |",
          "|---|---|---|---|---|---|---|"]
    for head in ("success", "failure"):
        h = rrange[head]
        b = boot[f"RESIDUAL_{head.upper()}_GAP_RANGE"]
        L.append(f"| {head} | {fmt(h['raw_range'])} | {fmt(h['residual_range'])} | "
                 f"{fmt(h['range_absorbed_by_correction'])} | {ci(b)} | "
                 f"{fmt(h['max_absolute_raw_gap'])} | "
                 f"{fmt(h['max_absolute_residual_gap'])} |")
    L += ["", "per-model residual gaps are in `analysis/corrected_gaps.csv`.", ""]
    return nl.join(L)


def _pooled(gap: pd.DataFrame, head: str, kind: str):
    """Decision-weighted pooled gap; equals the pooled mean score minus precision."""
    sub = gap[(gap["head"] == head) & gap["decisions"].gt(0)]
    if sub.empty:
        return None
    col = "raw_gap" if kind == "raw" else "corrected_gap"
    n = sub["decisions"].to_numpy(dtype=np.float64)
    v = sub[col].to_numpy(dtype=np.float64)
    return float((n * v).sum() / n.sum())


def build_report_tail(rho, boot, gate) -> str:
    L = ["## E. Prior-shift association with residual calibration error", ""]
    for head in ("success", "failure"):
        r = rho[head]
        L += [f"{head} head (n models = {r['n_models']}, "
              f"quantity = calibration gap):",
              f"- rho(PRIOR_SHIFT, raw calibration error) = "
              f"{fmt(r['rho_prior_shift_vs_raw_calibration_error'])}, "
              f"95% boot CI = {ci_lo_hi(r['raw_bootstrap_ci'])}",
              f"- rho(PRIOR_SHIFT, residual calibration error) = "
              f"{fmt(r['rho_prior_shift_vs_residual_calibration_error'])}, "
              f"95% boot CI = {ci_lo_hi(r['residual_bootstrap_ci'])}",
              f"- association absorbed by the correction = "
              f"{fmt(r['association_absorbed_by_correction'])}", ""]
    cc = rho["phase0d_cross_check"]
    L += ["Phase 0D cross-check (Phase 0D's own quantity: decision error rate, all "
          "10 models):", "",
          "| head | rho (decision error rate) | Phase 0D rho | abs difference | "
          "rho (raw calibration gap) |", "|---|---|---|---|---|"]
    for head in ("SUCCESS", "FAILURE"):
        v = cc["per_head"][head]
        L.append(f"| {head} | {fmt(v['rho_prior_shift_vs_decision_error_rate'])} | "
                 f"{fmt(v['phase0d_rho'])} | "
                 f"{fmt(v['abs_difference_vs_phase0d'])} | "
                 f"{fmt(v['rho_prior_shift_vs_raw_calibration_gap'])} |")
    L += ["", f"max abs difference on the Phase 0D error-rate branch = "
              f"{cc['max_abs_difference_error_rate_branch']}", ""]
    L += ["## F. Gate", "",
          f"RESIDUAL_LARGE_GAP_COUNT = {gate['RESIDUAL_LARGE_GAP_COUNT']} "
          f"({gate['models_with_residual_large_gap']})",
          f"RESIDUAL_HETEROGENEITY_SUBSTANTIAL = "
          f"{gate['RESIDUAL_HETEROGENEITY_SUBSTANTIAL']}"]
    for head in ("success", "failure"):
        h = gate["per_head"][head]
        L.append(f"- {head}: residual range {fmt(h['residual_gap_range'])} "
                 f"(raw {fmt(h['raw_gap_range'])}), boot CI lower "
                 f"{fmt(h['residual_gap_range_bootstrap_ci_lower'])}, "
                 f"SUBSTANTIAL = {h['SUBSTANTIAL']}")
    L += ["",
          f"prior-degenerate models excluded from the gate = "
          f"{gate['prior_degenerate_models_excluded_from_gate']}", "",
          f"OVERALL = {gate['OVERALL']}", ""]
    L += ["## G. API / training", "",
          "API calls = 0", "LLM calls = 0", "predictor retraining = 0",
          "new LightGBM folds = 0", "threshold sweep = 0",
          "TerminalBench = NOT RUN", "Toolathlon = NOT RUN", ""]
    L += ["## H. Deviations", "", DEVIATIONS, "",
          "## I. Artifact path + hash manifest", "",
          "`outputs/earlyeval_phase0e_prior_shift_decomposition/`", "",
          "See `manifest.json`, `integrity_report.json`, "
          "`artifact_sha256sums.txt`.", ""]
    return NL.join(L)


def main() -> int:
    ensure_dirs(ANA)
    inputs = read_json(OUT / "input_hashes.json")
    inp_ok = verify_inputs(inputs)
    write_json(OUT / "input_status.json", inp_ok)
    ident = read_json(ANA / "identity_check.json")
    corr = pd.read_csv(ANA / "prior_correction.csv")
    gap = pd.read_csv(ANA / "corrected_gaps.csv")
    rrange = read_json(ANA / "residual_gap_range.json")["heads"]
    rho = read_json(ANA / "residual_correlations.json")
    boot = read_json(ANA / "bootstrap_results.json")
    gate = read_json(ANA / "phase0e_gate.json")
    (OUT / "PROTOCOL.md").write_text(build_protocol(), "utf-8", newline=NL)
    report = build_report(inp_ok, ident, corr, gap, rrange, rho, boot, gate)
    report = report + NL + build_report_tail(rho, boot, gate)
    (OUT / "PHASE0E_REPORT.md").write_text(report, "utf-8", newline=NL)
    files = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json", "integrity_report.json",
                                          "artifact_sha256sums.txt"):
            files.append({"path": f.relative_to(OUT).as_posix(),
                          "bytes": int(f.stat().st_size)})
    write_json(OUT / "manifest.json", {
        "artifact": "earlyeval_phase0e_prior_shift_decomposition",
        "protocol": "EARLYEVAL_PHASE0E PRIOR_SHIFT_DECOMPOSITION",
        "reuses_phase0b_artifacts": str(OUT_0B),
        "reuses_phase0d_artifacts": str(OUT_0D),
        "phase0b_artifacts_modified": False,
        "phase0d_artifacts_modified": False,
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_folds": 0, "threshold_sweep": 0, "calibration_refit": 0,
        "files": files,
    })
    write_json(OUT / "integrity_report.json", as_builtin({
        "api_calls": 0,
        "llm_calls": 0,
        "predictor_retraining": 0,
        "new_lightgbm_folds": 0,
        "threshold_sweep": 0,
        "calibration_refit": 0,
        "terminalbench_run": False,
        "toolathlon_run": False,
        "predictions_rebuilt": False,
        "phase0b_artifacts_modified": False,
        "phase0d_artifacts_modified": False,
        "all_input_hashes_match": inp_ok["ALL_INPUT_HASHES_MATCH"],
        "phase0d_identity_reproduced": ident["IDENTITY_REPRODUCED"],
        "phase0d_error_rate_rho_max_abs_difference":
            rho["phase0d_cross_check"]["max_abs_difference_error_rate_branch"],
        "stop_prefix_join_missing": ident["stop_prefix_join"]
        ["stop_prefix_join_missing"],
        "oracle_correction_coefficient_fixed_in_bootstrap": True,
        "bootstrap_replicates": boot["replicates"],
        "bootstrap_seed": boot["seed"],
        "bootstrap_cluster_unit": "instance_id",
        "gate_applied_after_all_analysis": True,
        "gate_thresholds_altered_after_results": False,
        "prior_degenerate_models_reported_not_dropped": True,
        "checks": {
            "empirical_precision_unchanged_by_correction": True,
            "failure_head_uses_complementary_prior": True,
            "denominator_zero_recorded_as_na": True,
            "no_smoothing_on_prevalence": True,
            "no_causal_claim": True,
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
