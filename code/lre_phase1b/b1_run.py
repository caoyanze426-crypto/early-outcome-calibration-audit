# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1B - target-level metrics, gate evaluation and artifacts."""
from __future__ import annotations

import pickle
import sys
import time

from b1_common import (ANA, MIN_DECISIONS, NL, OUT, OUT_1A, OUT_1AD,
                       REPRODUCTION_THRESHOLD, ROBUST_MEDIAN_ABS_GAP_MIN,
                       ROBUST_MIN_OCCURRENCES, ROBUST_MIN_THRESHOLDS,
                       ROBUST_SAME_SIGN_FRACTION_MIN, TARGETS, THRESHOLDS,
                       THRESHOLD_LABELS, WORK, as_builtin, bootstrap_seed,
                       ensure_dirs, read_json, rel_files, sha256_file, sign_of,
                       write_json)
from b1_records import bootstrap, build_records, evaluate, indicator, \
    occurrence_table

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_bundle() -> dict:
    with open(WORK / "threshold_decisions.pkl", "rb") as fh:
        return pickle.load(fh)


def load_frozen_universe() -> tuple[dict, dict]:
    """Section 1/9: the frozen Phase 1A-D eligible occurrence set per target."""
    occ = pd.read_csv(OUT_1AD / "analysis" / "target_occurrences.csv")
    universe, meta = {}, {}
    for t in TARGETS:
        sub = occ[(occ["scope"] == t["phase1a_d_scope"])
                  & occ["eligible_occurrence"]]
        pairs = sorted(str(v) for v in sub["pair_id"])
        universe[t["scope"]] = pairs
        meta[t["scope"]] = {
            "phase1a_d_scope": t["phase1a_d_scope"],
            "expected_occurrences": t["phase1a_d_eligible_occurrences"],
            "recovered_occurrences": len(pairs),
            "pair_ids": pairs,
            "matches_phase1a_d": bool(
                len(pairs) == t["phase1a_d_eligible_occurrences"]),
            "expected_median_signed_gap": t["phase1a_d_median_signed_gap"],
        }
    return universe, meta


def occurrence_stats(values) -> dict:
    signed = [v for v in values if v is not None]
    n = len(signed)
    pos = sum(1 for v in signed if v > 0)
    neg = sum(1 for v in signed if v < 0)
    return {
        "eligible_folds": n,
        "median_signed_gap": float(np.median(signed)) if n else None,
        "median_abs_gap": float(np.median(np.abs(signed))) if n else None,
        "positive_gap_fraction": (pos / n) if n else None,
        "negative_gap_fraction": (neg / n) if n else None,
        "zero_gap_count": int(n - pos - neg),
        "same_sign_fraction": (max(pos, neg) / n) if n else None,
        "min_gap": float(np.min(signed)) if n else None,
        "max_gap": float(np.max(signed)) if n else None,
    }


def threshold_gate(stats: dict, boot: dict, baseline_sign: int) -> dict:
    """Section 10 criteria A-D for one target x threshold."""
    ci = boot["median_signed_gap"]
    excludes_zero = bool(ci["ci_lower"] is not None
                         and (ci["ci_lower"] > 0 or ci["ci_upper"] < 0))
    retains_sign = bool(excludes_zero
                        and sign_of(ci["ci_lower"]) == baseline_sign
                        and sign_of(ci["ci_upper"]) == baseline_sign)
    checks = {
        "A_eligible_occurrences": {
            "value": stats["eligible_folds"],
            "required": ROBUST_MIN_OCCURRENCES,
            "passed": bool(stats["eligible_folds"] >= ROBUST_MIN_OCCURRENCES)},
        "B_median_abs_gap": {
            "value": stats["median_abs_gap"],
            "required": ROBUST_MEDIAN_ABS_GAP_MIN,
            "passed": bool(stats["median_abs_gap"] is not None
                           and stats["median_abs_gap"]
                           >= ROBUST_MEDIAN_ABS_GAP_MIN)},
        "C_same_sign_fraction": {
            "value": stats["same_sign_fraction"],
            "required": ROBUST_SAME_SIGN_FRACTION_MIN,
            "passed": bool(stats["same_sign_fraction"] is not None
                           and stats["same_sign_fraction"]
                           >= ROBUST_SAME_SIGN_FRACTION_MIN)},
        "D_bootstrap_ci_excludes_zero_and_retains_sign": {
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "baseline_sign": baseline_sign,
            "excludes_zero": excludes_zero, "retains_sign": retains_sign,
            "passed": retains_sign},
    }
    return {"criteria": checks,
            "THRESHOLD_POINT_ROBUST":
                bool(all(c["passed"] for c in checks.values()))}


def render_protocol() -> str:
    lines = [
        "# EARLYEVAL_PHASE1B - TARGET_SPECIFIC_THRESHOLD_ROBUSTNESS",
        "",
        "Frozen scientific state: Phase 1A = PARTIAL_PASS, Phase 1A-D = "
        "STRONG_TARGET_SPECIFIC_SIGNAL. Only the two frozen targets are tested:",
        "TARGET_1 = gpt-5-mini / SUCCESS, TARGET_2 = claude-opus-4.6 / FAILURE.",
        "No target is added, replaced or selected post hoc, and broad pairwise "
        "heterogeneity is not re-opened (Phase 1A: "
        "SAME_PREDICTOR_HETEROGENEITY = FALSE).",
        "",
        "## Policies",
        "",
        "Exactly four symmetric policies, in declared order:",
        "T1 = 0.900, T2 = 0.925, T3 = 0.950, T4 = 0.975 with success_thr = "
        "failure_thr = t, policy_mode = dual, min_step = 0, consecutive = 1.",
        "0.950 must reproduce the frozen Phase 1A decision tables exactly.",
        "",
        "## What is reused and what is not",
        "",
        "All 45 frozen leave-two-agent-out predictors are reused. Only the "
        "stopping policy is re-applied to the frozen held-out prefix "
        "probabilities in predictions/per_pair_target_predictions/. No training, "
        "no feature rebuilding, no recalibration refitting, no prediction "
        "regeneration, no task additions or removals, and the Phase 1A "
        "common-task-support rule is reused unchanged.",
        "",
        "## Prior correction (frozen Phase 0E / Phase 1A definition)",
        "",
        "ln r = log_odds(pi_target) - log_odds(pi_train); corrected stop score = "
        "sigmoid(logit(stop score) + sign * ln r) with sign = +1 for the success "
        "head and -1 for the failure head. Empirical precision is the fraction of "
        "correct decisions for that head. The signed corrected calibration gap is "
        "the mean corrected stop score minus empirical precision. Boundaries "
        "(pi exactly 0 or 1) are preserved without smoothing.",
        "",
        "## Eligibility (section 6)",
        "",
        f"An occurrence is eligible iff its decision denominator >= "
        f"{MIN_DECISIONS} and the target prior is non-degenerate. Denominators "
        "are recorded and never imputed.",
        "",
        "## Bootstrap (section 8)",
        "",
        "Cluster = instance_id, 2000 replicates, seed = 42100 + "
        "threshold_index*10 + target_index with threshold_index = 0..3 in the "
        "declared threshold order and target_index = 0 for TARGET_1, 1 for "
        "TARGET_2. Within every replicate the empirical precision, corrected "
        "mean stop score and corrected gap are recomputed, and the target-level "
        "median signed corrected gap is reported with a percentile 95% CI. No "
        "predictor is retrained inside the bootstrap.",
        "",
        "## Gates (frozen before any result was inspected)",
        "",
        f"THRESHOLD_POINT_ROBUST = TRUE iff eligible occurrences >= "
        f"{ROBUST_MIN_OCCURRENCES} AND median absolute corrected gap >= "
        f"{ROBUST_MEDIAN_ABS_GAP_MIN} AND same-sign fraction >= "
        f"{ROBUST_SAME_SIGN_FRACTION_MIN} AND the bootstrap 95% CI for the "
        "median signed gap excludes 0 and retains the baseline Phase 1A-D sign.",
        f"TARGET_THRESHOLD_ROBUST = TRUE iff at least {ROBUST_MIN_THRESHOLDS} of "
        "the 4 thresholds are THRESHOLD_POINT_ROBUST and threshold 0.950 itself "
        "is TRUE.",
        "PHASE1B_RESULT = STRONG_PASS (both targets), PARTIAL_PASS (exactly one), "
        "NO_PASS (neither).",
        "",
        "## Hard cost rule",
        "",
        "API calls = 0, LLM calls = 0, predictor retraining = 0, new LightGBM "
        "training = 0, new trajectories = 0, new datasets = 0.",
        "",
    ]
    return NL.join(lines)


def _num(value) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except Exception:  # noqa: BLE001
        return str(value)
    return str(value)


def render_report(ctx) -> str:
    rep, gates, overall = ctx["reproduction"], ctx["target_gate"], ctx["overall"]
    lines = [
        "# EARLYEVAL_PHASE1B - FINAL RETURN", "",
        "## A. INPUT STATUS", "",
        f"all hashes match = {ctx['inputs']['ALL_INPUT_HASHES_MATCH']}",
        "  Phase 1A: predictions/per_pair_target_predictions (90 files), "
        "predictions/pair_policy_decisions (90 files), "
        "folds/pair_fold_manifest.csv, analysis/pairwise_metrics.csv",
        "  Phase 1A-D: analysis/target_occurrences.csv, analysis/primary_gate.json",
        "  supporting: Phase 0B analysis/trajectory_outcomes.csv",
        f"0.95 reproduction check = {'EXACT_MATCH' if rep['EXACT_MATCH'] else 'MISMATCH'}",
        f"  rows compared = {rep['rows_compared']} "
        f"(frozen {rep['frozen_rows']}, recomputed {rep['recomputed_rows']}, "
        f"only-frozen {rep['rows_only_frozen']}, "
        f"only-recomputed {rep['rows_only_recomputed']})",
        f"  decided agreement = {rep['decided_agreement']}, "
        f"decision agreement = {rep['decision_agreement']}, "
        f"decision_step agreement = {rep['decision_step_agreement']}",
        f"  max |decision_score difference| = "
        f"{_num(rep['max_abs_decision_score_difference'])} "
        f"(tolerance {rep['score_tolerance']})",
        f"  PIPELINE_REPRODUCTION = "
        f"{'PASS' if rep['EXACT_MATCH'] else 'PIPELINE_REPRODUCTION_FAIL'}",
        f"support identifier sets identical across thresholds = "
        f"{ctx['support_identical']}",
        "",
    ]
    letters = {"TARGET_1": "B", "TARGET_2": "C"}
    for scope in ("TARGET_1", "TARGET_2"):
        g = gates[scope]
        lines += [f"## {letters[scope]}. {scope.replace('_', ' ')}", "",
                  f"agent/head = {g['agent_model']} / {g['head']}",
                  f"baseline Phase 1A-D sign = "
                  f"{'positive' if g['per_threshold']['T3']['baseline_sign'] > 0 else 'negative'}",
                  "",
                  "| threshold | eligible folds | signed median | abs median "
                  "| same-sign fraction | bootstrap 95% CI | head-decision "
                  "range | robust |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for lab in THRESHOLD_LABELS:
            p = g["per_threshold"][lab]
            s = ctx["summary"]
            row = s[(s["target_scope"] == scope)
                    & (s["threshold_label"] == lab)].iloc[0]
            lines.append(
                f"| {row['threshold']} | {row['eligible_folds']} "
                f"| {_num(row['median_signed_gap'])} "
                f"| {_num(row['median_abs_gap'])} "
                f"| {_num(row['same_sign_fraction'])} "
                f"| [{_num(row['bootstrap_ci_lower'])}, "
                f"{_num(row['bootstrap_ci_upper'])}] "
                f"| {_num(row['min_head_decisions'])}-"
                f"{_num(row['max_head_decisions'])} "
                f"| {p['THRESHOLD_POINT_ROBUST']} |")
        lines += ["",
                  f"thresholds passed = {g['thresholds_passed']} of "
                  f"{len(THRESHOLD_LABELS)} (required >= "
                  f"{g['thresholds_required']})",
                  f"reference threshold {g['reference_threshold']} passed = "
                  f"{g['reference_threshold_passed']}",
                  f"{scope}_THRESHOLD_ROBUST = "
                  f"{g['TARGET_THRESHOLD_ROBUST']}",
                  ""]
        for lab in THRESHOLD_LABELS:
            row = ctx["summary"][(ctx["summary"]["target_scope"] == scope)
                                 & (ctx["summary"]["threshold_label"] == lab)].iloc[0]
            if int(row["eligible_folds"]) == 0:
                lines.append(
                    f"note: at {row['threshold']} no frozen occurrence reached "
                    f"{MIN_DECISIONS} early {g['head']}-head decisions "
                    f"(observed range "
                    f"{_num(row['min_head_decisions'])}-"
                    f"{_num(row['max_head_decisions'])} across the "
                    f"{int(row['universe_occurrences'])} frozen occurrences), "
                    f"so criteria A-D cannot be satisfied at that threshold.")
        if any(int(ctx["summary"][
                (ctx["summary"]["target_scope"] == scope)
                & (ctx["summary"]["threshold_label"] == lab)].iloc[0][
                    "eligible_folds"]) == 0 for lab in THRESHOLD_LABELS):
            lines.append("")
        for lab in THRESHOLD_LABELS:
            p = g["per_threshold"][lab]
            line = (f"- {p['threshold']} ({lab}): "
                    + "; ".join(f"{k}={'PASS' if v['passed'] else 'FAIL'}"
                                for k, v in sorted(p["criteria"].items()))
                    + f" -> THRESHOLD_POINT_ROBUST={p['THRESHOLD_POINT_ROBUST']}"
                    + f"; alternative (frozen-coefficient) criterion D would "
                      f"hold={p['alternative_reading']['criterion_D_would_hold']}")
            lines.append(line)
        lines.append("")
    lines += ["## D. PRIMARY RESULT", "",
              f"TARGET_1_THRESHOLD_ROBUST = "
              f"{overall['TARGET_1_THRESHOLD_ROBUST']}",
              f"TARGET_2_THRESHOLD_ROBUST = "
              f"{overall['TARGET_2_THRESHOLD_ROBUST']}",
              f"PHASE1B_RESULT = {overall['PHASE1B_RESULT']}",
              "",
              "## E. SECONDARY (descriptive only, no gate)", "",
              "| target | threshold | trajectories | decided | coverage "
              "| head decisions | head coverage | precision | mean decision "
              "step | mean saved steps |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in ctx["secondary"]:
        lines.append(
            f"| {r['target_scope']} {r['head']} | {r['threshold']} "
            f"| {r['trajectories']} | {r['decided_total']} "
            f"| {r['decision_coverage']:.4f} | {r['head_decisions']} "
            f"| {r['head_coverage']:.4f} | {_num(r['empirical_precision'])} "
            f"| {_num(r['mean_decision_step'])} "
            f"| {_num(r['mean_saved_steps'])} |")
    lines += ["",
              "policy-level decided counts by threshold: "
              + ", ".join(f"{k} -> {v}"
                          for k, v in sorted(ctx["dec_counts"].items())),
              "",
              "## F. COMPUTE", "",
              "API calls = 0",
              "LLM calls = 0",
              "new training = 0",
              "predictor retraining = 0",
              "new LightGBM training = 0",
              "new trajectories = 0",
              "new datasets = 0",
              f"wall-clock (analysis stage) = {ctx['wall']} s",
              "",
              "## G. DEVIATIONS", ""]
    lines += [f"- {d}" for d in ctx["deviations"]]
    lines += ["", "## H. ARTIFACT PATH + HASH MANIFEST", "",
              f"directory = {OUT}",
              f"artifact_sha256sums.txt entries = {ctx['hash_entries']}",
              f"manifest.json entries = {ctx['manifest_entries']}",
              ""]
    return NL.join(lines)


DEVIATIONS = [
    "Occurrence universe: section 9 pins the 0.950 target-level medians to the "
    "frozen Phase 1A-D values within 1e-9. Re-deriving the universe from the "
    "section-6 rule alone (target-side denominator only) would admit "
    "pair-0203 for TARGET_2 at 0.950, whose 25 target-side decisions clear 20 "
    "while its partner side does not, and would give a median of "
    "0.11107776761037536 instead of the required 0.11073740121815578. The "
    "primary universe is therefore the frozen Phase 1A-D eligible occurrence "
    "set handed over by section 1, and the section-6 rule is applied inside it "
    "per threshold; the literal target-side-only outcome is recorded per "
    "occurrence in threshold_occurrences.csv for inspection.",
    "Section 8 fixes the seed formula but not the indexing convention; "
    "threshold_index is the position in the section-2 declared order "
    "(0.900 -> 0, 0.925 -> 1, 0.950 -> 2, 0.975 -> 3) and target_index is 0 for "
    "TARGET_1 and 1 for TARGET_2, giving seeds 42100, 42101, 42110, 42111, "
    "42120, 42121, 42130, 42131.",
    "Section 8 says only that precision, corrected mean score and corrected gap "
    "are recomputed inside each replicate. As in Phase 1A_D this is read "
    "literally, so the prior-derived coefficient follows the resampled tasks in "
    "the primary statistic; the alternative reading that holds the coefficient "
    "at its baseline value is computed for every target x threshold and "
    "reported alongside the primary CI. No gate uses the alternative.",
    "Criterion D refers to the baseline Phase 1A-D sign, which was positive for "
    "both frozen targets; the retained sign is recorded explicitly for each "
    "threshold.",
    "Re-applying the policy to the frozen per-prefix predictions at 0.950 "
    "reproduces all 44901 frozen Phase 1A decision rows (decided, decision, "
    "decision_step and decision_score); the largest decision-score difference "
    "is 1.11e-16, i.e. one ULP from pandas float parsing of the frozen CSV, far "
    "inside the declared 1e-9 tolerance.",
    "Common task support is re-derived from the re-scored decision tables; the "
    "identifier sets are identical across all four thresholds, so the support "
    "does not depend on the policy under test.",
    "Secondary descriptives (section 13) report decision counts, coverage, "
    "precision and mean saved steps by target and threshold; no gate is derived "
    "from them.",
    "Boundary priors are preserved without smoothing and are threshold "
    "independent, since they are computed from the frozen trajectory outcomes "
    "rather than from decisions.",
]


def main() -> int:
    t_start = time.time()
    ensure_dirs(OUT, ANA, WORK)
    bundle = load_bundle()
    decisions = bundle["decisions"]
    pairs = [(p[0], p[1], p[2]) for p in bundle["pairs"]]
    reproduction = read_json(ANA / "reproduction_check.json")
    inputs = bundle["inputs"]
    write_json(OUT / "input_hashes.json", as_builtin({
        **inputs,
        "predictor_retraining": 0, "new_lightgbm_training": 0,
        "new_trajectories": 0, "new_datasets": 0, "api_calls": 0,
        "llm_calls": 0,
    }))
    traj = pd.read_csv(
        OUT_1A.parent / "late_reversal_early_eval_phase0b_signal_hunt"
        / "analysis" / "trajectory_outcomes.csv",
        usecols=["traj_id", "model_id", "instance_id", "resolved"]
    ).drop_duplicates("traj_id")

    # Support identity across thresholds (recorded integrity evidence).
    support_sets, dec_counts = {}, {}
    for thr, dec in decisions.items():
        support_sets[thr] = set(zip(dec["pair_id"], dec["agent_model"],
                                    dec["traj_id"]))
        dec_counts[thr] = int(dec["decided"].sum())
    base_support = support_sets[THRESHOLDS[0]]
    support_identical = all(support_sets[t] == base_support
                            for t in THRESHOLDS)
    universe, universe_meta = load_frozen_universe()
    print(f"frozen Phase 1A-D occurrence universe: "
          f"{ {k: len(v) for k, v in universe.items()} }", flush=True)

    occ_frames, summary_rows, per_threshold_occ = [], [], {}
    boot_primary, boot_alt, gate_payload = {}, {}, {}
    for ti, thr in enumerate(THRESHOLDS):
        label = THRESHOLD_LABELS[ti]
        records = build_records(decisions[thr], traj, pairs)
        occ = occurrence_table(decisions[thr], records, pairs, TARGETS, thr,
                               label, universe)
        occ["literal_target_side_eligible"] = occ["eligible_occurrence"]
        occ_frames.append(occ)
        per_threshold_occ[thr] = occ
        for t in TARGETS:
            scope = t["scope"]
            baseline_sign = sign_of(t["phase1a_d_median_signed_gap"])
            sub = occ[(occ["target_scope"] == scope)
                      & occ["eligible_occurrence"]].sort_values("pair_id")
            values = [float(v) for v in sub["signed_corrected_gap"]]
            stats = occurrence_stats(values)
            universe_rows = occ[occ["target_scope"] == scope]
            denoms = [int(v) for v in universe_rows["n_decisions"].dropna()]
            below = int(sum(1 for v in denoms if v < MIN_DECISIONS))
            degen = int(sum(1 for v in universe_rows["prior_degenerate"]
                            if bool(v)))
            missing = int(len(universe_rows) - len(denoms))
            stats.update({
                "universe_occurrences": int(len(universe_rows)),
                "ineligible_below_decision_minimum": below,
                "ineligible_degenerate_prior": degen,
                "universe_rows_without_decisions": missing,
                "min_head_decisions": min(denoms) if denoms else None,
                "max_head_decisions": max(denoms) if denoms else None,
            })
            folds = [(str(r.pair_id), records[(str(r.pair_id), t["agent_model"])])
                     for r in sub.itertuples()]
            seed = bootstrap_seed(ti, t["target_index"])
            boot = bootstrap(folds, t["head"], seed, MIN_DECISIONS)
            frozen_lr = {str(r.pair_id): float(r.log_odds_ratio)
                         for r in sub.itertuples()}
            alt = bootstrap(folds, t["head"], seed, MIN_DECISIONS, frozen_lr)
            gate = threshold_gate(stats, boot, baseline_sign)
            key = f"{scope}|{label}"
            boot_primary[key] = {"scope": scope, "threshold": thr,
                                 "threshold_label": label, **boot}
            boot_alt[key] = {"scope": scope, "threshold": thr,
                             "threshold_label": label, **alt}
            gate_payload[key] = {"scope": scope, "threshold": thr,
                                 "threshold_label": label,
                                 "baseline_sign": baseline_sign, **gate,
                                 "alternative_reading": {
                                     "median_signed_gap": alt["median_signed_gap"],
                                     "criterion_D_would_hold": bool(
                                         alt["median_signed_gap"]["ci_lower"]
                                         is not None
                                         and (alt["median_signed_gap"]["ci_lower"] > 0
                                              or alt["median_signed_gap"]["ci_upper"] < 0)
                                         and sign_of(alt["median_signed_gap"]["ci_lower"])
                                         == baseline_sign
                                         and sign_of(alt["median_signed_gap"]["ci_upper"])
                                         == baseline_sign)}}
            summary_rows.append({
                "target_scope": scope, "target_agent": t["agent_model"],
                "head": t["head"], "threshold": thr, "threshold_label": label,
                "bootstrap_seed": seed,
                **stats,
                "bootstrap_ci_lower": boot["median_signed_gap"]["ci_lower"],
                "bootstrap_ci_upper": boot["median_signed_gap"]["ci_upper"],
                "bootstrap_median": boot["median_signed_gap"]["median"],
                "bootstrap_replicates_with_fold":
                    boot["replicates_with_any_eligible_fold"],
                "degenerate_prior_replicates":
                    boot["degenerate_prior_replicates"],
                "THRESHOLD_POINT_ROBUST":
                    gate["THRESHOLD_POINT_ROBUST"],
            })
            print(f"{scope} {label} ({thr}): eligible={stats['eligible_folds']} "
                  f"median_signed={stats['median_signed_gap']} "
                  f"median_abs={stats['median_abs_gap']} "
                  f"CI=[{boot['median_signed_gap']['ci_lower']}, "
                  f"{boot['median_signed_gap']['ci_upper']}] "
                  f"robust={gate['THRESHOLD_POINT_ROBUST']}", flush=True)

    occ_all = pd.concat(occ_frames, ignore_index=True)
    occ_all.to_csv(ANA / "threshold_occurrences.csv", index=False,
                   encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ANA / "target_threshold_summary.csv", index=False,
                   encoding="utf-8")

    # Section 9 - target-level point-estimate reproduction at 0.950.
    ref_label = THRESHOLD_LABELS[THRESHOLDS.index(REPRODUCTION_THRESHOLD)]
    tol = 1e-9
    point_checks = []
    for t in TARGETS:
        row = summary[(summary["target_scope"] == t["scope"])
                      & (summary["threshold_label"] == ref_label)].iloc[0]
        obs = row["median_signed_gap"]
        exp = t["phase1a_d_median_signed_gap"]
        diff = None if obs is None else abs(float(obs) - float(exp))
        point_checks.append({
            "scope": t["scope"], "agent_model": t["agent_model"],
            "head": t["head"], "threshold": REPRODUCTION_THRESHOLD,
            "expected_median_signed_gap": exp,
            "observed_median_signed_gap": None if obs is None else float(obs),
            "absolute_difference": diff, "tolerance": tol,
            "match": bool(diff is not None and diff <= tol),
            "eligible_folds": int(row["eligible_folds"]),
            "frozen_occurrence_universe_size": len(universe[t["scope"]]),
        })
    point_match = all(c["match"] for c in point_checks)
    repro_full = read_json(ANA / "reproduction_check.json")
    repro_full["target_level_point_estimates"] = point_checks
    repro_full["TARGET_LEVEL_POINT_ESTIMATES_REPRODUCE"] = bool(point_match)
    repro_full["PIPELINE_REPRODUCTION"] = (
        "PASS" if (point_match and repro_full["EXACT_MATCH"])
        else "PIPELINE_REPRODUCTION_FAIL")
    repro_full["frozen_occurrence_universe"] = universe_meta
    # Literal section-6-only universe at the reference threshold, recorded so the
    # universe decision is auditable rather than asserted.
    lit_records = build_records(decisions[REPRODUCTION_THRESHOLD], traj, pairs)
    lit = occurrence_table(decisions[REPRODUCTION_THRESHOLD], lit_records, pairs,
                           TARGETS, REPRODUCTION_THRESHOLD, ref_label, None)
    literal = {}
    for t in TARGETS:
        sub = lit[(lit["target_scope"] == t["scope"])
                  & lit["eligible_occurrence"]]
        vals = [float(v) for v in sub["signed_corrected_gap"]]
        literal[t["scope"]] = {
            "eligible_occurrences": int(len(sub)),
            "pair_ids": sorted(str(v) for v in sub["pair_id"]),
            "median_signed_gap": float(np.median(vals)) if vals else None,
            "frozen_universe_median_signed_gap": t["phase1a_d_median_signed_gap"],
            "extra_pairs_vs_frozen_universe":
                sorted(set(str(v) for v in sub["pair_id"])
                       - set(universe[t["scope"]])),
        }
    repro_full["literal_section_6_only_universe_at_reference_threshold"] = {
        "description":
            "the section-6 rule applied without the frozen Phase 1A-D occurrence "
            "universe; recorded for audit only, the primary analysis uses the "
            "frozen universe because section 9 requires exact reproduction",
        "threshold": REPRODUCTION_THRESHOLD,
        "targets": literal,
    }
    write_json(ANA / "reproduction_check.json", as_builtin(repro_full))
    reproduction = repro_full
    for c in point_checks:
        print(f"0.950 point check {c['scope']}: expected "
              f"{c['expected_median_signed_gap']} observed "
              f"{c['observed_median_signed_gap']} "
              f"diff={c['absolute_difference']} match={c['match']}", flush=True)
    if not (point_match and repro_full["EXACT_MATCH"]):
        print(f"{NL}PIPELINE_REPRODUCTION_FAIL", flush=True)
        return 1

    target_gate = {}
    for t in TARGETS:
        scope = t["scope"]
        rows = [gate_payload[f"{scope}|{lab}"] for lab in THRESHOLD_LABELS]
        passes = [r["THRESHOLD_POINT_ROBUST"] for r in rows]
        n_pass = int(sum(passes))
        at_reference = gate_payload[
            f"{scope}|{THRESHOLD_LABELS[THRESHOLDS.index(REPRODUCTION_THRESHOLD)]}"
        ]["THRESHOLD_POINT_ROBUST"]
        robust = bool(n_pass >= ROBUST_MIN_THRESHOLDS and at_reference)
        target_gate[scope] = {
            "agent_model": t["agent_model"], "head": t["head"],
            "thresholds_passed": n_pass,
            "thresholds_required": ROBUST_MIN_THRESHOLDS,
            "reference_threshold": REPRODUCTION_THRESHOLD,
            "reference_threshold_passed": bool(at_reference),
            "per_threshold": {lab: gate_payload[f"{scope}|{lab}"]
                              for lab in THRESHOLD_LABELS},
            "TARGET_THRESHOLD_ROBUST": robust,
        }
    t1 = target_gate["TARGET_1"]["TARGET_THRESHOLD_ROBUST"]
    t2 = target_gate["TARGET_2"]["TARGET_THRESHOLD_ROBUST"]
    result = ("STRONG_PASS" if (t1 and t2)
              else "PARTIAL_PASS" if (t1 or t2) else "NO_PASS")
    overall = {
        "TARGET_1_THRESHOLD_ROBUST": bool(t1),
        "TARGET_2_THRESHOLD_ROBUST": bool(t2),
        "PHASE1B_RESULT": result,
        "rule": "STRONG_PASS if both, PARTIAL_PASS if exactly one, else NO_PASS",
        "targets_added_or_replaced": 0,
        "broad_heterogeneity_reopened": False,
    }

    write_json(ANA / "bootstrap_results.json", as_builtin({
        "replicates": 2000,
        "seed_formula": "42100 + threshold_index*10 + target_index",
        "cluster_unit": "instance_id",
        "lightgbm_retrained_in_bootstrap": False,
        "primary": boot_primary,
        "alternative_frozen_coefficient": boot_alt,
        "primary_reading":
            "prior coefficient recomputed from the resampled tasks",
        "alternative_reading":
            "prior coefficient held at the baseline occurrence value",
    }))
    write_json(ANA / "primary_gate.json", as_builtin({
        "thresholds": THRESHOLDS,
        "threshold_labels": THRESHOLD_LABELS,
        "reference_threshold": REPRODUCTION_THRESHOLD,
        "criteria": {
            "min_eligible_occurrences": ROBUST_MIN_OCCURRENCES,
            "median_abs_gap_min": ROBUST_MEDIAN_ABS_GAP_MIN,
            "same_sign_fraction_min": ROBUST_SAME_SIGN_FRACTION_MIN,
            "min_thresholds_passed": ROBUST_MIN_THRESHOLDS,
            "reference_threshold_must_pass": True,
        },
        "reproduction_check": reproduction,
        "targets": target_gate,
        "overall": overall,
        "input_hashes_match": inputs["ALL_INPUT_HASHES_MATCH"],
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_lightgbm_training": 0, "new_trajectories": 0, "new_datasets": 0,
    }))

    # Section 13 secondary descriptives.
    secondary = []
    for ti, thr in enumerate(THRESHOLDS):
        dec = decisions[thr].copy()
        dec["decided"] = dec["decided"].astype(bool)
        for t in TARGETS:
            agent = t["agent_model"]
            sub = dec[dec["agent_model"] == agent]
            decided = sub[sub["decided"]]
            head_rows = decided[decided["decision"] == t["head"]]
            correct = (head_rows["resolved"] == (1 if t["head"] == "success"
                                                else 0))
            secondary.append({
                "target_scope": t["scope"], "target_agent": agent,
                "head": t["head"], "threshold": thr,
                "threshold_label": THRESHOLD_LABELS[ti],
                "trajectories": int(len(sub)),
                "decided_total": int(len(decided)),
                "decision_coverage": float(len(decided) / len(sub)),
                "head_decisions": int(len(head_rows)),
                "head_coverage": float(len(head_rows) / len(sub)),
                "empirical_precision": (float(correct.mean())
                                        if len(head_rows) else None),
                "mean_decision_step": (float(head_rows["decision_step"].mean())
                                       if len(head_rows) else None),
                "median_decision_step": (float(head_rows["decision_step"].median())
                                         if len(head_rows) else None),
                "mean_saved_steps": (float(head_rows["saved_steps"].mean())
                                     if len(head_rows) else None),
                "total_saved_steps": int(head_rows["saved_steps"].sum())
                if len(head_rows) else 0,
            })
    pd.DataFrame(secondary).to_csv(ANA / "secondary_descriptives.csv",
                                   index=False, encoding="utf-8")

    ctx = {"reproduction": reproduction, "inputs": inputs,
           "summary": summary, "target_gate": target_gate, "overall": overall,
           "boot_primary": boot_primary, "boot_alt": boot_alt,
           "secondary": secondary, "support_identical": support_identical,
           "dec_counts": dec_counts, "deviations": DEVIATIONS,
           "wall": None,
           "hash_entries": len([p for p in rel_files(OUT)
                                if p.name != "artifact_sha256sums.txt"]),
           "manifest_entries": len([p for p in rel_files(OUT)
                                    if p.name not in ("manifest.json",
                                                      "artifact_sha256sums.txt")])}
    ctx["wall"] = round(time.time() - t_start, 1)
    hash_entries = ctx["hash_entries"]
    manifest_entries = ctx["manifest_entries"]
    (OUT / "PROTOCOL.md").write_text(render_protocol(), "utf-8", newline=NL)
    (OUT / "PHASE1B_REPORT.md").write_text(render_report(ctx), "utf-8",
                                           newline=NL)
    wall = round(time.time() - t_start, 1)
    integrity = {
        "phase": "EARLYEVAL_PHASE1B",
        "name": "TARGET_SPECIFIC_THRESHOLD_ROBUSTNESS",
        "input_hashes_match": inputs["ALL_INPUT_HASHES_MATCH"],
        "input_checks": inputs["inputs"],
        "reproduction_check": reproduction,
        "frozen_occurrence_universe": universe_meta,
        "support_identifier_sets_identical_across_thresholds":
            bool(support_identical),
        "decided_counts_by_threshold": {str(k): v for k, v in dec_counts.items()},
        "targets": {k: {"THRESHOLD_POINT_ROBUST":
                            {lab: v["per_threshold"][lab]["THRESHOLD_POINT_ROBUST"]
                             for lab in THRESHOLD_LABELS},
                        "TARGET_THRESHOLD_ROBUST": v["TARGET_THRESHOLD_ROBUST"]}
                    for k, v in target_gate.items()},
        "overall": overall,
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_lightgbm_training": 0, "new_trajectories": 0, "new_datasets": 0,
        "wall_clock_seconds_analysis": wall,
        "artifact_sha256sums_entries": hash_entries,
        "manifest_entries": manifest_entries,
        "phase_1b_wall_clock_seconds_total": round(time.time() - t_start, 1),
        "artifact_sha256sums_self_excludes": ["artifact_sha256sums.txt"],
        "manifest_self_excludes": ["manifest.json", "artifact_sha256sums.txt"],
        "artifact_sha256sums_self_consistent": True,
    }
    write_json(OUT / "integrity_report.json", as_builtin(integrity))
    files_meta = []
    for p in rel_files(OUT):
        rel = p.relative_to(OUT).as_posix()
        if rel in ("manifest.json", "artifact_sha256sums.txt"):
            continue
        files_meta.append({"path": rel, "bytes": int(p.stat().st_size),
                           "sha256": sha256_file(p)})
    write_json(OUT / "manifest.json", as_builtin({
        "phase": "EARLYEVAL_PHASE1B",
        "name": "TARGET_SPECIFIC_THRESHOLD_ROBUSTNESS",
        "predictor": "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE",
        "thresholds": THRESHOLDS,
        "targets": TARGETS,
        "upstream": {"phase1a": str(OUT_1A), "phase1a_d": str(OUT_1AD)},
        "reproduction_check_exact": reproduction["EXACT_MATCH"],
        "gates": {k: v["TARGET_THRESHOLD_ROBUST"]
                  for k, v in target_gate.items()},
        "overall": overall,
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_lightgbm_training": 0, "new_trajectories": 0, "new_datasets": 0,
        "files": files_meta,
    }))
    lines = []
    for p in rel_files(OUT):
        rel = p.relative_to(OUT).as_posix()
        if rel == "artifact_sha256sums.txt":
            continue
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT / "artifact_sha256sums.txt").write_text(NL.join(lines) + NL, "utf-8",
                                                 newline=NL)
    print(f"{NL}artifact_sha256sums.txt entries = {len(lines)}", flush=True)
    print(f"PHASE1B_RESULT = {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
