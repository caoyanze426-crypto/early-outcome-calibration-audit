# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A_D steps 3-8 - occurrence statistics and the robustness gate.

Every statistic re-evaluates frozen Phase 1A decisions on a task weighting.
No predictor is trained, no prediction is regenerated, no threshold is tuned.
"""
from __future__ import annotations

import pickle

from p1d_common import (BOOTSTRAP_SEED, HALF_MIN_ABS_MEDIAN_MIN,
                        HALF_MIN_DECISIONS, HALF_MIN_ELIGIBLE_FOLDS,
                        JACKKNIFE_MIN_ABS_MEDIAN_MIN, MODELS, OUT_1A,
                        PAIR_MIN_DECISIONS, REPLICATES,
                        TARGET_MEDIAN_ABS_GAP_MIN, TARGET_MIN_OCCURRENCES,
                        TARGET_SAME_SIGN_FRACTION_MIN, WORK, as_builtin,
                        median_or_none, percentile_ci)
from p1d_prep import evaluate, evaluate_frozen, indicator

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_bundle() -> dict:
    with open(WORK / "bundle.pkl", "rb") as fh:
        return pickle.load(fh)


def half_positions(rec, assignment: dict, half: str) -> np.ndarray:
    """Positions of the tasks whose frozen SHA256 half equals `half`."""
    half_of = assignment["half_of"]
    return np.array([i for i, t in enumerate(rec["tasks"])
                     if half_of.get(t) == half], dtype=np.int64)


def weights_from_positions(n_tasks: int, pos: np.ndarray) -> np.ndarray:
    mult = np.zeros(n_tasks, dtype=np.float64)
    mult[pos] += 1.0
    return mult


def sign_of(value) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def occurrence_stats(values) -> dict:
    signed = [v for v in values if v is not None]
    n = len(signed)
    pos = sum(1 for v in signed if v > 0)
    neg = sum(1 for v in signed if v < 0)
    return {
        "occurrences": n,
        "median_signed_gap": float(np.median(signed)) if n else None,
        "median_abs_gap": float(np.median(np.abs(signed))) if n else None,
        "min_signed_gap": float(np.min(signed)) if n else None,
        "max_signed_gap": float(np.max(signed)) if n else None,
        "positive_fraction": (pos / n) if n else None,
        "negative_fraction": (neg / n) if n else None,
        "same_sign_fraction": (max(pos, neg) / n) if n else None,
    }


def jackknife(values, pair_ids) -> dict:
    """Section 4: leave-one-partner-fold-out medians."""
    signed = [v for v in values if v is not None]
    baseline_sign = sign_of(float(np.median(signed))) if signed else 0
    rows = []
    for i in range(len(signed)):
        kept = signed[:i] + signed[i + 1:]
        if not kept:
            continue
        med = float(np.median(kept))
        rows.append({"left_out_index": i,
                     "left_out_pair_id": pair_ids[i] if i < len(pair_ids)
                     else None,
                     "remaining_occurrences": len(kept),
                     "median_signed_gap": med,
                     "median_abs_gap": float(np.median(np.abs(kept))),
                     "sign_preserved": bool(sign_of(med) == baseline_sign)})
    medians = [r["median_signed_gap"] for r in rows]
    abs_medians = [r["median_abs_gap"] for r in rows]
    return {
        "baseline_sign": baseline_sign,
        "folds": rows,
        "minimum_signed_magnitude": (min(abs(v) for v in medians)
                                     if medians else None),
        "minimum_signed_median": min(medians) if medians else None,
        "maximum_signed_median": max(medians) if medians else None,
        "minimum_abs_median": min(abs_medians) if abs_medians else None,
        "maximum_abs_median": max(abs_medians) if abs_medians else None,
        "sign_preserved_all_folds": (bool(all(r["sign_preserved"]
                                              for r in rows))
                                     if rows else None),
    }


def bootstrap(folds, head: str, seed: int, min_decisions: int,
              assignment: dict, half: str | None = None) -> dict:
    """Section 5 (and the half-restricted variant): task-cluster bootstrap."""
    rng = np.random.default_rng(int(seed))
    med_signed, med_abs = [], []
    degenerate_replicates = 0
    positions_cache = {}
    if half is not None:
        positions_cache = {pid: half_positions(rec, assignment, half)
                           for pid, rec in folds}
    for _ in range(REPLICATES):
        signed = []
        degenerate = False
        for pid, rec in folds:
            n_tasks = len(rec["tasks"])
            if half is None:
                idx = rng.integers(0, n_tasks, n_tasks)
            else:
                pos = positions_cache[pid]
                if len(pos) == 0:
                    continue
                idx = pos[rng.integers(0, len(pos), len(pos))]
            mult = np.bincount(idx, minlength=n_tasks).astype(np.float64)
            st = evaluate(rec, head, mult, min_decisions)
            if st is None:
                continue
            degenerate = degenerate or st["prior_degenerate"]
            signed.append(st["signed_corrected_gap"])
        if degenerate:
            degenerate_replicates += 1
        if signed:
            med_signed.append(float(np.median(signed)))
            med_abs.append(float(np.median(np.abs(signed))))
    return {
        "replicates": REPLICATES,
        "replicates_with_any_eligible_fold": len(med_signed),
        "median_signed_gap": percentile_ci(med_signed),
        "median_abs_gap": percentile_ci(med_abs),
        "degenerate_prior_replicates": degenerate_replicates,
        "seed": int(seed),
    }


def target_block(scope: str, agent: str, head: str, occ: pd.DataFrame,
                 records, assignment: dict, seed: int) -> dict:
    """Sections 3-7 for one frozen target/head."""
    sub = occ[occ["scope"] == scope].sort_values("pair_id")
    eligible = sub[sub["eligible_occurrence"]]
    pair_ids = [str(v) for v in eligible["pair_id"]]
    base_values = [float(v) for v in eligible["signed_corrected_gap"]]
    folds = [(pid, records[(pid, agent)]) for pid in pair_ids]
    baseline = occurrence_stats(base_values)
    jk = jackknife(base_values, pair_ids)
    boot = bootstrap(folds, head, seed, PAIR_MIN_DECISIONS, assignment)
    halves, half_occurrences = {}, []
    for half in ("A", "B"):
        rows, values = [], []
        for pid, rec in folds:
            mult = weights_from_positions(
                len(rec["tasks"]), half_positions(rec, assignment, half))
            st = evaluate(rec, head, mult, 0)
            st_min = evaluate(rec, head, mult, HALF_MIN_DECISIONS)
            ok = bool(st_min is not None and not st_min["prior_degenerate"])
            rows.append({
                "pair_id": pid,
                "half_tasks": int(mult.sum()),
                "n_decisions": None if st is None else st["n_decisions"],
                "empirical_precision":
                    None if st is None else st["empirical_precision"],
                "corrected_mean_score":
                    None if st is None else st["corrected_mean_score"],
                "signed_corrected_gap":
                    None if st is None else st["signed_corrected_gap"],
                "absolute_corrected_gap":
                    None if st is None else st["absolute_corrected_gap"],
                "pi_target": None if st is None else st["pi_target"],
                "pi_train": None if st is None else st["pi_train"],
                "prior_degenerate":
                    None if st is None else st["prior_degenerate"],
                "eligible_half_occurrence": ok,
            })
            if ok:
                values.append(float(st_min["signed_corrected_gap"]))
        halves[half] = {
            "half": half,
            "min_decisions_required": HALF_MIN_DECISIONS,
            "occurrences_evaluated": len(rows),
            "eligible_folds": len(values),
            "median_signed_gap": float(np.median(values)) if values else None,
            "median_abs_gap": (float(np.median(np.abs(values)))
                               if values else None),
            "occ_folds": rows,
        }
        for r in rows:
            half_occurrences.append({"scope": scope, "agent_model": agent,
                                     "head": head, "half": half, **r})
    return {"scope": scope, "agent_model": agent, "head": head,
            "baseline": baseline, "jackknife": jk, "bootstrap": boot,
            "halves": halves, "half_occurrences": half_occurrences,
            "occurrences": sub.to_dict(orient="records")}


def bootstrap_frozen_coefficient(folds, head: str, seed: int,
                                 log_r_by_pair: dict) -> dict:
    """Verification variant: the prior coefficient is held at its baseline.

    Section 5 can also be read as holding the frozen Phase 1A correction fixed
    while only the task sample varies. This is reported as an explicitly
    labelled alternative reading of the same declared statistic; it is not a
    primary statistic and no gate uses it.
    """
    rng = np.random.default_rng(int(seed))
    med_signed, med_abs = [], []
    for _ in range(REPLICATES):
        signed = []
        for pid, rec in folds:
            n_tasks = len(rec["tasks"])
            idx = rng.integers(0, n_tasks, n_tasks)
            mult = np.bincount(idx, minlength=n_tasks).astype(np.float64)
            st = evaluate_frozen(rec, head, mult, PAIR_MIN_DECISIONS,
                                 log_r_by_pair[pid])
            if st is None:
                continue
            signed.append(st["signed_corrected_gap"])
        if signed:
            med_signed.append(float(np.median(signed)))
            med_abs.append(float(np.median(np.abs(signed))))
    return {"replicates": REPLICATES, "seed": int(seed),
            "replicates_with_any_eligible_fold": len(med_signed),
            "median_signed_gap": percentile_ci(med_signed),
            "median_abs_gap": percentile_ci(med_abs),
            "coefficient": "frozen at the baseline occurrence log-odds ratio",
            "primary_statistic": False}


def robustness_gate(block: dict) -> dict:
    """Section 9 criteria A-H."""
    baseline = block["baseline"]
    jk = block["jackknife"]
    boot = block["bootstrap"]
    base_sign = sign_of(baseline["median_signed_gap"])
    ci = boot["median_signed_gap"]
    ci_excludes_zero = bool(ci["ci_lower"] is not None
                            and (ci["ci_lower"] > 0 or ci["ci_upper"] < 0))
    ci_retains_sign = bool(
        ci_excludes_zero and sign_of(ci["ci_lower"]) == base_sign
        and sign_of(ci["ci_upper"]) == base_sign)
    halves_ok, half_detail = True, {}
    for half, h in block["halves"].items():
        mag = h["median_signed_gap"]
        ok = bool(
            h["eligible_folds"] >= HALF_MIN_ELIGIBLE_FOLDS
            and mag is not None
            and sign_of(mag) == base_sign
            and abs(mag) >= HALF_MIN_ABS_MEDIAN_MIN)
        half_detail[half] = {
            "eligible_folds": h["eligible_folds"],
            "eligible_folds_required": HALF_MIN_ELIGIBLE_FOLDS,
            "median_signed_gap": mag,
            "sign_retained": bool(mag is not None
                                  and sign_of(mag) == base_sign),
            "abs_magnitude_required": HALF_MIN_ABS_MEDIAN_MIN,
            "passed": ok,
        }
        halves_ok = halves_ok and ok
    checks = {
        "A_eligible_occurrences": {
            "value": baseline["occurrences"],
            "required": TARGET_MIN_OCCURRENCES,
            "passed": bool(baseline["occurrences"] >= TARGET_MIN_OCCURRENCES)},
        "B_baseline_median_abs_gap": {
            "value": baseline["median_abs_gap"],
            "required": TARGET_MEDIAN_ABS_GAP_MIN,
            "passed": bool(baseline["median_abs_gap"] is not None
                           and baseline["median_abs_gap"]
                           >= TARGET_MEDIAN_ABS_GAP_MIN)},
        "C_baseline_same_sign_fraction": {
            "value": baseline["same_sign_fraction"],
            "required": TARGET_SAME_SIGN_FRACTION_MIN,
            "passed": bool(baseline["same_sign_fraction"] is not None
                           and baseline["same_sign_fraction"]
                           >= TARGET_SAME_SIGN_FRACTION_MIN)},
        "D_bootstrap_ci_excludes_zero_and_retains_sign": {
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "baseline_sign": base_sign,
            "excludes_zero": ci_excludes_zero,
            "retains_sign": ci_retains_sign,
            "passed": ci_retains_sign},
        "E_jackknife_sign_preserved_all_folds": {
            "value": jk["sign_preserved_all_folds"],
            "passed": bool(jk["sign_preserved_all_folds"] is True)},
        "F_jackknife_min_abs_median": {
            "value": jk["minimum_abs_median"],
            "required": JACKKNIFE_MIN_ABS_MEDIAN_MIN,
            "passed": bool(jk["minimum_abs_median"] is not None
                           and jk["minimum_abs_median"]
                           >= JACKKNIFE_MIN_ABS_MEDIAN_MIN)},
        "G_TASK_HALF_A": half_detail["A"],
        "H_TASK_HALF_B": half_detail["B"],
    }
    robust = all(v["passed"] for v in checks.values())
    return {"scope": block["scope"], "agent_model": block["agent_model"],
            "head": block["head"], "baseline_sign": base_sign,
            "criteria": checks, "TARGET_ROBUST": bool(robust)}
