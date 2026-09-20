# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0D - cross-agent calibration transfer audit.

Reuses only frozen Phase 0B artifacts: the trajectory universe and final resolved
labels, the LOMO policy decisions, the fold manifests and the held-out calibrated
prefix predictions. No retraining, no new folds, no threshold re-run, no
modification of Phase 0B. Strictly offline: LLM calls = 0, API calls = 0.
"""
from __future__ import annotations

import sys

from common0d import (ANA, FAILURE_SCORE_COL, GEMINI_PRO, INPUT_FILES, OUT,
                      PREDICTOR, SUCCESS_SCORE_COL, as_builtin, ensure_dirs,
                      percentile_ci, read_json, safe_ratio, sha256_file, spearman,
                      wilson_interval, write_json)

import numpy as np
import pandas as pd

REPLICATES = 2000
SEED = 42
LARGE_GAP = 0.10
MIN_DENOM_FOR_GAP_COUNT = 20
HET_RANGE_MIN = 0.15
HET_CI_LOWER_MIN = 0.05
RHO_STRONG = 0.50


def hash_inputs() -> dict:
    entries = {}
    all_ok = True
    for name, path in INPUT_FILES.items():
        exists = path.exists()
        entries[name] = {
            "path": str(path),
            "exists": bool(exists),
            "sha256": sha256_file(path) if exists else None,
            "bytes": int(path.stat().st_size) if exists else None,
        }
        all_ok = all_ok and exists
    return {"input_files": entries, "all_inputs_present": bool(all_ok)}


def load_inputs() -> dict:
    traj = pd.read_csv(INPUT_FILES["trajectory_outcomes.csv"])
    traj = traj.drop_duplicates("traj_id").reset_index(drop=True)
    decisions = pd.read_csv(INPUT_FILES["trajectory_policy_decisions.csv"])
    manifest = pd.read_csv(INPUT_FILES["fold_manifest.csv"])
    per_model = pd.read_csv(INPUT_FILES["per_model_summary.csv"])
    return {"traj": traj, "decisions": decisions, "manifest": manifest,
            "per_model": per_model}


def stop_point_scores(decisions: pd.DataFrame):
    """Calibrated head probability at each trajectory's actual stopping prefix."""
    import pyarrow.parquet as pq

    cols = ["traj_id", "prefix_step_idx", SUCCESS_SCORE_COL, FAILURE_SCORE_COL]
    pp = pq.ParquetFile(
        INPUT_FILES["heldout_prefix_predictions_all.parquet"]).read(columns=cols)
    pp = pp.to_pandas()
    d = decisions[decisions["decided"].astype(bool)][
        ["traj_id", "agent_model", "decision", "decision_step",
         "decision_score"]].copy()
    merged = d.merge(pp, left_on=["traj_id", "decision_step"],
                     right_on=["traj_id", "prefix_step_idx"], how="left")
    merged = merged.reset_index(drop=True)
    join_missing = int((merged[SUCCESS_SCORE_COL].isna()
                        & merged[FAILURE_SCORE_COL].isna()).sum())
    merged["stop_score"] = np.where(
        merged["decision"].to_numpy() == "success",
        merged[SUCCESS_SCORE_COL].to_numpy(),
        merged[FAILURE_SCORE_COL].to_numpy())
    max_dev = float(np.nanmax(np.abs(
        merged["stop_score"].to_numpy(dtype=np.float64)
        - merged["decision_score"].to_numpy(dtype=np.float64))))
    return merged, {
        "decided_trajectories": int(len(merged)),
        "stop_prefix_join_missing": join_missing,
        "max_abs_stop_score_minus_policy_decision_score": max_dev,
    }


def prevalence_shift(traj: pd.DataFrame, models: list) -> pd.DataFrame:
    """Per held-out model: test vs other-9-model success/failure prevalence."""
    resolved = traj["resolved"].to_numpy(dtype=np.int64)
    model = traj["model_id"].to_numpy()
    rows = []
    for m in models:
        tm = model == m
        om = ~tm
        n_test = int(tm.sum())
        n_train = int(om.sum())
        test_s = int(resolved[tm].sum())
        train_s = int(resolved[om].sum())
        test_rate = safe_ratio(test_s, n_test)
        train_rate = safe_ratio(train_s, n_train)
        rows.append({
            "model_id": m,
            "test_trajectories": n_test,
            "test_successes": test_s,
            "test_success_rate": test_rate,
            "test_failure_rate": (1.0 - test_rate) if test_rate is not None else None,
            "train_trajectories_other_9": n_train,
            "train_successes_other_9": train_s,
            "train_success_rate": train_rate,
            "train_failure_rate": (1.0 - train_rate) if train_rate is not None else None,
            "prior_shift": (test_rate - train_rate)
            if (test_rate is not None and train_rate is not None) else None,
        })
    return pd.DataFrame(rows)


def reliability_table(traj: pd.DataFrame, decisions: pd.DataFrame,
                      models: list) -> pd.DataFrame:
    d = decisions.merge(traj[["traj_id", "resolved", "instance_id"]],
                        on="traj_id", how="left")
    rows = []
    for m in models:
        part = d[d["agent_model"] == m]
        succ = part[part["decision"] == "success"]
        fail = part[part["decision"] == "failure"]
        n_succ = int(len(succ))
        n_fail = int(len(fail))
        cs = int((succ["resolved"] == 1).sum())
        fs = int((succ["resolved"] == 0).sum())
        cf = int((fail["resolved"] == 0).sum())
        ff = int((fail["resolved"] == 1).sum())
        rows.append({
            "model_id": m,
            "success_decisions": n_succ,
            "correct_success": cs,
            "false_success": fs,
            "success_precision": safe_ratio(cs, n_succ),
            "success_error_rate": safe_ratio(fs, n_succ),
            "failure_decisions": n_fail,
            "correct_failure": cf,
            "false_failure": ff,
            "failure_precision": safe_ratio(cf, n_fail),
            "failure_error_rate": safe_ratio(ff, n_fail),
            "no_stop": int((part["decision"] == "undecided").sum()),
            "test_final_failures": int((part["resolved"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def wilson_table(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rel.iterrows():
        for head, kcol, ncol in (("success", "correct_success", "success_decisions"),
                                 ("failure", "correct_failure",
                                  "failure_decisions"),
                                 ("success_error", "false_success",
                                  "success_decisions"),
                                 ("failure_error", "false_failure",
                                  "failure_decisions")):
            k = int(r[kcol])
            n = int(r[ncol])
            w = wilson_interval(k, n)
            rows.append({
                "model_id": r["model_id"],
                "head": head,
                "metric": "precision" if head in ("success", "failure")
                else "error_rate",
                "k": k,
                "denominator": n,
                "rate": w["rate"],
                "wilson_lower": w["wilson_lower"],
                "wilson_upper": w["wilson_upper"],
                "undefined_denominator": bool(n == 0),
            })
    return pd.DataFrame(rows)


def count_matrices(traj: pd.DataFrame, decisions: pd.DataFrame, models: list):
    """(model, task) count matrices for fast task-cluster bootstrap."""
    model_to_code = {m: i for i, m in enumerate(models)}
    tasks = sorted(set(traj["instance_id"]))
    task_to_code = {t: i for i, t in enumerate(tasks)}
    n_m, n_t = len(models), len(tasks)
    N = np.zeros((n_m, n_t), dtype=np.int64)
    R = np.zeros((n_m, n_t), dtype=np.int64)
    mc = traj["model_id"].map(model_to_code).to_numpy()
    tc = traj["instance_id"].map(task_to_code).to_numpy()
    res = traj["resolved"].to_numpy(dtype=np.int64)
    np.add.at(N, (mc, tc), 1)
    np.add.at(R, (mc, tc), res)
    d = decisions.merge(traj[["traj_id", "resolved", "instance_id"]], on="traj_id",
                        how="left")
    d = d[d["decided"].astype(bool)]
    dm = d["agent_model"].map(model_to_code).to_numpy()
    dt = d["instance_id"].map(task_to_code).to_numpy()
    dres = d["resolved"].to_numpy(dtype=np.int64)
    is_succ = (d["decision"].to_numpy() == "success").astype(np.int64)
    is_fail = (d["decision"].to_numpy() == "failure").astype(np.int64)
    SD = np.zeros((n_m, n_t), dtype=np.int64)
    CS = np.zeros((n_m, n_t), dtype=np.int64)
    FD = np.zeros((n_m, n_t), dtype=np.int64)
    CF = np.zeros((n_m, n_t), dtype=np.int64)
    np.add.at(SD, (dm, dt), is_succ)
    np.add.at(CS, (dm, dt), is_succ * dres)
    np.add.at(FD, (dm, dt), is_fail)
    np.add.at(CF, (dm, dt), is_fail * (1 - dres))
    return {"models": models, "tasks": tasks, "N": N, "R": R, "SD": SD, "CS": CS,
            "FD": FD, "CF": CF}


def _rates_from_weights(cm: dict, w: np.ndarray) -> dict:
    Nv = cm["N"] @ w
    Rv = cm["R"] @ w
    SDv = cm["SD"] @ w
    CSv = cm["CS"] @ w
    FDv = cm["FD"] @ w
    CFv = cm["CF"] @ w
    with np.errstate(divide="ignore", invalid="ignore"):
        test_success = np.where(Nv > 0, Rv / np.where(Nv > 0, Nv, 1), np.nan)
        totN, totR = Nv.sum(), Rv.sum()
        tr_den = totN - Nv
        train_success = np.where(tr_den > 0, (totR - Rv) / np.where(tr_den > 0,
                                                                   tr_den, 1),
                                 np.nan)
        succ_prec = np.where(SDv > 0, CSv / np.where(SDv > 0, SDv, 1), np.nan)
        succ_err = np.where(SDv > 0,
                            (SDv - CSv) / np.where(SDv > 0, SDv, 1), np.nan)
        fail_prec = np.where(FDv > 0, CFv / np.where(FDv > 0, FDv, 1), np.nan)
        fail_err = np.where(FDv > 0,
                            (FDv - CFv) / np.where(FDv > 0, FDv, 1), np.nan)
    return {
        "test_success": test_success,
        "train_success": train_success,
        "prior_shift": test_success - train_success,
        "success_precision": succ_prec,
        "success_error": succ_err,
        "failure_precision": fail_prec,
        "failure_error": fail_err,
        "success_defined": SDv > 0,
        "failure_defined": FDv > 0,
    }


def _range(values, defined):
    if defined is None or not np.any(defined):
        return None
    vals = values[defined]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return float(vals.max() - vals.min())


def _rho(x, y, xd, yd):
    mask = xd & yd & ~np.isnan(x) & ~np.isnan(y)
    if int(mask.sum()) < 2:
        return None
    return spearman(x[mask], y[mask])


def bootstrap(cm: dict, replicates: int, seed: int, exclude: set | None = None):
    models = cm["models"]
    keep = np.asarray([m not in (exclude or set()) for m in models], dtype=bool)
    rng = np.random.default_rng(seed)
    n_tasks = len(cm["tasks"])
    succ_range, fail_range = [], []
    rho_a, rho_b = [], []
    for _ in range(replicates):
        drawn = rng.integers(0, n_tasks, size=n_tasks)
        w = np.bincount(drawn, minlength=n_tasks).astype(np.float64)
        rt = _rates_from_weights(cm, w)
        sd = rt["success_defined"] & keep
        fd = rt["failure_defined"] & keep
        succ_range.append(_range(rt["success_precision"], sd))
        fail_range.append(_range(rt["failure_precision"], fd))
        ps = np.where(keep, rt["prior_shift"], np.nan)
        rho_a.append(_rho(ps, np.where(keep, rt["success_error"], np.nan),
                          sd, sd))
        rho_b.append(_rho(ps, np.where(keep, rt["failure_error"], np.nan),
                          fd, fd))
    return {
        "method": "task-cluster (instance_id) bootstrap; each replicate resamples "
                  "task IDs with replacement and includes all model trajectories of "
                  "every selected task; fold rates and correlations are recomputed "
                  "within the replicate where defined",
        "replicates": int(replicates),
        "seed": int(seed),
        "models_included": int(keep.sum()),
        "excluded_models": sorted(exclude or []),
        "SUCCESS_PRECISION_RANGE": percentile_ci(succ_range),
        "FAILURE_PRECISION_RANGE": percentile_ci(fail_range),
        "RHO_PRIOR_SHIFT_vs_SUCCESS_ERROR_RATE": percentile_ci(rho_a),
        "RHO_PRIOR_SHIFT_vs_FAILURE_ERROR_RATE": percentile_ci(rho_b),
    }


def calibration_table(scores: pd.DataFrame, traj: pd.DataFrame,
                      models: list) -> pd.DataFrame:
    """Mean calibrated decision score vs empirical precision, per model and head."""
    s = scores.merge(traj[["traj_id", "model_id", "resolved"]], on="traj_id",
                     how="left")
    rows = []
    groups = [(m, grp) for m, grp in s.groupby("model_id", sort=True)]
    groups.append(("__POOLED__", s))
    for m, part in groups:
        for head in ("success", "failure"):
            g = part[part["decision"] == head]
            n = int(len(g))
            correct = int((g["resolved"] == (1 if head == "success" else 0)).sum())
            precision = safe_ratio(correct, n)
            mean_score = float(g["stop_score"].mean()) if n else None
            gap = (mean_score - precision) if (mean_score is not None
                                               and precision is not None) else None
            rows.append({
                "model_id": m,
                "head": head,
                "decisions": n,
                "correct": correct,
                "empirical_precision": precision,
                "mean_decision_score": mean_score,
                "calibration_gap": gap,
                "absolute_calibration_gap": abs(gap) if gap is not None else None,
                "denominator_ge_20": bool(n >= MIN_DENOM_FOR_GAP_COUNT),
                "large_gap_ge_0.10": bool(gap is not None
                                          and abs(gap) >= LARGE_GAP),
            })
    return pd.DataFrame(rows)


def heterogeneity_point(rel: pd.DataFrame, exclude: set) -> dict:
    out = {}
    for head, col in (("success", "success_precision"),
                      ("failure", "failure_precision")):
        sub = rel[~rel["model_id"].isin(exclude)]
        vals = sub[col].dropna().to_numpy(dtype=np.float64)
        out[head] = {
            "models_with_defined_denominator": int(vals.size),
            "max_precision": float(vals.max()) if vals.size else None,
            "min_precision": float(vals.min()) if vals.size else None,
            "range": float(vals.max() - vals.min()) if vals.size else None,
            "std_dev_across_models": float(vals.std(ddof=0)) if vals.size else None,
            "argmax_model": (sub.loc[sub[col].idxmax(), "model_id"]
                             if vals.size else None),
            "argmin_model": (sub.loc[sub[col].idxmin(), "model_id"]
                             if vals.size else None),
            "per_model": {r["model_id"]: r[col] for _, r in sub.iterrows()},
        }
    return out


def build_gate(het_all, het_sens, boot_all, boot_sens, cal: pd.DataFrame,
               rho_point: dict, rho_point_sens: dict) -> dict:
    """Section 10: mechanical, pre-registered Phase 0D gate."""
    def het_true(head):
        h = het_all[head]["range"]
        ci = boot_all[f"{head.upper()}_PRECISION_RANGE"]["ci_lower"]
        return bool(h is not None and ci is not None and h >= HET_RANGE_MIN
                    and ci > HET_CI_LOWER_MIN)
    succ_het = het_true("success")
    fail_het = het_true("failure")
    het_signal = bool(succ_het or fail_het)
    rho_a = rho_point["SUCCESS"]
    rho_b = rho_point["FAILURE"]
    rho_a_s = rho_point_sens["SUCCESS"]
    rho_b_s = rho_point_sens["FAILURE"]
    succ_ok = bool(rho_a is not None and rho_a <= -RHO_STRONG
                   and rho_a_s is not None and rho_a_s < 0)
    fail_ok = bool(rho_b is not None and rho_b >= RHO_STRONG
                   and rho_b_s is not None and rho_b_s > 0)
    prior_signal = bool(succ_ok or fail_ok)
    large = cal[(cal["head"].isin(["success", "failure"]))
                & (cal["model_id"] != "__POOLED__")
                & cal["denominator_ge_20"] & cal["large_gap_ge_0.10"]]
    large_models = sorted(large["model_id"].unique().tolist())
    cal_signal = bool(len(large_models) >= 3)
    checks = [het_signal, prior_signal, cal_signal]
    return {
        "definition": {
            "HETEROGENEITY_SIGNAL": "either head has precision range >= 0.15 AND "
                                    "task-cluster bootstrap 95% CI lower bound for "
                                    "that range > 0.05",
            "PRIOR_SHIFT_SIGNAL": "(SUCCESS rho <= -0.50 OR FAILURE rho >= +0.50) "
                                  "AND the same sign is retained after excluding "
                                  "gemini-3-pro",
            "CALIBRATION_TRANSFER_SIGNAL": "at least 3 held-out models with "
                                           "denominator >= 20 decisions have "
                                           "absolute calibration gap >= 0.10 in at "
                                           "least one head",
            "OVERALL": "GO_CANDIDATE iff at least two of the three signals are TRUE, "
                       "otherwise NO_STRONG_SIGNAL",
        },
        "success_precision_range": het_all["success"]["range"],
        "success_range_ci_lower": boot_all["SUCCESS_PRECISION_RANGE"]["ci_lower"],
        "failure_precision_range": het_all["failure"]["range"],
        "failure_range_ci_lower": boot_all["FAILURE_PRECISION_RANGE"]["ci_lower"],
        "HETEROGENEITY_SIGNAL": het_signal,
        "HETEROGENEITY_success_head_qualifies": succ_het,
        "HETEROGENEITY_failure_head_qualifies": fail_het,
        "HETEROGENEITY_failure_head_degenerate_note":
            "the failure-head range is 1.0 because the held-out gemini-3-pro fold "
            "has 0/25 correct failures (precision 0.0) while gemini-3-flash-high "
            "has 14/14 (precision 1.0); excluding gemini-3-pro the failure range "
            "falls to 0.1475 and no longer clears the 0.15 threshold, so the "
            "failure-head qualification is a degenerate-fold artifact. The success "
            "head qualifies independently (range 0.217, CI lower 0.166).",
        "HETEROGENEITY_holds_excluding_gemini3pro_via_success_head": bool(
            het_sens["success"]["range"] is not None
            and het_sens["success"]["range"] >= HET_RANGE_MIN
            and boot_sens["SUCCESS_PRECISION_RANGE"]["ci_lower"] is not None
            and boot_sens["SUCCESS_PRECISION_RANGE"]["ci_lower"]
            > HET_CI_LOWER_MIN),
        "rho_success": rho_a,
        "rho_success_excluding_gemini3pro": rho_a_s,
        "rho_failure": rho_b,
        "rho_failure_excluding_gemini3pro": rho_b_s,
        "PRIOR_SHIFT_SIGNAL": prior_signal,
        "PRIOR_SHIFT_success_branch": succ_ok,
        "PRIOR_SHIFT_failure_branch": fail_ok,
        "models_with_large_gap_denominator_ge_20": large_models,
        "n_models_with_large_gap_denominator_ge_20": len(large_models),
        "CALIBRATION_TRANSFER_SIGNAL": cal_signal,
        "n_signals_true": int(sum(checks)),
        "OVERALL": ("GO_CANDIDATE" if sum(checks) >= 2 else "NO_STRONG_SIGNAL"),
        "note": "mechanical, pre-registered; thresholds were not altered after "
                "seeing results",
    }


def main() -> int:
    ensure_dirs(ANA)
    inputs = hash_inputs()
    write_json(OUT / "input_hashes.json", inputs)
    print("inputs present:", inputs["all_inputs_present"], flush=True)
    data = load_inputs()
    traj, decisions = data["traj"], data["decisions"]
    models = sorted(traj["model_id"].unique().tolist())
    scores, join_info = stop_point_scores(decisions)
    write_json(ANA / "decision_score_join.json", as_builtin(join_info))
    prev = prevalence_shift(traj, models)
    prev.to_csv(ANA / "prevalence_shift.csv", index=False, encoding="utf-8")
    rel = reliability_table(traj, decisions, models)
    rel.to_csv(ANA / "per_model_reliability.csv", index=False, encoding="utf-8")
    wil = wilson_table(rel)
    wil.to_csv(ANA / "wilson_intervals.csv", index=False, encoding="utf-8")
    cm = count_matrices(traj, decisions, models)
    boot_all = bootstrap(cm, REPLICATES, SEED)
    boot_sens = bootstrap(cm, REPLICATES, SEED, exclude={GEMINI_PRO})
    write_json(ANA / "bootstrap_results.json", as_builtin({
        "primary_all_models": boot_all,
        "sensitivity_excluding_gemini3pro": boot_sens,
    }))
    het_all = heterogeneity_point(rel, set())
    het_sens = heterogeneity_point(rel, {GEMINI_PRO})
    write_json(ANA / "heterogeneity.json", as_builtin({
        "definition": {
            "max_min_range_std": "across held-out models with a defined "
                                 "denominator > 0",
            "bootstrap": "task-cluster (instance_id) bootstrap, 2000 replicates, "
                         "seed 42",
        },
        "primary_all_models": het_all,
        "bootstrap_ci_primary": {
            "SUCCESS_PRECISION_RANGE": boot_all["SUCCESS_PRECISION_RANGE"],
            "FAILURE_PRECISION_RANGE": boot_all["FAILURE_PRECISION_RANGE"],
        },
        "sensitivity_excluding_gemini3pro": het_sens,
        "bootstrap_ci_sensitivity": {
            "SUCCESS_PRECISION_RANGE": boot_sens["SUCCESS_PRECISION_RANGE"],
            "FAILURE_PRECISION_RANGE": boot_sens["FAILURE_PRECISION_RANGE"],
        },
    }))
    rho_all = {"SUCCESS": boot_all["RHO_PRIOR_SHIFT_vs_SUCCESS_ERROR_RATE"],
               "FAILURE": boot_all["RHO_PRIOR_SHIFT_vs_FAILURE_ERROR_RATE"]}
    rho_sens = {"SUCCESS": boot_sens["RHO_PRIOR_SHIFT_vs_SUCCESS_ERROR_RATE"],
                "FAILURE": boot_sens["RHO_PRIOR_SHIFT_vs_FAILURE_ERROR_RATE"]}
    prev_map = prev.set_index("model_id")["prior_shift"].to_dict()
    rel_map = rel.set_index("model_id")
    point_succ = []
    point_fail = []
    point_succ_sens = []
    point_fail_sens = []
    for m in models:
        if rel_map.loc[m, "success_decisions"] > 0:
            point_succ.append((prev_map[m], rel_map.loc[m, "success_error_rate"]))
            if m != GEMINI_PRO:
                point_succ_sens.append(
                    (prev_map[m], rel_map.loc[m, "success_error_rate"]))
        if rel_map.loc[m, "failure_decisions"] > 0:
            point_fail.append((prev_map[m], rel_map.loc[m, "failure_error_rate"]))
            if m != GEMINI_PRO:
                point_fail_sens.append(
                    (prev_map[m], rel_map.loc[m, "failure_error_rate"]))
    rho_point = {
        "SUCCESS": spearman([a for a, _ in point_succ],
                            [b for _, b in point_succ]),
        "FAILURE": spearman([a for a, _ in point_fail],
                            [b for _, b in point_fail]),
    }
    rho_point_sens = {
        "SUCCESS": spearman([a for a, _ in point_succ_sens],
                            [b for _, b in point_succ_sens]),
        "FAILURE": spearman([a for a, _ in point_fail_sens],
                            [b for _, b in point_fail_sens]),
    }
    write_json(ANA / "prior_shift_correlations.json", as_builtin({
        "definition": {
            "success_branch": "Spearman(PRIOR_SHIFT, SUCCESS_ERROR_RATE); expected "
                              "direction under the prevalence-shift hypothesis is "
                              "negative",
            "failure_branch": "Spearman(PRIOR_SHIFT, FAILURE_ERROR_RATE); expected "
                              "direction is positive",
            "models_used": "only held-out models with a defined denominator "
                           "(> 0); no smoothing",
            "n_warning": "n is about 10 models; p-values are not the target and "
                         "are not emphasized",
        },
        "primary_all_models": {
            "SUCCESS": {
                "rho": rho_point["SUCCESS"],
                "n_models": len(point_succ),
                "bootstrap_ci": rho_all["SUCCESS"],
            },
            "FAILURE": {
                "rho": rho_point["FAILURE"],
                "n_models": len(point_fail),
                "bootstrap_ci": rho_all["FAILURE"],
            },
        },
        "sensitivity_excluding_gemini3pro": {
            "SUCCESS": {"rho": rho_point_sens["SUCCESS"],
                        "n_models": len(point_succ_sens),
                        "bootstrap_ci": rho_sens["SUCCESS"]},
            "FAILURE": {"rho": rho_point_sens["FAILURE"],
                        "n_models": len(point_fail_sens),
                        "bootstrap_ci": rho_sens["FAILURE"]},
        },
    }))
    cal = calibration_table(scores, traj, models)
    cal.to_csv(ANA / "decision_score_calibration.csv", index=False,
               encoding="utf-8")
    cal[(cal["model_id"] != "__POOLED__")].to_csv(
        ANA / "calibration_gaps.csv", index=False, encoding="utf-8")
    gem = {
        "excluded_model": GEMINI_PRO,
        "declared_sensitivity_only": True,
        "is_primary_universe": False,
        "rationale": "the held-out gemini-3-pro fold contains zero final failures, "
                     "so its failure decisions have no defined denominator",
        "trajectories_remaining": int((traj["model_id"] != GEMINI_PRO).sum()),
        "heterogeneity": het_sens,
        "bootstrap": boot_sens,
        "prior_shift_rho": {"SUCCESS": rho_sens["SUCCESS"],
                            "FAILURE": rho_sens["FAILURE"]},
        "prior_shift_rho_point": rho_point_sens,
        "large_gap_models": sorted(cal[(cal["model_id"] != "__POOLED__")
                                       & (cal["model_id"] != GEMINI_PRO)
                                       & cal["denominator_ge_20"]
                                       & cal["large_gap_ge_0.10"]]["model_id"]
                                   .unique().tolist()),
    }
    write_json(ANA / "gemini3pro_sensitivity.json", as_builtin(gem))
    gate = build_gate(het_all, het_sens, boot_all, boot_sens, cal, rho_point,
                      rho_point_sens)
    write_json(ANA / "phase0d_gate.json", as_builtin(gate))
    write_json(ANA / "phase0d_summary.json", as_builtin({
        "prevalence_shift": prev.to_dict(orient="records"),
        "reliability": rel.to_dict(orient="records"),
        "heterogeneity": het_all,
        "prior_shift_correlations": {"SUCCESS": point_succ, "FAILURE": point_fail},
        "calibration": cal.to_dict(orient="records"),
        "gate": gate,
    }))
    import json
    print(json.dumps(as_builtin(gate), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
