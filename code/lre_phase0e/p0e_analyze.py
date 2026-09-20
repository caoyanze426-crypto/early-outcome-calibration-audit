# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0E - PRIOR_SHIFT_DECOMPOSITION.

Inputs: the exact frozen Phase 0B stop-point predictions (the same join Phase 0D
used and verified) and the frozen Phase 0D fold priors. Applies an oracle
label-prior (base-rate) correction to the stop-point head probabilities and asks
whether agent-conditional miscalibration survives it.

Strictly offline: LLM calls = 0, API calls = 0, predictor retraining = 0, new folds
= 0, threshold sweep = 0. Phase 0B and Phase 0D artifacts are never modified.
"""
from __future__ import annotations

import sys

from common0e import (ANA, FAILURE_SCORE_COL, GEMINI_PRO, HEAD_CODE, INPUT_FILES,
                      OUT, OUT_0B, OUT_0D, PREDICTOR, PROVENANCE, SUCCESS_SCORE_COL,
                      as_builtin, ensure_dirs, log_odds, logit, percentile_ci,
                      read_json, read_manifest, safe_ratio, sha256_file, sigmoid,
                      spearman, write_json)

import numpy as np
import pandas as pd

REPLICATES = 2000
SEED = 42
TOL = 1e-9

LARGE_GAP = 0.10
MIN_DENOM_FOR_GAP_COUNT = 20
RESID_RANGE_MIN = 0.10
RESID_CI_LOWER_MIN = 0.05


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


def verify_provenance(inputs: dict) -> dict:
    """Check every consumed input against the producing phase's own manifest."""
    b_man = read_manifest(PROVENANCE["phase0b"])
    d_man = read_manifest(PROVENANCE["phase0d"])
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
    return {"phase0b_manifest": str(PROVENANCE["phase0b"]),
            "phase0b_manifest_entries": len(b_man),
            "phase0d_manifest": str(PROVENANCE["phase0d"]),
            "phase0d_manifest_entries": len(d_man),
            "inputs": checks, "ALL_INPUT_HASHES_MATCH": bool(all_match)}


def load_decided():
    """Rebuild the exact Phase 0D stop-point frame: head probability at the stop."""
    import pyarrow.parquet as pq

    traj = pd.read_csv(INPUT_FILES["trajectory_outcomes.csv"])
    traj = traj.drop_duplicates("traj_id").reset_index(drop=True)
    decisions = pd.read_csv(INPUT_FILES["trajectory_policy_decisions.csv"])
    cols = ["traj_id", "prefix_step_idx", SUCCESS_SCORE_COL, FAILURE_SCORE_COL]
    pp = pq.ParquetFile(
        INPUT_FILES["heldout_prefix_predictions_all.parquet"]).read(
            columns=cols).to_pandas()
    d = decisions[decisions["decided"].astype(bool)][
        ["traj_id", "agent_model", "decision", "decision_step",
         "decision_score"]].copy()
    m = d.merge(pp, left_on=["traj_id", "decision_step"],
                right_on=["traj_id", "prefix_step_idx"], how="left")
    m = m.merge(traj[["traj_id", "resolved", "instance_id"]], on="traj_id",
                how="left")
    join_missing = int((m[SUCCESS_SCORE_COL].isna()
                        & m[FAILURE_SCORE_COL].isna()).sum())
    m["p_raw"] = np.where(m["decision"].to_numpy() == "success",
                          m[SUCCESS_SCORE_COL].to_numpy(),
                          m[FAILURE_SCORE_COL].to_numpy())
    m["correct"] = np.where(m["decision"].to_numpy() == "success",
                            (m["resolved"].to_numpy() == 1).astype(np.int64),
                            (m["resolved"].to_numpy() == 0).astype(np.int64))
    m["head_code"] = m["decision"].map(HEAD_CODE).astype(np.int64)
    max_dev = float(np.nanmax(np.abs(
        m["p_raw"].to_numpy(dtype=np.float64)
        - m["decision_score"].to_numpy(dtype=np.float64))))
    return m, traj, {"decided_trajectories": int(len(m)),
                     "stop_prefix_join_missing": join_missing,
                     "max_abs_stop_score_minus_policy_decision_score": max_dev}


def prior_table(traj: pd.DataFrame, models: list) -> pd.DataFrame:
    """Recompute the Phase 0D fold priors from the frozen trajectory universe."""
    resolved = traj["resolved"].to_numpy(dtype=np.int64)
    model = traj["model_id"].to_numpy()
    rows = []
    for m in models:
        tm = model == m
        n_test = int(tm.sum())
        n_train = int((~tm).sum())
        test_s = int(resolved[tm].sum())
        train_s = int(resolved[~tm].sum())
        test_rate = safe_ratio(test_s, n_test)
        train_rate = safe_ratio(train_s, n_train)
        rows.append({
            "model_id": m,
            "test_trajectories": n_test,
            "test_successes": test_s,
            "test_success_prior": test_rate,
            "train_trajectories_other_9": n_train,
            "train_successes_other_9": train_s,
            "train_success_prior": train_rate,
            "prior_shift": (test_rate - train_rate)
            if (test_rate is not None and train_rate is not None) else None,
        })
    return pd.DataFrame(rows)


def correction_table(pri: pd.DataFrame) -> pd.DataFrame:
    """Oracle base-rate (label-prior) correction coefficient per held-out model.

    The success head's positive class is a resolved success with held-out prior
    pi_test and calibration-pool prior pi_train. The failure head's positive class
    is the complement, so its odds ratio is the exact reciprocal of the success
    head's, and its log shift is exactly -ln r_m.
    """
    rows = []
    for r in pri.itertuples(index=False):
        pt = r.test_success_prior
        ptr = r.train_success_prior
        degenerate = bool(pt in (0.0, 1.0) or ptr in (0.0, 1.0))
        with np.errstate(divide="ignore"):
            lt = float(log_odds(pt))
            ltr = float(log_odds(ptr))
        log_r = lt - ltr
        rows.append({
            "model_id": r.model_id,
            "train_success_prior": ptr,
            "test_success_prior": pt,
            "prior_shift": r.prior_shift,
            "train_success_odds": float(ptr / (1.0 - ptr)) if 0 < ptr < 1 else None,
            "test_success_odds": float(pt / (1.0 - pt)) if 0 < pt < 1 else None,
            "success_odds_ratio": float(np.exp(log_r)) if np.isfinite(log_r)
            else (np.inf if log_r > 0 else 0.0),
            "log_odds_ratio_success_head": log_r,
            "log_odds_ratio_failure_head": -log_r,
            "prior_degenerate": degenerate,
            "degenerate_reason": (
                "held-out success prior is 0 or 1, so the oracle odds ratio is 0 or "
                "+inf and the corrected probabilities sit on the boundary"
            ) if degenerate else "",
        })
    return pd.DataFrame(rows)


def apply_correction(m: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """Correct every stop-point head probability with its model's oracle prior."""
    lr = dict(zip(corr["model_id"], corr["log_odds_ratio_success_head"]))
    sign = np.where(m["head_code"].to_numpy() == 0, 1.0, -1.0)
    lr_row = m["agent_model"].map(lr).to_numpy(dtype=np.float64)
    z = logit(m["p_raw"].to_numpy(dtype=np.float64)) + sign * lr_row
    out = m.copy()
    out["log_r_applied"] = sign * lr_row
    out["p_corrected"] = sigmoid(z)
    return out


def gap_table(fr: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    for m in models:
        sub = fr[fr["agent_model"] == m]
        for head in ("success", "failure"):
            g = sub[sub["decision"] == head]
            n = int(len(g))
            if n == 0:
                rows.append({"model_id": m, "head": head, "decisions": 0,
                             "correct": 0, "empirical_precision": None,
                             "mean_raw_score": None, "mean_corrected_score": None,
                             "raw_gap": None, "corrected_gap": None,
                             "absolute_raw_gap": None,
                             "absolute_corrected_gap": None,
                             "gap_absorbed_by_correction": None})
                continue
            correct = int(g["correct"].sum())
            prec = safe_ratio(correct, n)
            raw = float(g["p_raw"].mean())
            cor = float(g["p_corrected"].mean())
            raw_gap = raw - prec
            cor_gap = cor - prec
            absorbed = (1.0 - abs(cor_gap) / abs(raw_gap)) if raw_gap != 0 else None
            rows.append({
                "model_id": m, "head": head, "decisions": n, "correct": correct,
                "empirical_precision": prec,
                "mean_raw_score": raw, "mean_corrected_score": cor,
                "raw_gap": raw_gap, "corrected_gap": cor_gap,
                "absolute_raw_gap": abs(raw_gap),
                "absolute_corrected_gap": abs(cor_gap),
                "gap_absorbed_by_correction": absorbed,
            })
    return pd.DataFrame(rows)


def identity_check(gap: pd.DataFrame, cal0d: pd.DataFrame, prev: pd.DataFrame,
                   pri: pd.DataFrame) -> dict:
    """Phase 0E must reproduce the frozen Phase 0D numbers exactly."""
    j = gap.merge(cal0d[["model_id", "head", "decisions", "correct",
                         "empirical_precision", "mean_decision_score",
                         "calibration_gap"]],
                  on=["model_id", "head"], how="left", suffixes=("", "_0d"))
    d_mean = float(np.nanmax(np.abs(
        j["mean_raw_score"].to_numpy(dtype=np.float64)
        - j["mean_decision_score"].to_numpy(dtype=np.float64))))
    d_prec = float(np.nanmax(np.abs(
        j["empirical_precision"].to_numpy(dtype=np.float64)
        - j["empirical_precision_0d"].to_numpy(dtype=np.float64))))
    d_gap = float(np.nanmax(np.abs(
        j["raw_gap"].to_numpy(dtype=np.float64)
        - j["calibration_gap"].to_numpy(dtype=np.float64))))
    d_dec = int((j["decisions"] != j["decisions_0d"]).sum())
    k = pri.merge(prev[["model_id", "test_success_rate", "train_success_rate",
                        "prior_shift"]], on="model_id", suffixes=("", "_0d"))
    d_test = float(np.max(np.abs(k["test_success_prior"]
                                 - k["test_success_rate"])))
    d_train = float(np.max(np.abs(k["train_success_prior"]
                                  - k["train_success_rate"])))
    d_shift = float(np.max(np.abs(k["prior_shift"] - k["prior_shift_0d"])))
    ok = bool(d_mean < TOL and d_prec < TOL and d_gap < TOL and d_dec == 0
              and d_test < TOL and d_train < TOL and d_shift < TOL)
    return {
        "n_model_head_rows": int(len(j)),
        "max_abs_mean_raw_score_minus_phase0d_mean_decision_score": d_mean,
        "max_abs_empirical_precision_minus_phase0d": d_prec,
        "max_abs_raw_gap_minus_phase0d_calibration_gap": d_gap,
        "model_head_rows_with_decision_count_mismatch": d_dec,
        "max_abs_test_prior_minus_phase0d_test_success_rate": d_test,
        "max_abs_train_prior_minus_phase0d_train_success_rate": d_train,
        "max_abs_prior_shift_minus_phase0d": d_shift,
        "tolerance": TOL,
        "IDENTITY_REPRODUCED": ok,
    }


def count_matrices(fr: pd.DataFrame, traj: pd.DataFrame, models: list):
    """Per-(model, task) count/sum matrices for the task-cluster bootstrap."""
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
    C = np.zeros((n_m, 2, n_t), dtype=np.int64)
    S = np.zeros((n_m, 2, n_t), dtype=np.float64)
    SR = np.zeros((n_m, 2, n_t), dtype=np.float64)
    K = np.zeros((n_m, 2, n_t), dtype=np.int64)
    fm = fr["agent_model"].map(model_to_code).to_numpy()
    ft = fr["instance_id"].map(task_to_code).to_numpy()
    fh = fr["head_code"].to_numpy()
    corr_scores = fr["p_corrected"].to_numpy(dtype=np.float64)
    correct = fr["correct"].to_numpy(dtype=np.int64)
    np.add.at(C, (fm, fh, ft), 1)
    np.add.at(S, (fm, fh, ft), corr_scores)
    np.add.at(SR, (fm, fh, ft), fr["p_raw"].to_numpy(dtype=np.float64))
    np.add.at(K, (fm, fh, ft), correct)
    return {"models": models, "tasks": tasks, "N": N, "R": R, "C": C, "S": S,
            "SR": SR, "K": K}


def _rho_masked(x, y, mask):
    m = mask & ~np.isnan(x) & ~np.isnan(y)
    if int(m.sum()) < 2:
        return None
    return spearman(x[m], y[m])


def _range_defined(values, defined):
    if defined is None or not np.any(defined):
        return None
    vals = values[defined]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return float(vals.max() - vals.min())


def _replicate_stats(cm: dict, w: np.ndarray, nondeg: np.ndarray) -> dict:
    Nv = cm["N"] @ w
    Rv = cm["R"] @ w
    totN, totR = Nv.sum(), Rv.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        test_prior = np.where(Nv > 0, Rv / np.where(Nv > 0, Nv, 1), np.nan)
        tr_den = totN - Nv
        train_prior = np.where(tr_den > 0,
                               (totR - Rv) / np.where(tr_den > 0, tr_den, 1),
                               np.nan)
        prior_shift = test_prior - train_prior
    out = {"prior_shift": prior_shift, "gap": {}, "raw_gap": {}, "defined": {}}
    for head, h in (("success", 0), ("failure", 1)):
        Cv = cm["C"][:, h, :] @ w
        Sv = cm["S"][:, h, :] @ w
        SRv = cm["SR"][:, h, :] @ w
        Kv = cm["K"][:, h, :] @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(Cv > 0, Sv / np.where(Cv > 0, Cv, 1), np.nan)
            raw_mean = np.where(Cv > 0, SRv / np.where(Cv > 0, Cv, 1), np.nan)
            prec = np.where(Cv > 0, Kv / np.where(Cv > 0, Cv, 1), np.nan)
        out["gap"][head] = mean - prec
        out["raw_gap"][head] = raw_mean - prec
        out["defined"][head] = (Cv > 0) & nondeg
    return out


def bootstrap(cm: dict, models: list, nondeg_models: set, replicates: int,
              seed: int) -> dict:
    nondeg = np.asarray([m in nondeg_models for m in models], dtype=bool)
    rng = np.random.default_rng(seed)
    n_tasks = len(cm["tasks"])
    succ_range, fail_range = [], []
    raw_succ_range, raw_fail_range = [], []
    rho_succ, rho_fail = [], []
    raw_rho_succ, raw_rho_fail = [], []
    for _ in range(replicates):
        drawn = rng.integers(0, n_tasks, size=n_tasks)
        w = np.bincount(drawn, minlength=n_tasks).astype(np.float64)
        rt = _replicate_stats(cm, w, nondeg)
        sd = rt["defined"]["success"]
        fd = rt["defined"]["failure"]
        succ_range.append(_range_defined(rt["gap"]["success"], sd))
        fail_range.append(_range_defined(rt["gap"]["failure"], fd))
        raw_succ_range.append(_range_defined(rt["raw_gap"]["success"], sd))
        raw_fail_range.append(_range_defined(rt["raw_gap"]["failure"], fd))
        ps = rt["prior_shift"]
        raw_rho_succ.append(_rho_masked(ps, rt["raw_gap"]["success"], sd))
        raw_rho_fail.append(_rho_masked(ps, rt["raw_gap"]["failure"], fd))
        rho_succ.append(_rho_masked(ps, rt["gap"]["success"], sd))
        rho_fail.append(_rho_masked(ps, rt["gap"]["failure"], fd))
    return {
        "method": "task-cluster (instance_id) bootstrap; each replicate resamples "
                  "task IDs with replacement, includes all model trajectories of "
                  "every selected task, and recomputes fold priors and residual "
                  "gaps within the replicate; the oracle correction coefficient is "
                  "held at its frozen point value",
        "replicates": int(replicates),
        "seed": int(seed),
        "models_used": sorted(nondeg_models),
        "RAW_SUCCESS_GAP_RANGE": percentile_ci(raw_succ_range),
        "RAW_FAILURE_GAP_RANGE": percentile_ci(raw_fail_range),
        "RESIDUAL_SUCCESS_GAP_RANGE": percentile_ci(succ_range),
        "RESIDUAL_FAILURE_GAP_RANGE": percentile_ci(fail_range),
        "RHO_PRIOR_SHIFT_vs_RAW_SUCCESS_GAP": percentile_ci(raw_rho_succ),
        "RHO_PRIOR_SHIFT_vs_RAW_FAILURE_GAP": percentile_ci(raw_rho_fail),
        "RHO_PRIOR_SHIFT_vs_RESIDUAL_SUCCESS_GAP": percentile_ci(rho_succ),
        "RHO_PRIOR_SHIFT_vs_RESIDUAL_FAILURE_GAP": percentile_ci(rho_fail),
    }


def build_gate(gap: pd.DataFrame, corr: pd.DataFrame, boot: dict,
               residual_range: dict) -> dict:
    """Gate: mechanical, pre-registered."""
    nondeg = set(corr[~corr["prior_degenerate"]]["model_id"])
    g = gap[gap["model_id"].isin(nondeg) & gap["decisions"].gt(0)].copy()
    g["denominator_ge_20"] = g["decisions"] >= MIN_DENOM_FOR_GAP_COUNT
    large = g[g["denominator_ge_20"]
              & (g["absolute_corrected_gap"] >= LARGE_GAP)]
    models_large = sorted(large["model_id"].unique().tolist())
    count = len(models_large)
    head_rows = {}
    substantial_any = False
    for head in ("success", "failure"):
        sub = g[g["head"] == head]
        vals = sub["corrected_gap"].dropna().to_numpy(dtype=np.float64)
        rng = float(vals.max() - vals.min()) if vals.size else None
        ci_lo = boot[f"RESIDUAL_{head.upper()}_GAP_RANGE"]["ci_lower"]
        substantial = bool(rng is not None and ci_lo is not None
                           and rng >= RESID_RANGE_MIN
                           and ci_lo > RESID_CI_LOWER_MIN)
        substantial_any = substantial_any or substantial
        head_rows[head] = {
            "models_with_defined_denominator": int(sub["decisions"].gt(0).sum()),
            "residual_gap_range": rng,
            "residual_gap_range_bootstrap_ci_lower": ci_lo,
            "residual_gap_range_bootstrap_ci_upper":
                boot[f"RESIDUAL_{head.upper()}_GAP_RANGE"]["ci_upper"],
            "raw_gap_range": residual_range[head]["raw_range"],
            "residual_range_absorbed_by_correction":
                residual_range[head]["range_absorbed_by_correction"],
            "SUBSTANTIAL": substantial,
        }
    if count >= 3 and substantial_any:
        overall = "GO"
    elif count <= 1 and not substantial_any:
        overall = "PIVOT_TO_PRIOR_SHIFT_ADAPTATION"
    else:
        overall = "PARTIAL_RESIDUAL_MISCALIBRATION"
    return {
        "definition": {
            "oracle_correction":
                "log-odds of each stop-point head probability shifted by ln r_m; "
                "success head r_m = [pi_test/(1-pi_test)]/[pi_train/(1-pi_train)], "
                "failure head uses the complementary prior so its log shift is "
                "exactly -ln r_m",
            "RESIDUAL_LARGE_GAP_COUNT":
                "held-out models that are not prior-degenerate, have >= 20 "
                "decisions in at least one head, and have |corrected calibration "
                "gap| >= 0.10 in at least one head",
            "RESIDUAL_HETEROGENEITY_SUBSTANTIAL":
                "either head has a corrected calibration-gap range >= 0.10 across "
                "non-degenerate models with a defined denominator AND a "
                "task-cluster bootstrap 95% CI lower bound > 0.05 for that range",
            "OVERALL":
                "GO iff RESIDUAL_LARGE_GAP_COUNT >= 3 AND "
                "RESIDUAL_HETEROGENEITY_SUBSTANTIAL; "
                "PIVOT_TO_PRIOR_SHIFT_ADAPTATION iff RESIDUAL_LARGE_GAP_COUNT <= 1 "
                "AND NOT RESIDUAL_HETEROGENEITY_SUBSTANTIAL; otherwise "
                "PARTIAL_RESIDUAL_MISCALIBRATION (a mechanically defined middle "
                "category the two-way rule does not cover)",
        },
        "RESIDUAL_LARGE_GAP_COUNT": count,
        "models_with_residual_large_gap": models_large,
        "RESIDUAL_HETEROGENEITY_SUBSTANTIAL": substantial_any,
        "per_head": head_rows,
        "prior_degenerate_models_excluded_from_gate": sorted(
            set(corr[corr["prior_degenerate"]]["model_id"])),
        "OVERALL": overall,
        "note": "mechanical, pre-registered; thresholds were not altered after "
                "seeing results",
    }


def main() -> int:
    ensure_dirs(ANA)
    inputs = hash_inputs()
    write_json(OUT / "input_hashes.json", inputs)
    prov = verify_provenance(inputs)
    write_json(OUT / "input_status.json", prov)
    print("inputs present:", inputs["all_inputs_present"],
          "| provenance all match:", prov["ALL_INPUT_HASHES_MATCH"], flush=True)

    fr, traj, join_info = load_decided()
    models = sorted(traj["model_id"].unique().tolist())
    pri = prior_table(traj, models)
    corr = correction_table(pri)
    fr = apply_correction(fr, corr)
    gap = gap_table(fr, models)

    cal0d = pd.read_csv(INPUT_FILES["phase0d_decision_score_calibration.csv"])
    prev0d = pd.read_csv(INPUT_FILES["phase0d_prevalence_shift.csv"])
    ident = identity_check(gap, cal0d, prev0d, pri)
    ident["stop_prefix_join"] = join_info
    write_json(ANA / "identity_check.json", as_builtin(ident))
    print("identity reproduced:", ident["IDENTITY_REPRODUCED"], flush=True)

    corr.to_csv(ANA / "prior_correction.csv", index=False, encoding="utf-8")
    write_json(ANA / "prior_correction.json",
               as_builtin(corr.to_dict(orient="records")))
    gap.to_csv(ANA / "corrected_gaps.csv", index=False, encoding="utf-8")

    nondeg = set(corr[~corr["prior_degenerate"]]["model_id"])
    residual_range = {}
    for head in ("success", "failure"):
        sub = gap[gap["head"] == head]
        nd = sub[sub["model_id"].isin(nondeg) & sub["decisions"].gt(0)]
        raw = nd["raw_gap"].dropna().to_numpy(dtype=np.float64)
        res = nd["corrected_gap"].dropna().to_numpy(dtype=np.float64)
        raw_range = float(raw.max() - raw.min()) if raw.size else None
        res_range = float(res.max() - res.min()) if res.size else None
        residual_range[head] = {
            "raw_range": raw_range,
            "residual_range": res_range,
            "range_absorbed_by_correction":
                (1.0 - res_range / raw_range)
                if (raw_range not in (None, 0.0) and res_range is not None)
                else None,
            "max_absolute_raw_gap":
                float(np.max(np.abs(raw))) if raw.size else None,
            "max_absolute_residual_gap":
                float(np.max(np.abs(res))) if res.size else None,
            "per_model_raw_gap": {str(a): float(b) for a, b in
                                  zip(nd["model_id"], nd["raw_gap"])},
            "per_model_residual_gap": {str(a): float(b) for a, b in
                                       zip(nd["model_id"], nd["corrected_gap"])},
        }
    write_json(ANA / "residual_gap_range.json", as_builtin({
        "definition": {
            "universe": "models that are not prior-degenerate, with a defined "
                        "denominator > 0 in that head",
            "correction": "oracle label-prior correction; coefficient held at its "
                          "frozen point value",
        },
        "heads": residual_range,
    }))

    cm = count_matrices(fr, traj, models)
    boot = bootstrap(cm, models, nondeg, REPLICATES, SEED)
    write_json(ANA / "bootstrap_results.json", as_builtin(boot))

    prior_map = dict(zip(pri["model_id"], pri["prior_shift"]))
    rho = {}
    for head in ("success", "failure"):
        sub = gap[gap["head"] == head]
        nd = sub[sub["model_id"].isin(nondeg) & sub["decisions"].gt(0)]
        x = [prior_map[m] for m in nd["model_id"]]
        raw_y = list(nd["raw_gap"].astype(float))
        res_y = list(nd["corrected_gap"].astype(float))
        raw_r = spearman(x, raw_y)
        res_r = spearman(x, res_y)
        up = head.upper()
        rho[head] = {
            "n_models": int(len(nd)),
            "universe": "non-degenerate held-out models with a defined "
                        "denominator > 0 in this head",
            "quantity": "calibration error = calibration gap = mean decision score "
                        "- empirical precision (the quantity the oracle prior "
                        "correction moves)",
            "rho_prior_shift_vs_raw_calibration_error": raw_r,
            "rho_prior_shift_vs_residual_calibration_error": res_r,
            "raw_bootstrap_ci": {
                "lower": boot[f"RHO_PRIOR_SHIFT_vs_RAW_{up}_GAP"]["ci_lower"],
                "upper": boot[f"RHO_PRIOR_SHIFT_vs_RAW_{up}_GAP"]["ci_upper"],
                "median": boot[f"RHO_PRIOR_SHIFT_vs_RAW_{up}_GAP"]["median"],
                "n_valid_replicates":
                    boot[f"RHO_PRIOR_SHIFT_vs_RAW_{up}_GAP"]
                    ["n_valid_replicates"],
            },
            "residual_bootstrap_ci": {
                "lower": boot[f"RHO_PRIOR_SHIFT_vs_RESIDUAL_{up}_GAP"]
                ["ci_lower"],
                "upper": boot[f"RHO_PRIOR_SHIFT_vs_RESIDUAL_{up}_GAP"]
                ["ci_upper"],
                "median": boot[f"RHO_PRIOR_SHIFT_vs_RESIDUAL_{up}_GAP"]
                ["median"],
                "n_valid_replicates":
                    boot[f"RHO_PRIOR_SHIFT_vs_RESIDUAL_{up}_GAP"]
                    ["n_valid_replicates"],
            },
            "association_absorbed_by_correction":
                (1.0 - abs(res_r) / abs(raw_r))
                if (raw_r not in (None, 0.0) and res_r is not None) else None,
        }
    # Cross-check against Phase 0D on Phase 0D's own quantity and universe: the
    # association of PRIOR_SHIFT with the DECISION ERROR RATE (1 - precision) over
    # all 10 held-out models. That is not the same statistic as the calibration-gap
    # association used above, so both are recorded.
    rho0d = read_json(INPUT_FILES["phase0d_prior_shift_correlations.json"])
    cross = {}
    for head in ("success", "failure"):
        s10 = gap[(gap["head"] == head) & gap["decisions"].gt(0)]
        xs = [prior_map[m] for m in s10["model_id"]]
        err = [1.0 - float(v) for v in s10["empirical_precision"]]
        gp = [float(v) for v in s10["raw_gap"]]
        r_err = spearman(xs, err)
        r_gap = spearman(xs, gp)
        p0d = rho0d["primary_all_models"][head.upper()]["rho"]
        cross[head.upper()] = {
            "n_models": int(len(s10)),
            "rho_prior_shift_vs_decision_error_rate": r_err,
            "phase0d_rho": p0d,
            "abs_difference_vs_phase0d": abs(r_err - p0d),
            "rho_prior_shift_vs_raw_calibration_gap": r_gap,
            "note": "Phase 0D's headline rho uses the decision error rate; the raw "
                    "calibration-gap association is the Phase 0E baseline for the "
                    "residual branch and differs from it in the success head",
        }
    rho["phase0d_cross_check"] = {
        "note": "Phase 0D's headline rho is Spearman(PRIOR_SHIFT, DECISION ERROR "
                "RATE) over all 10 held-out models. It is reproduced here exactly. "
                "Phase 0E's own baseline and residual figures use the CALIBRATION "
                "GAP on the 9-model non-degenerate universe.",
        "per_head": cross,
        "max_abs_difference_error_rate_branch":
            max(v["abs_difference_vs_phase0d"] for v in cross.values()),
    }
    write_json(ANA / "residual_correlations.json", as_builtin(rho))

    gate = build_gate(gap, corr, boot, residual_range)
    write_json(ANA / "phase0e_gate.json", as_builtin(gate))
    write_json(ANA / "phase0e_summary.json", as_builtin({
        "prior_correction": corr.to_dict(orient="records"),
        "corrected_gaps": gap.to_dict(orient="records"),
        "residual_range": residual_range,
        "residual_correlations": rho,
        "gate": gate,
    }))
    import json
    print(json.dumps(as_builtin({k: v for k, v in gate.items()
                                 if k != "definition"}),
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
