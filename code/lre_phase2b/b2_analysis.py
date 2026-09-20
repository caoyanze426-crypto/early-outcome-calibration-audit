# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - Sections 13/14/15/16/17/18/19/20/21/24.

Reads the frozen per-fold held-out predictions and policy decisions and produces
the per-target calibration metrics, the Phase 0E oracle label-prior correction,
the task-cluster bootstrap, the pre-registered cross-benchmark gate and the
descriptive comparison artifacts. No predictor is retrained and no decision is
changed here. Offline only.
"""
from __future__ import annotations

import sys

from b2_common import (ANA, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED_BASE,
                       FAILURE_SCORE_COL, HEAD_CODE, LARGE_GAP, MIN_DECISIONS,
                       OUT, OUT_2A, PRED, ROBUST_GAP, SUCCESS_SCORE_COL, WORK,
                       as_builtin, ensure_dirs, log_odds, logit, percentile_ci,
                       provider_family, read_json, sigmoid, spearman, write_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRED_DIR = PRED / "heldout_prefix_predictions"
DEC_DIR = PRED / "policy_decisions"
TOL = 1e-9
SWE_ROBUST_TARGETS = 2
SWE_SUCCESS_GAP = 0.1377
SWE_FAILURE_GAP = 0.1107


def load_frames():
    pre = read_json(WORK / "preflight.json")
    models = sorted(pre["eligible_models"])
    traj = pd.read_csv(WORK / "trajectory_index.csv")
    preds, decs = [], []
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        p = pd.read_parquet(PRED_DIR / (tag + ".parquet"), columns=[
            "traj_id", "model_id", "instance_id", "prefix_step_idx", "label",
            SUCCESS_SCORE_COL, FAILURE_SCORE_COL])
        p["fold_tag"] = tag
        preds.append(p)
        d = pd.read_csv(DEC_DIR / (tag + ".csv"))
        decs.append(d)
    pred = pd.concat(preds, ignore_index=True)
    dec = pd.concat(decs, ignore_index=True)
    return pre, models, traj, pred, dec


def prior_table(traj: pd.DataFrame, models: list) -> pd.DataFrame:
    resolved = traj["resolved"].to_numpy(dtype=np.int64)
    model = traj["model"].astype(str).to_numpy()
    rows = []
    for m in models:
        tm = model == m
        n_test = int(tm.sum())
        n_train = int((~tm).sum())
        s_test = int(resolved[tm].sum())
        s_train = int(resolved[~tm].sum())
        pt = s_test / n_test if n_test else None
        ptr = s_train / n_train if n_train else None
        degenerate = bool(pt in (0.0, 1.0) or ptr in (0.0, 1.0))
        lt = float(log_odds(pt)) if pt is not None else None
        ltr = float(log_odds(ptr)) if ptr is not None else None
        log_r = (lt - ltr) if (lt is not None and ltr is not None) else None
        rows.append({
            "model_id": m,
            "test_trajectories": n_test, "test_successes": s_test,
            "test_success_prior": pt,
            "train_trajectories_other_n1": n_train,
            "train_successes_other_n1": s_train,
            "train_success_prior": ptr,
            "prior_shift": (pt - ptr) if (pt is not None and ptr is not None)
            else None,
            "log_odds_ratio_success_head": log_r,
            "log_odds_ratio_failure_head": (-log_r if log_r is not None else None),
            "prior_degenerate": degenerate,
        })
    return pd.DataFrame(rows)


def stop_point_frame(dec: pd.DataFrame, pred: pd.DataFrame):
    d = dec[dec["decided"].astype(bool)][
        ["traj_id", "holdout_model", "decision", "decision_step",
         "decision_score", "label"]].copy()
    p = pred[["traj_id", "instance_id", "prefix_step_idx", SUCCESS_SCORE_COL,
              FAILURE_SCORE_COL]].copy()
    m = d.merge(p, left_on=["traj_id", "decision_step"],
                right_on=["traj_id", "prefix_step_idx"], how="left")
    m["p_raw"] = np.where(m["decision"].to_numpy() == "success",
                          m[SUCCESS_SCORE_COL].to_numpy(),
                          m[FAILURE_SCORE_COL].to_numpy())
    m["correct"] = np.where(m["decision"].to_numpy() == "success",
                            (m["label"].to_numpy() == 1).astype(np.int64),
                            (m["label"].to_numpy() == 0).astype(np.int64))
    m["head_code"] = m["decision"].map(HEAD_CODE).astype(np.int64)
    gap = np.abs(m["p_raw"].to_numpy(dtype=np.float64)
                 - m["decision_score"].to_numpy(dtype=np.float64))
    info = {
        "decided_trajectories": int(len(m)),
        "stop_prefix_join_missing": int(m["p_raw"].isna().sum()),
        "max_abs_stop_score_minus_policy_decision_score": (
            float(np.nanmax(gap)) if len(m) else None),
    }
    return m, info


def apply_correction(m: pd.DataFrame, pri: pd.DataFrame) -> pd.DataFrame:
    lr = dict(zip(pri["model_id"], pri["log_odds_ratio_success_head"]))
    sign = np.where(m["head_code"].to_numpy() == 0, 1.0, -1.0)
    lr_row = m["holdout_model"].map(lr).to_numpy(dtype=np.float64)
    z = logit(m["p_raw"].to_numpy(dtype=np.float64)) + sign * lr_row
    out = m.copy()
    out["log_r_applied"] = sign * lr_row
    out["p_corrected"] = sigmoid(z)
    return out


def gap_table(fr: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    for m in models:
        sub = fr[fr["holdout_model"] == m]
        for head in ("success", "failure"):
            g = sub[sub["decision"] == head]
            n = int(len(g))
            if n == 0:
                rows.append({"model_id": m, "head": head, "decisions": 0,
                             "correct": 0, "empirical_precision": None,
                             "mean_raw_score": None,
                             "mean_corrected_score": None, "raw_gap": None,
                             "corrected_gap": None,
                             "absolute_raw_gap": None,
                             "absolute_corrected_gap": None,
                             "distinct_tasks": 0})
                continue
            correct = int(g["correct"].sum())
            prec = correct / n
            raw = float(g["p_raw"].mean())
            cor = float(g["p_corrected"].mean())
            rows.append({
                "model_id": m, "head": head, "decisions": n, "correct": correct,
                "empirical_precision": prec,
                "mean_raw_score": raw, "mean_corrected_score": cor,
                "raw_gap": raw - prec, "corrected_gap": cor - prec,
                "absolute_raw_gap": abs(raw - prec),
                "absolute_corrected_gap": abs(cor - prec),
                "distinct_tasks": int(g["instance_id"].nunique()),
            })
    return pd.DataFrame(rows)


def bootstrap_gap(sub: pd.DataFrame, replicates: int, seed: int) -> dict:
    """Task-cluster bootstrap of the signed corrected calibration gap.

    Resampling unit is the task (instance_id) cluster: each replicate draws
    n_tasks task IDs with replacement and keeps every decision row of the drawn
    tasks. The oracle prior-correction coefficient is held at its frozen point
    value (Phase 0E lineage); only precision, corrected mean score and the gap
    are recomputed. The predictor is never retrained.
    """
    grp = sub.groupby("instance_id", sort=False)
    C = grp.size().to_numpy(dtype=np.float64)
    K = grp["correct"].sum().to_numpy(dtype=np.float64)
    S = grp["p_corrected"].sum().to_numpy(dtype=np.float64)
    n_tasks = int(len(C))
    tot_c, tot_k, tot_s = float(C.sum()), float(K.sum()), float(S.sum())
    point = (tot_s / tot_c) - (tot_k / tot_c)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_tasks, size=(int(replicates), n_tasks))
    Cs = C[draws].sum(axis=1)
    Ks = K[draws].sum(axis=1)
    Ss = S[draws].sum(axis=1)
    ok = Cs > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(ok, Ks / np.where(ok, Cs, 1.0), np.nan)
        mean = np.where(ok, Ss / np.where(ok, Cs, 1.0), np.nan)
    gaps = mean - prec
    ci = percentile_ci(gaps)
    return {
        "point_corrected_gap": float(point),
        "point_precision": float(tot_k / tot_c),
        "point_corrected_mean_score": float(tot_s / tot_c),
        "decisions": int(tot_c),
        "n_task_clusters": n_tasks,
        "replicates": int(replicates),
        "seed": int(seed),
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "bootstrap_median": ci["median"],
        "n_valid_replicates": ci["n_valid_replicates"],
    }


def robust_flag(point_gap, ci_lower, ci_upper) -> bool:
    if point_gap is None or ci_lower is None or ci_upper is None:
        return False
    if abs(point_gap) < ROBUST_GAP:
        return False
    if ci_lower > 0.0 and point_gap > 0.0:
        return True
    if ci_upper < 0.0 and point_gap < 0.0:
        return True
    return False


def head_signal(gap: pd.DataFrame, head: str, label: str) -> dict:
    sub = gap[(gap["head"] == head) & (~gap["prior_degenerate"])
              & gap["decisions"].ge(MIN_DECISIONS)].copy()
    vals = sub["corrected_gap"].astype(float).to_numpy()
    absv = np.abs(vals)
    return {
        "head": head,
        "eligible_targets": int(len(sub)),
        "eligibility": "decisions >= %d and non-degenerate target prior"
                       % MIN_DECISIONS,
        "median_abs_corrected_gap": float(np.median(absv)) if len(absv) else None,
        "max_abs_corrected_gap": float(absv.max()) if len(absv) else None,
        "residual_gap_range": (float(vals.max() - vals.min())
                               if len(vals) else None),
        "signed_corrected_gap_min": float(vals.min()) if len(vals) else None,
        "signed_corrected_gap_max": float(vals.max()) if len(vals) else None,
        "n_abs_gap_ge_0_05": int((absv >= 0.05).sum()),
        "n_abs_gap_ge_0_08": int((absv >= 0.08).sum()),
        "n_abs_gap_ge_0_10": int((absv >= 0.10).sum()),
        "per_target_abs_gap": {str(a): float(b) for a, b in
                               zip(sub["model_id"], absv)},
        "q4_label": label,
    }


def main() -> int:
    import time

    t0 = time.time()
    ensure_dirs(ANA)
    pre, models, traj, pred, dec = load_frames()
    model_index = {m: i for i, m in enumerate(models)}
    pri = prior_table(traj, models)
    fr, join_info = stop_point_frame(dec, pred)
    fr = apply_correction(fr, pri)
    gap = gap_table(fr, models)
    gap = gap.merge(pri[["model_id", "test_success_prior", "train_success_prior",
                         "prior_shift", "prior_degenerate",
                         "log_odds_ratio_success_head",
                         "log_odds_ratio_failure_head"]],
                    on="model_id", how="left")
    gap["provider_family"] = gap["model_id"].map(provider_family)

    # Section 16: task-cluster bootstrap per target/head with >= 20 decisions.
    boot = {}
    for m in models:
        sub_m = fr[fr["holdout_model"] == m]
        for head in ("success", "failure"):
            g = sub_m[sub_m["decision"] == head]
            if len(g) < MIN_DECISIONS:
                continue
            seed = BOOTSTRAP_SEED_BASE + model_index[m] * 10 + HEAD_CODE[head]
            boot[(m, head)] = bootstrap_gap(g, BOOTSTRAP_REPLICATES, seed)
    gap["bootstrap_ci_lower"] = [
        (boot.get((a, b)) or {}).get("ci_lower")
        for a, b in zip(gap["model_id"], gap["head"])]
    gap["bootstrap_ci_upper"] = [
        (boot.get((a, b)) or {}).get("ci_upper")
        for a, b in zip(gap["model_id"], gap["head"])]
    gap["bootstrap_median"] = [
        (boot.get((a, b)) or {}).get("bootstrap_median")
        for a, b in zip(gap["model_id"], gap["head"])]
    gap["bootstrap_n_task_clusters"] = [
        (boot.get((a, b)) or {}).get("n_task_clusters")
        for a, b in zip(gap["model_id"], gap["head"])]
    gap["target_head_large_gap"] = [
        robust_flag(p, lo, hi)
        for p, lo, hi in zip(gap["corrected_gap"], gap["bootstrap_ci_lower"],
                             gap["bootstrap_ci_upper"])]
    gap.to_csv(ANA / "per_target_metrics.csv", index=False, encoding="utf-8")
    write_json(ANA / "target_bootstrap.json", as_builtin({
        "method": "task-cluster (task_name / instance_id) bootstrap; each "
                  "replicate resamples task IDs with replacement, keeps every "
                  "decision row of the drawn tasks, and recomputes empirical "
                  "precision, corrected mean score and the signed corrected "
                  "calibration gap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed_rule": "43000 + model_index*10 + head_index (success=0, failure=1)",
        "correction_coefficient": "held at its frozen point value (Phase 0E "
                                  "lineage); the predictor is never retrained",
        "unit": "signed corrected calibration gap = corrected mean score - "
                "empirical precision",
        "ci": "2.5 / 97.5 percentile over replicates",
        "stop_point_join": join_info,
        "per_target_head": {"%s::%s" % (m, h): as_builtin(v)
                            for (m, h), v in sorted(boot.items())},
    }))

    # Section 13: per-target coverage / NO_STOP.
    cov_rows = []
    for m in models:
        sub = dec[dec["holdout_model"] == m]
        n_dec = int(sub["decided"].astype(bool).sum())
        cand = traj[traj["model"].astype(str) == m]
        n_test = int(len(cand))
        n_pred_traj = int(pred[pred["model_id"].astype(str) == m]["traj_id"].nunique())
        cov_rows.append({
            "model_id": m, "provider_family": provider_family(m),
            "usable_trajectories": n_test,
            "test_prefix_trajectories": n_pred_traj,
            "trajectory_count_match": bool(n_test == n_pred_traj),
            "decided": n_dec, "NO_STOP": int(len(sub) - n_dec),
            "coverage": (n_dec / n_test) if n_test else None,
            "coverage_pct": (100.0 * n_dec / n_test) if n_test else None,
            "original_resolve_rate": float(cand["resolved"].mean())
            if n_test else None,
        })
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(ANA / "per_target_coverage.csv", index=False, encoding="utf-8")

    # Section 15: target-specific signal per head.
    signal = {
        "success": head_signal(gap, "success", "SUCCESS"),
        "failure": head_signal(gap, "failure", "FAILURE"),
    }
    write_json(ANA / "target_signal.json", as_builtin({
        "section": "15 TARGET-SPECIFIC SIGNAL",
        "thresholds": {"MIN_DECISIONS": MIN_DECISIONS},
        "per_head": signal,
    }))

    # Section 17: robust large-gap targets.
    rob = gap[gap["target_head_large_gap"]].copy()
    rob = rob.sort_values(["model_id", "head"])
    rob_out = rob[[
        "model_id", "provider_family", "head", "decisions", "correct",
        "empirical_precision", "mean_corrected_score", "corrected_gap",
        "absolute_corrected_gap", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "bootstrap_median", "bootstrap_n_task_clusters",
        "test_success_prior", "train_success_prior", "prior_shift"]]
    rob_out.to_csv(ANA / "robust_targets.csv", index=False, encoding="utf-8")

    # Section 18: cross-benchmark replication gate.
    qualifying = []
    for m in models:
        rows = gap[(gap["model_id"] == m) & gap["target_head_large_gap"]]
        if not len(rows):
            continue
        mx = float(rows["absolute_corrected_gap"].max())
        qualifying.append({
            "model_id": m, "provider_family": provider_family(m),
            "heads_with_robust_large_gap": sorted(rows["head"].tolist()),
            "max_abs_corrected_gap_among_robust_heads": mx,
            "has_abs_gap_ge_0_10": bool(mx >= LARGE_GAP),
        })
    n_models = len(qualifying)
    n_ge_010 = int(sum(1 for q in qualifying if q["has_abs_gap_ge_0_10"]))
    families = sorted({q["provider_family"] for q in qualifying})
    not_single_family = len(families) > 1
    signal_true = bool(n_models >= 3 and n_ge_010 >= 2 and not_single_family)
    gate = {
        "section": "18 CROSS-BENCHMARK REPLICATION GATE",
        "definition": {
            "TARGET_HEAD_LARGE_GAP": "decision denominator >= %d AND "
                "|point corrected calibration gap| >= %.2f AND bootstrap 95%% CI "
                "excludes 0 AND the CI retains the point-estimate sign"
                % (MIN_DECISIONS, ROBUST_GAP),
            "TERMINAL_TARGET_SPECIFIC_SIGNAL": ">= 3 distinct exact models with "
                "TARGET_HEAD_LARGE_GAP = TRUE in at least one head AND >= 2 of "
                "those models have |corrected gap| >= %.2f AND the large-gap "
                "targets are not all from a single provider family" % LARGE_GAP,
        },
        "models_with_robust_large_gap": n_models,
        "of_which_abs_gap_ge_0_10": n_ge_010,
        "provider_families": families,
        "not_all_single_provider_family": bool(not_single_family),
        "qualifying_models": qualifying,
        "TERMINAL_TARGET_SPECIFIC_SIGNAL": signal_true,
        "PHASE2B_RESULT": ("CROSS_BENCHMARK_SIGNAL" if signal_true
                           else "NO_CROSS_BENCHMARK_SIGNAL"),
        "note": "mechanical, pre-registered; the gate was not altered after "
                "seeing results",
    }
    write_json(ANA / "phase2b_gate.json", as_builtin(gate))

    # Section 19: heterogeneity descriptive.
    desc = {}
    for head in ("success", "failure"):
        sub = gap[(gap["head"] == head) & (~gap["prior_degenerate"])
                  & gap["decisions"].ge(MIN_DECISIONS)]
        v = sub["corrected_gap"].astype(float).to_numpy()
        desc[head] = {
            "eligible_targets": int(len(sub)),
            "signed_corrected_gap_range": (float(v.max() - v.min())
                                           if len(v) else None),
            "min": float(v.min()) if len(v) else None,
            "max": float(v.max()) if len(v) else None,
            "rho_test_success_prior_vs_corrected_gap": spearman(
                list(sub["test_success_prior"].astype(float)), list(v)),
            "rho_train_success_prior_vs_corrected_gap": spearman(
                list(sub["train_success_prior"].astype(float)), list(v)),
        }
    write_json(ANA / "heterogeneity_descriptive.json", as_builtin({
        "section": "19 HETEROGENEITY DESCRIPTIVE",
        "note": "descriptive only; no broad-heterogeneity gate in Phase 2B",
        "per_head": desc,
    }))

    # Section 20: SWE-bench comparison (descriptive cross-benchmark context).
    gap_abs_all = gap[gap["decisions"].gt(0)]["absolute_corrected_gap"].astype(float)
    swe = {
        "section": "20 SWE-BENCH COMPARISON",
        "swe_bench_frozen": {
            "number_robust_frozen_targets": SWE_ROBUST_TARGETS,
            "abs_gap_success_approx": SWE_SUCCESS_GAP,
            "abs_gap_failure_approx": SWE_FAILURE_GAP,
        },
        "terminalbench": {
            "scaffold": "terminus-2",
            "number_models_with_robust_large_gap": n_models,
            "number_target_head_rows_with_robust_large_gap": int(len(rob)),
            "median_abs_corrected_gap_over_all_target_head_rows":
                float(gap_abs_all.median()) if len(gap_abs_all) else None,
            "max_abs_corrected_gap_over_all_target_head_rows":
                float(gap_abs_all.max()) if len(gap_abs_all) else None,
            "median_abs_corrected_gap_over_robust_rows":
                float(rob["absolute_corrected_gap"].median()) if len(rob) else None,
            "max_abs_corrected_gap_over_robust_rows":
                float(rob["absolute_corrected_gap"].max()) if len(rob) else None,
        },
        "same_model_names_required": False,
        "note": "descriptive cross-benchmark context only",
    }
    write_json(ANA / "swe_comparison.json", as_builtin(swe))

    # Section 21: provider-family concentration for robust large-gap targets.
    prov = {
        "section": "21 PROVIDER-FAMILY CONCENTRATION",
        "labels_from": "exact dataset model labels only",
        "large_gap_targets": [
            {"model_id": q["model_id"], "provider_family": q["provider_family"]}
            for q in qualifying],
        "provider_family_counts": {
            f: int(sum(1 for q in qualifying if q["provider_family"] == f))
            for f in families},
        "distinct_provider_families": len(families),
        "provider_performance_interpretation": False,
    }
    write_json(ANA / "provider_concentration.json", as_builtin(prov))

    # Section 24: secondary crossed-scaffold follow-up feasibility metadata only.
    SM = OUT_2A / "analysis" / "same_model_cross_scaffold.csv"
    TU = OUT_2A / "analysis" / "trajectory_usability.csv"
    TARGET_MODEL = "gpt-5-mini@openai"
    SCAFFOLDS_24 = ["terminus-2", "mini-swe-agent", "openhands", "codex"]
    pair_rows, summary_rows, per_scaffold = [], [], {}
    if SM.exists():
        sm = pd.read_csv(SM)
        pair_rows = sm[(sm["model"] == TARGET_MODEL)
                       & (sm["row_type"] == "scaffold_pair")].to_dict("records")
        summary_rows = sm[(sm["model"] == TARGET_MODEL)
                          & (sm["row_type"] == "model_summary")].to_dict("records")
    if TU.exists():
        tu = pd.read_csv(TU)
        tu = tu[tu["model"] == TARGET_MODEL]
        for r in tu.itertuples():
            per_scaffold[str(r.agent)] = {
                "all_public_trials": int(r.all_public_trials),
                "usable_full_trajectories": int(r.usable_full_trajectories),
                "usable_fraction": float(r.usable_fraction),
                "unique_tasks_usable": int(r.unique_tasks_usable),
            }
    feas = {
        "section": "24 SECONDARY CROSSED-SCAFFOLD FOLLOWUP (FEASIBILITY ONLY)",
        "run_now": False,
        "model": TARGET_MODEL,
        "scaffolds": SCAFFOLDS_24,
        "scaffolds_present_in_dataset": sorted(per_scaffold),
        "all_four_scaffolds_available": bool(
            set(SCAFFOLDS_24).issubset(set(per_scaffold))),
        "per_scaffold_metadata": per_scaffold,
        "model_summary_rows": summary_rows,
        "scaffold_pair_rows": pair_rows,
        "source": str(SM),
        "predictions_trained_for_other_scaffolds": 0,
        "note": "metadata only; no predictions and no training outside the fixed "
                "terminus-2 scaffold in Phase 2B",
    }
    write_json(ANA / "phase2c_feasibility.json", as_builtin(feas))

    summary = {
        "section": "PHASE2B ANALYSIS",
        "models": len(models),
        "stop_point_join": join_info,
        "gate": gate["TERMINAL_TARGET_SPECIFIC_SIGNAL"],
        "PHASE2B_RESULT": gate["PHASE2B_RESULT"],
        "robust_target_head_rows": int(len(rob)),
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "predictor_retraining": 0,
    }
    write_json(WORK / "analysis_summary.json", as_builtin(summary))
    print(as_builtin(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
