# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B_D - Sections 3-10: metrics, bootstrap and gates.

Reads the frozen Phase 2B held-out predictions plus the re-scanned threshold
decisions and produces the per-target/head/threshold calibration metrics, the
Phase 0E oracle label-prior correction, the task-cluster bootstrap, the
denominator adequacy table and the diagnostic gate. Predictor retraining = 0.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from d_common import (ANA, ANCHOR, B2, BOOTSTRAP_REPLICATES,
                      BOOTSTRAP_SEED_BASE, DECS, FAILURE_SCORE_COL, HEAD_CODE,
                      LARGE_GAP, MIN_DECISIONS, MIN_ELIGIBLE_TARGETS, OUT_2B,
                      PREDICTOR, PRED_2B, ROBUST_GAP, SUCCESS_SCORE_COL,
                      THRESHOLDS, WORK, anchor_tag, as_builtin, ensure_dirs,
                      log_odds, logit, percentile_ci, provider_family,
                      read_json, set_vendor_env, sigmoid, thr_tag, write_json)

set_vendor_env()

HEADS = ("success", "failure")


def load_inputs():
    pre = read_json(B2 / "preflight.json")
    models = sorted(pre["eligible_models"])
    traj = pd.read_csv(B2 / "trajectory_index.csv")
    preds = []
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        p = pd.read_parquet(PRED_2B / (tag + ".parquet"), columns=[
            "traj_id", "model_id", "instance_id", "prefix_step_idx", "label",
            SUCCESS_SCORE_COL, FAILURE_SCORE_COL])
        p["fold_tag"] = tag
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    decs = {}
    for thr in THRESHOLDS:
        parts = []
        for i, model in enumerate(models):
            tag = "fold-%02d" % i
            d = pd.read_csv(DECS / thr_tag(thr) / (tag + ".csv"))
            parts.append(d)
        decs[float(thr)] = pd.concat(parts, ignore_index=True)
    return pre, models, traj, pred, decs


def prior_table(traj: pd.DataFrame, models: list) -> pd.DataFrame:
    """Frozen Phase 0E / Phase 2B leave-one-model-out success priors."""
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
            "log_odds_ratio_failure_head": (-log_r
                                            if log_r is not None else None),
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
        for head in HEADS:
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
                "model_id": m, "head": head, "decisions": n,
                "correct": correct, "empirical_precision": prec,
                "mean_raw_score": raw, "mean_corrected_score": cor,
                "raw_gap": raw - prec, "corrected_gap": cor - prec,
                "absolute_raw_gap": abs(raw - prec),
                "absolute_corrected_gap": abs(cor - prec),
                "distinct_tasks": int(g["instance_id"].nunique()),
            })
    return pd.DataFrame(rows)


def bootstrap_gap(sub: pd.DataFrame, replicates: int, seed: int) -> dict:
    """Task-cluster bootstrap of the signed corrected calibration gap."""
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


def main() -> int:
    t0 = time.time()
    ensure_dirs(ANA)
    pre, models, traj, pred, decs = load_inputs()
    model_index = {m: i for i, m in enumerate(models)}
    pri = prior_table(traj, models)
    prior_map = pri.set_index("model_id")

    metric_frames, join_info_all, boot_all = [], {}, {}
    for thr in THRESHOLDS:
        key = "%.3f" % thr
        fr, join_info = stop_point_frame(decs[float(thr)], pred)
        fr = apply_correction(fr, pri)
        join_info_all[key] = join_info
        gap = gap_table(fr, models)
        gap = gap.merge(pri[["model_id", "test_success_prior",
                             "train_success_prior", "prior_shift",
                             "prior_degenerate",
                             "log_odds_ratio_success_head",
                             "log_odds_ratio_failure_head"]],
                        on="model_id", how="left")
        gap["provider_family"] = gap["model_id"].map(provider_family)
        gap.insert(0, "threshold", float(thr))
        boot = {}
        for m in models:
            sub_m = fr[fr["holdout_model"] == m]
            for head in HEADS:
                g = sub_m[sub_m["decision"] == head]
                if len(g) < MIN_DECISIONS:
                    continue
                seed = (BOOTSTRAP_SEED_BASE + model_index[m] * 10
                        + HEAD_CODE[head])
                boot[(m, head)] = bootstrap_gap(
                    g, BOOTSTRAP_REPLICATES, seed)
        boot_all[key] = {"%s::%s" % (m, h): as_builtin(v)
                         for (m, h), v in sorted(boot.items())}
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
            for p, lo, hi in zip(gap["corrected_gap"],
                                 gap["bootstrap_ci_lower"],
                                 gap["bootstrap_ci_upper"])]
        gap["prior_degenerate"] = gap["prior_degenerate"].astype(bool)
        metric_frames.append(gap)
        print("[%s] decided=%s robust_rows=%d" % (
            key, join_info["decided_trajectories"],
            int(gap["target_head_large_gap"].sum())), flush=True)

    metrics = pd.concat(metric_frames, ignore_index=True)
    cols = ["threshold", "model_id", "provider_family", "head", "decisions",
            "correct", "empirical_precision", "mean_raw_score",
            "mean_corrected_score", "raw_gap", "corrected_gap",
            "absolute_raw_gap", "absolute_corrected_gap", "distinct_tasks",
            "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_median",
            "bootstrap_n_task_clusters", "target_head_large_gap",
            "test_success_prior", "train_success_prior", "prior_shift",
            "prior_degenerate", "log_odds_ratio_success_head",
            "log_odds_ratio_failure_head"]
    metrics = metrics[cols]
    metrics.to_csv(ANA / "threshold_target_metrics.csv", index=False,
                   encoding="utf-8")

    write_json(ANA / "bootstrap_results.json", as_builtin({
        "section": "5 ROBUST LARGE GAP - BOOTSTRAP",
        "method": "task_name-cluster bootstrap; each replicate resamples task "
                  "IDs (instance_id) with replacement, keeps every decision "
                  "row of the drawn tasks, and recomputes empirical precision, "
                  "corrected mean score and the signed corrected gap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed_rule": "43000 + fold_index*10 + head_index (success=0, "
                     "failure=1), identical to the Phase 2B rule",
        "correction_coefficient": "held at its frozen Phase 0E point value; "
                                  "the predictor is never retrained",
        "unit": "signed corrected calibration gap = corrected mean score - "
                "empirical precision",
        "ci": "2.5 / 97.5 percentile over replicates",
        "stop_point_join": join_info_all,
        "per_threshold": boot_all,
    }))

    # Sections 6/9/10: denominator adequacy and per-head reports.
    den_rows, head_report = [], {}
    for thr in THRESHOLDS:
        key = "%.3f" % thr
        one = metrics[metrics["threshold"] == float(thr)]
        head_report[key] = {}
        for head in HEADS:
            sub = one[(one["head"] == head) & (~one["prior_degenerate"])
                      & one["decisions"].ge(MIN_DECISIONS)]
            n = int(len(sub))
            dec = sub["decisions"].astype(np.int64)
            rob = sub[sub["target_head_large_gap"]]
            adequate = bool(n >= MIN_ELIGIBLE_TARGETS)
            den_rows.append({
                "threshold": float(thr), "head": head,
                "eligible_targets": n,
                "total_decisions": int(dec.sum()),
                "median_decisions_per_eligible_target": (
                    float(dec.median()) if n else None),
                "min_decisions": int(dec.min()) if n else None,
                "max_decisions": int(dec.max()) if n else None,
                "robust_targets": int(len(rob)),
                "head_denominator_adequate": adequate,
            })
            head_report[key][head] = {
                "eligible_targets": n,
                "total_decisions": int(dec.sum()),
                "median_decisions_per_eligible_target": (
                    float(dec.median()) if n else None),
                "min_decisions": int(dec.min()) if n else None,
                "max_decisions": int(dec.max()) if n else None,
                "robust_targets": int(len(rob)),
                "head_denominator_adequate": adequate,
                "eligibility": "decisions >= %d and non-degenerate target prior"
                               % MIN_DECISIONS,
                "min_eligible_targets_rule": MIN_ELIGIBLE_TARGETS,
            }
    den = pd.DataFrame(den_rows)
    den.to_csv(ANA / "denominator_summary.csv", index=False, encoding="utf-8")

    rob_all = metrics[metrics["target_head_large_gap"]].copy()
    rob_all = rob_all.sort_values(["threshold", "model_id", "head"])
    rob_all[["threshold", "model_id", "provider_family", "head", "decisions",
             "correct", "empirical_precision", "mean_raw_score",
             "mean_corrected_score", "corrected_gap",
             "absolute_corrected_gap", "bootstrap_ci_lower",
             "bootstrap_ci_upper", "bootstrap_median",
             "bootstrap_n_task_clusters", "test_success_prior",
             "train_success_prior", "prior_shift", "prior_degenerate"]].to_csv(
        ANA / "robust_targets.csv", index=False, encoding="utf-8")

    # Sections 7/8: per-threshold diagnostic gate and the final gate.
    per_threshold, n_signal_primary, n_signal_variant = {}, 0, 0
    for thr in THRESHOLDS:
        key = "%.3f" % thr
        one = metrics[metrics["threshold"] == float(thr)]
        rob = one[one["target_head_large_gap"]]
        qualifying = []
        for m in sorted(set(rob["model_id"])):
            rows = rob[rob["model_id"] == m]
            mx = float(rows["absolute_corrected_gap"].max())
            qualifying.append({
                "model_id": m,
                "provider_family": provider_family(m),
                "heads_with_robust_large_gap": sorted(rows["head"].tolist()),
                "max_abs_corrected_gap_among_robust_heads": mx,
                "has_abs_gap_ge_0_10": bool(mx >= LARGE_GAP),
            })
        n_models = int(len(qualifying))
        n_ge_010 = int(sum(1 for q in qualifying if q["has_abs_gap_ge_0_10"]))
        families = sorted({q["provider_family"] for q in qualifying})
        contributing = sorted({h for q in qualifying
                               for h in q["heads_with_robust_large_gap"]})
        adequate = {h: bool(head_report[key][h]["head_denominator_adequate"])
                    for h in contributing}
        primary_ok = bool(all(adequate.values())) if contributing else False
        variant_ok = bool(any(adequate.values())) if contributing else False
        base = bool(n_models >= 3 and n_ge_010 >= 2 and len(families) >= 2)
        sig_primary = bool(base and primary_ok)
        sig_variant = bool(base and variant_ok)
        n_signal_primary += int(sig_primary)
        n_signal_variant += int(sig_variant)
        per_threshold[key] = {
            "threshold": float(thr),
            "models_with_robust_large_gap": n_models,
            "of_which_abs_gap_ge_0_10": n_ge_010,
            "provider_families": families,
            "distinct_provider_families": int(len(families)),
            "contributing_heads": contributing,
            "contributing_head_denominator_adequate": adequate,
            "base_conditions_met": base,
            "relevant_head_adequate_primary_all_heads": primary_ok,
            "relevant_head_adequate_variant_any_head": variant_ok,
            "THRESHOLD_SIGNAL": sig_primary,
            "THRESHOLD_SIGNAL_variant_any_head": sig_variant,
            "qualifying_models": qualifying,
        }
        print("[%s] robust_models=%d ge_0.10=%d families=%s signal=%s"
              % (key, n_models, n_ge_010, families, sig_primary), flush=True)
    result_primary = ("DENOMINATOR_RESOLVED_SIGNAL"
                      if n_signal_primary >= 2
                      else "NO_CROSS_BENCHMARK_REPLICATION")
    result_variant = ("DENOMINATOR_RESOLVED_SIGNAL"
                      if n_signal_variant >= 2
                      else "NO_CROSS_BENCHMARK_REPLICATION")
    gate = as_builtin({
        "section": "7/8 DIAGNOSTIC GATE",
        "definition": {
            "TARGET_HEAD_LARGE_GAP": "decisions >= %d AND |point corrected "
                "gap| >= %.2f AND bootstrap 95%% CI excludes 0 AND the CI "
                "retains the point-estimate sign" % (MIN_DECISIONS, ROBUST_GAP),
            "HEAD_DENOMINATOR_ADEQUATE": "eligible targets >= %d"
                                         % MIN_ELIGIBLE_TARGETS,
            "THRESHOLD_SIGNAL": ">= 3 distinct exact models with "
                "TARGET_HEAD_LARGE_GAP in >= 1 head AND >= 2 of them with "
                "|corrected gap| >= %.2f AND >= 2 provider families AND the "
                "relevant head has HEAD_DENOMINATOR_ADEQUATE = TRUE"
                % LARGE_GAP,
            "PHASE2B_D_RESULT": "DENOMINATOR_RESOLVED_SIGNAL iff "
                "THRESHOLD_SIGNAL = TRUE at >= 2 of the 3 thresholds; "
                "otherwise NO_CROSS_BENCHMARK_REPLICATION",
        },
        "relevant_head_readings": {
            "primary": "every head contributing a robust large gap must be "
                       "denominator-adequate (used for PHASE2B_D_RESULT)",
            "variant": "at least one contributing head is adequate "
                       "(sensitivity only)",
        },
        "per_threshold": per_threshold,
        "thresholds_with_signal_primary": int(n_signal_primary),
        "thresholds_with_signal_variant": int(n_signal_variant),
        "PHASE2B_D_RESULT": result_primary,
        "PHASE2B_D_RESULT_variant_any_head": result_variant,
        "PHASE2B_RESULT_frozen": read_json(
            OUT_2B / "analysis" / "phase2b_gate.json")["PHASE2B_RESULT"],
        "phase2b_primary_result_modified": False,
        "note": "mechanical, pre-registered in PROTOCOL.md before any metric "
                "was computed; the frozen Phase 2B primary result is unchanged",
    })
    write_json(ANA / "diagnostic_gate.json", gate)

    # Section 1/2: reproduction anchor (decision level + bootstrap level).
    grid = read_json(WORK / "policy_grid_summary.json")
    ref = read_json(OUT_2B / "analysis" / "target_bootstrap.json")
    ref_boot = ref["per_target_head"]
    new_boot = boot_all["%.3f" % ANCHOR]
    keys_ref = set(ref_boot)
    keys_new = set(new_boot)
    diffs = {"point_corrected_gap": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    exact = True
    for k in sorted(keys_ref & keys_new):
        a, b = ref_boot[k], new_boot[k]
        for f in ("point_corrected_gap", "ci_lower", "ci_upper"):
            if a.get(f) is None or b.get(f) is None:
                exact = exact and (a.get(f) is None and b.get(f) is None)
                continue
            d = abs(float(a[f]) - float(b[f]))
            diffs[f] = max(diffs[f], d)
            if d > 1e-12:
                exact = False
        for f in ("replicates", "seed", "n_task_clusters", "decisions"):
            if int(a[f]) != int(b[f]):
                exact = False
    repro = {
        "anchor_threshold": ANCHOR,
        "decision_level": grid["reproduction"],
        "bootstrap_level": {
            "comparison": "re-derived 0.950 task-cluster bootstrap vs frozen "
                          "Phase 2B analysis/target_bootstrap.json",
            "reference": str(OUT_2B / "analysis" / "target_bootstrap.json"),
            "reference_entries": int(len(keys_ref)),
            "rederived_entries": int(len(keys_new)),
            "key_sets_identical": bool(keys_ref == keys_new),
            "max_abs_diff_point_corrected_gap": diffs["point_corrected_gap"],
            "max_abs_diff_ci_lower": diffs["ci_lower"],
            "max_abs_diff_ci_upper": diffs["ci_upper"],
            "seed_and_counts_identical": bool(exact),
            "BOOTSTRAP_REPRODUCTION_PASS": bool(exact and keys_ref == keys_new),
        },
    }
    repro["REPRODUCTION_PASS"] = bool(
        repro["decision_level"]["REPRODUCTION_PASS"]
        and repro["bootstrap_level"]["BOOTSTRAP_REPRODUCTION_PASS"])
    write_json(ANA / "reproduction_check.json", as_builtin(repro))

    summary = {
        "section": "PHASE2B_D ANALYSIS",
        "models": len(models),
        "thresholds": list(THRESHOLDS),
        "stop_point_join": join_info_all,
        "per_head_denominator": head_report,
        "thresholds_with_signal_primary": int(n_signal_primary),
        "PHASE2B_D_RESULT": result_primary,
        "PHASE2B_D_RESULT_variant_any_head": result_variant,
        "REPRODUCTION_PASS": repro["REPRODUCTION_PASS"],
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
        "predictor_retraining": 0, "new_lightgbm_training": 0,
        "new_trajectories": 0,
    }
    write_json(WORK / "analysis_summary_d.json", as_builtin(summary))
    print(as_builtin({k: v for k, v in summary.items()
                      if k != "per_head_denominator"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
