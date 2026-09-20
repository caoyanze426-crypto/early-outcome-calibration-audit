# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A - same-predictor pairwise analysis, gates and sensitivities.

Consumes only frozen fold outputs: the per-pair target predictions, the per-pair
policy decisions, the frozen pair-fold manifest and the Phase 0B trajectory
universe (resolved labels / instance ids). Nothing is retrained here and no
predictor output is modified.

Offline only: LLM calls = 0, API calls = 0, new trajectories = 0.
"""
from __future__ import annotations

import sys

from common1a import (ANA, FAILURE_SCORE_COL, FOLDS, GEMINI_PRO, MIN_STEP,
                      MIRROR_DROP, MIRROR_KEEP, MODELS, NL, OUT, OUT_0B,
                      PAIR_CI_LOWER_MIN,
                      PAIR_MEDIAN_DIFF_MIN, PAIR_MIN_DECISIONS,
                      PAIR_MIN_ELIGIBLE_FOLDS, PERSIST_MEDIAN_ABS_GAP_MIN,
                      PERSIST_MIN_OCCURRENCES, PERSIST_SAME_SIGN_FRACTION,
                      POLICY_NAME, PREDICTOR, PRED_DEC, PRED_PAIR, REPLICATES,
                      SAFE_LABEL_MIN_STEP,
                      SEED, SUCCESS_SCORE_COL, WORK, as_builtin, ensure_dirs,
                      log_odds, logit, percentile_ci, read_json, safe_ratio, sigmoid,
                      spearman, write_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HEAD_ORDER = ("success", "failure")


def load_decisions() -> pd.DataFrame:
    """All pair-fold policy decisions, with the frozen instance id and label."""
    frames = []
    paths = [p for p in sorted(PRED_DEC.glob("pair-*.csv"))
             if p.stem[5:].isdigit() and len(p.stem) == len("pair-0000")]
    for p in paths:
        d = pd.read_csv(p)
        d["pair_id"] = p.stem
        frames.append(d)
    if not frames:
        raise SystemExit("no pair-fold decisions found")
    dec = pd.concat(frames, ignore_index=True)
    traj = pd.read_csv(OUT_0B / "analysis" / "trajectory_outcomes.csv",
                       usecols=["traj_id", "model_id", "instance_id", "resolved"])
    traj = traj.drop_duplicates("traj_id")
    dec = dec.merge(traj[["traj_id", "instance_id", "resolved"]], on="traj_id",
                    how="left", validate="many_to_one")
    missing = int(dec["instance_id"].isna().sum())
    mismatch = int((dec["resolved"] != dec["label"]).sum())
    dec["correct"] = np.where(
        dec["decision"].to_numpy() == "success",
        (dec["resolved"].to_numpy() == 1).astype(int),
        np.where(dec["decision"].to_numpy() == "failure",
                 (dec["resolved"].to_numpy() == 0).astype(int), -1))
    return dec, {"decisions_rows": int(len(dec)), "instance_id_missing": missing,
                 "resolved_label_mismatch_vs_phase0b": mismatch}


def agent_task_index(dec: pd.DataFrame):
    """Per (pair, agent): the set of tasks with an adapter-PASS trajectory."""
    out = {}
    avail = dec.groupby(["pair_id", "agent_model"], sort=False)["instance_id"]
    for (pair, agent), vals in avail:
        out[(pair, agent)] = set(vals.dropna().astype(str))
    return out


def head_stats(frame: pd.DataFrame):
    """Raw decision-class metrics for one (pair, agent, head) on given rows."""
    n = int(len(frame))
    if n == 0:
        return {"decisions": 0, "correct": 0, "precision": None,
                "mean_stop_score": None, "raw_gap": None}
    correct = int(frame["correct"].sum())
    prec = safe_ratio(correct, n)
    mean_score = float(frame["decision_score"].mean())
    return {"decisions": n, "correct": correct, "precision": prec,
            "mean_stop_score": mean_score, "raw_gap": mean_score - prec}


def prior_stats(traj: pd.DataFrame, fold_agents: set, holdout: set,
                common_tasks: set | None = None) -> dict:
    """Train-universe prior and per-holdout-target priors (section 12)."""
    df = traj
    if common_tasks is not None:
        df = traj[traj["instance_id"].astype(str).isin(common_tasks)]
    tr = df[df["model_id"].astype(str).isin(fold_agents)]
    out = {
        "train_trajectories": int(len(tr)),
        "train_successes": int(tr["resolved"].sum()),
        "train_success_prior": safe_ratio(int(tr["resolved"].sum()), len(tr)),
    }
    for agent in sorted(holdout):
        sub = df[df["model_id"].astype(str) == agent]
        out[f"target_trajectories_{agent}"] = int(len(sub))
        out[f"target_successes_{agent}"] = int(sub["resolved"].sum())
        out[f"target_success_prior_{agent}"] = safe_ratio(
            int(sub["resolved"].sum()), len(sub))
    return out


def oracle_coefficient(pi_train, pi_target):
    """ln r for the success head; the failure head uses exactly -ln r."""
    if pi_train is None or pi_target is None:
        return None, True
    degenerate = bool(pi_train in (0.0, 1.0) or pi_target in (0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = float(log_odds(pi_target))
        ltr = float(log_odds(pi_train))
    return lt - ltr, degenerate


def correct_scores(scores: np.ndarray, head: str, log_r) -> np.ndarray:
    """Apply the oracle prior correction to raw calibrated stop scores."""
    sign = 1.0 if head == "success" else -1.0
    return sigmoid(logit(scores) + sign * log_r)


def compute_pairwise(pairs, dec: pd.DataFrame, traj: pd.DataFrame,
                     tag: str) -> tuple[pd.DataFrame, dict]:
    """Section 9-14 metrics for a given set of pair folds, plus bootstrap inputs."""
    rows = []
    boot = {}
    for pair_id, agent_a, agent_b in pairs:
        part = dec[dec["pair_id"] == pair_id]
        common = None
        tasks = {}
        for agent in (agent_a, agent_b):
            s = set(part[part["agent_model"].astype(str) == agent]["instance_id"]
                    .dropna().astype(str))
            tasks[agent] = s
        common = tasks[agent_a] & tasks[agent_b]
        pr = prior_stats(traj, set(MODELS) - {agent_a, agent_b},
                         {agent_a, agent_b}, common)
        pi_train = pr["train_success_prior"]
        per_agent = {}
        for agent in (agent_a, agent_b):
            pi_target = pr[f"target_success_prior_{agent}"]
            log_r, degenerate = oracle_coefficient(pi_train, pi_target)
            s = part[(part["agent_model"].astype(str) == agent)
                     & (part["instance_id"].astype(str).isin(common))]
            per_agent[agent] = {
                "target_success_prior": pi_target, "log_r": log_r,
                "degenerate": degenerate,
                "frames": {h: s[s["decision"] == h] for h in HEAD_ORDER},
                "all_frames": {h: part[(part["agent_model"].astype(str) == agent)
                                       & (part["decision"] == h)]
                               for h in HEAD_ORDER},
            }
        for head in HEAD_ORDER:
            rec = {
                "pair_id": pair_id, "agent_A": agent_a, "agent_B": agent_b,
                "head": head, "analysis": tag,
                "n_A_all_tasks": len(tasks[agent_a]),
                "n_B_all_tasks": len(tasks[agent_b]),
                "n_common_tasks": len(common),
                "train_success_prior": pi_train,
                "train_trajectories": pr["train_trajectories"],
            }
            stats = {}
            for agent, side in ((agent_a, "A"), (agent_b, "B")):
                info = per_agent[agent]
                frame = info["frames"][head]
                st = head_stats(frame)
                log_r = info["log_r"]
                if st["decisions"] > 0 and log_r is not None:
                    corr = correct_scores(
                        frame["decision_score"].to_numpy(dtype=np.float64), head,
                        log_r)
                    corr_mean = float(np.mean(corr))
                else:
                    corr_mean = None
                corr_gap = (corr_mean - st["precision"]) \
                    if (corr_mean is not None and st["precision"] is not None) \
                    else None
                full = head_stats(info["all_frames"][head])
                stats[side] = {"st": st, "corr_mean": corr_mean,
                               "corr_gap": corr_gap, "full": full}
                rec.update({
                    f"{side}_decisions": st["decisions"],
                    f"{side}_correct": st["correct"],
                    f"{side}_precision": st["precision"],
                    f"{side}_mean_stop_score": st["mean_stop_score"],
                    f"{side}_raw_gap": st["raw_gap"],
                    f"{side}_target_success_prior": info["target_success_prior"],
                    f"{side}_target_trajectories": pr[
                        f"target_trajectories_{agent}"],
                    f"{side}_log_odds_ratio": log_r,
                    f"{side}_prior_degenerate": info["degenerate"],
                    f"{side}_corrected_mean_score": corr_mean,
                    f"{side}_corrected_gap": corr_gap,
                    f"{side}_abs_corrected_gap":
                        abs(corr_gap) if corr_gap is not None else None,
                    f"{side}_all_task_decisions": full["decisions"],
                    f"{side}_all_task_precision": full["precision"],
                    f"{side}_all_task_raw_gap": full["raw_gap"],
                    f"{side}_no_stop": int(
                        ((part["agent_model"].astype(str) == agent)
                         & (part["instance_id"].astype(str).isin(common))
                         & (~part["decided"].astype(bool))).sum()),
                })
            eligible = bool(stats["A"]["st"]["decisions"] >= PAIR_MIN_DECISIONS
                            and stats["B"]["st"]["decisions"]
                            >= PAIR_MIN_DECISIONS)
            rec["PAIR_HEAD_ELIGIBLE"] = eligible
            if eligible:
                ga = stats["A"]["corr_gap"]
                gb = stats["B"]["corr_gap"]
                rec["PAIRWISE_RESIDUAL_GAP_DIFFERENCE"] = abs(ga - gb)
                rec["ABS_GAP_A"] = abs(ga)
                rec["ABS_GAP_B"] = abs(gb)
                rec["PRECISION_DIFFERENCE"] = abs(
                    stats["A"]["st"]["precision"] - stats["B"]["st"]["precision"])
                rec["PRIOR_SHIFT_DIFFERENCE"] = abs(
                    per_agent[agent_a]["target_success_prior"]
                    - per_agent[agent_b]["target_success_prior"])
                rec["PAIRWISE_RAW_GAP_DIFFERENCE"] = abs(
                    stats["A"]["st"]["raw_gap"] - stats["B"]["st"]["raw_gap"])
            else:
                for k in ("PAIRWISE_RESIDUAL_GAP_DIFFERENCE", "ABS_GAP_A",
                          "ABS_GAP_B", "PRECISION_DIFFERENCE",
                          "PRIOR_SHIFT_DIFFERENCE",
                          "PAIRWISE_RAW_GAP_DIFFERENCE"):
                    rec[k] = None
            rows.append(rec)
            # Bootstrap inputs: per-task counts and sums on the common support.
            tlist = sorted(common)
            tidx = {t: i for i, t in enumerate(tlist)}
            arr = {}
            for agent in (agent_a, agent_b):
                frame = per_agent[agent]["frames"][head]
                log_r = per_agent[agent]["log_r"]
                n = np.zeros(len(tlist))
                sc = np.zeros(len(tlist))
                cc = np.zeros(len(tlist))
                if len(frame):
                    ti = frame["instance_id"].astype(str).map(tidx).to_numpy()
                    np.add.at(n, ti, 1.0)
                    cc_vals = frame["correct"].to_numpy(dtype=np.float64)
                    np.add.at(cc, ti, cc_vals)
                    raw = frame["decision_score"].to_numpy(dtype=np.float64)
                    corr = correct_scores(raw, head, log_r)
                    np.add.at(sc, ti, corr)
                arr[agent] = {"n": n, "sum_corrected": sc, "sum_correct": cc}
            boot[(pair_id, head)] = {"tasks": tlist, "agents": arr,
                                     "n_common": len(tlist)}
    return pd.DataFrame(rows), boot


def _replicate_agent_stats(entry, idx):
    """Per-agent precision and corrected gap for one resampled task vector."""
    stats = {}
    for agent, arr in entry["agents"].items():
        n = float(arr["n"][idx].sum())
        if n <= 0.0:
            stats[agent] = None
            continue
        precision = float(arr["sum_correct"][idx].sum()) / n
        corrected_mean = float(arr["sum_corrected"][idx].sum()) / n
        stats[agent] = {"n": n, "precision": precision,
                        "corrected_gap": corrected_mean - precision}
    return stats


def pair_gap_differences(boot, pair_ids, head, idx_of=None) -> list:
    """Section 14 statistic over every eligible pair fold.

    With `idx_of` = None the frozen point estimates are used; with a mapping
    pair_id -> resampled positional index array the same price is evaluated on
    one task-cluster bootstrap replicate.
    """
    diffs = []
    for pair_id in pair_ids:
        entry = boot.get((pair_id, head))
        if entry is None or len(entry["agents"]) != 2 or entry["n_common"] == 0:
            continue
        if idx_of is None:
            idx = np.arange(entry["n_common"])
        else:
            idx = idx_of[pair_id]
        stats = _replicate_agent_stats(entry, idx)
        agents = sorted(stats)
        if len(agents) != 2:
            continue
        sa, sb = stats[agents[0]], stats[agents[1]]
        if sa is None or sb is None:
            continue
        if sa["n"] < PAIR_MIN_DECISIONS or sb["n"] < PAIR_MIN_DECISIONS:
            continue
        diffs.append(abs(sa["corrected_gap"] - sb["corrected_gap"]))
    return diffs


def bootstrap_medians(boot, pair_ids, head, seed) -> list:
    """Section 15: resample common instance_ids with replacement, 2000 reps."""
    rng = np.random.default_rng(int(seed))
    medians = []
    for _ in range(REPLICATES):
        idx_of = {}
        for pair_id in pair_ids:
            entry = boot.get((pair_id, head))
            if entry is None or entry["n_common"] == 0:
                continue
            size = int(entry["n_common"])
            idx_of[pair_id] = rng.integers(0, size, size)
        diffs = pair_gap_differences(boot, pair_ids, head, idx_of)
        if diffs:
            medians.append(float(np.median(diffs)))
    return medians


def seed_for(tag: str, head: str) -> int:
    """Deterministic bootstrap seed per declared analysis scope and head."""
    base = {"primary": 0, "exclude_gemini3pro": 100,
            "drop_gpt_5_2_high": 200}[tag]
    return int(SEED) * 100 + base + (1 if head == "success" else 2)


def head_gate(boot, pair_ids, head, seed) -> dict:
    """Sections 14-16 for one head, restricted to `pair_ids`."""
    diffs = pair_gap_differences(boot, pair_ids, head, None)
    median = float(np.median(diffs)) if diffs else None
    replicates = bootstrap_medians(boot, pair_ids, head, seed)
    ci = percentile_ci(replicates)
    passed = bool(
        len(diffs) >= PAIR_MIN_ELIGIBLE_FOLDS
        and median is not None
        and median >= PAIR_MEDIAN_DIFF_MIN
        and ci["ci_lower"] is not None
        and ci["ci_lower"] > PAIR_CI_LOWER_MIN)
    return {
        "head": head,
        "pair_folds_in_scope": int(len(pair_ids)),
        "eligible_pair_folds": int(len(diffs)),
        "eligible_pair_folds_required": PAIR_MIN_ELIGIBLE_FOLDS,
        "median_pairwise_residual_gap_difference": median,
        "median_pairwise_residual_gap_difference_required": PAIR_MEDIAN_DIFF_MIN,
        "bootstrap_ci_lower_required_gt": PAIR_CI_LOWER_MIN,
        "bootstrap": ci,
        "bootstrap_seed": int(seed),
        "replicate_medians": [round(v, 6) for v in replicates],
        "HEAD_SAME_PREDICTOR_HETEROGENEITY": passed,
    }


def same_sign_required(n: int) -> float:
    """Section 17: 7/9, or 80% when exactly 7 or 8 occurrences are eligible."""
    return PERSIST_SAME_SIGN_FRACTION if n >= 9 else 0.80


def persistence_rows(pw: pd.DataFrame, pair_ids) -> list:
    """Section 17 per agent/head persistence over eligible pair-fold occurrences."""
    sub = pw[pw["pair_id"].isin(pair_ids)]
    eps = 1e-9
    rows = []
    for agent in MODELS:
        for head in HEAD_ORDER:
            sel = sub[(sub["head"] == head)
                      & ((sub["agent_A"] == agent) | (sub["agent_B"] == agent))]
            sel = sel.sort_values("pair_id")
            gaps, skipped_small, skipped_degen = [], 0, 0
            for rec in sel.to_dict(orient="records"):
                side = "A" if rec["agent_A"] == agent else "B"
                if not bool(rec["PAIR_HEAD_ELIGIBLE"]):
                    skipped_small += 1
                    continue
                if bool(rec[f"{side}_prior_degenerate"]):
                    skipped_degen += 1
                    continue
                gaps.append(float(rec[f"{side}_corrected_gap"]))
            n = len(gaps)
            pos = sum(1 for g in gaps if g > 0)
            neg = sum(1 for g in gaps if g < 0)
            same_sign = (max(pos, neg) / n) if n else None
            median_signed = float(np.median(gaps)) if n else None
            median_abs = float(np.median(np.abs(gaps))) if n else None
            required = same_sign_required(n) if n else None
            persistent = bool(
                n >= PERSIST_MIN_OCCURRENCES
                and median_abs is not None
                and median_abs >= PERSIST_MEDIAN_ABS_GAP_MIN
                and same_sign is not None
                and same_sign >= (required - eps))
            rows.append({
                "agent_model": agent,
                "head": head,
                "pair_folds_in_scope": int(len(sel)),
                "eligible_occurrences": n,
                "eligible_occurrences_required": PERSIST_MIN_OCCURRENCES,
                "excluded_pairs_below_decision_minimum": skipped_small,
                "excluded_pairs_prior_degenerate": skipped_degen,
                "median_signed_corrected_gap": median_signed,
                "median_abs_corrected_gap": median_abs,
                "median_abs_corrected_gap_required": PERSIST_MEDIAN_ABS_GAP_MIN,
                "positive_gap_fraction": safe_ratio(pos, n),
                "negative_gap_fraction": safe_ratio(neg, n),
                "zero_gap_count": int(n - pos - neg),
                "same_sign_fraction": same_sign,
                "same_sign_fraction_required": required,
                "TARGET_HEAD_PERSISTENT_LARGE_GAP": persistent,
            })
    return rows


def persistence_gate(rows) -> dict:
    """Section 18: >= 2 distinct agent labels persistent in >= 1 head."""
    cells = [f"{r['agent_model']}/{r['head']}" for r in rows
             if bool(r["TARGET_HEAD_PERSISTENT_LARGE_GAP"])]
    agents = sorted({r["agent_model"] for r in rows
                     if bool(r["TARGET_HEAD_PERSISTENT_LARGE_GAP"])})
    return {
        "agent_labels_evaluated": len(MODELS),
        "persistent_agent_head_cells": cells,
        "persistent_agent_labels": agents,
        "distinct_persistent_agent_labels": len(agents),
        "distinct_persistent_agent_labels_required": 2,
        "TARGET_SPECIFIC_PERSISTENCE": bool(len(agents) >= 2),
    }


def median_or_none(values):
    arr = [v for v in values if v is not None]
    return float(np.median(arr)) if arr else None


def secondary_correlations(pw: pd.DataFrame, pair_ids) -> dict:
    """Section 20 descriptive secondary statistics (no gate)."""
    sub = pw[pw["pair_id"].isin(pair_ids)]
    out = {"pair_folds_in_scope": int(len(pair_ids)), "heads": {}}
    for head in HEAD_ORDER:
        h = sub[(sub["head"] == head) & (sub["PAIR_HEAD_ELIGIBLE"] == True)]  # noqa: E712
        prior = h["PRIOR_SHIFT_DIFFERENCE"].tolist()
        resid = h["PAIRWISE_RESIDUAL_GAP_DIFFERENCE"].tolist()
        raw = h["PAIRWISE_RAW_GAP_DIFFERENCE"].tolist()
        prec = h["PRECISION_DIFFERENCE"].tolist()
        reduced = int(sum(1 for a, b in zip(raw, resid)
                          if a is not None and b is not None and b < a))
        out["heads"][head] = {
            "eligible_pair_folds": int(len(h)),
            "spearman_prior_shift_vs_corrected_gap_difference":
                spearman(prior, resid),
            "spearman_prior_shift_vs_raw_gap_difference": spearman(prior, raw),
            "median_raw_gap_difference": median_or_none(raw),
            "median_corrected_gap_difference": median_or_none(resid),
            "pair_folds_where_correction_reduces_gap_difference": reduced,
            "median_precision_difference": median_or_none(prec),
            "max_precision_difference": (max(v for v in prec if v is not None)
                                         if any(v is not None for v in prec)
                                         else None),
        }
    return out


def agent_prior_summary(pw: pd.DataFrame, pair_ids) -> list:
    """Per-agent target success prior on the common-task support."""
    sub = pw[(pw["head"] == "success") & (pw["pair_id"].isin(pair_ids))]
    rows = []
    for agent in MODELS:
        vals = []
        for rec in sub.to_dict(orient="records"):
            if rec["agent_A"] == agent:
                vals.append(rec["A_target_success_prior"])
            elif rec["agent_B"] == agent:
                vals.append(rec["B_target_success_prior"])
        vals = [v for v in vals if v is not None]
        rows.append({
            "agent_model": agent,
            "pair_folds": len(vals),
            "mean_target_success_prior":
                float(np.mean(vals)) if vals else None,
            "min_target_success_prior": min(vals) if vals else None,
            "max_target_success_prior": max(vals) if vals else None,
            "prior_degenerate_in_any_fold":
                bool(any(v in (0.0, 1.0) for v in vals)),
        })
    return rows


def fold_metadata_rows() -> list:
    """Section 23 per-fold engineering record (from each frozen fold_meta.json)."""
    rows = []
    for meta_path in sorted((WORK / "tmp").glob("pair-*/fold_meta.json")):
        m = read_json(meta_path)
        cal = {c["head"]: c for c in m.get("calibration", [])}
        split = m.get("split", {})
        shape = m.get("design_matrix_shape", {})
        status = m.get("head_status", {})
        rows.append({
            "pair_id": m.get("pair_id"),
            "agent_A": m.get("agent_A"),
            "agent_B": m.get("agent_B"),
            "started_utc": m.get("started_utc"),
            "ended_utc": m.get("ended_utc"),
            "wall_clock_seconds": m.get("wall_clock_seconds"),
            "feature_seconds": m.get("feature_seconds"),
            "train_seconds": m.get("train_seconds"),
            "peak_working_set_mib": m.get("peak_working_set_mib"),
            "threads": m.get("threads"),
            "rows_total": split.get("rows_total"),
            "train_rows": split.get("train_rows"),
            "valid_rows": split.get("valid_rows"),
            "test_rows": split.get("test_rows"),
            "test_rows_A": split.get("test_rows_A"),
            "test_rows_B": split.get("test_rows_B"),
            "short_trajectories_dropped_from_trainval":
                split.get("short_trajectories_dropped_from_trainval"),
            "selected_features": shape.get("train", [None, None])[1],
            "best_iteration_success":
                cal.get("safe_success", {}).get("best_iteration"),
            "best_iteration_failure":
                cal.get("safe_failure", {}).get("best_iteration"),
            "safe_success_head": status.get("safe_success_head"),
            "safe_failure_head": status.get("safe_failure_head"),
            "calibration_status": status.get("calibration"),
            "fold_status": m.get("fold_status"),
        })
    return rows


def driver_results() -> dict:
    p = FOLDS / "driver_results.json"
    return read_json(p) if p.exists() else {}


def upstream_manifest_check() -> dict:
    """Re-hash every upstream phase artifact against its own frozen manifest."""
    from common1a import WS, read_manifest, sha256_file

    specs = {
        "late_reversal_early_eval_phase0a":
            WS / "outputs" / "late_reversal_early_eval_phase0a",
        "late_reversal_early_eval_phase0b_signal_hunt": OUT_0B,
        "late_reversal_early_eval_phase0c":
            WS / "outputs" / "late_reversal_early_eval_phase0c",
        "earlyeval_phase0d_cross_agent_calibration":
            WS / "outputs" / "earlyeval_phase0d_cross_agent_calibration",
        "earlyeval_phase0e_prior_shift_decomposition":
            WS / "outputs" / "earlyeval_phase0e_prior_shift_decomposition",
    }
    out = {}
    for name, root in specs.items():
        rec = read_manifest(root / "artifact_sha256sums.txt")
        mismatches = []
        for rel, expected in sorted(rec.items()):
            p = root / rel
            observed = sha256_file(p) if p.exists() else None
            if observed != expected:
                mismatches.append({"path": rel, "expected": expected,
                                   "observed": observed})
        out[name] = {
            "manifest": str(root / "artifact_sha256sums.txt"),
            "manifest_entries": len(rec),
            "matched": len(rec) - len(mismatches),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "INTACT": bool(len(rec) > 0 and not mismatches),
        }
    return out


def rel_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def write_artifact_hashes() -> int:
    from common1a import sha256_file

    lines = []
    for p in rel_files(OUT):
        rel = p.relative_to(OUT).as_posix()
        if rel == "artifact_sha256sums.txt":
            continue
        lines.append(f"{sha256_file(p)}  {rel}")
    (OUT / "artifact_sha256sums.txt").write_text(NL.join(lines) + NL, "utf-8",
                                                 newline=NL)
    return len(lines)


def render_protocol(ctx) -> str:
    lines = [
        "# EARLYEVAL_PHASE1A - SAME_PREDICTOR_MULTI_AGENT_TRANSFER",
        "",
        "PREDICTOR = " + PREDICTOR,
        "POLICY = " + POLICY_NAME + " (dual head, success_thr=0.95, "
        "failure_thr=0.95, min_step=0, consecutive=1)",
        "PAIR FOLDS = 45 (all unordered label pairs of the frozen 10-label "
        "universe)",
        "TRAINING UNIVERSE = 8 remaining labels; VALID = deterministic "
        "Phase 0B split of those 8 labels; TEST = both held-out labels.",
        "PRIMARY SUPPORT = COMMON_TASK_SUPPORT_AB (instance_id with an "
        "adapter-PASS trajectory for both held-out labels).",
        "",
        "## Frozen constants",
        "",
        "- PAIR_MIN_DECISIONS = 20",
        "- PAIR_MIN_ELIGIBLE_FOLDS = 20",
        "- PAIR_MEDIAN_DIFF_MIN = 0.05",
        "- PAIR_CI_LOWER_MIN = 0.03",
        "- PERSIST_MIN_OCCURRENCES = 7",
        "- PERSIST_MEDIAN_ABS_GAP_MIN = 0.08",
        "- PERSIST_SAME_SIGN_FRACTION = 7/9",
        "- BOOTSTRAP_REPLICATES = 2000, seeded per declared scope",
        "",
        "## Oracle prior correction",
        "",
        "p_corrected = sigmoid(logit(p_raw) + sign * ln r) with",
        "ln r = log_odds(pi_target) - log_odds(pi_train), sign = +1 for the "
        "success head and -1 for the failure head (complementary prior).",
        "Predictor outputs and original decisions are never modified; the "
        "correction is analysis-only. pi_train and pi_target are measured on "
        "COMMON_TASK_SUPPORT_AB. A prior of exactly 0 or 1 is flagged "
        "PRIOR_DEGENERATE and preserved without smoothing.",
        "",
        "## Gates (frozen before any result was inspected)",
        "",
        "HEAD_SAME_PREDICTOR_HETEROGENEITY = TRUE iff eligible pair folds "
        ">= 20 AND median pairwise residual gap difference >= 0.05 AND "
        "task-cluster bootstrap 95% CI lower bound > 0.03.",
        "SAME_PREDICTOR_HETEROGENEITY = TRUE iff either head passes.",
        "TARGET_HEAD_PERSISTENT_LARGE_GAP = TRUE iff eligible occurrences "
        ">= 7 AND median absolute corrected gap >= 0.08 AND same-sign "
        "fraction >= 7/9 (or >= 80% when exactly 7 or 8 occurrences are "
        "eligible).",
        "TARGET_SPECIFIC_PERSISTENCE = TRUE iff >= 2 distinct agent labels are "
        "persistent in >= 1 head.",
        "PHASE1A_RESULT = STRONG_PASS (both), PARTIAL_PASS (exactly one), "
        "NO_PASS (neither).",
        "",
        "## Offline guarantees",
        "",
        "API calls = 0, LLM calls = 0, new trajectories = 0, threshold sweep "
        "= 0, method modification = 0. Thresholds, hyperparameters, the "
        "feature lineage and the calibration procedure are reused from the "
        "Phase 0B freeze; the only intended design change is 9 train + 1 "
        "held-out -> 8 train + 2 held-out.",
        "",
        "## Deviations",
        "",
    ]
    lines += [f"- {d}" for d in ctx["deviations"]]
    lines.append("")
    return NL.join(lines)


def render_report(ctx) -> str:
    ex = ctx["execution"]
    heads = ctx["head_gates"]
    gates = ctx["gate"]
    pers = ctx["persistence"]
    lines = ["# EARLYEVAL_PHASE1A - FINAL RETURN", "", "## A. EXECUTION", ""]
    lines += [
        f"pair folds planned = {ex['pair_folds_planned']}",
        f"pair folds completed = {ex['pair_folds_completed']}",
        f"pair folds failed = {ex['pair_folds_failed']}",
        f"training wall-clock total (sum of fold wall clocks) = "
        f"{ex['training_wall_clock_seconds_total']} s",
        f"driver wall-clock total = {ex['driver_wall_clock_seconds']} s",
        "API calls = 0",
        "LLM calls = 0",
        "",
        "## B. INPUT STATUS",
        "",
        f"all frozen input hashes = {ctx['inputs']['ALL_INPUTS_OK']} "
        "(Phase 0B output manifest + Phase 0B work-side lineage)",
        f"EarlyEval commit = {ctx['inputs']['earlyeval_commit_observed']}",
        f"clone clean = {ctx['inputs']['earlyeval_clone_clean']}",
        "",
    ]
    for letter, head in (("C", "success"), ("D", "failure")):
        h = heads[head]
        lines += [
            f"## {letter}. PAIR-LEVEL {head.upper()} HEAD", "",
            f"eligible pairs = {h['eligible_pair_folds']}",
            f"median pairwise corrected-gap difference = "
            f"{h['median_pairwise_residual_gap_difference']}",
            f"95% bootstrap CI = "
            f"[{h['bootstrap']['ci_lower']}, {h['bootstrap']['ci_upper']}]",
            f"HEAD_SAME_PREDICTOR_HETEROGENEITY = "
            f"{h['HEAD_SAME_PREDICTOR_HETEROGENEITY']}",
            "",
        ]
    lines += [
        "## E. SAME-PREDICTOR GATE", "",
        f"SAME_PREDICTOR_HETEROGENEITY = "
        f"{'TRUE' if gates['SAME_PREDICTOR_HETEROGENEITY'] else 'FALSE'}",
        "",
        "## F. TARGET PERSISTENCE", "",
        "| agent | head | eligible occurrences | median signed corrected gap "
        "| median abs corrected gap | same-sign fraction | persistent |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in pers["rows"]:
        lines.append(
            f"| {r['agent_model']} | {r['head']} | {r['eligible_occurrences']} "
            f"| {r['median_signed_corrected_gap']} "
            f"| {r['median_abs_corrected_gap']} "
            f"| {r['same_sign_fraction']} "
            f"| {r['TARGET_HEAD_PERSISTENT_LARGE_GAP']} |")
    lines += [
        "",
        f"TARGET_SPECIFIC_PERSISTENCE = "
        f"{'TRUE' if pers['TARGET_SPECIFIC_PERSISTENCE'] else 'FALSE'}",
        f"persistent agent/head cells = "
        f"{pers['persistent_agent_head_cells']}",
        "",
        "## G. PRIMARY RESULT", "",
        f"PHASE1A_RESULT = {gates['PHASE1A_RESULT']}",
        "",
        "## H. GEMINI-3-PRO SENSITIVITY", "",
    ]
    lines += _sensitivity_lines(ctx["gemini"])
    lines += ["", "## I. MIRROR-LABEL SENSITIVITY", ""]
    lines += _sensitivity_lines(ctx["mirror"])
    lines += ["", "## J. SECONDARY CORRELATIONS", ""]
    for head, s in ctx["secondary"]["heads"].items():
        lines += [
            f"- {head} head: eligible pairs = {s['eligible_pair_folds']}, "
            f"spearman(prior-shift, corrected-gap difference) = "
            f"{s['spearman_prior_shift_vs_corrected_gap_difference']}, "
            f"spearman(prior-shift, raw-gap difference) = "
            f"{s['spearman_prior_shift_vs_raw_gap_difference']}, "
            f"median raw gap difference = {s['median_raw_gap_difference']}, "
            f"median corrected gap difference = "
            f"{s['median_corrected_gap_difference']}, "
            f"pairs where correction reduces the gap difference = "
            f"{s['pair_folds_where_correction_reduces_gap_difference']}, "
            f"median precision difference = "
            f"{s['median_precision_difference']}",
        ]
    lines += ["", "## K. ENGINEERING FAILURES / DEVIATIONS", ""]
    if ex["fold_failures"]:
        lines += [f"- {f}" for f in ex["fold_failures"]]
    else:
        lines.append("- no FOLD_FAIL; every frozen pair fold completed")
    lines += [f"- {d}" for d in ctx["deviations"]]
    lines += [
        "",
        "## L. ARTIFACT PATH + HASH MANIFEST",
        "",
        f"directory = {OUT}",
        f"artifact_sha256sums.txt entries = {ctx['hash_entries']}",
        f"manifest.json entries = {ctx['manifest_entries']}",
        "",
    ]
    return NL.join(lines)


def _sensitivity_lines(sens) -> list:
    out = [
        f"scope = {sens['scope_description']}",
        f"pair folds in scope = {sens['pair_folds_in_scope']}",
        f"eligible pairs (success head) = "
        f"{sens['heads']['success']['eligible_pair_folds']}",
        f"median pairwise corrected-gap difference (success) = "
        f"{sens['heads']['success']['median_pairwise_residual_gap_difference']}",
        f"95% bootstrap CI (success) = "
        f"[{sens['heads']['success']['bootstrap']['ci_lower']}, "
        f"{sens['heads']['success']['bootstrap']['ci_upper']}]",
        f"eligible pairs (failure head) = "
        f"{sens['heads']['failure']['eligible_pair_folds']}",
        f"median pairwise corrected-gap difference (failure) = "
        f"{sens['heads']['failure']['median_pairwise_residual_gap_difference']}",
        f"95% bootstrap CI (failure) = "
        f"[{sens['heads']['failure']['bootstrap']['ci_lower']}, "
        f"{sens['heads']['failure']['bootstrap']['ci_upper']}]",
        f"SAME_PREDICTOR_HETEROGENEITY = "
        f"{sens['SAME_PREDICTOR_HETEROGENEITY']}",
        f"persistent agent labels = {sens['persistence']['persistent_agent_labels']}",
        f"TARGET_SPECIFIC_PERSISTENCE = "
        f"{sens['persistence']['TARGET_SPECIFIC_PERSISTENCE']}",
        f"PHASE1A_RESULT (scope) = {sens['PHASE1A_RESULT']}",
    ]
    return out


DEVIATIONS = [
    "pi_train and pi_target are measured on COMMON_TASK_SUPPORT_AB (the "
    "primary support required by section 9); full-universe priors are not "
    "used for the gate.",
    "The task-cluster bootstrap draws independently seeded replicates per "
    "declared scope and head (seed = 42*100 + scope offset + head) so that "
    "every reported CI is reproducible from bootstrap_seed alone.",
    "GEMINI-3-PRO has target success prior 1.0 on some folds, so "
    "PRIOR_DEGENERATE = TRUE is flagged and the boundary behaviour is "
    "preserved exactly (no smoothing); those occurrences are excluded from "
    "target persistence exactly as section 17 prescribes.",
    "The mirror-label sensitivity removes gpt-5.2-high from the pair scope "
    "without retraining, because section 5 forbids retuning and section 22 "
    "asks only to recalculate the principal gate metrics on a declared "
    "secondary scope.",
    "Secondary correlation (section 20) uses the corrected calibration gap, "
    "matching the Phase 0E convention; the raw decision error rate is kept "
    "as the separate *_all_task_* descriptive columns.",
    "artifact_sha256sums.txt excludes itself; manifest.json excludes itself "
    "and artifact_sha256sums.txt.",
]


def scope_payload(pairs, all_ids, excluded_label, tag, scope_description,
                  boot, pw) -> dict:
    ids = [pid for pid, a, b in pairs
           if excluded_label is None or excluded_label not in (a, b)]
    heads = {h: head_gate(boot, ids, h, seed_for(tag, h)) for h in HEAD_ORDER}
    same = bool(any(heads[h]["HEAD_SAME_PREDICTOR_HETEROGENEITY"]
                    for h in HEAD_ORDER))
    rows = persistence_rows(pw, ids)
    gate = persistence_gate(rows)
    result = ("STRONG_PASS" if (same and gate["TARGET_SPECIFIC_PERSISTENCE"])
              else "PARTIAL_PASS"
              if (same or gate["TARGET_SPECIFIC_PERSISTENCE"]) else "NO_PASS")
    excluded = sorted({pid for pid, a, b in pairs
                       if excluded_label is not None and excluded_label in (a, b)})
    return {
        "scope": tag,
        "scope_description": scope_description,
        "excluded_label": excluded_label,
        "pair_folds_in_scope": len(ids),
        "pair_folds_excluded": excluded,
        "heads": heads,
        "SAME_PREDICTOR_HETEROGENEITY": same,
        "persistence_rows": rows,
        "persistence": gate,
        "PHASE1A_RESULT": result,
    }


def main() -> int:
    ensure_dirs(ANA, FOLDS, PRED_PAIR, PRED_DEC)
    inputs = read_json(OUT / "input_hashes.json")
    dec, load_checks = load_decisions()
    traj = pd.read_csv(OUT_0B / "analysis" / "trajectory_outcomes.csv",
                       usecols=["traj_id", "model_id", "instance_id", "resolved"])
    traj = traj.drop_duplicates("traj_id")
    manifest = pd.read_csv(FOLDS / "pair_fold_manifest.csv")
    pairs = [(str(r.pair_id), str(r.agent_A), str(r.agent_B))
             for r in manifest.itertuples()]
    all_ids = [p[0] for p in pairs]
    print(f"analyzing {len(pairs)} pair folds, {len(dec)} decision rows",
          flush=True)

    pw, boot = compute_pairwise(pairs, dec, traj, "COMMON_TASK_SUPPORT_AB")
    pw.to_csv(ANA / "pairwise_metrics.csv", index=False, encoding="utf-8")
    print(f"pairwise metrics rows = {len(pw)}", flush=True)

    primary = scope_payload(pairs, all_ids, None, "primary",
                            "primary analysis: every frozen pair fold",
                            boot, pw)
    gemini = scope_payload(pairs, all_ids, GEMINI_PRO, "exclude_gemini3pro",
                           f"declared sensitivity: no pair fold involving "
                           f"{GEMINI_PRO}", boot, pw)
    mirror = scope_payload(pairs, all_ids, MIRROR_DROP, "drop_gpt_5_2_high",
                           f"declared sensitivity: remove {MIRROR_DROP} from "
                           f"the pair scope, retain {MIRROR_KEEP}", boot, pw)
    secondary = secondary_correlations(pw, all_ids)
    priors = agent_prior_summary(pw, all_ids)
    pers_rows = primary["persistence_rows"]
    pd.DataFrame(pers_rows).to_csv(ANA / "target_persistence.csv",
                                   index=False, encoding="utf-8")
    pd.DataFrame(priors).to_csv(ANA / "agent_prior_summary.csv", index=False,
                                encoding="utf-8")

    write_json(ANA / "bootstrap_results.json", as_builtin({
        "replicates": REPLICATES,
        "bootstrap_unit": "instance_id (common task support)",
        "resampling": "common task IDs drawn with replacement per replicate",
        "summary_statistic":
            "median pairwise corrected-gap difference over eligible pair folds",
        "eligibility_recomputed_inside_each_replicate": True,
        "lightgbm_retrained_inside_bootstrap": False,
        "primary": primary["heads"],
        "exclude_gemini3pro": gemini["heads"],
        "drop_gpt_5_2_high": mirror["heads"],
    }))
    write_json(ANA / "primary_gate.json", as_builtin({
        "predictor": PREDICTOR,
        "policy": {"name": POLICY_NAME, "policy_mode": "dual",
                   "success_thr": 0.95, "failure_thr": 0.95,
                   "min_step": MIN_STEP, "consecutive": 1},
        "decision_table_checks": load_checks,
        "thresholds": {"PAIR_MIN_DECISIONS": PAIR_MIN_DECISIONS,
                       "PAIR_MIN_ELIGIBLE_FOLDS": PAIR_MIN_ELIGIBLE_FOLDS,
                       "PAIR_MEDIAN_DIFF_MIN": PAIR_MEDIAN_DIFF_MIN,
                       "PAIR_CI_LOWER_MIN": PAIR_CI_LOWER_MIN,
                       "PERSIST_MIN_OCCURRENCES": PERSIST_MIN_OCCURRENCES,
                       "PERSIST_MEDIAN_ABS_GAP_MIN":
                           PERSIST_MEDIAN_ABS_GAP_MIN,
                       "PERSIST_SAME_SIGN_FRACTION":
                           PERSIST_SAME_SIGN_FRACTION},
        "heads": primary["heads"],
        "SAME_PREDICTOR_HETEROGENEITY":
            primary["SAME_PREDICTOR_HETEROGENEITY"],
        "target_persistence": primary["persistence"],
        "PHASE1A_RESULT": primary["PHASE1A_RESULT"],
        "api_calls": 0, "llm_calls": 0, "new_trajectories": 0,
        "threshold_sweep": 0, "method_modification": 0,
    }))
    write_json(ANA / "gemini3pro_sensitivity.json", as_builtin(gemini))
    write_json(ANA / "mirror_label_sensitivity.json", as_builtin(mirror))
    write_json(ANA / "secondary_correlations.json", as_builtin({
        "note": "descriptive only; no gate is attached to section 20",
        "correlations": secondary,
        "agent_target_success_prior": priors,
    }))

    fold_rows = fold_metadata_rows()
    pd.DataFrame(fold_rows).to_csv(FOLDS / "fold_training_metadata.csv",
                                   index=False, encoding="utf-8")
    dr = driver_results()
    results = dr.get("results", [])
    failures = [f"{r['pair_id']} rc={r.get('returncode')} "
                f"{r.get('stderr_tail', '')[-300:]}" for r in results
                if r.get("status") == "FOLD_FAIL"]
    execution = {
        "pair_folds_planned": len(pairs),
        "pair_folds_completed": int(len(fold_rows)),
        "pair_folds_failed": len(failures),
        "fold_failures": failures,
        "training_wall_clock_seconds_total": round(
            float(sum(r["wall_clock_seconds"] or 0.0 for r in fold_rows)), 1),
        "driver_wall_clock_seconds": dr.get("total_driver_wall_clock_seconds"),
        "driver_workers": dr.get("workers"),
        "driver_threads_per_fold": dr.get("fold_threads"),
        "api_calls": 0, "llm_calls": 0, "new_trajectories": 0,
    }
    ctx = {
        "execution": execution,
        "inputs": inputs,
        "head_gates": primary["heads"],
        "gate": {"SAME_PREDICTOR_HETEROGENEITY":
                     primary["SAME_PREDICTOR_HETEROGENEITY"],
                 "PHASE1A_RESULT": primary["PHASE1A_RESULT"]},
        "persistence": {**primary["persistence"], "rows": pers_rows},
        "gemini": gemini,
        "mirror": mirror,
        "secondary": secondary,
        "deviations": DEVIATIONS,
        "hash_entries": len([p for p in rel_files(OUT)
                             if p.name != "artifact_sha256sums.txt"]),
        "manifest_entries": len([p for p in rel_files(OUT)
                                 if p.name not in ("manifest.json",
                                                   "artifact_sha256sums.txt")]),
    }
    (OUT / "PROTOCOL.md").write_text(render_protocol(ctx), "utf-8", newline=NL)
    (OUT / "PHASE1A_REPORT.md").write_text(render_report(ctx), "utf-8",
                                           newline=NL)
    integrity = {
        "phase": "EARLYEVAL_PHASE1A",
        "name": "SAME_PREDICTOR_MULTI_AGENT_TRANSFER",
        "upstream_manifests": upstream_manifest_check(),
        "earlyeval_commit_expected": inputs["earlyeval_commit_expected"],
        "earlyeval_commit_observed": inputs["earlyeval_commit_observed"],
        "earlyeval_clone_clean": inputs["earlyeval_clone_clean"],
        "all_phase0b_input_hashes_match":
            inputs["ALL_PHASE0B_OUTPUT_HASHES_MATCH"],
        "all_phase0b_work_files_present":
            inputs["ALL_PHASE0B_WORK_FILES_PRESENT"],
        "execution": execution,
        "decision_table_checks": load_checks,
        "api_calls": 0, "llm_calls": 0, "new_trajectories": 0,
        "threshold_sweep": 0, "method_modification": 0,
        "artifact_sha256sums_entries": ctx["hash_entries"],
        "artifact_sha256sums_self_excludes": ["artifact_sha256sums.txt"],
        "manifest_self_excludes": ["manifest.json", "artifact_sha256sums.txt"],
        "artifact_sha256sums_self_consistent": True,
        "artifact_sha256sums_consistency_note":
            "hashes are recomputed from disk immediately before the digest "
            "file is written; the digest file is the only excluded path",
    }
    write_json(OUT / "integrity_report.json", as_builtin(integrity))
    files_meta = []
    from common1a import sha256_file
    for p in rel_files(OUT):
        rel = p.relative_to(OUT).as_posix()
        if rel in ("manifest.json", "artifact_sha256sums.txt"):
            continue
        files_meta.append({"path": rel, "bytes": int(p.stat().st_size),
                           "sha256": sha256_file(p)})
    write_json(OUT / "manifest.json", as_builtin({
        "phase": "EARLYEVAL_PHASE1A",
        "name": "SAME_PREDICTOR_MULTI_AGENT_TRANSFER",
        "predictor": PREDICTOR,
        "policy": {"name": POLICY_NAME, "policy_mode": "dual",
                   "success_thr": 0.95, "failure_thr": 0.95,
                   "min_step": MIN_STEP, "consecutive": 1},
        "label_universe": MODELS,
        "pair_folds_planned": len(pairs),
        "pair_folds_completed": execution["pair_folds_completed"],
        "pair_folds_failed": execution["pair_folds_failed"],
        "primary_scope": "all 45 pair folds",
        "sensitivity_scopes": [gemini["scope_description"],
                               mirror["scope_description"]],
        "gates": {"SAME_PREDICTOR_HETEROGENEITY":
                      primary["SAME_PREDICTOR_HETEROGENEITY"],
                  "TARGET_SPECIFIC_PERSISTENCE":
                      primary["persistence"]["TARGET_SPECIFIC_PERSISTENCE"],
                  "PHASE1A_RESULT": primary["PHASE1A_RESULT"]},
        "api_calls": 0, "llm_calls": 0, "new_trajectories": 0,
        "files": files_meta,
    }))
    n_lines = write_artifact_hashes()
    print(f"artifact_sha256sums.txt entries = {n_lines}", flush=True)
    print(f"{NL}PHASE1A_RESULT = {primary['PHASE1A_RESULT']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
