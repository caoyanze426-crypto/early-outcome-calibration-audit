# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1B - threshold-specific occurrence records and evaluation.

Every statistic re-applies the frozen oracle prior correction to frozen Phase 1A
prefix probabilities under one of the four declared stopping policies. Nothing
is trained, rebuilt or refitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from b1_common import (HEAD_SIGN, MIN_DECISIONS, MODELS, REPLICATES, log_odds,
                       logit, percentile_ci, sigmoid)


def build_records(dec: pd.DataFrame, traj: pd.DataFrame, pairs) -> dict:
    """Per (pair, agent): frozen task-level arrays plus head decision rows."""
    traj = traj.copy()
    traj["model_id"] = traj["model_id"].astype(str)
    traj["instance_id"] = traj["instance_id"].astype(str)
    records = {}
    for pair_id, agent_a, agent_b in pairs:
        part = dec[dec["pair_id"] == pair_id]
        tasks_a = set(part[part["agent_model"] == agent_a]["instance_id"])
        tasks_b = set(part[part["agent_model"] == agent_b]["instance_id"])
        common = sorted(tasks_a & tasks_b)
        pos = {t: i for i, t in enumerate(common)}
        train_agents = [m for m in MODELS if m not in (agent_a, agent_b)]
        tsub = traj[traj["instance_id"].isin(pos)]
        for agent in (agent_a, agent_b):
            partner = agent_b if agent == agent_a else agent_a
            sub = tsub[tsub["model_id"] == agent]
            n_t, s_t = np.zeros(len(common)), np.zeros(len(common))
            ti = sub["instance_id"].map(pos).to_numpy(dtype=np.int64)
            np.add.at(n_t, ti, 1.0)
            np.add.at(s_t, ti, sub["resolved"].to_numpy(dtype=np.float64))
            tr = tsub[tsub["model_id"].isin(train_agents)]
            n_tr, s_tr = np.zeros(len(common)), np.zeros(len(common))
            ti_tr = tr["instance_id"].map(pos).to_numpy(dtype=np.int64)
            np.add.at(n_tr, ti_tr, 1.0)
            np.add.at(s_tr, ti_tr, tr["resolved"].to_numpy(dtype=np.float64))
            heads = {}
            ag = part[(part["agent_model"] == agent)
                      & (part["instance_id"].isin(pos))]
            for head in ("success", "failure"):
                fr = ag[ag["decision"] == head]
                row_task = fr["instance_id"].map(pos).to_numpy(dtype=np.int64)
                resolved = fr["resolved"].to_numpy(dtype=np.int64)
                correct = (resolved == 1 if head == "success"
                           else resolved == 0).astype(np.float64)
                heads[head] = {
                    "row_task": row_task,
                    "score": fr["decision_score"].to_numpy(dtype=np.float64),
                    "correct": correct,
                    "saved_steps": fr["saved_steps"].to_numpy(dtype=np.float64),
                }
            records[(pair_id, agent)] = {
                "pair_id": pair_id, "agent_model": agent, "partner": partner,
                "tasks": common, "target_n": n_t, "target_s": s_t,
                "train_n": n_tr, "train_s": s_tr, "heads": heads,
            }
    return records


def _finish(rec, head, mult, dec_n, precision, pi_target, pi_train, log_r,
            degenerate, n, s, n_tr, s_tr, mean_saved) -> dict:
    h = rec["heads"][head]
    w = mult[h["row_task"]]
    corrected = sigmoid(logit(h["score"]) + HEAD_SIGN[head] * log_r)
    corrected_mean = float((w * corrected).sum()) / dec_n
    gap = corrected_mean - precision
    return {
        "n_decisions": int(dec_n),
        "empirical_precision": precision,
        "mean_stop_score": float((w * h["score"]).sum()) / dec_n,
        "corrected_mean_score": corrected_mean,
        "signed_corrected_gap": gap,
        "absolute_corrected_gap": abs(gap),
        "mean_saved_steps": mean_saved,
        "pi_target": pi_target, "pi_train": pi_train, "log_odds_ratio": log_r,
        "prior_degenerate": degenerate,
        "target_trajectories": int(n), "target_successes": int(s),
        "train_trajectories": int(n_tr), "train_successes": int(s_tr),
    }


def evaluate(rec, head, mult, min_decisions, frozen_log_r=None) -> dict | None:
    """One occurrence under a task weighting; optional frozen prior coefficient."""
    n = float((mult * rec["target_n"]).sum())
    s = float((mult * rec["target_s"]).sum())
    n_tr = float((mult * rec["train_n"]).sum())
    s_tr = float((mult * rec["train_s"]).sum())
    h = rec["heads"][head]
    w = mult[h["row_task"]]
    dec_n = float(w.sum())
    if dec_n < min_decisions:
        return None
    if h["score"].size:
        mean_saved = float((w * h["saved_steps"]).sum()) / dec_n
    else:
        mean_saved = None
    precision = float((w * h["correct"]).sum()) / dec_n
    if frozen_log_r is not None:
        return _finish(rec, head, mult, dec_n, precision,
                       s / n if n else None, s_tr / n_tr if n_tr else None,
                       float(frozen_log_r), False, n, s, n_tr, s_tr, mean_saved)
    if n <= 0.0 or n_tr <= 0.0:
        return None
    pi_target, pi_train = s / n, s_tr / n_tr
    degenerate = bool(pi_target in (0.0, 1.0) or pi_train in (0.0, 1.0))
    return _finish(rec, head, mult, dec_n, precision, pi_target, pi_train,
                   log_odds(pi_target) - log_odds(pi_train), degenerate, n, s,
                   n_tr, s_tr, mean_saved)


def indicator(n_tasks: int) -> np.ndarray:
    return np.ones(n_tasks, dtype=np.float64)


def bootstrap(folds, head: str, seed: int, min_decisions: int,
              frozen_log_r: dict | None = None) -> dict:
    """Section 8: task-cluster bootstrap of the target-level median signed gap."""
    rng = np.random.default_rng(int(seed))
    med_signed, med_abs = [], []
    degenerate_replicates = 0
    for _ in range(REPLICATES):
        signed, degenerate = [], False
        for pid, rec in folds:
            n_tasks = len(rec["tasks"])
            idx = rng.integers(0, n_tasks, n_tasks)
            mult = np.bincount(idx, minlength=n_tasks).astype(np.float64)
            flr = None if frozen_log_r is None else frozen_log_r[pid]
            st = evaluate(rec, head, mult, min_decisions, flr)
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
        "replicates": REPLICATES, "seed": int(seed),
        "cluster_unit": "instance_id",
        "replicates_with_any_eligible_fold": len(med_signed),
        "degenerate_prior_replicates": degenerate_replicates,
        "median_signed_gap": percentile_ci(med_signed),
        "median_abs_gap": percentile_ci(med_abs),
        "coefficient": ("recomputed from the resampled tasks"
                        if frozen_log_r is None
                        else "held at the baseline occurrence log-odds ratio"),
        "primary_statistic": frozen_log_r is None,
    }


def occurrence_table(dec: pd.DataFrame, records: dict, pairs, targets,
                     threshold: float, threshold_label: str,
                     universe: dict | None = None) -> pd.DataFrame:
    """Section 6/7 inputs: every target occurrence at one threshold.

    `universe` restricts each target to its frozen Phase 1A-D occurrence set,
    which is required by the section-9 reproduction check; the section-6
    eligibility rule is then applied per threshold inside that universe.
    """
    rows = []
    for t in targets:
        agent, head = t["agent_model"], t["head"]
        allowed = None if universe is None else set(universe[t["scope"]])
        for pair_id, agent_a, agent_b in pairs:
            if allowed is not None and pair_id not in allowed:
                continue
            rec = records.get((pair_id, agent))
            if rec is None:
                continue
            st = evaluate(rec, head, indicator(len(rec["tasks"])), 0)
            side = "A" if agent_a == agent else "B"
            eligible = bool(st is not None and st["n_decisions"] >= MIN_DECISIONS
                            and not st["prior_degenerate"])
            rows.append({
                "target_scope": t["scope"], "target_agent": agent, "head": head,
                "threshold": threshold, "threshold_label": threshold_label,
                "pair_id": pair_id, "partner_agent": rec["partner"],
                "side_in_pair": side, "common_tasks": len(rec["tasks"]),
                "eligible_occurrence": eligible,
                **{k: (None if st is None else st[k]) for k in (
                    "n_decisions", "empirical_precision", "mean_stop_score",
                    "corrected_mean_score", "signed_corrected_gap",
                    "absolute_corrected_gap", "mean_saved_steps", "pi_target",
                    "pi_train", "log_odds_ratio", "prior_degenerate",
                    "target_trajectories", "target_successes",
                    "train_trajectories", "train_successes")},
            })
    return pd.DataFrame(rows)
