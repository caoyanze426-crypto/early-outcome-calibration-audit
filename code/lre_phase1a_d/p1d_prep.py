# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A_D step 1 - input freeze, occurrence universe, task halves.

Nothing is regenerated: the Phase 1A decision tables and the Phase 0B trajectory
outcomes are read as frozen inputs. Every value computed here is cross-checked
against the Phase 1A artifacts before any Phase 1A_D statistic is produced.
"""
from __future__ import annotations

import pickle
import sys

from p1d_common import (ANA, BOOTSTRAP_SEED, HALF_MIN_DECISIONS, HEAD_ORDER,
                        HEAD_SIGN, MODELS, NL, OUT, OUT_0B, OUT_1A,
                        PAIR_MIN_DECISIONS, PHASE0B_INPUT, PHASE1A_INPUTS,
                        TARGETS, WORK, as_builtin, ensure_dirs, log_odds, read_json,
                        read_manifest, rel_files, sha256_file, sha256_text,
                        write_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEC_DIR = OUT_1A / "predictions" / "pair_policy_decisions"
PAIR_DIR = OUT_1A / "predictions" / "per_pair_target_predictions"


def verify_inputs() -> dict:
    """Section 1: every Phase 1A input hash must still match."""
    rec = read_manifest(OUT_1A / "artifact_sha256sums.txt")
    checks, ok = {}, True
    for rel in PHASE1A_INPUTS:
        p = OUT_1A / rel
        if p.is_dir():
            files = sorted(p.glob("*"))
            missing = [f.name for f in files
                       if rec.get(f.relative_to(OUT_1A).as_posix())
                       != sha256_file(f)]
            check = {"kind": "directory", "files": len(files),
                     "hash_mismatches": len(missing),
                     "mismatch_examples": missing[:5],
                     "match": bool(files and not missing)}
        else:
            obs = sha256_file(p) if p.exists() else None
            exp = rec.get(rel)
            check = {"kind": "file", "sha256": obs, "manifest_sha256": exp,
                     "match": bool(obs is not None and obs == exp)}
        ok = ok and check["match"]
        checks[rel] = check
    p0b = OUT_0B / PHASE0B_INPUT
    exp0b = read_manifest(OUT_0B / "artifact_sha256sums.txt").get(PHASE0B_INPUT)
    obs0b = sha256_file(p0b)
    ok = ok and obs0b == exp0b
    checks[PHASE0B_INPUT] = {"kind": "phase0b_file", "sha256": obs0b,
                             "manifest_sha256": exp0b,
                             "match": bool(obs0b == exp0b)}
    payload = {
        "phase1a_manifest": str(OUT_1A / "artifact_sha256sums.txt"),
        "phase1a_manifest_entries": len(rec),
        "phase0b_manifest": str(OUT_0B / "artifact_sha256sums.txt"),
        "inputs": checks,
        "ALL_INPUT_HASHES_MATCH": bool(ok),
        "predictor_retraining": 0,
        "predictions_regenerated": 0,
        "new_pair_folds": 0,
        "api_calls": 0,
        "llm_calls": 0,
        "threshold_sweep": 0,
        "method_design": 0,
    }
    write_json(OUT / "input_hashes.json", as_builtin(payload))
    return payload


def load_decisions() -> tuple[pd.DataFrame, dict]:
    frames = []
    for p in sorted(DEC_DIR.glob("pair-????.csv")):
        d = pd.read_csv(p)
        d["pair_id"] = p.stem
        frames.append(d)
    dec = pd.concat(frames, ignore_index=True)
    traj = pd.read_csv(OUT_0B / PHASE0B_INPUT,
                       usecols=["traj_id", "model_id", "instance_id",
                                "resolved"]).drop_duplicates("traj_id")
    dec = dec.merge(traj[["traj_id", "instance_id", "resolved"]], on="traj_id",
                    how="left", validate="many_to_one")
    dec["instance_id"] = dec["instance_id"].astype(str)
    dec["agent_model"] = dec["agent_model"].astype(str)
    dec["decided"] = dec["decided"].astype(bool)
    dec["decision"] = dec["decision"].astype(str)
    checks = {
        "decision_rows": int(len(dec)),
        "instance_id_missing": int((dec["instance_id"] == "nan").sum()),
        "resolved_label_mismatch_vs_phase0b":
            int((dec["resolved"] != dec["label"]).sum()),
    }
    return dec, traj, checks


def task_assignment(task_ids) -> pd.DataFrame:
    """Section 6: deterministic halves from SHA256(instance_id), lowest bit."""
    rows = []
    for t in sorted(task_ids):
        digest = sha256_text(t)
        bit = int(digest[-1], 16) & 1
        rows.append({"instance_id": t, "sha256": digest, "lowest_bit": bit,
                     "task_half": "A" if bit == 0 else "B"})
    frame = pd.DataFrame(rows)
    return frame


def build_pair_records(dec: pd.DataFrame, traj: pd.DataFrame, pairs) -> dict:
    """Per (pair, target agent, head): frozen arrays for every later statistic."""
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
            n_t, s_t = np.zeros(len(common)), np.zeros(len(common))
            sub = tsub[tsub["model_id"] == agent]
            ti = sub["instance_id"].map(pos).to_numpy(dtype=np.int64)
            np.add.at(n_t, ti, 1.0)
            np.add.at(s_t, ti, sub["resolved"].to_numpy(dtype=np.float64))
            tr = tsub[tsub["model_id"].isin(train_agents)]
            ti_tr = tr["instance_id"].map(pos).to_numpy(dtype=np.int64)
            n_tr, s_tr = np.zeros(len(common)), np.zeros(len(common))
            np.add.at(n_tr, ti_tr, 1.0)
            np.add.at(s_tr, ti_tr, tr["resolved"].to_numpy(dtype=np.float64))
            heads = {}
            # Primary support: only trajectories whose instance_id is in
            # COMMON_TASK_SUPPORT_AB, exactly as Phase 1A section 9 required.
            ag = part[(part["agent_model"] == agent)
                      & (part["instance_id"].isin(pos))]
            for head in HEAD_ORDER:
                fr = ag[ag["decision"] == head]
                if not fr["instance_id"].isin(pos).all():
                    raise SystemExit("non-common instance outside the support")
                row_task = fr["instance_id"].map(pos).to_numpy(dtype=np.int64)
                score = fr["decision_score"].to_numpy(dtype=np.float64)
                resolved = fr["resolved"].to_numpy(dtype=np.int64)
                correct = (resolved == 1 if head == "success"
                           else resolved == 0).astype(np.float64)
                heads[head] = {"row_task": row_task, "score": score,
                               "correct": correct}
            records[(pair_id, agent)] = {
                "pair_id": pair_id, "agent_model": agent, "partner": partner,
                "tasks": common, "pos": pos,
                "target_n": n_t, "target_s": s_t,
                "train_n": n_tr, "train_s": s_tr,
                "heads": heads,
            }
    return records


def evaluate(rec, head, mult, min_decisions) -> dict | None:
    """One occurrence evaluated on a task weighting (0/1 indicator or multiset)."""
    n = float((mult * rec["target_n"]).sum())
    s = float((mult * rec["target_s"]).sum())
    n_tr = float((mult * rec["train_n"]).sum())
    s_tr = float((mult * rec["train_s"]).sum())
    h = rec["heads"][head]
    w = mult[h["row_task"]]
    dec_n = float(w.sum())
    if dec_n < min_decisions:
        return None
    precision = float((w * h["correct"]).sum()) / dec_n
    if n <= 0.0 or n_tr <= 0.0:
        return None
    pi_target, pi_train = s / n, s_tr / n_tr
    degenerate = bool(pi_target in (0.0, 1.0) or pi_train in (0.0, 1.0))
    log_r = log_odds(pi_target) - log_odds(pi_train)
    return _finish(rec, head, mult, dec_n, precision, pi_target, pi_train,
                   log_r, degenerate, n, s, n_tr, s_tr)


def evaluate_frozen(rec, head, mult, min_decisions, log_r) -> dict | None:
    """Same evaluation but with the prior coefficient held at a frozen value."""
    n = float((mult * rec["target_n"]).sum())
    s = float((mult * rec["target_s"]).sum())
    n_tr = float((mult * rec["train_n"]).sum())
    s_tr = float((mult * rec["train_s"]).sum())
    h = rec["heads"][head]
    w = mult[h["row_task"]]
    dec_n = float(w.sum())
    if dec_n < min_decisions:
        return None
    precision = float((w * h["correct"]).sum()) / dec_n
    return _finish(rec, head, mult, dec_n, precision, s / n if n else None,
                   s_tr / n_tr if n_tr else None, float(log_r), False, n, s,
                   n_tr, s_tr)


def _finish(rec, head, mult, dec_n, precision, pi_target, pi_train, log_r,
            degenerate, n, s, n_tr, s_tr) -> dict:
    h = rec["heads"][head]
    w = mult[h["row_task"]]
    corrected = sigmoid_scores(h["score"], head, log_r)
    corrected_mean = float((w * corrected).sum()) / dec_n
    return {
        "n_decisions": int(dec_n),
        "empirical_precision": precision,
        "corrected_mean_score": corrected_mean,
        "signed_corrected_gap": corrected_mean - precision,
        "absolute_corrected_gap": abs(corrected_mean - precision),
        "pi_target": pi_target, "pi_train": pi_train, "log_odds_ratio": log_r,
        "prior_degenerate": degenerate,
        "target_trajectories": int(n), "target_successes": int(s),
        "train_trajectories": int(n_tr), "train_successes": int(s_tr),
    }


def sigmoid_scores(scores, head, log_r):
    from p1d_common import sigmoid

    return sigmoid(logit_scores(scores) + HEAD_SIGN[head] * log_r)


def logit_scores(scores, eps: float = 1e-12):
    return np.log(np.clip(scores, eps, 1.0 - eps)
                  / (1.0 - np.clip(scores, eps, 1.0 - eps)))


def indicator(n_tasks: int) -> np.ndarray:
    return np.ones(n_tasks, dtype=np.float64)


def main() -> int:
    ensure_dirs(OUT, ANA, WORK)
    inputs = verify_inputs()
    dec, traj, checks = load_decisions()
    print(f"loaded {len(dec)} decision rows, checks={checks}", flush=True)
    man = pd.read_csv(OUT_1A / "folds" / "pair_fold_manifest.csv")
    pair_agents = {str(r.pair_id): (str(r.agent_A), str(r.agent_B))
                   for r in man.itertuples()}
    targets = {(t["agent_model"], t["head"]) for t in TARGETS}
    relevant = sorted({pid for pid, (a, b) in pair_agents.items()
                       if any(t[0] in (a, b) for t in targets)})
    records = build_pair_records(dec, traj,
                                 [(p, *pair_agents[p]) for p in relevant])
    print(f"built {len(records)} (pair, agent) records over {len(relevant)} "
          f"pair folds", flush=True)

    # Task half assignment is frozen here, before any target statistic.
    all_tasks = set()
    for rec in records.values():
        all_tasks.update(rec["tasks"])
    assign = task_assignment(all_tasks)
    assign.to_csv(ANA / "task_half_assignment.csv", index=False,
                  encoding="utf-8")
    half_counts = assign["task_half"].value_counts().to_dict()
    print(f"task halves: {half_counts} (assignment written)", flush=True)

    # Baseline occurrences on the full common support, cross-checked vs Phase 1A.
    pw = pd.read_csv(OUT_1A / "analysis" / "pairwise_metrics.csv")
    rows, mismatch = [], []
    for t in TARGETS:
        agent, head = t["agent_model"], t["head"]
        for pid in relevant:
            rec = records.get((pid, agent))
            if rec is None:
                continue
            side = "A" if pair_agents[pid][0] == agent else "B"
            st = evaluate(rec, head, indicator(len(rec["tasks"])), 0)
            ref = pw[(pw["pair_id"] == pid) & (pw["head"] == head)].iloc[0]
            for key, col in (("n_decisions", f"{side}_decisions"),
                             ("empirical_precision", f"{side}_precision"),
                             ("corrected_mean_score",
                              f"{side}_corrected_mean_score"),
                             ("signed_corrected_gap", f"{side}_corrected_gap"),
                             ("pi_target", f"{side}_target_success_prior"),
                             ("log_odds_ratio", f"{side}_log_odds_ratio")):
                a = st[key]
                b = float(ref[col])
                if not (abs(a - b) < 1e-9):
                    mismatch.append({"pair_id": pid, "agent": agent,
                                     "head": head, "field": key,
                                     "recomputed": a, "phase1a": b})
            eligible = bool(ref["PAIR_HEAD_ELIGIBLE"]) and not st[
                "prior_degenerate"]
            rows.append({
                "scope": t["scope"], "target_agent": agent, "head": head,
                "pair_id": pid, "partner_agent": rec["partner"],
                "side_in_pair": side,
                "common_tasks": len(rec["tasks"]),
                "pair_head_eligible_phase1a": bool(ref["PAIR_HEAD_ELIGIBLE"]),
                "target_prior_degenerate": st["prior_degenerate"],
                "eligible_occurrence": eligible,
                **{k: st[k] for k in ("n_decisions", "empirical_precision",
                                      "corrected_mean_score",
                                      "signed_corrected_gap",
                                      "absolute_corrected_gap", "pi_target",
                                      "pi_train", "log_odds_ratio",
                                      "target_trajectories",
                                      "target_successes",
                                      "train_trajectories",
                                      "train_successes")},
            })
    occ = pd.DataFrame(rows).sort_values(
        ["scope", "head", "target_agent", "pair_id"]).reset_index(drop=True)
    occ.to_csv(ANA / "target_occurrences.csv", index=False, encoding="utf-8")
    observed = {t["scope"]: int(occ[(occ["scope"] == t["scope"])
                                    & occ["eligible_occurrence"]].shape[0])
                for t in TARGETS}
    tp = pd.read_csv(OUT_1A / "analysis" / "target_persistence.csv")
    reference = {}
    for t in TARGETS:
        ref = tp[(tp["agent_model"] == t["agent_model"])
                 & (tp["head"] == t["head"])].iloc[0]
        reference[t["scope"]] = {
            "phase1a_eligible_occurrences": int(ref["eligible_occurrences"]),
            "phase1a_median_signed_gap": float(ref["median_signed_corrected_gap"])
            if pd.notna(ref["median_signed_corrected_gap"]) else None,
            "phase1a_median_abs_gap": float(ref["median_abs_corrected_gap"])
            if pd.notna(ref["median_abs_corrected_gap"]) else None,
            "phase1a_same_sign_fraction": float(ref["same_sign_fraction"])
            if pd.notna(ref["same_sign_fraction"]) else None,
        }
    for scope, ref in reference.items():
        sub = occ[(occ["scope"] == scope) & occ["eligible_occurrence"]]
        ref["recomputed_eligible_occurrences"] = int(len(sub))
        ref["recomputed_median_signed_gap"] = float(
            np.median(sub["signed_corrected_gap"])) if len(sub) else None
        ref["recomputed_median_abs_gap"] = float(
            np.median(sub["absolute_corrected_gap"])) if len(sub) else None
    print(f"occurrence counts (eligible): {observed}", flush=True)
    print(f"phase1a reference: {reference}", flush=True)
    print(f"recomputation mismatches: {len(mismatch)}", flush=True)
    with open(WORK / "bundle.pkl", "wb") as fh:
        pickle.dump({"records": records, "pair_agents": pair_agents,
                     "relevant_pairs": relevant,
                     "assignment": assign.to_dict(orient="records"),
                     "occurrences": occ.to_dict(orient="records"),
                     "mismatches": mismatch, "reference": reference,
                     "input_checks": checks}, fh)
    write_json(ANA / "occurrence_recompute_check.json", as_builtin({
        "decision_table_checks": checks,
        "mismatches_vs_phase1a_pairwise_metrics": mismatch,
        "mismatch_count": len(mismatch),
        "field_tolerance": 1e-9,
        "occurrence_counts": observed,
        "phase1a_reference": reference,
        "task_half_counts": {k: int(v) for k, v in half_counts.items()},
        "half_rule": "task_half = A if int(sha256(instance_id)[-1], 16) & 1 == 0 "
                     "else B (lowest bit of the digest)",
        "half_rule_frozen_before_target_statistics": True,
        "ALL_RECOMPUTED_VALUES_MATCH_PHASE1A": bool(len(mismatch) == 0),
    }))
    print(f"{NL}ALL_RECOMPUTED_VALUES_MATCH_PHASE1A = "
          f"{len(mismatch) == 0}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
