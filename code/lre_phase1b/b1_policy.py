# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1B step 1 - re-score the frozen per-prefix predictions.

The only operation performed is applying the frozen stopping policy at four
symmetric thresholds to the Phase 1A held-out prefix probabilities. Threshold
0.950 is verified against the frozen Phase 1A decision tables.
"""
from __future__ import annotations

import pickle
import sys

from b1_common import (ANA, CONSECUTIVE, MIN_DECISIONS, MIN_STEP, MODELS, NL,
                       OUT, OUT_0B, OUT_1A, OUT_1AD, PHASE0B_INPUT,
                       PHASE1A_INPUTS, PHASE1AD_INPUTS, POLICY_MODE, POLICY_NAME,
                       PREDICTOR, REPRODUCTION_THRESHOLD, SCORE_MODE,
                       THRESHOLDS, WORK, as_builtin, ensure_dirs, read_json,
                       read_manifest, set_vendor_env, sha256_file, write_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

set_vendor_env()

from earlyeval.core.contracts import PolicySpec  # noqa: E402
from earlyeval.policies.safe_stop import apply_policy  # noqa: E402

DEC_DIR = OUT_1A / "predictions" / "pair_policy_decisions"
PAIR_DIR = OUT_1A / "predictions" / "per_pair_target_predictions"


def verify_inputs() -> dict:
    """Section 1: verify every consumed hash against its original manifest."""
    checks, ok = {}, True
    for root, rel_items, label in (
            (OUT_1A, PHASE1A_INPUTS, "phase1a"),
            (OUT_1AD, PHASE1AD_INPUTS, "phase1a_d")):
        rec = read_manifest(root / "artifact_sha256sums.txt")
        for rel in rel_items:
            p = root / rel
            if p.is_dir():
                files = sorted(p.glob("*"))
                bad = [f.name for f in files
                       if rec.get(f.relative_to(root).as_posix())
                       != sha256_file(f)]
                entry = {"kind": "directory", "files": len(files),
                         "hash_mismatches": len(bad),
                         "mismatch_examples": bad[:5],
                         "match": bool(files and not bad)}
            else:
                obs = sha256_file(p) if p.exists() else None
                exp = rec.get(rel)
                entry = {"kind": "file", "sha256": obs, "manifest_sha256": exp,
                         "match": bool(obs is not None and obs == exp)}
            ok = ok and entry["match"]
            checks[f"{label}:{rel}"] = entry
    p0b = OUT_0B / PHASE0B_INPUT
    exp0b = read_manifest(OUT_0B / "artifact_sha256sums.txt").get(PHASE0B_INPUT)
    obs0b = sha256_file(p0b)
    ok = ok and obs0b == exp0b
    checks[f"phase0b:{PHASE0B_INPUT}"] = {
        "kind": "phase0b_file", "sha256": obs0b, "manifest_sha256": exp0b,
        "match": bool(obs0b == exp0b)}
    return {"inputs": checks, "ALL_INPUT_HASHES_MATCH": bool(ok),
            "phase1a_manifest": str(OUT_1A / "artifact_sha256sums.txt"),
            "phase1a_d_manifest": str(OUT_1AD / "artifact_sha256sums.txt"),
            "phase0b_manifest": str(OUT_0B / "artifact_sha256sums.txt")}


def policy_for(thr: float) -> PolicySpec:
    return PolicySpec(name=POLICY_NAME, predictor=PREDICTOR,
                      score_mode=SCORE_MODE, policy_mode=POLICY_MODE,
                      success_thr=float(thr), failure_thr=float(thr),
                      min_step=MIN_STEP, consecutive=CONSECUTIVE)


def score_pair(pair_id: str, agent_a: str, agent_b: str, thr: float):
    """Apply one symmetric policy to the frozen held-out prefixes of a pair."""
    frames = [pd.read_parquet(PAIR_DIR / f"{pair_id}.{agent}.parquet")
              for agent in (agent_a, agent_b)]
    frame = pd.concat(frames, ignore_index=True)
    decisions, summary, per_agent = apply_policy(frame, policy_for(thr))
    decisions["pair_id"] = pair_id
    per_agent["pair_id"] = pair_id
    return decisions, summary, per_agent


def main() -> int:
    import time

    t0 = time.time()
    ensure_dirs(OUT, ANA, WORK)
    inputs = verify_inputs()
    print(f"inputs match = {inputs['ALL_INPUT_HASHES_MATCH']}", flush=True)
    write_json(OUT / "input_hashes.json", as_builtin({
        **inputs,
        "predictor_retraining": 0, "new_lightgbm_training": 0,
        "new_trajectories": 0, "new_datasets": 0, "api_calls": 0,
        "llm_calls": 0,
    }))
    man = pd.read_csv(OUT_1A / "folds" / "pair_fold_manifest.csv")
    pairs = [(str(r.pair_id), str(r.agent_A), str(r.agent_B))
             for r in man.itertuples()]

    by_threshold, summaries, per_agent_frames = {}, {}, []
    reproduction = {}
    for thr in THRESHOLDS:
        frames, sum_frames = [], []
        for pair_id, agent_a, agent_b in pairs:
            d, s, pa = score_pair(pair_id, agent_a, agent_b, thr)
            frames.append(d)
            sum_frames.append(s.assign(pair_id=pair_id))
            per_agent_frames.append(pa)
        dec = pd.concat(frames, ignore_index=True)
        by_threshold[thr] = dec
        summaries[thr] = pd.concat(sum_frames, ignore_index=True)
        print(f"threshold {thr}: {len(dec)} rows, "
              f"{int(dec['decided'].astype(bool).sum())} decided", flush=True)

    # Section 9 reproduction check against the frozen Phase 1A decisions.
    frozen = []
    for p in sorted(DEC_DIR.glob("pair-????.csv")):
        d = pd.read_csv(p)
        d["pair_id"] = p.stem
        frozen.append(d)
    frozen = pd.concat(frozen, ignore_index=True)
    new = by_threshold[REPRODUCTION_THRESHOLD]
    merged = frozen.merge(new[["traj_id", "pair_id", "decided", "decision",
                               "decision_step", "decision_score"]],
                          on=["traj_id", "pair_id"], how="outer",
                          suffixes=("_frozen", "_new"), indicator=True)
    only_frozen = int((merged["_merge"] == "left_only").sum())
    only_new = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    decided_agree = int((both["decided_frozen"].astype(bool)
                         == both["decided_new"].astype(bool)).sum())
    decision_agree = int((both["decision_frozen"].astype(str)
                          == both["decision_new"].astype(str)).sum())
    step_agree = int((both["decision_step_frozen"].astype(int)
                      == both["decision_step_new"].astype(int)).sum())
    diff = (both["decision_score_frozen"].fillna(-1.0)
            - both["decision_score_new"].fillna(-1.0)).abs()
    max_diff = float(diff.max()) if len(diff) else None
    reproduction = {
        "threshold": REPRODUCTION_THRESHOLD,
        "frozen_rows": int(len(frozen)), "recomputed_rows": int(len(new)),
        "rows_only_frozen": only_frozen, "rows_only_recomputed": only_new,
        "decided_agreement": decided_agree,
        "decision_agreement": decision_agree,
        "decision_step_agreement": step_agree,
        "rows_compared": int(len(both)),
        "max_abs_decision_score_difference": max_diff,
        "score_tolerance": 1e-9,
        "EXACT_MATCH": bool(only_frozen == 0 and only_new == 0
                            and decided_agree == len(both)
                            and decision_agree == len(both)
                            and step_agree == len(both)
                            and max_diff is not None and max_diff <= 1e-9),
    }
    print(f"0.950 reproduction vs frozen Phase 1A decisions: "
          f"{reproduction['EXACT_MATCH']} (rows={len(both)}, "
          f"max_score_diff={max_diff})", flush=True)

    traj = pd.read_csv(OUT_0B / PHASE0B_INPUT,
                       usecols=["traj_id", "instance_id", "resolved"]
                       ).drop_duplicates("traj_id")
    payload = {}
    for thr, dec in by_threshold.items():
        d = dec.merge(traj[["traj_id", "instance_id", "resolved"]], on="traj_id",
                      how="left", validate="many_to_one")
        d["instance_id"] = d["instance_id"].astype(str)
        d["agent_model"] = d["agent_model"].astype(str)
        d["decided"] = d["decided"].astype(bool)
        d["decision"] = d["decision"].astype(str)
        payload[thr] = d[["pair_id", "traj_id", "agent_model", "instance_id",
                          "label", "n_steps", "decided", "decision",
                          "decision_step", "decision_score", "saved_steps",
                          "resolved"]]
    with open(WORK / "threshold_decisions.pkl", "wb") as fh:
        pickle.dump({"decisions": payload,
                     "summaries": {k: v.to_dict(orient="records")
                                   for k, v in summaries.items()},
                     "per_agent": [f.to_dict(orient="records")
                                   for f in per_agent_frames],
                     "pair_manifest": man.to_dict(orient="records"),
                     "pairs": pairs, "inputs": inputs,
                     "reproduction": reproduction}, fh)
    write_json(ANA / "reproduction_check.json", as_builtin({
        **reproduction,
        "rescore_wall_clock_seconds": round(time.time() - t0, 1),
        "source": "frozen Phase 1A predictions/pair_policy_decisions",
        "pipeline": "apply_policy on frozen per_pair_target_predictions",
        "policy": {"name": POLICY_NAME, "policy_mode": POLICY_MODE,
                   "score_mode": SCORE_MODE, "min_step": MIN_STEP,
                   "consecutive": CONSECUTIVE,
                   "success_thr": REPRODUCTION_THRESHOLD,
                   "failure_thr": REPRODUCTION_THRESHOLD},
        "predictor_retrained": False, "calibrator_refit": False,
    }))
    print(f"{NL}EXACT_0.950_REPRODUCTION = {reproduction['EXACT_MATCH']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
