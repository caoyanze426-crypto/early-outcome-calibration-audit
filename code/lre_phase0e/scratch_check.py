# -*- coding: utf-8 -*-
"""Scratch sanity check for PHASE0E prior-shift decomposition (no artifacts)."""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = os.environ["PAPER2_ROOT"]
OUT_0B = WS + r"\outputs\late_reversal_early_eval_phase0b_signal_hunt"
OUT_0D = WS + r"\outputs\earlyeval_phase0d_cross_agent_calibration"
PRED = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SC = f"prob_cal_safe_success__{PRED}"
FC = f"prob_cal_safe_failure__{PRED}"

dec = pd.read_csv(OUT_0B + r"\predictions\trajectory_policy_decisions.csv")
traj = pd.read_csv(OUT_0B + r"\analysis\trajectory_outcomes.csv")
import pyarrow.parquet as pq
pp = pq.ParquetFile(OUT_0B + r"\predictions\heldout_prefix_predictions_all.parquet"
                    ).read(columns=["traj_id", "prefix_step_idx", SC, FC]).to_pandas()

d = dec[dec["decided"].astype(bool)][
    ["traj_id", "agent_model", "decision", "decision_step", "decision_score"]].copy()
m = d.merge(pp, left_on=["traj_id", "decision_step"],
            right_on=["traj_id", "prefix_step_idx"], how="left")
m = m.merge(traj[["traj_id", "resolved", "instance_id"]], on="traj_id", how="left")
print("decided", len(m), "score-missing", int(m[SC].isna().sum()),
      "fin-missing", int(m[FC].isna().sum()))
m["p"] = np.where(m["decision"].to_numpy() == "success",
                  m[SC].to_numpy(), m[FC].to_numpy())
m["correct"] = np.where(m["decision"].to_numpy() == "success",
                        (m["resolved"].to_numpy() == 1).astype(int),
                        (m["resolved"].to_numpy() == 0).astype(int))
print("score range", m["p"].min(), m["p"].max(), "exact 0/1:",
      int((m["p"].isin([0.0, 1.0])).sum()))

prev = pd.read_csv(OUT_0D + r"\analysis\prevalence_shift.csv")
cal0d = pd.read_csv(OUT_0D + r"\analysis\decision_score_calibration.csv")

EPS = 1e-12
m["z"] = np.log(np.clip(m["p"].to_numpy(), EPS, 1 - EPS) /
                (1 - np.clip(m["p"].to_numpy(), EPS, 1 - EPS)))

rows = []
for r in prev.itertuples(index=False):
    pt, ptr = r.test_success_rate, r.train_success_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = (np.log(pt / (1 - pt)) if pt not in (0.0, 1.0) else np.inf)
        logr_tr = np.log(ptr / (1 - ptr))
    lr = logr - logr_tr
    sub = m[m["agent_model"] == r.model_id]
    for head, sign in (("success", +1.0), ("failure", -1.0)):
        g = sub[sub["decision"] == head]
        n = len(g)
        if n == 0:
            continue
        raw_mean = float(g["p"].mean())
        zc = g["z"].to_numpy() + sign * lr
        corr = 1.0 / (1.0 + np.exp(-zc))
        corr_mean = float(np.mean(corr))
        prec = float(g["correct"].mean())
        rows.append({"model": r.model_id, "head": head, "n": n,
                     "precision": prec, "raw_mean": raw_mean, "corr_mean": corr_mean,
                     "raw_gap": raw_mean - prec, "corr_gap": corr_mean - prec,
                     "log_r": sign * lr})
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.round(4).to_string(index=False))

# identity check vs Phase 0D
chk = t.merge(cal0d, left_on=["model", "head"],
              right_on=["model_id", "head"], how="left")
print("max |raw_mean - 0D mean_decision_score| =",
      float(np.max(np.abs(chk["raw_mean"] - chk["mean_decision_score"]))))
print("max |precision - 0D empirical_precision| =",
      float(np.max(np.abs(chk["precision"] - chk["empirical_precision"]))))
for head in ("success", "failure"):
    s = t[t["head"] == head]
    print(head, "raw gap range", round(float(s["raw_gap"].max() - s["raw_gap"].min()), 4),
          "corr gap range", round(float(s["corr_gap"].max() - s["corr_gap"].min()), 4),
          "n |corr_gap|>=0.10", int((s["corr_gap"].abs() >= 0.10).sum()))
