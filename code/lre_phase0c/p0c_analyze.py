# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0C - signal decomposition and control analysis.

Reuses only frozen Phase 0B artifacts: the trajectory universe, the LOMO held-out
predictions, the 0.95/0.95 policy decisions and the deterministic post-stop
detectors. No retraining, no threshold re-run, no modification of Phase 0B.
Strictly offline: LLM calls = 0, API calls = 0.
"""
from __future__ import annotations

import re
import sys

from common0c import (ANA, DATASET_SNAPSHOT, INPUT_FILES, OUT, WORK_0B,
                      as_builtin, ensure_dirs, percentile_ci, read_json,
                      safe_ratio, sha256_file, write_json)

import numpy as np
import pandas as pd

BINS = [0.0, 0.25, 0.50, 0.75, 1.0000001]
BIN_LABELS = ["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
N_BINS = len(BIN_LABELS)
BOOT_REPLICATES = 2000
BOOT_SEED = 42
GEMINI_PRO = "gemini-3-pro"


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


def load_step_events(spec: dict) -> dict:
    """Recompute the frozen deterministic post-stop events from the step table."""
    import pyarrow.parquet as pq

    test_re = re.compile(spec["TEST_RE"])
    edit_re = re.compile(spec["EDIT_RE"])
    submit_marker = spec["SUBMIT_MARKER"]
    parts = sorted((WORK_0B / "step_table").glob("step_table.part-*.parquet"))
    cols = ["traj_id", "step_idx", "action_text", "tool_args_text",
            "action_subtypes", "test_pass_seen_this_step",
            "test_fail_seen_this_step", "traceback_seen_this_step",
            "tool_error_seen_this_step"]
    events: dict = {}
    files = 0
    rows = 0
    for p in parts:
        pf = pq.ParquetFile(p)
        tbl = pf.read(columns=cols)
        traj = tbl.column("traj_id").to_pylist()
        step = tbl.column("step_idx").to_pylist()
        act = tbl.column("action_text").to_pylist()
        args = tbl.column("tool_args_text").to_pylist()
        sub = tbl.column("action_subtypes").to_pylist()
        tpass = tbl.column("test_pass_seen_this_step").to_pylist()
        tfail = tbl.column("test_fail_seen_this_step").to_pylist()
        tback = tbl.column("traceback_seen_this_step").to_pylist()
        terr = tbl.column("tool_error_seen_this_step").to_pylist()
        for i in range(tbl.num_rows):
            a = act[i] or ""
            g = args[i] or ""
            combined = a + chr(10) + g
            subtypes = [str(x) for x in (sub[i] or [])]
            row = (
                int(step[i]),
                1 if edit_re.search(combined) else 0,
                1 if (("test" in subtypes) or test_re.search(a)) else 0,
                1 if tpass[i] else 0,
                1 if tfail[i] else 0,
                1 if (tback[i] or terr[i]) else 0,
                1 if (("submit" in subtypes) or (submit_marker in a)) else 0,
            )
            events.setdefault(str(traj[i]), []).append(row)
        del pf, tbl
        files += 1
        rows += len(traj)
    for key in events:
        events[key].sort(key=lambda r: r[0])
    return {"events": events, "step_table_files": files, "step_table_rows": rows}


NO_TAIL = {
    "POST_STOP_ACTIVITY": False, "POST_STOP_EDIT": False, "POST_STOP_TEST": False,
    "POST_STOP_TEST_PASS": False, "POST_STOP_TEST_FAIL": False,
    "POST_STOP_ERROR_OR_TRACEBACK": False, "POST_STOP_SUBMISSION": False,
    "POST_STOP_EDIT_COUNT": 0, "POST_STOP_TEST_COUNT": 0, "POST_STOP_STEPS": 0,
    "FIRST_TEST_PASS_AFTER_STOP": "NOT_OBSERVABLE",
}
EVENT_COLUMNS = [c for c in NO_TAIL if c != "FIRST_TEST_PASS_AFTER_STOP"]


def scan_post_stop(events, k: int) -> dict:
    tail = [r for r in events if r[0] > k]
    if not tail:
        return dict(NO_TAIL)
    pass_idx = [r[0] for r in tail if r[3]]
    return {
        "POST_STOP_ACTIVITY": True,
        "POST_STOP_EDIT": any(r[1] for r in tail),
        "POST_STOP_TEST": any(r[2] for r in tail),
        "POST_STOP_TEST_PASS": bool(pass_idx),
        "POST_STOP_TEST_FAIL": any(r[4] for r in tail),
        "POST_STOP_ERROR_OR_TRACEBACK": any(r[5] for r in tail),
        "POST_STOP_SUBMISSION": any(r[6] for r in tail),
        "POST_STOP_EDIT_COUNT": int(sum(r[1] for r in tail)),
        "POST_STOP_TEST_COUNT": int(sum(r[2] for r in tail)),
        "POST_STOP_STEPS": int(len(tail)),
        "FIRST_TEST_PASS_AFTER_STOP": "TRUE" if pass_idx else "FALSE",
    }


def detector_recheck(traj: pd.DataFrame, spec: dict) -> dict:
    """Re-run the frozen Phase 0B detectors from the frozen spec + step table."""
    loaded = load_step_events(spec)
    events = loaded["events"]
    decided = traj[traj["decided"].astype(bool)]
    mismatches = {c: [] for c in EVENT_COLUMNS}
    checked = 0
    worst = 0
    for r in decided.itertuples(index=False):
        tid = str(r.traj_id)
        recomputed = scan_post_stop(events.get(tid, []), int(r.decision_step))
        checked += 1
        for c in EVENT_COLUMNS:
            if bool(recomputed[c]) != bool(getattr(r, c)):
                mismatches[c].append(tid)
        worst = max(worst, int(recomputed["POST_STOP_STEPS"]))
    ff_sig = np.asarray(
        [bool(r.POST_STOP_EDIT) and (bool(r.POST_STOP_TEST_PASS)
                                     or bool(r.POST_STOP_SUBMISSION))
         for r in decided.itertuples(index=False)], dtype=bool)
    fs_sig = np.asarray(
        [bool(r.POST_STOP_TEST_FAIL) or bool(r.POST_STOP_ERROR_OR_TRACEBACK)
         for r in decided.itertuples(index=False)], dtype=bool)
    n_ff_mismatch = int((ff_sig
                         != decided["FF_LATE_RECOVERY_SIGNATURE"].to_numpy(bool)).sum())
    n_fs_mismatch = int((fs_sig
                         != decided["FS_LATE_COLLAPSE_SIGNATURE"].to_numpy(bool)).sum())
    return {
        "source": "frozen Phase 0B detector_spec.json re-applied to the frozen "
                  "step_table; compared against Phase 0B trajectory_outcomes.csv",
        "step_table_files": loaded["step_table_files"],
        "step_table_rows": loaded["step_table_rows"],
        "decided_trajectories_checked": int(checked),
        "max_post_stop_steps_observed": int(worst),
        "event_mismatch_counts": {c: len(v) for c, v in mismatches.items()},
        "event_mismatch_examples": {c: v[:5] for c, v in mismatches.items() if v},
        "ff_signature_mismatches": n_ff_mismatch,
        "fs_signature_mismatches": n_fs_mismatch,
        "all_events_reproduced": bool(
            all(len(v) == 0 for v in mismatches.values())
            and n_ff_mismatch == 0 and n_fs_mismatch == 0),
    }


def bin_codes(fraction: np.ndarray) -> np.ndarray:
    """[0,0.25) [0.25,0.50) [0.50,0.75) [0.75,1.00] as codes 0..3."""
    return np.digitize(fraction, [0.25, 0.50, 0.75], right=False).astype(np.int64)


class Universe:
    """Compact numpy view of the frozen Phase 0B trajectory outcome table."""

    def __init__(self, traj: pd.DataFrame):
        self.traj = traj
        self.decision = traj["decision"].to_numpy()
        self.cls = traj["outcome_class"].to_numpy()
        self.model = traj["model_id"].to_numpy()
        self.inst = traj["instance_id"].to_numpy()
        frac = traj["decision_fraction"].to_numpy(dtype=np.float64)
        self.decided = self.decision != "undecided"
        self.bin_code = np.full(len(traj), -1, dtype=np.int64)
        self.bin_code[self.decided] = bin_codes(frac[self.decided])
        self.fs_sig = traj["FS_LATE_COLLAPSE_SIGNATURE"].to_numpy(bool)
        self.ff_sig = traj["FF_LATE_RECOVERY_SIGNATURE"].to_numpy(bool)
        self.models = sorted(set(self.model))
        self.model_code = np.searchsorted(np.asarray(self.models), self.model)
        self.n_models = len(self.models)
        self.n_strata = self.n_models * N_BINS
        self.strata_code = (self.model_code * N_BINS
                            + np.where(self.bin_code < 0, 0, self.bin_code))
        self._task_codes()

    def _task_codes(self) -> None:
        uniq, codes = np.unique(self.inst, return_inverse=True)
        self.tasks = uniq
        self.task_code = codes
        self.n_tasks = len(uniq)
        self.order = np.argsort(codes, kind="stable")
        self.counts_per_task = np.bincount(codes, minlength=self.n_tasks)
        self.sorted_codes = codes[self.order]

    def task_bootstrap_index(self, rng) -> np.ndarray:
        drawn = rng.integers(0, self.n_tasks, size=self.n_tasks)
        rep_counts = np.bincount(drawn, minlength=self.n_tasks)
        return np.repeat(self.order, rep_counts[self.sorted_codes])


def confusion_metrics(dec, cls) -> dict:
    succ = dec == "success"
    fail = dec == "failure"
    n_succ = int(succ.sum())
    n_fail = int(fail.sum())
    cs = int(np.sum(succ & (cls == "CORRECT_SUCCESS")))
    fs = int(np.sum(succ & (cls == "FALSE_SUCCESS")))
    cf = int(np.sum(fail & (cls == "CORRECT_FAILURE")))
    ff = int(np.sum(fail & (cls == "FALSE_FAILURE")))
    n_err = fs + ff
    ser = safe_ratio(fs, n_succ)
    fer = safe_ratio(ff, n_fail)
    return {
        "EARLY_SUCCESS_TOTAL": n_succ,
        "EARLY_FAILURE_TOTAL": n_fail,
        "CORRECT_SUCCESS": cs,
        "FALSE_SUCCESS": fs,
        "CORRECT_FAILURE": cf,
        "FALSE_FAILURE": ff,
        "SUCCESS_DECISION_ERROR_RATE": ser,
        "FAILURE_DECISION_ERROR_RATE": fer,
        "SUCCESS_DECISION_PRECISION": safe_ratio(cs, n_succ),
        "FAILURE_DECISION_PRECISION": safe_ratio(cf, n_fail),
        "DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE":
            (fer - ser) if (ser is not None and fer is not None) else None,
        "N_ERRORS": n_err,
        "FALSE_SUCCESS_SHARE_OF_ERRORS": safe_ratio(fs, n_err),
        "FALSE_FAILURE_SHARE_OF_ERRORS": safe_ratio(ff, n_err),
    }


def standardized_control(err_mask, ctrl_mask, strata_code, sig, n_strata) -> dict:
    """Direct standardization on common support.

    The control group is reweighted to the error group's stratum distribution. A
    stratum contributes only when it holds both error and control observations, so
    the standardized error rate and the standardized control rate are computed on
    the identical stratum support and their difference is a coherent stratified risk
    difference. The share of the error distribution that lies inside that common
    support is reported as `covered_error_weight`; the remainder cannot be compared
    against any matched control in this frozen design.
    """
    es = strata_code[err_mask]
    cs = strata_code[ctrl_mask]
    ec = np.bincount(es, minlength=n_strata).astype(np.float64)
    cc = np.bincount(cs, minlength=n_strata).astype(np.float64)
    ek = np.bincount(es, weights=sig[err_mask].astype(np.float64),
                     minlength=n_strata)
    ck = np.bincount(cs, weights=sig[ctrl_mask].astype(np.float64),
                     minlength=n_strata)
    n_err = float(ec.sum())
    if n_err == 0:
        return {"standardized_error_rate": None, "standardized_control_rate": None,
                "risk_difference": None, "covered_error_weight": 0.0,
                "n_strata_used": 0, "n_strata_with_error": 0,
                "n_error_in_common_support": 0}
    usable = (cc > 0) & (ec > 0)
    covered = float(ec[usable].sum() / n_err)
    if ec[usable].sum() == 0:
        return {"standardized_error_rate": None, "standardized_control_rate": None,
                "risk_difference": None, "covered_error_weight": covered,
                "n_strata_used": 0, "n_strata_with_error": int((ec > 0).sum()),
                "n_error_in_common_support": 0}
    w = ec[usable] / ec[usable].sum()
    p_e = ek[usable] / ec[usable]
    p_c = ck[usable] / cc[usable]
    return {
        "standardized_error_rate": float((w * p_e).sum()),
        "standardized_control_rate": float((w * p_c).sum()),
        "risk_difference": float((w * (p_e - p_c)).sum()),
        "covered_error_weight": covered,
        "n_strata_used": int(usable.sum()),
        "n_strata_with_error": int((ec > 0).sum()),
        "n_error_in_common_support": int(ec[usable].sum()),
    }


def signature_block(u: "Universe", direction: str,
                    idx: np.ndarray | None = None) -> dict:
    """Error/control signature comparison for one decision class."""
    if direction == "FALSE_SUCCESS":
        error_label, control_label = "FALSE_SUCCESS", "CORRECT_SUCCESS"
        sig = u.fs_sig
    else:
        error_label, control_label = "FALSE_FAILURE", "CORRECT_FAILURE"
        sig = u.ff_sig
    cls = u.cls if idx is None else u.cls[idx]
    strata = u.strata_code if idx is None else u.strata_code[idx]
    s = sig if idx is None else sig[idx]
    models = u.model if idx is None else u.model[idx]
    err_mask = cls == error_label
    ctrl_mask = cls == control_label
    n_err = int(err_mask.sum())
    n_ctrl = int(ctrl_mask.sum())
    err_rate = float(s[err_mask].mean()) if n_err else None
    ctrl_rate = float(s[ctrl_mask].mean()) if n_ctrl else None
    std = standardized_control(err_mask, ctrl_mask, strata, s, u.n_strata)
    raw_rd = (err_rate - ctrl_rate) if (err_rate is not None
                                       and ctrl_rate is not None) else None
    std_rd = std["risk_difference"]
    composite_rd = (err_rate - std["standardized_control_rate"]) \
        if (err_rate is not None
            and std["standardized_control_rate"] is not None) else None
    return {
        "direction": direction,
        "error_class": error_label,
        "control_class": control_label,
        "n_error": n_err,
        "n_control": n_ctrl,
        "n_models_with_error": int(len(set(models[err_mask]))) if n_err else 0,
        "signature_error_rate": err_rate,
        "signature_raw_control_rate": ctrl_rate,
        "signature_standardized_error_rate_common_support":
            std["standardized_error_rate"],
        "signature_standardized_control_rate": std["standardized_control_rate"],
        "standardization_covered_error_weight": std["covered_error_weight"],
        "standardization_strata_used": std["n_strata_used"],
        "standardization_strata_with_error": std["n_strata_with_error"],
        "standardization_n_error_in_common_support":
            std["n_error_in_common_support"],
        "risk_difference_raw": raw_rd,
        "risk_difference_standardized": std_rd,
        "risk_difference_composite_full_error_minus_covered_control": composite_rd,
    }


def post_stop_profile(traj: pd.DataFrame, error_label: str, control_label: str,
                      sig_col: str) -> dict:
    components = ["POST_STOP_ACTIVITY", "POST_STOP_EDIT", "POST_STOP_TEST",
                  "POST_STOP_TEST_PASS", "POST_STOP_TEST_FAIL",
                  "POST_STOP_ERROR_OR_TRACEBACK", "POST_STOP_SUBMISSION"]

    def one(group: pd.DataFrame) -> dict:
        out = {"n": int(len(group))}
        for c in components:
            out[f"{c}_rate"] = (float(group[c].astype(bool).mean())
                                if len(group) else None)
        out[f"{sig_col}_rate"] = (float(group[sig_col].astype(bool).mean())
                                  if len(group) else None)
        steps = group["POST_STOP_STEPS"].astype(float)
        out["post_stop_steps"] = {
            "mean": float(steps.mean()) if len(group) else None,
            "median": float(steps.median()) if len(group) else None,
            "min": int(steps.min()) if len(group) else None,
            "max": int(steps.max()) if len(group) else None,
            "p25": float(np.percentile(steps, 25)) if len(group) else None,
            "p75": float(np.percentile(steps, 75)) if len(group) else None,
            "zero_steps_rate": float((steps == 0).mean()) if len(group) else None,
        }
        return out

    return {"error_class": error_label, "control_class": control_label,
            "signature_column": sig_col,
            "error_group": one(traj[traj["outcome_class"] == error_label]),
            "control_group": one(traj[traj["outcome_class"] == control_label])}


def strata_table(u: "Universe") -> pd.DataFrame:
    rows = []
    for direction, error_label, control_label, sig in (
            ("FALSE_SUCCESS", "FALSE_SUCCESS", "CORRECT_SUCCESS", u.fs_sig),
            ("FALSE_FAILURE", "FALSE_FAILURE", "CORRECT_FAILURE", u.ff_sig)):
        for model in u.models:
            for b in range(N_BINS):
                mask = (u.model == model) & (u.bin_code == b)
                err = mask & (u.cls == error_label)
                ctl = mask & (u.cls == control_label)
                n_err = int(err.sum())
                n_ctl = int(ctl.sum())
                err_rate = float(sig[err].mean()) if n_err else None
                ctl_rate = float(sig[ctl].mean()) if n_ctl else None
                rows.append({
                    "direction": direction,
                    "model_id": model,
                    "decision_fraction_bin": BIN_LABELS[b],
                    "n_error": n_err,
                    "n_control": n_ctl,
                    "signature_rate_error": err_rate,
                    "signature_rate_control": ctl_rate,
                    "risk_difference": (err_rate - ctl_rate)
                    if (err_rate is not None and ctl_rate is not None) else None,
                    "stratum_usable": bool(n_err > 0 and n_ctl > 0),
                })
    return pd.DataFrame(rows)


def per_model_conditional(u: "Universe",
                          idx: np.ndarray | None = None) -> pd.DataFrame:
    traj = u.traj if idx is None else u.traj.iloc[idx]
    rows = []
    for model, part in traj.groupby("model_id", sort=True):
        succ = part[part["decision"] == "success"]
        fail = part[part["decision"] == "failure"]
        n_succ = int(len(succ))
        n_fail = int(len(fail))
        fs = int((succ["outcome_class"] == "FALSE_SUCCESS").sum())
        ff = int((fail["outcome_class"] == "FALSE_FAILURE").sum())
        resolved = int(part["resolved"].sum())
        rows.append({
            "model_id": model,
            "trajectories": int(len(part)),
            "resolved": resolved,
            "resolve_rate": float(part["resolved"].mean()),
            "early_success_decisions": n_succ,
            "false_success": fs,
            "success_decision_error_rate": safe_ratio(fs, n_succ),
            "success_decision_precision": safe_ratio(n_succ - fs, n_succ),
            "early_failure_decisions": n_fail,
            "false_failure": ff,
            "failure_decision_error_rate": safe_ratio(ff, n_fail),
            "failure_decision_precision": safe_ratio(n_fail - ff, n_fail),
            "no_stop": int((part["decision"] == "undecided").sum()),
            "class_degenerate_heldout_model": bool(
                resolved == 0 or resolved == len(part)),
        })
    return pd.DataFrame(rows)


def concentration(u: "Universe") -> dict:
    out = {}
    for label in ("FALSE_SUCCESS", "FALSE_FAILURE"):
        sub = u.traj.loc[u.cls == label]
        n = int(len(sub))
        by_model = sub["model_id"].value_counts().to_dict()
        by_task = sub["instance_id"].value_counts().to_dict()
        top5 = sorted(by_model.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top10 = sorted(by_task.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        max_task = max(by_task.items(), key=lambda kv: kv[1])[0] if by_task else None
        out[label] = {
            "count": n,
            "by_model": {k: int(v) for k, v in sorted(by_model.items())},
            "n_models_with_error": int(len(by_model)),
            "largest_model": top5[0][0] if top5 else None,
            "largest_model_share": (float(top5[0][1]) / n) if (top5 and n) else None,
            "top5_models": [{"model_id": k, "count": int(v)} for k, v in top5],
            "unique_tasks": int(sub["instance_id"].nunique()),
            "max_trajectories_single_task": int(max(by_task.values()))
            if by_task else 0,
            "task_with_max_trajectories": max_task,
            "top10_tasks": [{"instance_id": k, "count": int(v)} for k, v in top10],
        }
    return out


def bootstrap(u: "Universe", replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ser, fer, serdiff = [], [], []
    fs_std, ff_std, fs_raw, ff_raw = [], [], [], []
    for _ in range(replicates):
        idx = u.task_bootstrap_index(rng)
        m = confusion_metrics(u.decision[idx], u.cls[idx])
        ser.append(m["SUCCESS_DECISION_ERROR_RATE"])
        fer.append(m["FAILURE_DECISION_ERROR_RATE"])
        serdiff.append(m["DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE"])
        fs = signature_block(u, "FALSE_SUCCESS", idx)
        ff = signature_block(u, "FALSE_FAILURE", idx)
        fs_std.append(fs["risk_difference_standardized"])
        ff_std.append(ff["risk_difference_standardized"])
        fs_raw.append(fs["risk_difference_raw"])
        ff_raw.append(ff["risk_difference_raw"])
    return {
        "method": "task-cluster (instance_id) bootstrap; each replicate resamples "
                  "task IDs with replacement and includes all model trajectories "
                  "of every selected task; prefix rows are never resampled",
        "replicates": int(replicates),
        "seed": int(seed),
        "SUCCESS_DECISION_ERROR_RATE": percentile_ci(ser),
        "FAILURE_DECISION_ERROR_RATE": percentile_ci(fer),
        "DIFFERENCE_FAILURE_MINUS_SUCCESS_ERROR_RATE": percentile_ci(serdiff),
        "FS_SIGNATURE_RISK_DIFFERENCE_RAW": percentile_ci(fs_raw),
        "FS_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED": percentile_ci(fs_std),
        "FF_SIGNATURE_RISK_DIFFERENCE_RAW": percentile_ci(ff_raw),
        "FF_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED": percentile_ci(ff_std),
    }


def adapter_failure_records() -> pd.DataFrame:
    import json
    df = pd.read_csv(INPUT_FILES["adapter_failures.csv"])
    rows = []
    for r in df.itertuples(index=False):
        rel = str(getattr(r, "file")).replace("\\", "/")
        parts = rel.split("/")
        model = parts[0]
        inst = parts[1] if len(parts) > 1 else None
        label = None
        cand = DATASET_SNAPSHOT / model / inst / f"{inst}.traj.json" if inst else None
        if cand is not None and cand.exists():
            try:
                doc = json.loads(cand.read_text("utf-8"))
                info = doc.get("info") or {}
                resolved = info.get("resolved")
                label = bool(resolved) if isinstance(resolved, bool) else None
            except (OSError, ValueError):
                label = None
        rows.append({
            "model_id": model,
            "instance_id": inst,
            "final_label": ("RESOLVED_TRUE" if label is True
                            else "RESOLVED_FALSE" if label is False else "NA"),
            "final_label_recovered": bool(label is not None),
            "stage": getattr(r, "stage"),
            "reason": getattr(r, "reason"),
            "raw_file": rel,
            "raw_file_found": bool(cand is not None and cand.exists()),
        })
    return pd.DataFrame(rows)


def build_gate(blocks: dict, boot: dict, concentration: dict) -> dict:
    """Section 11: mechanical, pre-registered Phase 0C gate."""
    fs = blocks["FALSE_SUCCESS"]
    ff = blocks["FALSE_FAILURE"]
    fs_rd = fs["risk_difference_standardized"]
    ff_rd = ff["risk_difference_standardized"]
    fs_ci = boot["FS_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED"]
    ff_ci = boot["FF_SIGNATURE_RISK_DIFFERENCE_STANDARDIZED"]
    fs_lo = fs_ci["ci_lower"]
    ff_lo = ff_ci["ci_lower"]
    gate = {
        "definition": {
            "FS_ENRICHED": "FS late-collapse signature risk difference >= +0.20 "
                           "AND bootstrap 95% CI lower bound > 0",
            "FF_ENRICHED": "FF late-recovery signature risk difference >= +0.20 "
                           "AND bootstrap 95% CI lower bound > 0",
            "A_bidirectional": "FS_ENRICHED AND FF_ENRICHED -> "
                               "ROBUST_BIDIRECTIONAL_SIGNAL",
            "B_one_direction": "one direction: risk difference >= +0.30, CI lower "
                               "bound > 0, errors >= 30, represented in >= 3 "
                               "models -> ROBUST_ONE_DIRECTION_SIGNAL",
            "otherwise": "SIGNAL_NOT_YET_ROBUST",
        },
        "FS_risk_difference_standardized": fs_rd,
        "FS_bootstrap_ci_lower": fs_lo,
        "FF_risk_difference_standardized": ff_rd,
        "FF_bootstrap_ci_lower": ff_lo,
        "FS_ENRICHED": bool(fs_rd is not None and fs_lo is not None
                            and fs_rd >= 0.20 and fs_lo > 0),
        "FF_ENRICHED": bool(ff_rd is not None and ff_lo is not None
                            and ff_rd >= 0.20 and ff_lo > 0),
    }
    if gate["FS_ENRICHED"] and gate["FF_ENRICHED"]:
        gate["OVERALL"] = "ROBUST_BIDIRECTIONAL_SIGNAL"
        gate["one_direction_strong_candidate"] = None
        gate["one_direction_checks"] = {}
    else:
        checks = {}
        strong = None
        for name, block, rd, lo in (
                ("FALSE_SUCCESS", fs, fs_rd, fs_lo),
                ("FALSE_FAILURE", ff, ff_rd, ff_lo)):
            ok = bool(rd is not None and lo is not None and rd >= 0.30 and lo > 0
                      and block["n_error"] >= 30
                      and block["n_models_with_error"] >= 3)
            checks[name] = {
                "risk_difference_standardized": rd,
                "bootstrap_ci_lower": lo,
                "n_error": block["n_error"],
                "n_models_with_error": block["n_models_with_error"],
                "meets_one_direction_strong_rule": ok,
            }
            if ok and strong is None:
                strong = name
        gate["one_direction_checks"] = checks
        gate["one_direction_strong_candidate"] = strong
        gate["OVERALL"] = ("ROBUST_ONE_DIRECTION_SIGNAL" if strong
                           else "SIGNAL_NOT_YET_ROBUST")
    gate["note"] = ("mechanical, pre-registered; thresholds were not altered "
                    "after seeing results")
    return gate


def json_dumps(payload) -> str:
    import json
    return json.dumps(as_builtin(payload), ensure_ascii=False, indent=2)


def main() -> int:
    ensure_dirs(ANA)
    inputs = hash_inputs()
    write_json(OUT / "input_hashes.json", inputs)
    print("input hashes recorded, all present:", inputs["all_inputs_present"],
          flush=True)
    traj = pd.read_csv(INPUT_FILES["trajectory_outcomes.csv"])
    spec = read_json(INPUT_FILES["detector_spec.json"])
    u = Universe(traj)
    traj["bin_code"] = u.bin_code
    recheck = detector_recheck(traj, spec)
    write_json(ANA / "detector_recheck.json", recheck)
    print("detector recheck all reproduced:", recheck["all_events_reproduced"],
          flush=True)
    conf = confusion_metrics(u.decision, u.cls)
    conf["QUANTITY_DISTINCTION"] = (
        "ERROR COUNT SHARE (FALSE_SUCCESS_SHARE_OF_ERRORS / "
        "FALSE_FAILURE_SHARE_OF_ERRORS) is a different quantity from the "
        "CONDITIONAL DECISION ERROR RATE (error rate within a decision class).")
    write_json(ANA / "decision_confusion.json", conf)
    per_model_conditional(u).to_csv(ANA / "per_model_conditional.csv", index=False,
                                    encoding="utf-8")
    fs_controls = post_stop_profile(traj, "FALSE_SUCCESS", "CORRECT_SUCCESS",
                                    "FS_LATE_COLLAPSE_SIGNATURE")
    fs_block = signature_block(u, "FALSE_SUCCESS")
    fs_controls["signature_control_comparison"] = fs_block
    pd.DataFrame([
        {"group": "FALSE_SUCCESS (error)", **fs_controls["error_group"]},
        {"group": "CORRECT_SUCCESS (control)", **fs_controls["control_group"]},
    ]).drop(columns=["post_stop_steps"]).to_csv(
        ANA / "success_signature_controls.csv", index=False, encoding="utf-8")
    ff_controls = post_stop_profile(traj, "FALSE_FAILURE", "CORRECT_FAILURE",
                                    "FF_LATE_RECOVERY_SIGNATURE")
    ff_block = signature_block(u, "FALSE_FAILURE")
    ff_controls["signature_control_comparison"] = ff_block
    pd.DataFrame([
        {"group": "FALSE_FAILURE (error)", **ff_controls["error_group"]},
        {"group": "CORRECT_FAILURE (control)", **ff_controls["control_group"]},
    ]).drop(columns=["post_stop_steps"]).to_csv(
        ANA / "failure_signature_controls.csv", index=False, encoding="utf-8")
    write_json(ANA / "post_stop_profiles.json",
               {"success_decisions": fs_controls, "failure_decisions": ff_controls})
    strata = strata_table(u)
    strata.to_csv(ANA / "strata.csv", index=False, encoding="utf-8")
    std_rates = {
        "FALSE_SUCCESS": fs_block,
        "FALSE_FAILURE": ff_block,
        "stratification": {
            "strata_definition": "held-out model x decision_fraction bin",
            "bins": BIN_LABELS,
            "bin_rule": "[0,0.25) [0.25,0.50) [0.50,0.75) [0.75,1.00]",
            "standardization": "direct standardization on common support; controls "
                               "are reweighted to the error group's stratum "
                               "distribution, and only strata holding both error and "
                               "control observations contribute. The standardized "
                               "error rate and standardized control rate therefore "
                               "share one stratum support and their difference is a "
                               "coherent stratified risk difference.",
            "common_support_coverage_reporting": "the share of the error "
                                                 "distribution inside the common "
                                                 "support is reported as "
                                                 "covered_error_weight, together "
                                                 "with n_error_in_common_support; "
                                                 "error mass outside it cannot be "
                                                 "compared with any matched control",
            "composite_diagnostic": "risk_difference_composite_full_error_minus_"
                                    "covered_control is reported for transparency "
                                    "only; it mixes the full-group error rate with a "
                                    "partially covered control rate and is not used "
                                    "for the gate",
            "no_propensity_modeling": True,
            "n_strata_grid": int(u.n_strata),
        },
    }
    write_json(ANA / "standardized_rates.json", as_builtin(std_rates))
    conc = concentration(u)
    write_json(ANA / "concentration.json", as_builtin(conc))
    boot = bootstrap(u, BOOT_REPLICATES, BOOT_SEED)
    write_json(ANA / "bootstrap_results.json", as_builtin(boot))
    gem_mask = np.flatnonzero(u.model != GEMINI_PRO)
    gem = {
        "excluded_model": GEMINI_PRO,
        "declared_sensitivity_only": True,
        "is_primary_universe": False,
        "rationale": "the held-out gemini-3-pro fold contains zero final failures, "
                     "so it cannot contribute CORRECT_FAILURE control rows; excluded "
                     "only as a declared sensitivity check",
        "trajectories_remaining": int(len(gem_mask)),
        "decision_confusion": confusion_metrics(u.decision[gem_mask], u.cls[gem_mask]),
        "FS_signature": signature_block(u, "FALSE_SUCCESS", gem_mask),
        "FF_signature": signature_block(u, "FALSE_FAILURE", gem_mask),
    }
    write_json(ANA / "gemini3pro_sensitivity.json", as_builtin(gem))
    af = adapter_failure_records()
    af.to_csv(ANA / "adapter_failures.csv", index=False, encoding="utf-8")
    af_summary = {
        "n_failures": int(len(af)),
        "by_model": {k: int(v) for k, v in af["model_id"].value_counts().items()},
        "by_final_label": {k: int(v)
                           for k, v in af["final_label"].value_counts().items()},
        "by_stage": {k: int(v) for k, v in af["stage"].value_counts().items()},
        "labels_recovered": int(af["final_label_recovered"].sum()),
        "unique_tasks": int(af["instance_id"].nunique()),
        "task_ids": sorted(af["instance_id"].dropna().unique().tolist()),
    }
    write_json(ANA / "adapter_failure_summary.json", af_summary)
    gate = build_gate({"FALSE_SUCCESS": fs_block, "FALSE_FAILURE": ff_block},
                      boot, conc)
    write_json(ANA / "phase0c_gate.json", as_builtin(gate))
    write_json(ANA / "phase0c_summary.json", as_builtin({
        "decision_confusion": conf,
        "FS_signature_block": fs_block,
        "FF_signature_block": ff_block,
        "gate": gate,
    }))
    print(json_dumps(gate), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
