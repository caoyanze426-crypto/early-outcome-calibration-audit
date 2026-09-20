# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2E - Sections 3-7: exact targets, primaries, status, gate.

Reads only frozen SWE Phase 1B and TerminalBench Phase 2B / 2B-D artifacts.
No predictor is retrained and no decision is changed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from e_common import (ANA, COLLAPSE_GAP, MIN_DECISIONS, OUT_1B, OUT_2B,
                      OUT_2BD, PERSIST_GAP, PRIMARY_THRESHOLD, TARGETS,
                      THRESHOLDS, WORK, as_builtin, ensure_dirs, read_json,
                      write_json)

SWE_OCC = OUT_1B / "analysis" / "threshold_occurrences.csv"
SWE_SUM = OUT_1B / "analysis" / "target_threshold_summary.csv"
SWE_BOOT = OUT_1B / "analysis" / "bootstrap_results.json"
TB_2B = OUT_2B / "analysis" / "per_target_metrics.csv"
TB_2BD = OUT_2BD / "analysis" / "threshold_target_metrics.csv"
TOL = 1e-9
LABEL_BY_THR = {0.9: "T1", 0.925: "T2", 0.95: "T3", 0.975: "T4"}


def swe_threshold_rows(occ, scope, thr):
    return occ[(occ["target_scope"] == scope)
               & np.isclose(occ["threshold"].astype(float), thr)]


def swe_point(occ, boot, summary, scope, thr):
    rows = swe_threshold_rows(occ, scope, thr)
    key = "%s|%s" % (scope, LABEL_BY_THR[float(thr)])
    s = summary[(summary["target_scope"] == scope)
                & np.isclose(summary["threshold"].astype(float), thr)]
    no_gap = None
    if len(s) and pd.notna(s["median_signed_gap"].iloc[0]):
        no_gap = float(s["median_signed_gap"].iloc[0])
    if not len(rows):
        return {"occurrences": 0, "decisions": 0, "decisions_min": None,
                "decisions_max": None, "decisions_median": None,
                "precision_pooled": None, "precision_median": None,
                "corrected_mean_score_pooled": None,
                "corrected_mean_score_median": None,
                "signed_corrected_gap": no_gap,
                "bootstrap_ci_lower": None, "bootstrap_ci_upper": None,
                "bootstrap_median": None, "bootstrap_replicates": None,
                "bootstrap_seed": None,
                "eligible_occurrences": int(s["eligible_folds"].iloc[0])
                if len(s) else 0}
    dec = rows["n_decisions"].to_numpy(dtype=np.int64)
    correct = np.rint(rows["empirical_precision"].to_numpy(dtype=float)
                      * dec).astype(np.int64)
    pooled_prec = float(correct.sum() / dec.sum()) if dec.sum() else None
    pooled_cor = float((rows["corrected_mean_score"].to_numpy(dtype=float)
                        * dec).sum() / dec.sum()) if dec.sum() else None
    b = boot.get(key, {})
    ci = b.get("median_signed_gap", {})
    return {"occurrences": int(len(rows)), "decisions": int(dec.sum()),
            "decisions_min": int(dec.min()), "decisions_max": int(dec.max()),
            "decisions_median": float(np.median(dec)),
            "precision_pooled": pooled_prec,
            "precision_median": float(rows["empirical_precision"].median()),
            "corrected_mean_score_pooled": pooled_cor,
            "corrected_mean_score_median":
                float(rows["corrected_mean_score"].median()),
            "signed_corrected_gap": no_gap,
            "bootstrap_ci_lower": ci.get("ci_lower"),
            "bootstrap_ci_upper": ci.get("ci_upper"),
            "bootstrap_median": ci.get("median"),
            "bootstrap_replicates": b.get("replicates"),
            "bootstrap_seed": b.get("seed"),
            "eligible_occurrences": int(s["eligible_folds"].iloc[0])
            if len(s) else int(len(rows))}


def tb_point(m2b, m2bd, model, head, thr):
    if np.isclose(thr, PRIMARY_THRESHOLD):
        r = m2b[(m2b["model_id"] == model) & (m2b["head"] == head)]
        src = "Phase 2B analysis/per_target_metrics.csv (0.950 frozen)"
        if len(r):
            r = r.iloc[0]
            return tb_row(r, src)
    r = m2bd[(m2bd["model_id"] == model) & (m2bd["head"] == head)
             & np.isclose(m2bd["threshold"].astype(float), thr)]
    src = "Phase 2B-D analysis/threshold_target_metrics.csv (%.3f)" % thr
    if not len(r):
        return {"decisions": 0, "empirical_precision": None,
                "mean_raw_score": None, "mean_corrected_score": None,
                "signed_corrected_gap": None, "absolute_corrected_gap": None,
                "bootstrap_ci_lower": None, "bootstrap_ci_upper": None,
                "distinct_tasks": 0, "source": src}
    return tb_row(r.iloc[0], src)


def tb_row(r, src):
    def val(col):
        v = r[col]
        return None if pd.isna(v) else float(v)

    return {"decisions": int(r["decisions"]),
            "empirical_precision": val("empirical_precision"),
            "mean_raw_score": val("mean_raw_score"),
            "mean_corrected_score": val("mean_corrected_score"),
            "signed_corrected_gap": val("corrected_gap"),
            "absolute_corrected_gap": val("absolute_corrected_gap"),
            "bootstrap_ci_lower": val("bootstrap_ci_lower"),
            "bootstrap_ci_upper": val("bootstrap_ci_upper"),
            "distinct_tasks": int(r["distinct_tasks"]), "source": src}


def ci_includes_zero(lo, hi):
    if lo is None or hi is None:
        return None
    return bool(float(lo) <= 0.0 <= float(hi))


def sign_of(value):
    if value is None:
        return None
    if float(value) > 0:
        return "positive"
    if float(value) < 0:
        return "negative"
    return "zero"


def classify(tb, swe_sign, identity):
    dec = int(tb["decisions"] or 0)
    gap = tb["signed_corrected_gap"]
    agap = None if gap is None else abs(float(gap))
    lo, hi = tb["bootstrap_ci_lower"], tb["bootstrap_ci_upper"]
    inc0 = ci_includes_zero(lo, hi)
    tb_sign = sign_of(gap)
    flags = {
        "tb_decisions": dec,
        "tb_decisions_ge_20": bool(dec >= MIN_DECISIONS),
        "abs_corrected_gap": agap,
        "abs_gap_ge_0_08": (None if agap is None
                            else bool(agap >= PERSIST_GAP)),
        "abs_gap_lt_0_04": (None if agap is None
                            else bool(agap < COLLAPSE_GAP)),
        "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi,
        "ci_excludes_zero": (None if inc0 is None else bool(not inc0)),
        "ci_includes_zero": inc0,
        "tb_gap_sign": tb_sign, "swe_gap_sign": swe_sign,
        "sign_matches_swe": (None if tb_sign is None or swe_sign is None
                             else bool(tb_sign == swe_sign)),
        "identity": identity,
        "identity_confirmed": bool(identity == "CONFIRMED"),
        "bootstrap_inference_available": bool(lo is not None
                                              and hi is not None),
    }
    if not flags["identity_confirmed"]:
        return "INDETERMINATE", flags
    if dec < MIN_DECISIONS:
        return "INDETERMINATE", flags
    if not flags["bootstrap_inference_available"]:
        return "INDETERMINATE", flags
    if (agap is not None and agap >= PERSIST_GAP and inc0 is False
            and flags["sign_matches_swe"]):
        return "PERSISTENT_CROSS_BENCHMARK", flags
    if agap is not None and agap < COLLAPSE_GAP and inc0 is True:
        return "COLLAPSED_ON_TERMINALBENCH", flags
    if flags["sign_matches_swe"] is False:
        return "SIGN_CHANGED", flags
    return "UNCLASSIFIED_BY_SPEC", flags


def cross_row(scope, head, benchmark, model, identity, d, basis, source,
              is_swe):
    gap = d["signed_corrected_gap"]
    return {
        "target_scope": scope, "head": head, "benchmark": benchmark,
        "model_label": model, "threshold": PRIMARY_THRESHOLD,
        "identity": identity, "decisions": d["decisions"],
        "decisions_basis": basis,
        "empirical_precision": (d["precision_pooled"] if is_swe
                                else d["empirical_precision"]),
        "corrected_mean_score": (d["corrected_mean_score_pooled"] if is_swe
                                 else d["mean_corrected_score"]),
        "signed_corrected_gap": gap,
        "absolute_corrected_gap": (None if gap is None
                                   else abs(float(gap))),
        "bootstrap_ci_lower": d["bootstrap_ci_lower"],
        "bootstrap_ci_upper": d["bootstrap_ci_upper"],
        "ci_includes_zero": ci_includes_zero(d["bootstrap_ci_lower"],
                                             d["bootstrap_ci_upper"]),
        "gap_sign": sign_of(gap),
        "occurrences_or_tasks": (d["occurrences"] if is_swe
                                 else d["distinct_tasks"]),
        "source": source,
    }


def main() -> int:
    t0 = time.time()
    ensure_dirs(WORK, ANA)
    occ = pd.read_csv(SWE_OCC)
    summary = pd.read_csv(SWE_SUM)
    boot = read_json(SWE_BOOT)["primary"]
    m2b = pd.read_csv(TB_2B)
    m2bd = pd.read_csv(TB_2BD)
    ident = read_json(ANA / "model_identity.json")["targets"]
    cross_rows, desc_rows, status = [], [], {}
    for spec in TARGETS:
        scope, head = spec["scope"], spec["head"]
        ident_state = ident[scope]["IDENTITY"]
        swe = swe_point(occ, boot, summary, scope, PRIMARY_THRESHOLD)
        tb = tb_point(m2b, m2bd, spec["tb_model"], head, PRIMARY_THRESHOLD)
        swe_sign = sign_of(swe["signed_corrected_gap"])
        status_name, flags = classify(tb, swe_sign, ident_state)
        cross_rows.append(cross_row(
            scope, head, "SWE-bench", spec["swe_model"], ident_state, swe,
            "sum over %d eligible frozen occurrences (per-occurrence min %s, "
            "max %s, median %s)"
            % (swe["occurrences"], swe["decisions_min"], swe["decisions_max"],
               swe["decisions_median"]),
            "%s + %s" % (SWE_SUM.name, SWE_BOOT.name), True))
        cross_rows.append(cross_row(
            scope, head, "TerminalBench", spec["tb_model"], ident_state, tb,
            "frozen Phase 2B decided-trajectory count for this model/head "
            "under the fixed scaffold terminus-2",
            tb["source"], False))
        per_thr = {}
        for thr in THRESHOLDS:
            s = swe_point(occ, boot, summary, scope, thr)
            t = tb_point(m2b, m2bd, spec["tb_model"], head, thr)
            st, st_flags = classify(t, sign_of(s["signed_corrected_gap"]),
                                    ident_state)
            per_thr["%.3f" % thr] = {"threshold": float(thr), "STATUS": st,
                                     "clauses": st_flags}
            for bench, model, d, is_swe in (
                    ("SWE-bench", spec["swe_model"], s, True),
                    ("TerminalBench", spec["tb_model"], t, False)):
                gap = d["signed_corrected_gap"]
                desc_rows.append({
                    "target_scope": scope, "head": head, "benchmark": bench,
                    "model_label": model, "threshold": float(thr),
                    "decisions": d["decisions"],
                    "empirical_precision": (d["precision_pooled"] if is_swe
                                            else d["empirical_precision"]),
                    "corrected_mean_score":
                        (d["corrected_mean_score_pooled"] if is_swe
                         else d["mean_corrected_score"]),
                    "signed_corrected_gap": gap,
                    "absolute_corrected_gap": (None if gap is None
                                               else abs(float(gap))),
                    "bootstrap_ci_lower": d["bootstrap_ci_lower"],
                    "bootstrap_ci_upper": d["bootstrap_ci_upper"],
                    "status_at_threshold": st,
                })
        status[scope] = {
            "scope": scope, "head": head, "swe_model": spec["swe_model"],
            "tb_model": spec["tb_model"], "identity": ident_state,
            "primary_threshold": PRIMARY_THRESHOLD,
            "swe_signed_corrected_gap": swe["signed_corrected_gap"],
            "swe_sign": swe_sign,
            "tb_signed_corrected_gap": tb["signed_corrected_gap"],
            "STATUS": status_name, "clauses": flags,
            "qualifies_for_primary_comparison": bool(ident_state == "CONFIRMED"),
            "per_threshold_status": per_thr,
        }
    return _finish(cross_rows, desc_rows, status, t0)


def _finish(cross_rows, desc_rows, status, t0):
    cross = pd.DataFrame(cross_rows)
    desc = pd.DataFrame(desc_rows)
    cross.to_csv(ANA / "target_cross_benchmark.csv", index=False,
                 encoding="utf-8")
    desc.to_csv(ANA / "threshold_descriptives.csv", index=False,
                encoding="utf-8")
    qualified = [k for k, v in status.items()
                 if v["qualifies_for_primary_comparison"]]
    persistent = [k for k, v in status.items()
                  if v["STATUS"] == "PERSISTENT_CROSS_BENCHMARK"]
    collapsed = [k for k, v in status.items()
                 if v["STATUS"] == "COLLAPSED_ON_TERMINALBENCH"]
    excluded = [k for k, v in status.items()
                if not v["qualifies_for_primary_comparison"]]
    if excluded:
        overall = "INSUFFICIENT_CROSS_BENCHMARK_SUPPORT"
    elif len(persistent) == len(qualified) and qualified:
        overall = "BOTH_PERSIST"
    elif len(persistent) == 1:
        overall = "ONE_PERSISTS"
    elif len(collapsed) == len(qualified) and qualified:
        overall = "BOTH_COLLAPSE"
    else:
        overall = "INSUFFICIENT_CROSS_BENCHMARK_SUPPORT"
    gate = read_json(OUT_2B / "analysis" / "phase2b_gate.json")
    payload = {
        "section": "6/7 TERMINALBENCH STATUS + OVERALL DESCRIPTIVE RESULT",
        "primary_threshold": PRIMARY_THRESHOLD,
        "criteria": {
            "PERSISTENT_CROSS_BENCHMARK": "TB decisions >= %d AND |TB "
                "corrected gap| >= %.2f AND TB bootstrap CI excludes 0 AND "
                "TB gap sign matches SWE" % (MIN_DECISIONS, PERSIST_GAP),
            "COLLAPSED_ON_TERMINALBENCH": "TB decisions >= %d AND |TB "
                "corrected gap| < %.2f AND TB bootstrap CI includes 0"
                % (MIN_DECISIONS, COLLAPSE_GAP),
            "SIGN_CHANGED": "TB decisions >= %d AND the robust corrected gap "
                "sign is opposite the SWE sign" % MIN_DECISIONS,
            "INDETERMINATE": "TB decisions < %d or identity not confirmed or "
                "bootstrap inference unavailable" % MIN_DECISIONS,
            "UNCLASSIFIED_BY_SPEC": "qualified target matches none of the "
                "four clauses above; treated as not persistent and reported "
                "as a deviation",
        },
        "targets": status,
        "qualified_targets": sorted(qualified),
        "excluded_targets": sorted(excluded),
        "persistent_targets": sorted(persistent),
        "collapsed_targets": sorted(collapsed),
        "OVERALL_RESULT": overall,
        "allowed_labels": ["BOTH_PERSIST", "ONE_PERSISTS", "BOTH_COLLAPSE",
                           "INSUFFICIENT_CROSS_BENCHMARK_SUPPORT"],
        "PHASE2B_RESULT_frozen": gate["PHASE2B_RESULT"],
        "phase2b_primary_result_modified": False,
        "descriptive_only": True,
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0, "predictor_retraining": 0,
        "new_lightgbm_training": 0, "new_trajectories": 0, "cloud_compute": 0,
    }
    write_json(ANA / "target_status.json", as_builtin(payload))
    for k, v in status.items():
        print("%s %s/%s identity=%s STATUS=%s (TB decisions=%s, TB gap=%s)"
              % (k, v["swe_model"], v["head"], v["identity"], v["STATUS"],
                 v["clauses"]["tb_decisions"], v["tb_signed_corrected_gap"]),
              flush=True)
        for thr, tv in sorted(v["per_threshold_status"].items()):
            print("    %.3f -> %s (decisions=%s, gap=%s)"
                  % (tv["threshold"], tv["STATUS"],
                     tv["clauses"]["tb_decisions"],
                     tv["clauses"]["abs_corrected_gap"]), flush=True)
    print("OVERALL_RESULT = %s" % overall, flush=True)
    write_json(WORK / "status_summary.json", as_builtin(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
