# -*- coding: utf-8 -*-
"""Record the Phase 0B end-to-end parity evidence, then clear parity scratch.

The parity fold ran the Phase 1A pipeline with a SINGLE holdout agent
(gpt-5-mini, i.e. the frozen Phase 0B fold-06 universe) so that its pipeline
outputs are directly comparable with the frozen Phase 0B fold-06 artifacts.
Bit-exact agreement proves the feature engineering, LightGBM training and the
sigmoid calibration were reproduced, not re-implemented.
"""
from __future__ import annotations

import json
import shutil
import sys

from common1a import (ANA, NL, OUT_0B, PREDICTOR, PRED_DEC, PRED_PAIR, WORK,
                      WORK_0B, as_builtin, ensure_dirs, write_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARITY_PAIR = "parity-fold06"
PARITY_AGENT = "gpt-5-mini"
PARITY_FOLD = "fold-06"


def main() -> int:
    ensure_dirs(ANA)
    a = pd.read_parquet(WORK / "tmp" / PARITY_PAIR /
                        "test_prefix_predictions.parquet")
    b = pd.read_parquet(OUT_0B / "folds" / PARITY_FOLD /
                        "test_prefix_predictions.parquet") \
        if (OUT_0B / "folds" / PARITY_FOLD).exists() else None
    if b is None:
        b = pd.read_parquet(WORK_0B / "folds" / PARITY_FOLD /
                            "test_prefix_predictions.parquet")
    a = a.sort_values("prefix_id").reset_index(drop=True)
    b = b.sort_values("prefix_id").reset_index(drop=True)
    cols = {
        "raw_success": f"prob_safe_success__{PREDICTOR}",
        "raw_failure": f"prob_safe_failure__{PREDICTOR}",
        "cal_success": f"prob_cal_safe_success__{PREDICTOR}",
        "cal_failure": f"prob_cal_safe_failure__{PREDICTOR}",
    }
    checks = {}
    all_exact = True
    for label, col in cols.items():
        d = np.abs(a[col].to_numpy(np.float64) - b[col].to_numpy(np.float64))
        max_abs = float(d.max()) if d.size else 0.0
        exact = bool(max_abs == 0.0)
        all_exact = all_exact and exact
        checks[label] = {
            "column": col, "max_abs_difference": max_abs,
            "n_rows_gt_1e-12": int((d > 1e-12).sum()),
            "bit_exact": exact,
        }
    ma = json.loads((WORK / "tmp" / PARITY_PAIR / "fold_meta.json").read_text(
        "utf-8"))
    cal_path = OUT_0B / "folds" / f"{PARITY_FOLD}.calibration_meta.json"
    if not cal_path.exists():
        cal_path = WORK_0B / "folds" / PARITY_FOLD / "calibration_meta.json"
    mb = json.loads(cal_path.read_text("utf-8"))
    payload = as_builtin({
        "check": "Phase 1A pipeline re-run with a SINGLE holdout agent "
                 "(gpt-5-mini) against the frozen Phase 0B fold-06 artifacts",
        "parity_mode": "single_holdout",
        "holdout_agent": PARITY_AGENT,
        "phase0b_fold": PARITY_FOLD,
        "n_rows": int(len(a)),
        "test_prefix_id_sets_identical": bool(set(a["prefix_id"]) ==
                                              set(b["prefix_id"])),
        "probability_checks": checks,
        "ALL_PROBABILITIES_BIT_EXACT": all_exact,
        "best_iteration_phase1a": {c["head"]: c["best_iteration"]
                                   for c in ma["calibration"]},
        "best_iteration_phase0b": {c["head"]: c["best_iteration"]
                                   for c in mb["calibration"]},
        "n_features_phase1a": ma["design_matrix_shape"]["train"][1],
        "n_features_phase0b": mb["rows"] and None,
        "feature_seconds_phase1a": ma["feature_seconds"],
        "peak_working_set_mib": ma.get("peak_working_set_mib"),
        "note": "the parity fold is engineering validation only and is excluded "
                "from every Phase 1A scientific artifact",
    })
    payload.pop("n_features_phase0b", None)
    write_json(ANA / "pipeline_parity_phase0b.json", payload)
    # Clear the parity scratch so it cannot be mistaken for a scientific fold.
    tmp = WORK / "tmp" / PARITY_PAIR
    if tmp.exists():
        shutil.rmtree(tmp)
    for p in list(PRED_PAIR.glob(f"{PARITY_PAIR}.*.parquet")) + \
            list(PRED_DEC.glob(f"{PARITY_PAIR}.*")):
        p.unlink()
    print(NL.join([f"{k}: max_abs={v['max_abs_difference']} "
                   f"bit_exact={v['bit_exact']}" for k, v in checks.items()]),
          flush=True)
    print(f"{NL}ALL_PROBABILITIES_BIT_EXACT = {all_exact}", flush=True)
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
