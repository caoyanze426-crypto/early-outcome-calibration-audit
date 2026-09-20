# -*- coding: utf-8 -*-
"""Phase 0B step 5: one fold's dual-head LightGBM + validation-only sigmoid calibration.

Mirrors safe_stop_dual_head_retrain.py: _safe_targets -> _fit_lgbm_with_cpu_fallback
-> fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid).
Offline only: no LLM / model API calls.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time

from common import WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from probability_calibration import (  # noqa: E402
    calibration_summary_row,
    fit_sigmoid_calibrator,
)
from trainer import save_model, train_lightgbm  # noqa: E402

FOLD_DIR = WORK / "folds"
PREDICTOR = "P2B_TerminalBench_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SAFE_LABEL_MIN_STEP = 10
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def safe_targets(prefix_step_idx, label, min_step):
    labels = np.asarray(label, dtype=int)
    steps = np.asarray(prefix_step_idx, dtype=int)
    eligible = steps >= int(min_step)
    return (((labels == 1) & eligible).astype(int),
            ((labels == 0) & eligible).astype(int))


def head_column(head, score_mode, predictor):
    if score_mode == "raw":
        return f"prob_safe_{head}__{predictor}"
    if score_mode == "calibrated":
        return f"prob_cal_safe_{head}__{predictor}"
    raise ValueError(score_mode)


def fit_lgbm_with_cpu_fallback(X_train, y_train, w_train, X_valid, y_valid,
                               w_valid, feature_names, model_name):
    original_params = dict(config.LGBM_PARAMS)
    try:
        return train_lightgbm(X_train=X_train, y_train=y_train, w_train=w_train,
                              X_valid=X_valid, y_valid=y_valid, w_valid=w_valid,
                              feature_names=feature_names, model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - repository fallback contract
        print(f"[{model_name}] LightGBM failed with current params: {exc}",
              flush=True)
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)
        try:
            return train_lightgbm(X_train=X_train, y_train=y_train,
                                  w_train=w_train, X_valid=X_valid,
                                  y_valid=y_valid, w_valid=w_valid,
                                  feature_names=feature_names,
                                  model_name=f"{model_name}_cpu")
        finally:
            config.LGBM_PARAMS.clear()
            config.LGBM_PARAMS.update(original_params)


def main():
    fold_tag = sys.argv[1]
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(n_threads)
    fold_dir = FOLD_DIR / fold_tag
    model_dir = fold_dir / "models"
    ensure_dirs(model_dir)
    t0 = time.time()
    meta = {name: pd.read_parquet(fold_dir / f"meta_{name}.parquet")
            for name in ("train", "valid", "test")}
    X = {name: np.load(fold_dir / f"X_{name}.npy") for name in
         ("train", "valid", "test")}
    y = {}
    w = {}
    for name in ("train", "valid", "test"):
        ys, yf = safe_targets(meta[name]["prefix_step_idx"], meta[name]["label"],
                              SAFE_LABEL_MIN_STEP)
        y[name] = {"success": ys, "failure": yf}
        w[name] = meta[name]["sample_weight"].to_numpy(dtype=np.float32)
    feature_names = json.loads((fold_dir / "feature_names.json").read_text("utf-8"))
    assert len(feature_names) == X["train"].shape[1], "feature name width mismatch"

    calibration_rows = []
    pred = meta["test"][["prefix_id", "traj_id", "instance_id", "model_id",
                         "prefix_step_idx", "label",
                         "n_steps_total_for_weighting"]].copy()
    pred = pred.rename(columns={"n_steps_total_for_weighting": "n_steps_total"})
    valid_pred = meta["valid"][["prefix_id", "traj_id", "instance_id", "model_id",
                                "prefix_step_idx", "label",
                                "n_steps_total_for_weighting"]].copy()
    valid_pred = valid_pred.rename(
        columns={"n_steps_total_for_weighting": "n_steps_total"})
    for head, column_prefix in (("safe_success", "success"),
                                ("safe_failure", "failure")):
        model_name = f"{PREDICTOR}__{head}"
        booster = fit_lgbm_with_cpu_fallback(
            X_train=X["train"], y_train=y["train"][column_prefix],
            w_train=w["train"], X_valid=X["valid"],
            y_valid=y["valid"][column_prefix], w_valid=w["valid"],
            feature_names=feature_names, model_name=model_name)
        save_model(booster, model_dir / f"{model_name}.lgb")
        valid_raw = np.asarray(booster.predict(X["valid"]), dtype=np.float64)
        test_raw = np.asarray(booster.predict(X["test"]), dtype=np.float64)
        calibrator = fit_sigmoid_calibrator(valid_raw, y["valid"][column_prefix],
                                            sample_weight=w["valid"])
        with open(model_dir / f"calibrator_{model_name}.pkl", "wb") as fh:
            pickle.dump(calibrator, fh)
        valid_cal = calibrator.predict(valid_raw)
        test_cal = calibrator.predict(test_raw)
        calibration_rows.append({
            "head": head,
            "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
            "n_features": int(X["train"].shape[1]),
            **calibration_summary_row(
                model_name=model_name, calibrator=calibrator,
                y_valid=y["valid"][column_prefix], raw_prob_valid=valid_raw,
                y_test=y["test"][column_prefix], raw_prob_test=test_raw),
        })
        pred[head_column(column_prefix, "raw", PREDICTOR)] = test_raw.astype(np.float32)
        pred[head_column(column_prefix, "calibrated", PREDICTOR)] = \
            test_cal.astype(np.float32)
        valid_pred[head_column(column_prefix, "raw", PREDICTOR)] = \
            valid_raw.astype(np.float32)
        valid_pred[head_column(column_prefix, "calibrated", PREDICTOR)] = \
            valid_cal.astype(np.float32)
        print(json.dumps({"fold": fold_tag, "head": head,
                          "best_iteration": calibration_rows[-1]["best_iteration"],
                          "seconds": round(time.time() - t0, 1)}), flush=True)
        del booster
        gc.collect()

    pred.to_parquet(fold_dir / "test_prefix_predictions.parquet", index=False,
                    compression="zstd")
    valid_pred.to_parquet(fold_dir / "valid_prefix_predictions.parquet",
                          index=False, compression="zstd")
    write_json(fold_dir / "calibration_meta.json", {
        "fold": fold_tag,
        "predictor": PREDICTOR,
        "safe_label_min_step": SAFE_LABEL_MIN_STEP,
        "safe_label_min_step_source": "repository CLI default --safe-label-min-step 10",
        "lgbm_params": dict(config.LGBM_PARAMS),
        "lgbm_params_note": ("repository config.LGBM_PARAMS; the repository's own "
                             "_fit_lgbm_with_cpu_fallback retries on device=cpu, which "
                             "is what happened here (no GPU build available)"),
        "threads": n_threads,
        "rows": {name: int(len(X[name])) for name in X},
        "positive_rates": {
            name: {h: float(y[name][h].mean()) for h in ("success", "failure")}
            for name in y},
        "calibration": calibration_rows,
        "seconds": round(time.time() - t0, 1),
    })
    print(json.dumps({"fold": fold_tag, "done_seconds": round(time.time() - t0, 1)}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
