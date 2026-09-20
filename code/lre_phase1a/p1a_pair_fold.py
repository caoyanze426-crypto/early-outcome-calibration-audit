# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A - one leave-TWO-agent-out pair fold.

Faithful port of the frozen Phase 0B lineage (`step4_fold_features.py` +
`step5_train_fold.py`) with the single intended design change:

    9 training agents + 1 held-out   ->   8 training agents + 2 held-out

Everything else is reused exactly: the global encoder artifacts, the reference-free
`action_feedback` TF-IDF level, `min_df`/`max_features`/ngram/SVD settings, the
per-instance TRAIN/VALID split algorithm and seed, the LightGBM hyperparameters
(including the repository's CPU fallback), the dual-head architecture, the
valid-only sigmoid calibration, the frozen 0.95/0.95 dual policy, and the safe-label
step. No hyperparameter was retuned for Phase 1A.

Exactly one predictor/calibrator is trained per pair fold and it scores BOTH
held-out agents.

Offline only: what this script may call is local LightGBM / numpy / scipy code.
LLM calls = 0, API calls = 0, new trajectories = 0.
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

from common1a import (ENC_DIR, NL, PRED_DEC, PRED_PAIR, PREFIX_TABLE, PREDICTOR,
                      ROW_INDEX, SEED, WORK, as_builtin, ensure_dirs,
                      set_vendor_env, sha256_file, write_json)

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import config  # noqa: E402
from feature_engineer import (  # noqa: E402
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TFIDF_ACTION_FEEDBACK,
)
from fe_lib import StreamTfidfFitter  # noqa: E402
from probability_calibration import (  # noqa: E402
    calibration_summary_row,
    fit_sigmoid_calibrator,
)
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize  # noqa: E402
from trainer import save_model, train_lightgbm  # noqa: E402

from earlyeval.core.contracts import PolicySpec  # noqa: E402
from earlyeval.policies.safe_stop import apply_policy  # noqa: E402

BLOCKS = list(TFIDF_ACTION_FEEDBACK.keys())
MIN_TRAJ_STEPS = int(config.MIN_TRAJECTORY_STEPS)
VALID_MODELS_PER_INSTANCE = 3
SPLIT_SEED = int(config.SPLIT_SEED)
SAFE_LABEL_MIN_STEP = 10
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

META = ["prefix_id", "traj_id", "instance_id", "model", "model_id",
        "prefix_step_idx", "label", "sample_weight",
        "n_steps_total_for_weighting"]


def is_reference_family_feature(name: str) -> bool:
    return name.startswith("gold_")


NUM = [c for c in NUMERIC_FEATURES if not is_reference_family_feature(c)]
BOOLS = [c for c in BOOL_FEATURES if not is_reference_family_feature(c)]
CATS = [c for c in CATEGORICAL_FEATURES if not is_reference_family_feature(c)]


class PeakRss:
    """Peak working-set of this process, in MiB (Windows)."""

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    def peak_mib(self):
        try:
            fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            fn.argtypes = [ctypes.wintypes.HANDLE,
                           ctypes.POINTER(self._PMC), ctypes.wintypes.DWORD]
            fn.restype = ctypes.wintypes.BOOL
            counters = self._PMC()
            counters.cb = ctypes.sizeof(self._PMC)
            ok = fn(ctypes.windll.kernel32.GetCurrentProcess(),
                    ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return round(counters.PeakWorkingSetSize / (1024.0 * 1024.0), 1)
        except Exception:  # noqa: BLE001 - measurement must never break the fold
            return None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def select_valid_model_pairs_per_instance(trainval, models_per_instance, seed):
    """Exact mirror of the vendored repository helper used by Phase 0B."""
    model_meta = (
        trainval[["instance_id", "model_id"]]
        .drop_duplicates()
        .assign(instance_id=lambda f: f["instance_id"].astype(str),
                model_id=lambda f: f["model_id"].astype(str))
    )
    rng = np.random.default_rng(seed + 3571)
    valid_pairs = set()
    for instance_id, part in model_meta.groupby("instance_id", sort=False):
        model_ids = part["model_id"].to_numpy()
        n_models = len(model_ids)
        if n_models <= 1:
            continue
        n_valid = min(models_per_instance, n_models - 1)
        chosen = rng.choice(model_ids, size=n_valid, replace=False).tolist()
        valid_pairs.update((str(instance_id), str(m)) for m in chosen)
    return valid_pairs


def build_split(row_index, holdout_models):
    """Phase 0B `build_split`, generalized to a two-agent holdout set."""
    model = row_index["model_id"].astype(str).to_numpy()
    traj = row_index["traj_id"].astype(str).to_numpy()
    pos = np.arange(len(row_index))
    is_holdout = np.isin(model, np.asarray(sorted(holdout_models), dtype=object))
    test_pos = pos[is_holdout]
    trainval_pos = pos[~is_holdout]
    tv = row_index.iloc[trainval_pos]
    first_steps = tv.groupby("traj_id")["n_steps_total_for_weighting"].first()
    short_ids = set(first_steps.loc[first_steps < MIN_TRAJ_STEPS].index)
    dropped_rows = 0
    if short_ids:
        keep = ~pd.Series(traj[trainval_pos]).isin(short_ids).to_numpy()
        dropped_rows = int((~keep).sum())
        trainval_pos = trainval_pos[keep]
    pool = row_index.iloc[trainval_pos]
    pairs = select_valid_model_pairs_per_instance(pool, VALID_MODELS_PER_INSTANCE,
                                                 SPLIT_SEED)
    pair_index = pd.MultiIndex.from_frame(pool[["instance_id", "model_id"]].astype(str))
    valid_mask = np.asarray(pair_index.isin(pairs))
    valid_pos = trainval_pos[valid_mask]
    train_pos = trainval_pos[~valid_mask]
    return {
        "train": train_pos, "valid": valid_pos, "test": test_pos,
        "short_trajectories_dropped_from_trainval": len(short_ids),
        "rows_dropped_from_trainval": dropped_rows,
        "valid_model_pairs": len(pairs),
    }


def load_global_csr(block):
    files = sorted(ENC_DIR.glob(f"gcsr.{block}.s*.npz"))
    mats = [sp.load_npz(f) for f in files]
    X = sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]
    del mats
    gc.collect()
    return X.tocsr()


def load_global_vocab(block):
    with open(ENC_DIR / f"gvocab.{block}.pkl", "rb") as fh:
        vocab = pickle.load(fh)["vocabulary"]
    n = len(vocab)
    terms_by_gid = [None] * n
    for term, gid in vocab.items():
        terms_by_gid[gid] = term
    return vocab, terms_by_gid, n


def load_dense_frame():
    import pyarrow.parquet as pq

    parts = sorted(PREFIX_TABLE.glob("prefix_table.part-*.parquet"))
    cols = list(NUM) + list(BOOLS) + list(CATS)
    frames = []
    for p in parts:
        tbl = pq.ParquetFile(p).read(columns=cols)
        frames.append(tbl.to_pandas())
        del tbl
    df = pd.concat(frames, ignore_index=True)
    del frames
    return df


def build_count_matrix(X_g, mapper, sink, positions, n_fold):
    S = X_g[positions]
    np.take(mapper, S.indices, out=S.indices)
    S.indices[S.indices < 0] = sink
    C = sp.csr_matrix((S.data, S.indices, S.indptr),
                      shape=(S.shape[0], n_fold + 1))
    del S
    return C


def apply_tfidf_inplace(X, idf, chunk_rows=16384):
    n_rows = X.shape[0]
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        lo, hi = int(X.indptr[start]), int(X.indptr[end])
        if hi <= lo:
            continue
        data = X.data[lo:hi]
        np.log(data, out=data)
        data += 1.0
        data *= idf.take(X.indices[lo:hi])
    normalize(X, norm="l2", copy=False)
    return X


def fold_block(block, column, load_x, n_glob, terms_by_gid, gid_of, pos_map):
    t0 = time.time()
    X_g = load_x()
    if X_g.shape[1] != n_glob:
        raise SystemExit(f"vocab/CSR width mismatch for {block}: "
                         f"{X_g.shape[1]} vs {n_glob}")
    sub = X_g[pos_map["train"]]
    n_train = sub.shape[0]
    df = np.bincount(sub.indices, minlength=n_glob)
    tf = np.zeros(n_glob, dtype=np.float64)
    chunk = 1 << 24
    for i in range(0, int(sub.nnz), chunk):
        idx = sub.indices[i:i + chunk]
        tf += np.bincount(idx, weights=sub.data[i:i + chunk].astype(np.float64),
                          minlength=n_glob)
    del sub
    gc.collect()
    present = np.flatnonzero(df > 0)
    fit = StreamTfidfFitter(block, column, config.TFIDF_NGRAM_RANGE,
                            config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
    fit.df = {terms_by_gid[int(i)]: int(df[i]) for i in present}
    fit.tf = {terms_by_gid[int(i)]: float(tf[i]) for i in present}
    fit.n_docs = int(n_train)
    fit.build_vocabulary()
    n_fold = len(fit.vocabulary_)
    vec = fit.make_vectorizer()
    mapper = np.full(n_glob, -1, dtype=np.int32)
    for term, fid in fit.vocabulary_.items():
        mapper[gid_of[term]] = fid
    sink = n_fold
    idf = np.concatenate([np.asarray(fit.idf_, dtype=np.float32),
                          np.zeros(1, dtype=np.float32)])
    del vec
    counts = {}
    for name in ("train", "valid", "test"):
        counts[name] = build_count_matrix(X_g, mapper, sink, pos_map[name], n_fold)
    del X_g, mapper
    gc.collect()
    transformed = {}
    for name in ("train", "valid", "test"):
        transformed[name] = apply_tfidf_inplace(counts[name], idf)
        del counts[name]
        gc.collect()
    n_comp = int(min(config.TFIDF_SVD_DIM_PER_BLOCK, n_fold - 1, n_train - 1))
    svd = TruncatedSVD(n_components=n_comp,
                       random_state=config.TFIDF_SVD_RANDOM_STATE)
    U = {"train": svd.fit_transform(transformed["train"]).astype(np.float32)}
    U["valid"] = svd.transform(transformed["valid"]).astype(np.float32)
    U["test"] = svd.transform(transformed["test"]).astype(np.float32)
    meta = {
        "block": block, "text_column": column, "train_rows": int(n_train),
        "global_terms": int(n_glob), "fold_terms": int(n_fold),
        "n_components": int(n_comp),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    del transformed, svd
    gc.collect()
    return U, meta


def build_dense(df, pos_map):
    num_df = df[NUM].copy()
    for c in num_df.columns:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(-1)
    num_vals = num_df.values.astype(np.float32)
    del num_df
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(num_vals[pos_map["train"]])
    bool_df = df[BOOLS].copy()
    for c in bool_df.columns:
        bool_df[c] = bool_df[c].astype(float).fillna(0)
    bool_vals = bool_df.values.astype(np.float32)
    del bool_df
    encoders, onehots, cat_names = {}, [], []
    for col in CATS:
        vals = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        le.fit(sorted(set(vals.iloc[pos_map["train"]].tolist()) | {"__MISSING__"}))
        encoders[col] = le
        known = set(le.classes_)
        unk = "__MISSING__" if "__MISSING__" in known else le.classes_[0]
        mapped = vals.apply(lambda x: x if x in known else unk)
        enc = le.transform(mapped)
        oh = np.zeros((len(df), len(le.classes_)), dtype=np.float32)
        oh[np.arange(len(df)), enc] = 1.0
        onehots.append(oh)
        cat_names.extend([f"{col}__{cls}" for cls in le.classes_])
    out = {}
    for name in ("train", "valid", "test"):
        p = pos_map[name]
        nv = scaler.transform(num_vals[p]).astype(np.float32)
        parts = [nv, bool_vals[p]] + [oh[p] for oh in onehots]
        out[name] = np.hstack(parts).astype(np.float32)
        del parts, nv
    feature_names = list(NUM) + list(BOOLS) + cat_names
    return out, feature_names, scaler, encoders


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


def fit_lgbm_with_cpu_fallback(X_train, y_train, w_train, X_valid, y_valid, w_valid,
                               feature_names, model_name):
    """Repository contract: retry once with device=cpu, then restore the params."""
    original_params = dict(config.LGBM_PARAMS)
    try:
        return train_lightgbm(X_train=X_train, y_train=y_train, w_train=w_train,
                              X_valid=X_valid, y_valid=y_valid, w_valid=w_valid,
                              feature_names=feature_names, model_name=model_name)
    except Exception as exc:  # noqa: BLE001
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


def main() -> int:
    pair_id = sys.argv[1]
    agent_a = sys.argv[2]
    agent_b = sys.argv[3]
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(n_threads)

    work_dir = WORK / "tmp" / pair_id
    ensure_dirs(work_dir, PRED_PAIR, PRED_DEC)
    peak = PeakRss()
    t_start = time.time()
    started = utc_now()
    single_holdout = agent_b == "-"
    holdout = {agent_a} if single_holdout else {agent_a, agent_b}
    holdout_agents = sorted(holdout)
    row_index = pd.read_parquet(ROW_INDEX)
    dense_frame = load_dense_frame()
    if len(dense_frame) != len(row_index):
        raise SystemExit(f"prefix_table rows {len(dense_frame)} != "
                         f"row_index rows {len(row_index)}")
    split = build_split(row_index, holdout)
    pos_map = {k: split[k] for k in ("train", "valid", "test")}
    split_info = {
        "pair_id": pair_id, "agent_A": agent_a, "agent_B": agent_b,
        "rows_total": int(len(row_index)),
        "train_rows": int(len(pos_map["train"])),
        "valid_rows": int(len(pos_map["valid"])),
        "test_rows": int(len(pos_map["test"])),
        "train_agents": sorted(set(
            row_index.iloc[pos_map["train"]]["model_id"].astype(str))),
        "valid_agents": sorted(set(
            row_index.iloc[pos_map["valid"]]["model_id"].astype(str))),
        "test_rows_A": int((row_index.iloc[pos_map["test"]]["model_id"]
                            .astype(str) == agent_a).sum()),
        "test_rows_B": int((row_index.iloc[pos_map["test"]]["model_id"]
                            .astype(str) == agent_b).sum()),
        "short_trajectories_dropped_from_trainval":
            split["short_trajectories_dropped_from_trainval"],
        "rows_dropped_from_trainval": split["rows_dropped_from_trainval"],
        "valid_model_pairs": split["valid_model_pairs"],
    }
    print(json.dumps(split_info), flush=True)

    dense, dense_names, scaler, encoders = build_dense(dense_frame, pos_map)
    del dense_frame
    gc.collect()
    block_meta = {}
    for block, column in TFIDF_ACTION_FEEDBACK.items():
        gid_of, terms_by_gid, n_glob = load_global_vocab(block)
        U, meta = fold_block(block, column, lambda b=block: load_global_csr(b),
                             n_glob, terms_by_gid, gid_of, pos_map)
        del gid_of, terms_by_gid
        gc.collect()
        block_meta[block] = meta
        print(json.dumps(meta), flush=True)
        for name in ("train", "valid", "test"):
            dense[name] = np.hstack([dense[name], U[name]]).astype(np.float32)
        dense_names.extend([f"{block}__svd_{i}"
                            for i in range(U["train"].shape[1])])
        del U
        gc.collect()
    feature_seconds = round(time.time() - t_start, 1)

    meta = {name: row_index.iloc[pos_map[name]][META].copy()
            for name in ("train", "valid", "test")}
    for name in ("train", "valid", "test"):
        meta[name]["row_position"] = pos_map[name]
        meta[name].to_parquet(work_dir / f"meta_{name}.parquet", index=False,
                              compression="zstd")
    X = dense
    y, w = {}, {}
    for name in ("train", "valid", "test"):
        ys, yf = safe_targets(meta[name]["prefix_step_idx"], meta[name]["label"],
                              SAFE_LABEL_MIN_STEP)
        y[name] = {"success": ys, "failure": yf}
        w[name] = meta[name]["sample_weight"].to_numpy(dtype=np.float32)
    feature_names = dense_names

    cal_rows = []
    status = {}
    preds = {}
    for name in ("valid", "test"):
        frame = meta[name][["prefix_id", "traj_id", "instance_id", "model_id",
                            "prefix_step_idx", "label",
                            "n_steps_total_for_weighting"]].copy()
        preds[name] = frame.rename(
            columns={"n_steps_total_for_weighting": "n_steps_total"})
    for head, column_prefix in (("safe_success", "success"),
                                ("safe_failure", "failure")):
        model_name = f"{PREDICTOR}__{head}"
        try:
            booster = fit_lgbm_with_cpu_fallback(
                X_train=X["train"], y_train=y["train"][column_prefix],
                w_train=w["train"], X_valid=X["valid"],
                y_valid=y["valid"][column_prefix], w_valid=w["valid"],
                feature_names=feature_names, model_name=model_name)
            save_model(booster, work_dir / f"{model_name}.lgb")
            valid_raw = np.asarray(booster.predict(X["valid"]), dtype=np.float64)
            test_raw = np.asarray(booster.predict(X["test"]), dtype=np.float64)
            calibrator = fit_sigmoid_calibrator(
                valid_raw, y["valid"][column_prefix], sample_weight=w["valid"])
            with open(work_dir / f"calibrator_{model_name}.pkl", "wb") as fh:
                pickle.dump(calibrator, fh)
            valid_cal = calibrator.predict(valid_raw)
            test_cal = calibrator.predict(test_raw)
            cal_rows.append({
                "head": head,
                "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
                "n_features": int(X["train"].shape[1]),
                **calibration_summary_row(
                    model_name=model_name, calibrator=calibrator,
                    y_valid=y["valid"][column_prefix], raw_prob_valid=valid_raw,
                    y_test=y["test"][column_prefix], raw_prob_test=test_raw),
            })
            preds["test"][head_column(column_prefix, "raw", PREDICTOR)] = \
                test_raw.astype(np.float32)
            preds["test"][head_column(column_prefix, "calibrated", PREDICTOR)] = \
                test_cal.astype(np.float32)
            preds["valid"][head_column(column_prefix, "raw", PREDICTOR)] = \
                valid_raw.astype(np.float32)
            preds["valid"][head_column(column_prefix, "calibrated", PREDICTOR)] = \
                valid_cal.astype(np.float32)
            status[f"{head}_head"] = "OK"
            status["calibration"] = "OK"
            del booster
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - record FOLD_FAIL, never repair
            status[f"{head}_head"] = f"FAIL:{type(exc).__name__}:{exc}"
            status["calibration"] = "NOT_REACHED"
            raise
    train_seconds = round(time.time() - t_start - feature_seconds, 1)

    test_pred = preds["test"]
    policy = PolicySpec(name="current_safe_stop", predictor=PREDICTOR,
                        score_mode="calibrated", policy_mode="dual",
                        success_thr=0.95, failure_thr=0.95, min_step=0,
                        consecutive=1)
    decisions, summary, per_agent = apply_policy(test_pred, policy)
    decisions["pair_id"] = pair_id
    per_agent["pair_id"] = pair_id

    test_pred.to_parquet(work_dir / "test_prefix_predictions.parquet", index=False,
                         compression="zstd")
    preds["valid"].to_parquet(work_dir / "valid_prefix_predictions.parquet",
                              index=False, compression="zstd")
    decisions.to_csv(work_dir / "pair_policy_decisions.csv", index=False,
                     encoding="utf-8")
    for agent in holdout_agents:
        sub = test_pred[test_pred["model_id"].astype(str) == agent]
        sub.to_parquet(PRED_PAIR / f"{pair_id}.{agent}.parquet", index=False,
                       compression="zstd")
    decisions.to_csv(PRED_DEC / f"{pair_id}.csv", index=False, encoding="utf-8")
    per_agent.to_csv(PRED_DEC / f"{pair_id}.per_agent.csv", index=False,
                     encoding="utf-8")

    write_json(work_dir / "fold_meta.json", as_builtin({
        "pair_id": pair_id, "agent_A": agent_a,
        "agent_B": None if single_holdout else agent_b,
        "single_holdout_parity_mode": single_holdout,
        "predictor": PREDICTOR,
        "started_utc": started, "ended_utc": utc_now(),
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "feature_seconds": feature_seconds,
        "train_seconds": train_seconds,
        "peak_working_set_mib": peak.peak_mib(),
        "threads": n_threads,
        "split": split_info,
        "safe_label_min_step": SAFE_LABEL_MIN_STEP,
        "lgbm_params": dict(config.LGBM_PARAMS),
        "design_matrix_shape": {name: [int(X[name].shape[0]),
                                       int(X[name].shape[1])]
                                for name in ("train", "valid", "test")},
        "positive_rates": {
            name: {h: float(y[name][h].mean()) for h in ("success", "failure")}
            for name in y},
        "blocks": block_meta,
        "calibration": cal_rows,
        "head_status": status,
        "fold_status": "OK",
    }))
    print(json.dumps({"pair_id": pair_id, "status": "OK",
                      "seconds": round(time.time() - t_start, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
