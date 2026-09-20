# -*- coding: utf-8 -*-
"""Phase 0B step 4: one leave-one-model-out fold, fold-local reference-free features.

Mirrors the repository's `--fit-feature-engineer-on-train` contract: the TF-IDF
vocabulary, idf, SVD basis, numeric scaler and label encoders are fitted on the
fold's TRAIN split only. The held-out model contributes no fitted parameter.
Offline only: no LLM / model API calls.
"""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time

from common import WORK, ensure_dirs, set_vendor_env, write_json

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
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

ENC_DIR = WORK / "encoded"
FOLD_DIR = WORK / "folds"
BLOCKS = list(TFIDF_ACTION_FEEDBACK.keys())
SEED = int(config.SPLIT_SEED)
MIN_TRAJ_STEPS = int(config.MIN_TRAJECTORY_STEPS)
VALID_MODELS_PER_INSTANCE = 3

META = ["prefix_id", "traj_id", "instance_id", "model", "model_id",
        "prefix_step_idx", "label", "sample_weight",
        "n_steps_total_for_weighting"]


def is_reference_family_feature(name: str) -> bool:
    """repository rule: model_holdout_shadow_valid_retrain._is_reference_family_feature"""
    return name.startswith("gold_")


NUM = [c for c in NUMERIC_FEATURES if not is_reference_family_feature(c)]
BOOLS = [c for c in BOOL_FEATURES if not is_reference_family_feature(c)]
CATS = [c for c in CATEGORICAL_FEATURES if not is_reference_family_feature(c)]


def select_valid_model_pairs_per_instance(trainval, models_per_instance, seed):
    """Exact mirror of _select_valid_model_pairs_per_instance in the vendored repo."""
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


def build_split(row_index, holdout_model):
    model = row_index["model_id"].astype(str).to_numpy()
    traj = row_index["traj_id"].astype(str).to_numpy()
    pos = np.arange(len(row_index))
    is_holdout = model == holdout_model
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
    pairs = select_valid_model_pairs_per_instance(pool, VALID_MODELS_PER_INSTANCE, SEED)
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
    """Read the reference-free dense feature columns from the prefix table.

    The prefix table parts are concatenated in the same sorted order used to
    build `row_index.parquet`, so row positions line up exactly.
    """
    import pyarrow.parquet as pq

    parts = sorted((WORK / "prefix_table").glob("prefix_table.part-*.parquet"))
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
    """float32 equivalent of TfidfTransformer(sublinear_tf, use_idf, norm='l2').transform.

    Same operations in the same order as sklearn, executed in place so that a block
    with hundreds of millions of non-zeros does not require a float64 duplicate.
    """
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
    """Fit the fold-local TF-IDF + SVD for one block and transform train/valid/test."""
    t0 = time.time()
    X_g = load_x()
    if X_g.shape[1] != n_glob:
        raise SystemExit(f"vocab/CSR width mismatch for {block}: "
                         f"{X_g.shape[1]} vs {n_glob}")
    sub = X_g[pos_map["train"]]
    n_train = sub.shape[0]
    df = np.bincount(sub.indices, minlength=n_glob)
    # Term frequencies are integer-valued; accumulate in chunks so the whole
    # 534M-nnz block does not need a full float64 duplicate of its data array.
    # Integer sums below 2**53 are exact regardless of accumulation order, so
    # this matches the single-shot bincount bitwise.
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
        "block": block,
        "text_column": column,
        "train_rows": int(n_train),
        "global_terms": int(n_glob),
        "fold_terms": int(n_fold),
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


def main():
    holdout_model = sys.argv[1]
    fold_tag = sys.argv[2]
    out_dir = FOLD_DIR / fold_tag
    ensure_dirs(out_dir)
    t_start = time.time()
    row_index = pd.read_parquet(WORK / "row_index.parquet")
    dense_frame = load_dense_frame()
    if len(dense_frame) != len(row_index):
        raise SystemExit(f"prefix_table rows {len(dense_frame)} != "
                         f"row_index rows {len(row_index)}")
    split = build_split(row_index, holdout_model)
    pos_map = {k: split[k] for k in ("train", "valid", "test")}
    print(json.dumps({
        "fold": fold_tag, "holdout_model": holdout_model,
        "rows": int(len(row_index)),
        "train_rows": int(len(pos_map["train"])),
        "valid_rows": int(len(pos_map["valid"])),
        "test_rows": int(len(pos_map["test"])),
        "short_trajectories_dropped_from_trainval":
            split["short_trajectories_dropped_from_trainval"],
        "rows_dropped_from_trainval": split["rows_dropped_from_trainval"],
    }), flush=True)
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
    for name in ("train", "valid", "test"):
        np.save(out_dir / f"X_{name}.npy", dense[name])
        p = pos_map[name]
        sub = row_index.iloc[p][META].copy()
        sub["row_position"] = p
        sub.to_parquet(out_dir / f"meta_{name}.parquet", index=False,
                       compression="zstd")
    write_json(out_dir / "feature_meta.json", {
        "fold": fold_tag,
        "holdout_model": holdout_model,
        "split_strategy": "per_instance_model (repository default)",
        "seed": SEED,
        "valid_models_per_instance": VALID_MODELS_PER_INSTANCE,
        "min_trajectory_steps": MIN_TRAJ_STEPS,
        "feature_engineer_fit_on_train": True,
        "tfidf_level": "action_feedback",
        "tfidf_level_note": ("repository default is 'with_thought'; the thought text "
                             "columns are degenerate on this corpus, so the "
                             "reference-free TF-IDF family used is 'action_feedback'"),
        "reference_family_removed": True,
        "removal_rule": "name.startswith('gold_')",
        "design_matrix_shape": {name: [int(dense[name].shape[0]),
                                       int(dense[name].shape[1])]
                                for name in ("train", "valid", "test")},
        "dense_feature_count": len(dense_names)
        - 5 * int(config.TFIDF_SVD_DIM_PER_BLOCK),
        "blocks": block_meta,
        "seconds": round(time.time() - t_start, 1),
    })
    (out_dir / "feature_names.json").write_text(json.dumps(dense_names), "utf-8")
    print(json.dumps({"fold": fold_tag, "cols": int(dense["train"].shape[1]),
                      "seconds": round(time.time() - t_start, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
