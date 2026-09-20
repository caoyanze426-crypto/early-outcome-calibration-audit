# -*- coding: utf-8 -*-
"""Phase 0B step 3c: assemble the reference-free design matrix from dense + SVD blocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from common import OUT, WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

import step3_features as S  # noqa: E402

FE_DIR = WORK / "features"
FEAT_DIR = OUT / "features"


def main() -> int:
    ensure_dirs(FE_DIR, FEAT_DIR)
    df = pd.read_parquet(FE_DIR / "dense_meta.parquet")
    n_rows = len(df)
    num_cols, bool_cols, cat_cols = S.NUM, S.BOOLS, S.CATS

    num_df = df[num_cols].copy()
    for c in num_df.columns:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(-1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    num_vals = scaler.fit_transform(num_df.values.astype(np.float32)).astype(np.float32)
    del num_df

    bool_vals = df[bool_cols].astype(float).fillna(0).values.astype(np.float32)

    encoders, onehots = {}, []
    for col in cat_cols:
        vals = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        le.fit(sorted(set(vals.tolist()) | {"__MISSING__"}))
        encoders[col] = le
        enc = le.transform(vals)
        oh = np.zeros((n_rows, len(le.classes_)), dtype=np.float32)
        oh[np.arange(n_rows), enc] = 1.0
        onehots.append(oh)
    del df

    parts = []
    names = (list(num_cols) + list(bool_cols)
             + [f"{c}__{cls}" for c in cat_cols for cls in encoders[c].classes_])
    parts.append(num_vals)
    parts.append(bool_vals)
    parts.extend(onehots)
    for block in TFIDF_ACTION_FEEDBACK:
        U = np.load(FE_DIR / f"{block}.svd.npy")
        if U.shape[0] != n_rows:
            raise SystemExit(f"row mismatch for {block}: {U.shape[0]} vs {n_rows}")
        parts.append(U.astype(np.float32))
        names.extend([f"{block}__svd_{i}" for i in range(U.shape[1])])
    X = np.hstack(parts).astype(np.float32)
    del parts
    np.save(FE_DIR / "design_matrix.npy", X)

    meta = pd.read_parquet(FE_DIR / "dense_meta.parquet")[S.META_COLS]
    meta["n_steps_total"] = meta["n_steps_total_for_weighting"]
    meta.to_parquet(FE_DIR / "row_meta.parquet", index=False, compression="zstd")

    write_json(FEAT_DIR / "feature_config.json", {
        "tfidf_level": "action_feedback",
        "base": "af",
        "predictor": "Tbl_NoReferenceFamily_LightGBM",
        "feature_families": ["behavioral_dense", "bool_status", "categorical", "textual_af_tfidf_svd"],
        "reference_family_removed": True,
        "removal_rule": ("work/.../model_holdout_shadow_valid_retrain.py::"
                         "_is_reference_family_feature == name.startswith('gold_')"),
        "removed_gold_numeric": [c for c in __import__(
            "feature_engineer").NUMERIC_FEATURES if c.startswith("gold_")],
        "kept_numeric": list(num_cols),
        "kept_bool": list(bool_cols),
        "kept_categorical": list(cat_cols),
        "categorical_classes": {c: list(encoders[c].classes_) for c in cat_cols},
        "tfidf_blocks": list(TFIDF_ACTION_FEEDBACK),
        "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
        "tfidf_min_df": int(config.TFIDF_MIN_DF),
        "tfidf_max_features": int(config.TFIDF_MAX_FEATURES),
        "svd_dim_per_block": int(config.TFIDF_SVD_DIM_PER_BLOCK),
        "svd_random_state": int(config.TFIDF_SVD_RANDOM_STATE),
        "dense_standardize": bool(config.DENSE_STANDARDIZE),
        "include_model_id": False,
        "design_matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "dense_block_width": int(num_vals.shape[1] + bool_vals.shape[1]
                                 + sum(oh.shape[1] for oh in onehots)),
    })
    (FE_DIR / "feature_names.json").write_text(json.dumps(names), "utf-8")
    print(json.dumps({"rows": int(X.shape[0]), "cols": int(X.shape[1]),
                      "n_dense": len(names) - 5 * config.TFIDF_SVD_DIM_PER_BLOCK,
                      "bytes": int(X.nbytes)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
