# -*- coding: utf-8 -*-
"""Phase 0B verification: the fold-local reconstruction equals the repository's
FeatureEngineer on the same documents.

Runs entirely on one prefix_table part (no corpus-wide state). Offline only.
"""
from __future__ import annotations

import json
import sys

from common import WORK, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import feature_engineer as FE  # noqa: E402
import config  # noqa: E402

import step4_fold_features as S4  # noqa: E402
from fe_lib import StreamTfidfFitter  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

FE.NUMERIC_FEATURES = list(S4.NUM)
FE.BOOL_FEATURES = list(S4.BOOLS)
FE.CATEGORICAL_FEATURES = list(S4.CATS)

PART = WORK / "prefix_table" / "prefix_table.part-0000.parquet"
GVOCAB_MIN_DF = int(config.TFIDF_MIN_DF)


def build_mini_global(block, column, texts):
    fit = StreamTfidfFitter(block, column, config.TFIDF_NGRAM_RANGE,
                            GVOCAB_MIN_DF, None)
    fit.update(fit.analyze(texts))
    fit.build_vocabulary()
    vec = CountVectorizer(ngram_range=tuple(config.TFIDF_NGRAM_RANGE),
                          vocabulary=fit.vocabulary_, dtype=np.float32)
    X = sp.csr_matrix(vec.fit_transform(texts))
    return fit.vocabulary_, X, X.shape[1]


def main() -> int:
    import pyarrow.parquet as pq

    cols = ["prefix_id", "traj_id", "instance_id", "model", "model_id",
            "prefix_step_idx", "label", "sample_weight",
            "n_steps_total_for_weighting"] + S4.NUM + S4.BOOLS + S4.CATS + \
        list(FE.TFIDF_ACTION_FEEDBACK.values())
    tbl = pq.ParquetFile(PART).read(columns=cols)
    df = tbl.to_pandas()
    n = len(df)
    print(f"verification rows={n}", flush=True)
    all_pos = np.arange(n)
    pos_map = {"train": all_pos, "valid": all_pos, "test": all_pos}

    report = {"rows": int(n), "blocks": {}, "dense": {}, "svd": {}}

    reference = FE.FeatureEngineer(include_model_id=False,
                                   tfidf_level="action_feedback")
    reference.fit(df)
    report["reference_feature_engineer"] = {
        "tfidf_level": reference.tfidf_level,
        "n_dense_features": len(reference.dense_feature_names),
        "dense_feature_names": reference.dense_feature_names,
        "tfidf_block_dims": {
            name: (reference.tfidf_reducers[name].n_components
                   if name in reference.tfidf_reducers
                   else len(reference.tfidf_vectorizers[name].vocabulary_))
            for name in reference.active_text_columns},
    }

    mine_dense, mine_names, _, _ = S4.build_dense(df, pos_map)
    ref_dense = reference.transform_dense(df)
    report["dense"]["names_match"] = (mine_names == reference.dense_feature_names)
    report["dense"]["shape"] = [int(mine_dense["train"].shape[0]),
                                int(mine_dense["train"].shape[1])]
    report["dense"]["max_abs_diff"] = float(
        np.max(np.abs(mine_dense["train"] - ref_dense)))
    report["dense"]["allclose"] = bool(
        np.allclose(mine_dense["train"], ref_dense, atol=1e-5))

    svd_mine = {}
    for order, (block, column) in enumerate(FE.TFIDF_ACTION_FEEDBACK.items()):
        texts = ["" if v is None else str(v) for v in df[column].tolist()]
        gid_of, X_g, n_glob = build_mini_global(block, column, texts)
        terms_by_gid = [None] * n_glob
        for term, gid in gid_of.items():
            terms_by_gid[gid] = term
        U, meta = S4.fold_block(block, column, lambda: X_g, n_glob,
                                terms_by_gid, gid_of, pos_map)
        svd_mine[block] = U["train"]
        ref_vec = reference.tfidf_vectorizers[block]
        ref_counts = sp.csr_matrix(ref_vec.transform(texts))
        ref_reducer = reference.tfidf_reducers[block]
        ref_svd = sp.csr_matrix(reference.transform_tfidf_subset(df, [block]))
        # column congruence between the two SVD embeddings
        ref_dense_emb = np.asarray(ref_svd.todense(), dtype=np.float64)
        z = np.abs(np.corrcoef(U["train"].T.astype(np.float64),
                               ref_dense_emb.T)[:U["train"].shape[1],
                                                U["train"].shape[1]:])
        report["blocks"][block] = {
            "fold_terms": meta["fold_terms"],
            "reference_terms": int(ref_counts.shape[1]),
            "global_terms": int(n_glob),
            "n_components": meta["n_components"],
            "explained_variance_ratio_sum": meta["explained_variance_ratio_sum"],
            "reference_evr_sum": float(ref_reducer.explained_variance_ratio_.sum()),
            "svd_component_max_abs_correlation": float(z.max()),
            "svd_component_mean_abs_correlation": float(z.mean()),
        }
        del X_g
    write_json(WORK / "features" / "fold_feature_equivalence.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
