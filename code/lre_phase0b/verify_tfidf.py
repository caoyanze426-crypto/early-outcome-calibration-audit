# -*- coding: utf-8 -*-
"""Phase 0B: prove the streamed TF-IDF reconstruction matches sklearn exactly on the sample."""
from __future__ import annotations

import json
import sys

from common import OUT, set_vendor_env

set_vendor_env()

import numpy as np  # noqa: E402

import config  # noqa: E402
from fe_lib import StreamTfidfFitter  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

import pandas as pd  # noqa: E402
from mapping_adapter import adapt_messages  # noqa: E402
from prefix_builder import build_prefix_samples_for_trajectory  # noqa: E402

SAMPLES = OUT.parent / "late_reversal_early_eval_phase0a" / "public_dataset" / "samples"


def sample_rows(limit=30):
    rows = []
    for f in sorted(SAMPLES.rglob("*.traj.json"))[:limit]:
        doc = json.loads(f.read_text("utf-8"))
        info = doc.get("info") or {}
        model = (info.get("docent") or {}).get("model_label") or f.parent.name
        rows.extend(build_prefix_samples_for_trajectory(pd.Series({
            "traj_id": str(model) + "::" + str(doc.get("instance_id")),
            "instance_id": doc.get("instance_id"),
            "resolved": bool(info.get("resolved")),
            "model": str(model),
            "messages": json.dumps(adapt_messages(doc.get("messages") or []),
                                   ensure_ascii=False)})))
    return pd.DataFrame(rows)


def main() -> int:
    df = sample_rows(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    report = {"prefix_rows": len(df), "blocks": {}, "all_match": True}
    for name, col in TFIDF_ACTION_FEEDBACK.items():
        texts = df[col].fillna("").astype(str).tolist()
        ref = TfidfVectorizer(ngram_range=config.TFIDF_NGRAM_RANGE,
                              min_df=config.TFIDF_MIN_DF,
                              max_features=config.TFIDF_MAX_FEATURES,
                              sublinear_tf=True, dtype=np.float32)
        Xfit = ref.fit_transform(texts)
        fit = StreamTfidfFitter(name, col, config.TFIDF_NGRAM_RANGE,
                                config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
        for i in range(0, len(texts), 256):
            fit.update(fit.analyze(texts[i:i + 256]))
        fit.build_vocabulary()
        vec = fit.make_vectorizer()
        Xrec = vec.transform(texts)
        # The sklearn fit path relabels columns (map_index.take) without re-sorting
        # rows; after canonicalizing row order the fit and transform paths must agree.
        Xref = ref.transform(texts)
        Xfit.sort_indices()
        Xref.sort_indices()
        Xrec.sort_indices()
        fit_path_equiv = (abs(Xfit - Xref).nnz == 0)
        same_vocab = ref.vocabulary_ == fit.vocabulary_
        same_idf = bool(np.array_equal(ref._tfidf.idf_, fit.idf_))
        same_shape = Xref.shape == Xrec.shape
        same_nnz = int(Xref.nnz) == int(Xrec.nnz)
        same_data = bool(np.array_equal(Xref.tocsr().data, Xrec.tocsr().data))
        same_indices = bool(np.array_equal(Xref.tocsr().indices, Xrec.tocsr().indices))
        same_indptr = bool(np.array_equal(Xref.tocsr().indptr, Xrec.tocsr().indptr))
        ok = all([same_vocab, same_idf, same_shape, same_nnz, same_data,
                  same_indices, same_indptr])
        report["all_match"] = report["all_match"] and ok
        report["blocks"][name] = {
            "match": ok,
            "sklearn_fit_vs_transform_numerically_identical": fit_path_equiv,
            "vocabulary_identical": same_vocab,
            "sklearn_terms": len(ref.vocabulary_), "recon_terms": len(fit.vocabulary_),
            "terms_seen": fit.n_terms_seen,
            "idf_bitwise_identical": same_idf,
            "shape": list(Xref.shape), "nnz": int(Xref.nnz),
            "matrix_bitwise_identical": bool(same_data and same_indices and same_indptr),
        }
        ref = None
        vec = None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
