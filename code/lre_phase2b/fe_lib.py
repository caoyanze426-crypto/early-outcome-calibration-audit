# -*- coding: utf-8 -*-
"""Phase 0B: streamed, exact reconstruction of sklearn TfidfVectorizer fit state."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


def iter_text_batches(part_paths, column, batch_rows=2048):
    """Yield (row_offset, list_of_texts) for one parquet column, in stable part order."""
    import pyarrow.parquet as parquet

    offset = 0
    for part in part_paths:
        pf = parquet.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[column]):
            col = batch.column(0)
            texts = ["" if v is None else str(v) for v in col.to_pylist()]
            yield offset, texts
            offset += len(texts)
        del pf


class StreamTfidfFitter:
    """Reproduce TfidfVectorizer(min_df, max_features, ngram_range, sublinear_tf).fit."""

    def __init__(self, name, text_column, ngram_range, min_df, max_features,
                 sublinear_tf=True, dtype=np.float32):
        self.name = name
        self.text_column = text_column
        self.ngram_range = tuple(ngram_range)
        self.min_df = int(min_df)
        self.max_features = int(max_features) if max_features else None
        self.sublinear_tf = bool(sublinear_tf)
        self.dtype = dtype
        self._analyzer = TfidfVectorizer(
            ngram_range=self.ngram_range, dtype=dtype).build_analyzer()
        self.df = Counter()
        self.tf = Counter()
        self.n_docs = 0
        self.vocabulary_ = None
        self.idf_ = None
        self.n_terms_seen = 0

    def analyze(self, texts):
        return [Counter(self._analyzer(t)) for t in texts]

    def update(self, counters):
        self.n_docs += len(counters)
        df, tf = self.df, self.tf
        for c in counters:
            for term, n in c.items():
                df[term] += 1
                tf[term] += n

    def build_vocabulary(self):
        terms = sorted(self.df)
        self.n_terms_seen = len(terms)
        df_arr = np.asarray([self.df[t] for t in terms], dtype=self.dtype)
        tf_arr = np.asarray([self.tf[t] for t in terms], dtype=self.dtype)
        mask = df_arr >= self.min_df
        if self.max_features is not None and int(mask.sum()) > self.max_features:
            mask_inds = (-tf_arr[mask]).argsort()[: self.max_features]
            new_mask = np.zeros(len(terms), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask
        kept = [t for t, m in zip(terms, mask) if m]
        if not kept:
            raise ValueError("no terms survived min_df/max_features")
        self.vocabulary_ = {t: i for i, t in enumerate(kept)}
        n = self.n_docs + 1
        df_kept = df_arr[mask] + 1.0
        idf = np.full_like(df_kept, fill_value=n, dtype=self.dtype)
        idf /= df_kept
        np.log(idf, out=idf)
        idf += 1.0
        self.idf_ = idf
        return self.vocabulary_

    def make_vectorizer(self):
        if self.vocabulary_ is None:
            raise RuntimeError("build_vocabulary() must run before make_vectorizer()")
        vec = TfidfVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df,
            max_features=self.max_features, sublinear_tf=self.sublinear_tf,
            dtype=self.dtype)
        vec.vocabulary_ = dict(self.vocabulary_)
        vec.fixed_vocabulary_ = True
        vec._tfidf = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True,
            sublinear_tf=self.sublinear_tf)
        vec._tfidf.idf_ = self.idf_
        vec._tfidf.n_features_in_ = len(self.vocabulary_)
        return vec

    def summary(self):
        return {
            "block": self.name,
            "text_column": self.text_column,
            "n_documents": self.n_docs,
            "n_terms_seen": self.n_terms_seen,
            "n_terms_kept": len(self.vocabulary_ or {}),
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "norm": "l2",
            "smooth_idf": True,
            "dtype": np.dtype(self.dtype).name,
        }
