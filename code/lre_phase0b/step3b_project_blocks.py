# -*- coding: utf-8 -*-
"""Phase 0B step 3b: build one TF-IDF block matrix, fit TruncatedSVD, persist the projection."""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time

from common import WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import config  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

PREFIX_DIR = WORK / "prefix_table"
FE_DIR = WORK / "features"
MEMO_MAX_CHARS = 8_000
MEMO_MAX_TOTAL = 1_000_000_000


def _mem_mb():
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round((m.ullTotalPhys - m.ullAvailPhys) / 1024 ** 2, 1)
    except Exception:
        return -1.0


def main() -> int:
    name = sys.argv[1]
    text_column = TFIDF_ACTION_FEEDBACK[name]
    ensure_dirs(FE_DIR)
    with open(FE_DIR / f"{name}.vocab.pkl", "rb") as fh:
        state = pickle.load(fh)
    from fe_lib import StreamTfidfFitter, iter_text_batches

    fit = StreamTfidfFitter(name, text_column, config.TFIDF_NGRAM_RANGE,
                            config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
    fit.vocabulary_ = state["vocabulary"]
    fit.idf_ = state["idf"]
    vec = fit.make_vectorizer()
    n_features = len(fit.vocabulary_)
    parts = sorted(PREFIX_DIR.glob("prefix_table.part-*.parquet"))
    t0 = time.time()
    chunks = []
    cache, cache_chars = {}, 0
    rows = 0
    nnz = 0
    batches = 0
    for offset, texts in iter_text_batches(parts, text_column):
        memo_idx, other_idx, other_texts = [], [], []
        for i, t in enumerate(texts):
            if len(t) <= MEMO_MAX_CHARS:
                memo_idx.append(i)
            else:
                other_idx.append(i)
                other_texts.append(t)
        rows_out = [None] * len(texts)
        if memo_idx:
            uniq = list(dict.fromkeys(texts[i] for i in memo_idx))
            new = [t for t in uniq if t not in cache]
            if new:
                Xn = vec.transform(new)
                for j, t in enumerate(new):
                    if cache_chars < MEMO_MAX_TOTAL:
                        cache[t] = Xn[j]
                        cache_chars += len(t)
                del Xn
            for i in memo_idx:
                rows_out[i] = cache[texts[i]]
        if other_idx:
            Xo = vec.transform(other_texts)
            for j, i in enumerate(other_idx):
                rows_out[i] = Xo[j]
            del Xo, other_texts
        X = sp.vstack(rows_out, format="csr") if len(rows_out) > 1 else rows_out[0]
        del rows_out
        chunks.append(X)
        rows += X.shape[0]
        nnz += X.nnz
        batches += 1
        if batches % 20 == 0:
            print(f"    {name}: rows={rows} nnz={nnz} usedRAM={_mem_mb()}MB "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    cache = None
    gc.collect()
    print(f"    {name}: assembling X rows={rows} nnz={nnz} chunks={len(chunks)}", flush=True)
    X = sp.vstack(chunks, format="csr")
    chunks = None
    gc.collect()
    print(f"    {name}: X {X.shape} nnz={X.nnz} usedRAM={_mem_mb()}MB", flush=True)
    n_comp = int(min(config.TFIDF_SVD_DIM_PER_BLOCK, n_features - 1, X.shape[0] - 1))
    t1 = time.time()
    svd = TruncatedSVD(n_components=n_comp, random_state=config.TFIDF_SVD_RANDOM_STATE)
    U = svd.fit_transform(X)
    svd_seconds = time.time() - t1
    np.save(FE_DIR / f"{name}.svd.npy", U.astype(np.float32))
    meta = {
        "block": name,
        "text_column": text_column,
        "rows": int(X.shape[0]),
        "tfidf_columns": int(X.shape[1]),
        "nnz": int(X.nnz),
        "n_components": n_comp,
        "max_components_allowed": int(config.TFIDF_SVD_DIM_PER_BLOCK),
        "random_state": int(config.TFIDF_SVD_RANDOM_STATE),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "singular_values_sum": float(svd.singular_values_.sum()),
        "svd_seconds": round(svd_seconds, 1),
        "build_seconds": round(svd_seconds + (t1 - t0), 1),
        "used_ram_mb_at_end": _mem_mb(),
    }
    write_json(FE_DIR / f"{name}.svd.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
