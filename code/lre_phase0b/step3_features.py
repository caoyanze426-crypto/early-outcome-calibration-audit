# -*- coding: utf-8 -*-
"""Phase 0B step 3: reference-free EarlyEval design matrix (dense + AF TF-IDF/SVD)."""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

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

PREFIX_DIR = WORK / "prefix_table"
FE_DIR = WORK / "features"
META_COLS = ["prefix_id", "traj_id", "instance_id", "model", "prefix_step_idx",
             "label", "sample_weight", "n_steps_total_for_weighting"]
BIG_BLOCKS = {"tfidf_prefix_action", "tfidf_prefix_feedback"}
BIG_WORKERS = 8
MEMO_MAX_CHARS = 8_000
MEMO_MAX_TOTAL = 500_000_000


def is_reference_family_feature(name: str) -> bool:
    """Repository rule: model_holdout_shadow_valid_retrain._is_reference_family_feature."""
    return name.startswith("gold_")


def admissible(cols):
    return [c for c in cols if not is_reference_family_feature(c)]


NUM = admissible(NUMERIC_FEATURES)
BOOLS = admissible(BOOL_FEATURES)
CATS = admissible(CATEGORICAL_FEATURES)


def _mem_mb():
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


def build_dense():
    import pyarrow.parquet as parquet

    parts = sorted(PREFIX_DIR.glob("prefix_table.part-*.parquet"))
    cols = META_COLS + NUM + BOOLS + CATS
    frames = []
    for p in parts:
        pf = parquet.ParquetFile(p)
        names = [n for n in cols if n in pf.schema_arrow.names]
        frames.append(pf.read(columns=names).to_pandas())
        del pf
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df


def _count_task(task):
    block, column, parts, out_path = task
    import pickle as _pickle

    from fe_lib import StreamTfidfFitter, iter_text_batches

    fit = StreamTfidfFitter(block, column, config.TFIDF_NGRAM_RANGE,
                            config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
    for _offset, texts in iter_text_batches(parts, column):
        fit.update(fit.analyze(texts))
    with open(out_path, "wb") as fh:
        _pickle.dump({"df": dict(fit.df), "tf": dict(fit.tf), "n_docs": fit.n_docs},
                     fh, protocol=5)
    return {"block": block, "task": out_path, "rows": fit.n_docs, "terms": len(fit.df)}


def _csrfy_task(task):
    block, column, parts, vocab_path, out_path = task
    import pickle as _pickle

    import scipy.sparse as _sp

    from fe_lib import StreamTfidfFitter, iter_text_batches

    with open(vocab_path, "rb") as fh:
        state = _pickle.load(fh)
    fit = StreamTfidfFitter(block, column, config.TFIDF_NGRAM_RANGE,
                            config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
    fit.vocabulary_ = state["vocabulary"]
    fit.idf_ = state["idf"]
    vec = fit.make_vectorizer()
    chunks = []
    cache, cache_chars = {}, 0
    rows = nnz = 0
    for _offset, texts in iter_text_batches(parts, column):
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
        chunk = _sp.vstack(rows_out, format="csr") if len(rows_out) > 1 else rows_out[0]
        del rows_out
        chunks.append(chunk)
        rows += chunk.shape[0]
        nnz += chunk.nnz
    X = _sp.vstack(chunks, format="csr") if len(chunks) > 1 else chunks[0]
    del chunks, cache
    _sp.save_npz(out_path, X)
    return {"block": block, "task": out_path, "rows": int(X.shape[0]),
            "cols": int(X.shape[1]), "nnz": int(X.nnz)}


def _groups(parts, n):
    n = max(1, min(n, len(parts)))
    size = (len(parts) + n - 1) // n
    return [parts[i:i + size] for i in range(0, len(parts), size)]


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "vocab"
    ensure_dirs(FE_DIR)
    parts = sorted(PREFIX_DIR.glob("prefix_table.part-*.parquet"))
    print(f"parts={len(parts)} stage={stage} ram_used={_mem_mb()}MB", flush=True)

    if stage == "vocab":
        tasks = []
        for block, col in TFIDF_ACTION_FEEDBACK.items():
            w = BIG_WORKERS if block in BIG_BLOCKS else 1
            for gi, grp in enumerate(_groups(parts, w)):
                tasks.append((block, col, [str(p) for p in grp],
                              str(FE_DIR / f"counts.{block}.g{gi:02d}.pkl")))
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(10, len(tasks))) as ex:
            done = list(ex.map(_count_task, tasks))
        print(f"count tasks done in {time.time() - t0:.0f}s", flush=True)
        blocks = {}
        for block, col in TFIDF_ACTION_FEEDBACK.items():
            fdf, ftf, ndocs = {}, {}, 0
            for r in done:
                if r["block"] != block:
                    continue
                with open(r["task"], "rb") as fh:
                    c = pickle.load(fh)
                ndocs += c["n_docs"]
                for k, v in c["df"].items():
                    fdf[k] = fdf.get(k, 0) + v
                for k, v in c["tf"].items():
                    ftf[k] = ftf.get(k, 0) + v
                del c
            from fe_lib import StreamTfidfFitter

            f = StreamTfidfFitter(block, col, config.TFIDF_NGRAM_RANGE,
                                  config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
            f.df, f.tf, f.n_docs = fdf, ftf, ndocs
            f.build_vocabulary()
            with open(FE_DIR / f"{block}.vocab.pkl", "wb") as fh:
                pickle.dump({"vocabulary": f.vocabulary_, "idf": f.idf_,
                             "n_docs": f.n_docs, "summary": f.summary()}, fh, protocol=5)
            blocks[block] = f.summary()
            print(json.dumps(blocks[block]), flush=True)
            fdf = ftf = f = None
            gc.collect()
        write_json(FE_DIR / "tfidf_vocab_report.json", {
            "blocks": blocks, "count_task_seconds": round(time.time() - t0, 1)})
        return 0

    if stage == "dense":
        t0 = time.time()
        df = build_dense()
        df.to_parquet(FE_DIR / "dense_meta.parquet", index=False, compression="zstd")
        print(json.dumps({"rows": int(len(df)), "columns": list(df.columns),
                          "seconds": round(time.time() - t0, 1)}, ensure_ascii=False),
              flush=True)
        return 0

    if stage == "blocks":
        import os

        summary = {}
        for block, col in TFIDF_ACTION_FEEDBACK.items():
            w = BIG_WORKERS if block in BIG_BLOCKS else 2
            tasks = [(block, col, [str(p) for p in grp],
                      str(FE_DIR / f"{block}.vocab.pkl"),
                      str(FE_DIR / f"csr.{block}.g{gi:02d}.npz"))
                     for gi, grp in enumerate(_groups(parts, w))]
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
                res = list(ex.map(_csrfy_task, tasks))
            print(f"  {block}: csr built in {time.time() - t0:.0f}s "
                  f"rows={sum(r['rows'] for r in res)}", flush=True)
            mats = [sp.load_npz(r["task"]) for r in res]
            X = sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]
            mats = None
            gc.collect()
            n_comp = int(min(config.TFIDF_SVD_DIM_PER_BLOCK, X.shape[1] - 1, X.shape[0] - 1))
            t1 = time.time()
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(n_components=n_comp,
                               random_state=config.TFIDF_SVD_RANDOM_STATE)
            U = svd.fit_transform(X)
            np.save(FE_DIR / f"{block}.svd.npy", U.astype(np.float32))
            summary[block] = {
                "rows": int(X.shape[0]), "tfidf_columns": int(X.shape[1]),
                "nnz": int(X.nnz), "n_components": n_comp,
                "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
                "csr_seconds": round(t1 - t0, 1),
                "svd_seconds": round(time.time() - t1, 1),
                "used_ram_mb": _mem_mb()}
            print(json.dumps(summary[block]), flush=True)
            for r in res:
                try:
                    os.remove(r["task"])
                except OSError:
                    pass
            X = svd = U = None
            gc.collect()
        write_json(FE_DIR / "tfidf_svd_report.json", summary)
        return 0

    raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    sys.exit(main())
