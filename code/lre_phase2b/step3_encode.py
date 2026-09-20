# -*- coding: utf-8 -*-
"""Phase 0B step 3: global term dictionary + encoded raw-count CSR.

The corpus is tokenised exactly twice (once to build a global term dictionary and
once to encode raw term counts). Fold-local FeatureEngineer fitting is then a
cheap re-indexing operation, so the held-out model never contributes to a fold's
vocabulary, idf, or SVD basis. Offline only: no LLM / model API calls.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from common import WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import config  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402

PREFIX_DIR = WORK / "prefix_table"
ENC_DIR = WORK / "encoded"
BLOCKS = list(TFIDF_ACTION_FEEDBACK.keys())
GVOCAB_MIN_DF = int(config.TFIDF_MIN_DF)
N_SHARDS = 5
META_COLS = ["prefix_id", "traj_id", "instance_id", "model", "model_id",
             "prefix_step_idx", "label", "sample_weight",
             "n_steps_total_for_weighting"]


def shard_groups(parts, n):
    n = max(1, min(int(n), len(parts)))
    size = (len(parts) + n - 1) // n
    return [parts[i:i + size] for i in range(0, len(parts), size)]


def part_paths():
    return sorted(PREFIX_DIR.glob("prefix_table.part-*.parquet"))


def _count_task(task):
    block, column, paths, out_path = task
    import pickle as pk

    from fe_lib import StreamTfidfFitter, iter_text_batches

    fit = StreamTfidfFitter(block, column, config.TFIDF_NGRAM_RANGE,
                            GVOCAB_MIN_DF, None)
    t0 = time.time()
    for _o, texts in iter_text_batches(paths, column):
        fit.update(fit.analyze(texts))
    with open(out_path, "wb") as fh:
        pk.dump({"df": dict(fit.df), "tf": dict(fit.tf), "n_docs": fit.n_docs},
                fh, protocol=5)
    return {"block": block, "rows": int(fit.n_docs), "terms": len(fit.df),
            "seconds": round(time.time() - t0, 1)}


def _encode_task(task):
    block, column, paths, vocab_path, out_path = task
    import pickle as pk

    from sklearn.feature_extraction.text import CountVectorizer

    from fe_lib import iter_text_batches

    with open(vocab_path, "rb") as fh:
        vocab = pk.load(fh)["vocabulary"]
    vec = CountVectorizer(ngram_range=tuple(config.TFIDF_NGRAM_RANGE),
                          vocabulary=vocab, dtype=np.float32)
    chunks = []
    rows = nnz = 0
    t0 = time.time()
    for _o, texts in iter_text_batches(paths, column, batch_rows=1024):
        X = sp.csr_matrix(vec.fit_transform(texts))
        chunks.append(X)
        rows += X.shape[0]
        nnz += int(X.nnz)
        del X
    X = sp.vstack(chunks, format="csr")
    del chunks
    sp.save_npz(out_path, X)
    return {"block": block, "rows": int(X.shape[0]), "cols": int(X.shape[1]),
            "nnz": int(X.nnz), "seconds": round(time.time() - t0, 1)}


def stage_meta() -> int:
    import pyarrow.parquet as pq
    import pandas as pd

    parts = part_paths()
    frames = []
    for p in parts:
        pf = pq.ParquetFile(p)
        names = [c for c in META_COLS if c in pf.schema_arrow.names]
        frames.append(pf.read(columns=names).to_pandas())
        del pf
    df = pd.concat(frames, ignore_index=True)
    del frames
    df.to_parquet(WORK / "row_index.parquet", index=False, compression="zstd")
    summary = {
        "parts": len(parts),
        "rows": int(len(df)),
        "trajectories": int(df["traj_id"].nunique()),
        "models": int(df["model_id"].nunique()),
        "resolved_rate_rows": float(df["label"].mean()),
        "columns": list(df.columns),
    }
    write_json(WORK / "row_index_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def stage_vocab() -> int:
    parts = part_paths()
    groups = shard_groups(parts, N_SHARDS)
    tasks = []
    for block, col in TFIDF_ACTION_FEEDBACK.items():
        for gi, grp in enumerate(groups):
            tasks.append((block, col, [str(p) for p in grp],
                          str(ENC_DIR / f"gcounts.{block}.s{gi:02d}.pkl")))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        done = list(ex.map(_count_task, tasks))
    blocks = {}
    for block, col in TFIDF_ACTION_FEEDBACK.items():
        fdf, ftf, ndocs = {}, {}, 0
        shard_files = sorted(ENC_DIR.glob(f"gcounts.{block}.s*.pkl"))
        for f in shard_files:
            with open(f, "rb") as fh:
                c = pickle.load(fh)
            ndocs += c["n_docs"]
            for k, v in c["df"].items():
                fdf[k] = fdf.get(k, 0) + v
            for k, v in c["tf"].items():
                ftf[k] = ftf.get(k, 0) + v
            del c
        from fe_lib import StreamTfidfFitter

        f = StreamTfidfFitter(block, col, config.TFIDF_NGRAM_RANGE,
                              GVOCAB_MIN_DF, None)
        f.df, f.tf, f.n_docs = fdf, ftf, ndocs
        f.build_vocabulary()
        with open(ENC_DIR / f"gvocab.{block}.pkl", "wb") as fh:
            pickle.dump({"vocabulary": f.vocabulary_, "n_docs": f.n_docs}, fh,
                        protocol=5)
        blocks[block] = {"n_documents": int(ndocs), "n_terms_seen": len(fdf),
                         "n_terms_kept": len(f.vocabulary_)}
        print(json.dumps({block: blocks[block]}), flush=True)
        fdf = ftf = f = None
    write_json(ENC_DIR / "global_vocab_report.json", {
        "rule": "TfidfVectorizer fit rule with min_df=%d and no max_features cap "
                "(lossless superset for any fold-local fit)" % GVOCAB_MIN_DF,
        "ngram_range": list(config.TFIDF_NGRAM_RANGE),
        "n_shards": len(groups),
        "blocks": blocks,
        "seconds": round(time.time() - t0, 1),
    })
    for f in ENC_DIR.glob("gcounts.*.pkl"):
        try:
            f.unlink()
        except OSError:
            pass
    return 0


def stage_encode() -> int:
    parts = part_paths()
    groups = shard_groups(parts, N_SHARDS)
    for f in ENC_DIR.glob("gcsr.*.npz"):
        f.unlink()
    for block, col in TFIDF_ACTION_FEEDBACK.items():
        vocab_path = ENC_DIR / f"gvocab.{block}.pkl"
        tasks = [(block, col, [str(p) for p in grp], str(vocab_path),
                  str(ENC_DIR / f"gcsr.{block}.s{gi:02d}.npz"))
                 for gi, grp in enumerate(groups)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            res = list(ex.map(_encode_task, tasks))
        print(json.dumps({"block": block, "rows": sum(r["rows"] for r in res),
                          "cols": res[0]["cols"], "nnz": sum(r["nnz"] for r in res),
                          "seconds": round(time.time() - t0, 1)}), flush=True)
    write_json(ENC_DIR / "encode_manifest.json", {
        "n_shards": len(groups),
        "shard_parts": [[p.name for p in g] for g in groups],
        "blocks": {b: {"shards": [f"gcsr.{b}.s{i:02d}.npz" for i in range(len(groups))]}
                   for b in BLOCKS},
    })
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "meta"
    ensure_dirs(ENC_DIR)
    if stage == "meta":
        return stage_meta()
    if stage == "vocab":
        return stage_vocab()
    if stage == "encode":
        return stage_encode()
    raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    sys.exit(main())
