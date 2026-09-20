# -*- coding: utf-8 -*-
"""Phase 0B step 3a: stream DF/TF over the frozen prefix table and build TF-IDF vocabularies."""
from __future__ import annotations

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from common import OUT, WORK, ensure_dirs, set_vendor_env, write_json

set_vendor_env()

import config  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402

PREFIX_DIR = WORK / "prefix_table"
FE_DIR = WORK / "features"


def _worker(payload):
    name, text_column, part_paths, out_path = payload
    import pickle as _pickle
    import time as _time

    from fe_lib import StreamTfidfFitter

    fit = StreamTfidfFitter(name, text_column, config.TFIDF_NGRAM_RANGE,
                            config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
    t0 = _time.time()
    rows = 0
    from fe_lib import iter_text_batches
    for offset, texts in iter_text_batches(part_paths, text_column):
        fit.update(fit.analyze(texts))
        rows += len(texts)
    with open(out_path, "wb") as fh:
        _pickle.dump({"df": dict(fit.df), "tf": dict(fit.tf),
                      "n_docs": fit.n_docs}, fh, protocol=5)
    return {"block": name, "rows": rows, "terms": len(fit.df),
            "elapsed_sec": round(_time.time() - t0, 1)}


def main() -> int:
    ensure_dirs(FE_DIR)
    parts = sorted(PREFIX_DIR.glob("prefix_table.part-*.parquet"))
    if not parts:
        raise SystemExit("no prefix table parts found; run step2 first")
    tasks = []
    for name, col in TFIDF_ACTION_FEEDBACK.items():
        tasks.append((name, col, [str(p) for p in parts], str(FE_DIR / f"{name}.counts.pkl")))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        results = list(ex.map(_worker, tasks))
    for r in results:
        print(json.dumps(r), flush=True)

    fitter = None  # vocabulary is built in the parent from the persisted counters
    from fe_lib import StreamTfidfFitter

    blocks = {}
    for name, col in TFIDF_ACTION_FEEDBACK.items():
        with open(FE_DIR / f"{name}.counts.pkl", "rb") as fh:
            counts = pickle.load(fh)
        f = StreamTfidfFitter(name, col, config.TFIDF_NGRAM_RANGE,
                              config.TFIDF_MIN_DF, config.TFIDF_MAX_FEATURES)
        f.df = counts["df"]
        f.tf = counts["tf"]
        f.n_docs = counts["n_docs"]
        f.build_vocabulary()
        blocks[name] = f.summary()
        with open(FE_DIR / f"{name}.vocab.pkl", "wb") as fh:
            pickle.dump({"vocabulary": f.vocabulary_, "idf": f.idf_,
                         "n_docs": f.n_docs, "summary": f.summary()}, fh, protocol=5)
        blocks[name]["idf_first3"] = [float(x) for x in f.idf_[:3]]
        del f

    write_json(FE_DIR / "tfidf_blocks.json", {
        "fit_corpus": "all prefix rows of the frozen prefix table (repository default: "
                      "one global FeatureEngineer fitted on the full prefix table)",
        "blocks": blocks,
        "fit_seconds": round(time.time() - t0, 1),
    })
    print(json.dumps(blocks, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
