# -*- coding: utf-8 -*-
"""Measure the frozen encoder block sizes so Phase 1A memory can be planned."""
from __future__ import annotations

import time

from common1a import ENC_DIR, NL  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

BLOCKS = ["tfidf_task_prompt", "tfidf_prefix_action", "tfidf_prefix_feedback",
          "tfidf_last_action", "tfidf_last_feedback"]


def main() -> int:
    for block in BLOCKS:
        files = sorted(ENC_DIR.glob(f"gcsr.{block}.s*.npz"))
        t0 = time.time()
        nnz = 0
        shape = None
        for f in files:
            m = sp.load_npz(f)
            nnz += int(m.nnz)
            shape = m.shape
            del m
        gb = nnz * 8 / 1e9
        print(f"{block:24s} shards={len(files)} shape={shape} nnz={nnz} "
              f"~{gb:.2f}GB(data+indices) load={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
