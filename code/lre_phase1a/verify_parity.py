# -*- coding: utf-8 -*-
"""Fidelity check: with a SINGLE holdout agent the Phase 1A split code must
reproduce the frozen Phase 0B fold split exactly (same train/valid/test rows).

This validates the generalized `build_split` before any Phase 1A training runs.
"""
from __future__ import annotations

import sys

from common1a import MODELS, NL, OUT, ROW_INDEX, as_builtin, write_json

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from p1a_pair_fold import build_split  # noqa: E402


def main() -> int:
    row_index = pd.read_parquet(ROW_INDEX)
    work_folds = ROW_INDEX.parent / "folds"
    results = {}
    for i, model in enumerate(MODELS):
        fold_tag = f"fold-{i:02d}"
        split = build_split(row_index, {model})
        rec = {}
        for name in ("train", "valid", "test"):
            ref = pd.read_parquet(work_folds / fold_tag / f"meta_{name}.parquet",
                                  columns=["row_position"])
            got = np.sort(np.asarray(split[name], dtype=np.int64))
            exp = np.sort(ref["row_position"].to_numpy(dtype=np.int64))
            same = bool(got.shape == exp.shape and np.array_equal(got, exp))
            n_diff = int(np.setxor1d(got, exp).size) if (got.size or exp.size) else 0
            rec[name] = {"n_expected": int(exp.size), "n_observed": int(got.size),
                         "identical": same, "n_symmetric_difference": n_diff}
        rec["fold_holdout_model"] = model
        rec["ALL_THREE_IDENTICAL"] = bool(
            rec["train"]["identical"] and rec["valid"]["identical"]
            and rec["test"]["identical"])
        results[fold_tag] = rec
        print(f"{fold_tag} {model:24s} train={rec['train']['identical']} "
              f"valid={rec['valid']['identical']} test={rec['test']['identical']} "
              f"(n={rec['train']['n_expected']}/"
              f"{rec['valid']['n_expected']}/{rec['test']['n_expected']})",
              flush=True)
    all_ok = all(r["ALL_THREE_IDENTICAL"] for r in results.values())
    write_json(OUT / "analysis" / "split_parity_check.json", as_builtin({
        "check": "Phase 1A generalized build_split with a single holdout must "
                 "reproduce the frozen Phase 0B fold split exactly",
        "folds_checked": len(results),
        "all_folds_identical": all_ok,
        "per_fold": results,
    }))
    print(f"{NL}ALL 10 PHASE 0B FOLDS REPRODUCED EXACTLY = {all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
