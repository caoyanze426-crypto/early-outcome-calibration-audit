# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A - freeze the 45 unordered held-out pair folds before training.

Section 3: pair_id / agent_A / agent_B must be frozen before any training runs.
"""
from __future__ import annotations

import sys

from common1a import (FOLDS, MODELS, NL, OUT, all_pairs, ensure_dirs, pair_id_of,
                      write_json)

import pandas as pd  # noqa: E402


def main() -> int:
    ensure_dirs(FOLDS)
    pairs = all_pairs()
    assert len(pairs) == 45, f"expected C(10,2)=45 pair folds, got {len(pairs)}"
    rows = []
    for a, b in pairs:
        rows.append({"pair_id": pair_id_of(a, b), "agent_A": a, "agent_B": b})
    df = pd.DataFrame(rows)
    df.to_csv(FOLDS / "pair_fold_manifest.csv", index=False, encoding="utf-8")
    write_json(OUT / "folds" / "pair_fold_manifest.json", {
        "frozen_before_training": True,
        "n_pair_folds": len(rows),
        "agent_labels": MODELS,
        "pairing_rule": "all unordered pairs of the 10 frozen agent labels",
        "pair_id_rule": "pair-<index(agent_A):02d><index(agent_B):02d> "
                        "with agent_A < agent_B in the frozen label order",
        "pairs": rows,
    })
    print(NL.join([f"{r['pair_id']}  {r['agent_A']}  {r['agent_B']}"
                   for r in rows]), flush=True)
    print(f"frozen pair folds = {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
