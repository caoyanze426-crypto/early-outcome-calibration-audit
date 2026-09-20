# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - integrity checks for the fold pipeline.

1. row_index.parquet is element-wise aligned with the prefix-table parts.
2. No held-out target model appears in any fold's TRAIN or VALID split, and each
   fold's TEST split contains only that fold's held-out target.
3. Every fold's design matrix width equals its saved feature-name list.
Offline only.
"""
from __future__ import annotations

import json
import sys

from common import FOLDS, WORK, read_json, write_json

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

META = ["prefix_id", "traj_id", "instance_id", "model_id", "prefix_step_idx"]
WORK_FOLDS = WORK / "folds"


def alignment() -> dict:
    ri = pd.read_parquet(WORK / "row_index.parquet", columns=META)
    frames = []
    for p in sorted((WORK / "prefix_table").glob("prefix_table.part-*.parquet")):
        frames.append(pq.ParquetFile(p).read(columns=META).to_pandas())
    pt = pd.concat(frames, ignore_index=True)
    del frames
    ok_len = len(ri) == len(pt)
    same = bool(ok_len and (ri["prefix_id"].to_numpy()
                            == pt["prefix_id"].to_numpy()).all())
    same_traj = bool(ok_len and (ri["traj_id"].to_numpy()
                                 == pt["traj_id"].to_numpy()).all())
    return {
        "row_index_rows": int(len(ri)), "prefix_table_rows": int(len(pt)),
        "length_match": bool(ok_len),
        "prefix_id_elementwise_equal": same,
        "traj_id_elementwise_equal": same_traj,
        "ALIGNED": bool(ok_len and same and same_traj),
    }


def folds() -> dict:
    pre = read_json(WORK / "preflight.json")
    models = sorted(pre["eligible_models"])
    checks, all_ok = [], True
    for i, model in enumerate(models):
        tag = "fold-%02d" % i
        d = WORK_FOLDS / tag
        entry = {"fold_tag": tag, "holdout_model": model}
        try:
            seen = {}
            for name in ("train", "valid", "test"):
                m = pd.read_parquet(d / ("meta_%s.parquet" % name),
                                    columns=["model_id", "traj_id"])
                seen[name] = sorted(m["model_id"].astype(str).unique().tolist())
                entry["%s_rows" % name] = int(len(m))
            entry["holdout_absent_from_train"] = bool(model not in seen["train"])
            entry["holdout_absent_from_valid"] = bool(model not in seen["valid"])
            entry["test_is_only_holdout"] = bool(seen["test"] == [model])
            entry["n_train_models"] = len(seen["train"])
            entry["n_valid_models"] = len(seen["valid"])
            X = np.load(d / "X_train.npy", mmap_mode="r")
            names = json.loads((d / "feature_names.json").read_text("utf-8"))
            entry["feature_width_match"] = bool(X.shape[1] == len(names))
            entry["n_features"] = int(X.shape[1])
            entry["train_rows_x"] = int(X.shape[0])
            del X
            entry["fold_ok"] = bool(
                entry["holdout_absent_from_train"]
                and entry["holdout_absent_from_valid"]
                and entry["test_is_only_holdout"]
                and entry["feature_width_match"])
        except Exception as exc:  # noqa: BLE001
            entry["fold_ok"] = False
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
        all_ok = all_ok and entry.get("fold_ok", False)
        checks.append(entry)
    return {"folds": checks, "ALL_FOLDS_VALID": bool(all_ok),
            "n_folds": len(checks)}


def main() -> int:
    al = alignment()
    fl = folds()
    payload = {"alignment": al, **fl}
    write_json(WORK / "verify_report.json", payload)
    summary = {"ALIGNED": al["ALIGNED"],
               "ALL_FOLDS_VALID": fl["ALL_FOLDS_VALID"],
               "n_folds": fl["n_folds"]}
    print(json.dumps(summary, indent=2), flush=True)
    bad = [f["fold_tag"] for f in fl["folds"] if not f.get("fold_ok")]
    if bad:
        print("folds failing: %s" % bad, flush=True)
    return 0 if (al["ALIGNED"] and fl["ALL_FOLDS_VALID"]) else 1


if __name__ == "__main__":
    sys.exit(main())
