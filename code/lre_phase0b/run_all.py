# -*- coding: utf-8 -*-
"""Phase 0B driver: encode -> fold features -> train (stages run in parallel)."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from common import WORK, set_vendor_env

set_vendor_env()

import pandas as pd  # noqa: E402

PY = sys.executable
HERE = WORK


def holdout_models():
    row_index = pd.read_parquet(WORK / "row_index.parquet", columns=["model_id"])
    return sorted(row_index["model_id"].astype(str).unique())


def run(args, label):
    print(f"### {label}: {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd=str(WORK), capture_output=True, text=True)
    tail = (proc.stdout or "")[-3000:]
    if proc.returncode != 0:
        print(tail, flush=True)
        print((proc.stderr or "")[-4000:], flush=True)
        raise SystemExit(f"{label} failed rc={proc.returncode}")
    print(tail[-900:], flush=True)


def main() -> int:
    stage = sys.argv[1]
    np_ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if stage == "encode":
        script = str(HERE / "step3_encode.py")
        for s in ("meta", "vocab", "encode"):
            run([PY, script, s], f"encode:{s}")
        return 0
    models = holdout_models()
    if stage == "folds":
        script = str(HERE / "step4_fold_features.py")
        args = [[PY, script, m, f"fold-{i:02d}"] for i, m in enumerate(models)]
    elif stage == "train":
        script = str(HERE / "step5_train_fold.py")
        args = [[PY, script, f"fold-{i:02d}", "6"] for i, _m in enumerate(models)]
    else:
        raise SystemExit(f"unknown stage {stage}")
    with ThreadPoolExecutor(max_workers=np_) as ex:
        list(ex.map(lambda a: run(a, a[2]), args))
    print(json.dumps({"stage": stage, "jobs": len(args), "parallel": np_}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
