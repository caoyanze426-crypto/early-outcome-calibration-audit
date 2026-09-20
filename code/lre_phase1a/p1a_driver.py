# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A driver: run the 45 frozen pair folds.

Sequential or bounded-parallel execution of `p1a_pair_fold.py`. A fold that fails
for engineering reasons is recorded as FOLD_FAIL and never repaired; the remaining
frozen pair folds continue (section 23).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common1a import FOLDS, NL, OUT, PRED_DEC, PRED_PAIR, WORK, ensure_dirs, write_json

import pandas as pd  # noqa: E402

PY = sys.executable
HERE = WORK
THREADS_PER_FOLD = 6


def run_one(row):
    pair_id = str(row["pair_id"])
    done = PRED_DEC / f"{pair_id}.csv"
    meta = WORK / "tmp" / pair_id / "fold_meta.json"
    if done.exists() and meta.exists():
        return {"pair_id": pair_id, "status": "SKIPPED_ALREADY_COMPLETE"}
    args = [PY, str(HERE / "p1a_pair_fold.py"), pair_id,
            str(row["agent_A"]), str(row["agent_B"]), str(THREADS_PER_FOLD)]
    t0 = time.time()
    proc = subprocess.run(args, cwd=str(HERE), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = (proc.stdout or "")[-1200:]
    err = (proc.stderr or "")[-2500:]
    ok = proc.returncode == 0 and done.exists()
    print(f"### {pair_id} {'OK' if ok else 'FOLD_FAIL'} "
          f"rc={proc.returncode} {time.time()-t0:.0f}s", flush=True)
    if not ok:
        print(tail, flush=True)
        print(err, flush=True)
    return {"pair_id": pair_id, "status": "OK" if ok else "FOLD_FAIL",
            "returncode": int(proc.returncode),
            "seconds": round(time.time() - t0, 1),
            "stdout_tail": tail if not ok else "",
            "stderr_tail": err if not ok else ""}


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ensure_dirs(FOLDS, PRED_PAIR, PRED_DEC, WORK / "tmp")
    man = pd.read_csv(FOLDS / "pair_fold_manifest.csv")
    rows = man.to_dict(orient="records")
    t0 = time.time()
    print(f"phase1a driver: folds={len(rows)} workers={workers} "
          f"threads_per_fold={THREADS_PER_FOLD}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run_one, rows))
    secs = round(time.time() - t0, 1)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FOLD_FAIL")
    n_skip = sum(1 for r in results if r["status"].startswith("SKIPPED"))
    write_json(OUT / "folds" / "driver_results.json", {
        "fold_threads": THREADS_PER_FOLD, "workers": workers,
        "folds_planned": len(rows), "folds_completed": n_ok,
        "folds_skipped_already_complete": n_skip, "folds_failed": n_fail,
        "total_driver_wall_clock_seconds": secs,
        "results": results,
    })
    print(NL + json.dumps({"planned": len(rows), "ok": n_ok, "fail": n_fail,
                           "skipped": n_skip, "wall_clock_seconds": secs}),
          flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
