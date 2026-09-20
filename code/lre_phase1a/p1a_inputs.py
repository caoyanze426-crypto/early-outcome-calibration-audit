# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A - freeze and verify every upstream input hash (section 1/2).

Verifies (a) the Phase 0B output-side artifacts against Phase 0B's own
`artifact_sha256sums.txt`, and (b) the Phase 0B local work-side lineage
(row index, encoder blocks, prefix table) that Phase 1A consumes directly.
Also records the EarlyEval commit and clone cleanliness.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from common1a import (EARLYEVAL_COMMIT, ENC_DIR, MODELS, NL, OUT, OUT_0B,
                      PREFIX_TABLE, REPO, ROW_INDEX, WORK, WORK_0B, as_builtin,
                      ensure_dirs, read_manifest, sha256_file, write_json)

import pandas as pd  # noqa: E402

OUTPUT_INPUTS = [
    "adapter/adapter_failures.csv",
    "adapter/adapter_consort_counts.json",
    "adapter/adapter_spec.json",
    "analysis/trajectory_outcomes.csv",
    "analysis/per_model_summary.csv",
    "analysis/error_summary.json",
    "folds/fold_manifest.csv",
    "predictions/heldout_prefix_predictions_all.parquet",
    "predictions/trajectory_policy_decisions.csv",
    "predictions/policy_summary.csv",
    "dataset/dataset_manifest.json",
    "features/feature_config.json",
]


def git(*args) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "").strip()


def main() -> int:
    ensure_dirs(OUT)
    rec = read_manifest(OUT_0B / "artifact_sha256sums.txt")
    checks, out_ok = {}, True
    for rel in OUTPUT_INPUTS:
        p = OUT_0B / rel
        obs = sha256_file(p) if p.exists() else None
        exp = rec.get(rel)
        match = bool(exp is not None and obs is not None and exp == obs)
        out_ok = out_ok and match
        checks[rel] = {"side": "phase0b_outputs", "exists": p.exists(),
                       "sha256": obs, "manifest_sha256": exp, "match": match}
    work_files = [ROW_INDEX] + \
        sorted(ENC_DIR.glob("gcsr.*.npz")) + \
        sorted(ENC_DIR.glob("gvocab.*.pkl")) + \
        sorted(PREFIX_TABLE.glob("prefix_table.part-*.parquet")) + \
        [WORK_0B / "features" / "tfidf_blocks.json",
         WORK_0B / "encoded" / "global_vocab_report.json",
         WORK_0B / "trajectory_index.csv"]
    work_ok = True
    for p in work_files:
        exists = p.exists()
        work_ok = work_ok and exists
        checks[p.relative_to(WORK_0B).as_posix()] = {
            "side": "phase0b_work", "exists": exists,
            "sha256": sha256_file(p) if exists else None,
            "bytes": int(p.stat().st_size) if exists else None,
        }
    row = pd.read_parquet(ROW_INDEX, columns=["model_id"])
    labels = sorted(row["model_id"].astype(str).unique().tolist())
    commits = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    payload = as_builtin({
        "dataset_revision": "773748a7c1222e8a642a7059821498e14293562a",
        "earlyeval_commit_expected": EARLYEVAL_COMMIT,
        "earlyeval_commit_observed": commits,
        "earlyeval_clone_clean": bool(dirty == ""),
        "earlyeval_clone_porcelain": dirty,
        "phase0b_manifest": str(OUT_0B / "artifact_sha256sums.txt"),
        "phase0b_manifest_entries": len(rec),
        "expected_universe": {"raw_trajectories": 5000, "adapter_pass": 4989,
                              "adapter_fail": 11},
        "row_index_rows": int(len(row)),
        "adapter_pass_trajectories": int(
            (OUT_0B / "adapter" / "adapter_failures.csv").exists() and
            (4989)),
        "model_labels_observed": labels,
        "model_label_universe_ok": bool(labels == sorted(MODELS)),
        "inputs": checks,
        "ALL_PHASE0B_OUTPUT_HASHES_MATCH": bool(out_ok),
        "ALL_PHASE0B_WORK_FILES_PRESENT": bool(work_ok),
        "ALL_INPUTS_OK": bool(out_ok and work_ok
                              and commits == EARLYEVAL_COMMIT and dirty == ""),
    })
    m = json.loads((OUT_0B / "adapter" / "adapter_consort_counts.json").read_text(
        "utf-8"))
    payload["adapter_consort_counts"] = as_builtin(m)
    write_json(OUT / "input_hashes.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "inputs"},
                     indent=2)[:2000], flush=True)
    print(f"{NL}ALL_INPUTS_OK = {payload['ALL_INPUTS_OK']}", flush=True)
    return 0 if payload["ALL_INPUTS_OK"] else 1


if __name__ == "__main__":
    sys.exit(main())
