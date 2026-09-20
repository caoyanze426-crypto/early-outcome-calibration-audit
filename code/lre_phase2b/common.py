# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - shared paths and helpers for the reused frozen pipeline.

The vendored-pipeline scripts copied into this directory (step3_encode.py,
step4_fold_features.py, step5_train_fold.py) import ``common``, ``fe_lib``,
``config`` and ``feature_engineer`` exactly as the frozen Phase 0B lineage did.
This module only re-points WORK/OUT at the Phase 2B namespace, so the feature
engineering, split, training and calibration code paths stay byte-identical.

Strictly offline: API calls = 0, LLM calls = 0, cloud compute = 0, new agent
generation = 0. Local LightGBM training is allowed by the Phase 2B spec.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

NL = chr(10)

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2b"
WORK_2A = WS / "work" / "lre_phase2a"
DATA = WORK_2A / "data"
OUT_2A = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
OUT = WS / "outputs" / "earlyeval_phase2b_terminalbench_fixed_scaffold"
ANA = OUT / "analysis"
FOLDS = OUT / "folds"
PRED = OUT / "predictions"
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
VENDOR = REPO / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
ADAPTER_DIR = OUT_2A / "adapter"
RUNTIME_ROOT = WORK / "vendor_runtime"

DATASET_REPO_ID = "yoonholee/terminalbench-trajectories"
DATASET_REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
EARLYEVAL_COMMIT = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"

SCAFFOLD = "terminus-2"
PREDICTOR = "P2B_TerminalBench_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
POLICY_NAME = "current_safe_stop"
POLICY_MODE = "dual"
SCORE_MODE = "calibrated"
SUCCESS_THR = 0.95
FAILURE_THR = 0.95
MIN_STEP = 0
CONSECUTIVE = 1


def set_vendor_env() -> None:
    """Make the unmodified vendored EarlyEval modules importable."""
    os.environ["EARLYEVAL_VENDOR_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    for p in (str(VENDOR), str(WORK), str(ADAPTER_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8",
                    newline=NL)
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text("utf-8"))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def hash_manifest(root: Path, rel_to: Path | None = None,
                  exclude: set[str] | None = None) -> list[str]:
    """Return sorted 'sha256  <relpath>' lines for every file under root."""
    base = Path(rel_to) if rel_to is not None else Path(root)
    skip = set(exclude or ())
    lines = []
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            rel = f.relative_to(base).as_posix()
            if rel in skip:
                continue
            lines.append(f"{sha256_file(f)}  {rel}")
    return lines


def read_manifest(path: Path) -> dict:
    """Parse a '<sha256>  <relative path>' manifest into {rel: sha}."""
    recorded = {}
    if Path(path).exists():
        for line in Path(path).read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    return recorded
