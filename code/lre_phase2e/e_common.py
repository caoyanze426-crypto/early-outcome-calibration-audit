# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2E - shared paths and frozen helpers.

Strictly offline: API = 0, LLM = 0, training = 0, cloud = 0.
"""
from __future__ import annotations
import os

import hashlib
import json
import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase2e"
OUT = WS / "outputs" / "earlyeval_phase2e_exact_target_cross_benchmark"
ANA = OUT / "analysis"

OUT_1A = WS / "outputs" / "earlyeval_phase1a_same_predictor_transfer"
OUT_1AD = WS / "outputs" / "earlyeval_phase1a_d_target_persistence"
OUT_1B = WS / "outputs" / "earlyeval_phase1b_threshold_robustness"
OUT_2B = WS / "outputs" / "earlyeval_phase2b_terminalbench_fixed_scaffold"
OUT_2BD = WS / "outputs" / "earlyeval_phase2b_d_threshold_diagnostic"
MANIFEST_ROOTS = (OUT_1A, OUT_1AD, OUT_1B, OUT_2B, OUT_2BD)

SWE_SNAPSHOT = (WS / "work" / "lre_phase0b" / "hf_home" / "hub"
                / "datasets--tarsur385--swebench-verified-trajectories"
                / "snapshots" / "773748a7c1222e8a642a7059821498e14293562a")
SWE_RAW = SWE_SNAPSHOT / "swebench_verified_raw"
TB_PARQUET = (WS / "work" / "lre_phase2a" / "data" / "data")
TB_README = WS / "work" / "lre_phase2a" / "data" / "README.md"

SWE_DATASET_REVISION = "773748a7c1222e8a642a7059821498e14293562a"
TB_DATASET_REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"

PRIMARY_THRESHOLD = 0.950
THRESHOLDS = (0.900, 0.925, 0.950)
MIN_DECISIONS = 20
PERSIST_GAP = 0.08
COLLAPSE_GAP = 0.04

TARGETS = (
    {"scope": "TARGET_1", "swe_model": "gpt-5-mini", "head": "success",
     "tb_model": "gpt-5-mini@openai"},
    {"scope": "TARGET_2", "swe_model": "claude-opus-4.6", "head": "failure",
     "tb_model": "claude-opus-4-6@anthropic"},
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def as_builtin(value):
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        np = None
    if isinstance(value, dict):
        return {k: as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.ndarray):
            return [as_builtin(v) for v in value.tolist()]
    if isinstance(value, float) and value != value:
        return None
    return value


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(as_builtin(payload), ensure_ascii=False, indent=2), "utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text("utf-8"))


def read_manifest(path: Path) -> dict:
    recorded = {}
    if Path(path).exists():
        for line in Path(path).read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    return recorded


def ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
