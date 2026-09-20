# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0C - shared paths and helpers.

Strictly offline: LLM calls = 0, API calls = 0, predictor retraining = 0, threshold
sweep = 0. Phase 0C only re-analyzes the frozen Phase 0B artifacts and never
modifies them.
"""
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

NL = chr(10)

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
WORK = WS / "work" / "lre_phase0c"
WORK_0B = WS / "work" / "lre_phase0b"
OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT = WS / "outputs" / "late_reversal_early_eval_phase0c"
ANA = OUT / "analysis"

DATASET_SNAPSHOT = (
    WS / "work" / "lre_phase0b" / "hf_home" / "hub"
    / "datasets--tarsur385--swebench-verified-trajectories"
    / "snapshots" / "773748a7c1222e8a642a7059821498e14293562a"
    / "swebench_verified_raw"
)

INPUT_FILES = {
    "trajectory_policy_decisions.csv":
        OUT_0B / "predictions" / "trajectory_policy_decisions.csv",
    "trajectory_outcomes.csv": OUT_0B / "analysis" / "trajectory_outcomes.csv",
    "late_reversal_signatures.csv":
        OUT_0B / "analysis" / "late_reversal_signatures.csv",
    "per_model_summary.csv": OUT_0B / "analysis" / "per_model_summary.csv",
    "detector_spec.json": OUT_0B / "features" / "detector_spec.json",
    "adapter_failures.csv": OUT_0B / "adapter" / "adapter_failures.csv",
    "error_summary.json": OUT_0B / "analysis" / "error_summary.json",
    "signal_gate.json": OUT_0B / "analysis" / "signal_gate.json",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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


def as_builtin(value):
    """Convert numpy scalars/arrays into plain JSON-serializable Python objects."""
    import numpy as np

    if isinstance(value, dict):
        return {k: as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [as_builtin(v) for v in value.tolist()]
    return value


def safe_ratio(num, den):
    import math

    if den is None or den == 0:
        return None
    return float(num) / float(den)


def percentile_ci(values, lo=2.5, hi=97.5):
    import numpy as np

    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n_valid_replicates": 0, "ci_lower": None, "ci_upper": None,
                "median": None}
    return {
        "n_valid_replicates": int(arr.size),
        "ci_lower": float(np.percentile(arr, lo)),
        "ci_upper": float(np.percentile(arr, hi)),
        "median": float(np.median(arr)),
    }


def math_isnan(value) -> bool:
    import math

    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return value is None
