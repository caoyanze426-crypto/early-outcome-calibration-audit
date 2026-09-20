# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A_D - shared paths, frozen constants and helpers.

Strictly offline reuse of the Phase 1A artifact namespace: no predictor is
trained, no prediction or decision is regenerated, no pair fold is created.
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
NL = chr(10)
WORK_1A = WS / "work" / "lre_phase1a"
WORK = WS / "work" / "lre_phase1a_d"
OUT_1A = WS / "outputs" / "earlyeval_phase1a_same_predictor_transfer"
OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT = WS / "outputs" / "earlyeval_phase1a_d_target_persistence"
ANA = OUT / "analysis"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
POLICY_NAME = "current_safe_stop"

# Phase 1A frozen pair-fold eligibility (section 14 of Phase 1A).
PAIR_MIN_DECISIONS = 20
# Phase 1A_D section 7: task-half eligibility.
HALF_MIN_DECISIONS = 10
HALF_MIN_ELIGIBLE_FOLDS = 5

# Phase 1A_D section 5: task-cluster bootstrap.
REPLICATES = 5000
BOOTSTRAP_SEED = 42031

# Phase 1A_D section 9 gate thresholds (frozen before any result was inspected).
TARGET_MIN_OCCURRENCES = 7
TARGET_MEDIAN_ABS_GAP_MIN = 0.08
TARGET_SAME_SIGN_FRACTION_MIN = 0.80
JACKKNIFE_MIN_ABS_MEDIAN_MIN = 0.06
HALF_MIN_ABS_MEDIAN_MIN = 0.06

# Phase 1A_D section 0: frozen primary targets and the sensitivity-only target.
TARGET_1 = {"agent_model": "gpt-5-mini", "head": "success",
            "scope": "PRIMARY_TARGET_1"}
TARGET_2 = {"agent_model": "claude-opus-4.6", "head": "failure",
            "scope": "PRIMARY_TARGET_2"}
SENSITIVITY_TARGET = {"agent_model": "minimax-m2.5-high", "head": "failure",
                      "scope": "SENSITIVITY_ONLY"}
TARGETS = [TARGET_1, TARGET_2, SENSITIVITY_TARGET]

MODELS = [
    "claude-4.5-haiku-high",
    "claude-4.5-opus-high",
    "claude-opus-4.6",
    "gemini-3-flash-high",
    "gemini-3-pro",
    "glm-5-high",
    "gpt-5-mini",
    "gpt-5.2-codex",
    "gpt-5.2-high",
    "minimax-m2.5-high",
]
HEAD_ORDER = ("success", "failure")
HEAD_SIGN = {"success": 1.0, "failure": -1.0}

# Inputs consumed from the Phase 1A namespace (section 1).
PHASE1A_INPUTS = [
    "analysis/pairwise_metrics.csv",
    "analysis/target_persistence.csv",
    "folds/pair_fold_manifest.csv",
    "predictions/pair_policy_decisions",
    "predictions/per_pair_target_predictions",
    "input_hashes.json",
]
PHASE0B_INPUT = "analysis/trajectory_outcomes.csv"


def sha256_file(path, chunk: int = 1 << 20) -> str:
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


def write_json(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8",
                 newline=NL)
    return p


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def read_manifest(path) -> dict:
    """Parse a `<sha256>  <relative path>` manifest into {rel: sha}."""
    recorded = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    return recorded


def as_builtin(value):
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
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [as_builtin(v) for v in value.tolist()]
    if isinstance(value, float) and value != value:
        return None
    return value


def sigmoid(z):
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(prob, eps: float = 1e-12):
    import numpy as np

    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def log_odds(prob) -> float:
    """Unclipped log-odds; +/-inf at the boundary (Phase 0E convention)."""
    import numpy as np

    p = float(prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(p / (1.0 - p)))


def safe_ratio(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def percentile_ci(values, lo=2.5, hi=97.5) -> dict:
    import numpy as np

    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n_valid_replicates": 0, "ci_lower": None, "ci_upper": None,
                "median": None}
    return {
        "n_valid_replicates": int(arr.size),
        "ci_lower": float(np.percentile(arr, lo)),
        "ci_upper": float(np.percentile(arr, hi)),
        "median": float(np.median(arr)),
    }


def median_or_none(values):
    import numpy as np

    arr = [v for v in values if v is not None]
    return float(np.median(arr)) if arr else None


def rel_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file())


def set_path() -> None:
    for p in (WORK, WORK_1A):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
