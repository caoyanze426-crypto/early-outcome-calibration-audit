# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1B - shared paths, frozen constants and helpers.

Strictly offline: the frozen Phase 1A per-prefix predictions are re-scored by the
stopping policy at four symmetric thresholds. No predictor is trained, no feature
is rebuilt, no calibrator is refit, no prediction is regenerated.
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

WORK_0B = WS / "work" / "lre_phase0b"
WORK_1A = WS / "work" / "lre_phase1a"
WORK_1AD = WS / "work" / "lre_phase1a_d"
WORK = WS / "work" / "lre_phase1b"

OUT_1A = WS / "outputs" / "earlyeval_phase1a_same_predictor_transfer"
OUT_1AD = WS / "outputs" / "earlyeval_phase1a_d_target_persistence"
OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT = WS / "outputs" / "earlyeval_phase1b_threshold_robustness"
ANA = OUT / "analysis"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
POLICY_NAME = "current_safe_stop"
SCORE_MODE = "calibrated"
POLICY_MODE = "dual"
MIN_STEP = 0
CONSECUTIVE = 1

# Section 2 - exactly four symmetric policies, in the order declared.
THRESHOLDS = [0.900, 0.925, 0.950, 0.975]
THRESHOLD_LABELS = ["T1", "T2", "T3", "T4"]
REPRODUCTION_THRESHOLD = 0.950

# Section 6 - occurrence eligibility.
MIN_DECISIONS = 20

# Section 8 - task-cluster bootstrap.
REPLICATES = 2000
BOOTSTRAP_SEED_BASE = 42100
BOOTSTRAP_SEED_STEP = 10

# Section 10 - per-threshold robustness criteria.
ROBUST_MIN_OCCURRENCES = 7
ROBUST_MEDIAN_ABS_GAP_MIN = 0.06
ROBUST_SAME_SIGN_FRACTION_MIN = 0.75
ROBUST_MIN_THRESHOLDS = 3

TARGET_1 = {"target_index": 0, "scope": "TARGET_1",
            "agent_model": "gpt-5-mini", "head": "success",
            "phase1a_d_scope": "PRIMARY_TARGET_1",
            "phase1a_d_median_signed_gap": 0.1377462284504496,
            "phase1a_d_eligible_occurrences": 9}
TARGET_2 = {"target_index": 1, "scope": "TARGET_2",
            "agent_model": "claude-opus-4.6", "head": "failure",
            "phase1a_d_scope": "PRIMARY_TARGET_2",
            "phase1a_d_median_signed_gap": 0.11073740121815578,
            "phase1a_d_eligible_occurrences": 8}
TARGETS = [TARGET_1, TARGET_2]

HEAD_SIGN = {"success": 1.0, "failure": -1.0}

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

PHASE1A_INPUTS = [
    "predictions/per_pair_target_predictions",
    "predictions/pair_policy_decisions",
    "folds/pair_fold_manifest.csv",
    "analysis/pairwise_metrics.csv",
]
PHASE1AD_INPUTS = [
    "analysis/target_occurrences.csv",
    "analysis/primary_gate.json",
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
    return {"n_valid_replicates": int(arr.size),
            "ci_lower": float(np.percentile(arr, lo)),
            "ci_upper": float(np.percentile(arr, hi)),
            "median": float(np.median(arr))}


def sign_of(value) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def bootstrap_seed(threshold_index: int, target_index: int) -> int:
    """Section 8: seed = 42100 + threshold_index*10 + target_index."""
    return (BOOTSTRAP_SEED_BASE + threshold_index * BOOTSTRAP_SEED_STEP
            + target_index)


def rel_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file())


def set_vendor_env() -> None:
    """Import contract identical to Phase 0B / Phase 1A."""
    for p in (WORK_0B, WORK_1A):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import common as common0b
    common0b.set_vendor_env()
