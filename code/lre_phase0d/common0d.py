# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0D - shared paths and helpers.

Strictly offline: LLM calls = 0, API calls = 0, predictor retraining = 0, new folds
= 0, threshold sweep = 0. Phase 0D only re-analyzes frozen Phase 0B artifacts and
never modifies them.
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
WORK = WS / "work" / "lre_phase0d"
OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT = WS / "outputs" / "earlyeval_phase0d_cross_agent_calibration"
ANA = OUT / "analysis"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SUCCESS_SCORE_COL = f"prob_cal_safe_success__{PREDICTOR}"
FAILURE_SCORE_COL = f"prob_cal_safe_failure__{PREDICTOR}"
GEMINI_PRO = "gemini-3-pro"

INPUT_FILES = {
    "trajectory_policy_decisions.csv":
        OUT_0B / "predictions" / "trajectory_policy_decisions.csv",
    "trajectory_outcomes.csv": OUT_0B / "analysis" / "trajectory_outcomes.csv",
    "per_model_summary.csv": OUT_0B / "analysis" / "per_model_summary.csv",
    "fold_manifest.csv": OUT_0B / "folds" / "fold_manifest.csv",
    "heldout_prefix_predictions_all.parquet":
        OUT_0B / "predictions" / "heldout_prefix_predictions_all.parquet",
    "policy_summary.csv": OUT_0B / "predictions" / "policy_summary.csv",
    "error_summary.json": OUT_0B / "analysis" / "error_summary.json",
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
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [as_builtin(v) for v in value.tolist()]
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def safe_ratio(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> dict:
    """95% Wilson score interval; no normal approximation."""
    if n <= 0:
        return {"n": 0, "k": int(k), "rate": None, "wilson_lower": None,
                "wilson_upper": None}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)) / denom
    return {
        "n": int(n),
        "k": int(k),
        "rate": float(p),
        "wilson_lower": float(max(0.0, center - half)),
        "wilson_upper": float(min(1.0, center + half)),
    }


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


def _rankdata(values):
    """Average-rank transform (ties share the mean rank)."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho with average ranks; None when undefined."""
    import numpy as np

    xs = [v for v in x]
    ys = [v for v in y]
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(((rx ** 2).sum() * (ry ** 2).sum()) ** 0.5)
    if denom == 0.0:
        return None
    return float((rx * ry).sum() / denom)
