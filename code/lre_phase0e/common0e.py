# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE0E - shared paths and helpers.

Strictly offline: LLM calls = 0, API calls = 0, predictor retraining = 0, new folds
= 0, threshold sweep = 0. Phase 0E re-analyzes the frozen Phase 0B stop-point
predictions and the frozen Phase 0D fold priors, and never modifies them.
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
WORK = WS / "work" / "lre_phase0e"
OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT_0D = WS / "outputs" / "earlyeval_phase0d_cross_agent_calibration"
OUT = WS / "outputs" / "earlyeval_phase0e_prior_shift_decomposition"
ANA = OUT / "analysis"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SUCCESS_SCORE_COL = f"prob_cal_safe_success__{PREDICTOR}"
FAILURE_SCORE_COL = f"prob_cal_safe_failure__{PREDICTOR}"
GEMINI_PRO = "gemini-3-pro"

# Frozen inputs consumed by Phase 0E. Provenance is checked against the artifact
# manifests written by the producing phase itself.
INPUT_FILES = {
    "trajectory_policy_decisions.csv":
        OUT_0B / "predictions" / "trajectory_policy_decisions.csv",
    "trajectory_outcomes.csv": OUT_0B / "analysis" / "trajectory_outcomes.csv",
    "heldout_prefix_predictions_all.parquet":
        OUT_0B / "predictions" / "heldout_prefix_predictions_all.parquet",
    "fold_manifest.csv": OUT_0B / "folds" / "fold_manifest.csv",
    "phase0d_prevalence_shift.csv":
        OUT_0D / "analysis" / "prevalence_shift.csv",
    "phase0d_decision_score_calibration.csv":
        OUT_0D / "analysis" / "decision_score_calibration.csv",
    "phase0d_per_model_reliability.csv":
        OUT_0D / "analysis" / "per_model_reliability.csv",
    "phase0d_prior_shift_correlations.json":
        OUT_0D / "analysis" / "prior_shift_correlations.json",
    "phase0d_heterogeneity.json": OUT_0D / "analysis" / "heterogeneity.json",
    "phase0d_gate.json": OUT_0D / "analysis" / "phase0d_gate.json",
}

PROVENANCE = {
    "phase0b": OUT_0B / "artifact_sha256sums.txt",
    "phase0d": OUT_0D / "artifact_sha256sums.txt",
}

HEAD_CODE = {"success": 0, "failure": 1}


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


def logit(prob, eps: float = 1e-12):
    """Log-odds with an explicit clip; exact 0/1 probabilities do not occur here."""
    import numpy as np

    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_odds(prob):
    """Unclipped log-odds; returns +/-inf at the boundary instead of raising."""
    import numpy as np

    p = np.asarray(prob, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.divide(p, 1.0 - p)
        return np.log(odds)


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


def read_manifest(path: Path) -> dict:
    """Parse a `<sha256>  <relative path>` manifest into {rel: sha}."""
    recorded = {}
    if Path(path).exists():
        for line in Path(path).read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    return recorded
