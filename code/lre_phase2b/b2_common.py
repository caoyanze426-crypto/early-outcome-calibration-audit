# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B - shared paths, helpers, frozen-vendor bootstrap.

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
SAFE_LABEL_MIN_STEP = 10  # repository CLI default --safe-label-min-step 10

SUCCESS_SCORE_COL = f"prob_cal_safe_success__{PREDICTOR}"
FAILURE_SCORE_COL = f"prob_cal_safe_failure__{PREDICTOR}"

# Section 5 primary missingness eligibility thresholds (frozen before training).
ELIG_USABLE = 100
ELIG_SUCCESS = 20
ELIG_FAILURE = 20
ELIG_USABLE_FRACTION = 0.60
ELIG_MISSINGNESS_GAP = 0.20
# Section 6 minimum model count gate.
MIN_MODELS = 12
# Section 15 target-specific signal thresholds.
MIN_DECISIONS = 20
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED_BASE = 43000
ROBUST_GAP = 0.08
LARGE_GAP = 0.10
HEAD_CODE = {"success": 0, "failure": 1}


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
    path.write_text(
        json.dumps(as_builtin(payload), ensure_ascii=False, indent=2), "utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text("utf-8"))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def as_builtin(value):
    """Convert numpy scalars/arrays into plain JSON-serializable objects."""
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


def safe_ratio(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def log_odds(prob):
    """Unclipped log-odds; +/-inf at the boundary instead of raising."""
    import numpy as np

    p = np.asarray(prob, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.divide(p, 1.0 - p))


def logit(prob, eps: float = 1e-12):
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
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
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
    if any(v is None for v in xs) or any(v is None for v in ys):
        return None
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


def provider_family(model_label: str) -> str:
    """Derive a provider family from the exact dataset model label only."""
    s = str(model_label)
    low = s.lower()
    if "@anthropic" in low or "claude" in low:
        return "Anthropic"
    if "gemini" in low or "@google" in low or "@gemini" in low:
        return "Google/Gemini"
    if "@xai" in low or "grok" in low:
        return "xAI"
    if "@openai" in low or "gpt-" in low or "openai/" in low:
        return "OpenAI"
    if "qwen" in low:
        return "Qwen/Alibaba"
    if "kimi" in low or "moonshot" in low:
        return "Moonshot"
    if "glm" in low or "z-ai" in low:
        return "Zhipu/Z.ai"
    if "minimax" in low:
        return "MiniMax"
    if "deepseek" in low:
        return "DeepSeek"
    if "fireworks" in low:
        return "Fireworks-hosted"
    if "together" in low:
        return "Together-hosted"
    return "Other/Unknown"
