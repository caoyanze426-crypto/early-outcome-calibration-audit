# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE1A - shared paths and helpers.

Strictly offline: LLM calls = 0, API calls = 0, new agent trajectories = 0,
threshold sweep = 0. Phase 1A re-uses the frozen Phase 0B adapter, encoder and
prefix lineage, and never modifies Phase 0B / 0C / 0D / 0E or the EarlyEval clone.

The Phase 0B module `common` is imported for `set_vendor_env()` so that the
vendored EarlyEval modules are importable exactly as Phase 0B imported them.
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
NL = chr(10)
WORK_0B = WS / "work" / "lre_phase0b"
WORK = WS / "work" / "lre_phase1a"
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
VENDOR = REPO / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"

OUT_0B = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
OUT = WS / "outputs" / "earlyeval_phase1a_same_predictor_transfer"
ANA = OUT / "analysis"
FOLDS = OUT / "folds"
PRED = OUT / "predictions"
PRED_PAIR = PRED / "per_pair_target_predictions"
PRED_DEC = PRED / "pair_policy_decisions"

ENC_DIR = WORK_0B / "encoded"
PREFIX_TABLE = WORK_0B / "prefix_table"
ROW_INDEX = WORK_0B / "row_index.parquet"

DATASET_REVISION = "773748a7c1222e8a642a7059821498e14293562a"
EARLYEVAL_COMMIT = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"

PREDICTOR = "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE"
SUCCESS_SCORE_COL = f"prob_cal_safe_success__{PREDICTOR}"
FAILURE_SCORE_COL = f"prob_cal_safe_failure__{PREDICTOR}"

# Frozen policy (section 8).
POLICY_NAME = "current_safe_stop"
SUCCESS_THR = 0.95
FAILURE_THR = 0.95
MIN_STEP = 0
CONSECUTIVE = 1
SAFE_LABEL_MIN_STEP = 10

REPLICATES = 2000
SEED = 42

# Pair-fold gate (section 16).
PAIR_MIN_DECISIONS = 20
PAIR_MIN_ELIGIBLE_FOLDS = 20
PAIR_MEDIAN_DIFF_MIN = 0.05
PAIR_CI_LOWER_MIN = 0.03

# Target persistence (section 17).
PERSIST_MIN_OCCURRENCES = 7
PERSIST_MEDIAN_ABS_GAP_MIN = 0.08
PERSIST_SAME_SIGN_FRACTION = 7.0 / 9.0

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

GEMINI_PRO = "gemini-3-pro"
MIRROR_DROP = "gpt-5.2-high"
MIRROR_KEEP = "gpt-5.2-codex"


def set_vendor_env() -> None:
    """Delegate to the Phase 0B helper so the import contract is identical."""
    if str(WORK_0B) not in sys.path:
        sys.path.insert(0, str(WORK_0B))
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    import common as common0b  # noqa: E402  (phase 0B module)
    common0b.set_vendor_env()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha1() if False else hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    import json
    from pathlib import Path as _P

    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8",
                 newline=NL)
    return p


def read_json(path):
    import json
    from pathlib import Path as _P

    return json.loads(_P(path).read_text("utf-8"))


def ensure_dirs(*paths) -> None:
    from pathlib import Path as _P

    for p in paths:
        _P(p).mkdir(parents=True, exist_ok=True)


def read_manifest(path) -> dict:
    """Parse a `<sha256>  <relative path>` manifest into {rel: sha}."""
    from pathlib import Path as _P

    recorded = {}
    p = _P(path)
    if p.exists():
        for line in p.read_text("utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                recorded[rel.strip()] = h.strip()
    return recorded


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


def log_odds(prob):
    """Unclipped log-odds; +/-inf at the boundary instead of raising."""
    import numpy as np

    p = np.asarray(prob, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.divide(p, 1.0 - p))


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


def pair_id_of(a: str, b: str) -> str:
    return f"pair-{MODELS.index(a):02d}{MODELS.index(b):02d}"


def all_pairs() -> list[tuple[str, str]]:
    out = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            out.append((MODELS[i], MODELS[j]))
    return out
