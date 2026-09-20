# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0B - shared paths and small helpers.

Strictly offline. No LLM / model API calls anywhere in this package.
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
WORK = WS / "work" / "lre_phase0b"
WORK_PHASE0A = WS / "work" / "lre_phase0a"
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
VENDOR = REPO / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
OUT = WS / "outputs" / "late_reversal_early_eval_phase0b_signal_hunt"
HF_HOME = WORK / "hf_home"
RUNTIME_ROOT = WORK / "vendor_runtime"

DATASET_REPO_ID = "tarsur385/swebench-verified-trajectories"
DATASET_REVISION = "773748a7c1222e8a642a7059821498e14293562a"
EARLYEVAL_COMMIT = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"


def set_vendor_env() -> None:
    """Make the unmodified vendored EarlyEval modules importable."""
    os.environ["EARLYEVAL_VENDOR_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    for p in (str(VENDOR), str(WORK), str(WORK_PHASE0A)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # `earlyeval` is imported as a package (earlyeval.core..., earlyeval.policies...),
    # so the repository root that contains the package directory must be importable.
    # Appended last so it cannot shadow the vendored top-level modules (config,
    # feature_engineer, trainer, ...).
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text("utf-8"))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def hash_manifest(root: Path, rel_to: Path | None = None) -> list[str]:
    """Return sorted 'sha256  <relpath>' lines for every file under root."""
    base = Path(rel_to) if rel_to is not None else Path(root)
    lines = []
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            rel = f.relative_to(base).as_posix()
            lines.append(f"{sha256_file(f)}  {rel}")
    return lines
