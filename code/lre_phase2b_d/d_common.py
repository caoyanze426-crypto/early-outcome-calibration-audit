# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B_D - shared paths and frozen Phase 2B helpers.

Strictly offline: API calls = 0, LLM calls = 0, cloud compute = 0,
no predictor retraining, no new trajectories.
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
B2 = WS / "work" / "lre_phase2b"
WORK = WS / "work" / "lre_phase2b_d"
OUT_2B = WS / "outputs" / "earlyeval_phase2b_terminalbench_fixed_scaffold"
OUT = WS / "outputs" / "earlyeval_phase2b_d_threshold_diagnostic"
ANA = OUT / "analysis"
DECS = WORK / "decisions"

B2_TAG_MANIFESTS = (OUT_2B / "artifact_sha256sums.txt",
                    OUT_2B / "manifest.json")
PRED_2B = OUT_2B / "predictions" / "heldout_prefix_predictions"
DEC_2B = OUT_2B / "predictions" / "policy_decisions"

THRESHOLDS = (0.900, 0.925, 0.950)
ANCHOR = 0.950
DEC_TOL = 1e-12
MIN_ELIGIBLE_TARGETS = 8

if str(B2) not in sys.path:
    sys.path.insert(0, str(B2))

from b2_common import (  # noqa: E402,F401
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED_BASE, FAILURE_SCORE_COL, HEAD_CODE,
    LARGE_GAP, MIN_DECISIONS, PREDICTOR, ROBUST_GAP, SUCCESS_SCORE_COL,
    as_builtin, ensure_dirs, log_odds, logit, percentile_ci, provider_family,
    read_json, read_manifest, set_vendor_env, sha256_file, sha256_text,
    sigmoid, spearman, write_json)


def thr_tag(thr: float) -> str:
    return "thr-%.3f" % float(thr)


def anchor_tag() -> str:
    return thr_tag(ANCHOR)
