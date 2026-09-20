# -*- coding: utf-8 -*-
"""Phase 0B scale probe: measure prefix text volume and TF-IDF token counts on the 30-file sample."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from common import OUT, WS, set_vendor_env

set_vendor_env()

import pandas as pd  # noqa: E402

import config  # noqa: E402
from feature_engineer import TFIDF_ACTION_FEEDBACK  # noqa: E402
from mapping_adapter import adapt_messages  # noqa: E402
from prefix_builder import build_prefix_samples_for_trajectory  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

SAMPLES = OUT.parent / "late_reversal_early_eval_phase0a" / "public_dataset" / "samples"
MAX_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main() -> int:
    files = sorted(SAMPLES.rglob("*.traj.json"))[:MAX_TRAJ]
    rows = []
    for f in files:
        doc = json.loads(f.read_text("utf-8"))
        info = doc.get("info") or {}
        model = (info.get("docent") or {}).get("model_label") or f.parent.name
        rec = pd.Series({
            "traj_id": str(model) + "::" + str(doc.get("instance_id")),
            "instance_id": doc.get("instance_id"),
            "resolved": bool(info.get("resolved")),
            "model": str(model),
            "messages": json.dumps(adapt_messages(doc.get("messages") or []),
                                   ensure_ascii=False),
        })
        rows.extend(build_prefix_samples_for_trajectory(rec))
    df = pd.DataFrame(rows)
    cols = [df[c].fillna("").astype(str) for c in TFIDF_ACTION_FEEDBACK.values()]
    n_prefix = len(df)
    out = {
        "trajectories": len(files),
        "prefix_rows": n_prefix,
        "per_text_column": {},
        "total_prefix_rows_extrapolated_5000": None,
    }
    total_tokens = 0
    total_unique_per_row = 0
    for name, col in TFIDF_ACTION_FEEDBACK.items():
        texts = df[col].fillna("").astype(str)
        chars = int(texts.str.len().sum())
        vec = CountVectorizer(ngram_range=config.TFIDF_NGRAM_RANGE,
                              dtype="int32", binary=True)
        X = vec.fit_transform(texts)
        tokens = int(X.sum())
        uniq = X.getnnz(axis=1)
        total_tokens += tokens
        total_unique_per_row += int(uniq.sum())
        out["per_text_column"][name] = {
            "text_column": col,
            "chars": chars,
            "chars_per_row": round(chars / n_prefix, 1),
            "vocab_on_sample": int(X.shape[1]),
            "token_occurrences": tokens,
            "mean_unique_terms_per_row": round(float(uniq.mean()), 1),
            "max_unique_terms_per_row": int(uniq.max()),
        }
        del X, vec
    out["total_token_occurrences_sample"] = total_tokens
    out["total_unique_entries_sample"] = total_unique_per_row
    scale = 5000.0 / max(len(files), 1)
    out["extrapolated_5000"] = {
        "prefix_rows": round(n_prefix * scale),
        "token_occurrences": round(total_tokens * scale),
        "sparse_matrix_nnz": round(total_unique_per_row * scale),
        "nnz_gib_float32_int32": round(total_unique_per_row * scale * 8 / (1024 ** 3), 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
