# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - Hugging Face dataset metadata probe.

Read-only public metadata access. No LLM/API scientific calls. No full-corpus download.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
OUT = WS / "outputs" / "late_reversal_early_eval_phase0a" / "public_dataset"
REPO_ID = "tarsur385/swebench-verified-trajectories"
NL = chr(10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    siblings = []
    total_bytes = 0
    for s in info.siblings or []:
        size = getattr(s, "size", None) or 0
        total_bytes += size
        siblings.append({"rfilename": s.rfilename, "size": size})
    meta = {
        "repo_id": REPO_ID,
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "card_data": json.loads(json.dumps(getattr(info, "card_data", None), default=str)),
        "license_from_tags": [t for t in (info.tags or []) if "license" in t.lower()],
        "all_tags": list(info.tags or []),
        "num_files": len(siblings),
        "total_bytes_listed": total_bytes,
        "total_gb_listed": round(total_bytes / 1e9, 3),
        "siblings": siblings,
    }
    (OUT / "metadata_raw.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           "utf-8", newline=NL)

    # lightweight README fetch (small public file)
    readme = None
    try:
        p = hf_hub_download(REPO_ID, "README.md", repo_type="dataset")
        readme = Path(p).read_text("utf-8", "replace")
        (OUT / "README_dataset.md").write_text(readme, "utf-8", newline="")
    except Exception as exc:  # noqa: BLE001
        readme = "FETCH_FAILED: " + type(exc).__name__ + ": " + str(exc)
        (OUT / "README_dataset.md").write_text(readme, "utf-8", newline="")

    top = {}
    for s in siblings:
        head = s["rfilename"].split("/")[0]
        top.setdefault(head, {"files": 0, "bytes": 0})
        top[head]["files"] += 1
        top[head]["bytes"] += s["size"]
    summary = {
        "repo_id": REPO_ID, "sha": meta["sha"], "num_files": len(siblings),
        "total_gb_listed": meta["total_gb_listed"],
        "license_from_tags": meta["license_from_tags"],
        "top_level_entries": {k: {"files": v["files"],
                                  "gb": round(v["bytes"] / 1e9, 4)}
                              for k, v in sorted(top.items())},
        "readme_head": readme[:2500] if readme else None,
    }
    (OUT / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       "utf-8", newline=NL)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
