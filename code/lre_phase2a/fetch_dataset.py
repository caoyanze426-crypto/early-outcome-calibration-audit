# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2A - public dataset metadata + bounded download.

Only public retrieval is performed: dataset metadata (revision, license, file
inventory, byte sizes) and, when requested, the dataset snapshot itself. No API
key is used, no LLM is called, no cloud compute is rented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
OUT = WS / "outputs" / "earlyeval_phase2a_terminalbench_coverage"
DATA = WS / "work" / "lre_phase2a" / "data"
REPO_ID = "yoonholee/terminalbench-trajectories"


def human(n) -> str:
    if n is None:
        return "n/a"
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.2f} {unit}"
        v /= 1024
    return f"{v:.2f} TiB"


def plain(value):
    """Deterministically coerce hub card objects into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    for attr in ("to_dict", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return plain(fn())
            except Exception:  # noqa: BLE001
                pass
    if hasattr(value, "__dict__"):
        return plain({k: v for k, v in vars(value).items()
                      if not k.startswith("_")})
    return str(value)


def meta() -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    siblings = getattr(info, "siblings", None) or []
    files = []
    for s in siblings:
        files.append({
            "path": getattr(s, "rfilename", None),
            "size": getattr(s, "size", None),
        })
    files.sort(key=lambda r: (-(r["size"] or 0)))
    total = sum((r["size"] or 0) for r in files)
    card = getattr(info, "cardData", None) or {}
    payload = {
        "repo_id": REPO_ID,
        "revision_requested": "main",
        "sha": getattr(info, "sha", None),
        "last_modified": str(getattr(info, "lastModified", None)),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "tags": getattr(info, "tags", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "private": getattr(info, "private", None),
        "gated": getattr(info, "gated", None),
        "file_count": len(files),
        "total_bytes": total,
        "total_human": human(total),
        "files": files,
        "card_data": plain(card),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (WS / "work" / "lre_phase2a").mkdir(parents=True, exist_ok=True)
    (WS / "work" / "lre_phase2a" / "remote_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("files", "card_data")}, indent=2))
    print("top 10 largest files:")
    for r in files[:10]:
        print(f"  {human(r['size']):>10}  {r['path']}")
    return 0


def download(revision: str | None) -> int:
    from huggingface_hub import snapshot_download

    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kwargs = dict(repo_id=REPO_ID, repo_type="dataset", local_dir=str(DATA))
    if revision:
        kwargs["revision"] = revision
    path = snapshot_download(**kwargs)
    secs = round(time.time() - t0, 1)
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    manifest = {
        "repo_id": REPO_ID, "revision": revision, "local_dir": str(path),
        "file_count": len(files), "total_bytes": total,
        "total_human": human(total),
        "wall_clock_seconds": secs,
        "files": [{"path": p.relative_to(Path(path)).as_posix(),
                   "bytes": p.stat().st_size} for p in files],
    }
    dest = WS / "work" / "lre_phase2a" / "local_download_manifest.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"},
                     indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["meta", "download"])
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    return meta() if a.stage == "meta" else download(a.revision)


if __name__ == "__main__":
    sys.exit(main())
