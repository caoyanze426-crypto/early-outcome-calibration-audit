# -*- coding: utf-8 -*-
"""Phase 0B step 1: download the frozen public trajectory corpus and hash it. Offline only."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from common import (  # noqa: E402
    DATASET_REPO_ID,
    DATASET_REVISION,
    HF_HOME,
    OUT,
    ensure_dirs,
    sha256_file,
    write_json,
)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

DATA_DIR = OUT / "dataset"
NL = chr(10)


def main() -> int:
    ensure_dirs(DATA_DIR)
    snapshot_dir = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["swebench_verified_raw/**", "README.md", ".gitattributes"],
        max_workers=8,
    )
    snap = Path(snapshot_dir)
    root = snap / "swebench_verified_raw"
    traj_files = sorted(root.rglob("*.traj.json"))
    per_model = Counter()
    per_model_bytes = Counter()
    total_bytes = 0
    manifest_lines = []
    zero_byte = []
    name_mismatch = []
    for f in traj_files:
        rel = f.relative_to(snap).as_posix()
        parts = rel.split("/")
        model = parts[1] if len(parts) > 1 else "__UNKNOWN__"
        size = f.stat().st_size
        total_bytes += size
        per_model[model] += 1
        per_model_bytes[model] += size
        if size == 0:
            zero_byte.append(rel)
        if f.name != f.parent.name + ".traj.json":
            name_mismatch.append(rel)
        manifest_lines.append(f"{sha256_file(f)}  {rel}")
    manifest_lines.sort()
    (DATA_DIR / "dataset_sha256sums.txt").write_text(
        NL.join(manifest_lines) + NL, "utf-8", newline=NL)
    instance_sets = {}
    for model in sorted(per_model):
        instance_sets[model] = sorted(p.name for p in (root / model).iterdir() if p.is_dir())
    sets = [set(v) for v in instance_sets.values()]
    common_instances = set.intersection(*sets) if sets else set()
    all_instances = set().union(*sets) if sets else set()
    payload = {
        "repo_id": DATASET_REPO_ID,
        "repo_type": "dataset",
        "revision_requested": DATASET_REVISION,
        "snapshot_dir": str(snap),
        "snapshot_subdir": str(root),
        "trajectory_file_count": len(traj_files),
        "trajectory_total_bytes": total_bytes,
        "trajectory_total_gib": round(total_bytes / (1024 ** 3), 6),
        "models": sorted(per_model),
        "model_count": len(per_model),
        "per_model_file_count": {m: per_model[m] for m in sorted(per_model)},
        "per_model_bytes": {m: per_model_bytes[m] for m in sorted(per_model)},
        "instances_per_model_count": {m: len(instance_sets[m]) for m in sorted(instance_sets)},
        "instances_common_to_all_models": len(common_instances),
        "instances_union": len(all_instances),
        "zero_byte_files": zero_byte,
        "filename_parent_mismatch": name_mismatch,
        "sha256_manifest": "dataset/dataset_sha256sums.txt",
        "sha256_manifest_lines": len(manifest_lines),
        "no_silent_substitution": True,
        "download_tool": "huggingface_hub.snapshot_download",
    }
    write_json(DATA_DIR / "dataset_manifest.json", payload)
    printable = {k: v for k, v in payload.items() if k != "per_model_bytes"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
