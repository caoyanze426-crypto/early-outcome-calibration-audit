# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - minimal public sample download.

Read-only public dataset access. No LLM/API scientific calls. Downloads a small number of
trajectory files only; never the full 821 MB corpus.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
HF_HOME = WS / "work" / "lre_phase0a" / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

OUT = WS / "outputs" / "late_reversal_early_eval_phase0a" / "public_dataset"
SAMPLES = OUT / "samples"
REPO_ID = "tarsur385/swebench-verified-trajectories"
NL = chr(10)
MODELS = ["claude-opus-4.6", "gpt-5.2-codex", "glm-5-high"]
PER_MODEL = 10


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    all_files = [s.rfilename for s in api.dataset_info(REPO_ID, files_metadata=False).siblings]
    traj = [f for f in all_files if f.endswith(".traj.json")]
    by_model = {}
    for f in traj:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "swebench_verified_raw":
            by_model.setdefault(parts[1], []).append(f)

    manifest = {
        "repo_id": REPO_ID,
        "models_available": sorted(by_model),
        "traj_file_count": len(traj),
        "per_model_file_counts": {m: len(v) for m, v in sorted(by_model.items())},
        "requested_models": MODELS, "per_model_target": PER_MODEL,
        "downloaded": [],
    }
    for model in MODELS:
        files = sorted(by_model.get(model, []))
        if not files:
            manifest["downloaded"].append({"model": model, "status": "MODEL_NOT_PRESENT"})
            continue
        step = max(1, len(files) // PER_MODEL)
        picks = files[::step][:PER_MODEL]
        for rel in picks:
            local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
            src = Path(local)
            dst = SAMPLES / model / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            manifest["downloaded"].append({
                "model": model, "repo_path": rel, "local_path": str(dst),
                "bytes": dst.stat().st_size})

    (OUT / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    print(json.dumps({
        "traj_file_count": len(traj),
        "models_available": sorted(by_model),
        "downloaded_files": len([d for d in manifest["downloaded"] if "local_path" in d]),
        "downloaded_bytes": sum(d.get("bytes", 0) for d in manifest["downloaded"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
