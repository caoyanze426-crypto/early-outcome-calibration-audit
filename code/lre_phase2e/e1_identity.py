# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2E - Section 2: model identity qualification.

Reads only frozen dataset/model metadata:
  * SWE side: per-trajectory info.config.model.model_name and
    info.docent.model_label from the frozen dataset snapshot.
  * TB side : the frozen dataset `model` column (format <slug>@<provider>)
    and the dataset card's own statement of that format.
No identity is inferred from name similarity.
"""
from __future__ import annotations

import collections
import json
import time

import pandas as pd

from e_common import (ANA, OUT, SWE_DATASET_REVISION, SWE_RAW, TARGETS,
                      TB_DATASET_REVISION, TB_PARQUET, TB_README, WORK,
                      ensure_dirs, sha256_file, write_json)


def swe_metadata(model_label: str) -> dict:
    root = SWE_RAW / model_label
    names = collections.Counter()
    docent = collections.Counter()
    model_types = collections.Counter()
    files = unreadable = 0
    for inst in sorted(root.iterdir()):
        for f in sorted(inst.iterdir()):
            files += 1
            try:
                j = json.loads(f.read_text("utf-8"))
            except Exception:  # noqa: BLE001
                unreadable += 1
                continue
            info = j.get("info", {})
            cfg = info.get("config", {}) or {}
            model = cfg.get("model", {}) or {}
            names[model.get("model_name")] += 1
            model_types[cfg.get("model_type")] += 1
            docent[(info.get("docent") or {}).get("model_label")] += 1
    return {
        "label": model_label,
        "label_dir": str(root),
        "trajectory_files": int(files),
        "unreadable_files": int(unreadable),
        "canonical_model_name_counts": dict(names),
        "docent_model_label_counts": dict(docent),
        "harness_model_type_counts": dict(model_types),
        "distinct_canonical_model_names": int(len(names)),
    }


def tb_labels() -> dict:
    frames = []
    for p in sorted(TB_PARQUET.glob("train-*.parquet")):
        frames.append(pd.read_parquet(p, columns=["agent", "model"]))
    df = pd.concat(frames, ignore_index=True)
    combos = (df.drop_duplicates().sort_values(["model", "agent"]))
    by_model = {}
    for m, part in combos.groupby("model", sort=True):
        by_model[str(m)] = sorted(part["agent"].astype(str).tolist())
    return {"rows": int(len(df)), "by_model": by_model}


def split_label(label: str) -> dict:
    if "@" in label:
        slug, provider = label.rsplit("@", 1)
    else:
        slug, provider = label, None
    return {"raw": label, "model_slug": slug, "provider_handle": provider}


def canon_from_model_name(name: str) -> dict:
    if name and "/" in name:
        provider, slug = name.split("/", 1)
    else:
        provider, slug = None, name
    return {"raw": name, "provider_handle": provider, "model_slug": slug}


def classify(swe: dict, tb: dict) -> tuple:
    swe_names = list(swe["canonical_model_name_counts"])
    reasons = []
    if swe["distinct_canonical_model_names"] != 1:
        return "AMBIGUOUS", ["SWE side has %d distinct canonical model_name "
                             "values: %s"
                             % (swe["distinct_canonical_model_names"],
                                swe_names)]
    canon = canon_from_model_name(swe_names[0])
    parts = split_label(tb["raw"])
    reasons.append("SWE config.model.model_name = %r (provider %r, slug %r)"
                   % (canon["raw"], canon["provider_handle"],
                      canon["model_slug"]))
    reasons.append("TB dataset model label = %r (provider handle %r, slug %r)"
                   % (parts["raw"], parts["provider_handle"],
                      parts["model_slug"]))
    same_provider = (str(canon["provider_handle"]).lower()
                     == str(parts["provider_handle"]).lower())
    same_slug = (str(canon["model_slug"]).lower()
                 == str(parts["model_slug"]).lower())
    if same_provider and same_slug:
        reasons.append("provider and model slug are equal after normalising "
                       "case only")
        return "CONFIRMED", reasons
    reasons.append("provider match = %s, model slug match = %s"
                   % (same_provider, same_slug))
    if same_provider or same_slug:
        return "AMBIGUOUS", reasons
    return "MISMATCH", reasons


def main() -> int:
    t0 = time.time()
    ensure_dirs(WORK, ANA)
    tb = tb_labels()
    readme = TB_README.read_text("utf-8")
    fmt_quote = ('| `model` | string | Underlying LLM (e.g. '
                 '`"claude-opus-4-6@anthropic"`, `"gpt-5@openai"`) |')
    card_documents_format = fmt_quote in readme
    targets = {}
    for spec in TARGETS:
        swe = swe_metadata(spec["swe_model"])
        tb_label = spec["tb_model"]
        present = tb_label in tb["by_model"]
        ident, reasons = classify(swe, {"raw": tb_label})
        if not present:
            ident = "MISMATCH"
            reasons.append("TB dataset does not contain the label %r"
                           % tb_label)
        variants = [m for m in tb["by_model"]
                    if m.split("@")[0].lower()
                    == spec["swe_model"].lower().replace(".", "-")
                    or m.split("@")[0].lower() == spec["swe_model"].lower()]
        targets[spec["scope"]] = {
            "scope": spec["scope"],
            "head": spec["head"],
            "swe_model_label": spec["swe_model"],
            "tb_model_label": tb_label,
            "swe_metadata": swe,
            "tb_label_present": bool(present),
            "tb_scaffolds_using_label": tb["by_model"].get(tb_label, []),
            "tb_labels_with_same_slug_or_spelling": variants,
            "IDENTITY": ident,
            "evidence": reasons,
        }
    write_json(ANA / "model_identity.json", {
        "section": "2 MODEL IDENTITY QUALIFICATION",
        "rule": "identity is taken from literal dataset/model metadata "
                "(SWE: info.config.model.model_name; TB: the model column "
                "slug@provider documented by the dataset card); never from "
                "name similarity",
        "sources": {
            "swe_dataset_repo_id": "tarsur385/swebench-verified-trajectories",
            "swe_dataset_revision": SWE_DATASET_REVISION,
            "swe_snapshot_dir": str(SWE_RAW),
            "swe_metadata_fields": [
                "info.config.model.model_name",
                "info.config.model_type",
                "info.docent.model_label",
            ],
            "tb_dataset_repo_id": "yoonholee/terminalbench-trajectories",
            "tb_dataset_revision": TB_DATASET_REVISION,
            "tb_dataset_card": str(TB_README),
            "tb_card_documents_label_format": bool(card_documents_format),
            "tb_card_format_quote": fmt_quote,
            "tb_model_column_distinct_labels": int(len(tb["by_model"])),
            "tb_rows_scanned": int(tb["rows"]),
        },
        "targets": targets,
        "all_identities_confirmed": bool(
            all(v["IDENTITY"] == "CONFIRMED" for v in targets.values())),
        "seconds": round(time.time() - t0, 1),
        "api_calls": 0, "llm_calls": 0,
    })
    for k, v in targets.items():
        print(k, v["swe_model_label"], "->", v["tb_model_label"],
              "IDENTITY =", v["IDENTITY"], flush=True)
        print("   variants seen with same spelling/slug:",
              v["tb_labels_with_same_slug_or_spelling"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
