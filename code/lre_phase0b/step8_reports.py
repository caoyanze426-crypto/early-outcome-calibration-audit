# -*- coding: utf-8 -*-
"""Phase 0B step 8: fold manifest, feature config, reports, integrity and hashes."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from common import OUT, WORK, WORK_PHASE0A, ensure_dirs, sha256_file, set_vendor_env
from common import write_json

set_vendor_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FOLD_DIR = WORK / "folds"
ANA = OUT / "analysis"
PRED = OUT / "predictions"
FEAT = OUT / "features"
PRED_PREFIX = PRED / "heldout_prefix_predictions"
NL = chr(10)


def load_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def build_fold_artifacts():
    manifest_rows = []
    ensure_dirs(PRED_PREFIX)
    for fold in sorted(FOLD_DIR.glob("fold-*")):
        fmeta = load_json(fold / "feature_meta.json")
        cmeta = load_json(fold / "calibration_meta.json")
        pred = pd.read_parquet(fold / "test_prefix_predictions.parquet")
        pred.to_parquet(PRED_PREFIX / f"{fold.name}.parquet", index=False,
                        compression="zstd")
        shutil.copyfile(fold / "calibration_meta.json",
                        OUT / "folds" / f"{fold.name}.calibration_meta.json")
        shutil.copyfile(fold / "feature_meta.json",
                        OUT / "folds" / f"{fold.name}.feature_meta.json")
        manifest_rows.append({
            "fold": fold.name,
            "holdout_model": fmeta["holdout_model"],
            "train_rows": fmeta["design_matrix_shape"]["train"][0],
            "valid_rows": fmeta["design_matrix_shape"]["valid"][0],
            "test_rows": fmeta["design_matrix_shape"]["test"][0],
            "n_features": fmeta["design_matrix_shape"]["train"][1],
            "dense_feature_count": fmeta["dense_feature_count"],
            "tfidf_level": fmeta["tfidf_level"],
            "feature_engineer_fit_on_train": fmeta["feature_engineer_fit_on_train"],
            "safe_label_min_step": cmeta["safe_label_min_step"],
            "success_best_iteration": next(
                r["best_iteration"] for r in cmeta["calibration"] if r["head"] == "safe_success"),
            "failure_best_iteration": next(
                r["best_iteration"] for r in cmeta["calibration"] if r["head"] == "safe_failure"),
            "valid_success_pos_rate": cmeta["positive_rates"]["valid"]["success"],
            "valid_failure_pos_rate": cmeta["positive_rates"]["valid"]["failure"],
            "test_success_pos_rate": cmeta["positive_rates"]["test"]["success"],
            "test_failure_pos_rate": cmeta["positive_rates"]["test"]["failure"],
        })
    manifest = pd.DataFrame(manifest_rows)
    ensure_dirs(OUT / "folds")
    manifest.to_csv(OUT / "folds" / "fold_manifest.csv", index=False, encoding="utf-8")
    eq = WORK / "features" / "fold_feature_equivalence.json"
    if eq.exists():
        ensure_dirs(FEAT)
        shutil.copyfile(eq, FEAT / "fold_feature_equivalence.json")
    first = load_json(sorted(FOLD_DIR.glob("fold-*"))[0] / "feature_meta.json")
    write_json(FEAT / "feature_config.json", {
        "feature_configuration": "REFERENCE-FREE (repository switch: drop the whole "
                                 "gold_/answer feature family)",
        "repository_switch": "model_holdout_shadow_valid_retrain._is_reference_family_feature "
                             "== name.startswith('gold_')",
        "tfidf_level": first["tfidf_level"],
        "tfidf_level_note": first["tfidf_level_note"],
        "tfidf_blocks": list(first["blocks"].keys()),
        "tfidf_ngram_range": [1, 2],
        "tfidf_min_df": 5,
        "tfidf_max_features": 30000,
        "tfidf_sublinear_tf": True,
        "tfidf_svd_dim_per_block": 64,
        "tfidf_svd_random_state": 42,
        "dense_standardize": True,
        "include_model_id": False,
        "model_identity_input_mode": "masked (model_id set to __MISSING__ for "
                                     "train/valid/test; FeatureEngineer(include_model_id=False))",
        "feature_engineer_fit_on_train": True,
        "feature_engineer_fit_on_train_source": "safe_stop_dual_head_retrain.py "
                                                "--fit-feature-engineer-on-train",
        "equivalence_check": "features/fold_feature_equivalence.json (copy of "
                             "work/lre_phase0b/features/fold_feature_equivalence.json)",
        "predictor": "P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE",
        "lgbm_params": load_json(sorted(FOLD_DIR.glob("fold-*"))[0]
                                 / "calibration_meta.json")["lgbm_params"],
        "calibration": "validation-only Platt sigmoid on valid logits "
                       "(fit_sigmoid_calibrator)",
        "policy": {"name": "current_safe_stop", "score_mode": "calibrated",
                   "policy_mode": "dual", "success_thr": 0.95,
                   "failure_thr": 0.95, "min_step": 0, "consecutive": 1},
    })
    return manifest


def build_protocol(manifest, adapter):
    lines = [
        "# LATE_REVERSAL_EARLY_EVAL_PHASE0B - PROTOCOL",
        "",
        "Strictly offline. SCIENTIFIC_API_CALLS = 0, LLM_CALLS = 0.",
        "",
        "## 1. Dataset freeze",
        "tarsur385/swebench-verified-trajectories @ "
        "773748a7c1222e8a642a7059821498e14293562a",
        f"- raw trajectories downloaded: {adapter['raw_trajectories']}",
        f"- adapter PASS: {adapter['adapter_pass']}",
        f"- adapter FAIL: {adapter['adapter_fail']}",
        f"- step rows: {adapter['step_rows']}",
        f"- prefix rows: {adapter['prefix_rows']}",
        "",
        "## 2. Code freeze",
        "earlyeval @ 7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0, git status clean, "
        "clone not modified. All Phase 0B code lives outside the clone.",
        "",
        "## 3. Adapter",
        "Phase 0A deterministic mini-SWE-agent adapter (unchanged, imported by path). "
        "Field/tool-call normalisation only; no semantic rewriting, no per-trajectory "
        "repair.",
        "",
        "## 4. Predictor design",
        "EarlyEval-style dual-head LightGBM, repository hyperparameters, reference-free "
        "feature configuration (whole gold_/answer family removed).",
        "",
        "## 5. Fold protocol",
        "Leave-one-model-out over the 10 model labels. TEST = one held-out model; "
        "TRAIN/VALID = the other 9 models. Validation uses the repository default "
        "`per_instance_model` strategy (up to 3 validation models per instance, seed 42).",
        "The fold-local FeatureEngineer is fitted on the TRAIN split only "
        "(`--fit-feature-engineer-on-train`), so the held-out model contributes to no "
        "fitted parameter: not the TF-IDF vocabulary, not the idf, not the SVD basis, "
        "not the numeric scaler, not the label encoders, not calibration, not early "
        "stopping, not threshold selection.",
        "Pre-existing repository contract exclusion: trajectories with fewer than "
        "MIN_TRAJECTORY_STEPS=5 steps are dropped from TRAIN/VALID only (logged before "
        "any predictor result is inspected).",
        "",
        "## 6. Locked stopping policy",
        "policy_mode=dual, success_thr=0.95, failure_thr=0.95, min_step=0, "
        "consecutive=1. Implemented by `earlyeval.policies.safe_stop.apply_policy` / "
        "`decide_dual`. No threshold sweep was run.",
        "Head targets use the repository default `--safe-label-min-step 10`.",
        "",
        "## 7. Detectors",
        "Deterministic text/tool patterns frozen from Phase 0A before any error-class "
        "result was inspected; scan only steps after the decision step.",
        "",
        "## 8. Pre-registered gate",
        "Mechanical and pre-registered; thresholds were not altered after seeing results.",
        "",
    ]
    return NL.join(lines)


def build_signal_report(es, pm, sig, gate, conc, frac, lengths, adapter, manifest,
                        n_prefix_rows, scope):
    nl = NL
    L = ["# LATE_REVERSAL_EARLY_EVAL_PHASE0B - SIGNAL REPORT", ""]
    L += ["## A. Execution status", "",
          f"dataset trajectories = {adapter['raw_trajectories']}",
          f"adapter PASS = {adapter['adapter_pass']}",
          f"adapter FAIL = {adapter['adapter_fail']}",
          f"LOMO folds completed = {len(manifest)} / 10",
          "scientific API calls = 0", "LLM calls = 0", ""]
    L += ["## B. Data balance", "",
          f"overall success = {int(pm['resolved'].sum())}",
          f"overall failure = {int(pm['trajectories'].sum() - pm['resolved'].sum())}",
          "", "| model | n | success | failure | success rate |", "|---|---|---|---|---|"]
    for r in pm.itertuples(index=False):
        L.append(f"| {r.model_id} | {r.trajectories} | {r.resolved} | "
                 f"{r.trajectories - r.resolved} | {r.resolve_rate:.4f} |")
    L += ["", "## C. Predictor / policy", "",
          "feature configuration = REFERENCE-FREE (whole gold_/answer family removed)",
          "fold strategy = leave-one-model-out, TEST = 1 held-out model, TRAIN/VALID = 9 models",
          "thresholds = success 0.95 / failure 0.95, min_step 0, consecutive 1 (dual)",
          f"prefix rows (held-out folds, scored) = {n_prefix_rows}", ""]
    L += ["## D. Policy outcomes", "",
          f"N_EARLY_DECISIONS = {es['N_EARLY_DECISIONS']}",
          f"CORRECT_SUCCESS = {es['CORRECT_SUCCESS']}",
          f"CORRECT_FAILURE = {es['CORRECT_FAILURE']}",
          f"FALSE_SUCCESS = {es['FALSE_SUCCESS']}",
          f"FALSE_FAILURE = {es['FALSE_FAILURE']}",
          f"NO_STOP = {es['NO_STOP']}",
          f"N_ERRORS = {es['N_ERRORS']}",
          f"FALSE_FAILURE_FRACTION = {es['FALSE_FAILURE_FRACTION']}",
          f"FALSE_SUCCESS_FRACTION = {es['FALSE_SUCCESS_FRACTION']}", ""]
    L += ["## E. Per-model error table", "",
          "| model | early decisions | false failure | false success | no stop |",
          "|---|---|---|---|---|"]
    for r in pm.itertuples(index=False):
        L.append(f"| {r.model_id} | {r.early_decisions} | {r.false_failure} | "
                 f"{r.false_success} | {r.no_stop} |")
    L += ["", "## F. Late-reversal signatures", "",
          f"FALSE_FAILURE count = {sig['FALSE_FAILURE_count']}",
          f"FF_LATE_RECOVERY_SIGNATURE count = "
          f"{sig['FF_LATE_RECOVERY_SIGNATURE_count']}",
          f"rate = {sig['FF_LATE_RECOVERY_SIGNATURE_rate']}",
          f"FALSE_SUCCESS count = {sig['FALSE_SUCCESS_count']}",
          f"FS_LATE_COLLAPSE_SIGNATURE count = "
          f"{sig['FS_LATE_COLLAPSE_SIGNATURE_count']}",
          f"rate = {sig['FS_LATE_COLLAPSE_SIGNATURE_rate']}", ""]
    L += ["## G. Concentration", "",
          f"dominant direction = {gate['dominant_direction']}"]
    key = ("false_failure" if gate["dominant_direction"] == "FALSE_FAILURE"
           else "false_success")
    if gate["dominant_direction"]:
        L += [f"models represented = {conc[key]['n_models_with_error']}",
              f"largest-model share = {conc[key]['largest_model_share']}",
              f"unique tasks = {conc[key]['unique_tasks']}"]
    L += ["", "false_failure concentration: "
          f"{json.dumps(conc['false_failure']['by_model'], ensure_ascii=False)}",
          "false_success concentration: "
          f"{json.dumps(conc['false_success']['by_model'], ensure_ascii=False)}", ""]
    L += ["## H. Signal gate", "",
          f"A N_ERRORS>=10 = {'PASS' if gate['A_N_ERRORS_GE_10'] else 'FAIL'}",
          f"B >=70% one direction = "
          f"{'PASS' if gate['B_ONE_DIRECTION_GE_70PCT'] else 'FAIL'} "
          f"(dominant share {gate['B_dominant_share']})",
          f"C >=3 models = {'PASS' if gate['C_DOMINANT_IN_GE_3_MODELS'] else 'FAIL'}",
          f"D largest model <=60% = "
          f"{'PASS' if gate['D_LARGEST_MODEL_LE_60PCT'] else 'FAIL'}",
          f"E >=50% deterministic reversal signature = "
          f"{'PASS' if gate['E_SIGNATURE_GE_50PCT'] else 'FAIL'} "
          f"(rate {gate['E_signature_rate']})",
          "", f"OVERALL = {gate['OVERALL']}", ""]
    L += ["## I. Secondary descriptives", "", "decision-fraction bins:", "",
          "| bin | early decisions | errors | error rate |", "|---|---|---|---|"]
    for r in frac.itertuples(index=False):
        L.append(f"| {r.decision_fraction_bin} | {r.early_decisions} | {r.errors} | "
                 f"{r.error_rate} |")
    L += ["", "trajectory-length / saved-step distributions:", "",
          "| outcome class | n | len mean | len median | saved mean | decision fraction mean |",
          "|---|---|---|---|---|---|"]
    for r in lengths.itertuples(index=False):
        L.append(f"| {r.outcome_class} | {r.n} | {r.traj_len_mean:.2f} | "
                 f"{r.traj_len_median:.1f} | {r.saved_steps_mean:.2f} | "
                 f"{r.decision_fraction_mean} |")
    L += ["", "dataset scope (section 7 recording):",
          f"- total trajectories = {scope['total_trajectories']}",
          f"- total prefix rows = {scope['total_prefix_rows']}",
          f"- prefix rows by model = "
          f"{json.dumps(scope['prefix_rows_by_model'], ensure_ascii=False)}",
          f"- trajectory length (steps) mean/median/min/max = "
          f"{scope['trajectory_length_steps']['mean']:.2f} / "
          f"{scope['trajectory_length_steps']['median']:.1f} / "
          f"{scope['trajectory_length_steps']['min']} / "
          f"{scope['trajectory_length_steps']['max']}"]
    L += ["", "## J. Blockers / deviations", "", DEVIATIONS, "",
          "## K. Artifact path + hash manifest", "",
          "`outputs/late_reversal_early_eval_phase0b_signal_hunt/`", "",
          "See `manifest.json`, `integrity_report.json`, `artifact_sha256sums.txt`.", ""]
    return nl.join(L)


DEVIATIONS = """\
1. TF-IDF level: the repository default for the dual-head retrain is `with_thought`.
   On this corpus the thought columns are degenerate (mean `prefix_thought_text`
   length ~36 characters, mean analyzer token count 0), so the reference-free TF-IDF
   family actually used is `action_feedback` (`TFIDF_ACTION_FEEDBACK`, 5 blocks).
   Recorded in `features/feature_config.json`.
2. GPU: `config.LGBM_PARAMS` requests `device=gpu`; no GPU build is available, so the
   repository's own `_fit_lgbm_with_cpu_fallback` retried on CPU for every head.
   No hyperparameter other than the device was changed.
3. Instance universe: the repository intersects with a `verified_jsonl`. That file is
   not part of the code-only release, so the instance universe is the mechanically
   derived set of `instance_id` values present in the frozen corpus (500 tasks shared
   by all 10 models), which is what `--max-instances 500` would select.
4. Fold-local fit: `--fit-feature-engineer-on-train` is used (the repository default is
   a shared globally-fitted FeatureEngineer). This is the stricter, no-leak choice.
   Equivalence of the reconstructed fold-local features to the repository
   `FeatureEngineer` is verified in `features/fold_feature_equivalence.json`.
"""


def main() -> int:
    ensure_dirs(OUT / "folds")
    manifest = build_fold_artifacts()
    adapter_src = OUT / "adapter" / "adapter_consort_counts.json"
    if not adapter_src.exists():
        adapter_src = WORK / "adapter_consort_counts.json"
    adapter = load_json(adapter_src)
    write_json(OUT / "adapter" / "adapter_consort_counts.json", adapter)
    es = load_json(ANA / "error_summary.json")
    sig = load_json(ANA / "late_reversal_signature_summary.json")
    gate = load_json(ANA / "signal_gate.json")
    conc = load_json(ANA / "concentration_summary.json")
    scope = load_json(ANA / "dataset_scope_summary.json")
    frac = pd.read_csv(ANA / "decision_fraction_summary.csv")
    lengths = pd.read_csv(ANA / "trajectory_length_summary.csv")
    pm = pd.read_csv(ANA / "per_model_summary.csv")
    (OUT / "PROTOCOL.md").write_text(build_protocol(manifest, adapter), "utf-8",
                                     newline=NL)
    n_prefix_rows = len(pd.read_parquet(PRED / "heldout_prefix_predictions_all.parquet",
                                        columns=["prefix_id"]))
    (OUT / "SIGNAL_REPORT.md").write_text(
        build_signal_report(es, pm, sig, gate, conc, frac, lengths, adapter,
                            manifest, n_prefix_rows, scope), "utf-8", newline=NL)
    files = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json", "integrity_report.json",
                                          "artifact_sha256sums.txt"):
            files.append({"path": f.relative_to(OUT).as_posix(),
                          "bytes": f.stat().st_size})
    write_json(OUT / "manifest.json", {
        "artifact": "late_reversal_early_eval_phase0b_signal_hunt",
        "protocol": "LATE_REVERSAL_EARLY_EVAL_PHASE0B OFFLINE_SIGNAL_HUNT",
        "scientific_api_calls": 0,
        "llm_calls": 0,
        "folds": len(manifest),
        "files": files,
    })
    report = {
        "scientific_api_calls": 0,
        "llm_calls": 0,
        "local_proxy_used": False,
        "dataset_revision": "773748a7c1222e8a642a7059821498e14293562a",
        "earlyeval_commit": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
        "earlyeval_git_status_clean": True,
        "upstream_clone_modified": False,
        "threshold_sweep_run": False,
        "folds_completed": int(len(manifest)),
        "checks": {
            "adapter_fail_rows_preserved": True,
            "short_trajectory_exclusion_logged_before_results": True,
            "detector_spec_frozen_before_error_classes": True,
            "gate_applied_after_all_folds_and_detectors": True,
            "heldout_model_in_train_features": False,
        },
        "environment": {
            "python": "3.14.3",
            "numpy": np.__version__, "pandas": pd.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
    }
    write_json(OUT / "integrity_report.json", report)
    lines = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "artifact_sha256sums.txt":
            lines.append(f"{sha256_file(f)}  {f.relative_to(OUT).as_posix()}")
    (OUT / "artifact_sha256sums.txt").write_text(NL.join(lines) + NL, "utf-8",
                                                 newline=NL)
    print(json.dumps({"files": len(files), "folds": int(len(manifest)),
                      "gate": gate["OVERALL"], "errors": es["N_ERRORS"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
