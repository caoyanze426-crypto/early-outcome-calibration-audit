import os
import json, hashlib
from pathlib import Path
# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
OUT = WS / "outputs" / "earlyeval_phase1b_threshold_robustness"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
rec = {}
for line in (OUT / "artifact_sha256sums.txt").read_text("utf-8").splitlines():
    if "  " in line:
        h, rel = line.split("  ", 1)
        rec[rel.strip()] = h.strip()
bad = [r for r, h in rec.items() if not (OUT / r).exists() or sha(OUT / r) != h]
on_disk = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
print("digest entries:", len(rec), "mismatches:", len(bad), "unlisted:", [p for p in on_disk if p not in rec])
required = ["PROTOCOL.md","manifest.json","input_hashes.json","analysis/threshold_occurrences.csv",
 "analysis/target_threshold_summary.csv","analysis/bootstrap_results.json","analysis/reproduction_check.json",
 "analysis/primary_gate.json","PHASE1B_REPORT.md","integrity_report.json","artifact_sha256sums.txt"]
print("missing required:", [r for r in required if not (OUT / r).exists()])
ir = json.loads((OUT / "integrity_report.json").read_text("utf-8"))
print("inputs match:", ir["input_hashes_match"], "| support identical:", ir["support_identifier_sets_identical_across_thresholds"])
print("reproduction:", ir["reproduction_check"]["PIPELINE_REPRODUCTION"], "| exact:", ir["reproduction_check"]["EXACT_MATCH"])
print("targets:", json.dumps(ir["targets"], indent=1))
print("overall:", json.dumps(ir["overall"], indent=1))
print("cost:", ir["api_calls"], ir["llm_calls"], ir["predictor_retraining"], ir["new_lightgbm_training"], ir["new_trajectories"], ir["new_datasets"])
print("hash entries declared:", ir["artifact_sha256sums_entries"], "actual:", len(rec))
for name, root in {
    "phase0a": WS/"outputs"/"late_reversal_early_eval_phase0a",
    "phase0b": WS/"outputs"/"late_reversal_early_eval_phase0b_signal_hunt",
    "phase0c": WS/"outputs"/"late_reversal_early_eval_phase0c",
    "phase0d": WS/"outputs"/"earlyeval_phase0d_cross_agent_calibration",
    "phase0e": WS/"outputs"/"earlyeval_phase0e_prior_shift_decomposition",
    "phase1a": WS/"outputs"/"earlyeval_phase1a_same_predictor_transfer",
    "phase1a_d": WS/"outputs"/"earlyeval_phase1a_d_target_persistence"}.items():
    lines = (root/"artifact_sha256sums.txt").read_text("utf-8").splitlines()
    bad = 0
    for line in lines:
        if "  " not in line: continue
        h, rel = line.split("  ", 1); rel = rel.strip()
        p = root/rel
        if not p.exists() or sha(p) != h: bad += 1
    print(f"{name}: entries={len(lines)} mismatches={bad}")
