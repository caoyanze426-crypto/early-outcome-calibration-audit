# -*- coding: utf-8 -*-
"""LATE_REVERSAL_EARLY_EVAL_PHASE0A - smoke fixture forensic (deterministic, local only).

No LLM/API calls. Reads the EarlyEval smoke policy outputs and the frozen fixture CSV.
"""
from __future__ import annotations
import os

import csv
import hashlib
import json
import sys
from pathlib import Path

# Project root of the frozen Paper 2 workspace; set PAPER2_ROOT
# (see reproducibility/RUN_INSTRUCTIONS.md).
WS = Path(os.environ["PAPER2_ROOT"]).resolve()
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
SMOKE = REPO / "outputs" / "current_safe_stop_smoke"
OUT = WS / "outputs" / "late_reversal_early_eval_phase0a"
NL = chr(10)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path):
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    for name in ("policy_decisions.csv", "policy_summary.csv", "policy_per_agent.csv",
                 "run_metadata.json"):
        (OUT / "smoke" / name).write_bytes((SMOKE / name).read_bytes())
    (OUT / "smoke" / "stdout.txt").write_text(
        (REPO / "work_smoke_stdout.txt").read_text("utf-8", "replace"), "utf-8", newline="")
    (OUT / "smoke" / "stderr.txt").write_text(
        (REPO / "work_smoke_stderr.txt").read_text("utf-8", "replace"), "utf-8", newline="")

    fixtures = read_csv(REPO / "examples" / "smoke_predictions.csv")
    decisions = read_csv(SMOKE / "policy_decisions.csv")

    prefixes = {}
    labels = {}
    for row in fixtures:
        tid = row["traj_id"]
        prefixes[tid] = prefixes.get(tid, 0) + 1
        labels[tid] = int(row["label"])

    rows, counts = [], {"FALSE_FAILURE": 0, "FALSE_SUCCESS": 0, "CORRECT_FAILURE": 0,
                        "CORRECT_SUCCESS": 0, "NO_STOP": 0}
    for d in decisions:
        tid = d["traj_id"]
        final_label = int(d["label"])
        final_outcome = "SUCCESS" if final_label == 1 else "FAILURE"
        halted = d["decided"].strip().lower() == "true"
        if not halted:
            early = "NO_STOP"
            cls = "NO_STOP"
        else:
            early = "SUCCESS" if d["decision"] == "success" else "FAILURE"
            if early == "FAILURE" and final_outcome == "SUCCESS":
                cls = "FALSE_FAILURE"
            elif early == "SUCCESS" and final_outcome == "FAILURE":
                cls = "FALSE_SUCCESS"
            elif early == "FAILURE" and final_outcome == "FAILURE":
                cls = "CORRECT_FAILURE"
            else:
                cls = "CORRECT_SUCCESS"
        counts[cls] += 1
        rows.append({
            "traj_id": tid, "instance_id": d["traj_id"].split("::")[-1],
            "agent_model": d["agent_model"], "final_label": final_label,
            "final_outcome": final_outcome,
            "fixture_prefix_rows": prefixes.get(tid),
            "decisions_n_steps": int(d["n_steps"]),
            "policy_halted": halted,
            "halt_step": int(d["decision_step"]) if halted else None,
            "predicted_early_class": early,
            "decision_score": d["decision_score"] or None,
            "saved_steps": int(d["saved_steps"]),
            "mechanical_class": cls,
        })

    n_steps_matches_prefixes = all(r["decisions_n_steps"] == r["fixture_prefix_rows"]
                                   for r in rows)
    # n_steps_total column vs prefix rows (offset check)
    n_total_map = {}
    for row in fixtures:
        n_total_map[row["traj_id"]] = int(row["n_steps_total"])

    with (OUT / "smoke" / "smoke_forensic.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    forensic = {
        "fixture": str(REPO / "examples" / "smoke_predictions.csv"),
        "fixture_sha256": sha_file(REPO / "examples" / "smoke_predictions.csv"),
        "number_of_trajectories": len(decisions),
        "number_of_prefix_rows": len(fixtures),
        "decisions_n_steps_equals_prefix_row_count": n_steps_matches_prefixes,
        "n_steps_total_vs_prefix_rows": {
            t: {"n_steps_total_fixture": n_total_map[t], "prefix_rows": prefixes[t]}
            for t in sorted(prefixes)},
        "mechanical_class_counts": counts,
        "per_trajectory": rows,
        "label_column_semantics": "label = ground-truth final outcome, 1 = resolved (SUCCESS)",
        "probability_columns_note": (
            "prob_cal_safe_success/failure columns are illustrative in the bundled fixture "
            "(examples/README.md); traj_id/instance_id/model_id/label/n_steps_total are real."),
    }
    (OUT / "smoke" / "smoke_forensic.json").write_text(
        json.dumps(forensic, ensure_ascii=False, indent=2), "utf-8", newline=NL)
    print(json.dumps({"number_of_trajectories": forensic["number_of_trajectories"],
                      "number_of_prefix_rows": forensic["number_of_prefix_rows"],
                      "n_steps_matches": n_steps_matches_prefixes,
                      "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
