# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2E - Sections 3/9: input hashes, report, manifest, sums."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from e_common import (ANA, MANIFEST_ROOTS, OUT, OUT_1A, OUT_1AD, OUT_1B,
                      OUT_2B, OUT_2BD, PRIMARY_THRESHOLD, SWE_DATASET_REVISION,
                      SWE_RAW, SWE_SNAPSHOT, TARGETS, TB_DATASET_REVISION,
                      TB_PARQUET, TB_README, THRESHOLDS, WORK, as_builtin,
                      ensure_dirs, read_json, read_manifest, sha256_file,
                      write_json)

WS = OUT.parent.parent
REPO = WS / "work" / "lre_phase0a" / "third_party" / "earlyeval"
EXPECTED_COMMIT = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def all_files(root) -> list:
    return sorted(p for p in root.rglob("*") if p.is_file())


def rel(p) -> str:
    return str(p.relative_to(OUT)).replace("\\", "/")


SWE_READ_FILES = [
    OUT_1B / "analysis" / "threshold_occurrences.csv",
    OUT_1B / "analysis" / "target_threshold_summary.csv",
    OUT_1B / "analysis" / "bootstrap_results.json",
    OUT_1B / "analysis" / "secondary_descriptives.csv",
    OUT_1B / "analysis" / "primary_gate.json",
    OUT_1A / "analysis" / "target_persistence.csv",
    OUT_1A / "input_hashes.json",
    OUT_1AD / "analysis" / "target_occurrences.csv",
    OUT_1A / "artifact_sha256sums.txt",
    OUT_1AD / "artifact_sha256sums.txt",
    OUT_1B / "artifact_sha256sums.txt",
]
TB_READ_FILES = [
    OUT_2B / "analysis" / "per_target_metrics.csv",
    OUT_2B / "analysis" / "target_bootstrap.json",
    OUT_2B / "analysis" / "phase2b_gate.json",
    OUT_2B / "analysis" / "per_target_coverage.csv",
    OUT_2BD / "analysis" / "threshold_target_metrics.csv",
    OUT_2BD / "analysis" / "bootstrap_results.json",
    OUT_2BD / "analysis" / "diagnostic_gate.json",
    OUT_2B / "artifact_sha256sums.txt",
    OUT_2BD / "artifact_sha256sums.txt",
]


def verify_manifests() -> dict:
    out = {}
    all_ok = True
    for root in MANIFEST_ROOTS:
        man = read_manifest(root / "artifact_sha256sums.txt")
        missing, bad = [], []
        for r, h in sorted(man.items()):
            p = root / r
            if not p.exists():
                missing.append(r)
            elif sha256_file(p) != h:
                bad.append(r)
        ok = bool(not missing and not bad)
        all_ok = all_ok and ok
        out[root.name] = {
            "manifest": str(root / "artifact_sha256sums.txt"),
            "entries": int(len(man)), "missing_files": int(len(missing)),
            "sha256_mismatches": int(len(bad)), "mismatch_examples": bad[:5],
            "match": ok,
        }
    out["ALL_UPSTREAM_MANIFESTS_MATCH"] = bool(all_ok)
    return out


def consumed_inputs(manifests: dict) -> tuple:
    man_by_root = {r.name: read_manifest(r / "artifact_sha256sums.txt")
                   for r in MANIFEST_ROOTS}
    records, fails = [], []

    def add(path, role, root=None):
        exist = path.exists()
        h = sha256_file(path) if exist else None
        exp = None
        if root is not None:
            key = str(path.relative_to(root)).replace("\\", "/")
            exp = man_by_root[root.name].get(key)
        ok = bool(exist and (root is None or (exp is not None and exp == h)))
        if not ok:
            fails.append(str(path))
        records.append({"path": str(path), "role": role, "exists": bool(exist),
                        "sha256": h, "manifest_sha256": exp, "match": ok})

    root_of = {p: r for r in MANIFEST_ROOTS
               for p in (r / "analysis").rglob("*")}
    for p in SWE_READ_FILES + TB_READ_FILES:
        root = next((r for r in MANIFEST_ROOTS
                     if str(p).startswith(str(r))), None)
        if p.name == "artifact_sha256sums.txt":
            add(p, "frozen manifest (self-excluded by design; verified by "
                   "the upstream manifest re-verification step above)")
            continue
        add(p, "frozen benchmark artifact", root)
    add(SWE_SNAPSHOT / "README.md", "SWE dataset card")
    add(WS / "work" / "lre_phase0b" / "hf_home" / "hub"
        / "datasets--tarsur385--swebench-verified-trajectories"
        / "snapshots" / SWE_DATASET_REVISION / "swebench_verified_raw"
        / "gpt-5-mini" / "astropy__astropy-12907"
        / "astropy__astropy-12907.traj.json",
        "SWE raw trajectory (gpt-5-mini sample; full 500-file scan recorded "
        "in analysis/model_identity.json)")
    add(WS / "work" / "lre_phase0b" / "hf_home" / "hub"
        / "datasets--tarsur385--swebench-verified-trajectories"
        / "snapshots" / SWE_DATASET_REVISION / "swebench_verified_raw"
        / "claude-opus-4.6" / "astropy__astropy-12907"
        / "astropy__astropy-12907.traj.json",
        "SWE raw trajectory (claude-opus-4.6 sample; full 500-file scan "
        "recorded in analysis/model_identity.json)")
    add(TB_README, "TB dataset card")
    for p in sorted(TB_PARQUET.glob("train-*.parquet")):
        add(p, "TB dataset shard (model/agent labels)")
    add(REPO / "earlyeval" / "policies" / "safe_stop.py",
        "frozen vendor policy source (reused by Phase 2B/2B-D lineage)")
    return records, fails


def fmt(value, digits=6):
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return str(value)


def fmt_any(value):
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "-"
    if isinstance(value, str) and value.lower() in ("nan", "none"):
        return "-"
    return str(value)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def build_report(ident, cross, desc, status, manifests, n_inputs, n_fail):
    L = []
    L.append("# EARLYEVAL_PHASE2E - EXACT_TARGET_CROSS_BENCHMARK_AUDIT")
    L.append("")
    L.append("Generated: " + utc())
    L.append("")
    L.append("    OVERALL_RESULT = " + status["OVERALL_RESULT"])
    L.append("")
    L.append("    PHASE2B_RESULT (frozen, unchanged) = "
             + status["PHASE2B_RESULT_frozen"])
    L.append("")
    L.append("Descriptive audit of the two exact SWE-bench targets that were "
             "robustly miscalibrated. Reuses frozen SWE Phase 1A / 1A-D / 1B "
             "and TerminalBench Phase 2B / 2B-D artifacts only. API calls = 0, "
             "LLM calls = 0, training = 0, cloud = 0.")
    L.append("")
    L.append("## A. IDENTITY")
    L.append("")
    for scope in sorted(ident["targets"]):
        t = ident["targets"][scope]
        L.append("- %s: SWE `%s` -> TB `%s` : IDENTITY = %s"
                 % (scope, t["swe_model_label"], t["tb_model_label"],
                    t["IDENTITY"]))
        for ev in t["evidence"]:
            L.append("    - " + ev)
        if t["tb_labels_with_same_slug_or_spelling"]:
            L.append("    - TB labels sharing this spelling/slug: %s"
                     % ", ".join("`%s`" % x
                                 for x in t["tb_labels_with_same_slug_or_"
                                            "spelling"]))
    L.append("")
    L.append("## B. TARGET 1 (gpt-5-mini / SUCCESS)")
    L.append("")
    L.extend(target_block(cross, desc, status, "TARGET_1"))
    L.append("")
    L.append("## C. TARGET 2 (claude-opus-4.6 / FAILURE)")
    L.append("")
    L.extend(target_block(cross, desc, status, "TARGET_2"))
    L.append("")
    L.append("## D. OVERALL")
    L.append("")
    L.append("    " + status["OVERALL_RESULT"])
    L.append("")
    L.append("- qualified targets: %s"
             % (", ".join(status["qualified_targets"]) or "none"))
    L.append("- excluded targets: %s"
             % (", ".join(status["excluded_targets"]) or "none"))
    L.append("- persistent: %s"
             % (", ".join(status["persistent_targets"]) or "none"))
    L.append("- collapsed: %s"
             % (", ".join(status["collapsed_targets"]) or "none"))
    L.append("")
    L.append("Statuses in section D are the primary 0.950 statuses. The "
             "secondary frozen-threshold descriptives in sections B and C "
             "show TARGET_2 classifying as COLLAPSED_ON_TERMINALBENCH at "
             "0.900 and 0.925 and UNCLASSIFIED_BY_SPEC at 0.950; no threshold "
             "is selected as best and no gate is built from them.")
    L.append("")
    L.append("Interpretation boundary: where an exact SWE target does not "
             "persist, the record states only that the target-specific "
             "calibration phenotype did not transfer unchanged across the two "
             "frozen benchmark settings. No claim is made that the benchmark "
             "causes the difference.")
    L.append("")
    L.append("## E. FROZEN CROSS-BENCHMARK GATE")
    L.append("")
    L.append("    Phase 2B remains: " + status["PHASE2B_RESULT_frozen"])
    L.append("")
    L.append("## F. COST")
    L.append("")
    L.append("    API = 0")
    L.append("    LLM = 0")
    L.append("    training = 0")
    L.append("    cloud = 0")
    L.append("")
    L.append("## G. ARTIFACT PATH + HASH MANIFEST")
    L.append("")
    L.append("    " + str(OUT))
    L.append("")
    L.append("- Upstream manifests re-verified: "
             + ", ".join("%s %d entries/%s" % (k, v["entries"],
                                               "match" if v["match"]
                                               else "MISMATCH")
                         for k, v in manifests.items()
                         if isinstance(v, dict)))
    L.append("- Consumed input files re-hashed: %d, failing %d."
             % (n_inputs, n_fail))
    L.append("")
    L.append("Files (%d); sha256 per file in artifact_sha256sums.txt and "
             "manifest.json:" % len(all_files(OUT)))
    L.append("")
    for r in sorted(rel(p) for p in all_files(OUT)):
        L.append("    " + r)
    L.append("")
    L.append("Independent post-build verification: "
             "work/lre_phase2e/verification_report.json")
    L.append("")
    return chr(10).join(L) + chr(10)


def target_block(cross, desc, status, scope):
    st = status["targets"][scope]
    L = []
    one = cross[cross["target_scope"] == scope]
    rows = []
    for r in one.itertuples():
        rows.append([r.benchmark, r.model_label, r.decisions,
                     fmt(r.empirical_precision), fmt(r.corrected_mean_score),
                     fmt(r.signed_corrected_gap),
                     "[" + fmt(r.bootstrap_ci_lower) + ", "
                     + fmt(r.bootstrap_ci_upper) + "]",
                     fmt_any(r.ci_includes_zero)])
    L.extend(md_table(["benchmark", "model", "decisions", "precision",
                       "corrected mean score", "signed corrected gap",
                       "bootstrap CI", "CI includes 0"], rows))
    L.append("")
    L.append("- SWE basis: " + str(one[one.benchmark == "SWE-bench"]
                                   ["decisions_basis"].iloc[0]))
    L.append("- TB basis: " + str(one[one.benchmark == "TerminalBench"]
                                  ["decisions_basis"].iloc[0]))
    L.append("- identity = %s" % st["identity"])
    L.append("- status at 0.950 = **%s**" % st["STATUS"])
    c = st["clauses"]
    L.append("- clauses: decisions=%s (>=20: %s), |gap|=%s (>=0.08: %s; "
             "<0.04: %s), CI excludes 0: %s, TB sign=%s, SWE sign=%s, "
             "sign matches: %s"
             % (c["tb_decisions"], c["tb_decisions_ge_20"],
                fmt(c["abs_corrected_gap"], 6), c["abs_gap_ge_0_08"],
                c["abs_gap_lt_0_04"], c["ci_excludes_zero"], c["tb_gap_sign"],
                c["swe_gap_sign"], c["sign_matches_swe"]))
    if st["STATUS"] == "UNCLASSIFIED_BY_SPEC":
        L.append("- NOTE (deviation 1): this target matches none of the four "
                 "pre-registered clauses at this threshold "
                 "(0.04 <= |gap| < 0.08 with a CI spanning 0 and an agreeing "
                 "sign). Recorded literally as UNCLASSIFIED_BY_SPEC, treated "
                 "as not persistent.")
    d = desc[desc["target_scope"] == scope]
    L.append("")
    L.append("Secondary frozen thresholds (descriptive):")
    L.append("")
    rows2 = []
    for thr in THRESHOLDS:
        for bench in ("SWE-bench", "TerminalBench"):
            r = d[(d["benchmark"] == bench)
                  & (d["threshold"].astype(float) == thr)]
            if not len(r):
                continue
            r = r.iloc[0]
            rows2.append(["%.3f" % thr, bench, r.decisions,
                          fmt(r.empirical_precision),
                          fmt(r.signed_corrected_gap),
                          "[" + fmt(r.bootstrap_ci_lower) + ", "
                          + fmt(r.bootstrap_ci_upper) + "]",
                          str(r.status_at_threshold)])
    L.extend(md_table(["threshold", "benchmark", "decisions", "precision",
                       "signed corrected gap", "bootstrap CI",
                       "TB status at threshold"], rows2))
    return L


def frozen_cross_checks(cross) -> tuple:
    """Verify frozen headline values and that carried values match them."""
    checks = {}
    occ = pd.read_csv(OUT_1B / "analysis" / "threshold_occurrences.csv")
    summ = pd.read_csv(OUT_1B / "analysis" / "target_threshold_summary.csv")
    boot = read_json(OUT_1B / "analysis" / "bootstrap_results.json")["primary"]
    label95 = "T3"
    recomputed, ok = {}, True
    for spec in TARGETS:
        scope = spec["scope"]
        rows = occ[(occ["target_scope"] == scope)
                   & (occ["threshold"].astype(float) == 0.95)]
        med = float(rows["signed_corrected_gap"].median())
        s = summ[(summ["target_scope"] == scope)
                 & (summ["threshold"].astype(float) == 0.95)]
        frozen = float(s["median_signed_gap"].iloc[0])
        b = boot["%s|%s" % (scope, label95)]["median_signed_gap"]
        ci_ok = abs(b["ci_lower"]
                    - float(s["bootstrap_ci_lower"].iloc[0])) <= 5e-7
        match = abs(med - frozen) <= 1e-12
        ok = ok and match and ci_ok
        recomputed[scope] = {
            "occurrences": int(len(rows)),
            "recomputed_median_signed_gap": med,
            "frozen_median_signed_gap": frozen,
            "abs_diff": abs(med - frozen), "match": bool(match),
            "frozen_bootstrap_ci_lower_full_precision": b["ci_lower"],
            "frozen_bootstrap_ci_upper_full_precision": b["ci_upper"],
            "frozen_summary_ci_matches_bootstrap_within_5e-7": bool(ci_ok)}
    checks["swe_headline_gap_reproduction"] = {
        "reference": str(OUT_1B / "analysis"
                         / "target_threshold_summary.csv"),
        "per_target": recomputed, "match": bool(ok)}
    spec_gaps = {"TARGET_1": 0.1377462284504496,
                 "TARGET_2": 0.11073740121815578}
    spec_ok, dev = True, {}
    for k, v in spec_gaps.items():
        frozen = recomputed[k]["frozen_median_signed_gap"]
        d = abs(frozen - v)
        dev[k] = {"spec_value": v, "frozen_artifact_value": frozen,
                  "abs_diff": d, "match_within_1e-9": bool(d <= 1e-9)}
        spec_ok = spec_ok and d <= 1e-9
    checks["spec_target_gap_values_match_frozen_artifacts"] = {
        "per_target": dev, "match": bool(spec_ok)}
    tb = cross[cross["benchmark"] == "TerminalBench"].set_index("target_scope")
    tb_check, ok_tb = {}, True
    m2b = pd.read_csv(OUT_2B / "analysis" / "per_target_metrics.csv")
    for spec in TARGETS:
        r = m2b[(m2b["model_id"] == spec["tb_model"])
                & (m2b["head"] == spec["head"])].iloc[0]
        row = tb.loc[spec["scope"]]
        same = int(r["decisions"]) == int(row["decisions"])
        if pd.isna(r["corrected_gap"]):
            same = same and bool(pd.isna(row["signed_corrected_gap"]))
        else:
            same = same and abs(float(r["corrected_gap"])
                                - float(row["signed_corrected_gap"])) <= 1e-12
        ok_tb = ok_tb and same
        tb_check[spec["scope"]] = {
            "model_id": spec["tb_model"], "head": spec["head"],
            "frozen_2b_decisions": int(r["decisions"]),
            "carried_decisions": int(row["decisions"]),
            "frozen_2b_corrected_gap": (None if pd.isna(r["corrected_gap"])
                                        else float(r["corrected_gap"])),
            "carried_corrected_gap": row["signed_corrected_gap"],
            "match": bool(same)}
    checks["terminalbench_values_match_frozen_phase2b"] = {
        "reference": str(OUT_2B / "analysis" / "per_target_metrics.csv"),
        "per_target": tb_check, "match": bool(ok_tb)}
    return as_builtin(checks), bool(ok and spec_ok and ok_tb)


def main() -> int:
    t0 = time.time()
    ensure_dirs(WORK, ANA)
    status = read_json(ANA / "target_status.json")
    ident = read_json(ANA / "model_identity.json")
    cross = pd.read_csv(ANA / "target_cross_benchmark.csv")
    desc = pd.read_csv(ANA / "threshold_descriptives.csv")
    manifests = verify_manifests()
    if not manifests["ALL_UPSTREAM_MANIFESTS_MATCH"]:
        print("STOP: upstream manifest verification failed", flush=True)
        return 2
    records, fails = consumed_inputs(manifests)
    checks, checks_ok = frozen_cross_checks(cross)
    all_ok = bool(not fails and checks_ok
                  and status["OVERALL_RESULT"]
                  in status["allowed_labels"])
    cost = {"api_calls": 0, "llm_calls": 0, "cloud_compute": 0,
            "predictor_retraining": 0, "new_lightgbm_training": 0,
            "new_trajectories": 0, "new_datasets": 0,
            "new_scaffolds": 0, "toolathlon_runs": 0, "paid_services": 0,
            "thresholds_evaluated": list(THRESHOLDS),
            "metrics_recomputed": "only the SWE pooled precision and pooled "
                                  "corrected mean score, by re-weighting "
                                  "frozen Phase 1B per-occurrence rows"}
    write_json(OUT / "input_hashes.json", {
        "phase": "EARLYEVAL_PHASE2E",
        "generated_utc": utc(), "artifact_root": str(OUT),
        "frozen_output_roots_consumed": [str(r) for r in MANIFEST_ROOTS],
        "frozen_dataset_revisions": {
            "swe_bench_verified_trajectories":
                {"repo_id": "tarsur385/swebench-verified-trajectories",
                 "revision": SWE_DATASET_REVISION},
            "terminalbench_trajectories":
                {"repo_id": "yoonholee/terminalbench-trajectories",
                 "revision": TB_DATASET_REVISION}},
        "upstream_manifest_verification": manifests,
        "consumed_inputs": records,
        "consumed_input_files": int(len(records)),
        "consumed_input_files_failing": int(len(fails)),
        "frozen_value_cross_checks": checks,
        "ALL_INPUT_HASHES_MATCH": all_ok,
        "OVERALL_RESULT": status["OVERALL_RESULT"],
        "PHASE2B_RESULT_frozen": status["PHASE2B_RESULT_frozen"],
        "phase2b_primary_result_modified": False,
        **cost})
    (OUT / "PHASE2E_REPORT.md").write_text(
        build_report(ident, cross, desc, status, manifests, len(records),
                     len(fails)), "utf-8")
    skip_integrity = {"integrity_report.json", "manifest.json",
                      "artifact_sha256sums.txt"}
    integ_entries = {rel(p): sha256_file(p) for p in all_files(OUT)
                     if rel(p) not in skip_integrity}
    write_json(OUT / "integrity_report.json", {
        "phase": "EARLYEVAL_PHASE2E", "generated_utc": utc(),
        "artifact_root": str(OUT),
        "self_consistency": {
            "manifest": "artifact_sha256sums.txt",
            "entries_covered_here": int(len(integ_entries)),
            "entries_excluded_here": sorted(skip_integrity),
            "note": "this file lists every artifact this phase produced "
                    "except manifest.json, artifact_sha256sums.txt and this "
                    "file itself; manifest.json covers that same set plus "
                    "this file; the independent post-build verification "
                    "recomputes all three",
            "entries": [{"path": k, "sha256": v}
                        for k, v in sorted(integ_entries.items())]},
        "upstream_manifest_verification": manifests,
        "consumed_inputs_verified": {
            "files": int(len(records)), "failing": int(len(fails)),
            "failing_paths": fails[:10], "ALL_INPUT_HASHES_MATCH": all_ok},
        "frozen_value_cross_checks": checks,
        "OVERALL_RESULT": status["OVERALL_RESULT"],
        "target_status": {k: v["STATUS"] for k, v in status["targets"].items()},
        "identity": {k: v["IDENTITY"] for k, v in ident["targets"].items()},
        "PHASE2B_RESULT_frozen": status["PHASE2B_RESULT_frozen"],
        "phase2b_primary_result_modified": False,
        "frozen_upstream_repo": {"path": str(REPO),
                                 "expected_commit": EXPECTED_COMMIT},
        "cost": cost, "history_preserved": True,
        "wall_clock_seconds": round(time.time() - t0, 1)})
    skip_manifest = {"manifest.json", "artifact_sha256sums.txt"}
    man_entries = {rel(p): sha256_file(p) for p in all_files(OUT)
                   if rel(p) not in skip_manifest}
    write_json(OUT / "manifest.json", {
        "phase": "EARLYEVAL_PHASE2E",
        "title": "EXACT_TARGET_CROSS_BENCHMARK_AUDIT",
        "generated_utc": utc(), "artifact_root": str(OUT),
        "excludes": sorted(skip_manifest), "n_entries": int(len(man_entries)),
        "entries": [{"path": k, "sha256": v,
                     "bytes": (OUT / k).stat().st_size}
                    for k, v in sorted(man_entries.items())]})
    sums = []
    for p in all_files(OUT):
        r = rel(p)
        if r == "artifact_sha256sums.txt":
            continue
        sums.append(sha256_file(p) + "  " + r)
    (OUT / "artifact_sha256sums.txt").write_text(chr(10).join(sums) + chr(10),
                                                 "utf-8")
    print("inputs=%d failing=%d checks_ok=%s matches=%s"
          % (len(records), len(fails), checks_ok, all_ok), flush=True)
    print("OVERALL_RESULT=%s manifest=%d sums=%d"
          % (status["OVERALL_RESULT"], len(man_entries), len(sums)), flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
