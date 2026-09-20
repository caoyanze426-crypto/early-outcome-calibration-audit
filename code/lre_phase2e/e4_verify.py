# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2E - independent post-build verification of the artifacts."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone

from e_common import (MANIFEST_ROOTS, OUT, WORK, read_json, read_manifest,
                      sha256_file, write_json)

REPO = (OUT.parent.parent / "work" / "lre_phase0a" / "third_party"
        / "earlyeval")
EXPECTED_COMMIT = "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0"


def main() -> int:
    t0 = time.time()
    sums = read_manifest(OUT / "artifact_sha256sums.txt")
    man = read_json(OUT / "manifest.json")
    integ = read_json(OUT / "integrity_report.json")
    on_disk = sorted(str(p.relative_to(OUT)).replace("\\", "/")
                     for p in OUT.rglob("*") if p.is_file())
    sums_missing = [r for r in sums if not (OUT / r).exists()]
    sums_bad = [r for r, h in sums.items()
                if (OUT / r).exists() and sha256_file(OUT / r) != h]
    covered = set(sums) | {"artifact_sha256sums.txt"}
    uncovered = [r for r in on_disk if r not in covered]
    man_bad = [e["path"] for e in man["entries"]
               if not (OUT / e["path"]).exists()
               or sha256_file(OUT / e["path"]) != e["sha256"]]
    man_paths = {e["path"] for e in man["entries"]}
    integ_paths = {e["path"] for e in integ["self_consistency"]["entries"]}
    integ_hashes = {e["path"]: e["sha256"]
                    for e in integ["self_consistency"]["entries"]}
    integ_bad = [p for p, h in integ_hashes.items()
                 if not (OUT / p).exists() or sha256_file(OUT / p) != h]
    man_map = {e["path"]: e["sha256"] for e in man["entries"]}
    cross = [p for p in integ_paths if man_map.get(p) != integ_hashes[p]]
    expected_man = set(on_disk) - {"manifest.json", "artifact_sha256sums.txt"}
    expected_integ = expected_man - {"integrity_report.json"}
    upstream = {}
    up_ok = True
    for root in MANIFEST_ROOTS:
        rec = read_manifest(root / "artifact_sha256sums.txt")
        miss = [r for r in rec if not (root / r).exists()]
        bad = [r for r, h in rec.items()
               if (root / r).exists() and sha256_file(root / r) != h]
        upstream[root.name] = {"entries": len(rec), "missing": len(miss),
                               "mismatches": len(bad),
                               "match": not miss and not bad}
        up_ok = up_ok and not miss and not bad
    git = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    report = {
        "phase": "EARLYEVAL_PHASE2E",
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "artifact_root": str(OUT), "files_on_disk": len(on_disk),
        "sums_file": {"entries": len(sums), "missing_files": len(sums_missing),
                      "sha256_mismatches": len(sums_bad),
                      "mismatch_examples": sums_bad[:5],
                      "match": not sums_missing and not sums_bad},
        "sums_coverage": {"files_not_covered_by_sums": uncovered,
                          "only_sums_file_excluded":
                              uncovered in ([], ["artifact_sha256sums.txt"])},
        "manifest": {"entries": len(man["entries"]),
                     "sha256_mismatches": len(man_bad),
                     "mismatch_examples": man_bad[:5],
                     "covers_every_file_except_self_and_sums":
                         man_paths == expected_man,
                     "match": not man_bad and man_paths == expected_man},
        "integrity_report": {
            "entries": len(integ_paths),
            "sha256_mismatches": len(integ_bad),
            "covers_every_file_except_self_manifest_and_sums":
                integ_paths == expected_integ,
            "integrity_vs_manifest_mismatches": len(cross),
            "match": (not integ_bad and integ_paths == expected_integ
                      and not cross)},
        "upstream_manifests": upstream,
        "ALL_UPSTREAM_MANIFESTS_MATCH": bool(up_ok),
        "frozen_vendor_repo": {
            "path": str(REPO), "head": head.stdout.strip(),
            "expected_commit": EXPECTED_COMMIT,
            "porcelain": git.stdout.strip(), "clean": git.stdout.strip() == ""},
        "claims": {
            "OVERALL_RESULT": integ["OVERALL_RESULT"],
            "target_status": integ["target_status"],
            "identity": integ["identity"],
            "PHASE2B_RESULT_frozen": integ["PHASE2B_RESULT_frozen"],
            "phase2b_primary_result_modified":
                integ["phase2b_primary_result_modified"],
            "ALL_INPUT_HASHES_MATCH":
                integ["consumed_inputs_verified"]["ALL_INPUT_HASHES_MATCH"],
            "cost": integ["cost"]},
        "seconds": round(time.time() - t0, 1)}
    report["VERIFICATION_PASS"] = bool(
        all(report[k]["match"] for k in ("sums_file", "manifest",
                                         "integrity_report"))
        and up_ok and report["frozen_vendor_repo"]["clean"]
        and report["frozen_vendor_repo"]["head"] == EXPECTED_COMMIT)
    write_json(WORK / "verification_report.json", report)
    print("VERIFICATION_PASS = %s" % report["VERIFICATION_PASS"], flush=True)
    print("sums=%d manifest=%d integrity=%d on_disk=%d"
          % (len(sums), len(man["entries"]), len(integ_paths), len(on_disk)),
          flush=True)
    return 0 if report["VERIFICATION_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
