# -*- coding: utf-8 -*-
"""EARLYEVAL_PHASE2B_D - independent post-build verification of the artifacts."""
from __future__ import annotations

import subprocess
import time

from d_common import OUT, OUT_2B, WORK, read_json, read_manifest, sha256_file, \
    write_json

REPO = (OUT.parent.parent / "work" / "lre_phase0a" / "third_party"
        / "earlyeval")


def main() -> int:
    t0 = time.time()
    sums = read_manifest(OUT / "artifact_sha256sums.txt")
    man = read_json(OUT / "manifest.json")
    integ = read_json(OUT / "integrity_report.json")
    on_disk = sorted(str(p.relative_to(OUT)).replace("\\", "/")
                     for p in OUT.rglob("*") if p.is_file())
    sums_missing, sums_bad = [], []
    for r, h in sorted(sums.items()):
        p = OUT / r
        if not p.exists():
            sums_missing.append(r)
        elif sha256_file(p) != h:
            sums_bad.append(r)
    covered = set(sums) | {"artifact_sha256sums.txt"}
    uncovered = [r for r in on_disk if r not in covered]
    man_bad = []
    for e in man["entries"]:
        p = OUT / e["path"]
        if not p.exists() or sha256_file(p) != e["sha256"]:
            man_bad.append(e["path"])
    man_paths = {e["path"] for e in man["entries"]}
    integ_paths = {e["path"] for e in integ["self_consistency"]["entries"]}
    integ_hashes = {e["path"]: e["sha256"]
                    for e in integ["self_consistency"]["entries"]}
    integ_bad = [p for p, h in integ_hashes.items()
                 if not (OUT / p).exists() or sha256_file(OUT / p) != h]
    man_map = {e["path"]: e["sha256"] for e in man["entries"]}
    cross = [p for p in integ_paths
             if man_map.get(p) != integ_hashes[p]]
    expected_manifest_paths = set(on_disk) - {"manifest.json",
                                             "artifact_sha256sums.txt"}
    expected_integ_paths = expected_manifest_paths - {"integrity_report.json"}
    frozen_man = read_manifest(OUT_2B / "artifact_sha256sums.txt")
    f_missing, f_bad = [], []
    for r, h in sorted(frozen_man.items()):
        p = OUT_2B / r
        if not p.exists():
            f_missing.append(r)
        elif sha256_file(p) != h:
            f_bad.append(r)
    git = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    report = {
        "phase": "EARLYEVAL_PHASE2B_D",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        "artifact_root": str(OUT),
        "files_on_disk": int(len(on_disk)),
        "sums_file": {
            "entries": int(len(sums)), "missing_files": len(sums_missing),
            "sha256_mismatches": len(sums_bad),
            "mismatch_examples": sums_bad[:5], "match": not sums_missing
            and not sums_bad,
        },
        "sums_coverage": {
            "files_not_covered_by_sums": uncovered,
            "only_sums_file_excluded": bool(
                uncovered == ["artifact_sha256sums.txt"] or not uncovered),
        },
        "manifest": {
            "entries": int(len(man["entries"])),
            "sha256_mismatches": len(man_bad),
            "mismatch_examples": man_bad[:5],
            "covers_every_file_except_self_and_sums": bool(
                man_paths == expected_manifest_paths),
            "match": not man_bad and man_paths == expected_manifest_paths,
        },
        "integrity_report": {
            "entries": int(len(integ_paths)),
            "sha256_mismatches": len(integ_bad),
            "covers_every_file_except_self_manifest_and_sums": bool(
                integ_paths == expected_integ_paths),
            "integrity_vs_manifest_mismatches": len(cross),
            "match": bool(not integ_bad and integ_paths == expected_integ_paths
                          and not cross),
        },
        "frozen_phase2b_manifest": {
            "entries": int(len(frozen_man)), "missing_files": len(f_missing),
            "sha256_mismatches": len(f_bad), "match": not f_missing
            and not f_bad,
        },
        "frozen_vendor_repo": {
            "path": str(REPO), "head": head.stdout.strip(),
            "expected_commit": "7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0",
            "porcelain": git.stdout.strip(),
            "clean": git.stdout.strip() == "",
        },
        "claims": {
            "PHASE2B_D_RESULT": integ["PHASE2B_D_RESULT"],
            "PHASE2B_RESULT_frozen": integ["PHASE2B_RESULT_frozen"],
            "REPRODUCTION_PASS": integ["reproduction"]["REPRODUCTION_PASS"],
            "ALL_INPUT_HASHES_MATCH":
                integ["consumed_inputs_verified"]["ALL_INPUT_HASHES_MATCH"],
            "cost": integ["cost"],
        },
        "seconds": round(time.time() - t0, 1),
    }
    ok = all(report[k]["match"] for k in (
        "sums_file", "manifest", "integrity_report",
        "frozen_phase2b_manifest"))
    report["VERIFICATION_PASS"] = bool(ok and report["frozen_vendor_repo"]
                                       ["clean"])
    write_json(WORK / "verification_report.json", report)
    print("VERIFICATION_PASS = %s" % report["VERIFICATION_PASS"], flush=True)
    print("sums=%d manifest=%d integrity=%d on_disk=%d"
          % (len(sums), len(man["entries"]), len(integ_paths), len(on_disk)),
          flush=True)
    return 0 if report["VERIFICATION_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
