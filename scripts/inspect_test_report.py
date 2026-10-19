"""Read ONE phone report; never merge unrelated/stale files into a new run.

Compare independent Flow thread fixtures only when version/build/fixture/file
stamps match. Stamps are identity hints, not cryptographic model verification.
"""
from __future__ import annotations
import argparse
from array import array
import base64
import json
import math
from pathlib import Path
import sys

def flow_thread_comparisons(report: dict) -> list[dict]:
    groups: dict[str, dict[int, tuple[dict, dict, array]]] = {}
    for run in report.get("runs", []):
        if run.get("kind") != "module-experiment" or run.get("state") != "passed":
            continue
        result = run.get("result", {})
        for row in result.get("rows", []):
            if not row.get("melF32Base64") or not row.get("finite"):
                continue
            if row.get("completedRuns") != 3 or row.get("melShape") != [80, 140]:
                continue
            stamps = row.get("modelFileStamps", {})
            if len(stamps) < 7 or any(s.get("bytes", 0) <= 0 or s.get("modifiedEpoch", 0) <= 0 for s in stamps.values()):
                continue
            threads = row.get("threads")
            if threads not in (2, 3, 4) or row.get("actualThreads") != threads:
                continue
            if row.get("fixtureID") != "real70-prompt87-effective314-noise-v1-f32-native" or row.get("byteOrder") != "little-endian":
                continue
            try:
                raw = base64.b64decode(row["melF32Base64"], validate=True)
                if len(raw) != 80 * 140 * 4:
                    continue
                mel = array("f"); mel.frombytes(raw)
                if sys.byteorder != "little": mel.byteswap()
                if not all(math.isfinite(x) for x in mel): continue
            except (ValueError, TypeError):
                continue
            key = json.dumps([run.get("version"), run.get("build"), row["fixtureID"], stamps], sort_keys=True)
            groups.setdefault(key, {})[threads] = (run, row, mel)
    comparisons = []
    for group in groups.values():
        if 4 not in group:
            comparisons.append({"state": "missing-matching-4-thread-reference", "availableThreads": sorted(group)})
            continue
        ref_run, _, ref = group[4]
        energy = sum(x*x for x in ref)
        if energy <= 1e-20:
            comparisons.append({"state": "invalid-silent-reference", "referenceRun": ref_run["id"]})
            continue
        for threads, (run, row, mel) in sorted(group.items()):
            l2 = math.sqrt(sum((a-b)**2 for a,b in zip(ref,mel))/energy)
            comparisons.append({"referenceRun": ref_run["id"], "candidateRun": run["id"],
                "threads": threads, "melRelativeL2": l2, "numericalGatePassed": l2 <= 1e-3,
                "warmTwoStepsMeanSeconds": row.get("warmTwoStepsMeanSeconds"), "audioQualityValidated": False})
    return comparisons

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    if report.get("schemaVersion") != 1:
        raise ValueError("Unknown report schema")
    print(json.dumps({"runs": [{"id": r.get("id"), "label": r.get("label"), "state": r.get("state"),
                    "version": r.get("version"), "build": r.get("build")} for r in report.get("runs", [])],
                      "flowThreadComparisons": flow_thread_comparisons(report)}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
