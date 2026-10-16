"""Audit every required ZIP member, reporting *all* upstream mismatches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from download_model import EXPECTED_BYTES, EXPECTED_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--report", type=Path, default=root / "reports" / "zip-audit.json")
    args = parser.parse_args()
    archive = args.zip_path.resolve()
    manifest = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
    archive_bytes = archive.stat().st_size
    archive_sha256 = sha256_file(archive)
    if archive_bytes != EXPECTED_BYTES or archive_sha256 != EXPECTED_SHA256:
        raise RuntimeError("Archive size or published SHA-256 mismatch; do not extract or run")
    print("Published ZIP size and SHA-256: PASS", flush=True)
    results = []
    with zipfile.ZipFile(archive) as source:
        by_name = {Path(info.filename).name: info for info in source.infolist()}
        for spec in manifest["files"]:
            name = spec["name"]
            info = by_name.get(name)
            if info is None:
                results.append({"name": name, "status": "missing"})
                continue
            digest = hashlib.sha256()
            count = 0
            with source.open(info) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            actual = digest.hexdigest().upper()
            passed = count == spec["bytes"] and actual == spec["sha256"]
            results.append({"name": name, "status": "match" if passed else "mismatch",
                            "bytes": count, "expected_bytes": spec["bytes"],
                            "sha256": actual, "expected_sha256": spec["sha256"]})
            print(f"{name}: {'MATCH' if passed else 'MISMATCH'}", flush=True)
    report = {"zip_path": str(archive), "zip_bytes": archive_bytes, "zip_sha256": archive_sha256,
              "published_archive_sha256_match": True, "files": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report.resolve()}")
    return 0 if all(item["status"] == "match" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
