"""Verify an extracted CosyVoice3-MNN model directory without loading weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--sha256", action="store_true", help="hash every model file")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "model-manifest.json").read_text(encoding="utf-8"))
    directory = args.model_dir.resolve()
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 2
    failures = []
    for spec in manifest["files"]:
        path = directory / spec["name"]
        if not path.is_file():
            failures.append(f"MISSING {spec['name']}")
            continue
        actual = path.stat().st_size
        if actual != spec["bytes"]:
            failures.append(f"SIZE {spec['name']}: {actual} != {spec['bytes']}")
            continue
        if args.sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().upper() != spec["sha256"]:
                failures.append(f"SHA256 {spec['name']}")
    print(f"Checked {len(manifest['files'])} files in {directory}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("PASS: sizes" + (" and SHA-256" if args.sha256 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
