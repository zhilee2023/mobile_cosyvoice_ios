"""Verify an extracted CosyVoice3-MNN model directory without loading weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from extract_model import KNOWN_FLOW_GRAPH_NAME, KNOWN_FLOW_GRAPH_SHA256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--sha256", action="store_true", help="hash every model file")
    parser.add_argument("--accept-known-flow-graph-mismatch", action="store_true",
                        help="accept only the documented Flow graph hash from the verified release ZIP")
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
            actual_sha256 = digest.hexdigest().upper()
            if actual_sha256 != spec["sha256"] and not (args.accept_known_flow_graph_mismatch
                    and spec["name"] == KNOWN_FLOW_GRAPH_NAME and actual_sha256 == KNOWN_FLOW_GRAPH_SHA256):
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
