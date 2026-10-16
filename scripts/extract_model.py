"""Safely extract the required model files and verify their hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from download_model import EXPECTED_BYTES, EXPECTED_SHA256


KNOWN_FLOW_GRAPH_NAME = "flow.cfg-student-2step.batch1.fp16.mnn"
KNOWN_FLOW_GRAPH_SHA256 = "B74ECE0A1A2E165EEF53DBC8D628310A1B464323D4729959069744BAECF418D7"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest().upper()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output", type=Path, default=root / "models" / "CosyVoice3")
    parser.add_argument("--accept-known-flow-graph-mismatch", action="store_true",
                        help="Only after full published ZIP SHA check; records the known upstream graph mismatch")
    args = parser.parse_args()
    archive = args.zip_path.resolve()
    output = args.output.resolve()
    if not archive.is_file():
        print(f"ZIP not found: {archive}", file=sys.stderr)
        return 2
    if args.accept_known_flow_graph_mismatch:
        if archive.stat().st_size != EXPECTED_BYTES or digest(archive) != EXPECTED_SHA256:
            print("Cannot use exception: published ZIP size/SHA-256 failed", file=sys.stderr)
            return 1
    manifest = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
    specs = {item["name"]: item for item in manifest["files"]}
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = {}
        for info in source.infolist():
            name = Path(info.filename).name
            if name in specs:
                if name in members:
                    print(f"Duplicate model file in ZIP: {name}", file=sys.stderr)
                    return 1
                members[name] = info
        absent = sorted(set(specs) - set(members))
        if absent:
            print("ZIP lacks: " + ", ".join(absent), file=sys.stderr)
            return 1
        for name, spec in specs.items():
            destination = output / name
            def accepted(path: Path) -> bool:
                if path.stat().st_size != spec["bytes"]:
                    return False
                actual = digest(path)
                return actual == spec["sha256"] or (args.accept_known_flow_graph_mismatch
                    and name == KNOWN_FLOW_GRAPH_NAME and actual == KNOWN_FLOW_GRAPH_SHA256)
            if destination.is_file() and accepted(destination):
                print(f"verified existing: {name}" if digest(destination) == spec["sha256"]
                      else f"known upstream SHA mismatch accepted after full ZIP check: {name}")
                continue
            if destination.exists():
                print(f"Existing file differs; move it aside manually: {destination}", file=sys.stderr)
                return 1
            part = output / f"{name}.importing"
            if part.exists():
                print(f"Partial extraction exists; move it aside manually: {part}", file=sys.stderr)
                return 1
            with source.open(members[name]) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if not accepted(part):
                print(f"Size/SHA-256 mismatch: {name}. Partial file retained.", file=sys.stderr)
                return 1
            part.replace(destination)
            print(f"extracted: {name}" if digest(destination) == spec["sha256"]
                  else f"extracted with known upstream SHA exception: {name}")
    print(f"PASS: {len(specs)} verified files in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
