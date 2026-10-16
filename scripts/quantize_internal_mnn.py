"""Quantize a self-contained FP32 MNN component and verify its actual size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error("Source graph is missing")
    if Path(str(source) + ".weight").exists():
        parser.error("Source has external weights; use original ONNX/FP32 export instead")
    if output.exists():
        parser.error("Refusing to overwrite an existing candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    import _tools

    command = ["MNNConvert", "-f", "MNN", "--modelFile", str(source),
               "--MNNModel", str(output), "--weightQuantBits", str(args.bits),
               "--weightQuantBlock", "128"]
    if args.bits == 4:
        command.append("--hqq")
    print("converter flags:", " ".join(command[1:]), flush=True)
    _tools.mnnconvert(command)
    source_bytes = source.stat().st_size
    candidate_bytes = output.stat().st_size if output.is_file() else 0
    report = {
        "source": str(source), "candidate": str(output), "bits": args.bits,
        "recipe": "MNN FP32 internal -> weight-only block 128" + (" HQQ" if args.bits == 4 else ""),
        "source_bytes": source_bytes, "candidate_bytes": candidate_bytes,
        "size_reduction_fraction": 1 - candidate_bytes / source_bytes if candidate_bytes else 0,
        "materially_compressed": bool(candidate_bytes and candidate_bytes < source_bytes * 0.9),
        "note": "Some low-input-channel convolutions fall back to channel-wise quantization; see converter log.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["materially_compressed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
