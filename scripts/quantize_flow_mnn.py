"""Try MNN 3.6.1 weight-only quantization of the published FP16 Flow graph.

This is an experiment: the original graph already stores FP16 weights, and
MNNConvert may leave those weights untouched. Size and inference are checked
separately; converter success alone is not accepted as quantization success.
"""

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
    source_weight = Path(str(source) + ".weight")
    if not source.is_file() or not source_weight.is_file():
        parser.error("Source graph and external .weight are both required")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output) + ".weight").exists():
        parser.error("Refusing to overwrite an existing candidate")

    # Call the C++ converter directly; the Python CLI wrapper separately
    # installs an optional telemetry SDK and is unsuitable for a clean CI job.
    import _tools

    command = ["MNNConvert", "-f", "MNN", "--modelFile", str(source),
               "--MNNModel", str(output), "--weightQuantBits", str(args.bits),
               "--weightQuantBlock", "128", "--saveExternalData"]
    if args.bits == 4:
        command.append("--hqq")
    print("converter flags:", " ".join(command[1:]), flush=True)
    _tools.mnnconvert(command)

    candidate_weight = Path(str(output) + ".weight")
    source_bytes = source.stat().st_size + source_weight.stat().st_size
    candidate_bytes = (output.stat().st_size if output.exists() else 0) + (
        candidate_weight.stat().st_size if candidate_weight.exists() else 0)
    complete = output.is_file() and candidate_weight.is_file() and candidate_weight.stat().st_size > 0
    size_reduction = 1 - candidate_bytes / source_bytes if candidate_bytes else 0
    result = {
        "source": str(source), "candidate": str(output), "bits": args.bits,
        "recipe": "MNN->MNN weight-only, block 128" + (", HQQ" if args.bits == 4 else ""),
        "source_bytes": source_bytes, "candidate_bytes": candidate_bytes,
        "size_reduction_fraction": size_reduction,
        "complete_external_weights": complete,
        "materially_compressed": bool(complete and size_reduction >= 0.10),
        "warning": "FP16 source may not be eligible for MNNConvert weightQuantBits; inspect size and output accuracy.",
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
