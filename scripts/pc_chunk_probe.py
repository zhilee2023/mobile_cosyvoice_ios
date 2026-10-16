"""Exercise multiple token lengths on one Windows MNN Conditioner session."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import MNN
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"


def main() -> int:
    with (MODEL_DIR / "prompt-speech-tokens.csv").open(newline="", encoding="utf-8") as source:
        prompt = [int(item) for row in csv.reader(source) for item in row if item.strip()]
    net = MNN.Interpreter(str(MODEL_DIR / "flow-conditioner.fp32.mnn"))
    session = net.createSession({"backend": "CPU", "precision": "high", "memory": "low", "numThread": 4})
    report = []
    previous_25 = None
    for count in (12, 25, 50, 25):
        target = (prompt * (count // len(prompt) + 1))[:count]
        ids = np.asarray([prompt + target], dtype=np.int32)
        start_resize = time.perf_counter()
        net.resizeTensor(net.getSessionInput(session, "tokens"), tuple(ids.shape))
        net.resizeSession(session)
        resize_seconds = time.perf_counter() - start_resize
        tensor = net.getSessionInput(session, "tokens")
        host = MNN.Tensor(ids.shape, MNN.Halide_Type_Int, ids, MNN.Tensor_DimensionType_Caffe)
        if not tensor.copyFrom(host):
            raise RuntimeError("Could not refeed token input")
        start_run = time.perf_counter()
        net.runSession(session)
        output = np.asarray(net.getSessionOutput(session, "mu").getNumpyData(), dtype=np.float32).copy()
        run_seconds = time.perf_counter() - start_run
        if output.size != 80 * 2 * ids.size or not np.isfinite(output).all():
            raise RuntimeError(f"Invalid output at target length {count}")
        same_as_prior_25 = None
        if count == 25:
            same_as_prior_25 = None if previous_25 is None else bool(np.array_equal(output, previous_25))
            previous_25 = output
        report.append({"target_tokens": count, "input_tokens": int(ids.size), "output_values": int(output.size),
                       "resize_seconds": round(resize_seconds, 5), "run_seconds": round(run_seconds, 5),
                       "finite": True, "same_as_prior_25": same_as_prior_25})
    if report[-1]["same_as_prior_25"] is not True:
        raise RuntimeError("Returning to 25 tokens did not reproduce the same output")
    result = {"test": "single-session-dynamic-conditioner", "steps": report,
              "scope_note": "Synthetic speech token IDs; not audio generation"}
    path = ROOT / "reports" / "pc-conditioner-dynamic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
