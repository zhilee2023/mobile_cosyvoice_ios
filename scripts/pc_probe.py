"""Smoke-test real CosyVoice3 MNN graphs on Windows CPU.

The Conditioner test uses public prompt speech tokens as a synthetic target. It
proves graph loading, dynamic input resizing, and finite output; it does not
prove the LLM, Flow/HiFT, voice quality, or end-to-end streaming.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--target-tokens", type=int, choices=(12, 25, 50), default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import MNN  # Use the Windows MNN 3.6.1 environment, not system Python.
    import numpy as np

    model_dir = args.model_dir.resolve()
    model = model_dir / "flow-conditioner.fp32.mnn"
    prompt_file = model_dir / "prompt-speech-tokens.csv"
    if not model.is_file() or not prompt_file.is_file():
        parser.error(f"Missing {model.name} or {prompt_file.name} in {model_dir}")

    with prompt_file.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    prompt = [int(value) for row in rows for value in row if value.strip()]
    if not prompt:
        parser.error("Prompt token CSV is empty")
    target = (prompt * ((args.target_tokens // len(prompt)) + 1))[: args.target_tokens]
    ids = np.asarray([prompt + target], dtype=np.int32)

    started = time.perf_counter()
    net = MNN.Interpreter(str(model))
    loaded = time.perf_counter()
    session = net.createSession({"backend": "CPU", "precision": "high", "memory": "low", "numThread": 4})
    created = time.perf_counter()
    token_input = net.getSessionInput(session, "tokens")
    net.resizeTensor(token_input, tuple(ids.shape))
    net.resizeSession(session)
    resized = time.perf_counter()
    token_input = net.getSessionInput(session, "tokens")
    input_type = str(token_input.getDataType())
    if token_input.getDataType() != MNN.Halide_Type_Int:
        raise RuntimeError(f"Conditioner input is not int32: {input_type}")
    host = MNN.Tensor(ids.shape, MNN.Halide_Type_Int, ids, MNN.Tensor_DimensionType_Caffe)
    runs = []
    output = None
    repeat_max_abs_difference = None
    repeat_relative_l2 = None
    for _ in range(2):
        if not token_input.copyFrom(host):
            raise RuntimeError("Could not copy speech token IDs into Conditioner")
        start = time.perf_counter()
        net.runSession(session)
        value = np.asarray(net.getSessionOutput(session, "mu").getNumpyData(), dtype=np.float32).copy()
        runs.append(time.perf_counter() - start)
        if output is None:
            output = value
        else:
            delta = value - output
            repeat_max_abs_difference = float(np.max(np.abs(delta)))
            denominator = float(np.linalg.norm(output.ravel()))
            repeat_relative_l2 = float(np.linalg.norm(delta.ravel()) / denominator) if denominator else None
    assert output is not None
    expected_values = 80 * 2 * ids.shape[1]
    if output.size != expected_values or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid Conditioner output: values={output.size}, expected={expected_values}")

    report = {
        "test": "synthetic-conditioner-cpu",
        "model_dir": str(model_dir),
        "prompt_tokens": len(prompt),
        "synthetic_target_tokens": len(target),
        "input_shape": list(ids.shape),
        "output_values": int(output.size),
        "output_max_abs": float(np.max(np.abs(output))),
        "output_finite": True,
        "repeat_identical": repeat_max_abs_difference == 0,
        "repeat_max_abs_difference": repeat_max_abs_difference,
        "repeat_relative_l2": repeat_relative_l2,
        "load_seconds": round(loaded - started, 4),
        "create_session_seconds": round(created - loaded, 4),
        "resize_seconds": round(resized - created, 4),
        "run_seconds": [round(seconds, 4) for seconds in runs],
        "scope_note": "Not an end-to-end speech generation or streaming test",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
