"""Local per-component W8/W4 sensitivity test on fixed real speech tokens.

The external-FP16 Flow and already mixed-precision LLM are left unchanged.
Each candidate changes only one component; all downstream stages use the same
inputs and model versions. WAVs support subjective checks, not automatic MOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cloud_flow_quant_eval import metrics, write_wav
from pc_flow_hift_probe import (
    conditioner, csv_tokens, flow, get, hift, model_session, put,
)
import MNN


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"
OUTPUT = ROOT / "builds" / "local-sensitivity"
CANDIDATES = {
    "conditioner": {
        "w8": ROOT / "builds" / "conditioner-w8-test.mnn",
        "w4": ROOT / "builds" / "conditioner-w4-test.mnn",
    },
    "f0": {
        "w8": ROOT / "builds" / "hift-f0-w8-test.mnn",
        "w4": ROOT / "builds" / "hift-f0-w4-test.mnn",
    },
    "hift_core": {
        "w8": ROOT / "builds" / "hift-core-w8-test.mnn",
        "w4": ROOT / "builds" / "hift-core-w4-test.mnn",
        "w4_block32": ROOT / "builds" / "hift-core-w4-block32-test.mnn",
    },
}
SAMPLES = {
    "zh_short": ROOT / "testdata" / "pc-real-74-speech-tokens.csv",
    "zh_long": OUTPUT / "sample2-tokens.csv",
    "en_short": OUTPUT / "sample3-tokens.csv",
}


def f0_forward(mel: np.ndarray, path: Path) -> np.ndarray:
    value = np.ascontiguousarray(mel.reshape(1, 80, -1))
    network, session = model_session(path)
    network.resizeTensor(network.getSessionInput(session, "mel"), tuple(value.shape))
    network.resizeSession(session)
    put(network, session, "mel", value, MNN.Halide_Type_Float)
    network.runSession(session)
    result = get(network, session, "f0")
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite F0")
    return result


def checked_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise RuntimeError("Shape mismatch or non-finite candidate")
    return metrics(reference, candidate)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    report = {
        "test": "local per-component weight-quant sensitivity",
        "backend": "Windows MNN 3.6.1 CPU normal precision",
        "baseline": "LLM as published (already mixed W4/W8); Conditioner FP32; Flow FP16; HiFT-F0 FP32; HiFT-core FP32",
        "limitations": "Three generated token streams, one prompt voice; no iPhone Metal measurements or blind listening scores.",
        "candidates": {}, "samples": {},
    }
    for part, variants in CANDIDATES.items():
        source = MODEL_DIR / {"conditioner": "flow-conditioner.fp32.mnn",
                              "f0": "hift-f0.fp32.mnn",
                              "hift_core": "hift-core.fp32.mnn"}[part]
        report["candidates"][part] = {
            label: {"path": str(path), "bytes": path.stat().st_size,
                    "source_bytes": source.stat().st_size,
                    "size_reduction_fraction": 1 - path.stat().st_size / source.stat().st_size}
            for label, path in variants.items()
        }
    for sample_name, token_path in SAMPLES.items():
        target = csv_tokens(token_path)
        print(f"sample={sample_name} tokens={len(target)}", flush=True)
        base_mu, _ = conditioner(MODEL_DIR, prompt, target)
        padded = 256 if base_mu.shape[1] <= 256 else 512
        base_mel, _ = flow(MODEL_DIR, base_mu, runtime_length=padded, memory="low")
        base_pcm, _ = hift(MODEL_DIR, base_mel)
        base_f0 = f0_forward(base_mel, MODEL_DIR / "hift-f0.fp32.mnn")
        write_wav(OUTPUT / f"{sample_name}-baseline.wav", base_pcm)
        sample = {"target_tokens": len(target), "audio_seconds": len(base_pcm) / 24000,
                  "flow_padded_frames": padded, "conditioner": {}, "f0": {}, "hift_core": {}}
        for label, path in CANDIDATES["conditioner"].items():
            try:
                mu, _ = conditioner(MODEL_DIR, prompt, target, model_path=path)
                result = {"mu": checked_metrics(base_mu, mu)}
                mel, _ = flow(MODEL_DIR, mu, runtime_length=padded, memory="low")
                result["mel"] = checked_metrics(base_mel, mel)
                pcm, _ = hift(MODEL_DIR, mel)
                result["pcm"] = checked_metrics(base_pcm, pcm)
                write_wav(OUTPUT / f"{sample_name}-conditioner-{label}.wav", pcm)
                sample["conditioner"][label] = result
            except Exception as error:
                sample["conditioner"][label] = {"error": str(error)}
        for label, path in CANDIDATES["f0"].items():
            try:
                f0 = f0_forward(base_mel, path)
                pcm, _ = hift(MODEL_DIR, base_mel, f0_model_path=path)
                sample["f0"][label] = {"f0": checked_metrics(base_f0, f0),
                                       "pcm": checked_metrics(base_pcm, pcm)}
                write_wav(OUTPUT / f"{sample_name}-f0-{label}.wav", pcm)
            except Exception as error:
                sample["f0"][label] = {"error": str(error)}
        for label, path in CANDIDATES["hift_core"].items():
            try:
                pcm, _ = hift(MODEL_DIR, base_mel, core_model_path=path)
                sample["hift_core"][label] = {"pcm": checked_metrics(base_pcm, pcm)}
                write_wav(OUTPUT / f"{sample_name}-core-{label}.wav", pcm)
            except Exception as error:
                sample["hift_core"][label] = {"error": str(error)}
        report["samples"][sample_name] = sample
        (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"sample={sample_name} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
