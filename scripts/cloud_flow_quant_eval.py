"""Compare W8/W4 Flow candidates against the published FP16 MNN baseline."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import wave

import numpy as np

from pc_flow_hift_probe import SAMPLE_RATE, conditioner, csv_tokens, flow, hift


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"
OUTPUT = ROOT / "reports" / "cloud-flow-quant"
SOURCE = MODEL_DIR / "flow.cfg-student-2step.batch1.fp16.mnn"
CANDIDATES = {
    "w8": ROOT / "quant" / "cloud-flow-w8.mnn",
    "w4": ROOT / "quant" / "cloud-flow-w4.mnn",
}


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    scale = max(float(np.linalg.norm(reference64)), 1e-12)
    correlation = float(np.corrcoef(reference64.ravel(), candidate64.ravel())[0, 1])
    return {
        "relative_l2": float(np.linalg.norm(difference) / scale),
        "max_abs_error": float(np.max(np.abs(difference))),
        "correlation": correlation,
        "finite": bool(np.isfinite(candidate).all()),
        "peak": float(np.max(np.abs(candidate))),
        "rms": float(np.sqrt(np.mean(candidate64 * candidate64))),
    }


def write_wav(path: Path, pcm: np.ndarray) -> None:
    peak = float(np.max(np.abs(pcm)))
    safe = pcm * np.float32(min(1.0, 0.98 / peak)) if peak else pcm
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(SAMPLE_RATE)
        destination.writeframes((np.clip(safe, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    speech = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")
    report = {
        "test": "published-fp16-vs-postconverted-w8-w4",
        "model": "Fun-CosyVoice3-0.5B-2512 distilled 2-step Flow",
        "token_source": "74 speech tokens generated on Windows from 你好，欢迎来到离线语音测试。",
        "backend": "MNN 3.6.1 Linux CPU normal precision, low memory mode",
        "limitations": "Cloud CPU correctness test only; not iPhone Metal memory/speed or listening quality.",
        "cases": {},
    }
    for name, target, frames in (("first25", speech[:25], 256), ("full74", speech, 512)):
        mu, _ = conditioner(MODEL_DIR, prompt, target)
        base, base_seconds = flow(MODEL_DIR, mu, runtime_length=frames, model_path=SOURCE, memory="low")
        if not np.isfinite(base).all():
            raise RuntimeError(f"FP16 baseline is non-finite: {name}")
        case = {
            "input_frames": int(mu.shape[1]), "padded_frames": frames,
            "output_mel_frames": int(base.shape[1]),
            "baseline_flow_seconds": base_seconds,
            "variants": {},
        }
        base_pcm = None
        if name == "full74":
            base_pcm, _ = hift(MODEL_DIR, base)
            write_wav(OUTPUT / "full74-fp16.wav", base_pcm)
        for label, path in CANDIDATES.items():
            weight = Path(str(path) + ".weight")
            if not path.is_file() or not weight.is_file() or weight.stat().st_size == 0:
                case["variants"][label] = {
                    "status": "invalid_candidate",
                    "message": "External Flow weights were not exported; do not treat converter success as quantization.",
                }
                continue
            try:
                mel, seconds = flow(MODEL_DIR, mu, runtime_length=frames, model_path=path, memory="low")
                if mel.shape != base.shape or not np.isfinite(mel).all():
                    raise RuntimeError("Shape mismatch or non-finite mel")
                result = {"status": "ran", "flow_seconds": seconds, "mel": metrics(base, mel)}
                if base_pcm is not None:
                    pcm, _ = hift(MODEL_DIR, mel)
                    if pcm.shape != base_pcm.shape or not np.isfinite(pcm).all():
                        raise RuntimeError("Shape mismatch or non-finite PCM")
                    result["pcm"] = metrics(base_pcm, pcm)
                    write_wav(OUTPUT / f"full74-{label}.wav", pcm)
                case["variants"][label] = result
            except Exception as error:
                case["variants"][label] = {"status": "error", "message": str(error)}
            gc.collect()
        report["cases"][name] = case
        gc.collect()
    (OUTPUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(v["status"] == "ran" for c in report["cases"].values()
                    for v in c["variants"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
