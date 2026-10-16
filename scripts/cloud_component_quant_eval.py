"""Compare separate Conditioner and F0 W8/W4 recipes end to end on cloud CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import MNN

from cloud_flow_quant_eval import metrics, write_wav
from audio_sensitivity_metrics import compare as audio_compare
from pc_flow_hift_probe import conditioner, csv_tokens, flow, get, hift, model_session, put


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"
OUTPUT = ROOT / "reports" / "cloud-component-quant"


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


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    speech = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")
    report = {
        "test": "Separate Conditioner and HiFT-F0 FP32/W8/W4 recipes",
        "backend": "MNN 3.6.1 Linux CPU normal precision",
        "token_source": "74 real speech tokens generated from 你好，欢迎来到离线语音测试。",
        "limitations": "One short sample; PCM waveforms are not a substitute for listening or phone memory tests.",
        "cases": {},
    }
    for name, target, padded in (("first25", speech[:25], 256), ("full74", speech, 512)):
        base_mu, _ = conditioner(MODEL_DIR, prompt, target)
        base_mel, _ = flow(MODEL_DIR, base_mu, runtime_length=padded, memory="low")
        base_pcm, _ = hift(MODEL_DIR, base_mel)
        base_f0 = f0_forward(base_mel, MODEL_DIR / "hift-f0.fp32.mnn")
        write_wav(OUTPUT / f"{name}-fp32.wav", base_pcm)
        case = {"target_speech_tokens": len(target), "conditioner": {}, "f0": {}}
        for label in ("w8", "w4"):
            path = ROOT / "quant" / f"cloud-conditioner-{label}.mnn"
            try:
                mu, elapsed = conditioner(MODEL_DIR, prompt, target, model_path=path)
                if mu.shape != base_mu.shape or not np.isfinite(mu).all():
                    raise RuntimeError("Shape mismatch or non-finite Conditioner mu")
                result = {"status": "ran", "seconds": elapsed, "mu": metrics(base_mu, mu)}
                mel, _ = flow(MODEL_DIR, mu, runtime_length=padded, memory="low")
                pcm, _ = hift(MODEL_DIR, mel)
                result["mel"] = metrics(base_mel, mel)
                result["pcm"] = metrics(base_pcm, pcm)
                result["audio_spectrum"] = audio_compare(base_pcm, pcm)["fft_1024"]
                write_wav(OUTPUT / f"{name}-conditioner-{label}.wav", pcm)
                case["conditioner"][label] = result
            except Exception as error:
                case["conditioner"][label] = {"status": "error", "message": str(error)}
        for label in ("w8", "w4"):
            path = ROOT / "quant" / f"cloud-f0-{label}.mnn"
            try:
                f0 = f0_forward(base_mel, path)
                pcm, timing = hift(MODEL_DIR, base_mel, f0_model_path=path)
                if pcm.shape != base_pcm.shape or not np.isfinite(pcm).all():
                    raise RuntimeError("Shape mismatch or non-finite F0 PCM")
                case["f0"][label] = {"status": "ran", "timing": timing,
                                     "f0": metrics(base_f0, f0),
                                     "pcm": metrics(base_pcm, pcm),
                                     "audio_spectrum": audio_compare(base_pcm, pcm)["fft_1024"]}
                write_wav(OUTPUT / f"{name}-f0-{label}.wav", pcm)
            except Exception as error:
                case["f0"][label] = {"status": "error", "message": str(error)}
        report["cases"][name] = case
    (OUTPUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(v["status"] == "ran" for c in report["cases"].values()
                    for part in ("conditioner", "f0") for v in c[part].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
