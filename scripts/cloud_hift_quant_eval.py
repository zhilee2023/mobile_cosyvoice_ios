"""Compare W8/W4 HiFT core on the same FP16 Flow mel and real-token sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_sensitivity_metrics import compare as audio_compare
from cloud_flow_quant_eval import metrics, write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, flow, hift


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"
OUTPUT = ROOT / "reports" / "cloud-hift-quant"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    speech = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")
    report = {
        "test": "HiFT core FP32 vs W8/W4 HQQ, same FP16 Flow mel",
        "token_source": "74 real speech tokens generated from 你好，欢迎来到离线语音测试。",
        "backend": "MNN 3.6.1 Linux CPU normal precision",
        "limitations": "One short Chinese sample; no phone Metal memory or subjective listening validation.",
        "cases": {},
    }
    for name, target, padded in (("first25", speech[:25], 256), ("full74", speech, 512)):
        mu, _ = conditioner(MODEL_DIR, prompt, target)
        mel, flow_seconds = flow(MODEL_DIR, mu, runtime_length=padded, memory="low")
        baseline, baseline_time = hift(MODEL_DIR, mel)
        if not np.isfinite(baseline).all():
            raise RuntimeError(f"Baseline PCM is non-finite: {name}")
        case = {"mel_frames": int(mel.shape[1]), "flow_seconds": flow_seconds,
                "baseline_hift": baseline_time, "variants": {}}
        write_wav(OUTPUT / f"{name}-fp32.wav", baseline)
        for label in ("w8", "w4"):
            path = ROOT / "quant" / f"cloud-hift-core-{label}.mnn"
            if not path.is_file():
                case["variants"][label] = {"status": "missing_candidate"}
                continue
            try:
                pcm, timing = hift(MODEL_DIR, mel, core_model_path=path)
                if pcm.shape != baseline.shape or not np.isfinite(pcm).all():
                    raise RuntimeError("Shape mismatch or non-finite PCM")
                case["variants"][label] = {"status": "ran", "hift": timing,
                                            "pcm": metrics(baseline, pcm),
                                            "audio_spectrum": audio_compare(baseline, pcm)["fft_1024"]}
                write_wav(OUTPUT / f"{name}-{label}.wav", pcm)
            except Exception as error:
                case["variants"][label] = {"status": "error", "message": str(error)}
        report["cases"][name] = case
    (OUTPUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(v["status"] == "ran" for c in report["cases"].values()
                    for v in c["variants"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
