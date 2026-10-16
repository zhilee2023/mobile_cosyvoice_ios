"""Compare the fixed iPhone token sample on Windows CPU normal/low precision.

Windows CPU low may fall back to normal precision. This is a graph regression
check, not proof of the iPhone ARM FP16 path's accuracy or memory use.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from audio_sensitivity_metrics import compare
from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, flow, hift


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    model = ROOT / "models" / "CosyVoice3"
    output = ROOT / "builds" / "cpu-flow-precision-check"
    output.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    target = csv_tokens(ROOT / "testdata" / "pc-real-first25-speech-tokens.csv")
    assert len(prompt) == 87 and len(target) == 25
    mu, _ = conditioner(model, prompt, target)
    report = {"scope": "Windows CPU regression; ARM CPU Low still requires iPhone validation",
              "effective_frames": int(mu.shape[1]), "flow_frames": 256, "cases": {}}
    results = {}
    for precision in ("normal", "high", "low"):
        started = time.perf_counter()
        mel, flow_seconds = flow(model, mu, runtime_length=256, memory="low", precision=precision)
        pcm, times = hift(model, mel)
        assert mel.shape == (80, 50) and len(pcm) == 24000
        assert np.isfinite(mel).all() and np.isfinite(pcm).all()
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        assert rms > 0.001
        write_wav(output / f"cpu-{precision}.wav", pcm)
        results[precision] = (mel, pcm)
        report["cases"][precision] = {"flow_seconds": flow_seconds, **times,
                                       "total_seconds": time.perf_counter() - started,
                                       "pcm_peak": float(np.max(np.abs(pcm))), "pcm_rms": rms}
    reference, candidate = results["normal"][0], results["low"][0]
    report["mel_relative_l2"] = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    report["audio"] = compare(results["normal"][1], results["low"][1])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
