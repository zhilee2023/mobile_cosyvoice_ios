"""Regress the bounded text-to-WAV CPU pipeline used by iOS v0.4.0."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, flow, hift


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output = ROOT / "builds" / "text-chain-v040"
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / "models" / "CosyVoice3"
    started = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "pc_llm_probe.py"), str(model),
                    "--max-new-tokens", "161", "--output", str(output / "llm.json"),
                    "--tokens-output", str(output / "speech-tokens.csv")], check=True)
    llm = json.loads((output / "llm.json").read_text(encoding="utf-8"))
    target = csv_tokens(output / "speech-tokens.csv")
    assert llm["eos_seen"] and llm["invalid_token_count"] == 0 and 1 <= len(target) <= 160
    mu, _ = conditioner(model, csv_tokens(model / "prompt-speech-tokens.csv"), target)
    bucket = 256 if mu.shape[1] <= 256 else (384 if mu.shape[1] <= 384 else 512)
    mel, flow_seconds = flow(model, mu, runtime_length=bucket, memory="low", precision="high")
    reference, _ = flow(model, mu, runtime_length=512, memory="low", precision="high")
    relative_l2 = float(np.linalg.norm(mel - reference) / max(np.linalg.norm(reference), 1e-12))
    assert relative_l2 < 1e-3
    pcm, hift_times = hift(model, mel)
    assert len(pcm) == len(target) * 2 * 480 and np.isfinite(pcm).all()
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    assert rms > 0.001
    write_wav(output / "text-to-voice-cpu.wav", pcm)
    report = {"text": llm["text"], "llm": llm, "target_tokens": len(target),
              "audio_seconds": len(pcm) / 24000, "effective_frames": int(mu.shape[1]),
              "bucket_frames": bucket, "mel_relative_l2_vs_512": relative_l2,
              "flow_seconds": flow_seconds, **hift_times, "pcm_rms": rms,
              "pcm_peak": float(np.max(np.abs(pcm))), "total_seconds": time.perf_counter() - started,
              "scope": "Windows CPU full text-to-WAV; not iPhone runtime or true streaming"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "llm"}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
