"""Regress 32-aligned Flow padding against the shipped 128-frame buckets.

Uses fixed real token prefixes; 149-token case repeats a real sequence only for
shape sensitivity, NOT to claim natural speech. iPhone still needs validation.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from pc_flow_hift_probe import conditioner, csv_tokens, flow

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    model = ROOT / "models" / "CosyVoice3"
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    real = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")
    results = []
    for count in (25, 74, 149, 160):
        target = (real * 3)[:count]
        mu, _ = conditioner(model, prompt, target)
        effective = int(mu.shape[1])
        baseline = 256 if effective <= 256 else 384 if effective <= 384 else 512
        aligned = (effective + 31) // 32 * 32
        outputs, times = {}, {}
        for name, length in (("baseline", baseline), ("aligned", aligned)):
            started = time.perf_counter()
            outputs[name], forward = flow(model, mu, runtime_length=length, memory="low", precision="high")
            times[name] = {"forward_seconds": forward, "load_resize_forward_seconds": time.perf_counter() - started}
        reference, candidate = outputs["baseline"], outputs["aligned"]
        l2 = float(np.linalg.norm(reference - candidate) / max(np.linalg.norm(reference), 1e-12))
        assert np.isfinite(candidate).all() and l2 < 1e-3, (count, l2)
        results.append({"speech_tokens": count, "effective_frames": effective,
                        "baseline_frames": baseline, "aligned_frames": aligned,
                        "mel_relative_l2": l2, "times": times})
        print(json.dumps(results[-1]), flush=True)
    output = ROOT / "builds" / "sentence-v050"
    output.mkdir(parents=True, exist_ok=True)
    (output / "aligned-report.json").write_text(json.dumps({"scope": "Windows CPU shape regression, not iPhone speed/quality", "cases": results}, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
