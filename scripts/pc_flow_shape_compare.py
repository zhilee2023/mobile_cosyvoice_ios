"""Check whether a 256-frame Flow session matches the existing 512-frame bucket.

This is a Windows CPU shape/quality diagnostic, not an iPhone memory estimate.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from pc_flow_hift_probe import conditioner, csv_tokens, flow


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"


def main() -> int:
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    target = prompt[:25]
    mu, _ = conditioner(MODEL_DIR, prompt, target)
    outputs: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for frames in (256, 512):
        outputs[frames], timings[frames] = flow(MODEL_DIR, mu, runtime_length=frames)
        gc.collect()
    difference = outputs[256] - outputs[512]
    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(outputs[512]), 1e-12))
    result = {
        "test": "flow-256-vs-512-same-synthetic-condition",
        "prompt_tokens": len(prompt),
        "target_tokens": len(target),
        "actual_input_frames": int(mu.shape[1]),
        "padded_frames": [256, 512],
        "mel_shape": list(outputs[256].shape),
        "both_finite": bool(all(np.isfinite(value).all() for value in outputs.values())),
        "relative_l2": relative_l2,
        "max_abs_difference": float(np.max(np.abs(difference))),
        "seconds": {str(key): round(value, 3) for key, value in timings.items()},
        "scope_note": "Synthetic target tokens. CPU shape check only; iPhone Metal and audio quality untested.",
    }
    path = ROOT / "reports" / "pc-flow-shape-compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["both_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
