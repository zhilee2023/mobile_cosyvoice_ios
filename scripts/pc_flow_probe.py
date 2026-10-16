"""Run a provisional Conditioner → Flow CPU smoke test before HiFT arrives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pc_flow_hift_probe import conditioner, csv_tokens, flow


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "CosyVoice3"


def main() -> int:
    prompt = csv_tokens(MODEL_DIR / "prompt-speech-tokens.csv")
    target = prompt[:25]  # Synthetic data for a graph smoke test, not text TTS.
    mu, conditioner_seconds = conditioner(MODEL_DIR, prompt, target)
    mel, flow_seconds = flow(MODEL_DIR, mu)
    report = {"test": "synthetic-conditioner-flow-pc", "target_tokens": len(target),
              "conditioner_seconds": conditioner_seconds, "flow_seconds": flow_seconds,
              "mel_shape": list(mel.shape), "mel_finite": bool(np.isfinite(mel).all()),
              "mel_max_abs": float(np.max(np.abs(mel))),
              "flow_graph_manifest_mismatch": True,
              "scope_note": "No HiFT/WAV or text-to-speech; full ZIP audit still pending"}
    path = ROOT / "reports" / "pc-flow-synthetic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["mel_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
