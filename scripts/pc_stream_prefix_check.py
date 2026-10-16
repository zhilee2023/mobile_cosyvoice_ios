"""Measure prefix sensitivity of the published offline distilled MNN graph.

This is not an implementation or validation of official streaming. In
particular, this graph does not expose the official DiT streaming flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, flow, hift


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    model = ROOT / "models" / "CosyVoice3"
    output = ROOT / "builds" / "stream-prefix-feasibility"
    output.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    target = csv_tokens(ROOT / "builds" / "text-chain-v040" / "speech-tokens.csv")
    hop, lookahead = 25, 3
    padding = (-len(prompt)) % hop
    emitted_tokens = hop + padding
    prefix_count = emitted_tokens + lookahead
    assert len(target) > prefix_count
    full_mu, _ = conditioner(model, prompt, target)
    prefix_mu, _ = conditioner(model, prompt, target[:prefix_count])
    # Earlier pre-lookahead outputs may be compared without pretending that
    # the exported offline DiT has been given the official causal mask.
    prefix_mu = np.ascontiguousarray(prefix_mu[:, :-lookahead * 2])
    prefix_mel, prefix_seconds = flow(model, prefix_mu, runtime_length=256, memory="low", precision="high")
    full_mel, full_seconds = flow(model, full_mu, runtime_length=512, memory="low", precision="high")
    reference = full_mel[:, :prefix_mel.shape[1]]
    relative_l2 = float(np.linalg.norm(prefix_mel - reference) / np.linalg.norm(reference))
    prefix_pcm, _ = hift(model, prefix_mel)
    full_pcm, _ = hift(model, full_mel)
    write_wav(output / "partial-context.wav", prefix_pcm)
    write_wav(output / "full-context.wav", full_pcm)
    report = {"target_tokens": len(target), "prompt_tokens": len(prompt), "hop": hop,
              "prompt_alignment_pad": padding, "lookahead_tokens": lookahead,
              "first_prefix_tokens": prefix_count, "emitted_tokens": emitted_tokens,
              "prefix_audio_seconds": len(prefix_pcm) / 24000,
              "full_audio_seconds": len(full_pcm) / 24000,
              "prefix_mel_relative_l2_vs_final": relative_l2,
              "prefix_flow_seconds": prefix_seconds, "full_flow_seconds": full_seconds,
              "scope": "Prefix sensitivity of offline MNN student; not official causal streaming or constant-memory proof"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
