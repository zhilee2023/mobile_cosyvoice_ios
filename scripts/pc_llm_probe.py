"""Bounded Windows MNN-LLM speech-token generation probe.

This only tests LLM token generation. It does not synthesize audio. The public
CosyVoice3-MNN Android port's fixed prompt and speech-token offset are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time


SPEECH_OFFSET = 151924
SPEECH_MAX = SPEECH_OFFSET + 6560
SPEECH_EOS = SPEECH_OFFSET + 6562
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--text", default="你好，欢迎来到离线语音测试。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sampler", choices=("mixed", "greedy", "cosyvoice_ras"), default="mixed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be 1..256")

    import MNN.llm as mllm

    model_dir = args.model_dir.resolve()
    required = ("config-cpu-cosyvoice-ras.json", "llm_config.json", "llm.mnn", "llm.mnn.weight", "embeddings_bf16.bin", "tokenizer.mtok", "prompt-speech-tokens.csv")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        parser.error("Missing model files: " + ", ".join(missing))

    config_source = json.loads((model_dir / "config-cpu-cosyvoice-ras.json").read_text(encoding="utf-8"))
    # The public Android runtime now uses a mixed sampler with explicit
    # repetition penalty; the packaged static config still says cosyvoice_ras.
    config_source.update({"sampler_type": args.sampler, "mixed_samplers": ["penalty", "topK", "topP", "temperature"],
                          "repetition_penalty": 1.15, "max_new_tokens": args.max_new_tokens})
    config = model_dir / "config-pc-runtime.json"
    config.write_text(json.dumps(config_source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (model_dir / "prompt-speech-tokens.csv").open(newline="", encoding="utf-8") as source:
        prompt_speech_tokens = [int(value) for row in csv.reader(source) for value in row if value.strip()]
    started = time.perf_counter()
    llm = mllm.create(str(config))
    if not llm:
        raise RuntimeError("MNN-LLM create failed")
    # The Windows Python wrapper's load() returns None even on success; errors
    # surface as exceptions or when the following tokenizer/generate calls run.
    llm.load()
    loaded = time.perf_counter()
    prompt = llm.apply_chat_template(PROMPT_PREFIX + args.text)
    if not prompt:
        prompt = PROMPT_PREFIX + args.text
    ids = list(llm.tokenizer_encode(prompt))
    ids.extend(token + SPEECH_OFFSET for token in prompt_speech_tokens)
    encoded = time.perf_counter()
    output_ids = list(llm.generate(ids))
    generated = time.perf_counter()
    context = llm.context
    speech_tokens = [token - SPEECH_OFFSET for token in output_ids if SPEECH_OFFSET <= token <= SPEECH_MAX]
    invalid = [token for token in output_ids if not SPEECH_OFFSET <= token <= SPEECH_MAX and token != SPEECH_EOS]
    report = {
        "test": "pc-mnn-llm-speech-token-probe",
        "text": args.text,
        "input_token_count": len(ids),
        "output_token_count": len(output_ids),
        "speech_token_count": len(speech_tokens),
        "invalid_token_count": len(invalid),
        "eos_seen": SPEECH_EOS in output_ids,
        "load_seconds": round(loaded - started, 4),
        "encode_seconds": round(encoded - loaded, 4),
        "generate_seconds": round(generated - encoded, 4),
        "first_16_raw_output_ids": output_ids[:16],
        "first_16_speech_tokens": speech_tokens[:16],
        "mnn_prompt_len": context.prompt_len,
        "mnn_generated_len": context.gen_seq_len,
        "mnn_prefill_seconds": round(context.prefill_us / 1_000_000, 4),
        "mnn_decode_seconds": round(context.decode_us / 1_000_000, 4),
        "mnn_status": context.status,
        "scope_note": "No Flow, HiFT, audio, or streaming validation yet",
    }
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    if args.tokens_output:
        args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_output.write_text(",".join(str(token) for token in speech_tokens) + "\n", encoding="ascii")
    print(result)
    if not speech_tokens or invalid:
        raise RuntimeError("LLM did not produce a valid speech-token sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
