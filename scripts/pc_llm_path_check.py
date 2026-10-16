"""Reproduce v0.4.0's doubled MNN paths, then test v0.4.1 resolution.

Runs bad/good loads in separate processes: never call a failed tokenizer.
Windows correctness regression only, not an iOS memory/performance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from pc_llm_probe import PROMPT_PREFIX, SPEECH_OFFSET, SPEECH_MAX, SPEECH_EOS

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"llm_model": "llm.mnn", "llm_weight": "llm.mnn.weight",
          "tokenizer_file": "tokenizer.mtok", "llm_config": "llm_config.json",
          "embedding_file": "embeddings_bf16.bin"}


def worker(config_path: Path, output: Path, bad: bool) -> None:
    import MNN.llm as mllm

    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    llm = mllm.create(str(config_path))
    llm.load()  # Python wrapper returns None even on success.
    report = {"status_after_load": int(llm.context.status),
              "load_seconds": time.perf_counter() - started}
    if bad:
        assert report["status_after_load"] == -1, report
        # Do not enter tokenizer/generate after NOT_LOADED.
    else:
        assert report["status_after_load"] == 0, report
        model = Path(config["base_dir"])
        prompt = [int(x) for x in (model / "prompt-speech-tokens.csv").read_text().strip().split(",")]
        text = PROMPT_PREFIX + "你好，欢迎来到离线语音测试。"
        encoded = list(llm.tokenizer_encode(llm.apply_chat_template(text) or text))
        encoded.extend(x + SPEECH_OFFSET for x in prompt)
        tokens = list(llm._c_obj.generate(encoded, 161))
        speech = [x - SPEECH_OFFSET for x in tokens if SPEECH_OFFSET <= x <= SPEECH_MAX]
        invalid = [x for x in tokens if not SPEECH_OFFSET <= x <= SPEECH_MAX and x != SPEECH_EOS]
        report.update(input_tokens=len(encoded), speech_tokens=len(speech), invalid_tokens=len(invalid),
                      eos=SPEECH_EOS in tokens or int(llm.context.status) == 1,
                      final_status=int(llm.context.status),
                      scope="Windows CPU; not iPhone generation or streaming")
        assert speech and len(speech) <= 160 and not invalid and report["eos"], report
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bad", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output, args.bad)
        return 0
    model = ROOT / "models" / "CosyVoice3"
    output = ROOT / "builds" / "llm-path-v041"
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads((model / "config-cpu-cosyvoice-ras.json").read_text())
    base.update(base_dir=model.as_posix() + "/", backend_type="cpu", thread_num=4,
                precision="low", memory="low", sampler_type="mixed",
                mixed_samplers=["penalty", "topK", "topP", "temperature"],
                repetition_penalty=1.15, max_new_tokens=161, **{"async": False})
    base.update(FIELDS)
    assert all((model / value).is_file() and (model / value).stat().st_size > 0 for value in FIELDS.values())
    for bad in (True, False):
        name = "bad-absolute" if bad else "good-relative"
        config = base.copy()
        if bad:
            for key in ("llm_model", "llm_weight", "tokenizer_file"):
                config[key] = (model / FIELDS[key]).as_posix()
        resolved = {key: config["base_dir"] + config[key] for key in FIELDS}
        if bad:
            assert not Path(resolved["tokenizer_file"]).exists()
        else:
            assert all(Path(path).is_file() for path in resolved.values())
        config_path = output / (name + ".json")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path),
               "--output", str(output / (name + "-report.json"))]
        if bad:
            cmd.append("--bad")
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (output / (name + ".log")).write_bytes(completed.stdout)
        if completed.returncode:
            print(completed.stdout.decode("utf-8", errors="replace"))
            raise RuntimeError(f"{name} exited {completed.returncode}")
        report = json.loads((output / (name + "-report.json")).read_text())
        if bad:
            assert b"tokenizer file not found" in completed.stdout, completed.stdout[-2000:]
        print(name + ": " + json.dumps(report))
    print("PASS: doubled-path failure reproduced; relative paths load and generate speech tokens with EOS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
