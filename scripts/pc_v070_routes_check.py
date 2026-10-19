"""Controlled v0.7.0 input/reset regression; Windows timings are NOT iPhone RTF.

Uses the same real 70-token prefix as the native bucket matrix. No BF16/ARM
claim is possible on this Windows CPU. Generates only ignored build artifacts.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
import MNN.llm as mllm
import numpy as np
from pc_flow_hift_probe import conditioner, csv_tokens, flow
from pc_llm_probe import PROMPT_PREFIX, SPEECH_OFFSET, SPEECH_MAX, SPEECH_EOS

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    model = ROOT / "models" / "CosyVoice3"
    output = ROOT / "builds" / "v070-pc-routes"
    output.mkdir(parents=True, exist_ok=True)
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    target = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")[:70]
    mu, _ = conditioner(model, prompt, target)
    assert mu.shape == (80, 314)
    buckets = []
    reference = None
    for size in (384, 352, 320):
        mel, seconds = flow(model, mu, runtime_length=size, memory="low", precision="high")
        assert mel.shape == (80, 140) and np.isfinite(mel).all()
        if reference is None:
            reference = mel.copy()
        l2 = float(np.linalg.norm(mel-reference)/np.linalg.norm(reference))
        assert l2 < 1e-3, (size, l2)
        buckets.append(dict(frames=size, mel_l2=l2, windows_forward_seconds=seconds))
        print(json.dumps(buckets[-1]), flush=True)
    config = json.loads((model / "config-cpu-cosyvoice-ras.json").read_text())
    config.update(base_dir=model.as_posix()+"/", backend_type="cpu", precision="low", memory="low",
                  thread_num=4, sampler_type="mixed", mixed_samplers=["penalty", "greedy"],
                  repetition_penalty=1.15, max_new_tokens=106)
    path = output / "llm.json"
    path.write_text(json.dumps(config))
    reference = None
    requests = []
    for reuse in (False, True):
        llm = None
        for run in range(3):
            if llm is None:
                llm = mllm.create(str(path)); llm.load()
            if reuse:
                llm.reset()
            content = PROMPT_PREFIX + "今天天气很好。"
            formatted = llm.apply_chat_template(content) or content
            inputs = list(llm.tokenizer_encode(formatted)) + [t+SPEECH_OFFSET for t in prompt]
            raw = list(llm._c_obj.generate(inputs,106))
            eos = SPEECH_EOS in raw or int(llm.context.status) == 1
            speech = [t-SPEECH_OFFSET for t in raw if SPEECH_OFFSET <= t <= SPEECH_MAX]
            illegal = [t for t in raw if not SPEECH_OFFSET <= t <= SPEECH_MAX and t != SPEECH_EOS]
            assert eos and speech and len(speech) <= 105 and not illegal, (eos,len(speech),raw[-5:])
            if reference is None:
                reference = speech
            assert speech == reference, "Reset/thread isolation did not preserve deterministic output"
            requests.append(dict(reuse=reuse,run=run+1,speech_tokens=len(speech),eos=eos,exact_match=True))
            print(json.dumps(requests[-1]), flush=True)
            if not reuse:
                del llm; llm = None; gc.collect()
        del llm
        gc.collect()
    report = dict(scope="Windows numerical/reset regression ONLY; BF16, memory, speed and iPhone playback untested",
                  buckets=buckets,requests=requests,passed=True)
    (output / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
