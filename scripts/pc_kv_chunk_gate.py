"""Gate incremental KV + offline-MNN prefix acoustics, NOT official streaming.

Greedy forward-loop replay checks KV survives pauses for Flow/HiFT Low.
The acoustic gate compares emitted prefix PCM with same-token final PCM;
failure deliberately prevents approving this offline graph for token streaming.
No weights are changed and no iPhone memory claim is made.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import MNN.llm as mllm
import numpy as np

from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, flow, hift
from pc_llm_probe import PROMPT_PREFIX, SPEECH_EOS, SPEECH_MAX, SPEECH_OFFSET

ROOT = Path(__file__).resolve().parents[1]
CAP = 105
LOOKAHEAD = 3
HOP = 25
HOLD_SAMPLES = 4800  # Conservative research holdback; not exported finalize semantics.


def main() -> int:
    output = ROOT / "builds" / "kv-chunk-gate-v061"
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / "models" / "CosyVoice3"
    config = json.loads((model / "config-cpu-cosyvoice-ras.json").read_text())
    config.update(base_dir=model.as_posix()+"/", backend_type="cpu", thread_num=4,
                  precision="low", memory="low", sampler_type="greedy", max_new_tokens=CAP+1)
    path = output / "llm.json"
    path.write_text(json.dumps(config))
    llm = mllm.create(str(path)); llm.load()
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    assert len(prompt) == 87
    text = "今天天气很好。"
    formatted = llm.apply_chat_template(PROMPT_PREFIX+text) or PROMPT_PREFIX+text
    inputs = list(llm.tokenizer_encode(formatted)) + [x+SPEECH_OFFSET for x in prompt]
    threshold = HOP + (-len(prompt)) % HOP + LOOKAHEAD  # 41, not 28.
    assert threshold == 41

    def generate(on_prefix=None):
        llm.reset(); llm.generate_init()
        raw = []
        logits = np.array(llm.forward(inputs).read(), copy=True).reshape(-1)
        prefills = 1
        for _ in range(CAP+1):
            # A declared deterministic research sampler, not production mixed RAS.
            scores = logits.copy()
            if raw:
                used = np.unique(raw)
                scores[used] = np.where(scores[used] < 0, scores[used]*1.15, scores[used]/1.15)
            token = int(np.argmax(scores))
            if token == SPEECH_EOS:
                assert raw
                return raw, prefills
            assert SPEECH_OFFSET <= token <= SPEECH_MAX, token
            raw.append(token)
            assert len(raw) <= CAP, "No EOS in 105 speech tokens; refuse truncation"
            if on_prefix and len(raw) >= threshold and (len(raw)-threshold) % HOP == 0:
                on_prefix([x-SPEECH_OFFSET for x in raw], int(llm.context.all_seq_len))
            logits = np.array(llm.forward([token]).read(), copy=True).reshape(-1)
        raise AssertionError("Missing EOS")

    reference, ref_prefills = generate()
    print(json.dumps(dict(stage="reference",speech=len(reference),prefills=ref_prefills)),flush=True)
    events, emitted, stable_mels = [], [], []
    emitted_samples = 0
    started = time.perf_counter()

    def acoustic(prefix, kv_before):
        nonlocal emitted_samples
        t = time.perf_counter()
        mu,_ = conditioner(model,prompt,prefix)
        mu = np.ascontiguousarray(mu[:,:-LOOKAHEAD*2])
        runtime = 256 if mu.shape[1] <= 256 else 384
        mel,_ = flow(model,mu,runtime_length=runtime,memory="low",precision="high")
        pcm,_ = hift(model,mel,core_precision="low",core_memory="low")
        end = max(0,len(pcm)-HOLD_SAMPLES)
        assert emitted_samples < end
        block = pcm[emitted_samples:end].copy()
        assert np.isfinite(block).all() and 0 < len(block) <= 4*24000
        emitted.append(block)
        stable_mels.append((mel.copy(),end))
        event = dict(speech_tokens_ready=len(prefix),kv_before=kv_before,
                     kv_after=int(llm.context.all_seq_len),first_pcm_before_eos=True,
                     sample_begin=emitted_samples,sample_end=end,chunk_seconds=len(block)/24000,
                     flow_frames=runtime,compute_seconds=time.perf_counter()-t,
                     wall_seconds=time.perf_counter()-started)
        assert event["kv_after"] == kv_before  # Acoustics did not mutate LLM KV.
        emitted_samples = end
        events.append(event)
        print(json.dumps(event),flush=True)

    replay, replay_prefills = generate(acoustic)
    eos_time = time.perf_counter()-started
    assert replay == reference and replay_prefills == 1
    target = [x-SPEECH_OFFSET for x in replay]
    mu,_ = conditioner(model,prompt,target)
    mel,_ = flow(model,mu,runtime_length=384,memory="low",precision="high")
    pcm,_ = hift(model,mel,core_precision="low",core_memory="low")
    assert len(pcm) == len(target)*960
    # Final flush: exact remaining sample range, including sub-hop tail.
    emitted.append(pcm[emitted_samples:].copy())
    joined = np.concatenate(emitted) if events else pcm.copy()
    assert len(joined) == len(pcm) and np.isfinite(joined).all()
    reference_rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
    pcm_l2 = float(np.linalg.norm(joined-pcm)/max(np.linalg.norm(pcm),1e-12))
    mel_errors = [float(np.linalg.norm(m-mel[:,:m.shape[1]])/max(np.linalg.norm(mel[:,:m.shape[1]]),1e-12)) for m,_ in stable_mels]
    # Strict equivalence gate: do not silently bless a noncausal offline export.
    acoustic_pass = bool(events) and pcm_l2 <= 1e-3 and max(mel_errors,default=1) <= 1e-3
    assert reference_rms > .001
    write_wav(output / "offline-prefix-experiment.wav",joined)
    write_wav(output / "same-token-full-context.wav",pcm)
    report = dict(scope="Windows greedy KV research; NOT production mixed sampler, iPhone memory or official causal streaming",
                  text=text,kv_replay_exact=replay==reference,prefills_per_utterance=1,
                  token_cap=CAP,token_count=len(target),first_chunk_threshold=threshold,
                  events=events,eos_wall_seconds=eos_time,first_pcm_before_eos=bool(events),
                  audio_seconds=len(pcm)/24000,tail_samples=len(pcm)-emitted_samples,
                  pcm_relative_l2=pcm_l2,prefix_mel_relative_l2=mel_errors,
                  acoustic_gate_passed=acoustic_pass,approved_for_phone_token_stream=False)
    (output / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return 0  # A failed acoustic gate is a measured finding, not hidden as success.


if __name__ == "__main__":
    raise SystemExit(main())
