"""Real-text, bounded whole-sentence regression for the v0.6.0 Flow lifecycle.

Windows CPU numerics/call order only; NOT iPhone residency, playback or RTF.
Each sentence gets a new LLM; ONE 384-frame Flow survives all HiFT calls.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import MNN.llm as mllm
import numpy as np

from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, finite, flow, get, hift, model_session, put
from pc_llm_probe import PROMPT_PREFIX, SPEECH_EOS, SPEECH_MAX, SPEECH_OFFSET

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 384
PROMPT_FRAMES = 174
CAP = (FRAMES - PROMPT_FRAMES) // 2


def accepted(raw: list[int], status: int) -> tuple[list[int], bool]:
    speech = [x - SPEECH_OFFSET for x in raw if SPEECH_OFFSET <= x <= SPEECH_MAX]
    invalid = [x for x in raw if not SPEECH_OFFSET <= x <= SPEECH_MAX and x != SPEECH_EOS]
    eos = SPEECH_EOS in raw or status == 1
    return speech, bool(speech) and not invalid and eos and len(speech) <= CAP


def main() -> int:
    started = time.perf_counter()
    model = ROOT / "models" / "CosyVoice3"
    output = ROOT / "builds" / "resident-sentences-v060"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config-cpu-cosyvoice-ras.json").read_text())
    config.update(base_dir=model.as_posix() + "/", backend_type="cpu", thread_num=4,
                  precision="low", memory="low", sampler_type="mixed",
                  mixed_samplers=["penalty", "topK", "topP", "temperature"],
                  repetition_penalty=1.15, max_new_tokens=CAP + 1)
    config_path = output / "llm.json"
    config_path.write_text(json.dumps(config))
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    assert len(prompt) == 87 and CAP == 105
    # Fail closed for an over-limit EOS, missing EOS, empty or illegal output.
    assert not accepted([SPEECH_OFFSET] * 106 + [SPEECH_EOS], 1)[1]
    assert not accepted([SPEECH_OFFSET] * 106, 2)[1]
    assert not accepted([SPEECH_EOS], 1)[1]
    assert not accepted([123, SPEECH_EOS], 1)[1]
    assert accepted([SPEECH_OFFSET] * 105 + [SPEECH_EOS], 1)[1]
    net, session = model_session(model / "flow.cfg-student-2step.batch1.fp16.mnn", "high", "low")
    shapes = {"x": (1,80,FRAMES), "mu": (1,80,FRAMES), "cond": (1,80,FRAMES),
              "mask": (1,1,FRAMES), "spks": (1,80), "t": (1,)}
    for name, shape in shapes.items():
        net.resizeTensor(net.getSessionInput(session, name), shape)
    net.resizeSession(session)  # Exactly once, including changes in sentence length.
    noise = np.fromfile(model / "rand-noise.bin", dtype="<f4").reshape(80,15000)
    prompt_cond = np.fromfile(model / "prompt-cond.bin", dtype="<f4").reshape(80,174)
    spks = np.fromfile(model / "spks.bin", dtype="<f4").reshape(1,80)
    midpoint = np.float32(1-np.cos(np.pi/4))

    def reuse(mu: np.ndarray) -> np.ndarray:
        effective = mu.shape[1]
        assert PROMPT_FRAMES < effective <= FRAMES
        x = np.zeros((1,80,FRAMES), np.float32)
        padded = np.zeros_like(x)
        cond = np.zeros_like(x)
        mask = np.zeros((1,1,FRAMES), np.float32)
        x[0,:,:effective] = noise[:,:effective]
        padded[0,:,:effective] = mu
        cond[0,:,:174] = prompt_cond
        mask[:,:,:effective] = 1
        for t, delta in ((np.float32(0),midpoint),(midpoint,np.float32(1)-midpoint)):
            for name, value in (("x",x),("mu",padded),("cond",cond),("mask",mask),
                                ("spks",spks),("t",np.array([t],np.float32))):
                put(net,session,name,value)
            net.runSession(session)
            velocity = get(net,session,"estimator_out").reshape(x.shape)
            finite("Flow",velocity)
            x += delta*velocity
        return x[0,:,174:effective].copy()

    reports, pcms = [], []
    first_mu = first_mel = None
    for index, text in enumerate(("你好。", "今天天气很好。", "欢迎再来。")):
        begin = time.perf_counter()
        llm = mllm.create(str(config_path))
        llm.load()
        templated = llm.apply_chat_template(PROMPT_PREFIX + text) or PROMPT_PREFIX + text
        inputs = list(llm.tokenizer_encode(templated)) + [x+SPEECH_OFFSET for x in prompt]
        raw = list(llm._c_obj.generate(inputs,CAP+1))
        status = int(llm.context.status)
        target, ok = accepted(raw,status)
        assert ok, (text,len(target),status,raw[-4:])
        llm_seconds = time.perf_counter()-begin
        del llm
        gc.collect()  # No live LLM/KV when Conditioner/HiFT run.
        mu,_ = conditioner(model,prompt,target)
        mel = reuse(mu)
        if first_mu is None:
            first_mu,first_mel = mu.copy(),mel.copy()
        pcm,times = hift(model,mel)  # High F0/core; resident Flow remains alive.
        assert len(pcm) == len(target)*2*480 and np.isfinite(pcm).all()
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        assert rms > .001 and np.abs(pcm).max() <= 1.001
        write_wav(output / f"sentence-{index+1}.wav",pcm)
        pcms.append(pcm)
        reports.append(dict(text=text,speech_tokens=len(target),eos=True,llm_seconds=llm_seconds,
                            audio_seconds=len(pcm)/24000,rms=rms,hift=times))
        print(json.dumps(reports[-1],ensure_ascii=True),flush=True)
        gc.collect()
    repeated = reuse(first_mu)
    repeat_l2 = float(np.linalg.norm(repeated-first_mel)/np.linalg.norm(first_mel))
    assert repeat_l2 < 1e-3  # Different lengths/conditioning did not poison cached inputs.
    del net,session
    gc.collect()
    fresh,_ = flow(model,first_mu,runtime_length=FRAMES,memory="low",precision="high")
    fresh_l2 = float(np.linalg.norm(fresh-first_mel)/np.linalg.norm(first_mel))
    assert fresh_l2 < 1e-3
    write_wav(output / "three-sentences.wav",np.concatenate(pcms))
    report = dict(scope="Windows real-text/call-order/numerics ONLY; no iPhone memory/RTF/playback claim",
                  resident_flow_resize_count=1,flow_frames=FRAMES,steps=2,speech_cap=CAP,
                  repeat_l2=repeat_l2,fresh_l2=fresh_l2,sentences=reports,
                  wall_seconds=time.perf_counter()-started)
    (output / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
