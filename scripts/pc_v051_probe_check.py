"""Validate v0.5.1's fixed-input comparisons and resident-call order on PC.

Windows CPU Low is a fallback here: this does NOT certify ARM FP16 or iPhone
residency memory. No new weights or modified model graphs are produced.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
import MNN
import MNN.llm as mllm
import numpy as np
from pc_flow_hift_probe import (conditioner, csv_tokens, model_session, put, get,
                                finite, excitation, stft, istft)
from pc_llm_probe import PROMPT_PREFIX, SPEECH_OFFSET, SPEECH_MAX
from cloud_flow_quant_eval import write_wav

ROOT=Path(__file__).resolve().parents[1]

def main() -> int:
    model=ROOT/"models"/"CosyVoice3"
    output=ROOT/"builds"/"v051-probes"
    output.mkdir(parents=True,exist_ok=True)
    prompt=csv_tokens(model/"prompt-speech-tokens.csv")
    target=csv_tokens(ROOT/"testdata"/"pc-real-74-speech-tokens.csv")
    assert len(prompt)==87 and len(target)==74
    mu,_=conditioner(model,prompt,target)
    frames=384;effective=322
    assert mu.shape==(80,effective)
    noise=np.fromfile(model/"rand-noise.bin",dtype="<f4").reshape(80,15000)
    initial=np.zeros((1,80,frames),np.float32); initial[0,:,:effective]=noise[:,:effective]
    padded=np.zeros_like(initial);padded[0,:,:effective]=mu
    cond=np.zeros_like(initial);cond[0,:,:174]=np.fromfile(model/"prompt-cond.bin",dtype="<f4").reshape(80,174)
    mask=np.zeros((1,1,frames),np.float32);mask[:,:,:effective]=1
    spks=np.fromfile(model/"spks.bin",dtype="<f4").reshape(1,80)
    net,session=model_session(model/"flow.cfg-student-2step.batch1.fp16.mnn",precision="high",memory="low")
    for name,shape in {"x":initial.shape,"mu":padded.shape,"cond":cond.shape,"mask":mask.shape,"spks":spks.shape,"t":(1,)}.items():
        net.resizeTensor(net.getSessionInput(session,name),shape)
    net.resizeSession(session)  # Exactly one resize; same session all three phases.
    def run_flow():
        x=initial.copy(); mid=np.float32(1-np.cos(np.pi/4));started=time.perf_counter()
        for t,delta in ((np.float32(0),mid),(mid,np.float32(1)-mid)):
            for name,value in (("x",x),("mu",padded),("cond",cond),("mask",mask),("spks",spks),("t",np.array([t],np.float32))):
                put(net,session,name,value)
            net.runSession(session)
            velocity=get(net,session,"estimator_out").reshape(x.shape)
            finite("velocity",velocity);x+=delta*velocity
        finite("Flow latent",x)
        return x,time.perf_counter()-started
    reference,baseline_time=run_flow()
    config=json.loads((model/"config-cpu-cosyvoice-ras.json").read_text())
    config.update(base_dir=model.as_posix()+"/",sampler_type="mixed",mixed_samplers=["penalty","topK","topP","temperature"],repetition_penalty=1.15)
    config_path=output/"llm.json";config_path.write_text(json.dumps(config))
    llm=mllm.create(str(config_path));llm.load();assert int(llm.context.status)==0
    text=PROMPT_PREFIX+"你好，欢迎来到离线语音测试。"
    encoded=list(llm.tokenizer_encode(llm.apply_chat_template(text) or text))+[x+SPEECH_OFFSET for x in prompt]
    tokens=list(llm._c_obj.generate(encoded,2))
    assert tokens and all(SPEECH_OFFSET<=x<=SPEECH_MAX for x in tokens),tokens
    resident,resident_time=run_flow()  # llm variable remains strongly alive.
    assert np.linalg.norm(resident-reference)/np.linalg.norm(reference)<1e-3
    del llm;gc.collect()
    reused,reused_time=run_flow()
    relative=float(np.linalg.norm(reused-reference)/np.linalg.norm(reference));assert relative<1e-3
    mel=reference[0,:,174:effective].copy();del net,session;gc.collect()
    f0net,f0session=model_session(model/"hift-f0.fp32.mnn",precision="high",memory="low")
    shaped=mel.reshape(1,80,148)
    f0net.resizeTensor(f0net.getSessionInput(f0session,"mel"),shaped.shape);f0net.resizeSession(f0session)
    put(f0net,f0session,"mel",shaped);f0net.runSession(f0session);f0=get(f0net,f0session,"f0")
    del f0net,f0session;gc.collect()
    source=excitation(f0,np.fromfile(model/"source-linear-weight.bin",dtype="<f4"),float(np.fromfile(model/"source-linear-bias.bin",dtype="<f4")[0]))
    source_stft=stft(source).reshape(1,18,148*120+1)
    pcms={};times={}
    for precision in ("high","low"):
        core,core_session=model_session(model/"hift-core.fp32.mnn",precision=precision,memory="low")
        for name,shape in (("mel",shaped.shape),("source_stft",source_stft.shape)):
            core.resizeTensor(core.getSessionInput(core_session,name),shape)
        core.resizeSession(core_session);run_times=[]
        for _ in range(3):
            started=time.perf_counter();put(core,core_session,"mel",shaped);put(core,core_session,"source_stft",source_stft)
            core.runSession(core_session);spectrum=get(core,core_session,"spectral").reshape(18,-1)
            finite("HiFT spectrum",spectrum);run_times.append(time.perf_counter()-started)
        pcm=istft(spectrum);finite("PCM",pcm);assert len(pcm)==71040 and np.sqrt(np.mean(pcm**2))>0.001
        pcms[precision]=pcm;times[precision]=run_times
        write_wav(output/f"hift-{precision}.wav",pcm)
        del core,core_session;gc.collect()
    pcm_l2=float(np.linalg.norm(pcms["low"]-pcms["high"])/np.linalg.norm(pcms["high"]))
    assert pcm_l2<1e-3  # Windows reports fp16 unsupported; no ARM Low claim.
    report={"scope":"Windows call-order/numerical test ONLY; no iPhone memory or ARM Low validation",
            "resize_count":1,"short_llm_tokens":len(tokens),"flow_times":[baseline_time,resident_time,reused_time],
            "flow_reuse_relative_l2":relative,"hift_frames":148,"audio_seconds":2.96,"core_times":times,"pcm_relative_l2":pcm_l2}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
