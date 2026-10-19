"""v0.7.3 lifecycle numerics. Windows CPU only; NOT phone RAM/RTF validation.

Controlled 384-frame inputs, penalty+greedy sampling, exact speech/EOS checks.
Fresh baseline vs retained LLM through acoustics vs two-request stage batching.
All routes use the same two-step Flow and selected HiFT Low.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
import MNN.llm as mllm
import numpy as np
from cloud_flow_quant_eval import write_wav
from pc_flow_hift_probe import conditioner, csv_tokens, finite, get, hift, model_session, put
from pc_llm_probe import PROMPT_PREFIX, SPEECH_OFFSET, SPEECH_MAX, SPEECH_EOS

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "CosyVoice3"
OUT = ROOT / "builds" / "v073-pc-pipeline"
TEXTS = ["你好。", "今天天气很好。", "欢迎再来。"]

class FlowSession:
    def __init__(self):
        self.net, self.session = model_session(MODEL / "flow.cfg-student-2step.batch1.fp16.mnn", "high", "low")
        for name, shape in {"x": (1,80,384), "mu": (1,80,384), "cond": (1,80,384),
                            "mask": (1,1,384), "spks": (1,80), "t": (1,)}.items():
            self.net.resizeTensor(self.net.getSessionInput(self.session,name),shape)
        self.net.resizeSession(self.session)
    def run(self, mu):
        effective = mu.shape[1]
        assert 174 < effective <= 384
        noise = np.fromfile(MODEL / "rand-noise.bin",dtype="<f4").reshape(80,15000)
        prompt = np.fromfile(MODEL / "prompt-cond.bin",dtype="<f4").reshape(80,174)
        spks = np.fromfile(MODEL / "spks.bin",dtype="<f4").reshape(1,80)
        x = np.zeros((1,80,384),np.float32); padded=np.zeros_like(x); cond=np.zeros_like(x)
        mask=np.zeros((1,1,384),np.float32)
        x[0,:,:effective]=noise[:,:effective];padded[0,:,:effective]=mu;cond[0,:,:174]=prompt;mask[:,:,:effective]=1
        mid=np.float32(1-np.cos(np.pi/4))
        for t,delta in ((np.float32(0),mid),(mid,np.float32(1)-mid)):
            for name,value in (("x",x),("mu",padded),("cond",cond),("mask",mask),("spks",spks),("t",np.array([t],np.float32))):
                put(self.net,self.session,name,value)
            self.net.runSession(self.session)
            v=get(self.net,self.session,"estimator_out").reshape(x.shape);finite("Flow",v);x+=delta*v
        return x[0,:,174:effective].copy()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((MODEL / "config-cpu-cosyvoice-ras.json").read_text())
    config.update(base_dir=MODEL.as_posix()+"/",backend_type="cpu",precision="low",memory="low",thread_num=4,
                  sampler_type="mixed",mixed_samplers=["penalty","greedy"],repetition_penalty=1.15,max_new_tokens=161)
    path=OUT / "llm.json";path.write_text(json.dumps(config))
    prompt=csv_tokens(MODEL / "prompt-speech-tokens.csv")
    def load():
        llm=mllm.create(str(path));llm.load();return llm
    def raw(llm,text,limit):
        llm.reset()
        content=PROMPT_PREFIX+text;formatted=llm.apply_chat_template(content) or content
        inputs=list(llm.tokenizer_encode(formatted))+[t+SPEECH_OFFSET for t in prompt]
        ids=list(llm._c_obj.generate(inputs,limit))
        eos=SPEECH_EOS in ids or int(llm.context.status)==1
        return ids,eos
    def tokens(llm,text):
        ids,eos=raw(llm,text,106)
        speech=[t-SPEECH_OFFSET for t in ids if SPEECH_OFFSET<=t<=SPEECH_MAX]
        illegal=[t for t in ids if not SPEECH_OFFSET<=t<=SPEECH_MAX and t!=SPEECH_EOS]
        assert eos and 0<len(speech)<=105 and not illegal,(text,eos,len(speech))
        return speech
    # Actual MNN early-token-limit destruction, then subsequent clean requests.
    for _ in range(2):
        llm=load();ids,eos=raw(llm,"你好，欢迎来到离线语音测试。我们正在测试按句生成和播放。",2)
        assert not eos and len(ids)==2
        del llm;gc.collect()
    references=[]; rows=[]
    def acoustic(route,index,speech,f):
        mu,_=conditioner(MODEL,prompt,speech);mel=f.run(mu)
        pcm,_=hift(MODEL,mel,core_precision="low",core_memory="low")
        assert np.isfinite(pcm).all() and len(pcm)==len(speech)*960
        rms=float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        assert rms>.001 and np.abs(pcm).max()<=1.001
        if route=="fresh": references.append((speech,mel.copy(),pcm.copy()))
        refspeech,refmel,refpcm=references[index]
        assert speech==refspeech
        ml=float(np.linalg.norm(mel-refmel)/np.linalg.norm(refmel))
        pl=float(np.linalg.norm(pcm-refpcm)/np.linalg.norm(refpcm))
        assert ml<1e-3 and pl<1e-3,(route,index,ml,pl)
        write_wav(OUT / f"{route}-{index+1}.wav",pcm)
        row=dict(route=route,index=index+1,speech_tokens=len(speech),eos=True,mel_l2=ml,pcm_l2=pl,audio_seconds=len(pcm)/24000,rms=rms)
        rows.append(row);print(json.dumps(row),flush=True)
    for index,text in enumerate(TEXTS):
        llm=load();speech=tokens(llm,text);del llm;gc.collect()
        f=FlowSession();acoustic("fresh",index,speech,f);del f;gc.collect()
    llm=load()
    for index,text in enumerate(TEXTS):
        speech=tokens(llm,text)
        f=FlowSession();acoustic("llm-retained",index,speech,f);del f;gc.collect()
    del llm;gc.collect()
    for start in range(0,len(TEXTS),2):
        llm=load();batch=[tokens(llm,t) for t in TEXTS[start:start+2]]
        del llm;gc.collect()
        f=FlowSession()
        for offset,speech in enumerate(batch):acoustic("two-staged",start+offset,speech,f)
        del f;gc.collect()
    report=dict(passed=True,scope="Windows MNN lifecycle/numerics only; not native iPhone ARC/RAM/RTF or listening approval",
                forced_no_eos_destructions=2,rows=rows,flow_frames=384,steps=2,hift="low",sampling="penalty+greedy")
    (OUT / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS pipeline numerical regression",flush=True)

if __name__=="__main__":main()
