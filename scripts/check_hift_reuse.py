"""PC-only fresh/reused HiFT Low numerical replay, changing/repeated lengths."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
from pc_flow_hift_probe import model_session,put,get,excitation,stft,istft,finite
from flow_quant_common import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--mels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=['zh-question','zh-numbers','zh-numbers','en-heldout','mixed-heldout','zh-question']
    weights=np.fromfile(a.assets/'source-linear-weight.bin',dtype='<f4');bias=np.fromfile(a.assets/'source-linear-bias.bin',dtype='<f4')[0]
    def decode(mel,cache):
        frames=mel.shape[1];shaped=mel.reshape(1,80,frames)
        def graph(name,precision):
            if name not in cache:cache[name]=[*model_session(a.assets/name,precision,'low'),0]
            return cache[name]
        f=graph('hift-f0.fp32.mnn','high');n,s,old=f
        if old!=frames:n.resizeTensor(n.getSessionInput(s,'mel'),shaped.shape);n.resizeSession(s);f[2]=frames
        put(n,s,'mel',shaped);assert n.runSession(s)==0
        f0=get(n,s,'f0');finite('f0',f0)
        source=stft(excitation(f0,weights,float(bias))).reshape(1,18,frames*120+1)
        c=graph('hift-core.fp32.mnn','low');n,s,old=c
        if old!=frames:
            for name,value in [('mel',shaped),('source_stft',source)]:n.resizeTensor(n.getSessionInput(s,name),value.shape)
            n.resizeSession(s);c[2]=frames
        put(n,s,'mel',shaped);put(n,s,'source_stft',source);assert n.runSession(s)==0
        spectral=get(n,s,'spectral').reshape(18,frames*120+1);finite('spectral',spectral)
        pcm=istft(spectral)[:frames*480];finite('pcm',pcm)
        return f0,spectral,pcm
    report={'state':'running','scope':'Windows CPU; not iPhone speed/quality or whole-chain acceptance','rows':[],'models':{n:digest(a.assets/n) for n in ['hift-f0.fp32.mnn','hift-core.fp32.mnn']}}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    cache={};save()
    try:
        for case in cases:
            print('HiFT reuse',case,flush=True);mel=np.load(a.mels/(case+'-mel.npy'))
            fresh={};reference=decode(mel,fresh);fresh.clear();gc.collect()
            actual=decode(mel,cache)
            errors=[float(np.linalg.norm((v-r).astype(np.float64))/max(1e-30,np.linalg.norm(r.astype(np.float64)))) for v,r in zip(actual,reference)]
            report['rows'].append({'case':case,'frames':mel.shape[1],'relative_l2_f0_spectral_pcm':errors,'exact':[bool(np.array_equal(v,r)) for v,r in zip(actual,reference)]});save()
            if max(errors)>1e-5:raise ValueError('Reuse mismatch')
        report['state']='completed';report['sessions_retained']=len(cache);cache.clear();save()
    except Exception as e:report['state']='failed';report['error']=str(e);save();raise
if __name__=='__main__':main()
