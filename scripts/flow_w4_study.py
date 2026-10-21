"""RTN/HQQ W4 isolated sensitivity; verified frozen20+5 corpus, no training."""
import argparse,json,subprocess,sys,statistics,struct,wave
from pathlib import Path
from flow_projection_matrix import verify_corpus,GATES
from flow_quant_common import digest
from repack_flow_weights import repack,external_regions
from audit_mnn_weight_layout import FlatBuffer

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);corpus=verify_corpus(a.corpus,a.assets)
    report={'state':'running','scope':'Windows CPU High, not iPhone throughput/listening acceptance; no automatic deployment',
      'corpus':{'sha256':digest(a.corpus/'calibration-report.json'),'calibration':20,'validation':5},'gates':GATES,'profiles':{}}
    def save():(a.output/'w4-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save()
    for scope,mode in [('ffn','ffn-w8'),('projections','ffn-qvout-w8')]:
      for method in ['rtn','hqq']:
        key=scope+'-w4-'+method;print('REPACK',key,flush=True)
        folder=a.output/key;recipe=repack(a.assets,folder,mode,64,bits=4,algorithm=method,w4_scope=scope)
        entry={'recipe':recipe,'evaluations':{}};report['profiles'][key]=entry
        # Independent region/bias audit; no rewrite is allowed outside selection.
        source_graph=a.assets/'flow.cfg-student-2step.batch1.fp16.mnn';source_weight=Path(str(source_graph)+'.weight')
        old=external_regions(FlatBuffer(source_graph.read_bytes()),source_weight.stat().st_size)
        new=external_regions(FlatBuffer((folder/'flow.mnn').read_bytes()),(folder/'flow.mnn.weight').stat().st_size)
        formats={r['name']:r['format'] for r in recipe['operators']};preserved=0;biases=0
        assert [(r['kind'],r['name']) for r in old]==[(r['kind'],r['name']) for r in new]
        with source_weight.open('rb') as src,(folder/'flow.mnn.weight').open('rb') as dst:
          for x,y in zip(old,new):
            oe,ne=x['external'],y['external'];src.seek(oe[0]);dst.seek(ne[0]);v=src.read(sum(oe[1:]));w=dst.read(sum(ne[1:]))
            if formats.get(x['name'],'original-fp16')=='original-fp16':assert v==w;preserved+=1
            else:assert v[oe[1]:]==w[ne[1]+ne[2]:];biases+=1
        entry['integrity']={'preserved_regions_exact':preserved,'selected_biases_exact':biases,'passed':True};save()
        for memory,split in [('normal','validation'),('low','all')]:
          dest=a.output/(key+'-'+memory);log=a.output/(key+'-'+memory+'.log')
          command=[sys.executable,str(Path(__file__).with_name('evaluate_flow_quant.py')),'--model',str(folder/'flow.mnn'),'--corpus',str(a.corpus),'--output',str(dest),'--split',split,'--memory',memory]
          print('EVALUATE',key,memory,flush=True)
          with log.open('xb') as stream:
            try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:code='timeout'
          result=json.loads((dest/'report.json').read_text(encoding='utf-8')) if (dest/'report.json').exists() else {}
          summary={}
          for part,count in [('calibration',20),('validation',5)]:
            rows=[r for r in result.get('cases',[]) if r['split']==part]
            if not rows:continue
            s={'cases':len(rows),'max_teacher_velocity_l2':max(m['relative_l2'] for r in rows for m in r['teacher_velocity']),
               'max_closed_loop_mel_l2':max(r['closed_loop_mel']['relative_l2'] for r in rows),
               'mean_closed_loop_mel_l2':statistics.mean(r['closed_loop_mel']['relative_l2'] for r in rows)}
            s['screen_pass']=code==0 and result.get('state')=='completed' and len(rows)==count and all(s[k]<=v for k,v in GATES.items());summary[part]=s
          entry['evaluations'][memory]={'exit':code,'state':result.get('state','failed'),'summary':summary,'report':str(dest/'report.json'),'error':result.get('error')};save()
          print(key,memory,summary,flush=True)
        entry['listening']=[]
        # Save audible held-out Low-memory trajectories even if screen fails;
        # they stay explicitly experimental, never packaged for deployment.
        result=json.loads((a.output/(key+'-low')/'report.json').read_text(encoding='utf-8'))
        if result.get('state')=='completed':
          import numpy as np
          from pc_flow_hift_probe import hift
          listen=folder/'listening';listen.mkdir()
          for c in corpus['cases']:
            if c['split']!='validation':continue
            mel=np.load(a.output/(key+'-low')/(c['id']+'-mel.npy'))
            pcm,t=hift(a.assets,mel,core_precision='high',core_memory='normal')
            if not np.isfinite(pcm).all():raise ValueError('Nonfinite PCM')
            path=listen/(c['id']+'.wav')
            with wave.open(str(path),'wb') as stream:
              stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000);stream.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
            entry['listening'].append({'case':c['id'],'wav':str(path),'sha256':digest(path),'audio_seconds':len(pcm)/24000});save()
    report['state']='completed';save()
if __name__=='__main__':main()
