"""Isolated projection W8 sensitivity: frozen 20 calibration + 5 held-out texts.

All recipes retain existing FFN W8/block64 and add named attention projections.
Min/max weight-only encoding is NOT activation-calibration fitting. Windows
CPU timings, allocator estimates and audio proxies are not phone RTF or MOS.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from flow_quant_common import digest
from repack_flow_weights import repack

RECIPES=['ffn-q-w8','ffn-k-w8','ffn-v-w8','ffn-out-w8','ffn-qkv-w8','ffn-qkvo-w8']
BASE_HASHES={'graph':'ac9e126e8e2847db1421af956499fb0c99babb8924336924070ac996c34bfc35',
             'weight':'f5d713123341572a85fd3639c02a852e971916f18c80b181861f20cda4f662fd'}
GATES={'max_teacher_velocity_l2':.03,'max_closed_loop_mel_l2':.03,'mean_closed_loop_mel_l2':.015}

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def verify_corpus(path,assets):
    import numpy as np
    d=json.loads((path/'calibration-report.json').read_text(encoding='utf-8'))
    if d.get('state')!='completed' or not d.get('verification',{}).get('real_two_step_trajectories_exact'):
        raise ValueError('Completed verified real-trajectory corpus required')
    if Counter(c['split'] for c in d['cases'])!={'calibration':20,'validation':5}:raise ValueError('Expected frozen 20+5 split')
    if len({c['id'] for c in d['cases']})!=25 or len({c['text'] for c in d['cases']})!=25:raise ValueError('Duplicate corpus text/ID')
    for name in ['flow.cfg-student-2step.batch1.fp16.mnn','flow.cfg-student-2step.batch1.fp16.mnn.weight']:
        if digest(assets/name)!=d['model_files'][name]['sha256']:raise ValueError('Source model changed')
    for c in d['cases']:
        if not c['llm']['eos'] or c['llm']['invalid_tokens']:raise ValueError('Incomplete speech token reference')
        if digest(Path(c['audio']))!=c['audio_sha256']:raise ValueError('Baseline audio hash mismatch')
        arrays=[]
        for snap in c['snapshots']:
            p=Path(snap['directory'])/'tensors.npz'
            if digest(p)!=snap['tensor_archive_sha256']:raise ValueError('Fixture archive hash mismatch')
            with np.load(p) as a:arrays.append({k:a[k].copy() for k in a.files})
        if len(arrays)!=2 or not all(np.isfinite(v).all() for a in arrays for v in a.values()):raise ValueError('Invalid trajectory')
        x=arrays[0]['x']+np.float32(c['snapshots'][0]['delta'])*arrays[0]['estimator_out']
        if not np.array_equal(x,arrays[1]['x']):raise ValueError('Broken real step2 input')
        final=x+np.float32(c['snapshots'][1]['delta'])*arrays[1]['estimator_out']
        expected=np.load(Path(c['audio']).parent/'target-mel.npy')
        if not np.array_equal(final[0,:,174:c['effective_frames']],expected):raise ValueError('Broken reference Mel')
    return d

def run_child(command,log,timeout=900):
    with log.open('xb') as stream:
        try:return subprocess.run([sys.executable,*command],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:return 'timeout'

def sensitivity(args,report):
    root=args.output
    corpus=verify_corpus(args.corpus,args.assets)
    report['corpus']={'sha256':digest(args.corpus/'calibration-report.json'),'calibration':20,'validation':5,
        'snapshots':50,'verified_now':True,'note':'One prompt voice; screening, not full perceptual coverage'}
    report['gates']=GATES
    report['profiles']={}
    for mode in ['ffn-w8',*args.recipes]:
        if mode=='ffn-w8' and args.control_evaluations:
            previous=json.loads((args.control_evaluations/'projection-report.json').read_text(encoding='utf-8'))
            if previous.get('state')!='completed' or previous.get('corpus',{}).get('sha256')!=report['corpus']['sha256']:
                raise ValueError('Only completed, identical-corpus FFN numeric control may be reused')
            profile=previous['profiles']['ffn-w8']
            graph=Path(profile['model']);weight=Path(str(graph)+'.weight')
            if {'graph':digest(graph),'weight':digest(weight)}!=BASE_HASHES:raise ValueError('Reused control changed')
            if any(profile['evaluations'][m]['state']!='completed' for m in ['normal','low']):raise ValueError('Incomplete reused control')
            profile['numeric_control_reused_from']=str(args.control_evaluations)
            report['profiles'][mode]=profile;write(root/'projection-report.json',report)
            print('REUSE BYTE-IDENTICAL NUMERIC CONTROL; timing will be remeasured',flush=True)
            continue
        destination=root/mode
        print('REPACK',mode,flush=True)
        recipe=repack(args.assets,destination,mode,64)
        if mode=='ffn-w8' and {'graph':recipe['graph_sha256'],'weight':recipe['weight_sha256']}!=BASE_HASHES:
            raise ValueError('Control repack is not byte-identical to deployed FFN W8')
        profile={'mode':mode,'model':str(destination/'flow.mnn'),'recipe':recipe,'evaluations':{}}
        report['profiles'][mode]=profile;write(root/'projection-report.json',report)
        for memory,split in [('normal','validation'),('low','all')]:
            dest=root/(mode+'-'+memory)
            print('REPLAY',mode,memory,split,flush=True)
            code=run_child([str(Path(__file__).with_name('evaluate_flow_quant.py')),'--model',str(destination/'flow.mnn'),
                '--corpus',str(args.corpus),'--output',str(dest),'--split',split,'--memory',memory],root/(mode+'-'+memory+'.log'))
            result=json.loads((dest/'report.json').read_text(encoding='utf-8')) if (dest/'report.json').exists() else {}
            summaries={}
            for part in ['calibration','validation']:
                rows=[r for r in result.get('cases',[]) if r['split']==part]
                if not rows:continue
                values={'cases':len(rows),'max_teacher_velocity_l2':max(v['relative_l2'] for r in rows for v in r['teacher_velocity']),
                    'max_closed_loop_mel_l2':max(r['closed_loop_mel']['relative_l2'] for r in rows),
                    'mean_closed_loop_mel_l2':statistics.mean(r['closed_loop_mel']['relative_l2'] for r in rows)}
                expected=20 if part=='calibration' else 5
                values['numeric_screen_pass']=code==0 and result.get('state')=='completed' and len(rows)==expected and all(values[k]<=limit for k,limit in GATES.items())
                summaries[part]=values
            profile['evaluations'][memory]={'exit':code,'state':result.get('state','failed-no-report'),'error':result.get('error'),
                'summary':summaries,'report':str(dest/'report.json')}
            write(root/'projection-report.json',report)
            print(mode,memory,summaries,flush=True)
    report['sensitivity_state']='completed';write(root/'projection-report.json',report)
    return corpus

def timing(args,report,corpus):
    import numpy as np
    from pc_flow_hift_probe import model_session,put,get
    profiles=list(report['profiles'])
    cases=[c for c in corpus['cases'] if c['split']=='validation']
    report['timing']={'state':'running','warm_repeats':2,'memory':'low','precision':'high','threads':4,
        'method':'Recipe order rotated by held-out case, one fresh session at a time, first + two warm closed-loop runs; no teacher forward between timed runs',
        'limitations':['Windows CPU only; this machine lacks ARM int8 dot instructions','Single sweep, no confidence intervals','Session allocator is not process footprint','Not phone/whole-chain RTF'],'rows':[]}
    for i,case in enumerate(cases):
        with np.load(Path(case['snapshots'][0]['directory'])/'tensors.npz') as archive:
            initial={k:archive[k].copy() for k in archive.files if k!='estimator_out'}
        reference=np.load(Path(case['audio']).parent/'target-mel.npy')
        at=i%len(profiles)
        for mode in profiles[at:]+profiles[:at]:
            print('MATCHED TIMING',case['id'],mode,flush=True)
            start=time.perf_counter();net,session=model_session(Path(report['profiles'][mode]['model']),memory='low')
            for key,value in initial.items():net.resizeTensor(net.getSessionInput(session,key),value.shape)
            net.resizeSession(session);setup=time.perf_counter()-start
            values=[];first=None;repeat_error=0
            for trial in range(3):
                x=initial['x'].copy();start=time.perf_counter()
                for step in case['snapshots']:
                    for key,value in initial.items():
                        put(net,session,key,x if key=='x' else np.asarray([step['time']],dtype=np.float32) if key=='t' else value)
                    if net.runSession(session)!=0:raise RuntimeError('Timed MNN forward failed')
                    velocity=get(net,session,'estimator_out').reshape(x.shape)
                    if not np.isfinite(velocity).all():raise RuntimeError('Nonfinite timed forward')
                    x=x+np.float32(step['delta'])*velocity
                values.append(time.perf_counter()-start)
                mel=x[0,:,174:case['effective_frames']]
                if first is None:first=mel.copy()
                repeat_error=max(repeat_error,float(np.linalg.norm((first-mel).astype(np.float64))/max(1e-30,np.linalg.norm(first.astype(np.float64)))))
            report['timing']['rows'].append({'case':case['id'],'mode':mode,'frames':case['runtime_frames'],
                'audio_seconds':case['audio_seconds'],'setup_seconds':setup,'first_seconds':values[0],
                'warm_seconds':values[1:],'warm_median_seconds':statistics.median(values[1:]),
                'allocator_mb':net.getSessionInfo(session,0),'repeat_mel_l2':repeat_error,
                'mel_l2':float(np.linalg.norm((mel-reference).astype(np.float64))/np.linalg.norm(reference.astype(np.float64)))})
            del net,session;gc.collect();write(args.output/'projection-report.json',report)
    totals={}
    for mode in profiles:
        rows=[r for r in report['timing']['rows'] if r['mode']==mode]
        totals[mode]={'cases':len(rows),'warm_total_seconds':sum(r['warm_median_seconds'] for r in rows),
            'audio_seconds':sum(r['audio_seconds'] for r in rows),'median_allocator_mb':statistics.median(r['allocator_mb'] for r in rows),
            'max_repeat_mel_l2':max(r['repeat_mel_l2'] for r in rows)}
    for mode,summary in totals.items():summary['descriptive_speedup_vs_ffn']=totals['ffn-w8']['warm_total_seconds']/summary['warm_total_seconds']
    report['timing']['summary']=totals;report['timing']['state']='completed';write(args.output/'projection-report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-timing',action='store_true')
    from repack_flow_weights import GROUPS
    p.add_argument('--recipes',nargs='+',choices=[m for m in GROUPS if m!='ffn-w8'],default=RECIPES)
    p.add_argument('--control-evaluations',type=Path,help='Reuse completed identical-corpus FFN numeric replay only; still retime control')
    args=p.parse_args()
    args.assets=args.assets.resolve();args.corpus=args.corpus.resolve();args.output=args.output.resolve()
    if args.control_evaluations:args.control_evaluations=args.control_evaluations.resolve()
    if len(args.recipes)!=len(set(args.recipes)):p.error('Duplicate recipe')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'state':'running','scope':'Windows MNN CPU projection W8 sensitivity; not phone RTF/streaming/listening acceptance',
        'started_epoch':time.time(),'encoding':'Weight min/max W8/block64, no activation calibration fitting',
        'automatic_promotion':False,'original_models_modified':False,'script_sha256':digest(Path(__file__))}
    try:
        corpus=sensitivity(args,report)
        if not args.skip_timing:timing(args,report,corpus)
        report['state']='completed'
    except Exception as error:
        report['state']='failed';report['error']=str(error);raise
    finally:
        report['finished_epoch']=time.time();write(args.output/'projection-report.json',report)

if __name__=='__main__':main()
