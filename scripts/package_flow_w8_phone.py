"""Package isolated FFN-W8 weights and all five frozen held-out Flow fixtures.

Never edits original model files. Generated model/fixture bytes stay in builds/;
only the small identity manifest is bundled in the app.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def package(project: Path, corpus: Path, quant: Path, output: Path):
    import numpy as np
    reference=json.loads((corpus/'calibration-report.json').read_text(encoding='utf-8'))
    if reference.get('state')!='completed' or not reference.get('verification',{}).get('real_two_step_trajectories_exact'):
        raise ValueError('Verified real-trajectory corpus required')
    recipe=json.loads((quant/'ffn-w8-g64/repack-report.json').read_text(encoding='utf-8'))
    for memory in ('normal','low'):
        result=json.loads((quant/f'validation-g64-{memory}/report.json').read_text(encoding='utf-8'))
        if result.get('state')!='completed' or len(result['cases'])!=5 or not result.get('binding_memory_enum_verified'):
            raise ValueError('Complete corrected-enum held-out validation required')
    source=corpus.parents[1]/'models/CosyVoice3'
    files={}
    for key,name in (('sourceGraph','flow.cfg-student-2step.batch1.fp16.mnn'),('sourceWeight','flow.cfg-student-2step.batch1.fp16.mnn.weight')):
        path=source/name; sha=digest(path)
        if sha!=reference['model_files'][name]['sha256']:raise ValueError('Source identity changed')
        files[key]={'file':name,'bytes':path.stat().st_size,'sha256':sha}
    if recipe['source_hashes']!={'graph':files['sourceGraph']['sha256'],'weight':files['sourceWeight']['sha256']}:
        raise ValueError('Candidate lineage mismatch')
    output.mkdir(parents=True,exist_ok=False)
    for key,suffix in (('candidateGraph',''),('candidateWeight','.weight')):
        original=quant/'ffn-w8-g64'/('flow.mnn'+suffix)
        expected=recipe['weight_sha256' if suffix else 'graph_sha256']
        if digest(original)!=expected:raise ValueError('Candidate hash mismatch')
        target=output/('flow-ffn-w8-g64.mnn'+suffix);shutil.copyfile(original,target)
        files[key]={'file':target.name,'bytes':target.stat().st_size,'sha256':expected}
    fixtures=[]; path=output/'flow-w8-fixtures.f32.bin'
    with path.open('wb') as stream:
        for case in reference['cases']:
            if case['split']!='validation':continue
            archives=[np.load(Path(s['directory'])/'tensors.npz') for s in case['snapshots']]
            values={n:archives[0][n] for n in ('x','mask','mu','spks','cond','t')}
            values.update({'velocity1':archives[0]['estimator_out'],'velocity2':archives[1]['estimator_out'],
                           'referenceMel':np.load(Path(case['audio']).parent/'target-mel.npy')})
            sections={}
            for name,value in values.items():
                if not np.isfinite(value).all():raise ValueError('Nonfinite fixture')
                encoded=np.ascontiguousarray(value,dtype='<f4').tobytes()
                sections[name]={'offset':stream.tell(),'count':value.size,'shape':list(value.shape)}
                stream.write(encoded)
            fixtures.append({'id':case['id'],'text':case['text'],'frames':case['runtime_frames'],
                'effectiveFrames':case['effective_frames'],'audioSeconds':case['audio_seconds'],
                'times':[s['time'] for s in case['snapshots']], 'deltas':[s['delta'] for s in case['snapshots']],
                'sections':sections})
            for archive in archives:archive.close()
    if len(fixtures)!=5:raise ValueError('Exactly five held-out fixtures required')
    files['fixtures']={'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)}
    manifest={'schemaVersion':1,'recipe':'cosyvoice3-flow-ffn-w8-g64-v1','byteOrder':'little-endian',
        'files':files,'cases':fixtures,'quantizedOperators':44,'preservedFP16Operators':147,
        'scope':'Frozen five held-out cases; no LLM/HiFT, no whole-chain RTF or subjective acceptance',
        'gates':{'baselineMelRelativeL2':0.001,'normalMelRelativeL2':0.01,'lowMelRelativeL2':0.03,'repeatRelativeL2':0.001,
                 'note':'Exploratory numeric limits, not perceptual quality acceptance; no auto promotion'}}
    encoded=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    (project/'flow-w8-manifest.json').write_text(encoded,encoding='utf-8')
    (output/'flow-w8-manifest.json').write_text(encoded,encoding='utf-8')
    print(json.dumps({'files':files,'cases':len(fixtures),'output':str(output)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--quant',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();package(Path(__file__).resolve().parents[1],args.corpus.resolve(),args.quant.resolve(),args.output.resolve())
