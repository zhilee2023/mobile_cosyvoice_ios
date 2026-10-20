"""Decode saved held-out Mels with unchanged HiFT; no repeat Flow/LLM run."""
import argparse
import json
from pathlib import Path
import time
import wave
import numpy as np
from pc_flow_hift_probe import hift
from audio_sensitivity_metrics import read_wav,compare
from flow_quant_common import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',required=True)
    a=p.parse_args();d=json.loads((a.corpus/'calibration-report.json').read_text(encoding='utf-8'))
    cases=[c for c in d['cases'] if c['split']=='validation']
    if len(cases)!=5:raise ValueError('Need all five held-out cases')
    for name in ['hift-f0.fp32.mnn','hift-core.fp32.mnn','source-linear-weight.bin','source-linear-bias.bin']:
        if digest(a.assets/name)!=d['model_files'][name]['sha256']:raise ValueError('HiFT original changed')
    matrix=json.loads((a.root/'projection-report.json').read_text(encoding='utf-8'))
    # Follow-up sweeps can reuse the byte-identical control's numeric replay.
    # Its Mels remain in the original evaluation directory, not in this root.
    evaluations={mode:Path(matrix['profiles'][mode]['evaluations']['low']['report']) for mode in a.modes}
    for path in evaluations.values():
        if not path.is_file():raise ValueError('Missing saved replay: '+str(path))
    a.output.mkdir(parents=True,exist_ok=False)
    report={'state':'running','scope':'Same saved speech-token trajectories; fixed original F0/HiFT High, deterministic excitation; audio proxies are not MOS',
        'listening_validated':False,'modes':{},'generated_from_saved_mel':True}
    def save():(a.output/'audio-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    for mode in a.modes:
        evaluation=json.loads(evaluations[mode].read_text(encoding='utf-8'))
        if evaluation.get('state')!='completed':raise ValueError('Incomplete replay '+mode)
        rows=[];report['modes'][mode]=rows
        for c in cases:
            print('AUDIO',mode,c['id'],flush=True)
            mel_path=evaluations[mode].parent/(c['id']+'-mel.npy');mel=np.load(mel_path)
            if mel.shape!=(80,c['effective_frames']-174) or not np.isfinite(mel).all():raise ValueError('Invalid saved Mel')
            start=time.perf_counter();pcm,times=hift(a.assets,mel,core_precision='high',core_memory='normal')
            if not np.isfinite(pcm).all() or np.max(np.abs(pcm))>4 or np.sqrt(np.mean(pcm.astype(np.float64)**2))<.001:raise ValueError('Unhealthy PCM')
            path=a.output/(mode+'-'+c['id']+'.wav')
            with wave.open(str(path),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
                stream.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
            original=read_wav(Path(c['audio']));candidate=read_wav(path)
            rows.append({'case':c['id'],'text':c['text'],'wav':str(path.resolve()),'sha256':digest(path),
                'audio_seconds':len(pcm)/24000,'mel_sha256':digest(mel_path),'render_seconds':time.perf_counter()-start,
                'peak':float(np.max(np.abs(pcm))),'clip_fraction':float(np.mean(np.abs(pcm)>=1)),
                'hift_stage_seconds':times,'proxies_vs_original':compare(original,candidate)})
            save()
    report['state']='completed';save()

if __name__=='__main__':main()
