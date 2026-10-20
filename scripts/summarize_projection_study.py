"""Consolidate both exploratory sweeps, integrity audits and listening artifacts."""
import argparse
import json
from pathlib import Path
from flow_quant_common import digest
from summarize_projection_matrix import summarize

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--audio',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[];corpus=None
    for root in a.roots:
        raw=json.loads((root/'projection-report.json').read_text(encoding='utf-8'))
        audit=json.loads((root/'layout-integrity.json').read_text(encoding='utf-8'))
        if raw.get('state')!='completed' or audit.get('state')!='completed' or not audit.get('original_assets_unchanged'):raise ValueError('Incomplete or unaudited sweep')
        if corpus and corpus!=raw['corpus']['sha256']:raise ValueError('Different frozen corpus')
        corpus=raw['corpus']['sha256']
        rows.append({'name':root.name,'report_sha256':digest(root/'projection-report.json'),'integrity_sha256':digest(root/'layout-integrity.json'),**summarize(raw)})
    audio=json.loads(a.audio.read_text(encoding='utf-8'))
    if audio.get('state')!='completed':raise ValueError('Incomplete audio generation')
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    out={'state':'completed','unique_frozen_texts':25,'calibration_texts':20,'heldout_texts':5,'corpus_sha256':corpus,
        'sweeps':rows,'selected_phone_candidate':manifest,'listening_artifacts':audio,
        'original_assets_unchanged':True,'ACE_modified':False,'automatic_promotion':False,'listening_validated':False,
        'scope':'Pilot numerical screening and descriptive Windows timings; phone model remains an optional unvalidated module candidate'}
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':out['state'],'recipes':len({r['mode'] for s in rows for r in s['recipes']}),
        'candidate':manifest['candidateMode'],'wavs':sum(len(v) for v in audio['modes'].values()),'output':str(a.output)},indent=2))

if __name__=='__main__':main()
