"""Pair complete same-build phone FFN-control/projection module results."""
import argparse
import json
from pathlib import Path
IDS={'zh-numbers','zh-question','zh-heldout','en-heldout','mixed-heldout'}

def compare(report):
    arms={}
    for r in report.get('runs',[]):
        route=r.get('parameters',{}).get('route')
        if route in ['flow-projection-reference','flow-projection-candidate'] and r.get('kind')=='module-experiment' and r.get('state')=='passed' and r.get('result',{}).get('success') is True:
            arms[route]=r
    if len(arms)!=2:return {'state':'missing-complete-pair','available':list(arms)}
    a,b=(arms[k] for k in ['flow-projection-reference','flow-projection-candidate'])
    if any(a.get(k)!=b.get(k) for k in ['version','build']):return {'state':'version-mismatch'}
    def index(r):
        rows=r['result'].get('rows',[])
        return {x['caseID']:x for x in rows} if len(rows)==5 and {x.get('caseID') for x in rows}==IDS else None
    aa,bb=index(a),index(b)
    if aa is None or bb is None:return {'state':'requires-five-cases'}
    for key in IDS:
        x,y=aa[key],bb[key]
        for field in ['manifestSHA256','fileSHA256','frames','effectiveFrames','audioSeconds','precision','memory','backend','actualThreads','candidateMode']:
            if x.get(field) is None or x.get(field)!=y.get(field):return {'state':'identity-mismatch','field':field,'case':key}
        if y.get('pairedReferenceMelSHA256')!=x.get('targetMelSHA256') or not x.get('targetMelSHA256'):
            return {'state':'reference-mismatch','case':key}
        for row in [x,y]:
            runs=row.get('runs',[])
            if row.get('status')!='numeric-probe-pass' or row.get('completedRuns')!=3 or len(runs)!=3 or not all(r.get('status')=='finite-completed' and r.get('completedSteps')==2 for r in runs):return {'state':'incomplete-forward','case':key}
    def metrics(r):
        rows=r['result']['rows'];return {
            'warm_two_step_total_seconds':sum(x['warmTwoStepsMeanSeconds'] for x in rows),
            'audio_seconds':sum(x['audioSeconds'] for x in rows),
            'peakFootprintMB':r['result'].get('peakFootprintMB'),'minAvailableMB':r['result'].get('minAvailableMB'),
            'max_pc_original_mel_l2':max(x['pcBaselineMelRelativeL2'] for x in rows),
            'max_paired_mel_l2':max(x['pairedMelRelativeL2'] for x in rows),
            'thermal':[(x.get('thermalStart'),x.get('thermalEnd')) for x in rows], 'processID':r.get('processID')}
    control,candidate=metrics(a),metrics(b)
    return {'state':'matched-five-case-module-pair','runs':[a['id'],b['id']],'control':control,'candidate':candidate,
        'descriptive_speedup':control['warm_two_step_total_seconds']/candidate['warm_two_step_total_seconds'],
        'wholeChainRTFMeasured':False,'listeningValidated':False,'automaticPromotion':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);a=p.parse_args()
    print(json.dumps(compare(json.loads(a.report.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2))
