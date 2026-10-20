"""Read one unified phone report; pair only matching complete end-to-end runs."""
import argparse
import json
from pathlib import Path

PREFIX='three-sentence-staged-w8-head-'

def compare(report):
    arms={}
    for run in report.get('runs',[]):
        mode=run.get('parameters',{}).get('pipelineMode','')
        if mode in (PREFIX+'full',PREFIX+'reduced') and run.get('kind')=='sentence-generation' and run.get('state')=='completed':
            arms[mode.removeprefix(PREFIX)]=run
    if set(arms)!={'full','reduced'}:return {'state':'missing-complete-pair','available':list(arms)}
    a,b=arms['full'],arms['reduced']
    for field in ('version','build'):
        if a.get(field)!=b.get(field):return {'state':'identity-mismatch','field':field}
    for field in ('text','threads','hiFTLow','playbackMode','optimized'):
        if a.get('parameters',{}).get(field)!=b.get('parameters',{}).get(field):return {'state':'configuration-mismatch','field':field}
    for field in ('flowIdentity','headIdentity'):
        left=a.get('result',{}).get('lifecycle',{}).get(field)
        right=b.get('result',{}).get('lifecycle',{}).get(field)
        # Verification duration differs; actual model identities must not.
        if not left or not right or left.get('files')!=right.get('files') or left.get('manifestSHA256')!=right.get('manifestSHA256'):
            return {'state':'identity-mismatch','field':field}
    def requests(run):
        return [row for attempt in run['result'].get('attempts',[]) if attempt.get('phase')=='prepare-batch'
                for row in attempt.get('result',{}).get('llmRequests',[])]
    aa,bb=requests(a),requests(b)
    if len(aa)!=5 or len(bb)!=5:return {'state':'requires-five-complete-requests','counts':[len(aa),len(bb)]}
    rows=[]
    for left,right in zip(aa,bb):
        if left.get('text')!=right.get('text') or left.get('seed')!=right.get('seed') or left.get('seed') is None:
            return {'state':'request-mismatch'}
        ids=left.get('tokens',[]);other=right.get('tokens',[])
        valid=all(r.get('eosValidated') is True and r.get('invalidTokens')==0 for r in (left,right))
        legal=bool(ids) and bool(other) and all(isinstance(t,int) and 0<=t<=6560 for t in ids+other)
        rows.append({'text':left['text'],'seed':left['seed'],'eosAndLegalTokens':valid and legal,
                     'sameCompleteTokens':valid and legal and ids==other,'counts':[len(ids),len(other)]})
    matched=all(r['sameCompleteTokens'] for r in rows)
    def metrics(run):
        r=run['result'];return {k:r.get(k) for k in ('computeRTF','generationWallRTF','audioSeconds','computeSeconds','peakFootprintMB','minAvailableMB')}
    return {'state':'matched-complete-tokens' if matched else 'token-divergence','runs':[a['id'],b['id']],
            'sentences':rows,'full':metrics(a),'reduced':metrics(b),'listeningValidated':False,
            'scope':'Full-vs-reduced output head, both restricted speech/EOS and fresh per-sentence LLM; not a comparison against default within-batch LLM reuse'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path)
    args=p.parse_args();print(json.dumps(compare(json.loads(args.report.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2))
