"""Compare same-build five-sentence A/B/C; never treat module RTF as E2E."""
import argparse,json
from pathlib import Path
MODES=['three-sentence-staged-w8-head-reduced-reuse','three-sentence-staged-projection-head-reuse','three-sentence-staged-projection-head-reuse-hift-reuse']
HASHES=['f5d713123341572a85fd3639c02a852e971916f18c80b181861f20cda4f662fd','884b43e1fae71b55e2ce31aaf199434da9fb64429047a8eb771e34f903ea9205','884b43e1fae71b55e2ce31aaf199434da9fb64429047a8eb771e34f903ea9205']
def compare(report):
    selected={}
    for run in report.get('runs',[]):
        mode=run.get('parameters',{}).get('pipelineMode')
        if mode in MODES and run.get('kind')=='sentence-generation' and run.get('state')=='completed':selected[mode]=run
    if len(selected)!=3:return {'state':'missing-complete-three-arm-set','available':list(selected)}
    runs=[selected[m] for m in MODES];first=runs[0];summaries=[];sequences=[];acoustics=[]
    for index,run in enumerate(runs):
        p=run['parameters'];r=run.get('result',{})
        for field in ['version','build']:
            if run.get(field)!=first.get(field):return {'state':'version-mismatch'}
        if any(p.get(k)!=first['parameters'].get(k) for k in ['text','threads','hiFTLow','optimized','playbackMode']):return {'state':'configuration-mismatch'}
        if p.get('threads')!=4 or p.get('hiFTLow') is not True or p.get('playbackMode')!='complete':return {'state':'not-controlled-five-sentence-test'}
        if r.get('completedAudio') is not True or r.get('generatedSegments')!=5 or r.get('plannedSegments')!=5 or r.get('headMode')!='reduced':return {'state':'incomplete-audio'}
        llm=[];rows=[];stages={}
        for a in r.get('attempts',[]):
            v=a.get('result',{})
            if a.get('phase')=='prepare-batch':llm.extend(v.get('llmRequests',[]))
            if a.get('phase')=='sentence':
                rows.append(v)
                if v.get('success') is not True or v.get('flowSteps')!=2 or v.get('flowPrecision')!='High' or v.get('flowMemory')!='Low' or v.get('hiftCorePrecision')!='low':return {'state':'acoustic-configuration-mismatch'}
                actual=v.get('flowIdentity',{}).get('files',{}).get('candidateWeight',{}).get('sha256')
                if actual!=HASHES[index]:return {'state':'model-identity-mismatch'}
                for k,value in v.get('stageSeconds',{}).items():stages[k]=stages.get(k,0)+value
        if len(llm)!=5 or len(rows)!=5 or not all(x.get('eosValidated') is True and x.get('invalidTokens')==0 and x.get('samplerResetPerRequest') is True and x.get('seed')==2307+i for i,x in enumerate(llm)):return {'state':'missing-complete-eos-and-seed-evidence'}
        sequence=[x.get('speechTokenSHA256') for x in llm]
        if not all(sequence) or sequence!=[x.get('speechTokenSHA256') for x in rows]:return {'state':'llm-acoustic-token-mismatch'}
        if any(x.get('hiFTBatchReuse') is not (index==2) for x in rows):return {'state':'hift-reuse-flag-mismatch'}
        sequences.append(sequence);acoustics.append(rows)
        summaries.append({'mode':MODES[index],'runID':run['id'],'computeSeconds':r.get('computeSeconds'),'computeRTF':r.get('computeRTF'),
          'generationWallSeconds':r.get('generationWallSeconds'),'audioSeconds':r.get('audioSeconds'),'firstQueuedSeconds':r.get('firstQueuedSeconds'),
          'peakFootprintMB':r.get('peakFootprintMB'),'minAvailableMB':r.get('minAvailableMB'),'stageSums':stages,
          'llm':{k:sum(x.get(k,0) for x in llm) for k in ['loadSeconds','prefillSeconds','decodeSeconds']},
          'speechTokens':sum(x['speechTokens'] for x in llm),'hiFTLifecycle':[x.get('hiFTLifecycle') for x in rows]})
    if sequences[0]!=sequences[1] or sequences[1]!=sequences[2]:return {'state':'speech-tokens-differ-no-attribution','summaries':summaries}
    mel=[x.get('melSHA256') for x in acoustics[1]];pcm=[x.get('pcmSHA256') for x in acoustics[1]]
    if not all(mel+pcm):return {'state':'missing-mel-pcm-hashes'}
    identical=mel==[x.get('melSHA256') for x in acoustics[2]] and pcm==[x.get('pcmSHA256') for x in acoustics[2]]
    return {'state':'matched-three-arm-set' if identical else 'hift-output-differs-review-required','summaries':summaries,
       'projection_time_reduction_fraction':1-summaries[1]['computeSeconds']/summaries[0]['computeSeconds'],
       'hift_reuse_time_reduction_fraction':1-summaries[2]['computeSeconds']/summaries[1]['computeSeconds'],
       'same_process':len({r.get('processID') for r in runs})==1,'listeningValidated':False,'automaticPromotion':False}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);a=p.parse_args()
    print(json.dumps(compare(json.loads(a.report.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2))
