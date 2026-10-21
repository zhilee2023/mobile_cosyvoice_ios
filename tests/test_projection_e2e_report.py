import unittest,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_projection_e2e import compare,MODES,HASHES
def fixture():
    runs=[]
    for n,mode in enumerate(MODES):
        llm=[{'seed':2307+i,'eosValidated':True,'invalidTokens':0,'samplerResetPerRequest':True,'speechTokenSHA256':str(i),'speechTokens':70} for i in range(5)]
        rows=[{'phase':'sentence','result':{'success':True,'flowSteps':2,'flowPrecision':'High','flowMemory':'Low','hiftCorePrecision':'low','speechTokenSHA256':str(i),
          'flowIdentity':{'files':{'candidateWeight':{'sha256':HASHES[n]}}},'hiFTBatchReuse':n==2,'melSHA256':'mel'+str(i),'pcmSHA256':'pcm'+str(i)}} for i in range(5)]
        runs.append({'id':str(n),'kind':'sentence-generation','state':'completed','version':'0.7.16','build':'30','processID':str(n),
          'parameters':{'pipelineMode':mode,'text':'same five sentences','threads':4,'hiFTLow':True,'optimized':True,'playbackMode':'complete'},
          'result':{'completedAudio':True,'generatedSegments':5,'plannedSegments':5,'headMode':'reduced','computeSeconds':20-n,'audioSeconds':14,
            'attempts':[{'phase':'prepare-batch','result':{'llmRequests':llm}},*rows]}})
    return {'runs':runs}
class E2EReport(unittest.TestCase):
    def test_complete_and_no_auto_promotion(self):
        r=compare(fixture());self.assertEqual(r['state'],'matched-three-arm-set');self.assertFalse(r['automaticPromotion']);self.assertFalse(r['same_process'])
    def test_reject_missing_or_wrong_build(self):
        r=fixture();r['runs'].pop();self.assertEqual(compare(r)['state'],'missing-complete-three-arm-set')
        r=fixture();r['runs'][1]['build']='29';self.assertEqual(compare(r)['state'],'version-mismatch')
    def test_reject_wrong_model_seed_and_token(self):
        r=fixture();r['runs'][1]['result']['attempts'][1]['result']['flowIdentity']['files']['candidateWeight']['sha256']='other'
        self.assertEqual(compare(r)['state'],'model-identity-mismatch')
        r=fixture();r['runs'][1]['result']['attempts'][0]['result']['llmRequests'][1]['seed']=4
        self.assertEqual(compare(r)['state'],'missing-complete-eos-and-seed-evidence')
        r=fixture();r['runs'][1]['result']['attempts'][0]['result']['llmRequests'][1]['speechTokenSHA256']='different'
        self.assertEqual(compare(r)['state'],'llm-acoustic-token-mismatch')
    def test_not_identical_pcm_requires_review(self):
        r=fixture();r['runs'][2]['result']['attempts'][1]['result']['pcmSHA256']='different'
        self.assertEqual(compare(r)['state'],'hift-output-differs-review-required')
    def test_cannot_compare_playback_modes(self):
        r=fixture();r['runs'][2]['parameters']['playbackMode']='buffered'
        self.assertEqual(compare(r)['state'],'configuration-mismatch')
if __name__=='__main__':unittest.main()
