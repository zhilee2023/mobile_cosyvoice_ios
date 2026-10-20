from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_head_e2e import compare,PREFIX

def run(mode):
    requests=[{'text':str(i),'seed':2307+i,'tokens':[1,2,3],'eosValidated':True,'invalidTokens':0} for i in range(5)]
    identity={'files':{'weight':'same-identity-set'},'manifestSHA256':'manifest','verificationSeconds':1}
    return {'id':mode,'kind':'sentence-generation','state':'completed','version':'0.7.13','build':'27',
            'parameters':{'pipelineMode':PREFIX+mode,'text':'same text','threads':4,'hiFTLow':True,'playbackMode':'complete','optimized':True},
            'result':{'lifecycle':{'flowIdentity':identity,'headIdentity':identity},
                      'attempts':[{'phase':'prepare-batch','result':{'llmRequests':requests}}]}}

class HeadReport(unittest.TestCase):
    def test_complete_pair(self):
        self.assertEqual(compare({'runs':[run('full'),run('reduced')]})['state'],'matched-complete-tokens')
    def test_failure_or_old_version_not_accepted(self):
        a,b=run('full'),run('reduced');b['state']='not-completed'
        self.assertEqual(compare({'runs':[a,b]})['state'],'missing-complete-pair')
        b['state']='completed';b['build']='26'
        self.assertEqual(compare({'runs':[a,b]})['state'],'identity-mismatch')
    def test_token_divergence_is_not_pass(self):
        a,b=run('full'),run('reduced');b['result']['attempts'][0]['result']['llmRequests'][2]['tokens']=[1,2,4]
        self.assertEqual(compare({'runs':[a,b]})['state'],'token-divergence')
    def test_seed_and_asset_changes_rejected(self):
        a,b=run('full'),run('reduced');b['result']['attempts'][0]['result']['llmRequests'][0]['seed']=5
        self.assertEqual(compare({'runs':[a,b]})['state'],'request-mismatch')
        b=run('reduced');b['result']['lifecycle']['flowIdentity']=copy.deepcopy(b['result']['lifecycle']['flowIdentity'])
        b['result']['lifecycle']['flowIdentity']['manifestSHA256']='different'
        self.assertEqual(compare({'runs':[a,b]})['state'],'identity-mismatch')

if __name__=='__main__':unittest.main()
