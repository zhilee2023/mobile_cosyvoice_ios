from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_projection_phone import compare,IDS

def pair():
    runs=[]
    for arm in ['reference','candidate']:
        rows=[]
        for key in sorted(IDS):rows.append({'caseID':key,'manifestSHA256':'sha','fileSHA256':{'all':'same'},'frames':384,'effectiveFrames':356,
            'audioSeconds':3.64,'precision':'High','memory':'Low','backend':0,'actualThreads':4,'candidateMode':'ffn-qout-w8',
            'targetMelSHA256':'mel-'+key,'pairedReferenceMelSHA256':'mel-'+key,'status':'numeric-probe-pass','completedRuns':3,
            'runs':[{'status':'finite-completed','completedSteps':2} for _ in range(3)],'warmTwoStepsMeanSeconds':2,
            'pcBaselineMelRelativeL2':.01,'pairedMelRelativeL2':.01})
        runs.append({'id':arm,'version':'0.7.15','build':'29','state':'passed','kind':'module-experiment',
            'parameters':{'route':'flow-projection-'+arm},'result':{'success':True,'rows':rows}})
    return {'runs':runs}

class PhonePair(unittest.TestCase):
    def test_good_pair_and_no_auto_promotion(self):
        r=compare(pair());self.assertEqual(r['state'],'matched-five-case-module-pair');self.assertFalse(r['automaticPromotion'])
    def test_failure_version_input_and_reference_rejected(self):
        for mutation,state in [('failure','missing-complete-pair'),('version','version-mismatch'),('input','identity-mismatch'),('ref','reference-mismatch'),('steps','incomplete-forward')]:
            d=copy.deepcopy(pair());b=d['runs'][1];row=b['result']['rows'][0]
            if mutation=='failure':b['state']='not-passed'
            if mutation=='version':b['build']='28'
            if mutation=='input':row['frames']=512
            if mutation=='ref':row['pairedReferenceMelSHA256']='wrong'
            if mutation=='steps':row['runs'][1]['completedSteps']=1
            self.assertEqual(compare(d)['state'],state)

if __name__=='__main__':unittest.main()
