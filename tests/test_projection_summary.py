from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_projection_matrix import summarize

def fixture():
    part=lambda n:{'cases':n,'numeric_screen_pass':True}
    return {'state':'completed','sensitivity_state':'completed',
        'profiles':{'ffn-q-w8':{'evaluations':{'low':{'state':'completed','summary':{'calibration':part(20),'validation':part(5)}},
            'normal':{'state':'completed','summary':{'validation':part(5)}}}}},
        'timing':{'state':'completed','summary':{'ffn-q-w8':{'cases':5,'max_repeat_mel_l2':0}}}}

class SummaryChecks(unittest.TestCase):
    def test_requires_complete_finite_replay_and_repeats(self):
        r=fixture();self.assertTrue(summarize(r)['recipes'][0]['eligible_for_phone_review'])
        for change in ['cases','screen','state','repeat','negative','timing']:
            d=copy.deepcopy(r)
            if change=='cases':d['profiles']['ffn-q-w8']['evaluations']['low']['summary']['calibration']['cases']=19
            if change=='screen':d['profiles']['ffn-q-w8']['evaluations']['normal']['summary']['validation']['numeric_screen_pass']=False
            if change=='state':d['sensitivity_state']='running'
            if change=='repeat':d['timing']['summary']['ffn-q-w8']['max_repeat_mel_l2']=.01
            if change=='negative':d['timing']['summary']['ffn-q-w8']['max_repeat_mel_l2']=-.01
            if change=='timing':d['timing']['state']='running'
            self.assertFalse(summarize(d)['recipes'][0]['eligible_for_phone_review'])
    def test_empty_is_not_accepted(self):
        self.assertEqual(summarize({})['recipes'],[])
        self.assertFalse(summarize(fixture())['automatic_promotion'])

if __name__=='__main__':unittest.main()
