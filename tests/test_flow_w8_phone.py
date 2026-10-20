import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FlowW8PhoneChecks(unittest.TestCase):
    def test_manifest_and_layout(self):
        m=json.loads((ROOT/'flow-w8-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['recipe'],'cosyvoice3-flow-ffn-w8-g64-v1')
        self.assertEqual(m['quantizedOperators'],44)
        self.assertEqual(len(m['cases']),5)
        self.assertEqual(len({c['id'] for c in m['cases']}),5)
        self.assertEqual(set(m['files']),{'sourceGraph','sourceWeight','candidateGraph','candidateWeight','fixtures'})
        ranges=[]
        for entry in m['files'].values():
            self.assertEqual(Path(entry['file']).name,entry['file'])
            self.assertGreater(entry['bytes'],0);self.assertEqual(len(entry['sha256']),64)
        for case in m['cases']:
            n=case['frames'];effective=case['effectiveFrames']
            self.assertEqual(n%32,0);self.assertGreater(effective,174);self.assertLessEqual(effective,n)
            self.assertAlmostEqual(case['times'][1],1-2**-.5)
            expected={'x':80*n,'mask':n,'mu':80*n,'spks':80,'cond':80*n,'t':1,
                      'velocity1':80*n,'velocity2':80*n,'referenceMel':80*(effective-174)}
            self.assertEqual(set(case['sections']),set(expected))
            for name,s in case['sections'].items():
                self.assertEqual(s['count'],expected[name]);self.assertEqual(s['offset']%4,0)
                ranges.append((s['offset'],s['offset']+s['count']*4))
        position=0
        for start,end in sorted(ranges):self.assertEqual(start,position);position=end
        self.assertEqual(position,m['files']['fixtures']['bytes'])

    def test_routes_remain_opt_in(self):
        route=(ROOT/'ios/Bridge/CosyVoiceFlowW8Experiment.inc').read_text(encoding='utf-8')
        self.assertIn('saved[@"version"]',route)
        self.assertIn('targetMelSHA256',route)
        self.assertIn('manifestSHA256',route)
        self.assertIn('sampledBudgetOK(trace)',route)
        self.assertNotIn('generateSpeechTokens(',route)
        self.assertNotIn('runAtDirectory:',route)
        self.assertNotIn('flow.releaseModel()',route)
        default=(ROOT/'ios/Bridge/CosyVoiceGenerate.mm').read_text(encoding='utf-8')
        self.assertIn('bool normalMemory = false',default)
        self.assertNotIn('flow-ffn-w8-g64.mnn',default)


if __name__=='__main__':unittest.main()
