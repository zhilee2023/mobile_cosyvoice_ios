from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parents[1]

class ProjectionPhone(unittest.TestCase):
    def test_bundled_manifest_pins_safe_selected_recipe(self):
        m=json.loads((ROOT/'flow-projection-manifest.json').read_text(encoding='utf-8'))
        old=json.loads((ROOT/'flow-w8-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['candidateMode'],'ffn-qvout-w8')
        self.assertEqual(m['quantizedOperators'],110)
        self.assertEqual(m['preservedFP16Operators'],81)
        self.assertEqual(m['files']['sourceGraph'],old['files']['candidateGraph'])
        self.assertEqual(m['files']['sourceWeight'],old['files']['candidateWeight'])
        self.assertEqual(m['files']['fixtures'],old['files']['fixtures'])
        self.assertEqual(m['cases'],old['cases'])
        self.assertTrue(m['pcScreen']['eligible_for_phone_review'])
        self.assertEqual(m['pcScreen']['low_calibration']['cases'],20)
        self.assertEqual(m['pcScreen']['low_validation']['cases'],5)
        self.assertFalse(m['automaticPromotion'])
    def test_independent_cpu_control_and_candidate(self):
        route=(ROOT/'ios/Bridge/CosyVoiceFlowW8Experiment.inc').read_text(encoding='utf-8')
        for key in ['flow-projection-reference','flow-projection-candidate','flow-projection-manifest',
                    'last-success-flow-projection-reference.json','flow-projection-cpu-%@.f32']:
            self.assertIn(key,route)
        self.assertIn('low=projection||',route)
        self.assertIn('pc.relativeL2<=0.03',route)
        self.assertIn('projection?1e-5',route)
        self.assertIn('saved[@"version"]',route)
        self.assertIn('matched[@"targetMelSHA256"]',route)
        self.assertNotIn('generateSpeechTokens(',route)
        self.assertNotIn('runAtDirectory:',route)
        finish=(ROOT/'ios/Bridge/CosyVoiceRouteExperiments.inc').read_text(encoding='utf-8')
        self.assertIn('[route isEqual:@"flow-projection-reference"]&&[summary[@"success"] boolValue]&&rows.count==5',finish)
        self.assertIn('last-success-flow-projection-reference.json',finish)

    def test_no_automatic_generation_promotion(self):
        native=(ROOT/'ios/Bridge/CosyVoicePipeline.inc').read_text(encoding='utf-8')
        self.assertIn('owner->_projection=[mode hasPrefix:@"three-sentence-staged-projection-"]',native)
        self.assertIn('owner->_projection?verifyFlowProjection',native)
        ui=(ROOT/'ios/App/ContentView.swift').read_text(encoding='utf-8')
        self.assertIn('mode ?? (useW8Flow ? "three-sentence-staged-w8" : "three-sentence-staged")',ui)
        package=(ROOT/'scripts/package_projection_phone.py').read_text(encoding='utf-8')
        self.assertIn('eligible_for_phone_review',package)
        self.assertIn('Bit-exact preservation audit required',package)
        self.assertIn("'automaticPromotion':False",package)

if __name__=='__main__':unittest.main()
