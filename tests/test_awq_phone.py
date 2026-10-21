import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from package_awq_phone import package


class AwqPhone(unittest.TestCase):
    def test_review_does_not_relabel_failed_numeric_gate(self):
        m = json.loads((ROOT / 'flow-awq-manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(m['experimentalOnly'])
        self.assertTrue(m['userListeningReviewed'])
        self.assertFalse(m['pcNumericScreenPassed'])
        self.assertFalse(m['phoneValidated'])
        self.assertFalse(m['automaticPromotion'])
        self.assertEqual(m['pcScreen']['calibration']['cases'], 20)
        self.assertEqual(m['pcScreen']['validation']['cases'], 5)
        self.assertEqual(m['ffnW4Operators'], 44)
        self.assertEqual(m['projectionW8Operators'], 66)
        self.assertEqual(m['inverseScaleOperators'], 44)
        self.assertEqual(m['files']['candidateWeight']['sha256'],
                         '2404ace0f9fa511ccffb1a6e65d6a1dd7a19af216369eb793bde72ac9630799c')
        self.assertEqual(sum(x['bytes'] for x in m['files'].values()), 489346380)
        self.assertEqual(m['unchangedNumericGates']['mean_closed_loop_mel_l2'], .015)

    def test_packaging_requires_explicit_listening_decision(self):
        with self.assertRaisesRegex(ValueError, 'Explicit user listening review'):
            package(Path('not-read'), Path('not-created'))

    def test_selected_recipe_and_bounded_lifetimes(self):
        p = (ROOT / 'ios/Bridge/CosyVoicePipeline.inc').read_text(encoding='utf-8')
        self.assertIn('owner->_awq?verifyFlowAwq', p)
        self.assertIn('||owner->_awq', p)
        self.assertIn('reuseHiFT:_hift.get()', p)
        self.assertIn('_hift.reset(); _flow.reset(); _llm.reset()', p)
        self.assertIn('os_proc_available_memory()<1200000000ULL', p)
        self.assertIn('seed=2307+int(_requests)-1', p)
        v = (ROOT / 'ios/Bridge/CosyVoiceVerifiedAssets.inc').read_text(encoding='utf-8')
        self.assertIn('verifyFlowAwq', v)
        self.assertIn('verifySelectedAssets(directory,@"flow-awq-manifest"', v)
        self.assertIn('pcNumericScreenPassed', v)
        self.assertIn('没有自动回退原模型', v)

    def test_opt_in_current_text_and_default_unchanged(self):
        ui = (ROOT / 'ios/App/ContentView.swift').read_text(encoding='utf-8')
        self.assertIn('mode ?? (useW8Flow ? "three-sentence-staged-w8" : "three-sentence-staged")', ui)
        self.assertIn('useCurrentText: true', ui)
        self.assertIn('fixedBenchmark ? LongTextBenchmarks.fiveSentenceText : text', ui)
        self.assertIn('!models.awqReady', ui)
        self.assertIn('headPair ? .complete : playbackMode', ui)
        self.assertIn('end-to-end-awq.wav', ui)
        self.assertIn('flow-awq-manifest.json', (ROOT / 'project.yml').read_text(encoding='utf-8'))
        native = (ROOT / 'ios/Bridge/CosyVoiceGenerate.mm').read_text(encoding='utf-8')
        self.assertIn('completed[@"pcNumericScreenPassed"]=@NO', native)
        self.assertIn('completed[@"flowSteps"]=@2', native)
        self.assertIn('AWQ FFN W4/g32 scaling+clipping', native)


if __name__ == '__main__':
    unittest.main()
