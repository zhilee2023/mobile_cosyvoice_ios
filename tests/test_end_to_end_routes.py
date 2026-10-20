import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class EndToEndRoutes(unittest.TestCase):
    def test_head_identity_and_mapping(self):
        m=json.loads((ROOT/'speech-head-manifest.json').read_text())
        self.assertEqual(m['candidateWidth'],6563)
        self.assertEqual(m['globalVocabulary'],158720)
        self.assertEqual(m['eos'],m['speechOffset']+6562)
        self.assertEqual(m['reserved'],m['speechOffset']+6561)
        self.assertNotEqual(m['sourceWeight']['sha256'],m['candidateWeight']['sha256'])

    def test_default_and_opt_in_are_separate(self):
        native=(ROOT/'ios/Bridge/CosyVoiceGenerate.mm').read_text(encoding='utf-8')
        pipeline=(ROOT/'ios/Bridge/CosyVoicePipeline.inc').read_text(encoding='utf-8')
        self.assertIn('removeObjectForKey:@"cosyvoice_head_experiment"',native)
        self.assertIn('flowW8:NO',native)
        self.assertIn('_headMode?nullptr:&_llm',pipeline)
        self.assertIn('seed=2307+int(_requests)-1',pipeline)
        self.assertIn('llmMetrics[@"tokens"]',pipeline)
        self.assertIn('f.flowW8=_useW8',pipeline)
        self.assertIn('verifySpeechHead(directory,trace,&failure)',pipeline)

    def test_failed_retry_preserves_reference(self):
        native=(ROOT/'ios/Bridge/CosyVoiceRouteExperiments.inc').read_text(encoding='utf-8')
        self.assertIn('[summary[@"success"] boolValue]&&rows.count==5',native)
        self.assertIn('last-success-flow-w8-reference.json',native)
        route=(ROOT/'ios/Bridge/CosyVoiceFlowW8Experiment.inc').read_text(encoding='utf-8')
        self.assertIn('last-success-flow-w8-reference.json',route)

    def test_paired_ui_uses_five_complete_sentences(self):
        ui=(ROOT/'ios/App/ContentView.swift').read_text(encoding='utf-8')
        self.assertIn('headPair ? LongTextBenchmarks.fiveSentenceText : text',ui)
        self.assertIn('headPair ? .complete : playbackMode',ui)
        stream=(ROOT/'ios/App/SentenceStream.swift').read_text(encoding='utf-8')
        self.assertIn('LongTextBenchmarks.audioArchiveName(experimentMode)',stream)
        self.assertIn('copyStart',stream)

if __name__=='__main__':unittest.main()
