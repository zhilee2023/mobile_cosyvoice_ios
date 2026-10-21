from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ProjectionE2E(unittest.TestCase):
    def test_explicit_recipe_and_lifetime(self):
        p=(ROOT/'ios/Bridge/CosyVoicePipeline.inc').read_text(encoding='utf-8')
        self.assertIn('owner->_projection?verifyFlowProjection',p)
        self.assertIn('_hift.reset(); _flow.reset(); _llm.reset()',p)
        self.assertIn('reuseHiFT:_hift.get()',p)
        self.assertIn('if(done)',p)
        self.assertIn('_hift.reset();_flow.reset();_prepared.clear()',p)
        self.assertIn('seed=2307+int(_requests)-1',p)
    def test_dynamic_shape_keeps_metadata(self):
        p=(ROOT/'ios/Bridge/CosyVoiceHiFTBatch.inc').read_text(encoding='utf-8')
        self.assertNotIn('releaseModel()',p)
        self.assertIn('previous!=frames',p)
        self.assertIn('frames>210',p)
        self.assertIn('RESIZE_STATUS',p)
        self.assertIn('coreLow!=low',p)
        self.assertIn('slot->traceLog=&trace',p)
        self.assertIn('owner->unbind()',p)
    def test_metrics_and_repeated_inputs(self):
        p=(ROOT/'ios/Bridge/CosyVoiceGenerate.mm').read_text(encoding='utf-8')
        for key in ['hiFTLifecycle','hiFTBatchReuse','melSHA256','pcmSHA256']:self.assertIn(key,p)
        self.assertIn('graph.inputFloat("mel", mel) || !graph.inputFloat("source_stft", stft)',p)
        self.assertIn('HiFTTraceBinding hiftBinding{reuseHiFT}',p)
        ui=(ROOT/'ios/App/ContentView.swift').read_text(encoding='utf-8')
        self.assertIn('lowHiFT: headPair ? true : lowHiFT',ui)
        self.assertIn('headPair ? .complete : playbackMode',ui)
if __name__=='__main__':unittest.main()
