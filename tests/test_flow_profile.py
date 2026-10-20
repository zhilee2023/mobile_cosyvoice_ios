from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class FlowProfile(unittest.TestCase):
    def test_no_intermediate_reads_and_bounded_metadata(self):
        source=(ROOT/'ios/Bridge/CosyVoiceFlowProfile.inc').read_text(encoding='utf-8')
        self.assertIn('runSessionWithCallBackInfo',source)
        self.assertNotIn('metalTensorStats',source)
        self.assertNotIn('readMap',source)
        self.assertIn('profileVsOrdinaryMelL2',source)
        self.assertIn('profileOverheadRatio',source)
        self.assertIn('secondsSince(start)>60',source)
        header=(ROOT/'ios/Bridge/CosyVoiceOpProfile.hpp').read_text(encoding='utf-8')
        self.assertIn('limit = 4096',header)
        self.assertIn('!std::isfinite(elapsed)',header)

    def test_gpu_reference_and_safety(self):
        source=(ROOT/'ios/Bridge/CosyVoiceFlowProfile.inc').read_text(encoding='utf-8')
        self.assertIn('saved[@"version"]',source)
        self.assertIn('saved[@"build"]',source)
        self.assertIn('savedIdentity[@"manifestSHA256"]',source)
        self.assertIn('saved[@"melSHA256"]',source)
        self.assertIn('backend!=MNN_FORWARD_METAL',source)
        self.assertIn('flow.mainBackend()',source)
        self.assertIn('os_proc_available_memory()<1100000000ULL',source)
        self.assertIn('memory-gate-no-forward',source)
        self.assertIn('readSyncSeconds',source)
        self.assertNotIn('generateSpeechTokens(',source)
        self.assertNotIn('Precision_Low',source)
        # Only one Graph is alive per route. No CPU/GPU co-resident comparison.
        self.assertLess(source.index('return finish(passed,'),source.index('bool ok=[&]'))

    def test_backend_info_uses_required_array(self):
        bridge=ROOT/'ios/Bridge'
        native=(bridge/'CosyVoiceGenerate.mm').read_text(encoding='utf-8')
        self.assertIn('int backends[2]={-1,-1}',native)
        for p in list(bridge.glob('*.inc'))+list(bridge.glob('*.mm')):
            for line in p.read_text(encoding='utf-8').splitlines():
                if 'getSessionInfo' in line and 'BACKENDS' in line:
                    self.assertNotRegex(line,r'BACKENDS\s*,\s*&\w+')

if __name__=='__main__':unittest.main()
