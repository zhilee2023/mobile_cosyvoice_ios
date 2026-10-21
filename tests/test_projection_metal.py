from pathlib import Path
import json
import plistlib
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ProjectionMetal(unittest.TestCase):
    def test_current_weights_high_precision_and_isolated_references(self):
        p=(ROOT/'ios/Bridge/CosyVoiceMetalExperiment.inc').read_text(encoding='utf-8')
        for marker in ['verifyFlowProjection', 'projectionIdentity[@"files"][@"candidateGraph"]',
                       'metal&&!projection', 'last-success-%@.json',
                       'candidate[@"modelIdentity"][@"manifestSHA256"]',
                       'candidate[@"schedule"]', 'candidate[@"cpuThreads"]',
                       'saved[@"version"]', 'saved[@"build"]',
                       'candidate[@"inputSHA256"]', 'savedRow[@"melSHA256"]']:
            self.assertIn(marker,p)
        self.assertIn('if (!cpuOnly && splitReference)',p)
        self.assertIn('if (cpuOnly)',p)
        self.assertLess(p.index('if (cpuOnly)'),p.index('const bool gpuOK ='))
        self.assertNotIn('generateSpeechTokens(',p)
        self.assertIn('if (!gpuOK) return finish(false',p)

    def test_no_relaxed_memory_or_hidden_acceptance(self):
        p=(ROOT/'ios/Bridge/CosyVoiceMetalExperiment.inc').read_text(encoding='utf-8')
        for marker in ['os_proc_available_memory() < 1100000000ULL',
                       'sampledBudgetOK(trace)', '2900000000ULL',
                       'projection?1e-5:1e-3', 'projection?0.01:0.001',
                       '@"strictNumericPassed":@(strict)', '@"listeningValidated": @NO',
                       'last-success-flow-projection-metal-high-256.json',
                       'priorComparison[@"numericProbePassed"]', 'flow.releaseModel()',
                       'readSyncSeconds', 'runSeconds', 'completedSteps',
                       'footprintAfterResizeMB', 'availableAfterReleaseModelMB']:
            self.assertIn(marker,p)

    def test_success_cache_only_and_opt_in_routes(self):
        p=(ROOT/'ios/Bridge/CosyVoiceRouteExperiments.inc').read_text(encoding='utf-8')
        for suffix in ['cpu-reference-256','cpu-reference-384','metal-high-256','metal-high-384']:
            self.assertIn('@"flow-projection-'+suffix+'"',p)
        self.assertIn('&&[summary[@"success"] boolValue]',p)
        self.assertIn('last-success-%@.json',p)
        catalog=(ROOT/'ios/App/ExperimentCatalog.swift').read_text(encoding='utf-8')
        self.assertIn('"flow-projection-profile", title: "投影 W8 · 五条 CPU 热点", state: .archived',catalog)
        self.assertIn('"flow-awq-profile", title: "AWQ W4 · 五条 CPU 热点", state: .archived',catalog)
        ui=(ROOT/'ios/App/ContentView.swift').read_text(encoding='utf-8')
        self.assertIn('正式完整生成仍走CPU',ui)
        version=plistlib.loads((ROOT/'ios/Info.plist').read_bytes())
        self.assertEqual((version['CFBundleShortVersionString'],version['CFBundleVersion']),('0.7.19','33'))
        manifest=json.loads((ROOT/'flow-projection-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['files']['candidateWeight']['sha256'],
                         '884b43e1fae71b55e2ce31aaf199434da9fb64429047a8eb771e34f903ea9205')


if __name__=='__main__':unittest.main()
