import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patch_mnn_speech_head import patch_text, SAMPLE_ANCHOR, SAMPLE_PATCH, SEED_ANCHOR, SEED_PATCH, RESET_ANCHOR, RESET_PATCH


class PatchTests(unittest.TestCase):
    def test_ios_exceptions_disabled_failure_is_not_eos(self):
        self.assertNotIn('throw ',SAMPLE_PATCH)
        self.assertIn('LlmStatus::INTERNAL_ERROR',SAMPLE_PATCH)
        self.assertIn('return 0;',SAMPLE_PATCH)
    def test_exact_anchors(self):
        for anchor, replacement in [(SAMPLE_ANCHOR,SAMPLE_PATCH),(SEED_ANCHOR,SEED_PATCH),(RESET_ANCHOR,RESET_PATCH)]:
            self.assertEqual(patch_text("prefix\n"+anchor+"\nsuffix",anchor,replacement),
                             "prefix\n"+replacement+"\nsuffix")
            for invalid in ["",anchor+anchor,replacement]:
                with self.assertRaises(ValueError): patch_text(invalid,anchor,replacement)

    def test_reset_recreates_only_opt_in_sampler(self):
        self.assertIn('cosyvoice_head_experiment',RESET_PATCH)
        self.assertIn('cosyvoice_experiment_seed',RESET_PATCH)
        self.assertIn('mSampler.reset(Sampler::createSampler(mContext, mConfig))',RESET_PATCH)
        self.assertNotIn('mModule',RESET_PATCH)
        self.assertNotIn('throw ',RESET_PATCH)


if __name__ == "__main__": unittest.main()
