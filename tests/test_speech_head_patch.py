import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patch_mnn_speech_head import patch_text, SAMPLE_ANCHOR, SAMPLE_PATCH, SEED_ANCHOR, SEED_PATCH


class PatchTests(unittest.TestCase):
    def test_ios_exceptions_disabled_failure_is_not_eos(self):
        self.assertNotIn('throw ',SAMPLE_PATCH)
        self.assertIn('LlmStatus::INTERNAL_ERROR',SAMPLE_PATCH)
        self.assertIn('return 0;',SAMPLE_PATCH)
    def test_exact_anchors(self):
        for anchor, replacement in [(SAMPLE_ANCHOR,SAMPLE_PATCH),(SEED_ANCHOR,SEED_PATCH)]:
            self.assertEqual(patch_text("prefix\n"+anchor+"\nsuffix",anchor,replacement),
                             "prefix\n"+replacement+"\nsuffix")
            for invalid in ["",anchor+anchor,replacement]:
                with self.assertRaises(ValueError): patch_text(invalid,anchor,replacement)


if __name__ == "__main__": unittest.main()
