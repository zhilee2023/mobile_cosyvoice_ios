import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from convert_flow_official_w4 import converter_arguments, validate_payload


class OfficialConverterTests(unittest.TestCase):
    def test_hqq_is_explicit_asymmetric_without_transformer_fusion(self):
        flags = converter_arguments(Path('in.mnn'), Path('out.mnn'), 'hqq', 32)
        self.assertIn('--hqq', flags)
        # The pinned Windows parser uses stream >> bool without boolalpha;
        # spelling this as a bare flag or '=true' silently disables HQQ.
        self.assertIn('--weightQuantAsymmetric=1', flags)
        self.assertEqual(flags[flags.index('--optimizeLevel') + 1], '1')
        self.assertEqual(flags[flags.index('--transformerFuseC4') + 1], '0')
        self.assertNotIn('--fp16', flags)

    def test_minmax_does_not_claim_hqq(self):
        flags = converter_arguments(Path('in.mnn'), Path('out.mnn'), 'minmax', 64)
        self.assertNotIn('--hqq', flags)
        self.assertEqual(flags[flags.index('--weightQuantBlock') + 1], '64')

    def test_empty_external_or_unchanged_fp16_is_not_w4(self):
        for layout, size in [({'convolution_counts': {'quant_type=3,bits=0': 191}}, 100),
                             ({'convolution_counts': {'quant_type=1,bits=4': 191}}, 0)]:
            with self.assertRaises(ValueError):
                validate_payload(layout, 100, size)

    def test_valid_payload_still_requires_later_accuracy_acceptance(self):
        self.assertEqual(validate_payload(
            {'convolution_counts': {'quant_type=1,bits=4': 190}}, 100, 1000), 190)


if __name__ == '__main__':
    unittest.main()
