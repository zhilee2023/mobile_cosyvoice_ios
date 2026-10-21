import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_mnn_weight_layout import FlatBuffer
from flow_awq import search_scale, search_clip, encode_awq_weight, quantize_awq_weight
from flow_awq_graph import table, vector, link, string, insert_inverse_scales
from summarize_flow_awq import subset


def toy_graph(w):
    data = bytearray(bytes(4))
    root, rf = table(data, 12, {3: ('I', 0), 7: ('I', 0), 8: ('i', 2)})
    ops = vector(data, 'I', [0, 0])
    names = vector(data, 'I', [0, 0])
    link(data, rf[3], ops)
    link(data, rf[7], names)
    for index, text in enumerate(('x', 'y')):
        link(data, names + 4 + 4 * index, string(data, text))
    inp, inf = table(data, 7, {1: ('B', 21), 2: ('I', 0), 3: ('I', 0),
                              4: ('I', 0), 5: ('i', 34), 6: ('B', 0)})
    param, pf = table(data, 3, {0: ('I', 0), 1: ('i', 1), 2: ('B', 0)})
    link(data, inf[2], param)
    link(data, pf[0], vector(data, 'i', [1, w.shape[1], 1, 1]))
    link(data, inf[3], string(data, 'x'))
    link(data, inf[4], vector(data, 'i', [0]))
    link(data, ops + 4, inp)
    conv, cf = table(data, 7, {0: ('I', 0), 1: ('B', 9), 2: ('I', 0),
                               3: ('I', 0), 4: ('I', 0), 5: ('i', 12), 6: ('B', 0)})
    cp, cpf = table(data, 7, {0: ('I', 0), 1: ('I', 0), 2: ('I', 0)})
    common, _ = table(data, 17, {2: ('i', 1), 3: ('i', 1), 9: ('i', 1),
                                  10: ('i', w.shape[0]), 11: ('i', w.shape[1])})
    link(data, cf[2], cp)
    link(data, cpf[0], common)
    link(data, cpf[1], vector(data, 'f', w.ravel().tolist()))
    link(data, cpf[2], vector(data, 'f', [0.] * w.shape[0]))
    link(data, cf[0], vector(data, 'i', [0]))
    link(data, cf[4], vector(data, 'i', [1]))
    link(data, cf[3], string(data, 'linear'))
    link(data, ops + 8, conv)
    import struct
    struct.pack_into('<I', data, 0, root)
    return bytes(data)


class AWQTests(unittest.TestCase):
    def test_quality_screen_keeps_original_gates_and_requires_complete_split(self):
        row = {'split': 'validation', 'teacher_velocity': [{'relative_l2': .01}, {'relative_l2': .02}],
               'closed_loop_mel': {'relative_l2': .01}}
        self.assertFalse(subset([row] * 4, 'validation')['screen_pass'])
        self.assertTrue(subset([row] * 5, 'validation')['screen_pass'])
        bad = {**row, 'closed_loop_mel': {'relative_l2': .04}}
        self.assertFalse(subset([bad] * 5, 'validation')['screen_pass'])

    def test_native_codec_matches_searched_dequantizer(self):
        w = np.random.default_rng(17).normal(size=(8, 64)).astype(np.float32)
        payload, alpha, rows, expected = encode_awq_weight(w, 32)
        packed = np.frombuffer(payload[26:], np.uint8)
        q = np.stack((packed >> 4, packed & 15), axis=1).reshape(rows, 32).astype(np.float32) - 8
        a = np.frombuffer(alpha, '<f4').reshape(rows, 2)
        actual = q * a[:, 1, None] + (a[:, 0] + 8 * a[:, 1])[:, None]
        np.testing.assert_array_equal(actual.reshape(w.shape), expected)

    def test_search_never_worse_than_its_unscaled_calibration_control(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=(16, 64)).astype(np.float32)
        x = rng.normal(size=(40, 64)).astype(np.float32)
        x[:, 0] *= 30
        scale, result = search_scale(w, x, 32)
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue((scale > 0).all())
        self.assertLessEqual(result['awq_calibration_loss'], result['unscaled_calibration_loss'])
        self.assertEqual(len(result['search']), 21)
        self.assertFalse(result['clipping'])

    def test_graph_metadata_and_input_validation(self):
        source = toy_graph(np.ones((4, 4), np.float32))
        result = insert_inverse_scales(source, {'linear': np.array([.5, 3, 1, 2], np.float32)})
        f = FlatBuffer(result)
        root = f.number('I', 0)
        self.assertEqual(len(f.table_vector(root, 3)), 3)
        self.assertEqual(f.scalar(root, 8), 3)
        for scales in ({'wrong': np.ones(4)}, {'linear': np.array([1, 0, 1, 1])}):
            with self.assertRaises(ValueError):
                insert_inverse_scales(source, scales)

    def test_clip_covariance_matches_direct_activation_output_objective(self):
        rng = np.random.default_rng(8)
        w = rng.normal(size=(8, 64)).astype(np.float32)
        x = rng.normal(size=(40, 64)).astype(np.float32)
        bound, metrics = search_clip(w, x, 32)
        clipped = np.clip(w.reshape(8, 2, 32), -bound, bound).reshape(8, 64)
        d = (quantize_awq_weight(clipped, 32)[0] - w).reshape(8, 2, 32)
        direct = np.einsum('ngb,ogb->ong', x.reshape(40, 2, 32).astype(np.float64), d.astype(np.float64))
        self.assertAlmostEqual(metrics['clipped_group_output_loss'], float(np.mean(direct ** 2)), places=10)
        self.assertLessEqual(metrics['clipped_group_output_loss'], metrics['unclipped_group_output_loss'])

    @unittest.skipUnless(importlib.util.find_spec('MNN'), 'MNN runtime is optional for unit tests')
    def test_native_scale_compensation_preserves_linear_output(self):
        import MNN
        from pc_flow_hift_probe import put, get
        w = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        scale = np.array([.5, 3, 1, 2], np.float32)
        # Tiny schema fixture checks one batch; multi-row Flow compensation
        # is independently checked against all real held-out trajectories.
        x = np.array([[1, 2, 3, 4]], np.float32).reshape(1, 4, 1, 1)
        graph = insert_inverse_scales(toy_graph(w * scale[None, :]), {'linear': scale})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            p = Path(folder) / 'toy.mnn'
            p.write_bytes(graph)
            n = MNN.Interpreter(str(p))
            s = n.createSession({'backend': 'CPU', 'precision': 1, 'memory': 2, 'numThread': 4})
            put(n, s, 'x', x)
            self.assertEqual(n.runSession(s), 0)
            target = n.getSessionOutput(s, 'y')
            host = MNN.Tensor((1, 4, 1, 1), MNN.Halide_Type_Float,
                              np.zeros((1, 4, 1, 1), np.float32), MNN.Tensor_DimensionType_Caffe)
            self.assertTrue(target.copyToHostTensor(host))
            y = np.asarray(host.getNumpyData()).reshape(1, 4)
            np.testing.assert_allclose(y, x.reshape(1, 4) @ w.T, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
