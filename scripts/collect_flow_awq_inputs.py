"""Capture bounded real FFN inputs from 20 calibration texts / two steps.

No held-out inputs or padded rows are used to fit AWQ. Raw sampled rows are
kept only in the local experiment directory, never bundled in the iOS app.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from flow_projection_matrix import verify_corpus
from flow_quant_common import digest
from repack_flow_weights import external_regions, FLOW
from audit_mnn_weight_layout import FlatBuffer
from pc_flow_hift_probe import model_session, put, get


def collect(assets, corpus, output):
    data = verify_corpus(corpus, assets)
    cases = [c for c in data['cases'] if c['split'] == 'calibration']
    output.mkdir(parents=True, exist_ok=False)
    f = FlatBuffer((assets / FLOW).read_bytes())
    regions = external_regions(f, (assets / (FLOW + '.weight')).stat().st_size)
    names = [r['name'] for r in regions if r['kind'] == 9 and '/ff/' in r['name']]
    if len(names) != 44 or len(cases) != 20:
        raise ValueError('Expected pinned 44 FFN linears and 20 calibration cases')
    report = {'state': 'running', 'algorithm_input': 'AWQ real per-linear activation samples',
              'source_hashes': {'graph': digest(assets / FLOW), 'weight': digest(assets / (FLOW + '.weight'))},
              'corpus_sha256': digest(corpus / 'calibration-report.json'),
              'calibration_case_ids': [c['id'] for c in cases], 'validation_used_for_fit': False,
              'rows_per_snapshot': 8, 'snapshots_completed': 0, 'layers': [],
              'coverage': 'one prompt voice; eight stratified valid positions per real step'}
    def save():
        (output / 'awq-inputs-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    captured = {name: [] for name in names}
    n, s = model_session(assets / FLOW, 'high', 'normal')
    last = None
    try:
        for case in cases:
            for snapshot in case['snapshots']:
                with np.load(Path(snapshot['directory']) / 'tensors.npz') as z:
                    inputs = {k: z[k].copy() for k in z.files if k != 'estimator_out'}
                    expected = z['estimator_out'].copy()
                shapes = {k: v.shape for k, v in inputs.items()}
                if shapes != last:
                    for key, v in inputs.items():
                        n.resizeTensor(n.getSessionInput(s, key), v.shape)
                    n.resizeSession(s)
                    last = shapes
                for key, v in inputs.items():
                    put(n, s, key, v)
                seen, errors = set(), []
                def before(tensors, info):
                    name = info.getName()
                    if name not in captured:
                        return True
                    try:
                        if name in seen:
                            raise ValueError('Duplicate FFN callback')
                        seen.add(name)
                        v = np.asarray(tensors[0].getNumpyData(), dtype=np.float32)
                        if v.ndim != 4 or v.shape[0] != case['runtime_frames'] or v.shape[2:] != (1, 1):
                            raise ValueError('Unexpected pointwise layout ' + str(v.shape))
                        v = v.reshape(v.shape[0], v.shape[1])[:case['effective_frames']]
                        indices = np.linspace(0, len(v) - 1, 8, dtype=np.int64)
                        x = v[indices].copy()
                        if not np.isfinite(x).all():
                            raise ValueError('Nonfinite FFN activation')
                        captured[name].append(x)
                        return True
                    except Exception as error:
                        errors.append(str(error))
                        return False
                code = n.runSessionWithCallBackInfo(s, before, lambda tensors, info: True)
                if code or errors or seen != set(names):
                    raise ValueError('Incomplete FFN callback: ' + str(errors))
                actual = get(n, s, 'estimator_out').reshape(expected.shape)
                relative = float(np.linalg.norm((actual - expected).astype(np.float64)) /
                                 max(1e-30, np.linalg.norm(expected.astype(np.float64))))
                if not np.isfinite(actual).all() or relative > 1e-5:
                    raise ValueError('Calibration callback changed reference result')
                report['snapshots_completed'] += 1
                save()
                print('AWQ CAPTURE', case['id'], report['snapshots_completed'], '/40', flush=True)
        for index, name in enumerate(names):
            if len(captured[name]) != 40:
                raise ValueError('Missing real-step activation')
            x = np.concatenate(captured[name], axis=0)
            path = output / f'ffn-{index:02d}.npz'
            np.savez_compressed(path, x=x)
            report['layers'].append({'name': name, 'file': path.name, 'sha256': digest(path),
                                     'samples': len(x), 'input_channels': x.shape[1], 'snapshots': 40})
        report['state'] = 'completed'
    except Exception as error:
        report.update(state='failed', error=str(error))
        save()
        raise
    save()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    collect(a.assets.resolve(), a.corpus.resolve(), a.output.resolve())
