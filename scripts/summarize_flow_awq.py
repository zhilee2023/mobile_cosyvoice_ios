"""One local AWQ report: frozen split, integrity, quality gates, optional WAVs.

Not a phone benchmark or perceptual acceptance. Failed profiles stay local.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import wave
from flow_projection_matrix import GATES
from flow_quant_common import digest

PROFILES = ('unit-ffn-g32', 'awq-ffn-g32', 'unit-ffn-g64', 'awq-ffn-g64', 'awq-ffn-g32-clipped')


def subset(rows, split):
    rows = [r for r in rows if r['split'] == split]
    if not rows:
        return None
    result = {'cases': len(rows),
              'max_teacher_velocity_l2': max(m['relative_l2'] for r in rows for m in r['teacher_velocity']),
              'max_closed_loop_mel_l2': max(r['closed_loop_mel']['relative_l2'] for r in rows),
              'mean_closed_loop_mel_l2': statistics.mean(r['closed_loop_mel']['relative_l2'] for r in rows)}
    result['screen_pass'] = len(rows) == (20 if split == 'calibration' else 5) and all(result[k] <= v for k, v in GATES.items())
    return result


def summarize(root, corpus, assets, audio=False):
    capture = json.loads((root / 'calibration-inputs' / 'awq-inputs-report.json').read_text(encoding='utf-8'))
    report = {'state': 'running', 'scope': 'Windows CPU offline AWQ sensitivity, not phone RTF/MOS',
              'algorithm': 'AWQ duo scaling and optional activation-aware clipping, per-linear CosyVoice NumPy adaptation',
              'upstream_cli_bit_exact': False, 'calibration_texts': 20, 'validation_texts': 5,
              'real_calibration_steps': capture['snapshots_completed'], 'rows_per_ffn': 320,
              'validation_used_for_fit': False, 'gates_unchanged': GATES,
              'source_hashes': capture['source_hashes'], 'corpus_sha256': capture['corpus_sha256'],
              'limitations': ['One prompt voice', 'Repeated screening corpus, not final independent acceptance',
                             'Explicit inverse Scale operators add runtime work',
                             'Memory Low may dynamically quantize activations on eligible CPU kernels; not claimed W4A16',
                             'Concurrent PC test timings are not speed rankings'],
              'profiles': {}, 'audio': [], 'phone_deployment': False}
    control = json.loads((root / 'scale-only-validation-low' / 'report.json').read_text(encoding='utf-8'))
    report['float_compensation_control'] = {'state': control['state'], **control['summary']}
    compensation_ok = control['state'] == 'completed' and control['summary']['cases'] == 5 and control['summary']['max_teacher_velocity_l2'] <= 1e-4
    report['float_compensation_control']['passed_1e_minus4_teacher_gate'] = compensation_ok
    incomplete = False
    for key in PROFILES:
        recipe = json.loads((root / key / 'repack-report.json').read_text(encoding='utf-8'))
        model = root / key / 'flow.mnn'
        identity_ok = digest(model) == recipe['graph_sha256'] and digest(Path(str(model) + '.weight')) == recipe['weight_sha256']
        entry = {'recipe': recipe, 'identity_verified': identity_ok, 'evaluations': {}}
        audit_path = root / key / 'independent-audit.json'
        entry['independent_audit'] = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else None
        for memory in ('low', 'normal'):
            path = root / (key + '-' + memory) / 'report.json'
            if not path.is_file():
                if memory == 'low' or key.startswith('awq-'):
                    incomplete = True
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            entry['evaluations'][memory] = {'state': data['state'], 'memory_mode': data['memory_mode'],
                'session_config': data['session_config'], 'calibration': subset(data['cases'], 'calibration'),
                'validation': subset(data['cases'], 'validation')}
            if data['state'] != 'completed':
                incomplete = True
        low = entry['evaluations'].get('low', {})
        entry['low_quality_screen_pass'] = bool(identity_ok and compensation_ok and
            entry['independent_audit'] and entry['independent_audit'].get('passed') and
            low.get('state') == 'completed' and (low.get('calibration') or {}).get('screen_pass') and
            (low.get('validation') or {}).get('screen_pass'))
        report['profiles'][key] = entry
    if incomplete:
        report['state'] = 'incomplete'
    else:
        report['state'] = 'completed'
    if audio:
        if incomplete:
            raise ValueError('Wait for all required evaluations before rendering audio')
        import numpy as np
        from pc_flow_hift_probe import hift
        fixtures = json.loads((corpus / 'calibration-report.json').read_text(encoding='utf-8'))
        key = 'awq-ffn-g32-clipped'
        folder = root / 'listening'
        folder.mkdir(exist_ok=True)
        for case in fixtures['cases']:
            if case['split'] != 'validation':
                continue
            path = folder / (case['id'] + '-awq-g32-clipped.wav')
            if path.exists():
                raise FileExistsError('Do not silently overwrite an existing listening fixture')
            mel = np.load(root / (key + '-low') / (case['id'] + '-mel.npy'))
            pcm, _ = hift(assets, mel, core_precision='high', core_memory='normal')
            if not np.isfinite(pcm).all():
                raise ValueError('Nonfinite candidate audio')
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
                stream.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
            report['audio'].append({'case': case['id'], 'candidate': str(path), 'sha256': digest(path),
                'reference_original_fp16': case['audio'], 'audio_seconds': len(pcm) / 24000,
                'scope': 'Same High HiFT as reference to isolate Flow; not iPhone HiFT Low/MOS acceptance'})
    path = root / 'cosyvoice-awq-report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'state': report['state'], 'report': str(path), 'low_validation': {
        key: entry['evaluations'].get('low', {}).get('validation') for key, entry in report['profiles'].items()}}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--audio', action='store_true')
    a = p.parse_args()
    summarize(a.root.resolve(), a.corpus.resolve(), a.assets.resolve(), a.audio)
