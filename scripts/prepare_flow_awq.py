"""Fit AWQ FFN scales, or package native-IDST W4 plus explicit inverse scales.

This is a disclosed NumPy/MNN adaptation (per-linear reconstruction, optional
activation-aware clipping), not the original LLM AWQ CLI or Q-DiT. Sources and the
successful phone projection-W8 baseline remain untouched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import shutil
import numpy as np
from audit_mnn_weight_layout import FlatBuffer, audit
from flow_quant_common import digest
from repack_flow_weights import (FLOW, external_regions, append_quant_table,
                                clear_table_field)
from flow_awq import search_scale, search_clip, encode_awq_weight
from flow_awq_graph import insert_inverse_scales


def load_original(assets):
    graph = assets / FLOW
    weight = assets / (FLOW + '.weight')
    hashes = {'graph': digest(graph), 'weight': digest(weight)}
    if hashes != {'graph': 'b74ece0a1a2e165eef53dbc8d628310a1b464323d4729959069744baecf418d7',
                  'weight': 'e8d363544063702a9da0800f3a2bbba5669ea719943a5de15a8b08b0f28d24bc'}:
        raise ValueError('Pinned original model identity mismatch')
    f = FlatBuffer(graph.read_bytes())
    regions = external_regions(f, weight.stat().st_size)
    return f, regions, weight, hashes


def fit(assets, inputs, output, block):
    f, regions, path, hashes = load_original(assets)
    capture = json.loads((inputs / 'awq-inputs-report.json').read_text(encoding='utf-8'))
    if (capture.get('state') != 'completed' or capture.get('source_hashes') != hashes
            or capture.get('validation_used_for_fit') is not False
            or len(capture.get('calibration_case_ids', [])) != 20
            or capture.get('snapshots_completed') != 40 or len(capture.get('layers', [])) != 44):
        raise ValueError('Incomplete AWQ calibration capture')
    layers = {row['name']: row for row in capture['layers']}
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'running', 'algorithm': 'AWQ scale search / CosyVoice per-linear NumPy adaptation',
              'source_hashes': hashes, 'block': block, 'clipping': False, 'grid': 20,
              'validation_used_for_fit': False, 'calibration_texts': 20, 'real_steps': 40,
              'inputs_report_sha256': digest(inputs / 'awq-inputs-report.json'),
              'corpus_sha256': capture['corpus_sha256'], 'layers': [],
              'scope': 'Offline CPU calibration; not upstream CLI bit-exact, no phone RTF claim'}
    def save():
        (output / 'awq-search-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        with path.open('rb') as stream:
            for region in regions:
                if region['name'] not in layers:
                    continue
                name = region['name']
                row = layers[name]
                p = inputs / row['file']
                if digest(p) != row['sha256'] or row['samples'] != 320 or row['snapshots'] != 40:
                    raise ValueError('Activation file identity/coverage mismatch')
                with np.load(p) as z:
                    x = z['x'].copy()
                common = f.pointer(region['main'], 0)
                oc, ic = f.scalar(common, 10), f.scalar(common, 11)
                e = region['external']
                stream.seek(e[0])
                w = np.frombuffer(stream.read(e[1]), '<f2').astype(np.float32).reshape(oc, ic)
                scale, result = search_scale(w, x, block)
                filename = f'scale-{len(report["layers"]):02d}.npy'
                np.save(output / filename, scale)
                report['layers'].append({'name': name, 'scale_file': filename,
                    'scale_sha256': digest(output / filename), 'samples': len(x),
                    'input_channels': ic, 'output_channels': oc, **result})
                save()
                print('AWQ FIT', block, len(report['layers']), '/44', name,
                      'ratio', result['ratio'], 'calibration loss ratio',
                      result['awq_calibration_loss'] / max(1e-30, result['unscaled_calibration_loss']), flush=True)
        if len(report['layers']) != 44:
            raise ValueError('Missing FFN search result')
        report['state'] = 'completed'
    except Exception as error:
        report.update(state='failed', error=str(error))
        save()
        raise
    save()
    return report


def build(assets, search, output, base=None, floating=False, unit=False):
    original, original_regions, original_weight, hashes = load_original(assets)
    fitted = json.loads((search / 'awq-search-report.json').read_text(encoding='utf-8'))
    if fitted.get('state') != 'completed' or fitted.get('source_hashes') != hashes or fitted.get('validation_used_for_fit') is not False:
        raise ValueError('Completed uncontaminated search required')
    scales = {}
    for row in fitted['layers']:
        path = search / row['scale_file']
        if digest(path) != row['scale_sha256']:
            raise ValueError('Scale file identity mismatch')
        scales[row['name']] = np.ones(row['input_channels'], np.float32) if unit else np.load(path)
    base = base or assets / FLOW
    base_weight = Path(str(base) + '.weight')
    base_hashes = {'graph': digest(base), 'weight': digest(base_weight)}
    if base_hashes not in [hashes, {
            'graph': '79b0b28e341aabab0198ef60da4402181f1b4dc9c9ff062e20b60d2712221815',
            'weight': '884b43e1fae71b55e2ce31aaf199434da9fb64429047a8eb771e34f903ea9205'}]:
        raise ValueError('Only original or verified phone projection-W8 baseline is allowed')
    f = FlatBuffer(base.read_bytes())
    graph = bytearray(f.data)
    regions = external_regions(f, base_weight.stat().st_size)
    originals = {r['name']: r for r in original_regions}
    output.mkdir(parents=True, exist_ok=False)
    model = output / 'flow.mnn'
    weight = Path(str(model) + '.weight')
    changed = []
    preserved = 0
    with original_weight.open('rb') as src, base_weight.open('rb') as reader, weight.open('xb') as writer:
        for region in regions:
            e = region['external']
            reader.seek(e[0])
            old = reader.read(sum(e[1:]))
            offset = writer.tell()
            name = region['name']
            if name not in scales:
                writer.write(old)
                updated = [offset, *e[1:]]
                preserved += 1
            else:
                r = originals[name]
                oe = r['external']
                common = original.pointer(r['main'], 0)
                oc, ic = original.scalar(common, 10), original.scalar(common, 11)
                src.seek(oe[0])
                raw = src.read(sum(oe[1:]))
                w = np.frombuffer(raw[:oe[1]], '<f2').astype(np.float32).reshape(oc, ic)
                w *= scales[name][None, :]
                row = next(r for r in fitted['layers'] if r['name'] == name)
                if row.get('clip_file') and not unit:
                    clip_path = search / row['clip_file']
                    if floating:
                        raise ValueError('Scale-only control must never apply lossy clipping')
                    if digest(clip_path) != row['clip_sha256']:
                        raise ValueError('Clipping threshold identity mismatch')
                    bound = np.load(clip_path)
                    grouped = w.reshape(oc, -1, fitted['block'])
                    if bound.shape != grouped.shape[:2] + (1,):
                        raise ValueError('Clipping threshold layout mismatch')
                    w = np.clip(grouped, -bound, bound).reshape(oc, ic)
                bias = raw[oe[1]:]
                if floating:
                    payload = w.astype('<f4').tobytes()
                    writer.write(payload)
                    writer.write(bias)
                    updated = [offset, len(payload), len(bias)]
                    clear_table_field(graph, region['main'], 3)
                else:
                    payload, alpha, rows, _ = encode_awq_weight(w, fitted['block'])
                    writer.write(payload)
                    writer.write(alpha)
                    writer.write(bias)
                    updated = [offset, len(payload), len(alpha), len(bias), 0]
                    q = append_quant_table(graph, rows, 4)
                    field = f.field(region['main'], 3)
                    struct.pack_into('<I', graph, field, q - field)
                changed.append(name)
            v = region['vector']
            struct.pack_into('<I', graph, v, len(updated))
            for index, value in enumerate(updated):
                struct.pack_into('<q', graph, v + 4 + index * 8, value)
    if len(changed) != 44:
        raise ValueError('Expected all 44 FFN compensation edges')
    graph = insert_inverse_scales(bytes(graph), scales)
    model.write_bytes(graph)
    report = {'mode': 'awq-scale-only-fp32' if floating else 'unscaled-w4-control' if unit else 'awq-ffn-w4',
              'source_hashes': hashes, 'base_hashes': base_hashes,
              'algorithm': ('Unscaled AWQ-codec W4 control' if unit else
                            'AWQ scaling + activation-aware clipping / CosyVoice adaptation' if fitted.get('clipping') else
                            'AWQ scale-search / CosyVoice per-linear adaptation; clipping off'),
              'clipping': bool(fitted.get('clipping')) and not unit,
              'block': fitted['block'], 'search_report_sha256': digest(search / 'awq-search-report.json'),
              'corpus_sha256': fitted['corpus_sha256'], 'validation_used_for_fit': False,
              'graph_sha256': digest(model), 'weight_sha256': digest(weight),
              'graph_bytes': model.stat().st_size, 'weight_bytes': weight.stat().st_size,
              'inverse_scale_operators': 44, 'changed_ffn_weights': 44,
              'nonselected_regions_copied_exact': preserved, 'selected_biases_copied_exact': 44,
              'layout_audit': audit(model), 'state': 'built-not-accepted',
              'note': 'Explicit input / scale and quantized weight * scale; no silent nonlinear folding; no phone acceptance'}
    if {'graph': digest(base), 'weight': digest(base_weight)} != base_hashes:
        raise ValueError('Baseline files changed')
    (output / 'repack-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def fit_clipping(assets, inputs, search, output):
    f, regions, path, hashes = load_original(assets)
    captured = json.loads((inputs / 'awq-inputs-report.json').read_text(encoding='utf-8'))
    previous = json.loads((search / 'awq-search-report.json').read_text(encoding='utf-8'))
    if (previous.get('state') != 'completed' or previous.get('source_hashes') != hashes
            or captured.get('state') != 'completed' or captured.get('source_hashes') != hashes
            or previous.get('inputs_report_sha256') != digest(inputs / 'awq-inputs-report.json')
            or previous.get('validation_used_for_fit') is not False):
        raise ValueError('Clipping must reuse exactly the original calibration-only capture')
    captures = {r['name']: r for r in captured['layers']}
    region_map = {r['name']: r for r in regions}
    output.mkdir(parents=True, exist_ok=False)
    report = {**previous, 'state': 'running', 'layers': [], 'clipping': True,
              'algorithm': 'AWQ duo scale search + activation-aware group clipping / CosyVoice NumPy adaptation',
              'scale_search_report_sha256': digest(search / 'awq-search-report.json')}
    def save():
        (output / 'awq-search-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        with path.open('rb') as stream:
            for index, oldrow in enumerate(previous['layers']):
                name = oldrow['name']
                cp = inputs / captures[name]['file']
                sp = search / oldrow['scale_file']
                if digest(cp) != captures[name]['sha256'] or digest(sp) != oldrow['scale_sha256']:
                    raise ValueError('Calibration activation/scale identity changed')
                with np.load(cp) as z:
                    x = z['x'].copy()
                scale = np.load(sp)
                r = region_map[name]
                stream.seek(r['external'][0])
                w = np.frombuffer(stream.read(r['external'][1]), '<f2').astype(np.float32)
                w = w.reshape(oldrow['output_channels'], oldrow['input_channels'])
                bound, metrics = search_clip(w * scale[None, :], x / scale[None, :], previous['block'])
                shutil.copyfile(sp, output / oldrow['scale_file'])
                filename = f'clip-{index:02d}.npy'
                np.save(output / filename, bound)
                report['layers'].append({**oldrow, 'clip_file': filename,
                    'clip_sha256': digest(output / filename), 'clip_metrics': metrics})
                save()
                print('AWQ CLIP', index + 1, '/44', 'group loss ratio',
                      metrics['clipped_group_output_loss'] / max(1e-30, metrics['unclipped_group_output_loss']), flush=True)
        report['state'] = 'completed'
    except Exception as error:
        report.update(state='failed', error=str(error)); save(); raise
    save()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('fit', 'clip', 'build'))
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--search', type=Path)
    p.add_argument('--base', type=Path)
    p.add_argument('--block', type=int, choices=(32, 64), default=32)
    p.add_argument('--floating', action='store_true')
    p.add_argument('--unit', action='store_true')
    a = p.parse_args()
    if a.action == 'fit':
        if a.inputs is None:
            p.error('--inputs required for fit')
        fit(a.assets.resolve(), a.inputs.resolve(), a.output.resolve(), a.block)
    elif a.action == 'clip':
        if a.inputs is None or a.search is None:
            p.error('--inputs and --search required for clipping')
        fit_clipping(a.assets.resolve(), a.inputs.resolve(), a.search.resolve(), a.output.resolve())
    else:
        if a.search is None:
            p.error('--search required for build')
        print(json.dumps(build(a.assets.resolve(), a.search.resolve(), a.output.resolve(),
                               a.base.resolve() if a.base else None, a.floating, a.unit), indent=2))
