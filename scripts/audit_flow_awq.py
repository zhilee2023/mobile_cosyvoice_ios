"""Independent byte/edge/scale audit of prepared AWQ Flow candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_mnn_weight_layout import FlatBuffer
from repack_flow_weights import external_regions
from flow_quant_common import digest


def audit_candidate(model, base, search):
    report_path = model.parent / 'repack-report.json'
    recipe = json.loads(report_path.read_text(encoding='utf-8'))
    fitted = json.loads((search / 'awq-search-report.json').read_text(encoding='utf-8'))
    ff = FlatBuffer(base.read_bytes())
    cf = FlatBuffer(model.read_bytes())
    bw, cw = Path(str(base) + '.weight'), Path(str(model) + '.weight')
    old = external_regions(ff, bw.stat().st_size)
    new = external_regions(cf, cw.stat().st_size)
    if [(r['kind'], r['name']) for r in old] != [(r['kind'], r['name']) for r in new]:
        raise ValueError('External parameter region identity changed')
    fits = {r['name']: r for r in fitted['layers']}
    preserved, biases = 0, 0
    with bw.open('rb') as src, cw.open('rb') as dst:
        for x, y in zip(old, new):
            oe, ne = x['external'], y['external']
            src.seek(oe[0]); dst.seek(ne[0])
            a, b = src.read(sum(oe[1:])), dst.read(sum(ne[1:]))
            if x['name'] not in fits:
                if a != b:
                    raise ValueError('Nonselected region changed: ' + x['name'])
                preserved += 1
            else:
                if a[oe[1] + oe[2]:oe[1] + oe[2] + oe[3]] != b[ne[1] + ne[2]:ne[1] + ne[2] + ne[3]]:
                    raise ValueError('Selected bias changed')
                q = cf.pointer(y['main'], 3)
                if cf.scalar(q, 2) != 1 or cf.scalar(q, 7) != 4:
                    raise ValueError('Selected FFN is not native W4')
                biases += 1
    def ops(f):
        result = {}
        for p in f.table_vector(f.number('I', 0), 3):
            s = f.pointer(p, 3)
            name = f.data[s + 4:s + 4 + f.number('I', s)].decode()
            if name in result and (name in fits or name.startswith('__awq_inverse_scale_')):
                raise ValueError('Duplicate operator name')
            result[name] = p
        return result
    oldops, newops = ops(ff), ops(cf)
    scales = {name: p for name, p in newops.items() if name.startswith('__awq_inverse_scale_')}
    if len(scales) != 44 or len(newops) != len(oldops) + 44:
        raise ValueError('Unexpected compensation operator count')
    old_order = ff.table_vector(ff.number('I', 0), 3)
    new_order = [p for p in cf.table_vector(cf.number('I', 0), 3) if p not in scales.values()]
    if len(old_order) != len(new_order):
        raise ValueError('Original operator order/count changed')
    def ints(f, op, slot):
        v = f.pointer(op, slot)
        return [f.number('i', v + 4 + 4 * i) for i in range(f.number('I', v))] if v else []
    for o, n in zip(old_order, new_order):
        s = ff.pointer(o, 3)
        name = ff.data[s + 4:s + 4 + ff.number('I', s)].decode()
        if ff.scalar(o, 5) != cf.scalar(n, 5) or ints(ff, o, 4) != ints(cf, n, 4):
            raise ValueError('Original op type/output edge changed')
        if name not in fits and ints(ff, o, 0) != ints(cf, n, 0):
            raise ValueError('Unselected consumer input edge changed')
    # Check each inserted op is the sole producer of the rerouted input, with
    # the original input edge and correct inverse learned channel scales.
    producers = {}
    for p in scales.values():
        v = cf.pointer(p, 4)
        if cf.number('I', v) != 1 or cf.scalar(p, 5) != 77 or cf.scalar(p, 1, fmt='B') != 58:
            raise ValueError('Expected native Scale operation')
        producers[cf.number('i', v + 4)] = p
    for name, row in fits.items():
        o, n = oldops[name], newops[name]
        p = producers[cf.number('i', cf.pointer(n, 0) + 4)]
        if cf.number('i', cf.pointer(p, 0) + 4) != ff.number('i', ff.pointer(o, 0) + 4):
            raise ValueError('Compensation input edge changed')
        params = cf.pointer(p, 2)
        v = cf.pointer(params, 1)
        count = cf.number('I', v)
        value = np.frombuffer(cf.data, '<f4', count=count, offset=v + 4)
        learned = np.load(search / row['scale_file'])
        expected = np.ones_like(learned) if recipe['mode'] == 'unscaled-w4-control' else 1 / learned
        np.testing.assert_array_equal(value, expected)
    result = {'passed': True, 'nonselected_regions_exact': preserved,
              'selected_biases_exact': biases, 'native_w4_ffn_ops': biases,
              'inverse_scales_and_edges_verified': 44,
              'candidate_graph_sha256': digest(model), 'candidate_weight_sha256': digest(cw)}
    (model.parent / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--search', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit_candidate(a.model.resolve(), a.base.resolve(), a.search.resolve()), indent=2))
