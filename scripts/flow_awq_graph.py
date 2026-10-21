"""Insert explicit inverse-channel Scale ops using pinned MNN 3.6.1 schema.

No scale is silently pushed through GELU/AdaLN/residual branches. Only each
selected linear input edge is rerouted. Other consumers retain the old edge.
New root/op/name vectors precede a relocated unchanged original buffer so
all unsigned FlatBuffer offsets remain forward; source subgraphs rejected.
"""
from __future__ import annotations
import struct
import numpy as np
from audit_mnn_weight_layout import FlatBuffer


def align(data, unit=4):
    data.extend(bytes((-len(data)) % unit))


def table(data, count, values):
    """Four-byte field cells, including padded schema byte fields."""
    align(data)
    vt = len(data)
    offsets = [4 + 4 * i if i in values else 0 for i in range(count)]
    data.extend(struct.pack('<HH' + 'H' * count, 4 + 2 * count, 4 + 4 * count, *offsets))
    align(data)
    obj = len(data)
    data.extend(bytes(4 + 4 * count))
    struct.pack_into('<i', data, obj, obj - vt)
    fields = {}
    for slot, (fmt, value) in values.items():
        fields[slot] = obj + 4 + 4 * slot
        struct.pack_into('<' + fmt, data, fields[slot], value)
    return obj, fields


def vector(data, fmt, values):
    align(data)
    pos = len(data)
    data.extend(struct.pack('<I', len(values)))
    if len(values):
        data.extend(struct.pack('<' + fmt * len(values), *values))
    return pos


def string(data, value):
    align(data)
    pos = len(data)
    raw = value.encode('utf-8')
    data.extend(struct.pack('<I', len(raw)) + raw + b'\0')
    return pos


def link(data, field, target):
    if not field or target <= field:
        raise ValueError('FlatBuffer offset must be forward and non-null')
    struct.pack_into('<I', data, field, target - field)


def insert_inverse_scales(original: bytes, scales: dict[str, np.ndarray]) -> bytes:
    f = FlatBuffer(original)
    oldroot = f.number('I', 0)
    if f.table_vector(oldroot, 10):
        raise ValueError('Only reviewed top-level Flow graph supported')
    oldops = f.table_vector(oldroot, 3)
    oldnames = f.pointer(oldroot, 7)
    count_names = f.number('I', oldnames)
    names = {}
    for op in oldops:
        p = f.pointer(op, 3)
        name = original[p + 4:p + 4 + f.number('I', p)].decode() if p else ''
        if name in scales:
            if name in names or f.scalar(op, 1, fmt='B') != 9:
                raise ValueError('Expected unique convolution name for AWQ')
            main = f.pointer(op, 2)
            common = f.pointer(main, 0)
            if f.scalar(common, 2, 1) != 1 or f.scalar(common, 3, 1) != 1 or f.scalar(common, 9, 1) != 1:
                raise ValueError('Scale insertion restricted to pointwise dense FFN')
            scale = np.asarray(scales[name], np.float32)
            if (scale.shape != (f.scalar(common, 11),) or len(scale) % 4
                    or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError('Invalid per-input-channel AWQ scale')
            names[name] = op
    if set(names) != set(scales) or not names:
        raise ValueError('Scale names do not match graph')
    # Net pointer slots from schema/default/MNN.fbs; scalar slots are bytes
    # except tensorNumber. Preserve all existing metadata pointers.
    values = {slot: ('I', 0) for slot in (0, 1, 2, 3, 4, 7, 10, 11) if f.field(oldroot, slot)}
    values.update({5: ('B', f.scalar(oldroot, 5, fmt='B')),
                   6: ('B', f.scalar(oldroot, 6, fmt='B')),
                   8: ('I', count_names + len(names)),
                   9: ('B', f.scalar(oldroot, 9, fmt='B'))})
    data = bytearray(bytes(4))
    newroot, fields = table(data, 12, values)
    opvec = vector(data, 'I', [0] * (len(oldops) + len(names)))
    namevec = vector(data, 'I', [0] * (count_names + len(names)))
    align(data, 8)
    shift = len(data)
    data.extend(original)
    struct.pack_into('<I', data, 0, newroot)
    for slot in values:
        if slot in (5, 6, 8, 9):
            continue
        target = opvec if slot == 3 else namevec if slot == 7 else shift + f.pointer(oldroot, slot)
        link(data, fields[slot], target)
    for index in range(count_names):
        oldfield = oldnames + 4 + 4 * index
        link(data, namevec + 4 + 4 * index, shift + oldfield + f.number('I', oldfield))
    selected = {op: name for name, op in names.items()}
    index, added = 0, 0
    for op in oldops:
        if op in selected:
            name = selected[op]
            scale = np.asarray(scales[name], np.float32)
            original_inputs = f.pointer(op, 0)
            if f.number('I', original_inputs) != 1:
                raise ValueError('FFN must have exactly one activation input')
            old_input = f.number('i', original_inputs + 4)
            new_id = count_names + added
            scaled_name = '__awq_inverse_scale_' + str(added)
            # OpParameter::Scale=58, OpType::Scale=77, NCHW=0.
            newop, of = table(data, 7, {0: ('I', 0), 1: ('B', 58), 2: ('I', 0),
                                       3: ('I', 0), 4: ('I', 0), 5: ('i', 77), 6: ('B', 0)})
            params, pf = table(data, 4, {0: ('i', len(scale)), 1: ('I', 0), 2: ('I', 0)})
            link(data, of[2], params)
            link(data, pf[1], vector(data, 'f', (1 / scale).astype(np.float32).tolist()))
            link(data, pf[2], vector(data, 'f', [0.] * len(scale)))
            link(data, of[0], vector(data, 'i', [old_input]))
            link(data, of[4], vector(data, 'i', [new_id]))
            text = string(data, scaled_name)
            link(data, of[3], text)
            link(data, namevec + 4 + 4 * new_id, text)
            link(data, shift + f.field(op, 0), vector(data, 'i', [new_id]))
            link(data, opvec + 4 + 4 * index, newop)
            index += 1
            added += 1
        link(data, opvec + 4 + 4 * index, shift + op)
        index += 1
    if index != len(oldops) + len(names):
        raise ValueError('Incomplete op vector construction')
    return bytes(data)
