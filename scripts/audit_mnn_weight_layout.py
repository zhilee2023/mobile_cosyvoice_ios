"""Read-only audit of MNN Convolution2D weight metadata.

Field slots follow MNN 3.6.1 schema/default/{MNN,CaffeOp}.fbs. This does
not load external weights or estimate complete runtime memory allocations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import struct


class FlatBuffer:
    def __init__(self, data: bytes):
        self.data = data

    def number(self, fmt: str, position: int) -> int:
        return struct.unpack_from("<" + fmt, self.data, position)[0]

    def field(self, table: int, slot: int) -> int:
        vtable = table - self.number("i", table)
        offset = 4 + 2 * slot
        if offset >= self.number("H", vtable):
            return 0
        value = self.number("H", vtable + offset)
        return table + value if value else 0

    def scalar(self, table: int, slot: int, default: int = 0, fmt: str = "i") -> int:
        position = self.field(table, slot)
        return self.number(fmt, position) if position else default

    def pointer(self, table: int, slot: int) -> int:
        position = self.field(table, slot)
        return position + self.number("I", position) if position else 0

    def table_vector(self, table: int, slot: int) -> list[int]:
        vector = self.pointer(table, slot)
        if not vector:
            return []
        return [position + self.number("I", position)
                for position in range(vector + 4, vector + 4 + 4 * self.number("I", vector), 4)]

    def int64_vector(self, table: int, slot: int) -> list[int]:
        vector = self.pointer(table, slot)
        return [self.number("q", vector + 4 + 8 * index)
                for index in range(self.number("I", vector))] if vector else []


def audit(path: Path) -> dict:
    buffer = FlatBuffer(path.read_bytes())
    net = buffer.number("I", 0)
    operators = buffer.table_vector(net, 3)
    for subgraph in buffer.table_vector(net, 10):
        operators.extend(buffer.table_vector(subgraph, 4))
    kinds = Counter()
    parameters = Counter()
    external_bytes = Counter()
    for operator in operators:
        if buffer.scalar(operator, 1, fmt="B") != 9:  # OpParameter::Convolution2D
            continue
        convolution = buffer.pointer(operator, 2)
        common = buffer.pointer(convolution, 0)
        quant = buffer.pointer(convolution, 3)
        quant_type = buffer.scalar(quant, 2) if quant else 0
        bits = buffer.scalar(quant, 7) if quant else 0
        kind = f"quant_type={quant_type},bits={bits}"
        group = buffer.scalar(common, 9, 1)
        count = (buffer.scalar(common, 10) * buffer.scalar(common, 11)
                 * buffer.scalar(common, 2, 1) * buffer.scalar(common, 3, 1) // group)
        kinds[kind] += 1
        parameters[kind] += count
        external = buffer.int64_vector(convolution, 6)
        external_bytes[kind] += external[1] if len(external) >= 2 else 0
    return {"path": str(path.resolve()), "graph_bytes": path.stat().st_size,
            "operators": len(operators), "convolution_counts": dict(kinds),
            "convolution_weight_values": dict(parameters),
            "external_convolution_weight_bytes": dict(external_bytes),
            "convolution_weights_if_all_fp32_bytes": 4 * sum(parameters.values()),
            "scope": "Metadata only; excludes activations, padding, staging, aliases, and non-convolution weights"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.models], indent=2))
