"""Run the installed official MNN converter, isolated from inference imports.

Requires an audited, exactly materialized FP32 Flow input. Conversion is NOT
quality acceptance, nor evidence of iPhone integer-kernel speed or memory use.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from audit_mnn_weight_layout import audit
from flow_quant_common import digest


def converter_arguments(source: Path, output: Path, method: str, block: int) -> list[str]:
    if method not in ('minmax', 'hqq') or block not in (32, 64, 128):
        raise ValueError('Unknown official W4 recipe')
    arguments = ['MNNConvert', '-f', 'MNN', '--modelFile', str(source),
                 '--MNNModel', str(output), '--optimizeLevel', '1',
                 '--transformerFuseC4', '0',
                 '--weightQuantBits', '4', '--weightQuantBlock', str(block),
                 '--weightQuantAsymmetric=1', '--saveExternalData=1']
    if method == 'hqq':
        arguments.append('--hqq')
    return arguments


def validate_payload(layout: dict, graph_bytes: int, weight_bytes: int) -> int:
    count = sum(value for key, value in layout['convolution_counts'].items()
                if key == 'quant_type=1,bits=4')
    if count == 0 or graph_bytes <= 0 or weight_bytes <= 0:
        raise ValueError('No verified external W4 convolution payload; converter success alone is insufficient')
    return count


def convert(source: Path, output: Path, method: str, block: int) -> dict:
    if importlib.metadata.version('MNN') != '3.6.1':
        raise ValueError('This experiment is pinned to MNN 3.6.1')
    source_weight = Path(str(source) + '.weight')
    provenance = json.loads((source.parent / 'repack-report.json').read_text(encoding='utf-8'))
    if provenance.get('mode') != 'fp32' or provenance.get('quantized_operator_count') != 0:
        raise ValueError('Expected the audited FP16-to-exact-FP32 materialization')
    before = {'graph': digest(source), 'weight': digest(source_weight)}
    if before != {'graph': provenance['graph_sha256'], 'weight': provenance['weight_sha256']}:
        raise ValueError('Materialized input hashes do not match provenance')
    output.mkdir(parents=True, exist_ok=False)
    model = output / 'flow.mnn'
    arguments = converter_arguments(source, model, method, block)
    # Only the installed upstream converter handles quantization here. No
    # in-house RTN/HQQ implementation is used by this experiment.
    command = [sys.executable, '-c',
               'import sys, _tools; _tools.mnnconvert(sys.argv[1:])', *arguments]
    report = {'state': 'running', 'converter': 'official installed MNN 3.6.1 _tools',
              'arguments': arguments, 'method': method, 'block': block,
              'source_hashes': before, 'original_source_hashes': provenance['source_hashes'],
              'scope': 'Conversion only; no accuracy, listening or phone acceptance',
              'source': str(source), 'output': str(model)}
    report_path = output / 'official-conversion-report.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        # MNN 3.6.1 addBizCode only unpacks the MNN graph; postTreat reads
        # external offsets from this literal scratch filename, not model.weight.
        # Stage a verified copy in a unique working directory. The converter
        # removes its own scratch file; never run against source weights.
        scratch = output / 'converter-work'
        scratch.mkdir(exist_ok=False)
        scratch_weight = scratch / '.__convert_external_data.bin'
        shutil.copyfile(source_weight, scratch_weight)
        if digest(scratch_weight) != before['weight']:
            raise ValueError('Converter scratch copy hash mismatch')
        with (output / 'converter.log').open('xb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=900, check=False, cwd=scratch)
        report['converter_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Converter exited with {result.returncode}; see converter.log')
        log = (output / 'converter.log').read_text(encoding='utf-8', errors='replace')
        if "Can't open file:" in log or 'Disable Hqq currently' in log:
            raise ValueError('Converter reported missing input or disabled HQQ; reject candidate')
        if method == 'hqq' and 'Use HQQ to quant weight' not in log:
            raise ValueError('No official HQQ activation confirmation in converter log')
        weight = Path(str(model) + '.weight')
        report['layout'] = audit(model)
        report['graph_bytes'] = model.stat().st_size
        report['weight_bytes'] = weight.stat().st_size
        report['verified_w4_convolutions'] = validate_payload(
            report['layout'], report['graph_bytes'], report['weight_bytes'])
        report['graph_sha256'] = digest(model)
        report['weight_sha256'] = digest(weight)
        if before != {'graph': digest(source), 'weight': digest(source_weight)}:
            raise ValueError('Source input changed during conversion')
        report['operator_count_changed'] = (
            report['layout']['operators'] != provenance['layout_audit']['operators'])
        report['source_unchanged'] = True
        report['state'] = 'converted-not-evaluated'
    except Exception as error:
        report.update(state='failed', error=str(error))
        save()
        raise
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('minmax', 'hqq'), default='hqq')
    parser.add_argument('--block', type=int, choices=(32, 64, 128), default=32)
    args = parser.parse_args()
    print(json.dumps(convert(args.source.resolve(), args.output.resolve(),
                             args.method, args.block), indent=2))
