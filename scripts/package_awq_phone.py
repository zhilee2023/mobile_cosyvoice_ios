"""Package the explicitly user-listening-reviewed AWQ experimental pair.

Preserves failed numerical gates and baseline identity. This is a human
decision for a phone trial, not an automatic numerical-quality promotion.
"""
import argparse
import json
from pathlib import Path
import shutil
from flow_quant_common import digest


def package(root, output, listening_reviewed=False):
    if not listening_reviewed:
        raise ValueError('Explicit user listening review required for this experimental candidate')
    report = json.loads((root / 'cosyvoice-awq-report.json').read_text(encoding='utf-8'))
    key = 'awq-ffn-g32-clipped'
    profile = report['profiles'][key]
    recipe = profile['recipe']
    if (report['state'] != 'completed' or not profile['identity_verified']
            or not profile['independent_audit']['passed'] or recipe['block'] != 32
            or recipe.get('clipping') is not True or recipe['inverse_scale_operators'] != 44
            or profile['evaluations']['low']['state'] != 'completed'
            or profile['evaluations']['low']['calibration']['cases'] != 20
            or profile['evaluations']['low']['validation']['cases'] != 5):
        raise ValueError('Complete experimental identity/integrity/replay evidence required')
    model = root / key / 'flow.mnn'
    weights = Path(str(model) + '.weight')
    if digest(model) != recipe['graph_sha256'] or digest(weights) != recipe['weight_sha256']:
        raise ValueError('AWQ candidate changed')
    listened = next(a for a in report['audio'] if a['case'] == 'zh-heldout')
    if digest(Path(listened['candidate'])) != listened['sha256']:
        raise ValueError('Listened candidate audio changed')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source, suffix in [('candidateGraph', model, ''), ('candidateWeight', weights, '.weight')]:
        target = output / ('flow-awq-ffn-w4-g32-clipped.mnn' + suffix)
        shutil.copyfile(source, target)
        if digest(target) != digest(source):
            raise ValueError('Import copy identity mismatch')
        files[name] = {'file': target.name, 'bytes': target.stat().st_size, 'sha256': digest(target)}
    manifest = {'schemaVersion': 1, 'recipe': 'cosyvoice3-flow-awq-ffn-w4-g32-clipped-v1',
        'candidateMode': key, 'files': files, 'byteOrder': 'little-endian',
        'awqAdaptation': 'CosyVoice per-linear duo scaling + activation-aware clipping; native IDST W4; not upstream CLI bit-exact',
        'ffnW4Operators': 44, 'projectionW8Operators': 66, 'preservedFP16Operators': 81,
        'inverseScaleOperators': 44, 'baseHashes': recipe['base_hashes'],
        'originalSourceHashes': recipe['source_hashes'],
        'pcReportSHA256': digest(root / 'cosyvoice-awq-report.json'),
        'independentAuditSHA256': digest(root / key / 'independent-audit.json'),
        'pcNumericScreenPassed': bool(profile['low_quality_screen_pass']),
        'pcScreen': profile['evaluations']['low'], 'unchangedNumericGates': report['gates_unchanged'],
        'userListeningReviewed': True, 'listenedCase': 'zh-heldout', 'listenedAudioSHA256': listened['sha256'],
        'listeningScope': 'User accepted this 4.24-second PC clip with High HiFT; not iPhone HiFT Low or broad listening acceptance',
        'phoneValidated': False, 'automaticPromotion': False, 'experimentalOnly': True,
        'scope': 'Explicit opt-in CPU High/Memory Low two-step Flow, pruned-head batch LLM and bounded HiFT Low reuse. W8 baseline preserved.'}
    (output / 'flow-awq-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--user-listening-reviewed', action='store_true')
    a = p.parse_args()
    m = package(a.root.resolve(), a.output.resolve(), a.user_listening_reviewed)
    print(json.dumps(m, ensure_ascii=False, indent=2))
