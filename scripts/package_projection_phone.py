"""Package one numerically screened projection candidate; never promote defaults."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from flow_quant_common import digest
from summarize_projection_matrix import summarize

def package(root,mode,base_manifest,output):
    report=json.loads((root/'projection-report.json').read_text(encoding='utf-8'))
    reviewed=next((r for r in summarize(report)['recipes'] if r['mode']==mode),None)
    if not reviewed or not reviewed['eligible_for_phone_review']:raise ValueError('Complete calibration/validation/repeat screening required')
    audit=json.loads((root/'layout-integrity.json').read_text(encoding='utf-8'))
    if audit.get('state')!='completed' or not audit.get('original_assets_unchanged') or mode not in audit.get('profiles',{}):raise ValueError('Bit-exact preservation audit required')
    base=json.loads(base_manifest.read_text(encoding='utf-8'))
    if base['recipe']!='cosyvoice3-flow-ffn-w8-g64-v1' or len(base['cases'])!=5:raise ValueError('Unknown phone fixture format')
    control=report['profiles']['ffn-w8'];candidate=report['profiles'][mode]
    for p in [control,candidate]:
        graph=Path(p['model']);weight=Path(str(graph)+'.weight');r=p['recipe']
        if digest(graph)!=r['graph_sha256'] or digest(weight)!=r['weight_sha256']:raise ValueError('Candidate/control changed')
    for key,sha_key in [('candidateGraph','graph_sha256'),('candidateWeight','weight_sha256')]:
        if base['files'][key]['sha256']!=control['recipe'][sha_key]:raise ValueError('Phone FFN control differs')
    output.mkdir(parents=True,exist_ok=False)
    files={'sourceGraph':copy.deepcopy(base['files']['candidateGraph']),'sourceWeight':copy.deepcopy(base['files']['candidateWeight']),
           'fixtures':copy.deepcopy(base['files']['fixtures'])}
    for key,suffix in [('candidateGraph',''),('candidateWeight','.weight')]:
        source=Path(candidate['model']+suffix);target=output/('flow-projection-'+mode+'-g64.mnn'+suffix)
        shutil.copyfile(source,target)
        files[key]={'file':target.name,'bytes':target.stat().st_size,'sha256':digest(target)}
    manifest={'schemaVersion':1,'recipe':'cosyvoice3-flow-projection-w8-g64-v1','candidateMode':mode,
        'byteOrder':'little-endian','files':files,'cases':copy.deepcopy(base['cases']),
        'quantizedOperators':candidate['recipe']['quantized_operator_count'],
        'preservedFP16Operators':191-candidate['recipe']['quantized_operator_count'],
        'pcScreen':reviewed,'pcReportSHA256':digest(root/'projection-report.json'),'auditSHA256':digest(root/'layout-integrity.json'),
        'automaticPromotion':False,'listeningValidated':False,
        'scope':'Independent CPU High/Memory Low five-fixture phone module trial; no LLM/HiFT or whole-chain RTF; original FFN control preserved',
        'gates':{'pcOriginalMelRelativeL2':.03,'pairedPhoneFFNMelRelativeL2':.03,'repeatRelativeL2':1e-5}}
    (output/'flow-projection-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',required=True)
    p.add_argument('--base-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=package(a.root,a.mode,a.base_manifest,a.output)
    print(json.dumps({'candidateMode':m['candidateMode'],'files':m['files'],'output':str(a.output)},indent=2))
