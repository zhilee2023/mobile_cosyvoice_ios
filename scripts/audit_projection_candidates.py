"""Verify selected W8 metadata and bit-exact preservation of every other region."""
import argparse
from collections import Counter
import json
from pathlib import Path
from audit_mnn_weight_layout import FlatBuffer
from flow_quant_common import digest
from repack_flow_weights import FLOW,external_regions,selected_operator,GROUPS

def audit_candidates(root,assets):
    report=json.loads((root/'projection-report.json').read_text(encoding='utf-8'))
    source_graph=assets/FLOW;source_weight=assets/(FLOW+'.weight')
    source=FlatBuffer(source_graph.read_bytes());regions=external_regions(source,source_weight.stat().st_size)
    hashes={'graph':digest(source_graph),'weight':digest(source_weight)}
    out={'state':'running','source_hashes':hashes,'profiles':{},'original_assets_unchanged':False}
    for mode,profile in report['profiles'].items():
        recipe=profile['recipe'];graph=Path(profile['model']);weight=Path(str(graph)+'.weight')
        if hashes!=recipe['source_hashes'] or digest(graph)!=recipe['graph_sha256'] or digest(weight)!=recipe['weight_sha256']:
            raise ValueError('Source/candidate identity changed: '+mode)
        f=FlatBuffer(graph.read_bytes());candidate=external_regions(f,weight.stat().st_size)
        if [(r['kind'],r['name']) for r in regions]!=[(r['kind'],r['name']) for r in candidate]:raise ValueError('Graph regions/order changed')
        counts=Counter()
        with source_weight.open('rb') as a,weight.open('rb') as b:
            for old,new in zip(regions,candidate):
                oe,ne=old['external'],new['external'];selected=old['kind']==9 and selected_operator(old['name'],mode)
                a.seek(oe[0]);old_data=a.read(sum(oe[1:]));b.seek(ne[0]);new_data=b.read(sum(ne[1:]))
                if not selected:
                    if old_data!=new_data:raise ValueError('Preserved region changed: '+old['name'])
                    counts['preserved_regions_bit_exact']+=1
                else:
                    q=f.pointer(new['main'],3)
                    if not q or f.scalar(q,2)!=1 or f.scalar(q,7)!=8:raise ValueError('Selected operator not native dense W8')
                    if old_data[oe[1]:]!=new_data[ne[1]+ne[2]:]:raise ValueError('Bias changed: '+old['name'])
                    counts['w8_operators']+=1;counts['selected_biases_bit_exact']+=1
        if counts['w8_operators']!=44+22*len(GROUPS[mode]):raise ValueError('Wrong projection count')
        out['profiles'][mode]=dict(counts)
    out['original_assets_unchanged']=True;out['state']='completed'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();value=audit_candidates(a.root,a.assets)
    (a.root/'layout-integrity.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,indent=2))
