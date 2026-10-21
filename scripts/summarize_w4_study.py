"""One compact combined report for local PTQ studies; no phone speed claims."""
import argparse,json
from pathlib import Path
from flow_quant_common import digest
def summarize(roots,activation):
    rows=[];complete=True
    for root in roots:
        path=root/'w4-report.json';d=json.loads(path.read_text(encoding='utf-8'))
        complete=complete and d.get('state')=='completed'
        for name,p in d.get('profiles',{}).items():
            recipe=p['recipe'];e=p.get('evaluations',{})
            low=e.get('low',{});normal=e.get('normal',{});summary=low.get('summary',{})
            calibration=summary.get('calibration',{});validation=summary.get('validation',{})
            passed=d.get('state')=='completed' and low.get('state')=='completed' and calibration.get('screen_pass') is True and validation.get('screen_pass') is True
            rows.append({'profile':name,'block':recipe['block'],'algorithm':recipe.get('algorithm'),'weight_bytes':recipe['candidate_weight_bytes'],
              'w4_operators':recipe.get('w4_operators'),'w8_operators':recipe.get('w8_operators'),'integrity':p.get('integrity'),
              'normal_validation':normal.get('summary',{}).get('validation'),'low_calibration':calibration,'low_validation':validation,
              'numeric_screen_pass':passed,'listening':p.get('listening',[]),'source_report':str(path),'source_report_sha256':digest(path)})
    ar=json.loads((activation/'activation-report.json').read_text(encoding='utf-8'))
    return {'state':'completed' if complete else 'running','scope':'Local Windows CPU MNN actual forwards; not phone RTF/MOS, no automatic deployment',
      'methods_note':'HQQ NumPy proximal adaptation; GPTQ BLOCK-DIAGONAL approximation, not full original GPTQ/Q-DiT; MSE is weight-only clipping',
      'qdit_status':'Original official Q-DiT NOT integrated. Block32/64 comparison only tests a grouping idea, no evolutionary layer allocation or matching dynamic-activation implementation.',
      'activation_calibration':{'state':ar['state'],'utterances':len(ar['calibration_case_ids']),'snapshots':ar['snapshots_completed'],'layers':len(ar['layers']),
         'validation_used_for_fit':ar['validation_used_for_fit'],'report':str(activation/'activation-report.json'),'sha256':digest(activation/'activation-report.json')},
      'profiles':rows}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--activation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=summarize(a.roots,a.activation);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':r['state'],'profiles':len(r['profiles']),'passed':[x['profile']+':'+str(x['block']) for x in r['profiles'] if x['numeric_screen_pass']]},indent=2))
