"""Summarize a single projection report without promoting a phone model."""
import argparse
import json
from pathlib import Path

def summarize(report):
    rows=[]
    for mode,p in report.get('profiles',{}).items():
        ev=p.get('evaluations',{});low=ev.get('low',{});normal=ev.get('normal',{})
        slices=[low.get('summary',{}).get('calibration',{}),low.get('summary',{}).get('validation',{}),normal.get('summary',{}).get('validation',{})]
        numeric=(report.get('sensitivity_state')=='completed' and low.get('state')==normal.get('state')=='completed'
            and [s.get('cases') for s in slices]==[20,5,5] and all(s.get('numeric_screen_pass') is True for s in slices))
        time=report.get('timing',{}).get('summary',{}).get(mode,{})
        repeat_ok=(report.get('timing',{}).get('state')=='completed' and time.get('cases')==5 and 0<=time.get('max_repeat_mel_l2',1)<=1e-5)
        rows.append({'mode':mode,'w8_operators':p.get('recipe',{}).get('quantized_operator_count'),
            'weight_bytes':p.get('recipe',{}).get('candidate_weight_bytes'),
            'normal_validation':slices[2], 'low_calibration':slices[0], 'low_validation':slices[1],
            'windows_timing':time,'numeric_screen_pass':numeric,
            'eligible_for_phone_review':numeric and repeat_ok,'listening_validated':False})
    return {'state':report.get('state'),'scope':report.get('scope'),'gates':report.get('gates'),
        'recipes':rows,'automatic_promotion':False,
        'warning':'Screening and Windows descriptive timing only; not phone speed/memory, MOS or listening acceptance'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();value=json.dumps(summarize(json.loads(args.report.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(value+'\n',encoding='utf-8')
    print(value)
