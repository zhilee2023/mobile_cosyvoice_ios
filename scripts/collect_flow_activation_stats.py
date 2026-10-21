"""Bounded calibration-only per-linear input/Hessian statistics; no raw dumps."""
import argparse,json
from pathlib import Path
import numpy as np
from flow_projection_matrix import verify_corpus
from flow_quant_common import digest
from repack_flow_weights import external_regions,selected_operator,FLOW
from audit_mnn_weight_layout import FlatBuffer
from pc_flow_hift_probe import model_session,put,get
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=verify_corpus(a.corpus,a.assets);cases=[c for c in d['cases'] if c['split']=='calibration'];assert len(cases)==20
    a.output.mkdir(parents=True,exist_ok=False)
    regions=external_regions(FlatBuffer((a.assets/FLOW).read_bytes()),(a.assets/(FLOW+'.weight')).stat().st_size)
    names=[r['name'] for r in regions if r['kind']==9 and selected_operator(r['name'],'ffn-qvout-w8')];assert len(names)==110
    stats={};report={'state':'running','source_graph_sha256':digest(a.assets/FLOW),'source_weight_sha256':digest(a.assets/(FLOW+'.weight')),
      'corpus_sha256':digest(a.corpus/'calibration-report.json'),'calibration_case_ids':[c['id'] for c in cases],
      'validation_used_for_fit':False,'block':64,'max_rows_per_snapshot':32,'scope':'block-diagonal Hessian only; omits cross-block correlations; one prompt voice',
      'snapshots_completed':0,'layers':[]}
    def save():(a.output/'activation-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save();n,s=model_session(a.assets/FLOW,'high','normal');last=None
    try:
      for case in cases:
       for snap in case['snapshots']:
        with np.load(Path(snap['directory'])/'tensors.npz') as z:inputs={k:z[k].copy() for k in z.files if k!='estimator_out'};expected=z['estimator_out'].copy()
        shapes={k:v.shape for k,v in inputs.items()}
        if shapes!=last:
         for name,v in inputs.items():n.resizeTensor(n.getSessionInput(s,name),v.shape)
         n.resizeSession(s);last=shapes
        for name,v in inputs.items():put(n,s,name,v)
        seen=set();errors=[]
        def before(tensors,info):
         name=info.getName()
         if name not in names:return True
         try:
          if name in seen:raise ValueError('Duplicate selected layer callback')
          seen.add(name);v=np.asarray(tensors[0].getNumpyData(),dtype=np.float32)
          if v.ndim!=4 or v.shape[0]!=case['runtime_frames'] or v.shape[2:]!=(1,1) or v.shape[1]%64:raise ValueError('Unexpected linear input layout '+str(v.shape))
          matrix=v.reshape(v.shape[0],v.shape[1])[:case['effective_frames']]
          idx=np.linspace(0,len(matrix)-1,min(32,len(matrix)),dtype=np.int64);x=matrix[idx].copy()
          if not np.isfinite(x).all():raise ValueError('Nonfinite calibration activation')
          if name not in stats:stats[name]={'h':np.zeros((x.shape[1]//64,64,64),dtype=np.float64),'sum_abs':np.zeros(x.shape[1],dtype=np.float64),'max_abs':np.zeros(x.shape[1],dtype=np.float32),'samples':0,'snapshots':0}
          st=stats[name];blocks=x.reshape(len(x),-1,64)
          st['h']+=np.einsum('ngb,ngc->gbc',blocks,blocks,optimize=True).astype(np.float64)
          st['sum_abs']+=np.sum(np.abs(x),axis=0,dtype=np.float64);st['max_abs']=np.maximum(st['max_abs'],np.max(np.abs(x),axis=0))
          st['samples']+=len(x);st['snapshots']+=1
          return True
         except Exception as e:errors.append(str(e));return False
        code=n.runSessionWithCallBackInfo(s,before,lambda tensors,info:True)
        if errors or code or seen!=set(names):raise ValueError('Callback capture failed '+str(errors)+' '+str(len(seen)))
        actual=get(n,s,'estimator_out').reshape(expected.shape)
        l2=float(np.linalg.norm((actual-expected).astype(np.float64))/max(1e-30,np.linalg.norm(expected.astype(np.float64))))
        if not np.isfinite(actual).all() or l2>1e-5:raise ValueError('Callback changed reference output '+str(l2))
        report['snapshots_completed']+=1;save();print('CALIBRATION',case['id'],report['snapshots_completed'],'/40; layers',len(seen),flush=True)
      for i,name in enumerate(names):
       st=stats[name];assert st['snapshots']==40
       path=a.output/(f'layer-{i:03d}.npz')
       np.savez_compressed(path,hessian=2*st['h']/st['samples'],mean_abs=st['sum_abs']/st['samples'],max_abs=st['max_abs'])
       report['layers'].append({'name':name,'file':path.name,'sha256':digest(path),'input_channels':len(st['sum_abs']),'samples':st['samples'],'snapshots':st['snapshots']})
      report['state']='completed';save()
    except Exception as e:report['state']='failed';report['error']=str(e);save();raise
if __name__=='__main__':main()
