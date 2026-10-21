"""Native MNN IDST W4 codec, RTN or NumPy HQQ proximal-v2 adaptation.

Sources: alibaba/MNN 3.6.1 source/core/IDSTEncoder.hpp (MSB-first packing),
mobiusml/hqq hqq/core/optimize.py optimize_weights_proximal_v2 (Apache-2.0,
Hicham Badri/Mobius Labs). This is an independent NumPy adaptation, not an
invocation of MNNConvert --hqq or bit-identical upstream Torch output.
Weight-only optimization; no fitted activation scales, GPTQ or AWQ claim.
"""
import struct
import numpy as np

def encode_w4(weights,output_channels,block,algorithm='rtn',hessian=None):
    if algorithm not in ['rtn','hqq','mse','gptq-blockdiag']:raise ValueError('Unknown PTQ method')
    w=np.asarray(weights,dtype=np.float32).reshape(output_channels,-1)
    if w.shape[1]%block:raise ValueError('Block must divide input width')
    w=w.reshape(-1,block);minimum=w.min(axis=1,keepdims=True)
    step=(w.max(axis=1,keepdims=True)-minimum)/np.float32(15)
    good=step>np.float32(1e-6);safe=np.where(good,step,np.float32(1))
    if algorithm in ['rtn','mse']:
        q=np.floor((w-minimum)/safe+np.float32(.5)).clip(0,15)
        if algorithm=='mse':
            # Per-group symmetric clipping of the asymmetric min/max interval.
            # Include RTN (factor1), select by original weight MSE only.
            # This is OMSE-style weight clipping, NOT activation-aware AWQ.
            center=(w.max(axis=1,keepdims=True)+minimum)/2
            half=(w.max(axis=1,keepdims=True)-minimum)/2
            best=np.mean((q*step+minimum-w)**2,axis=1,keepdims=True)
            best_q=q.copy();best_min=minimum.copy();best_step=step.copy()
            for factor in np.linspace(.99,.70,30,dtype=np.float32):
                lower=center-half*factor;candidate_step=step*factor
                cq=np.floor((w-lower)/np.where(good,candidate_step,1)+np.float32(.5)).clip(0,15)
                error=np.mean((cq*candidate_step+lower-w)**2,axis=1,keepdims=True)
                choose=error<best;best=np.where(choose,error,best)
                best_q=np.where(choose,cq,best_q);best_min=np.where(choose,lower,best_min);best_step=np.where(choose,candidate_step,best_step)
            q,minimum,step=best_q,best_min,best_step
    elif algorithm=='hqq':
        inv=np.float32(1)/safe;zero=-minimum*inv
        best_zero=zero.copy();best_error=float('inf');beta=10.
        # HQQ v2 fixed scale, fractional zero proximal updates, axis=1.
        for _ in range(20):
            q=np.rint(w*inv+zero).clip(0,15);restored=(q-zero)/inv
            error=float(np.mean(np.abs(w-restored),dtype=np.float64))
            if error<best_error:best_error=error;best_zero=zero.copy()
            else:break
            residual=w-restored;mag=np.abs(residual)
            shrink=np.sign(residual)*np.maximum(mag-(1/beta)*np.maximum(mag,1e-12)**np.float32(-.3),0)
            zero=np.mean(q-(w-shrink)*inv,axis=1,keepdims=True,dtype=np.float32)
            beta*=1.01
        q=np.rint(w*inv+best_zero).clip(0,15);minimum=-best_zero/inv
    else:
        # GPTQ sequential error feedback within each 64-input block only.
        # Cross-block Hessian correlations are intentionally omitted; no actorder.
        # Algorithm source: IST-DASLab/gptq/gptq.py fasterquant (Apache-2.0).
        groups=w.shape[0]//output_channels
        if hessian is None or hessian.shape!=(groups,block,block) or not np.isfinite(hessian).all():raise ValueError('Need matching calibration block Hessians')
        work=w.reshape(output_channels,groups,block).copy();q=np.zeros_like(work)
        mins=minimum.reshape(output_channels,groups);steps=safe.reshape(output_channels,groups)
        for g in range(groups):
            h=hessian[g].astype(np.float64).copy();h=(h+h.T)/2
            diag=np.diag_indices(block);h[diag]+=max(1e-8,float(np.mean(np.diag(h)))*.01)
            # Damped SPD inverse, then upper Cholesky as in GPTQ.
            inv=np.linalg.inv(h);upper=np.linalg.cholesky((inv+inv.T)/2).T.astype(np.float32)
            for i in range(block):
                current=work[:,g,i].copy()
                indices=np.floor((current-mins[:,g])/steps[:,g]+np.float32(.5)).clip(0,15)
                reconstructed=indices*steps[:,g]+mins[:,g];q[:,g,i]=indices
                error=(current-reconstructed)/upper[i,i]
                work[:,g,i:]-=error[:,None]*upper[i,i:][None,:]
        q=q.reshape(-1,block)
    q=np.where(good,q,0).astype(np.uint8)
    minimum=np.where(good,minimum,w[:,:1]);step=np.where(good,step,0)
    alpha=np.concatenate((minimum,step),axis=1).astype('<f4')
    flat=q.ravel();packed=((flat[::2]<<4)|flat[1::2]).tobytes()
    header=bytes([2])+struct.pack('<II',len(w),block)+bytes([16])+bytes(range(248,256))+bytes(range(8))
    decoded=(q.astype(np.float32)-8)*step+(minimum+8*step)
    error=float(np.linalg.norm((decoded-w).astype(np.float64))/max(1e-30,np.linalg.norm(w.astype(np.float64))))
    return header+packed,alpha.tobytes(),len(w),error
