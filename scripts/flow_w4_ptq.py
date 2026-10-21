"""Native MNN IDST W4 codec, RTN or NumPy HQQ proximal-v2 adaptation.

Sources: alibaba/MNN 3.6.1 source/core/IDSTEncoder.hpp (MSB-first packing),
mobiusml/hqq hqq/core/optimize.py optimize_weights_proximal_v2 (Apache-2.0,
Hicham Badri/Mobius Labs). This is an independent NumPy adaptation, not an
invocation of MNNConvert --hqq or bit-identical upstream Torch output.
Weight-only optimization; no fitted activation scales, GPTQ or AWQ claim.
"""
import struct
import numpy as np

def encode_w4(weights,output_channels,block,algorithm='rtn'):
    if algorithm not in ['rtn','hqq']:raise ValueError('Unknown PTQ method')
    w=np.asarray(weights,dtype=np.float32).reshape(output_channels,-1)
    if w.shape[1]%block:raise ValueError('Block must divide input width')
    w=w.reshape(-1,block);minimum=w.min(axis=1,keepdims=True)
    step=(w.max(axis=1,keepdims=True)-minimum)/np.float32(15)
    good=step>np.float32(1e-6);safe=np.where(good,step,np.float32(1))
    if algorithm=='rtn':q=np.floor((w-minimum)/safe+np.float32(.5)).clip(0,15)
    else:
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
    q=np.where(good,q,0).astype(np.uint8)
    minimum=np.where(good,minimum,w[:,:1]);step=np.where(good,step,0)
    alpha=np.concatenate((minimum,step),axis=1).astype('<f4')
    flat=q.ravel();packed=((flat[::2]<<4)|flat[1::2]).tobytes()
    header=bytes([2])+struct.pack('<II',len(w),block)+bytes([16])+bytes(range(248,256))+bytes(range(8))
    decoded=(q.astype(np.float32)-8)*step+(minimum+8*step)
    error=float(np.linalg.norm((decoded-w).astype(np.float64))/max(1e-30,np.linalg.norm(w.astype(np.float64))))
    return header+packed,alpha.tobytes(),len(w),error
