import unittest,sys,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from flow_w4_ptq import encode_w4
from repack_flow_weights import append_quant_table
from audit_mnn_weight_layout import FlatBuffer
class W4(unittest.TestCase):
    def test_msb_first_and_decode(self):
        w=np.arange(64,dtype=np.float32).reshape(1,64)%16
        payload,alpha,rows,error=encode_w4(w,1,64)
        self.assertEqual(payload[26:34],bytes.fromhex('0123456789abcdef'))
        self.assertEqual(struct.unpack_from('<II',payload,1),(1,64))
        q=np.frombuffer(payload[26:],dtype=np.uint8);q=np.stack((q>>4,q&15),axis=1).ravel()
        a=np.frombuffer(alpha,dtype='<f4');decoded=(q.astype(np.float32)-8)*a[1]+(a[0]+8*a[1])
        np.testing.assert_array_equal(decoded,w.ravel());self.assertEqual(error,0)
    def test_constant_and_hqq_finite(self):
        for method in ['rtn','hqq','mse']:
            w=np.full((2,64),.5,dtype=np.float32)
            payload,alpha,rows,error=encode_w4(w,2,64,method)
            self.assertEqual(error,0);self.assertEqual(rows,2)
            w=np.random.default_rng(1).normal(size=(4,128)).astype(np.float32)
            self.assertTrue(np.isfinite(encode_w4(w,4,64,method)[3]))
    def test_metadata(self):
        data=bytearray(bytes(12));table=append_quant_table(data,256,4);f=FlatBuffer(bytes(data))
        self.assertEqual(f.scalar(table,7),4);self.assertEqual(f.scalar(table,8),-8)
    def test_mse_search_contains_rtn(self):
        w=np.random.default_rng(8).normal(size=(8,256)).astype(np.float32);w[0,0]=12
        def decoded(method):
            payload,alpha,rows,_=encode_w4(w,8,64,method)
            packed=np.frombuffer(payload[26:],dtype=np.uint8)
            q=np.stack((packed>>4,packed&15),axis=1).reshape(rows,64).astype(np.float32)-8
            a=np.frombuffer(alpha,dtype='<f4').reshape(rows,2)
            return q*a[:,1,None]+(a[:,0]+8*a[:,1])[:,None]
        mse=np.mean((decoded('mse')-w.reshape(-1,64))**2,axis=1)
        rtn=np.mean((decoded('rtn')-w.reshape(-1,64))**2,axis=1)
        self.assertTrue(np.all(mse<=rtn+1e-7))
    def test_gptq_diagonal_identity_reduces_to_rtn(self):
        w=np.random.default_rng(7).normal(size=(4,128)).astype(np.float32)
        actual=encode_w4(w,4,64,'gptq-blockdiag',np.tile(np.eye(64)[None,:,:],(2,1,1)))
        reference=encode_w4(w,4,64,'rtn')
        self.assertEqual(actual[:3],reference[:3])
        with self.assertRaises(ValueError):encode_w4(w,4,64,'gptq-blockdiag')
if __name__=='__main__':unittest.main()
