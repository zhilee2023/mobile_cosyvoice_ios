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
        for method in ['rtn','hqq']:
            w=np.full((2,64),.5,dtype=np.float32)
            payload,alpha,rows,error=encode_w4(w,2,64,method)
            self.assertEqual(error,0);self.assertEqual(rows,2)
            w=np.random.default_rng(1).normal(size=(4,128)).astype(np.float32)
            self.assertTrue(np.isfinite(encode_w4(w,4,64,method)[3]))
    def test_metadata(self):
        data=bytearray(bytes(12));table=append_quant_table(data,256,4);f=FlatBuffer(bytes(data))
        self.assertEqual(f.scalar(table,7),4);self.assertEqual(f.scalar(table,8),-8)
if __name__=='__main__':unittest.main()
