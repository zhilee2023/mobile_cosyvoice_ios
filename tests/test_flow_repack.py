import importlib.util
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from repack_flow_weights import append_quant_table,clear_table_field,encode_w8,selected_operator,projection_group,GROUPS
from audit_mnn_weight_layout import FlatBuffer


class LayoutChecks(unittest.TestCase):
    def test_exact_projection_groups(self):
        for group,op in [('q','to_q'),('k','to_k'),('v','to_v'),('out','to_out.0')]:
            name='/transformer_blocks.21/attn/'+op+'/Add_output_0__matmul_converted'
            self.assertEqual(projection_group(name),group)
            for mode,groups in GROUPS.items():self.assertEqual(selected_operator(name,mode),group in groups)
        self.assertIsNone(projection_group('/transformer_blocks.22/attn/to_q/Add'))
        self.assertIsNone(projection_group('/other/attn/to_q/Add'))
        self.assertIsNone(projection_group('/transformer_blocks.1/attn/to_query/Add'))
        for mode in GROUPS:self.assertTrue(selected_operator('/transformer_blocks.0/ff/ff/ff.0/Add',mode))
        self.assertFalse(selected_operator('/time_mlp/linear_1/Add','ffn-qkvo-w8'))
        with self.assertRaises(ValueError):selected_operator('anything','typo')
    def test_appended_quant_metadata(self):
        data=bytearray(bytes(12));table=append_quant_table(data,1024)
        f=FlatBuffer(bytes(data))
        self.assertEqual(f.scalar(table,2),1)
        self.assertEqual(f.scalar(table,7),8)
        self.assertEqual(f.scalar(table,8),-128)
        self.assertEqual(f.scalar(table,9),1024)
        self.assertTrue(f.scalar(table,11,fmt="?"))
        self.assertEqual(f.pointer(table,0),0)

    def test_does_not_change_shared_vtable(self):
        data=bytearray(struct.pack("<HHHH",8,8,4,0)+struct.pack("<ii",8,7)+struct.pack("<ii",16,9))
        clear_table_field(data,8,0)
        f=FlatBuffer(bytes(data))
        self.assertEqual(f.field(8,0),0)
        self.assertEqual(f.scalar(16,0),9)

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"numeric fixture needs numpy")
    def test_idst_w8_decode_and_constant_block(self):
        import numpy as np
        w=np.vstack((np.linspace(-1,2,128,dtype=np.float32),np.full(128,.5,dtype=np.float32)))
        payload,alpha,rows,error=encode_w8(w,2,64)
        self.assertEqual(struct.unpack_from("<II",payload,1),(4,64))
        q=np.frombuffer(payload[266:],dtype=np.uint8).astype(np.float32).reshape(4,64)-128
        scales=np.frombuffer(alpha,dtype="<f4").reshape(4,2)
        decoded=q*scales[:,1,None]+(scales[:,0]+128*scales[:,1])[:,None]
        self.assertEqual(rows,4)
        self.assertLess(np.max(np.abs(decoded.reshape(2,128)-w)),.007)
        self.assertTrue(np.array_equal(decoded.reshape(2,128)[1],w[1]))
        self.assertLess(error,.01)


if __name__=="__main__":unittest.main()
