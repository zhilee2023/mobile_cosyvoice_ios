import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mnn_session_config import cpu_session_config


class SessionConfigChecks(unittest.TestCase):
    def test_binding_requires_actual_integer_enums(self):
        self.assertEqual(cpu_session_config("high","low",4),
                         {"backend":"CPU","precision":1,"memory":2,"numThread":4})
        self.assertEqual(cpu_session_config("high","normal")["memory"],0)
        self.assertEqual(cpu_session_config("low","low")["precision"],2)
        for key in ("memory","precision"):self.assertIs(type(cpu_session_config()[key]),int)

    def test_invalid_modes_rejected(self):
        with self.assertRaises(ValueError):cpu_session_config(memory="typo")
        with self.assertRaises(ValueError):cpu_session_config(threads=0)


if __name__=="__main__":unittest.main()
