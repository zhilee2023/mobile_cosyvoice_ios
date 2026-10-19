from array import array
import base64
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_test_report import flow_thread_comparisons

def run(threads, scale=1):
    mel = array("f", [scale*(1+i%7) for i in range(80*140)])
    if sys.byteorder != "little": mel.byteswap()
    return dict(id=str(threads),kind="module-experiment",state="passed",version="0.7.1",build="15",
        result=dict(rows=[dict(threads=threads,actualThreads=threads,finite=True,completedRuns=3,
          fixtureID="real70-prompt87-effective314-noise-v1-f32-native",melShape=[80,140],byteOrder="little-endian",
          melF32Base64=base64.b64encode(mel).decode(),warmTwoStepsMeanSeconds=1,
          modelFileStamps={str(i):dict(bytes=1,modifiedEpoch=1) for i in range(7)})]))

class ReportAnalysisChecks(unittest.TestCase):
    def test_same_fixture(self):
        out = flow_thread_comparisons(dict(runs=[run(4),run(2),run(3,1.1)]))
        self.assertEqual(len(out),3)
        self.assertTrue(out[0]["numericalGatePassed"])
        self.assertFalse(out[1]["numericalGatePassed"])
    def test_mismatched_model_and_build_not_compared(self):
        a,b = run(4),run(2)
        b["build"] = "16"
        out=flow_thread_comparisons(dict(runs=[a,b]))
        self.assertTrue(any(x.get("state")=="missing-matching-4-thread-reference" for x in out))
        self.assertFalse(any(x.get("threads")==2 for x in out))
        b=run(2); b["result"]["rows"][0]["modelFileStamps"]["0"]["modifiedEpoch"]=2
        self.assertFalse(any(x.get("threads")==2 for x in flow_thread_comparisons(dict(runs=[a,b]))))
    def test_failed_legacy_and_invalid_data_rejected(self):
        b=run(2); b["state"]="not-passed"
        self.assertFalse(any(x.get("threads")==2 for x in flow_thread_comparisons(dict(runs=[run(4),b]))))
        b=run(2); b["kind"]="legacy-import"
        self.assertFalse(any(x.get("threads")==2 for x in flow_thread_comparisons(dict(runs=[run(4),b]))))
        b=run(2); b["result"]["rows"][0]["melF32Base64"]="bad data"
        self.assertEqual(len(flow_thread_comparisons(dict(runs=[b]))),0)

if __name__ == "__main__": unittest.main()
