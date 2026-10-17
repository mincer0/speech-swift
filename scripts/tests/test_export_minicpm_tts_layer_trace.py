import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "export_minicpm_tts_layer_trace.py"
spec = importlib.util.spec_from_file_location("tts_layer_trace", path)
trace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_module)


class TeacherReplayPlanTests(unittest.TestCase):
    def fixture(self):
        return {
            "chunks": [{"expected_key": f"chunk{i}.expected"} for i in range(3)],
            "trace": [dict(text_start=start, condition_length=length,
                           sampled_count=26, committed=list(range(25)), cache_offset=end)
                      for start, length, end in [(0, 4, 29), (29, 5, 59), (59, 3, 87)]],
        }

    def testDiscardedSampleIsNeverFedBeforeNextCondition(self):
        plan = trace_module.teacher_forced_plan(self.fixture(), 1, 0)
        self.assertEqual(len(plan), 27)
        self.assertEqual(plan[-2]["input_code"], 24)
        self.assertIsNone(plan[-1]["input_code"])
        self.assertEqual(plan[-1]["cache_offset_before"], 29)
        self.assertEqual(plan[-1]["sequence_length"], 5)

    def testOneTokenStepUsesReturnedCodeAndOfficialCache(self):
        plan = trace_module.teacher_forced_plan(self.fixture(), 2, 1)
        self.assertEqual(plan[-1]["input_code"], 0)
        self.assertEqual(plan[-1]["cache_offset_before"], 62)
        self.assertEqual(plan[-1]["sequence_length"], 1)

    def testRejectsUnobservedDecoderStep(self):
        with self.assertRaises(ValueError):
            trace_module.teacher_forced_plan(self.fixture(), 0, 26)


if __name__ == "__main__":
    unittest.main()
