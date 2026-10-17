import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "export_minicpm_tts_semantic_oracle.py"
SPEC = importlib.util.spec_from_file_location("tts_oracle_exporter", MODULE_PATH)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class GreedyPrefixContractTests(unittest.TestCase):
    def test_exact_tie_stops_prefix_and_invalidates_later_chunks(self):
        first = EXPORTER.greedy_prefix_contract(
            np.array([[3, 2, 0], [2, 3, 0], [3, 3, 0], [2, 4, 0]]),
            [0, 1, 1], eos_token=2, prefix_is_valid=True)
        self.assertEqual(first["stable_prefix"], [0, 1])
        self.assertEqual(first["first_argmax_tie"], {
            "step_index": 2, "top_gap": 0.0, "top_logit": 3.0, "tokens": [0, 1]})
        self.assertFalse(first["prefix_is_valid_after_chunk"])
        later = EXPORTER.greedy_prefix_contract(
            np.array([[3, 2, 0], [2, 3, 0]]), [0], eos_token=2,
            prefix_is_valid=first["prefix_is_valid_after_chunk"])
        self.assertEqual(later["stable_prefix"], [])

    def test_force_no_stop_excludes_eos_from_tie_detection(self):
        result = EXPORTER.greedy_prefix_contract(
            np.array([[3, 2, 3], [2, 3, 100]]), [0],
            eos_token=2, prefix_is_valid=True)
        self.assertEqual(result["stable_prefix"], [0])
        self.assertIsNone(result["first_argmax_tie"])
        self.assertTrue(result["prefix_is_valid_after_chunk"])
        self.assertEqual(result["argmax_top_gaps"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
