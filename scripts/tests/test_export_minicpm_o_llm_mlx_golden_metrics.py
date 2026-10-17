#!/usr/bin/env python3
"""Check fixture metrics with tiny files, without MLX or model imports."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_minicpm_o_llm_golden import ArtifactWriter  # noqa: E402
from export_minicpm_o_llm_mlx_golden import load_manifest_array, output_metrics  # noqa: E402


class FixtureMetricTests(unittest.TestCase):
    @staticmethod
    def record(writer: ArtifactWriter, tokens: list[int]) -> dict[str, str]:
        return {
            "hidden": writer.array("hidden", np.array([[1.0, 2.0]])),
            "logits_selected": writer.array("logits", np.array([[3.0, 4.0]])),
            "top20": writer.array("top20", np.array([tokens]), dtype="i32"),
        }

    def test_manifest_loader_preserves_integer_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ArtifactWriter(root)
            key = writer.array("ids", np.array([151705, 151706, 151707]), dtype="i32")
            actual = load_manifest_array(root, {"arrays": writer.arrays}, key)
            self.assertEqual(actual.dtype, np.dtype("<i4"))
            self.assertEqual(actual.tolist(), [151705, 151706, 151707])

    def test_overlap_counts_token_ids_instead_of_float_reinterpretations(self) -> None:
        with tempfile.TemporaryDirectory() as actual_dir, tempfile.TemporaryDirectory() as reference_dir:
            actual_root, reference_root = Path(actual_dir), Path(reference_dir)
            actual_writer, reference_writer = ArtifactWriter(actual_root), ArtifactWriter(reference_root)
            actual = self.record(actual_writer, [151705, 151706, 151707, 151708])
            expected = self.record(reference_writer, [151706, 151708, 151709, 151710])
            metrics = output_metrics(actual_root, actual_writer, actual, reference_root,
                                     {"arrays": reference_writer.arrays}, expected)
            self.assertEqual(metrics["top20_overlap"], {"min": 2, "rows": [2]})
            self.assertEqual(metrics["hidden"]["max_abs"], 0)
            self.assertEqual(metrics["logits_selected"]["cosine"], 1)

    def test_disjoint_ids_have_zero_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as actual_dir, tempfile.TemporaryDirectory() as reference_dir:
            actual_root, reference_root = Path(actual_dir), Path(reference_dir)
            actual_writer, reference_writer = ArtifactWriter(actual_root), ArtifactWriter(reference_root)
            actual = self.record(actual_writer, list(range(20)))
            expected = self.record(reference_writer, list(range(20, 40)))
            metrics = output_metrics(actual_root, actual_writer, actual, reference_root,
                                     {"arrays": reference_writer.arrays}, expected)
            self.assertEqual(metrics["top20_overlap"], {"min": 0, "rows": [0]})


if __name__ == "__main__":
    unittest.main()
