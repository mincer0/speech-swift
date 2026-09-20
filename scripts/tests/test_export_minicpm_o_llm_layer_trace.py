#!/usr/bin/env python3
"""Portable CLI regression tests for the LLM layer-trace exporter.

The exporter is imported only for its argument and mode helpers.  These tests
deliberately do not import torch/transformers or load any model weights.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "export_minicpm_o_llm_layer_trace.py"
SPEC = importlib.util.spec_from_file_location("export_minicpm_o_llm_layer_trace", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"unable to import exporter from {SCRIPT_PATH}")
TRACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACE
SPEC.loader.exec_module(TRACE)


class _FakeTensor:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _FakeTorch:
    def __init__(self, grad_enabled: bool = True) -> None:
        self.grad_enabled = grad_enabled

    def set_grad_enabled(self, enabled: bool) -> None:
        self.grad_enabled = enabled

    def is_grad_enabled(self) -> bool:
        return self.grad_enabled

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)


class LayerTraceCliTests(unittest.TestCase):
    """Check parse and branch semantics without constructing a decoder."""

    @staticmethod
    def _parse(
        *trace_layer: int,
        token_count: int | None = None,
        token_ids: str | None = None,
        cached_prefix_count: int | None = None,
        trace_all_layers: bool = False,
    ) -> object:
        argv = [
            "--source-model", "source",
            "--mlx-bundle", "bundle",
            "--output", "output",
        ]
        if trace_layer:
            argv.extend(("--trace-layer", str(trace_layer[0])))
        if token_count is not None:
            argv.extend(("--token-count", str(token_count)))
        if token_ids is not None:
            argv.extend(("--token-ids", token_ids))
        if cached_prefix_count is not None:
            argv.extend(("--cached-prefix-count", str(cached_prefix_count)))
        if trace_all_layers:
            argv.append("--trace-all-layers")
        return TRACE.parse_args(argv)

    def test_omitted_trace_layer_keeps_full_summary_mode(self) -> None:
        args = self._parse()
        self.assertIsNone(args.trace_layer)
        self.assertFalse(TRACE.is_targeted_trace(args.trace_layer))

    def test_explicit_zero_selects_targeted_mode(self) -> None:
        args = self._parse(0)
        self.assertEqual(args.trace_layer, 0)
        self.assertTrue(TRACE.is_targeted_trace(args.trace_layer))

    def test_explicit_nonzero_layer_selects_targeted_mode(self) -> None:
        args = self._parse(17)
        self.assertEqual(args.trace_layer, 17)
        self.assertTrue(TRACE.is_targeted_trace(args.trace_layer))

    def test_default_prefill_keeps_four_fixed_tokens(self) -> None:
        args = self._parse()
        self.assertEqual(args.token_count, 4)
        self.assertEqual(TRACE.trace_token_ids(args.token_count), [151643, 198, 108, 151645])
        self.assertEqual(
            TRACE.trace_input_provenance(TRACE.trace_token_ids(args.token_count)),
            {"token_ids": [151643, 198, 108, 151645], "input_length": 4},
        )

    def test_single_token_prefill_is_explicit_and_cache_free(self) -> None:
        args = self._parse(token_count=1)
        self.assertEqual(args.token_count, 1)
        token_ids = TRACE.trace_token_ids(args.token_count)
        self.assertEqual(token_ids, [151643])
        self.assertEqual(TRACE.trace_input_provenance(token_ids)["input_length"], 1)

    def test_explicit_token_ids_parse_and_override_token_count(self) -> None:
        args = self._parse(token_count=1, token_ids="151643, 151645,198")
        self.assertEqual(args.token_ids, [151643, 151645, 198])
        self.assertEqual(
            TRACE.resolve_trace_token_ids(args.token_count, args.token_ids),
            [151643, 151645, 198],
        )
        self.assertEqual(
            TRACE.trace_token_ids(0, args.token_ids), [151643, 151645, 198]
        )

    def test_explicit_token_ids_accepts_a_single_id(self) -> None:
        self.assertEqual(TRACE.parse_token_ids("151643"), [151643])
        self.assertIsNone(TRACE.parse_token_ids(None))

    def test_explicit_token_ids_reject_empty_or_invalid_values(self) -> None:
        for value in ("", ",", "151643,", ",151643", "not-an-int", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TRACE.parse_token_ids(value)

    def test_explicit_token_ids_override_invalid_token_count(self) -> None:
        self.assertEqual(
            TRACE.resolve_trace_token_ids(0, "151643,151645,198"),
            [151643, 151645, 198],
        )

    def test_cached_all_layer_flag_is_opt_in(self) -> None:
        self.assertFalse(self._parse().trace_all_layers)
        self.assertTrue(self._parse(trace_all_layers=True).trace_all_layers)

    def test_cached_prefix_defaults_to_cache_free_and_accepts_explicit_prefix(self) -> None:
        self.assertEqual(self._parse().cached_prefix_count, 0)
        self.assertEqual(self._parse(cached_prefix_count=2).cached_prefix_count, 2)

    def test_cached_prefix_range_keeps_a_successor_token(self) -> None:
        for value in (0, 1, 3):
            with self.subTest(value=value):
                TRACE.validate_cached_prefix_count(value, 4)
        for value in (-1, 4, 5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, r"--cached-prefix-count must be in \[1, 3\]"
                ):
                    TRACE.validate_cached_prefix_count(value, 4)

    def test_cached_prefix_range_uses_explicit_sequence_length(self) -> None:
        token_ids = TRACE.resolve_trace_token_ids(1, "151643,151645,198")
        for value in (0, 1, 2):
            with self.subTest(value=value):
                TRACE.validate_cached_prefix_count(value, len(token_ids))
        with self.assertRaisesRegex(
            ValueError, r"--cached-prefix-count must be in \[1, 2\]"
        ):
            TRACE.validate_cached_prefix_count(3, len(token_ids))

    def test_cached_trace_provenance_records_token_split_and_backend(self) -> None:
        provenance = TRACE.cached_trace_provenance(
            [151643, 198], [108], trace_layer=0, attn_implementation="sdpa"
        )
        self.assertEqual(
            provenance,
            {
                "mode": "cached_decode",
                "token_ids": [151643, 198, 108],
                "prefix_token_ids": [151643, 198],
                "current_token_ids": [108],
                "prefix_count": 2,
                "current_length": 1,
                "prefix_length": 2,
                "input_length": 3,
                "trace_layer": 0,
                "attn_implementation": "sdpa",
            },
        )

    def test_cached_trace_provenance_keeps_explicit_token_sequence(self) -> None:
        provenance = TRACE.cached_trace_provenance(
            [151643, 151645], [198], trace_layer=7,
            attn_implementation="sdpa", token_ids=[151643, 151645, 198],
        )
        self.assertEqual(provenance["token_ids"], [151643, 151645, 198])
        self.assertEqual(provenance["prefix_token_ids"], [151643, 151645])
        self.assertEqual(provenance["current_token_ids"], [198])

    def test_qwen3_451_head_counts_fall_back_to_model_config(self) -> None:
        class Config:
            num_attention_heads = 32
            num_key_value_heads = 8

        class AttentionWithoutHeadAttributes:
            pass

        attention = AttentionWithoutHeadAttributes()
        self.assertEqual(
            TRACE._resolve_attention_heads(attention, Config(), key_value=False), 32
        )
        self.assertEqual(
            TRACE._resolve_attention_heads(attention, Config(), key_value=True), 8
        )

    def test_attention_head_instance_attributes_take_precedence(self) -> None:
        class Config:
            num_attention_heads = 32
            num_key_value_heads = 8

        class Attention:
            num_heads = 16
            num_key_value_heads = 4

        attention = Attention()
        self.assertEqual(
            TRACE._resolve_attention_heads(attention, Config(), key_value=False), 16
        )
        self.assertEqual(
            TRACE._resolve_attention_heads(attention, Config(), key_value=True), 4
        )

    def test_token_count_range_is_rejected(self) -> None:
        for value in (0, 5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"--token-count must be in \[1, 4\]"):
                    TRACE.trace_token_ids(value)

    def test_range_validation_allows_none_and_decoder_boundaries(self) -> None:
        for value in (None, 0, 35):
            with self.subTest(value=value):
                TRACE.validate_trace_layer(value, 36)

    def test_range_validation_rejects_negative_and_upper_bound(self) -> None:
        for value in (-1, 36):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"--trace-layer must be in \[0, 35\]"):
                    TRACE.validate_trace_layer(value, 36)

    def test_help_describes_omission_and_layer_zero(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                TRACE.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("omit this option for the full decoder layer-output summary", help_text)
        self.assertIn("layer 0 is valid", help_text)
        self.assertIn("--token-count COUNT", help_text)
        self.assertIn("without a KV cache", help_text)
        self.assertIn("--cached-prefix-count COUNT", help_text)
        self.assertIn("DynamicCache", help_text)
        self.assertIn("--token-ids IDS", help_text)
        self.assertIn("--trace-all-layers", help_text)

    def test_no_grad_entry_is_verified_and_provenance_is_explicit(self) -> None:
        torch = _FakeTorch(grad_enabled=True)
        TRACE.enter_no_grad_inference(torch)
        self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(
            TRACE.trace_provenance(torch),
            {"grad_enabled": False, "inference_semantics": "torch.no_grad"},
        )

    def test_no_grad_entry_fails_closed_when_gradients_remain_enabled(self) -> None:
        class StubbornTorch(_FakeTorch):
            def set_grad_enabled(self, enabled: bool) -> None:
                del enabled

        with self.assertRaisesRegex(RuntimeError, "gradients remain enabled"):
            TRACE.enter_no_grad_inference(StubbornTorch())

    def test_record_guard_rejects_gradient_tracked_tensors(self) -> None:
        torch = _FakeTorch(grad_enabled=False)
        with self.assertRaisesRegex(RuntimeError, "q unexpectedly requires_grad=True"):
            TRACE.assert_trace_tensor(torch, "q", _FakeTensor(requires_grad=True))


if __name__ == "__main__":
    unittest.main()
