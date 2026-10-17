#!/usr/bin/env python3
"""Check small sampler fixtures against the checkpoint's actual decode source.

The AST loader compiles only StreamDecoder.decode and its top-k/top-p helper.
It never constructs or loads a model. Uniform draws replace multinomial only
to make the categorical decisions reproducible across PyTorch and NumPy.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_minicpm_o_llm_golden import (  # noqa: E402
    ArtifactWriter, deterministic_sampler_case, filtered_sample, make_sampling_cases,
)


PROTOCOL = dict(listen=1, chunk_eos=2, chunk_tts_eos=3, turn_eos=4,
                speak=5, unit_end=6, tts_bos=7)


class SamplerReferenceTests(unittest.TestCase):
    def test_chat_and_duplex_negative_repetition_are_explicit(self) -> None:
        logits = np.full(32, -100.0)
        logits[20], logits[21] = -1.0, -1.02
        common = dict(protocol=PROTOCOL, uniforms=[0.2, 0.2], mode="greedy",
                      previous_tokens=[20], repetition_penalty=1.05, length_penalty=1)
        self.assertEqual(deterministic_sampler_case(logits, **common)[0], 21)
        self.assertEqual(deterministic_sampler_case(
            logits, **common, repetition_policy="official_duplex")[0], 20)

    def test_top_k_keeps_every_threshold_tie(self) -> None:
        logits = np.array([3.0, 2.0, 2.0, -100.0])
        tokens = {filtered_sample(logits, uniform=draw, mode="sampling",
                                  temperature=1, top_k=2, top_p=1)
                  for draw in (0.0, 0.7, 0.999)}
        self.assertEqual(tokens, {0, 1, 2})

    def test_zero_window_is_explicitly_policy_dependent(self) -> None:
        logits = np.full(32, -100.0)
        logits[20], logits[21] = 5.1, 5.0
        common = dict(protocol=PROTOCOL, uniforms=[0.2, 0.2], mode="greedy",
                      previous_tokens=[20], repetition_penalty=1.05,
                      repetition_window_size=0, length_penalty=1)
        self.assertEqual(deterministic_sampler_case(logits, **common)[0], 20)
        self.assertEqual(deterministic_sampler_case(
            logits, **common, repetition_policy="official_duplex")[0], 21)

    def test_fixture_cases_identify_policy_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases = make_sampling_cases(ArtifactWriter(Path(directory)), PROTOCOL)
        self.assertEqual({case["repetition_policy"] for case in cases},
                         {"hugging_face", "official_duplex"})
        for case in cases:
            self.assertEqual(case["reference_backend"], "numpy_runtime_policy")
            self.assertEqual(case["reference_policy"], case["repetition_policy"])
        self.assertIn("suppressed_natural_chunk_eos", {case["name"] for case in cases})

    def test_unknown_policy_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported repetition policy"):
            deterministic_sampler_case(np.zeros(32), protocol=PROTOCOL,
                                       uniforms=[0.2], repetition_policy="implicit")


class OfficialSourceSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise unittest.SkipTest("PyTorch is required for tiny official-source checks") from exc
        default_source = (Path(__file__).resolve().parents[4]
                          / "models/MiniCPM-o-4_5/utils.py")
        source = Path(os.environ.get("MINICPM_OFFICIAL_UTILS", str(default_source)))
        if not source.is_file():
            raise unittest.SkipTest("Set MINICPM_OFFICIAL_UTILS to the checkpoint's utils.py")
        tree = ast.parse(source.read_text())
        filtering = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef)
                         and node.name == "top_k_top_p_filtering")
        decoder = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "StreamDecoder")
        decode = next(node for node in decoder.body
                      if isinstance(node, ast.FunctionDef) and node.name == "decode")
        namespace = dict(torch=torch, F=functional, Literal=Literal)
        selected = ast.fix_missing_locations(ast.Module(body=[filtering, decode], type_ignores=[]))
        exec(compile(selected, str(source), "exec"), namespace)
        cls.torch = torch
        cls.decode = staticmethod(namespace["decode"])
        cls.filtering = staticmethod(namespace["top_k_top_p_filtering"])

    def compare(self, logits: np.ndarray, **options: object) -> SimpleNamespace:
        uniforms = [0.2, 0.7]
        common = dict(mode="greedy", temperature=1.0, top_k=20, top_p=0.8,
                      previous_tokens=[], forbidden=[], repetition_penalty=1.05,
                      repetition_window_size=512,
                      length_penalty=1.0, listen_top_k=None, listen_probability_scale=1.0)
        common.update(options)
        expected, draws = deterministic_sampler_case(
            logits, protocol=PROTOCOL, uniforms=uniforms,
            repetition_policy="official_duplex", **common)
        tokenizer = SimpleNamespace(decode=lambda token: "<|listen|>"
                                    if int(token.item()) == PROTOCOL["listen"] else str(token.item()))
        # Explicit listen is intentional: the raw class initializer aliases it
        # to tokenizer EOS, whereas the unified duplex protocol has its own ID.
        decoder = SimpleNamespace(tokenizer=tokenizer, listen_id=PROTOCOL["listen"],
                                  chunk_eos_id=PROTOCOL["chunk_eos"],
                                  forbidden_token_ids=list(common["forbidden"]),
                                  generated_tokens=list(common["previous_tokens"]),
                                  generated_special_tokens=[], special_token_ids=[], context="")
        draw_iterator = iter(uniforms)
        consumed = []

        def multinomial(probabilities: object, num_samples: int) -> object:
            self.assertEqual(num_samples, 1)
            draw = next(draw_iterator)
            consumed.append(draw)
            values = probabilities.reshape(-1)
            order = self.torch.argsort(values, descending=True, stable=True)
            cumulative = values[order].cumsum(0)
            rank = min(int((cumulative < draw).sum()), len(order) - 1)
            return order[rank].reshape(1)

        with mock.patch.object(self.torch, "multinomial", multinomial):
            actual = self.decode(
                decoder, self.torch.tensor(logits, dtype=self.torch.float64).reshape(1, -1),
                mode=common["mode"], temperature=common["temperature"],
                top_k=common["top_k"], top_p=common["top_p"],
                listen_top_k=common["listen_top_k"],
                listen_prob_scale=common["listen_probability_scale"],
                text_repetition_penalty=common["repetition_penalty"],
                text_repetition_window_size=common["repetition_window_size"])
        self.assertEqual(int(actual.item()), expected)
        self.assertEqual(len(consumed), draws)
        return decoder

    def test_negative_and_positive_repetition(self) -> None:
        for values, penalty in [((-1.0, -1.02), 1.05), ((5.0, 4.9), 1.05),
                                ((-1.0, -1.02), 0.9)]:
            logits = np.full(32, -100.0)
            logits[20], logits[21] = values
            self.compare(logits, previous_tokens=[20], repetition_penalty=penalty)

    def test_protocol_history_and_no_length_penalty(self) -> None:
        logits = np.full(32, -100.0)
        logits[PROTOCOL["listen"]], logits[20] = 5.1, 5.0
        self.compare(logits, previous_tokens=[PROTOCOL["listen"]])
        logits[PROTOCOL["listen"]] = -100
        logits[PROTOCOL["turn_eos"]], logits[20] = 9.5, 9.0
        self.compare(logits, length_penalty=2.0)

    def test_zero_window_uses_complete_official_history(self) -> None:
        logits = np.full(32, -100.0)
        logits[20], logits[21] = 5.1, 5.0
        self.compare(logits, previous_tokens=[20], repetition_window_size=0)

    def test_listen_scale_multiplies_the_logit(self) -> None:
        logits = np.full(32, -100.0)
        logits[PROTOCOL["listen"]], logits[20] = -1.0, -0.75
        self.compare(logits, listen_probability_scale=0.5)

    def test_early_returns_do_not_record_history_or_use_final_draw(self) -> None:
        logits = np.full(32, -100.0)
        logits[PROTOCOL["chunk_eos"]] = 100.0
        decoder = self.compare(logits, mode="sampling", forbidden=[PROTOCOL["chunk_eos"]])
        self.assertEqual(decoder.generated_tokens, [])
        logits[PROTOCOL["chunk_eos"]] = -100
        logits[PROTOCOL["listen"]] = 100
        decoder = self.compare(logits, mode="sampling", listen_top_k=1)
        self.assertEqual(decoder.generated_tokens, [])

    def test_filtered_protocol_sample_is_recorded(self) -> None:
        logits = np.full(32, -100.0)
        logits[PROTOCOL["listen"]] = 100.0
        self.assertEqual(self.compare(logits).generated_tokens, [PROTOCOL["listen"]])

    def test_official_top_k_preserves_all_boundary_ties(self) -> None:
        logits = self.torch.tensor([[3.0, 2.0, 2.0, -100.0]])
        filtered = self.filtering(logits, top_k=2, top_p=0)
        self.assertEqual(self.torch.isfinite(filtered).tolist(), [[True, True, True, False]])


if __name__ == "__main__":
    unittest.main()
