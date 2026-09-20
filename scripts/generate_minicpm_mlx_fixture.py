#!/usr/bin/env python3
"""Generate the small cross-language MiniCPM MLX oracle fixture.

This is intentionally independent of the 8B checkpoint.  A one-layer Qwen3
configuration keeps the committed safetensors tiny while exercising the exact
packed embedding/linear keys, external-embedding path, KV state, and
snapshot/rollback protocol used by the production bundle.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from minicpm_mlx.llm import MLXStreamDecoder, Qwen3MLXConfig  # noqa: E402


class FixtureTokenizer:
    _ids = {
        "<|listen|>": 5,
        "<|chunk_eos|>": 6,
        "<|chunk_tts_eos|>": 7,
        "<|turn_eos|>": 8,
        "<|speak|>": 9,
        "</unit>": 10,
        "<|tts_bos|>": 11,
    }

    def convert_tokens_to_ids(self, token):
        return self._ids[token]


def _numpy(value):
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def _sha256(path: Path) -> str | None:
    """Return a source fingerprint, or ``None`` when the source is absent.

    The oracle is intentionally tied to the exact upstream implementation
    used to produce it.  Recording hashes prevents a future mlx-lm update or
    an edited local decoder from silently changing the reference numbers.
    """

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "Tests" / "MiniCPMLLMTests" / "Fixtures"
    output.mkdir(parents=True, exist_ok=True)
    config = Qwen3MLXConfig(
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=32,
        rope_theta=1_000_000,
        tie_word_embeddings=False,
    )
    mx.random.seed(20260804)
    model = Model(config.to_model_args())
    # Match production MLX affine 8bit layout exactly.
    nn.quantize(model, group_size=32, bits=8, mode="affine")
    mx.eval(model.parameters())
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(output / "model.safetensors"), flat, metadata={"format": "mlx"})
    raw_config = {
        **vars(config),
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "quantization": {"bits": 8, "group_size": 32, "mode": "affine"},
        "quantization_config": {"bits": 8, "group_size": 32, "mode": "affine"},
    }
    (output / "config.json").write_text(json.dumps(raw_config, indent=2) + "\n")

    decoder = MLXStreamDecoder(
        output,
        FixtureTokenizer(),
        quantize_bits=0,
        config=config,
        model=model,
    )
    token_ids = [1, 2]
    logits, hidden = decoder.feed(decoder.embed_tokens(token_ids), return_logits=True)
    mx.eval(logits, hidden)
    external = (mx.arange(64, dtype=mx.float32).reshape(2, 32) - 31.5) / 32
    external_logits, external_hidden = decoder.feed(external, return_logits=True)
    mx.eval(external_logits, external_hidden)

    decoder.register_unit_start()
    rollback_input = (mx.arange(32, dtype=mx.float32).reshape(1, 32) + 0.25) / 16
    rollback_logits, rollback_hidden = decoder.feed(rollback_input, return_logits=True)
    mx.eval(rollback_logits, rollback_hidden)
    cache_before_rollback = decoder.get_cache_length()
    removed = decoder.rollback_pending_unit()
    cache_after_rollback = decoder.get_cache_length()

    def record(logit, hid):
        logit_array = _numpy(logit)
        hidden_array = _numpy(hid)
        values = logit_array.reshape(-1, logit_array.shape[-1])[-1].tolist()
        return {
            "hidden": hidden_array.reshape(-1, hidden_array.shape[-1])[-1].tolist(),
            "logits": values,
            "top20": np.argsort(-np.asarray(values))[:20].astype(int).tolist(),
        }

    runtime_root = Path(__file__).resolve().parents[1]
    source_metadata = {
        "fixture_script_sha256": _sha256(Path(__file__).resolve()),
        "python_decoder_sha256": _sha256(ROOT / "minicpm_mlx" / "llm.py"),
        "mlx_lm_qwen3_sha256": _sha256(Path(inspect.getfile(Model))),
        "swift_llm_sha256": _sha256(
            runtime_root / "Sources" / "MiniCPMLLM" / "MiniCPMLanguageModel.swift"
        ),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
    }
    oracle = {
        "metadata": {"source": source_metadata},
        "token_ids": token_ids,
        "external_embeddings": _numpy(external).tolist(),
        "steps": [record(logits, hidden), record(external_logits, external_hidden)],
        "rollback": {
            "cache_before": cache_before_rollback,
            "cache_after": cache_after_rollback,
            "removed": removed,
            "step": record(rollback_logits, rollback_hidden),
        },
        "config": raw_config,
    }
    (output / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    print(json.dumps({"output": str(output), "tensors": len(flat), "cache": cache_after_rollback}, indent=2))


if __name__ == "__main__":
    main()
