#!/usr/bin/env python3
"""Export a Python-MLX reference for the native MiniCPM-o 8-bit LLM.

The normal LLM golden exporter deliberately uses the pinned official PyTorch
BF16 implementation as its oracle.  That is the correct reference for the
dense bundle, but it is the wrong reference for a quantized bundle: affine
8-bit quantization is expected to change hidden states, logits, and KV values.

This exporter runs the *same converted 8-bit safetensors* through Python MLX
and writes the same fixture schema consumed by the Swift XCTest.  It also
records a separate quantization-error report against the existing official
BF16 fixture.  Consequently a Swift 8-bit parity failure can be distinguished
from an intentional 8-bit-vs-BF16 quality delta.

The reference intentionally uses the public ``mlx_lm`` Qwen3 implementation
and the public MLX KV cache.  The small session/window adapter below mirrors
the state transitions exercised by ``MiniCPMSession``; it does not import the
Swift implementation or use the Swift output to manufacture an oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the audited, format-stable fixture helpers.  Importing the official
# exporter does not load torch; all heavyweight imports in that script are
# lazy and the helpers below only deal with NumPy/JSON/MLX values.
from export_minicpm_o_llm_golden import (  # noqa: E402
    ArtifactWriter,
    array_hash,
    deterministic_external_embeddings,
    deterministic_sampler_case,
    output_record,
    output_records,
    protocol_special_ids,
    sha256_file,
    sha256_tree,
    to_numpy,
)


SCHEMA = "minicpm-o-llm-golden/v1"
REFERENCE_BACKEND = "python_mlx_quantized"
QUANTIZATION_REPORT_SCHEMA = "minicpm-o-llm-quantization-error/v1"
EXPECTED_BITS = 8
EXPECTED_GROUP_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-bundle", type=Path, required=True)
    parser.add_argument(
        "--official-golden-dir",
        type=Path,
        required=True,
        help="Official PyTorch BF16 fixture used only for the quantization report.",
    )
    parser.add_argument(
        "--audio-golden",
        type=Path,
        required=True,
        help="P2-A golden.safetensors containing the real projected embeddings tensor.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clone_array(value: Any) -> Any:
    """Make an eager, independent MLX copy while preserving the dtype."""

    import mlx.core as mx

    mx.eval(value)
    array = np.asarray(value.astype(mx.float32)).copy(order="C")
    return mx.array(array).astype(value.dtype)


def clone_cache(model: Any, cache: list[Any]) -> list[Any]:
    """Clone MLX KV storage, including the logical offset but not spare data."""

    from mlx_lm.models.cache import make_prompt_cache

    result = make_prompt_cache(model)
    for source, target in zip(cache, result):
        target.offset = int(source.offset)
        if source.keys is None:
            continue
        target.keys = clone_array(source.keys)
        target.values = clone_array(source.values)
        # KVCache has no other state in the scenarios below.  Preserve the
        # attributes used by newer cache implementations if they exist.
        for name in ("_idx", "max_size", "keep"):
            if hasattr(source, name) and hasattr(target, name):
                setattr(target, name, getattr(source, name))
    return result


def cache_pairs(cache: list[Any]) -> list[tuple[Any, Any] | None]:
    import mlx.core as mx

    result: list[tuple[Any, Any] | None] = []
    for item in cache:
        if item.keys is None or int(item.offset) == 0:
            result.append(None)
            continue
        result.append(
            (
                item.keys[..., : int(item.offset), :],
                item.values[..., : int(item.offset), :],
            )
        )
    mx.eval(*[value for pair in result if pair for value in pair])
    return result


def cache_length(cache: list[Any]) -> int:
    return int(cache[0].offset) if cache else 0


def digest(value: Any) -> dict[str, Any]:
    array = to_numpy(value)
    flat = array.reshape(-1)
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": "float32-le",
        "sum": float(np.sum(flat, dtype=np.float64)),
        "sum_sq": float(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64)),
        "first8": [float(item) for item in flat[:8]],
        "last8": [float(item) for item in flat[-8:]],
        "sha256": array_hash(array),
    }


def cache_checksum(cache: list[Any]) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for index, pair in enumerate(cache_pairs(cache)):
        if pair is None:
            layers.append({"layer": index, "length": 0})
            continue
        keys, values = pair
        layers.append(
            {
                "layer": index,
                "length": int(keys.shape[2]),
                "keys": digest(keys),
                "values": digest(values),
            }
        )
    return {"cache_length": cache_length(cache), "layers": layers}


def trace_digest(value: Any) -> dict[str, Any]:
    flat = to_numpy(value).reshape(-1)
    return {
        "sum": float(np.sum(flat, dtype=np.float64)),
        "sum_sq": float(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64)),
        "first8": [float(item) for item in flat[:8]],
        "last8": [float(item) for item in flat[-8:]],
        "sha256": array_hash(flat),
    }


def cache_position_trace(cache: list[Any]) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for index, pair in enumerate(cache_pairs(cache)):
        if pair is None:
            layers.append({"layer": index, "length": 0, "positions": []})
            continue
        keys, values = pair
        length = int(keys.shape[2])
        positions = []
        for position in range(length):
            positions.append(
                {
                    "position": position,
                    "keys": trace_digest(keys[..., position, :]),
                    "values": trace_digest(values[..., position, :]),
                }
            )
        layers.append({"layer": index, "length": length, "positions": positions})
    return {"cache_length": cache_length(cache), "layers": layers}


def rope_realign(keys: Any, old_start: int, new_start: int, rope_theta: float) -> Any:
    """Mirror MiniCPMCacheRoPE.realign, including FP32 trig and BF16 stores."""

    import mlx.core as mx

    length = int(keys.shape[2])
    if length == 0:
        return keys
    head_dim = int(keys.shape[3])
    half = head_dim // 2
    inverse = mx.array(
        [float(rope_theta ** (-index / half)) for index in range(half)],
        dtype=mx.float32,
    ).reshape(1, half)
    positions = mx.arange(old_start, old_start + length, dtype=mx.float32).reshape(
        length, 1
    )
    old_angles = mx.matmul(positions, inverse)
    new_positions = mx.arange(new_start, new_start + length, dtype=mx.float32).reshape(
        length, 1
    )
    new_angles = mx.matmul(new_positions, inverse)
    old_cos = mx.concatenate([mx.cos(old_angles), mx.cos(old_angles)], axis=-1)
    old_sin = mx.concatenate([mx.sin(old_angles), mx.sin(old_angles)], axis=-1)
    new_cos = mx.concatenate([mx.cos(new_angles), mx.cos(new_angles)], axis=-1)
    new_sin = mx.concatenate([mx.sin(new_angles), mx.sin(new_angles)], axis=-1)
    old_cos = old_cos.astype(keys.dtype)[None, None, ...]
    old_sin = old_sin.astype(keys.dtype)[None, None, ...]
    new_cos = new_cos.astype(keys.dtype)[None, None, ...]
    new_sin = new_sin.astype(keys.dtype)[None, None, ...]

    def rotate_half(value: Any) -> Any:
        return mx.concatenate([-value[..., half:], value[..., :half]], axis=-1)

    unrotated = old_cos * keys - old_sin * rotate_half(keys)
    return new_cos * unrotated + new_sin * rotate_half(unrotated)


class MLXReferenceDecoder:
    """Small deterministic MLX session used only by the golden exporter."""

    def __init__(self, model: Any, protocol: Mapping[str, int], rope_theta: float):
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self.protocol = protocol
        self.rope_theta = float(rope_theta)
        self._make_prompt_cache = make_prompt_cache
        self.cache: list[Any] = []
        self._rollback_cache: list[Any] | None = None
        self._unit_history: list[dict[str, Any]] = []
        self._pending_unit: dict[str, Any] | None = None
        self._system_prefix_length = 0
        self._window: dict[str, Any] = {"mode": "off"}
        self._window_enabled = False
        self._previous_marker_token_ids: list[int] = []
        self._previous_token_ids: list[int] = []
        self._previous_text = ""
        self._last_rebuild_trace: dict[str, Any] | None = None
        self.reset()

    def reset(self) -> None:
        self.cache = self._make_prompt_cache(self.model)
        self._rollback_cache = None
        self._unit_history = []
        self._pending_unit = None
        self._system_prefix_length = 0
        self._previous_token_ids = []
        self._previous_text = ""
        self._last_rebuild_trace = None

    def embed_tokens(self, token_ids: list[int]) -> Any:
        import mlx.core as mx

        ids = mx.array(token_ids, dtype=mx.int32)
        return self.model.model.embed_tokens(ids)

    def embed_token(self, token_id: int) -> Any:
        return self.embed_tokens([int(token_id)])

    def _forward(self, embeddings: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        embeddings = embeddings.astype(mx.bfloat16)
        dummy = mx.zeros((int(embeddings.shape[0]), int(embeddings.shape[1])), dtype=mx.int32)
        hidden = self.model.model(dummy, cache=self.cache, input_embeddings=embeddings)
        logits = self.model.lm_head(hidden)
        mx.eval(hidden, logits, *[value for pair in cache_pairs(self.cache) if pair for value in pair])
        return logits, hidden

    def feed(self, embeddings: Any, return_logits: bool = True) -> tuple[Any, Any]:
        return self._forward(embeddings)

    def get_cache_length(self) -> int:
        return cache_length(self.cache)

    def register_unit_start(self) -> int:
        self._rollback_cache = clone_cache(self.model, self.cache)
        self._pending_unit = {"start": self.get_cache_length()}
        return self.get_cache_length()

    def register_unit_end(self, *, input_type: str = "audio", generated_tokens: list[int] | None = None, generated_text: str = "") -> None:
        if self._pending_unit is None:
            return
        self._pending_unit.update(
            {
                "input_type": input_type,
                "generated_tokens": [int(item) for item in (generated_tokens or [])],
                "generated_text": generated_text,
                "length": self.get_cache_length() - int(self._pending_unit["start"]),
            }
        )

    def commit_pending_unit(self) -> None:
        if self._pending_unit is None:
            return
        item = dict(self._pending_unit)
        item["unit_id"] = len(self._unit_history)
        self._unit_history.append(item)
        self._pending_unit = None
        self._rollback_cache = None

    def rollback_pending_unit(self) -> int:
        if self._rollback_cache is None:
            return 0
        before = self.get_cache_length()
        self.cache = self._rollback_cache
        self._rollback_cache = None
        self._pending_unit = None
        return before - self.get_cache_length()

    def trim_cache_to(self, target_length: int) -> int:
        before = self.get_cache_length()
        target = max(0, min(int(target_length), before))
        for item in self.cache:
            if item.keys is None:
                continue
            item.offset = target
        return before - target

    def register_system_prompt(self) -> None:
        self._system_prefix_length = self.get_cache_length()

    def set_window_config(self, config: Mapping[str, Any]) -> None:
        self._window = dict(config)

    def set_window_enabled(self, enabled: bool) -> None:
        self._window_enabled = bool(enabled)

    def enforce_window(self) -> bool:
        if not self._window_enabled or self._window.get("sliding_window_mode") != "basic":
            return False
        high = int(self._window.get("basic_window_high_tokens", 4000))
        low = int(self._window.get("basic_window_low_tokens", 3500))
        triggered = False
        if self.get_cache_length() <= high:
            return False
        triggered = True
        while self.get_cache_length() > low:
            index = next((i for i, item in enumerate(self._unit_history) if item.get("input_type") != "system"), None)
            if index is None:
                break
            unit = self._unit_history[index]
            length = int(unit.get("length", 0))
            if length <= 0:
                self._unit_history.pop(index)
                continue
            if self.get_cache_length() - length < low:
                # The Swift implementation stops when the low watermark is
                # reached; this keeps a whole unit rather than splitting it.
                break
            self._drop_cache_segment(self._system_prefix_length, length)
            self._unit_history.pop(index)
        return triggered

    def _cache_from_pairs(self, pairs: list[tuple[Any, Any] | None]) -> list[Any]:
        result = self._make_prompt_cache(self.model)
        for item, pair in zip(result, pairs):
            if pair is None:
                continue
            keys, values = pair
            item.keys = keys
            item.values = values
            item.offset = int(keys.shape[2])
        return result

    def _drop_cache_segment(self, preserve: int, length: int) -> None:
        current = self.get_cache_length()
        pairs: list[tuple[Any, Any] | None] = []
        for pair in cache_pairs(self.cache):
            if pair is None:
                pairs.append(None)
                continue
            keys, values = pair
            prefix_keys = keys[..., :preserve, :]
            prefix_values = values[..., :preserve, :]
            suffix_start = preserve + length
            suffix_keys = keys[..., suffix_start:current, :]
            suffix_values = values[..., suffix_start:current, :]
            reindexed = rope_realign(
                suffix_keys, suffix_start, preserve, self.rope_theta
            )
            pairs.append(
                (
                    mx_concatenate([prefix_keys, reindexed], axis=2),
                    mx_concatenate([prefix_values, suffix_values], axis=2),
                )
            )
        self.cache = self._cache_from_pairs(pairs)

    def register_system_prompt_with_context(self, *, suffix_token_ids: list[int]) -> None:
        self._system_prefix_length = self.get_cache_length()
        self._previous_marker_token_ids = [271, 19702, 25, 220]
        self._context_suffix_token_ids = [int(item) for item in suffix_token_ids]

    def get_previous_context(self) -> tuple[str, list[int]]:
        return self._previous_text, list(self._previous_token_ids)

    def _rebuild_context(
        self,
        *,
        prefix_length: int,
        previous_tokens: list[int],
        suffix_token_ids: list[int],
        retained_length: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        current = self.get_cache_length()
        retained_start = current - retained_length
        old_pairs = cache_pairs(self.cache)
        prefix_pairs = [
            (pair[0][..., :prefix_length, :], pair[1][..., :prefix_length, :])
            if pair is not None
            else None
            for pair in old_pairs
        ]
        retained_pairs = [
            (pair[0][..., retained_start:current, :], pair[1][..., retained_start:current, :])
            if pair is not None
            else None
            for pair in old_pairs
        ]
        prefix_cache = self._cache_from_pairs(
            [
                (clone_array(pair[0]), clone_array(pair[1])) if pair is not None else None
                for pair in prefix_pairs
            ]
        )
        # ``KVCache`` appends into its backing arrays during ``feed``.  Keep
        # the prefix trace before installing this cache as the active cache;
        # otherwise the trace aliases the rebuilt cache and reports the
        # prefix-plus-segment length instead of the original prefix length.
        prefix_trace = cache_position_trace(prefix_cache)
        segment_ids = [int(item) for item in previous_tokens] + [int(item) for item in suffix_token_ids]
        old_cache = self.cache
        self.cache = prefix_cache
        if segment_ids:
            self.feed(self.embed_tokens(segment_ids))
        recomputed_cache = self.cache
        new_system_length = self.get_cache_length()
        reindexed_pairs: list[tuple[Any, Any] | None] = []
        final_pairs: list[tuple[Any, Any] | None] = []
        for pair in retained_pairs:
            if pair is None:
                reindexed_pairs.append(None)
                final_pairs.append(None)
                continue
            keys, values = pair
            reindexed = rope_realign(
                keys, retained_start, new_system_length, self.rope_theta
            )
            reindexed_pairs.append((reindexed, values))
        for head, tail in zip(cache_pairs(recomputed_cache), reindexed_pairs):
            if head is None:
                final_pairs.append(tail)
            elif tail is None:
                final_pairs.append(head)
            else:
                final_pairs.append(
                    (
                        mx_concatenate([head[0], tail[0]], axis=2),
                        mx_concatenate([head[1], tail[1]], axis=2),
                    )
                )
        self.cache = old_cache
        trace = {
            "source": cache_position_trace(old_cache),
            "prefix": prefix_trace,
            "retained_original": cache_position_trace(
                self._cache_from_pairs(retained_pairs)
            ),
            "retained_start": retained_start,
            "prefix_length": prefix_length,
            "retained_length": retained_length,
            "recomputed_previous_suffix": cache_position_trace(recomputed_cache),
            "retained_reindexed": cache_position_trace(
                self._cache_from_pairs(reindexed_pairs)
            ),
            "final": cache_position_trace(self._cache_from_pairs(final_pairs)),
        }
        return self._cache_from_pairs(final_pairs), trace

    def context_rebuild_trace(
        self,
        *,
        prefix_length: int,
        previous_tokens: list[int],
        suffix_token_ids: list[int],
        retained_length: int,
    ) -> dict[str, Any]:
        _, trace = self._rebuild_context(
            prefix_length=prefix_length,
            previous_tokens=previous_tokens,
            suffix_token_ids=suffix_token_ids,
            retained_length=retained_length,
        )
        return trace

    def enforce_window_with_context(self) -> bool:
        if not self._window_enabled or self._window.get("sliding_window_mode") != "context":
            return False
        maximum = int(self._window.get("context_max_units", 24))
        triggered = False
        while len(self._unit_history) > maximum:
            index = next((i for i, item in enumerate(self._unit_history) if item.get("input_type") != "system"), None)
            if index is None:
                break
            removed = self._unit_history.pop(index)
            if not removed.get("generated_tokens"):
                continue
            marker = list(self._previous_marker_token_ids)
            content = [int(item) for item in removed.get("generated_tokens", [])]
            if not self._previous_token_ids:
                self._previous_token_ids = marker + content
            else:
                self._previous_token_ids.extend(content)
            max_tokens = int(self._window.get("context_previous_max_tokens", 500))
            self._previous_token_ids = marker + self._previous_token_ids[len(marker):][-max_tokens:]
            self._previous_text += str(removed.get("generated_text", ""))
            if len(self._previous_text) > max_tokens * 4:
                self._previous_text = self._previous_text[-max_tokens * 4 :]
            retained = sum(int(item.get("length", 0)) for item in self._unit_history)
            self.cache, self._last_rebuild_trace = self._rebuild_context(
                prefix_length=self._system_prefix_length,
                previous_tokens=self._previous_token_ids,
                suffix_token_ids=getattr(self, "_context_suffix_token_ids", []),
                retained_length=retained,
            )
            triggered = True
        return triggered


def mx_concatenate(values: list[Any], axis: int) -> Any:
    import mlx.core as mx

    values = [value for value in values if int(value.shape[axis]) > 0]
    if not values:
        raise ValueError("cannot concatenate an empty cache segment")
    return values[0] if len(values) == 1 else mx.concatenate(values, axis=axis)


def output_manifest_record(writer: ArtifactWriter, name: str, output: tuple[Any, Any], selected: list[int]) -> dict[str, Any]:
    return output_record(writer, name, output, selected)


def output_manifest_records(writer: ArtifactWriter, name: str, outputs: list[tuple[Any, Any]], selected: list[int]) -> dict[str, Any]:
    return output_records(writer, name, outputs, selected)


def load_float_array(root: Path, manifest: Mapping[str, Any], name: str) -> np.ndarray:
    spec = manifest["arrays"][name]
    return np.fromfile(root / spec["path"], dtype="<f4").reshape(spec["shape"])


def output_metrics(
    output_root: Path,
    writer: ArtifactWriter,
    record: Mapping[str, Any],
    official_root: Path,
    official_manifest: Mapping[str, Any],
    official_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one MLX output record to the official BF16 record."""

    def metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        expected = np.asarray(expected, dtype=np.float64).reshape(-1)
        delta = actual - expected
        expected_norm = float(np.linalg.norm(expected))
        actual_norm = float(np.linalg.norm(actual))
        cosine = (
            float(np.dot(actual, expected) / (actual_norm * expected_norm))
            if actual_norm > 0 and expected_norm > 0
            else (1.0 if np.all(delta == 0) else 0.0)
        )
        return {
            "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "relative_l2": float(np.linalg.norm(delta) / max(expected_norm, 1e-12)),
            "cosine": cosine,
        }

    def read_writer_array(name: str) -> np.ndarray:
        spec = writer.arrays[name]
        return np.fromfile(output_root / spec["path"], dtype="<f4").reshape(spec["shape"])

    actual_hidden_name = str(record["hidden"])
    actual_logits_name = str(record["logits_selected"])
    expected_hidden_name = str(official_record["hidden"])
    expected_logits_name = str(official_record["logits_selected"])
    actual_hidden = read_writer_array(actual_hidden_name)
    actual_logits = read_writer_array(actual_logits_name)
    expected_hidden = load_float_array(official_root, official_manifest, expected_hidden_name)
    expected_logits = load_float_array(official_root, official_manifest, expected_logits_name)
    actual_top = read_writer_array(str(record["top20"])).astype(np.int64)
    expected_top = load_float_array(official_root, official_manifest, str(official_record["top20"])).astype(np.int64)
    overlaps = []
    for a, e in zip(actual_top.reshape(-1, actual_top.shape[-1]), expected_top.reshape(-1, expected_top.shape[-1])):
        overlaps.append(int(len(set(a.tolist()).intersection(e.tolist()))))
    return {
        "hidden": metric(actual_hidden, expected_hidden).copy(),
        "logits_selected": metric(actual_logits, expected_logits).copy(),
        "top20_overlap": {"min": min(overlaps) if overlaps else 0, "rows": overlaps},
    }


def load_audio_embeddings(path: Path) -> tuple[np.ndarray, str]:
    import mlx.core as mx

    arrays, metadata = mx.load(str(path), return_metadata=True)
    del metadata
    if "embeddings" not in arrays:
        raise ValueError(f"audio golden has no embeddings tensor: {path}")
    value = to_numpy(arrays["embeddings"])
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 4096:
        raise ValueError(f"audio embeddings must be [1,T,4096], got {value.shape}")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def make_sampling_cases(writer: ArtifactWriter, protocol: Mapping[str, int]) -> list[dict[str, Any]]:
    """Keep sampler coverage identical to the official fixture."""

    width = max(32, max(int(value) for value in protocol.values()) + 1)
    cases: list[dict[str, Any]] = []

    def add(name: str, logits: np.ndarray, **kwargs: Any) -> None:
        uniforms = list(kwargs.pop("uniforms"))
        expected, draws = deterministic_sampler_case(
            logits, protocol=protocol, uniforms=uniforms, **kwargs
        )
        logits_name = writer.array(f"sampling_{name}_logits", logits)
        cases.append(
            {
                "name": name,
                "logits": logits_name,
                "uniforms": uniforms,
                "expected_token": expected,
                "draw_count": draws,
                **kwargs,
            }
        )

    base = np.full(width, -10.0, dtype=np.float32)
    values = base.copy(); values[protocol["chunk_eos"]] = 10
    add("natural_chunk_eos", values, uniforms=[0.2], mode="sampling", temperature=0.7, top_k=20, top_p=0.8, forbidden=[protocol["chunk_eos"]], previous_tokens=[], repetition_penalty=1.05, length_penalty=1.1, listen_top_k=None, listen_probability_scale=1.0)
    values = base.copy(); values[0] = 10; values[1] = 9
    add("forbidden_after_probe", values, uniforms=[0.1, 0.1], mode="sampling", temperature=1.0, top_k=0, top_p=1.0, forbidden=[0], previous_tokens=[], repetition_penalty=1.0, length_penalty=1.0, listen_top_k=None, listen_probability_scale=1.0)
    values = base.copy(); values[20] = 5; values[21] = 4.9
    add("repetition_penalty", values, uniforms=[0.2, 0.2], mode="sampling", temperature=1.0, top_k=2, top_p=1.0, forbidden=[], previous_tokens=[20], repetition_penalty=2.0, length_penalty=1.0, listen_top_k=None, listen_probability_scale=1.0)
    values = base.copy(); values[protocol["listen"]] = 4; values[10] = 3
    add("listen_rank", values, uniforms=[0.2, 0.2], mode="sampling", temperature=1.0, top_k=20, top_p=1.0, forbidden=[], previous_tokens=[], repetition_penalty=1.0, length_penalty=1.0, listen_top_k=2, listen_probability_scale=1.0)
    values = np.full(width, -10.0, dtype=np.float32); values[0:3] = [3, 2, 1]
    add("top_p_boundary", values, uniforms=[0.2, 0.99], mode="sampling", temperature=1.0, top_k=0, top_p=0.5, forbidden=[], previous_tokens=[], repetition_penalty=1.0, length_penalty=1.0, listen_top_k=None, listen_probability_scale=1.0)
    return cases


def main() -> None:
    args = parse_args()
    bundle = args.mlx_bundle.expanduser().resolve()
    official_root = args.official_golden_dir.expanduser().resolve()
    audio_path = args.audio_golden.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (bundle, official_root, audio_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        if not output.is_dir():
            raise SystemExit(f"output is not a directory: {output}")
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)

    official_manifest = read_json(official_root / "manifest.json")
    bundle_config = read_json(bundle / "config.json")
    quant = bundle_config.get("quantization") or bundle_config.get("quantization_config") or {}
    bits = int(quant.get("bits", -1))
    group_size = int(quant.get("group_size", 0))
    if bits != EXPECTED_BITS or group_size != EXPECTED_GROUP_SIZE or str(quant.get("mode")) != "affine":
        raise SystemExit("--mlx-bundle must be the pinned affine 8-bit/group-64 bundle")

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer, mlx_config = load(
        str(bundle),
        tokenizer_config={"trust_remote_code": True},
        return_config=True,
        lazy=True,
    )
    del mlx_config
    protocol = {str(key): int(value) for key, value in (official_manifest.get("protocol_ids") or {}).items()}
    if int(tokenizer.eos_token_id) != protocol["eos"]:
        raise SystemExit("bundle tokenizer EOS disagrees with the official fixture")
    selected_indices = [int(item) for item in (official_manifest["scenarios"]["prefill"]["output"]["selected_indices"])]
    special_ids = protocol_special_ids(tokenizer, protocol)
    decoder = MLXReferenceDecoder(model, protocol, float(bundle_config["rope_theta"]))
    writer = ArtifactWriter(output)

    token_ids = [151643, 198, 108, protocol["eos"]]
    token_embeddings = to_numpy(decoder.embed_tokens(token_ids))
    token_embeddings_name = writer.array("token_embeddings", token_embeddings)
    external = deterministic_external_embeddings(int(bundle_config["hidden_size"]))
    external_name = writer.array("external_embeddings", external)
    token_ids_name = writer.array("token_ids", np.asarray(token_ids, dtype=np.int32), dtype="i32")

    decoder.reset()
    prefill_output = decoder.feed(decoder.embed_tokens(token_ids))
    prefill_record = output_manifest_record(writer, "prefill", prefill_output, selected_indices)
    prefill_kv = cache_checksum(decoder.cache)

    decoder.reset()
    incremental_outputs: list[tuple[Any, Any]] = []
    incremental_kv: list[dict[str, Any]] = []
    for token in token_ids:
        incremental_outputs.append(decoder.feed(decoder.embed_token(token)))
        incremental_kv.append(cache_checksum(decoder.cache))
    incremental_record = output_manifest_records(writer, "incremental", incremental_outputs, selected_indices)

    decoder.reset()
    external_record = output_manifest_record(writer, "external", decoder.feed(mx.array(external)), selected_indices)

    decoder.reset()
    rollback_prefix = [151643, 198]
    rollback_speculative = [protocol["listen"], protocol["speak"]]
    rollback_canonical = [protocol["unit"], protocol["unit_end"]]
    decoder.feed(decoder.embed_tokens(rollback_prefix))
    rollback_cache_before = decoder.get_cache_length()
    decoder.register_unit_start()
    for token in rollback_speculative:
        decoder.feed(decoder.embed_token(token))
    rollback_cache_speculative = decoder.get_cache_length()
    rollback_removed = decoder.rollback_pending_unit()
    rollback_cache_after = decoder.get_cache_length()
    rollback_outputs = [decoder.feed(decoder.embed_token(token)) for token in rollback_canonical]
    rollback_record = output_manifest_records(writer, "rollback_canonical", rollback_outputs, selected_indices)

    decoder.reset()
    trim_prefix = [151643, 198, 108, protocol["eos"]]
    decoder.feed(decoder.embed_tokens(trim_prefix))
    trim_target = 2
    trim_removed = decoder.trim_cache_to(trim_target)
    trim_probe = decoder.feed(decoder.embed_token(protocol["unit"]))
    trim_record = output_manifest_record(writer, "trim_probe", trim_probe, selected_indices)
    trim_cache_after_probe = decoder.get_cache_length()

    decoder.reset()
    decoder.set_window_config({"sliding_window_mode": "basic", "basic_window_high_tokens": 4, "basic_window_low_tokens": 2})
    decoder.set_window_enabled(True)
    decoder.feed(decoder.embed_token(151643))
    decoder.register_system_prompt()
    basic_units = [198, 108, 151683, 151705, 151706]
    basic_events: list[dict[str, Any]] = []
    for unit_id, token in enumerate(basic_units):
        before_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        before_cache = decoder.get_cache_length()
        decoder.register_unit_start()
        decoder.feed(decoder.embed_token(token))
        decoder.register_unit_end(input_type="audio", generated_tokens=[token])
        decoder.commit_pending_unit()
        triggered = decoder.enforce_window()
        after_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        if triggered:
            basic_events.append({"unit_id": unit_id, "triggered": True, "before_unit_ids": before_ids, "after_unit_ids": after_ids, "dropped_unit_ids": [item for item in before_ids if item not in after_ids], "cache_before": before_cache + 1, "cache_after": decoder.get_cache_length()})
    basic_probe = decoder.feed(decoder.embed_token(protocol["unit_end"]))
    basic_probe_record = output_manifest_record(writer, "sliding_basic_probe", basic_probe, selected_indices)
    basic_kv = cache_checksum(decoder.cache)

    decoder.reset()
    decoder.set_window_config({"sliding_window_mode": "context", "context_max_units": 1, "context_previous_max_tokens": 8})
    decoder.set_window_enabled(True)
    decoder.feed(decoder.embed_token(151643))
    decoder.register_system_prompt_with_context(suffix_token_ids=[protocol["eos"]])
    context_trace: dict[str, Any] = {"prefix": cache_position_trace(decoder.cache)}
    decoder.feed(decoder.embed_token(protocol["eos"]))
    context_trace["prefix_suffix"] = cache_position_trace(decoder.cache)
    marker_ids = list(decoder._previous_marker_token_ids)
    context_units = [{"token": 198, "generated": [20, 21], "text": "first"}, {"token": 108, "generated": [22], "text": "second"}]
    context_events: list[dict[str, Any]] = []
    for unit_id, unit in enumerate(context_units):
        before_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        decoder.register_unit_start()
        decoder.feed(decoder.embed_token(unit["token"]))
        decoder.register_unit_end(input_type="audio", generated_tokens=unit["generated"], generated_text=unit["text"])
        decoder.commit_pending_unit()
        context_trace[f"before_enforce_unit_{unit_id}"] = cache_position_trace(decoder.cache)
        if unit_id == 1:
            context_trace["rebuild"] = decoder.context_rebuild_trace(prefix_length=1, previous_tokens=marker_ids + context_units[0]["generated"], suffix_token_ids=[protocol["eos"]], retained_length=1)
        triggered = decoder.enforce_window_with_context()
        after_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        if triggered:
            previous_text, previous_ids = decoder.get_previous_context()
            context_events.append({"unit_id": unit_id, "triggered": True, "before_unit_ids": before_ids, "after_unit_ids": after_ids, "dropped_unit_ids": [item for item in before_ids if item not in after_ids], "cache_after": decoder.get_cache_length(), "previous_text": previous_text, "previous_token_ids": previous_ids})
            context_trace["final"] = cache_position_trace(decoder.cache)
    context_probe = decoder.feed(decoder.embed_token(protocol["unit_end"]))
    context_probe_record = output_manifest_record(writer, "sliding_context_probe", context_probe, selected_indices)
    context_kv = cache_checksum(decoder.cache)
    context_trace["after_probe"] = cache_position_trace(decoder.cache)

    audio_embeddings, audio_sha = load_audio_embeddings(audio_path)
    audio_name = writer.array("audio_bridge_embeddings", audio_embeddings)
    bridge_cases: list[dict[str, Any]] = []

    def bridge_case(name: str, operations: list[tuple[str, int | None]], *, repair: bool = False) -> None:
        decoder.reset()
        outputs: list[tuple[Any, Any]] = []
        for kind, value in operations:
            if kind == "audio":
                outputs.append(decoder.feed(mx.array(audio_embeddings)))
            else:
                assert value is not None
                outputs.append(decoder.feed(decoder.embed_token(value)))
        if repair:
            decoder.reset()
            outputs = [decoder.feed(mx.array(audio_embeddings))]
            decoder.register_unit_start()
            for token in (protocol["listen"], protocol["speak"]):
                decoder.feed(decoder.embed_token(token))
            decoder.rollback_pending_unit()
            outputs.extend(decoder.feed(decoder.embed_token(token)) for token in (protocol["turn_eos"], protocol["unit_end"]))
        record_name = f"audio_bridge_{name}"
        record = output_manifest_records(writer, record_name, outputs, selected_indices)
        cache_name = writer.json(f"kv_audio_bridge_{name}", cache_checksum(decoder.cache))
        bridge_cases.append({"name": name, "audio_array": audio_name, "operations": [{"kind": kind, **({"token_id": value} if value is not None else {})} for kind, value in operations], "repair": repair, "output": record, "cache": cache_name})

    bridge_case("first_audio_unit", [("audio", None)])
    bridge_case("listen_unit", [("audio", None), ("token", protocol["listen"])])
    bridge_case("spoken_continuation", [("audio", None), ("token", protocol["speak"]), ("token", protocol["tts_bos"])])
    bridge_case("repair_next_unit", [("audio", None)], repair=True)

    sampling_cases = make_sampling_cases(writer, protocol)

    # The MLX fixture is intentionally compared with the existing official
    # fixture only in this report.  The XCTest reads this fixture's own output
    # arrays as its strict 8-bit oracle.
    scenarios = {
        "prefill": {"token_ids": token_ids, "cache": writer.json("kv_prefill", prefill_kv), "output": prefill_record},
        "incremental": {"token_ids": token_ids, "cache": writer.json("kv_incremental", incremental_kv), "output": incremental_record, "batch_last_step": len(token_ids) - 1},
        "external": {"shape": list(external.shape), "output": external_record},
        "rollback": {"prefix_token_ids": rollback_prefix, "speculative_token_ids": rollback_speculative, "canonical_token_ids": rollback_canonical, "cache_before": rollback_cache_before, "cache_speculative": rollback_cache_speculative, "cache_after": rollback_cache_after, "removed": rollback_removed, "output": rollback_record},
        "trim": {"prefix_token_ids": trim_prefix, "target_length": trim_target, "removed": trim_removed, "probe_token_id": protocol["unit"], "cache_after_probe": trim_cache_after_probe, "output": trim_record},
        "sliding_basic": {"config": {"mode": "basic", "high": 4, "low": 2}, "prefix_token_ids": [151643], "units": [{"input_token_id": token, "generated_tokens": [token]} for token in basic_units], "events": basic_events, "cache": writer.json("kv_sliding_basic", basic_kv), "probe_token_id": protocol["unit_end"], "output": basic_probe_record},
        "sliding_context": {"config": {"mode": "context", "max_units": 1, "previous_max_tokens": 8}, "prefix_token_ids": [151643], "suffix_token_ids": [protocol["eos"]], "marker_token_ids": marker_ids, "units": context_units, "events": context_events, "cache": writer.json("kv_sliding_context", context_kv), "probe_token_id": protocol["unit_end"], "output": context_probe_record, "trace": context_trace},
        "sampling": {"vocab_size": max(int(value) for value in protocol.values()) + 1, "cases": sampling_cases},
        "audio_bridge": {"audio_array": audio_name, "audio_shape": list(audio_embeddings.shape), "audio_golden_sha256": audio_sha, "cases": bridge_cases},
    }

    # Build quantization report from the output arrays and the official BF16
    # arrays.  Keep the thresholds explicit but separate from Swift parity
    # thresholds: these numbers describe model-quality loss, not backend
    # agreement.
    quantization_report: dict[str, Any] = {
        "schema": QUANTIZATION_REPORT_SCHEMA,
        "under_test": {"backend": REFERENCE_BACKEND, "bundle": bundle.name, "bits": bits, "group_size": group_size},
        "reference": {"backend": official_manifest.get("oracle_backend", "official_pytorch"), "fixture": official_root.name},
        "scenarios": {},
        "thresholds": {"hidden_max_abs": 16.0, "hidden_relative_l2": 0.75, "logits_selected_cosine": 0.85, "top20_overlap_min": 0},
    }
    for name in ["prefill", "incremental", "external", "rollback", "trim", "sliding_basic", "sliding_context"]:
        record = scenarios[name]["output"]
        official_record = official_manifest["scenarios"][name]["output"]
        quantization_report["scenarios"][name] = output_metrics(output, writer, record, official_root, official_manifest, official_record)
    thresholds = quantization_report["thresholds"]
    passed = True
    for value in quantization_report["scenarios"].values():
        hidden = value["hidden"]
        logits = value["logits_selected"]
        overlap = int(value["top20_overlap"]["min"])
        passed = passed and hidden["max_abs"] <= thresholds["hidden_max_abs"] and hidden["relative_l2"] <= thresholds["hidden_relative_l2"] and logits["cosine"] >= thresholds["logits_selected_cosine"] and overlap >= thresholds["top20_overlap_min"]
    quantization_report["status"] = "passed" if passed else "failed"
    quantization_report_name = "quantization_error.json"
    write_json(output / quantization_report_name, quantization_report)

    repo_root = SCRIPT_DIR.parent
    config_path = bundle / "config.json"
    manifest = {
        "schema": SCHEMA,
        "oracle_backend": REFERENCE_BACKEND,
        "source": official_manifest.get("source", {}),
        "model": {
            "bundle_name": bundle.name,
            "repo_relative_path": bundle.relative_to(repo_root).as_posix() if bundle.is_relative_to(repo_root) else None,
            "tree_sha256": sha256_tree(bundle),
            "config_sha256": sha256_file(config_path),
            "config_file": "config.json",
            "quantization": {"bits": bits, "group_size": group_size, "mode": str(quant.get("mode"))},
            "hidden_size": int(bundle_config["hidden_size"]),
            "vocab_size": int(bundle_config["vocab_size"]),
        },
        "converter": official_manifest.get("converter", {}),
        "runtime": {
            "oracle_backend": REFERENCE_BACKEND,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "cache_class": "mlx_lm.models.cache.KVCache",
            "attention_implementation": "mlx_lm.qwen3.scaled_dot_product_attention",
            "seed": args.seed,
            "official_fixture": official_root.name,
            "audio_golden": audio_path.name,
        },
        "protocol_ids": protocol,
        "protocol_special_ids": special_ids,
        "inputs": {"token_ids": token_ids_name, "token_ids_values": token_ids, "token_embeddings": token_embeddings_name, "external_embeddings": external_name},
        "arrays": writer.arrays,
        "scenarios": scenarios,
        "quantization_report": quantization_report_name,
        "tolerances": {
            "bf16": {"hidden_max_abs": 2e-3, "logits_cosine": 0.995, "top20_overlap": 19},
            "mlx_8bit": {"hidden_max_abs": 0.02, "logits_cosine": 0.995, "top20_overlap": 20},
            "kv_checksum": {"sum_abs": 0.2, "sum_sq_abs": 1.0},
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "arrays": len(writer.arrays), "schema": SCHEMA, "oracle_backend": REFERENCE_BACKEND, "quantization_report": quantization_report_name, "quantization_status": quantization_report["status"]}, indent=2))


if __name__ == "__main__":
    main()
