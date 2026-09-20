#!/usr/bin/env python3
"""Export a deterministic MiniCPM-o Qwen3/MLX LLM golden fixture.

The official MiniCPM-o checkpoint contains several independent models.  This
fixture intentionally exercises only the Qwen3 language backbone and the
state/transaction policy around it.  Audio, vision and TTS embeddings can be
fed into the same API later, but mixing them into this oracle would make a
cache or protocol regression impossible to localise.

The exporter is deliberately *re-generatable*.  It records logical model
identifiers and content hashes, never a machine-local absolute path.  The
expected values come from the pinned upstream PyTorch Qwen3 LLM and official
``StreamDecoder``; the pre-converted MLX bundle is recorded only as the model
under test.  The bundle may be dense BF16 or the native affine 8-bit export;
neither bundle is used to produce the oracle values.  The exporter writes
little-endian float32 arrays plus compact selected-logit/top-k records.  A real
8B export is expensive; callers must invoke this script explicitly.  When
``--audio-golden`` is supplied, the fixture also covers the fixed P2-A
projected audio embedding through the four audio→LLM bridge transactions.

Example::

    .venv-minicpmo/bin/python scripts/export_minicpm_o_llm_golden.py \
      --source-model models/MiniCPM-o-4_5 \
      --mlx-bundle models/MiniCPM-o-4_5-llm-mlx-bf16 \
      --output test-output/minicpm-llm-golden-bf16 \
      --source-revision 073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1 \
      --overwrite

The Swift XCTest consumes ``manifest.json`` and the files it references.  No
full 151748-wide logits or complete KV tensors are written: every tensor has a
shape/dtype/hash, while logits additionally contain selected protocol IDs and
top-k indices/values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PINNED_UPSTREAM_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
PINNED_MODEL_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEFAULT_MODEL_ID = "OpenBMB/MiniCPM-o-4_5"
SCHEMA = "minicpm-o-llm-golden/v1"
EXPECTED_EOS_ID = 151645
EXPECTED_LISTEN_ID = 151705
_TORCHVISION_COMPAT_LIB: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--mlx-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Optional HF/local weight revision; never use the Demo source commit here.",
    )
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=Path("/tmp/MiniCPM-o-Demo"),
        help="Pinned upstream checkout used for source hashes.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().with_name("convert_minicpm_o_llm_to_mlx.py"),
        help="Converter whose content hash is recorded in the manifest.",
    )
    parser.add_argument(
        "--audio-golden",
        type=Path,
        default=None,
        help=(
            "Optional P2-A golden.safetensors containing the real projected "
            "audio embeddings tensor."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory (never touches model weights).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory deterministically without recording its absolute path."""

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "little"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return value.strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def protocol_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve protocol markers independently from the normal EOS token.

    ``StreamDecoder`` in older upstream revisions used ``tokenizer.eos`` for
    listen, while the MiniCPM-o unified duplex protocol uses the explicit
    ``<|listen|>`` marker.  The two IDs are different in the pinned checkpoint
    and this assertion prevents a stale tokenizer or accidental alias from
    producing a plausible but invalid fixture.
    """

    def token_id(token: str) -> int:
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None or int(value) < 0:
            raise ValueError(f"tokenizer is missing {token}")
        return int(value)

    eos = int(getattr(tokenizer, "eos_token_id", -1))
    listen = token_id("<|listen|>")
    if eos != EXPECTED_EOS_ID:
        raise ValueError(f"unexpected MiniCPM EOS id: {eos} (expected {EXPECTED_EOS_ID})")
    if listen != EXPECTED_LISTEN_ID:
        raise ValueError(
            f"unexpected MiniCPM listen id: {listen} (expected {EXPECTED_LISTEN_ID})"
        )

    names = {
        "eos": "<|im_end|>",
        "listen": "<|listen|>",
        "speak": "<|speak|>",
        "chunk_eos": "<|chunk_eos|>",
        "chunk_tts_eos": "<|chunk_tts_eos|>",
        "turn_eos": "<|turn_eos|>",
        "tts_bos": "<|tts_bos|>",
        "unit": "<unit>",
        "unit_end": "</unit>",
    }
    result = {name: (eos if name == "eos" else token_id(token)) for name, token in names.items()}
    result["tokenizer_eos_token_id"] = eos
    return result


def protocol_special_ids(tokenizer: Any, ids: Mapping[str, int]) -> list[int]:
    values = set(int(item) for item in (getattr(tokenizer, "all_special_ids", ()) or ()))
    values.update(int(value) for value in ids.values())
    return sorted(values)


def to_numpy(value: Any, *, dtype: np.dtype = np.dtype("<f4")) -> np.ndarray:
    """Materialize an MLX/Torch value as contiguous little-endian NumPy."""

    # MLX BF16 arrays do not expose a NumPy buffer directly.  The explicit
    # float32 conversion is part of the fixture contract.
    try:
        import mlx.core as mx

        if isinstance(value, mx.array):
            value = value.astype(mx.float32)
    except ImportError:
        pass
    if hasattr(value, "detach"):
        value = value.detach()
    # NumPy cannot materialize a torch BF16 tensor directly.  The official
    # PyTorch oracle runs the Qwen3 backbone in BF16, while the fixture's
    # serialization contract is explicitly float32 little-endian.
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype).copy(order="C")


def array_hash(value: np.ndarray) -> str:
    arr = np.asarray(value, dtype="<f4").copy(order="C")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


class ArtifactWriter:
    def __init__(self, root: Path):
        self.root = root
        self.arrays: dict[str, dict[str, Any]] = {}

    def array(self, name: str, value: Any, *, dtype: str = "f32") -> str:
        if dtype == "i32":
            array = np.asarray(value, dtype="<i4").copy(order="C")
            suffix = ".i32"
            logical_dtype = "int32-le"
        else:
            array = to_numpy(value)
            suffix = ".f32"
            logical_dtype = "float32-le"
        path = self.root / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(array.tobytes(order="C"))
        key = name.replace("/", "_")
        self.arrays[key] = {
            "path": path.relative_to(self.root).as_posix(),
            "shape": [int(item) for item in array.shape],
            "dtype": logical_dtype,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": int(array.size),
        }
        return key

    def json(self, name: str, value: Any) -> str:
        path = self.root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path.relative_to(self.root).as_posix()


def cache_checksum(decoder: Any) -> dict[str, Any]:
    """Compactly fingerprint every valid PyTorch KV layer."""

    layers: list[dict[str, Any]] = []
    cache = decoder.cache
    if cache is None:
        return {"cache_length": 0, "layers": layers}
    # Transformers DynamicCache (used by the pinned official Qwen3 model)
    # stores parallel key/value lists; older revisions return a tuple of
    # ``(key, value)`` pairs.  Normalize both without involving MLX.
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    elif hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    else:
        pairs = list(cache)
    for index, pair in enumerate(pairs):
        if pair is None:
            layers.append({"layer": index, "length": 0})
            continue
        if len(pair) < 2:
            layers.append({"layer": index, "length": 0})
            continue
        keys, values = pair[0], pair[1]
        if keys is None or keys.numel() == 0:
            layers.append({"layer": index, "length": 0})
            continue
        keys_np = to_numpy(keys)
        values_np = to_numpy(values)

        def digest(array: np.ndarray) -> dict[str, Any]:
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

        layers.append(
            {
                "layer": index,
                "length": int(keys.shape[2]),
                "keys": digest(keys_np),
                "values": digest(values_np),
            }
        )
    return {"cache_length": int(decoder.get_cache_length()), "layers": layers}


def _cache_pairs(cache: Any) -> list[tuple[Any, Any]]:
    """Normalize the cache containers used by Transformers revisions.

    The context-window diagnostic intentionally keeps this helper separate
    from ``cache_checksum``: a trace needs to inspect every sequence position,
    while the normal fixture checksum only records one digest per layer.
    """

    if cache is None:
        return []
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [(pair[0], pair[1]) for pair in cache if pair is not None and len(pair) >= 2]


def _trace_digest(array: np.ndarray) -> dict[str, Any]:
    """Small deterministic digest for one layer/position K or V slice."""

    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    return {
        "sum": float(np.sum(flat, dtype=np.float64)),
        "sum_sq": float(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64)),
        "first8": [float(item) for item in flat[:8]],
        "last8": [float(item) for item in flat[-8:]],
        "sha256": array_hash(flat),
    }


def cache_position_trace(cache_or_decoder: Any) -> dict[str, Any]:
    """Trace every layer and sequence position of a PyTorch KV cache.

    This is deliberately a diagnostic artifact rather than a production
    golden: position-level checksums let the Swift test report whether a
    mismatch first appears in the fixed prefix, the rebuilt segment, or the
    retained tail after RoPE relocation without serializing the full 8B KV
    tensors.
    """

    cache = getattr(cache_or_decoder, "cache", cache_or_decoder)
    layers: list[dict[str, Any]] = []
    for index, pair in enumerate(_cache_pairs(cache)):
        keys, values = pair
        if keys is None or values is None or keys.numel() == 0:
            layers.append({"layer": index, "length": 0, "positions": []})
            continue
        keys_np = to_numpy(keys)
        values_np = to_numpy(values)
        length = int(keys_np.shape[2])
        positions: list[dict[str, Any]] = []
        for position in range(length):
            positions.append(
                {
                    "position": position,
                    "keys": _trace_digest(keys_np[:, :, position, :]),
                    "values": _trace_digest(values_np[:, :, position, :]),
                }
            )
        layers.append({"layer": index, "length": length, "positions": positions})
    cache_length = 0
    if layers:
        cache_length = int(layers[0].get("length", 0))
    return {"cache_length": cache_length, "layers": layers}


def operator_position_trace(value: Any) -> dict[str, Any]:
    """Digest a layer-0 operator output independently at every time step."""

    array = to_numpy(value)
    if array.ndim < 2:
        return {"shape": [int(item) for item in array.shape], "positions": []}
    length_axis = 2 if array.ndim == 4 else 1
    length = int(array.shape[length_axis])
    positions: list[dict[str, Any]] = []
    for position in range(length):
        if array.ndim == 4:
            slice_value = array[:, :, position, :]
        else:
            slice_value = array[:, position, ...]
        positions.append(
            {"position": position, "digest": _trace_digest(slice_value)}
        )
    return {
        "shape": [int(item) for item in array.shape],
        "length": length,
        "positions": positions,
    }


def output_record(
    writer: ArtifactWriter,
    name: str,
    output: tuple[Any, Any],
    selected_indices: list[int],
) -> dict[str, Any]:
    logits, hidden = output
    logits_np = to_numpy(logits).reshape(-1, int(logits.shape[-1]))[-1]
    hidden_np = to_numpy(hidden).reshape(-1, int(hidden.shape[-1]))[-1]
    selected = logits_np[np.asarray(selected_indices, dtype=np.int64)]
    top = np.argsort(-logits_np, kind="stable")[:20].astype(np.int32)
    top_values = logits_np[top]
    return {
        "logits_selected": writer.array(f"{name}_logits_selected", selected),
        "hidden": writer.array(f"{name}_hidden", hidden_np),
        "top20": writer.array(f"{name}_top20", top, dtype="i32"),
        "top20_values": writer.array(f"{name}_top20_values", top_values),
        "selected_indices": selected_indices,
        "logits_dim": int(logits_np.size),
        "hidden_dim": int(hidden_np.size),
    }


def output_records(
    writer: ArtifactWriter,
    name: str,
    outputs: list[tuple[Any, Any]],
    selected_indices: list[int],
) -> dict[str, Any]:
    logits_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    top_rows: list[np.ndarray] = []
    top_value_rows: list[np.ndarray] = []
    vocab = hidden_dim = 0
    for logits, hidden in outputs:
        logits_np = to_numpy(logits).reshape(-1, int(logits.shape[-1]))[-1]
        hidden_np = to_numpy(hidden).reshape(-1, int(hidden.shape[-1]))[-1]
        vocab = int(logits_np.size)
        hidden_dim = int(hidden_np.size)
        logits_rows.append(logits_np[np.asarray(selected_indices, dtype=np.int64)])
        hidden_rows.append(hidden_np)
        top = np.argsort(-logits_np, kind="stable")[:20].astype(np.int32)
        top_rows.append(top)
        top_value_rows.append(logits_np[top])
    return {
        "logits_selected": writer.array(f"{name}_logits_selected", np.stack(logits_rows)),
        "hidden": writer.array(f"{name}_hidden", np.stack(hidden_rows)),
        "top20": writer.array(f"{name}_top20", np.stack(top_rows), dtype="i32"),
        "top20_values": writer.array(f"{name}_top20_values", np.stack(top_value_rows)),
        "selected_indices": selected_indices,
        "steps": len(outputs),
        "logits_dim": vocab,
        "hidden_dim": hidden_dim,
    }


def deterministic_external_embeddings(hidden_size: int) -> np.ndarray:
    values = np.linspace(-0.875, 0.875, num=2 * hidden_size, dtype=np.float32)
    return values.reshape(2, hidden_size)


def load_audio_embeddings(path: Path) -> tuple[np.ndarray, str]:
    """Load the fixed P2-A projected embedding without recomputing audio.

    The audio golden is a safetensors bundle produced by the pinned native
    audio parity gate. P2-B consumes its ``embeddings`` tensor directly so an
    LLM mismatch cannot be hidden by rerunning a different frontend or MLX
    version.
    """

    from safetensors.numpy import load_file

    arrays = load_file(str(path))
    if "embeddings" not in arrays:
        raise ValueError(f"audio golden has no embeddings tensor: {path}")
    value = np.asarray(arrays["embeddings"], dtype=np.float32).copy(order="C")
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 4096:
        raise ValueError(f"audio embeddings must be [1,T,4096], got {value.shape}")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def categorical_with_uniform(logits: np.ndarray, uniform: float) -> int:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    probabilities = np.exp(values)
    probabilities /= np.sum(probabilities)
    target = min(max(float(uniform), 0.0), 0.999999999) * float(np.sum(probabilities))
    return int(np.searchsorted(np.cumsum(probabilities), target, side="right"))


def filtered_sample(
    logits: np.ndarray,
    *,
    uniform: float,
    mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if mode == "greedy":
        return int(np.argmax(logits))
    scores = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    order = np.argsort(-scores, kind="stable")
    if top_k > 0:
        order = order[: min(int(top_k), order.size)]
    ranked = scores[order]
    keep = np.ones(order.size, dtype=bool)
    if 0 < float(top_p) < 1:
        probs = np.exp(ranked - np.max(ranked))
        probs /= np.sum(probs)
        shifted = np.concatenate([[0.0], np.cumsum(probs)[:-1]])
        keep = shifted <= float(top_p)
    candidates = order[keep]
    candidate_scores = scores[candidates]
    return int(candidates[categorical_with_uniform(candidate_scores, uniform)])


def deterministic_sampler_case(
    logits: np.ndarray,
    *,
    protocol: Mapping[str, int],
    uniforms: list[float],
    forbidden: Iterable[int] = (),
    previous_tokens: Iterable[int] = (),
    mode: str = "sampling",
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    length_penalty: float = 1.1,
    listen_top_k: int | None = None,
    listen_probability_scale: float = 1.0,
) -> tuple[int, int]:
    """Numpy reference for ``MiniCPMSampler``'s two-stage draw order."""

    if not uniforms:
        raise ValueError("sampler fixture needs at least one uniform")
    natural = filtered_sample(logits, uniform=uniforms[0], mode=mode, temperature=1, top_k=0, top_p=1)
    if natural == protocol["chunk_eos"]:
        return natural, 1 if mode != "greedy" else 0

    scores = np.asarray(logits, dtype=np.float64).copy()
    for token in forbidden:
        if 0 <= int(token) < scores.size:
            scores[int(token)] = -np.inf
    special = {
        protocol[name]
        for name in ("listen", "chunk_eos", "chunk_tts_eos", "turn_eos", "speak", "unit_end", "tts_bos")
        if name in protocol
    }
    for token in set(int(item) for item in previous_tokens):
        if 0 <= token < scores.size and token not in special and repetition_penalty != 1:
            scores[token] = scores[token] / repetition_penalty if repetition_penalty > 1 else scores[token] * (1 / repetition_penalty)
    turn_eos = protocol["turn_eos"]
    if length_penalty != 1 and np.isfinite(scores[turn_eos]):
        scores[turn_eos] = scores[turn_eos] / length_penalty if scores[turn_eos] > 0 else scores[turn_eos] * length_penalty
    listen = protocol["listen"]
    if listen_probability_scale != 1 and np.isfinite(scores[listen]):
        scores[listen] *= float(listen_probability_scale)
    if listen_top_k is not None:
        rank = int(np.sum(scores > scores[listen]))
        if rank < int(listen_top_k):
            return listen, 1 if mode != "greedy" else 0
    final_uniform = uniforms[1] if mode != "greedy" and len(uniforms) > 1 else uniforms[0]
    token = filtered_sample(scores, uniform=final_uniform, mode=mode, temperature=temperature, top_k=top_k, top_p=top_p)
    return token, 2 if mode != "greedy" else 0


def ensure_torchvision_compat() -> None:
    """Define optional torchvision NMS schemas before Transformers imports.

    The pinned PyTorch environment may contain a CPU-only torchvision wheel
    without compiled custom operators.  Transformers imports torchvision while
    importing the official Qwen/Llama classes, whose fake-kernel registration
    otherwise aborts on the missing schemas.  The official LLM never invokes
    these operators; keeping this schema library alive is numerically inert.
    """

    global _TORCHVISION_COMPAT_LIB
    import torch

    library_factory = getattr(getattr(torch, "library", None), "Library", None)
    if library_factory is None:
        return
    try:
        _TORCHVISION_COMPAT_LIB = library_factory("torchvision", "DEF")
        for name in ("nms", "qnms"):
            try:
                _TORCHVISION_COMPAT_LIB.define(
                    f"{name}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
                )
            except RuntimeError:
                pass
    except (RuntimeError, TypeError):
        _TORCHVISION_COMPAT_LIB = None


class OfficialTorchStreamDecoder:
    """Small adapter around the pinned official PyTorch StreamDecoder.

    The golden values must come from the upstream PyTorch implementation, not
    the MLX model under test.  This adapter only supplies the handful of
    convenience methods used by the fixture exporter and keeps all forward,
    RoPE, cache and sliding-window logic in ``MiniCPMO45.utils.StreamDecoder``.
    """

    def __init__(self, llm: Any, tokenizer: Any, stream_decoder_type: Any, window_type: Any):
        self.m = llm
        self._decoder = stream_decoder_type(llm=llm, tokenizer=tokenizer)
        self._window_type = window_type
        self._rollback_cache: Any = None

    @property
    def cache(self) -> Any:
        return self._decoder.cache

    @cache.setter
    def cache(self, value: Any) -> None:
        self._decoder.cache = value

    @property
    def hidden_size(self) -> int:
        return int(self.m.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.m.config.vocab_size)

    def __getattr__(self, name: str) -> Any:
        # Expose official unit/context diagnostics (``_unit_history``,
        # ``_previous_marker_token_ids``) without reimplementing them here.
        return getattr(self._decoder, name)

    def embed_tokens(self, token_ids: list[int]) -> Any:
        return self._decoder.embed_tokens(token_ids)

    def embed_token(self, token_id: int) -> Any:
        return self._decoder.embed_token(token_id)

    def feed(self, embeds: Any, return_logits: bool = False) -> tuple[Any, Any]:
        import torch

        parameter = next(self.m.parameters())
        if not torch.is_tensor(embeds):
            embeds = torch.as_tensor(embeds)
        # P2-A stores projected audio as ``[1, T, H]`` because that is the
        # native MLX/session boundary.  The pinned upstream StreamDecoder
        # takes its embedding sequence as ``[T, H]`` and adds the batch axis
        # internally; normalize only the singleton batch form here.
        if embeds.ndim == 3:
            if int(embeds.shape[0]) != 1:
                raise ValueError(f"expected singleton embedding batch, got {tuple(embeds.shape)}")
            embeds = embeds[0]
        embeds = embeds.to(device=parameter.device, dtype=parameter.dtype)
        return self._decoder.feed(embeds, return_logits=return_logits)

    def reset(self) -> None:
        self._decoder.reset()
        self._rollback_cache = None

    def get_cache_length(self) -> int:
        return self._decoder.get_cache_length()

    def register_unit_start(self) -> int:
        self._rollback_cache = self._clone_cache(self._decoder.cache)
        return self._decoder.register_unit_start()

    def register_unit_end(self, *args: Any, **kwargs: Any) -> Any:
        return self._decoder.register_unit_end(*args, **kwargs)

    def commit_pending_unit(self) -> None:
        # The upstream implementation commits synchronously in
        # ``register_unit_end``.  Keep the call as a no-op for the scenario
        # table shared with the Swift session tests.
        return None

    def rollback_pending_unit(self) -> int:
        if self._rollback_cache is None:
            return 0
        before = self._decoder.get_cache_length()
        self._decoder.cache = self._rollback_cache
        self._decoder._pending_unit_id = None
        self._decoder._pending_unit_start_cache_len = 0
        self._rollback_cache = None
        return max(0, before - self._decoder.get_cache_length())

    def trim_cache_to(self, target_length: int) -> int:
        before = self._decoder.get_cache_length()
        target = max(0, min(int(target_length), before))
        if target < before:
            self._decoder.cache = self._decoder._slice_cache(0, target)
        return before - target

    def set_window_config(self, config: Mapping[str, Any]) -> None:
        self._decoder.set_window_config(
            self._window_type(
                sliding_window_mode=str(config.get("sliding_window_mode", "off")),
                basic_window_high_tokens=int(config.get("basic_window_high_tokens", 4000)),
                basic_window_low_tokens=int(config.get("basic_window_low_tokens", 3500)),
                context_previous_max_tokens=int(config.get("context_previous_max_tokens", 500)),
                context_max_units=int(config.get("context_max_units", 24)),
            )
        )

    def set_window_enabled(self, enabled: bool) -> None:
        self._decoder.set_window_enabled(enabled)

    def context_rebuild_trace(
        self,
        *,
        prefix_length: int,
        previous_tokens: list[int],
        suffix_token_ids: list[int],
        retained_length: int,
    ) -> dict[str, Any]:
        """Mirror one official context rebuild without mutating the decoder.

        ``StreamDecoder._rebuild_cache_with_previous`` keeps the retained tail
        out of the forward pass, then RoPE-reindexes only that tail before
        concatenating it.  The upstream method intentionally exposes no
        intermediate tensors, so the diagnostic reproduces those exact
        private steps on cloned cache objects and records each stage.  This is
        only used by the context-sliding fixture exporter; the production
        adapter still delegates execution to the pinned official method.
        """

        decoder = self._decoder
        old_cache = decoder.cache
        total_cache_len = decoder.get_cache_length()
        retained = max(0, int(retained_length))
        retained_start = total_cache_len - retained

        prefix_cache = decoder._slice_cache(0, int(prefix_length))
        retained_cache = (
            decoder._slice_cache(retained_start, None) if retained > 0 else None
        )
        trace: dict[str, Any] = {
            "source": cache_position_trace(old_cache),
            "prefix": cache_position_trace(prefix_cache),
            "retained_original": cache_position_trace(retained_cache),
            "retained_start": retained_start,
            "prefix_length": int(prefix_length),
            "retained_length": retained,
        }

        prev_suffix_tokens = list(previous_tokens) + list(suffix_token_ids)
        rebuilt = prefix_cache
        operator_trace: dict[str, Any] = {}
        if prev_suffix_tokens:
            embeds = decoder.embed_tokens(prev_suffix_tokens)
            device = embeds.device
            start_pos = int(prefix_length) + int(decoder._position_offset)
            position_ids = __import__("torch").arange(
                start_pos,
                start_pos + len(prev_suffix_tokens),
                device=device,
            ).unsqueeze(0)
            # Observe only layer 0 while running the exact official forward.
            # The hooks are local to this diagnostic and are removed in the
            # ``finally`` block before the real StreamDecoder rebuild runs.
            layer = decoder.m.model.layers[0]
            attention = layer.self_attn
            q_heads = int(
                getattr(attention, "num_heads", None)
                or getattr(decoder.m.config, "num_attention_heads")
            )
            kv_heads = int(
                getattr(attention, "num_key_value_heads", None)
                or getattr(decoder.m.config, "num_key_value_heads")
            )
            torch = __import__("torch")

            def first_tensor(value: Any) -> Any:
                if torch.is_tensor(value):
                    return value
                if isinstance(value, (tuple, list)):
                    for item in value:
                        found = first_tensor(item)
                        if found is not None:
                            return found
                return None

            def heads_first(value: Any, heads: int) -> Any:
                if value.ndim == 4:
                    if int(value.shape[1]) == heads and int(value.shape[2]) != heads:
                        return value
                    return value.transpose(1, 2)
                if value.ndim != 3:
                    return value
                batch, length, width = value.shape
                return value.view(batch, length, heads, int(width) // heads).transpose(1, 2)

            def record_operator(name: str, value: Any) -> None:
                if value is not None and name not in operator_trace:
                    operator_trace[name] = operator_position_trace(value)

            def output_hook(name: str, transform: Any | None = None):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    value = first_tensor(output)
                    record_operator(name, transform(value) if transform and value is not None else value)

                return hook

            def pre_hook(name: str):
                def hook(_module: Any, inputs: Any) -> None:
                    record_operator(name, first_tensor(inputs))

                return hook

            handles = [
                attention.v_proj.register_forward_hook(
                    output_hook("v", lambda value: heads_first(value, kv_heads))
                ),
                attention.o_proj.register_forward_pre_hook(pre_hook("sdpa_merge")),
                layer.register_forward_hook(output_hook("layer_out")),
            ]
            attention_globals = getattr(
                getattr(attention.forward, "__func__", attention.forward),
                "__globals__",
                {},
            )
            original_rotary = attention_globals.get("apply_rotary_pos_emb")
            active_attention = True

            def capture_rotary(*args: Any, **kwargs: Any) -> Any:
                result = original_rotary(*args, **kwargs)
                if active_attention and isinstance(result, (tuple, list)) and len(result) >= 2:
                    record_operator("q_rope", result[0])
                    record_operator("k_rope", result[1])
                return result

            try:
                if original_rotary is not None:
                    attention_globals["apply_rotary_pos_emb"] = capture_rotary
                with torch.no_grad():
                    outputs = decoder.m(
                        inputs_embeds=embeds.unsqueeze(0)
                        if embeds.dim() == 2
                        else embeds,
                        position_ids=position_ids,
                        past_key_values=prefix_cache,
                        use_cache=True,
                        return_dict=True,
                    )
            finally:
                active_attention = False
                if original_rotary is not None:
                    attention_globals["apply_rotary_pos_emb"] = original_rotary
                for handle in handles:
                    handle.remove()
            required_ops = ("q_rope", "k_rope", "v", "sdpa_merge", "layer_out")
            missing_ops = [name for name in required_ops if name not in operator_trace]
            if missing_ops:
                raise RuntimeError(
                    "context rebuild layer-0 trace did not capture: "
                    + ", ".join(missing_ops)
                )
            trace["layer0_ops"] = operator_trace
        rebuilt = outputs.past_key_values
        trace["recomputed_previous_suffix"] = cache_position_trace(rebuilt)

        new_system_total = int(prefix_length) + len(prev_suffix_tokens)
        reindexed = None
        if retained_cache is not None and retained > 0:
            # _reindex_rope_for_cache returns a fresh DynamicCache/tuple for
            # the cloned slice, matching the official implementation.
            reindexed = decoder._reindex_rope_for_cache(
                retained_cache,
                retained_start,
                new_system_total,
                retained,
            )
        trace["retained_reindexed"] = cache_position_trace(reindexed)
        final_cache = decoder._concat_caches(rebuilt, reindexed)
        trace["final"] = cache_position_trace(final_cache)
        return trace

    @staticmethod
    def _clone_cache(cache: Any) -> Any:
        if cache is None:
            return None
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            # Importing DynamicCache lazily keeps ``--help`` dependency-free.
            from transformers.cache_utils import DynamicCache

            cloned = DynamicCache()
            cloned.key_cache = [item.clone() for item in cache.key_cache]
            cloned.value_cache = [item.clone() for item in cache.value_cache]
            return cloned
        return tuple((keys.clone(), values.clone()) for keys, values in cache)


def load_official_torch_decoder(source_model: Path, bundle: Path, upstream: Path):
    """Load the pinned upstream PyTorch Qwen3 LLM as the numerical oracle.

    ``bundle`` is intentionally unused for inference.  It remains an explicit
    argument so the caller can record and later verify the MLX bundle's
    provenance without accidentally using it to generate expected outputs.
    """

    del bundle
    # Keep heavyweight imports lazy so ``--help`` and static linting work on a
    # machine without torch/transformers/safetensors installed.
    import torch

    ensure_torchvision_compat()
    from transformers import AutoTokenizer

    upstream = upstream.expanduser().resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from MiniCPMO45.configuration_minicpmo import MiniCPMOConfig
    from MiniCPMO45.modeling_minicpmo import Qwen3ForCausalLM
    from MiniCPMO45.utils import DuplexWindowConfig, StreamDecoder
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(
        str(source_model), trust_remote_code=True, local_files_only=True
    )
    ids = protocol_ids(tokenizer)
    config = MiniCPMOConfig.from_pretrained(str(source_model), local_files_only=True)
    llm = Qwen3ForCausalLM(config)
    llm.to(dtype=torch.bfloat16)

    index_path = source_model / "model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"source model has no safetensors weight map: {index_path}")
    loaded: set[str] = set()
    for shard_name in sorted(set(str(item) for item in weight_map.values())):
        shard = load_file(str(source_model / shard_name), device="cpu")
        llm_shard = {
            str(key)[len("llm.") :]: value
            for key, value in shard.items()
            if str(key).startswith("llm.")
        }
        if llm_shard:
            llm.load_state_dict(llm_shard, strict=False)
            loaded.update(llm_shard)
        del shard
    missing = set(llm.state_dict()) - loaded
    if missing:
        raise ValueError(f"official LLM weights missing tensors: {sorted(missing)[:8]}")
    llm.eval()
    if torch.backends.mps.is_available():
        llm.to(device="mps")
    decoder = OfficialTorchStreamDecoder(llm, tokenizer, StreamDecoder, DuplexWindowConfig)
    return decoder, tokenizer, ids


def main() -> None:
    args = parse_args()
    source_model = args.source_model.expanduser().resolve()
    bundle = args.mlx_bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    upstream = args.upstream_source.expanduser().resolve()
    converter = args.converter.expanduser().resolve()
    audio_path = (
        args.audio_golden.expanduser().resolve()
        if args.audio_golden is not None
        else None
    )
    if not source_model.is_dir() or not bundle.is_dir():
        raise SystemExit("--source-model and --mlx-bundle must be directories")
    if audio_path is not None and not audio_path.is_file():
        raise SystemExit(f"--audio-golden must be a file: {audio_path}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        if not output.is_dir():
            raise SystemExit(f"output is not a directory: {output}")
        for child in output.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    source_config = read_json(source_model / "config.json")
    source_index_path = source_model / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise SystemExit(
            "source model must contain model.safetensors.index.json so weight provenance is fail-closed"
        )
    bundle_config = read_json(bundle / "config.json")
    if args.source_revision != PINNED_MODEL_REVISION:
        raise SystemExit(
            "--source-revision must be the pinned MiniCPM-o-4_5 model revision "
            f"{PINNED_MODEL_REVISION}, got {args.source_revision!r}"
        )
    quant = bundle_config.get("quantization") or bundle_config.get("quantization_config") or {}
    bundle_bits = int(quant.get("bits", -1))
    bundle_mode = str(quant.get("mode", "none"))
    if bundle_bits not in (0, 8):
        raise SystemExit(
            "--mlx-bundle must be a dense BF16 or affine 8-bit export "
            "with quantization.bits=0 or 8"
        )
    expected_mode = "none" if bundle_bits == 0 else "affine"
    if bundle_mode != expected_mode:
        raise SystemExit(
            f"--mlx-bundle quantization.mode={bundle_mode!r} is incompatible "
            f"with quantization.bits={bundle_bits}; expected {expected_mode!r}"
        )
    upstream_commit = git_commit(upstream)
    if upstream_commit != PINNED_UPSTREAM_COMMIT:
        raise SystemExit(
            f"upstream checkout must be {PINNED_UPSTREAM_COMMIT}, got {upstream_commit!r}: {upstream}"
        )
    if not converter.is_file():
        raise SystemExit(f"converter script not found: {converter}")

    decoder, tokenizer, ids = load_official_torch_decoder(source_model, bundle, upstream)
    hidden_size = int(decoder.hidden_size)
    vocab_size = int(decoder.vocab_size)
    special_ids = protocol_special_ids(tokenizer, ids)
    selected_indices = sorted(
        set([0, 1, 2, 3, 151643, EXPECTED_EOS_ID, EXPECTED_LISTEN_ID] + list(ids.values()))
    )
    selected_indices = [index for index in selected_indices if 0 <= index < vocab_size]

    writer = ArtifactWriter(output)
    token_ids = [151643, 198, 108, EXPECTED_EOS_ID]
    token_embeddings = to_numpy(decoder.embed_tokens(token_ids))
    token_embedding_array = writer.array("token_embeddings", token_embeddings)
    external = deterministic_external_embeddings(hidden_size)
    external_array = writer.array("external_embeddings", external)
    token_ids_array = writer.array("token_ids", np.asarray(token_ids, dtype=np.int32), dtype="i32")

    # One batch prefill and the same sequence one token at a time.
    decoder.reset()
    prefill = decoder.feed(decoder.embed_tokens(token_ids), return_logits=True)
    prefill_record = output_record(writer, "prefill", prefill, selected_indices)
    prefill_kv = cache_checksum(decoder)

    decoder.reset()
    incremental_outputs: list[tuple[Any, Any]] = []
    incremental_kv: list[dict[str, Any]] = []
    for token in token_ids:
        incremental_outputs.append(
            decoder.feed(decoder.embed_token(token), return_logits=True)
        )
        incremental_kv.append(cache_checksum(decoder))
    incremental_record = output_records(writer, "incremental", incremental_outputs, selected_indices)

    decoder.reset()
    external_output = decoder.feed(external, return_logits=True)
    external_record = output_record(writer, "external", external_output, selected_indices)

    # Snapshot/rollback: speculative input is never allowed to influence the
    # canonical suffix or the generated-token history.
    decoder.reset()
    rollback_prefix = [151643, 198]
    rollback_speculative = [ids["listen"], ids["speak"]]
    rollback_canonical = [ids["unit"], ids["unit_end"]]
    decoder.feed(decoder.embed_tokens(rollback_prefix), return_logits=True)
    rollback_cache_before = decoder.get_cache_length()
    decoder.register_unit_start()
    for token in rollback_speculative:
        decoder.feed(decoder.embed_token(token), return_logits=True)
    rollback_cache_speculative = decoder.get_cache_length()
    rollback_removed = decoder.rollback_pending_unit()
    rollback_cache_after = decoder.get_cache_length()
    canonical_outputs = [
        decoder.feed(decoder.embed_token(token), return_logits=True)
        for token in rollback_canonical
    ]
    rollback_record = output_records(writer, "rollback_canonical", canonical_outputs, selected_indices)

    # Direct trim oracle.  Swift has a value-semantic state trim helper; the
    # fixture also records the equivalent transaction lengths for older builds.
    decoder.reset()
    trim_prefix = [151643, 198, 108, EXPECTED_EOS_ID]
    decoder.feed(decoder.embed_tokens(trim_prefix), return_logits=True)
    trim_target = 2
    trim_removed = decoder.trim_cache_to(trim_target)
    trim_probe = decoder.feed(decoder.embed_token(ids["unit"]), return_logits=True)
    trim_record = output_record(writer, "trim_probe", trim_probe, selected_indices)
    trim_cache_after_probe = decoder.get_cache_length()

    # Basic sliding window (prefix + one-token units, high=4, low=2).
    decoder.reset()
    decoder.set_window_config(
        {"sliding_window_mode": "basic", "basic_window_high_tokens": 4, "basic_window_low_tokens": 2}
    )
    decoder.set_window_enabled(True)
    decoder.feed(decoder.embed_token(151643), return_logits=True)
    decoder.register_system_prompt()
    basic_units = [198, 108, 151683, 151705, 151706]
    basic_events: list[dict[str, Any]] = []
    for unit_id, token in enumerate(basic_units):
        before_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        before_cache = decoder.get_cache_length()
        decoder.register_unit_start()
        decoder.feed(decoder.embed_token(token), return_logits=True)
        decoder.register_unit_end(input_type="audio", generated_tokens=[token])
        decoder.commit_pending_unit()
        triggered = decoder.enforce_window()
        after_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        if triggered:
            basic_events.append(
                {
                    "unit_id": unit_id,
                    "triggered": True,
                    "before_unit_ids": before_ids,
                    "after_unit_ids": after_ids,
                    "dropped_unit_ids": [item for item in before_ids if item not in after_ids],
                    "cache_before": before_cache + 1,
                    "cache_after": decoder.get_cache_length(),
                }
            )
    basic_probe = decoder.feed(decoder.embed_token(ids["unit_end"]), return_logits=True)
    basic_probe_record = output_record(writer, "sliding_basic_probe", basic_probe, selected_indices)
    basic_kv = cache_checksum(decoder)

    # Context sliding: retain prefix/suffix, move generated normal tokens to
    # ``previous`` and replay the retained unit at compressed RoPE positions.
    decoder.reset()
    decoder.set_window_config(
        {
            "sliding_window_mode": "context",
            "context_max_units": 1,
            "context_previous_max_tokens": 8,
        }
    )
    decoder.set_window_enabled(True)
    decoder.feed(decoder.embed_token(151643), return_logits=True)
    context_trace: dict[str, Any] = {
        "prefix": cache_position_trace(decoder),
    }
    decoder.register_system_prompt_with_context(suffix_token_ids=[EXPECTED_EOS_ID])
    decoder.feed(decoder.embed_token(EXPECTED_EOS_ID), return_logits=True)
    context_trace["prefix_suffix"] = cache_position_trace(decoder)
    marker_ids = list(decoder._previous_marker_token_ids)
    context_units = [
        {"token": 198, "generated": [20, 21], "text": "first"},
        {"token": 108, "generated": [22], "text": "second"},
    ]
    context_events: list[dict[str, Any]] = []
    for unit_id, unit in enumerate(context_units):
        before_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        decoder.register_unit_start()
        decoder.feed(decoder.embed_token(unit["token"]), return_logits=True)
        decoder.register_unit_end(
            input_type="audio",
            generated_tokens=unit["generated"],
            generated_text=unit["text"],
        )
        decoder.commit_pending_unit()
        context_trace[f"before_enforce_unit_{unit_id}"] = cache_position_trace(decoder)
        if unit_id == 1:
            # The second unit triggers the only context rebuild in this tiny
            # fixture. Capture each upstream intermediate before the real
            # method mutates ``decoder.cache``.
            context_trace["rebuild"] = decoder.context_rebuild_trace(
                prefix_length=1,
                previous_tokens=marker_ids + context_units[0]["generated"],
                suffix_token_ids=[EXPECTED_EOS_ID],
                retained_length=1,
            )
        triggered = decoder.enforce_window_with_context()
        after_ids = [int(item["unit_id"]) for item in decoder._unit_history]
        if triggered:
            previous_text, previous_ids = decoder.get_previous_context()
            context_events.append(
                {
                    "unit_id": unit_id,
                    "triggered": True,
                    "before_unit_ids": before_ids,
                    "after_unit_ids": after_ids,
                    "dropped_unit_ids": [item for item in before_ids if item not in after_ids],
                    "cache_after": decoder.get_cache_length(),
                    "previous_text": previous_text,
                    "previous_token_ids": previous_ids,
                }
            )
    context_trace["final"] = cache_position_trace(decoder)
    context_probe = decoder.feed(decoder.embed_token(ids["unit_end"]), return_logits=True)
    context_probe_record = output_record(writer, "sliding_context_probe", context_probe, selected_indices)
    context_kv = cache_checksum(decoder)
    context_trace["after_probe"] = cache_position_trace(decoder)

    audio_bridge_scenario: dict[str, Any] | None = None
    if audio_path is not None:
        audio_embeddings, audio_sha = load_audio_embeddings(audio_path)
        audio_name = writer.array("audio_bridge_embeddings", audio_embeddings)
        bridge_cases: list[dict[str, Any]] = []

        def bridge_case(
            name: str,
            operations: list[tuple[str, int | None]],
            *,
            repair: bool = False,
        ) -> None:
            decoder.reset()
            outputs: list[tuple[Any, Any]] = []
            for kind, value in operations:
                if kind == "audio":
                    outputs.append(decoder.feed(audio_embeddings, return_logits=True))
                else:
                    assert value is not None
                    outputs.append(
                        decoder.feed(decoder.embed_token(value), return_logits=True)
                    )
            if repair:
                decoder.reset()
                outputs = [decoder.feed(audio_embeddings, return_logits=True)]
                decoder.register_unit_start()
                for token in (ids["listen"], ids["speak"]):
                    decoder.feed(decoder.embed_token(token), return_logits=True)
                decoder.rollback_pending_unit()
                outputs.extend(
                    decoder.feed(decoder.embed_token(token), return_logits=True)
                    for token in (ids["turn_eos"], ids["unit_end"])
            )
            record_name = f"audio_bridge_{name}"
            record = output_records(writer, record_name, outputs, selected_indices)
            cache_name = writer.json(
                f"kv_audio_bridge_{name}", cache_checksum(decoder)
            )
            bridge_cases.append(
                {
                    "name": name,
                    "audio_array": audio_name,
                    "operations": [
                        {
                            "kind": kind,
                            **({"token_id": value} if value is not None else {}),
                        }
                        for kind, value in operations
                    ],
                    "repair": repair,
                    "output": record,
                    "cache": cache_name,
                }
            )

        bridge_case("first_audio_unit", [("audio", None)])
        bridge_case(
            "listen_unit",
            [("audio", None), ("token", ids["listen"])],
        )
        bridge_case(
            "spoken_continuation",
            [
                ("audio", None),
                ("token", ids["speak"]),
                ("token", ids["tts_bos"]),
            ],
        )
        bridge_case("repair_next_unit", [("audio", None)], repair=True)
        audio_bridge_scenario = {
            "audio_array": audio_name,
            "audio_shape": list(audio_embeddings.shape),
            "audio_golden_sha256": audio_sha,
            "cases": bridge_cases,
        }

    # Protocol-aware deterministic sampler cases.  The arrays are tiny and
    # intentionally independent of model logits so sampling regressions do not
    # require loading the 8B bundle in XCTest.
    sampling_cases: list[dict[str, Any]] = []
    width = max(32, max(ids.values()) + 1)

    def case(name: str, logits: np.ndarray, **kwargs: Any) -> None:
        uniforms = list(kwargs.pop("uniforms"))
        token, draws = deterministic_sampler_case(
            logits, protocol=ids, uniforms=uniforms, **kwargs
        )
        array_name = writer.array(f"sampling_{name}_logits", logits)
        sampling_cases.append(
            {
                "name": name,
                "logits": array_name,
                "uniforms": uniforms,
                "expected_token": token,
                "draw_count": draws,
                **kwargs,
            }
        )

    base = np.full(width, -10.0, dtype=np.float32)
    natural = base.copy()
    natural[ids["chunk_eos"]] = 10
    case(
        "natural_chunk_eos",
        natural,
        uniforms=[0.2],
        mode="sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        forbidden=[ids["chunk_eos"]],
        previous_tokens=[],
        repetition_penalty=1.05,
        length_penalty=1.1,
        listen_top_k=None,
        listen_probability_scale=1.0,
    )
    forbidden = base.copy()
    forbidden[0] = 10
    forbidden[1] = 9
    case(
        "forbidden_after_probe",
        forbidden,
        uniforms=[0.1, 0.1],
        mode="sampling",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        forbidden=[0],
        previous_tokens=[],
        repetition_penalty=1.0,
        length_penalty=1.0,
        listen_top_k=None,
        listen_probability_scale=1.0,
    )
    repetition = base.copy()
    repetition[20] = 5
    repetition[21] = 4.9
    case(
        "repetition_penalty",
        repetition,
        uniforms=[0.2, 0.2],
        mode="sampling",
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        forbidden=[],
        previous_tokens=[20],
        repetition_penalty=2.0,
        length_penalty=1.0,
        listen_top_k=None,
        listen_probability_scale=1.0,
    )
    listen = base.copy()
    listen[ids["listen"]] = 4.0
    listen[10] = 3.0
    case(
        "listen_rank",
        listen,
        uniforms=[0.2, 0.2],
        mode="sampling",
        temperature=1.0,
        top_k=20,
        top_p=1.0,
        forbidden=[],
        previous_tokens=[],
        repetition_penalty=1.0,
        length_penalty=1.0,
        listen_top_k=2,
        listen_probability_scale=1.0,
    )
    top_p = np.full(width, -10.0, dtype=np.float32)
    top_p[0], top_p[1], top_p[2] = 3.0, 2.0, 1.0
    case(
        "top_p_boundary",
        top_p,
        uniforms=[0.2, 0.99],
        mode="sampling",
        temperature=1.0,
        top_k=0,
        top_p=0.5,
        forbidden=[],
        previous_tokens=[],
        repetition_penalty=1.0,
        length_penalty=1.0,
        listen_top_k=None,
        listen_probability_scale=1.0,
    )

    # Persist metadata only after all arrays are written.  JSON is sorted and
    # has no timestamp, so byte-for-byte reruns are possible.
    upstream_files = {
        rel: sha256_file(upstream / rel)
        for rel in (
            "MiniCPMO45/utils.py",
            "MiniCPMO45/modeling_minicpmo_unified.py",
            "MiniCPMO45/modeling_minicpmo.py",
            "MiniCPMO45/configuration_minicpmo.py",
        )
        if (upstream / rel).is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "oracle_backend": "official_pytorch",
        "source": {
            "model_id": args.source_model_id,
            "revision": args.source_revision,
            "revision_kind": "explicit" if args.source_revision else "local-tree-sha256",
            "directory_name": source_model.name,
            "tree_sha256": sha256_tree(source_model),
            "config_sha256": sha256_file(source_model / "config.json"),
            "index_sha256": sha256_file(source_index_path),
            "upstream_commit": upstream_commit,
            "upstream_files": upstream_files,
            "config_model_type": source_config.get("model_type"),
        },
        "model": {
            "bundle_name": bundle.name,
            # A portable resolver may locate the bundle from a checked-out
            # repository ancestor, but must never infer it from the fixture's
            # own sibling directory.
            "repo_relative_path": (
                bundle.relative_to(Path(__file__).resolve().parents[1]).as_posix()
                if bundle.is_relative_to(Path(__file__).resolve().parents[1])
                else None
            ),
            "tree_sha256": sha256_tree(bundle),
            "config_sha256": sha256_file(bundle / "config.json"),
            "config_file": "config.json",
            "quantization": {
                "bits": int(quant.get("bits", 0)),
                "group_size": int(quant.get("group_size", 0)),
                "mode": str(quant.get("mode", "affine")),
            },
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        },
        "converter": {
            "logical_path": converter.name,
            "sha256": sha256_file(converter),
        },
        "runtime": {
            "oracle_backend": "official_pytorch",
            "attention_implementation": str(
                getattr(decoder.m.config, "_attn_implementation", "unknown")
            ),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "oracle_source_files": upstream_files,
            "seed": args.seed,
        },
        "protocol_ids": ids,
        "protocol_special_ids": special_ids,
        "inputs": {
            "token_ids": token_ids_array,
            "token_ids_values": token_ids,
            "token_embeddings": token_embedding_array,
            "external_embeddings": external_array,
        },
        "arrays": writer.arrays,
        "scenarios": {
            "prefill": {
                "token_ids": token_ids,
                "cache": writer.json("kv_prefill", prefill_kv),
                "output": prefill_record,
            },
            "incremental": {
                "token_ids": token_ids,
                "cache": writer.json("kv_incremental", incremental_kv),
                "output": incremental_record,
                "batch_last_step": len(token_ids) - 1,
            },
            "external": {"shape": list(external.shape), "output": external_record},
            "rollback": {
                "prefix_token_ids": rollback_prefix,
                "speculative_token_ids": rollback_speculative,
                "canonical_token_ids": rollback_canonical,
                "cache_before": rollback_cache_before,
                "cache_speculative": rollback_cache_speculative,
                "cache_after": rollback_cache_after,
                "removed": rollback_removed,
                "output": rollback_record,
            },
            "trim": {
                "prefix_token_ids": trim_prefix,
                "target_length": trim_target,
                "removed": trim_removed,
                "probe_token_id": ids["unit"],
                "cache_after_probe": trim_cache_after_probe,
                "output": trim_record,
            },
            "sliding_basic": {
                "config": {"mode": "basic", "high": 4, "low": 2},
                "prefix_token_ids": [151643],
                "units": [{"input_token_id": token, "generated_tokens": [token]} for token in basic_units],
                "events": basic_events,
                "cache": writer.json("kv_sliding_basic", basic_kv),
                "probe_token_id": ids["unit_end"],
                "output": basic_probe_record,
            },
            "sliding_context": {
                "config": {"mode": "context", "max_units": 1, "previous_max_tokens": 8},
                "prefix_token_ids": [151643],
                "suffix_token_ids": [EXPECTED_EOS_ID],
                "marker_token_ids": marker_ids,
                "units": context_units,
                "events": context_events,
                "cache": writer.json("kv_sliding_context", context_kv),
                "probe_token_id": ids["unit_end"],
                "output": context_probe_record,
                "trace": context_trace,
            },
            "sampling": {
                "vocab_size": width,
                "cases": sampling_cases,
            },
        },
        "tolerances": {
            "bf16": {"hidden_max_abs": 2e-3, "logits_cosine": 0.995, "top20_overlap": 19},
            "mlx_8bit": {"hidden_max_abs": 0.2, "logits_cosine": 0.96, "top20_overlap": 18},
            "kv_checksum": {"sum_abs": 0.2, "sum_sq_abs": 1.0},
        },
    }
    if audio_bridge_scenario is not None:
        assert audio_path is not None
        manifest["runtime"]["audio_golden"] = audio_path.name
        manifest["scenarios"]["audio_bridge"] = audio_bridge_scenario
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "arrays": len(writer.arrays), "schema": SCHEMA}, indent=2))


if __name__ == "__main__":
    main()
