#!/usr/bin/env python3
"""Export a deterministic PyTorch oracle for MiniCPM-o's audio path.

The oracle deliberately imports the pinned upstream implementation rather
than importing Python files from the model directory.  A model directory is
an artifact (config/weights); it is not an implementation authority.  The
side-car manifest records enough provenance to make a fixture auditable and
reproducible without embedding any machine-specific paths.

This script is intentionally an offline fixture generator.  It is not part
of the MLX runtime and should only be run in the pinned PyTorch environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Keep this in lock-step with the source manifest used by the other MiniCPM
# exporters.  A different upstream tree can change subtle Whisper cache
# semantics while leaving the tensor names unchanged.
UPSTREAM_REVISION = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
MANIFEST_SCHEMA = "minicpm-o-audio-golden/v2"
SOURCE_PREFIXES = ("apm.", "audio_projection_layer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Original MiniCPM-o model directory")
    parser.add_argument(
        "output",
        type=Path,
        help="Output golden.safetensors path (manifest is written beside it)",
    )
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=Path("/tmp/MiniCPM-o-Demo"),
        help="Pinned checkout containing the official MiniCPMO45 Python modules",
    )
    parser.add_argument(
        "--upstream-revision",
        default=UPSTREAM_REVISION,
        help="Expected upstream git commit (defaults to the pinned revision)",
    )
    parser.add_argument(
        "--source-model-id",
        default=None,
        help="Stable model identifier recorded in the manifest (no local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Optional model repository revision; never interpreted as the upstream code revision",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing golden file and manifest",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Hash a directory by relative names and bytes, excluding VCS metadata."""

    digest = hashlib.sha256()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    ]
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _relative_source_file(source: Path, candidate: str) -> Path:
    """Resolve an index entry and reject path traversal/missing shards."""

    relative = Path(candidate)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe safetensors index path: {candidate!r}")
    path = (source / relative).resolve()
    if path.parent != source.resolve() and source.resolve() not in path.parents:
        raise RuntimeError(f"safetensors index path escapes model directory: {candidate!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing safetensors shard referenced by index: {candidate}")
    return path


def source_provenance(source: Path, model_id: str, revision: str | None) -> dict[str, Any]:
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(
            "model.safetensors.index.json is required; refusing to infer shard ownership"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("invalid safetensors index: weight_map must be non-empty")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards = []
    for name in shard_names:
        path = _relative_source_file(source, name)
        shards.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "model_id": model_id,
        "source_revision": revision,
        "revision_kind": "model_repository" if revision else "unspecified",
        "tree_sha256": _sha256_tree(source),
        "config_sha256": _sha256_file(config_path),
        "index_sha256": _sha256_file(index_path),
        "shards": shards,
    }


def _assert_upstream_revision(source: Path, expected: str) -> str:
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"upstream source must be a git checkout at {expected}: {source}"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            "upstream MiniCPM implementation revision mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual


_TORCHVISION_COMPAT_LIB = None


def ensure_torchvision_compat() -> None:
    """Define schemas required by old torchvision builds before transformers import."""

    global _TORCHVISION_COMPAT_LIB
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


def load_local_module(package: str, name: str, path: Path):
    qualified = f"{package}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def import_minicpm(upstream_source: Path, expected_revision: str):
    """Import only the official processor/encoder/projector implementation."""

    upstream_source = upstream_source.expanduser().resolve()
    actual_revision = _assert_upstream_revision(upstream_source, expected_revision)
    ensure_torchvision_compat()
    package = "_minicpmo_audio_upstream"
    root = types.ModuleType(package)
    root.__path__ = [str(upstream_source / "MiniCPMO45")]
    sys.modules[package] = root
    module_root = upstream_source / "MiniCPMO45"
    load_local_module(package, "configuration_minicpmo", module_root / "configuration_minicpmo.py")
    modeling = load_local_module(package, "modeling_minicpmo", module_root / "modeling_minicpmo.py")
    processing = load_local_module(package, "processing_minicpmo", module_root / "processing_minicpmo.py")
    return modeling, processing, actual_revision


def load_selected(source: Path) -> dict[str, torch.Tensor]:
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("invalid safetensors index: missing weight_map")
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith(SOURCE_PREFIXES)
    }
    if not selected:
        raise RuntimeError("safetensors index contains no MiniCPM audio tensors")
    result: dict[str, torch.Tensor] = {}
    for shard in sorted(set(selected.values())):
        shard_path = _relative_source_file(source, shard)
        names = [name for name, candidate in selected.items() if candidate == shard]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in names:
                if name not in available:
                    raise RuntimeError(f"index maps missing tensor {name!r} in {shard!r}")
                result[name] = handle.get_tensor(name)
    return result


def deterministic_waveform(count: int = 16_000, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(count, dtype=np.float32) / sample_rate
    chirp_phase = 2 * np.pi * (180 * t + 0.5 * 620 * t * t)
    signal = 0.12 * np.sin(2 * np.pi * 220 * t) + 0.07 * np.sin(chirp_phase)
    signal *= np.minimum(1, t * 20) * np.minimum(1, (1 - t) * 20)
    return signal.astype(np.float32)


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _cache_layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        return []
    if hasattr(cache, "self_attention_cache"):
        cache = cache.self_attention_cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError(f"unsupported MiniCPM audio cache type: {type(cache)!r}")


def _clone_cache(cache: Any) -> Any:
    """Clone a cache through its public legacy representation when possible."""

    layers = _cache_layers(cache)
    if not layers:
        return None
    # The pinned transformers version accepts a tuple-of-tuples as
    # ``past_key_values`` and upgrades it to EncoderDecoderCache internally.
    # The pinned upstream encoder explicitly upgrades a *list* legacy cache;
    # a tuple falls through to the new Cache API and is rejected on the next
    # ``self_attention_cache`` access.
    return [(keys.clone(), values.clone()) for keys, values in layers]


def _cache_tensors(prefix: str, cache: Any, tensors: dict[str, torch.Tensor]) -> None:
    for index, (keys, values) in enumerate(_cache_layers(cache)):
        tensors[f"{prefix}_layer_{index:02d}_keys"] = _as_float_tensor(keys)
        tensors[f"{prefix}_layer_{index:02d}_values"] = _as_float_tensor(values)


# Streaming KV bucket length (in post-convolution encoder positions). Both
# the oracle and the MLX runtime lay every streaming chunk's attention keys
# out as [real_past | zeros | current] with a constant total length
# (KV_BUCKET_POSITIONS) and mask the zero middle out of the softmax with a
# large finite negative value, so attention tensor shapes are identical for
# every chunk of a session and compiled graph plans are reused.
# exp(-10000 - max) underflows to exactly +0 in FP32, so the zero block
# contributes nothing and the layout is mathematically transparent.
KV_BUCKET_POSITIONS = 256


def _bucketize_cache(
    cache: Any,
    past_real: int,
    current: int,
) -> tuple[Any, int]:
    """Lay the cache out as [zeros | real_past] padded so that appending the
    current chunk's keys yields exactly KV_BUCKET_POSITIONS total positions.
    Returns (padded cache, head_pad) where head_pad = zero positions placed
    BEFORE the real keys."""

    head_pad = KV_BUCKET_POSITIONS - past_real - current
    if head_pad < 0:
        raise RuntimeError(
            f"KV bucket {KV_BUCKET_POSITIONS} too small for real length {past_real + current}")
    layers = _cache_layers(cache)
    if not layers:
        # First chunk: synthesize an all-zero prefix cache.
        return None, head_pad
    padded = []
    for keys, values in layers:
        zk = torch.zeros(
            (keys.shape[0], keys.shape[1], head_pad, keys.shape[3]),
            dtype=keys.dtype, device=keys.device)
        zv = torch.zeros(
            (values.shape[0], values.shape[1], head_pad, values.shape[3]),
            dtype=values.dtype, device=values.device)
        padded.append((torch.cat([zk, keys], dim=2), torch.cat([zv, values], dim=2)))
    return padded, head_pad


def _run_encoder_chunk(
    encoder: torch.nn.Module,
    projection: torch.nn.Module,
    chunk: torch.Tensor,
    *,
    cache: Any,
    device: torch.device,
    use_extra_context: bool = False,
    prefix_extra_frames: int = 0,
    suffix_extra_frames: int = 0,
    audio_dtype: torch.dtype,
    pool_step: int = 5,
    bucket_pad: bool = False,
    bucket_real_length: int = 0,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    _bucket_head_pad: list[int] = [0]
    chunk = chunk.to(device=device, dtype=audio_dtype)
    current = (chunk.shape[-1] - 1) // 2 + 1
    if use_extra_context:
        current -= (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
        current -= (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0
    if current <= 0:
        raise RuntimeError("audio chunk is empty after CNN context removal")
    # Static-shape bucket layout: [zeros | real_past | current] with total
    # length KV_BUCKET_POSITIONS. The mask exposes only the real positions
    # (head zeros + real past + current are 0; the zero middle is -10000).
    # The MLX runtime applies the identical layout and mask geometry.
    if bucket_pad:
        run_cache, head_pad = _bucketize_cache(
            cache, bucket_real_length - 0, current)
        past_real = bucket_real_length
        total = KV_BUCKET_POSITIONS
        mask = torch.zeros((1, 1, current, total), dtype=audio_dtype, device=device)
        _bucket_head_pad[0] = head_pad
        real_start = head_pad
        real_end = head_pad + past_real + current
        if real_start > 0:
            mask[..., :real_start] = -10_000.0
        if real_end < total:
            mask[..., real_end:] = -10_000.0
        if run_cache is None and head_pad > 0:
            # First chunk with no cache: synthesize a zero prefix cache so
            # the total key length still equals the bucket. Geometry matches
            # the pinned audio encoder (24 layers, 16 heads, head_dim 64).
            synth = []
            for _ in range(24):
                zk = torch.zeros((1, 16, head_pad, 64), dtype=audio_dtype, device=device)
                synth.append((zk, zk.clone()))
            run_cache = synth
    else:
        past = _cache_layers(cache)[0][0].shape[2] if _cache_layers(cache) else 0
        mask = torch.zeros((1, 1, current, past + current), dtype=audio_dtype, device=device)
        run_cache = cache
    with torch.inference_mode():
        result = encoder(
            chunk,
            past_key_values=run_cache,
            use_cache=True,
            attention_mask=mask,
            use_extra_context=use_extra_context,
            prefix_extra_frames=prefix_extra_frames,
            suffix_extra_frames=suffix_extra_frames,
            return_dict=True,
        )
        next_cache = result.past_key_values
        if bucket_pad and next_cache is not None:
            # Store only the real positions (they sit at
            # [head_pad, head_pad + past_real + current) in the static
            # layout) so the next chunk's bucket math stays exact.
            layers = _cache_layers(next_cache)
            start = _bucket_head_pad[0] if _bucket_head_pad else (
                KV_BUCKET_POSITIONS - bucket_real_length - current)
            end = start + bucket_real_length + current
            next_cache = [(k[..., start:end, :], v[..., start:end, :])
                          for k, v in layers]
        states = result.last_hidden_state
        projected = projection(states)
        embeddings = torch.nn.functional.avg_pool1d(
            projected.transpose(1, 2), kernel_size=pool_step, stride=pool_step
        ).transpose(1, 2)
    return next_cache, states, embeddings


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order="C")
    values = tensor.float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": tensor.numel(),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sum": float(values.sum().item()) if values.numel() else 0.0,
        "sum_sq": float(torch.square(values).sum().item()) if values.numel() else 0.0,
        "max_abs": float(values.abs().max().item()) if values.numel() else 0.0,
        "mean_abs": float(values.abs().mean().item()) if values.numel() else 0.0,
        "finite": bool(torch.isfinite(values).all().item()),
    }


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.stem + ".manifest.json")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = _manifest_path(output)
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"refusing to overwrite existing fixture; pass --overwrite: {output}"
        )
    source_config = json.loads((source / "config.json").read_text())
    model_id = args.source_model_id or str(
        source_config.get("_name_or_path") or source.name
    )
    if Path(model_id).is_absolute():
        model_id = source.name
    modeling, processing, upstream_revision = import_minicpm(
        args.upstream_source, args.upstream_revision
    )
    audio_config = modeling.WhisperConfig(**source_config["audio_config"])

    encoder = modeling.MiniCPMWhisperEncoder(audio_config)
    projection = modeling.MultiModalProjector(
        in_dim=audio_config.d_model,
        out_dim=source_config["hidden_size"],
    )
    selected = load_selected(source)
    source_dtypes = {tensor.dtype for tensor in selected.values()}
    if len(source_dtypes) != 1:
        raise RuntimeError(f"audio tensors have mixed dtypes: {sorted(map(str, source_dtypes))}")
    encoder.load_state_dict(
        {name.removeprefix("apm."): value for name, value in selected.items() if name.startswith("apm.")},
        strict=True,
    )
    projection.load_state_dict(
        {
            name.removeprefix("audio_projection_layer."): value
            for name, value in selected.items()
            if name.startswith("audio_projection_layer.")
        },
        strict=True,
    )
    device = torch.device(args.device)
    # Preserve the source checkpoint dtype through the oracle.  Loading a
    # BF16 shard into a freshly constructed FP32 module would silently create
    # a different numerical contract from the MLX bundle, whose tensors retain
    # the checkpoint dtype.
    source_dtype = next(iter(selected.values())).dtype
    encoder = encoder.to(device=device, dtype=source_dtype).eval()
    projection = projection.to(device=device, dtype=source_dtype).eval()
    audio_dtype = encoder.conv1.weight.dtype
    pool_step = int(source_config["audio_pool_step"])

    processor = processing.MiniCPMAAudioProcessor.from_pretrained(source)
    waveform = deterministic_waveform()
    batch = processor(
        waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    )
    features_nct = batch.input_features.to(device=device, dtype=audio_dtype)
    with torch.inference_mode():
        states = encoder(features_nct, return_dict=True).last_hidden_state
        projected = projection(states)
        embeddings = torch.nn.functional.avg_pool1d(
            projected.transpose(1, 2), kernel_size=pool_step, stride=pool_step
        ).transpose(1, 2)

    tensors: dict[str, torch.Tensor] = {
        "waveform": torch.from_numpy(waveform),
        "features": _as_float_tensor(features_nct.transpose(1, 2)),
        "encoder_states": _as_float_tensor(states),
        "projected_states": _as_float_tensor(projected),
        "embeddings": _as_float_tensor(embeddings),
    }

    # Use the exact first-window/regular-window geometry used by the native
    # streaming frontend.  The stable frame ranges are deliberately sliced
    # from cumulative extraction so that every chunk has the same centered
    # waveform boundary as the official processor.
    streaming_waveform = deterministic_waveform(count=16_480 + 3 * 16_000)
    fixed_processor = processing.MiniCPMAAudioProcessor.from_pretrained(source)
    fixed_processor.set_spac_log_norm(log_floor_db=-10)
    first_full = fixed_processor(
        streaming_waveform[:16_480],
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    ).input_features
    cumulative = fixed_processor(
        streaming_waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    ).input_features
    feature_chunks = [
        first_full[:, :, :100],
        cumulative[:, :, 100:200],
        cumulative[:, :, 200:300],
    ]
    tensors["stream_waveform"] = torch.from_numpy(streaming_waveform)
    cache = None
    stream_real_length = 0
    for index, original_chunk in enumerate(feature_chunks):
        current_real = (original_chunk.shape[-1] - 1) // 2 + 1
        cache, stream_states, stream_embeddings = _run_encoder_chunk(
            encoder,
            projection,
            original_chunk,
            cache=cache,
            device=device,
            audio_dtype=audio_dtype,
            pool_step=pool_step,
            bucket_pad=False,
            bucket_real_length=stream_real_length,
        )
        stream_real_length += current_real
        tensors[f"stream_features_{index}"] = _as_float_tensor(original_chunk.transpose(1, 2))
        tensors[f"stream_encoder_states_{index}"] = _as_float_tensor(stream_states)
        # ``stream_states`` is created under ``torch.inference_mode()`` in
        # ``_run_encoder_chunk``.  Re-enter inference mode for this diagnostic
        # projection as well; PyTorch 2.8 otherwise tries to retain an
        # inference tensor for backward and rejects the export.
        with torch.inference_mode():
            stream_projected = projection(stream_states)
        tensors[f"stream_projected_states_{index}"] = _as_float_tensor(
            stream_projected
        )
        tensors[f"stream_embeddings_{index}"] = _as_float_tensor(stream_embeddings)
        _cache_tensors(f"stream_cache_{index}", cache, tensors)

    # Context mode removes 20 ms (two mel frames) of redundant CNN input on
    # each applicable side.  Keep this as a separate sequence so a parity
    # test can distinguish ordinary streaming from the duplex protocol.
    context_cache = None
    context_real_length = 0
    for index, original_chunk in enumerate(feature_chunks[:2]):
        prefix = 0 if index == 0 else 2
        suffix = 2
        current_real = (original_chunk.shape[-1] - 1) // 2 + 1
        current_real -= (prefix + 1) // 2 if prefix > 0 else 0
        current_real -= (suffix + 1) // 2
        context_cache, context_states, context_embeddings = _run_encoder_chunk(
            encoder,
            projection,
            original_chunk,
            cache=context_cache,
            device=device,
            use_extra_context=True,
            prefix_extra_frames=prefix,
            suffix_extra_frames=suffix,
            audio_dtype=audio_dtype,
            pool_step=pool_step,
            bucket_pad=False,
            bucket_real_length=context_real_length,
        )
        context_real_length += current_real
        tensors[f"context_stream_features_{index}"] = _as_float_tensor(
            original_chunk.transpose(1, 2)
        )
        tensors[f"context_stream_encoder_states_{index}"] = _as_float_tensor(context_states)
        with torch.inference_mode():
            context_projected = projection(context_states)
        tensors[f"context_stream_projected_states_{index}"] = _as_float_tensor(
            context_projected
        )
        tensors[f"context_stream_embeddings_{index}"] = _as_float_tensor(context_embeddings)
        _cache_tensors(f"context_cache_{index}", context_cache, tensors)

    # Snapshot/rollback oracle: run the second chunk speculatively, rebuild
    # the exact first-chunk boundary (the official cache is immutable from the
    # native runtime's perspective), then run the canonical continuation.
    first_cache, _, _ = _run_encoder_chunk(
        encoder,
        projection,
        feature_chunks[0],
        cache=None,
        device=device,
        audio_dtype=audio_dtype,
        pool_step=pool_step,
    )
    speculative_cache, speculative_states, speculative_embeddings = _run_encoder_chunk(
        encoder,
        projection,
        feature_chunks[1],
        cache=_clone_cache(first_cache),
        device=device,
        audio_dtype=audio_dtype,
        pool_step=pool_step,
        bucket_pad=False,
        bucket_real_length=(feature_chunks[0].shape[-1] - 1) // 2 + 1,
    )
    restored_cache, _, _ = _run_encoder_chunk(
        encoder,
        projection,
        feature_chunks[0],
        cache=None,
        device=device,
        audio_dtype=audio_dtype,
        pool_step=pool_step,
    )
    canonical_cache, canonical_states, canonical_embeddings = _run_encoder_chunk(
        encoder,
        projection,
        feature_chunks[1],
        cache=_clone_cache(restored_cache),
        device=device,
        audio_dtype=audio_dtype,
        pool_step=pool_step,
        bucket_pad=False,
        bucket_real_length=(feature_chunks[0].shape[-1] - 1) // 2 + 1,
    )
    tensors["rollback_stream_features_1"] = _as_float_tensor(feature_chunks[1].transpose(1, 2))
    tensors["rollback_speculative_encoder_states_1"] = _as_float_tensor(speculative_states)
    tensors["rollback_speculative_embeddings_1"] = _as_float_tensor(speculative_embeddings)
    tensors["rollback_canonical_encoder_states_1"] = _as_float_tensor(canonical_states)
    tensors["rollback_canonical_embeddings_1"] = _as_float_tensor(canonical_embeddings)
    _cache_tensors("rollback_speculative_cache_1", speculative_cache, tensors)
    _cache_tensors("rollback_canonical_cache_1", canonical_cache, tensors)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            "format": MANIFEST_SCHEMA,
            "oracle_backend": "official_pytorch",
            "upstream_revision": upstream_revision,
            "model_id": model_id,
            "dtype": "float32",
        },
    )

    converter_path = Path(__file__).resolve()
    provenance = source_provenance(source, model_id, args.source_revision)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "oracle_backend": "official_pytorch",
        "upstream": {
            "repository": "OpenBMB/MiniCPM-o-Demo",
            "revision": upstream_revision,
            "revision_kind": "git_commit",
        },
        "source": provenance,
        "converter": {
            "path": converter_path.name,
            "sha256": _sha256_file(converter_path),
        },
        "runtime": {
            "device": str(device),
            "source_dtype": str(audio_dtype).replace("torch.", ""),
            "output_dtype": "float32",
            "sample_rate": 16_000,
            "fft_size": 400,
            "hop_length": 160,
            "mel_bins": int(source_config["audio_config"]["num_mel_bins"]),
            "fixed_log_floor_db": -10.0,
            "first_chunk_samples": 16_480,
            "regular_chunk_samples": 16_000,
            "redundancy_frames": 2,
            "pool_step": pool_step,
        },
        "artifact": {
            "path": output.name,
            "sha256": _sha256_file(output),
            "tensor_count": len(tensors),
            "tensors": {
                name: _tensor_metadata(tensor)
                for name, tensor in sorted(tensors.items())
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": output.name,
                "manifest": manifest_path.name,
                "tensor_count": len(tensors),
                "upstream_revision": upstream_revision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
