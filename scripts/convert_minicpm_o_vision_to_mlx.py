#!/usr/bin/env python3
"""Convert MiniCPM-o SigLIP + Resampler weights to an MLX bundle.

Only the visual path is read from the original checkpoint.  The converter
uses ``safe_open`` and therefore does not materialise the 19GB language model:
the MiniCPM-o shards are inspected by key and only tensors beginning with
``vpm.`` or ``resampler.`` are copied.  The resulting ``model.safetensors``
can be loaded with ``mx.load`` and passed to :class:`MiniCPMOVisionEncoder`.

Example::

    python scripts/convert_minicpm_o_vision_to_mlx.py \
      --input-dir models/MiniCPM-o-4_5 \
      --output-dir models/MiniCPM-o-4_5-vision-mlx

No PyTorch import is performed.  ``safetensors`` is used only for streaming
reads of the source files, while MLX writes the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import mlx.core as mx
import numpy as np
from safetensors import safe_open


VISION_PREFIX = "vpm."
RESAMPLER_PREFIX = "resampler."
FORMAT_VERSION = 1
MANIFEST_SCHEMA = "minicpmo-vision-mlx/v1"
DEFAULT_SOURCE_MODEL_ID = "OpenBMB/MiniCPM-o-4_5"
# Keep checkpoint and Demo implementation provenance independent.  The
# historical ``UPSTREAM_COMMIT`` name remains the implementation identity for
# callers that imported it from this script.
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
UPSTREAM_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
DEMO_IMPLEMENTATION_COMMIT = UPSTREAM_COMMIT
COMPONENT_ID = "minicpmo-4_5.vision-resampler"
DEFAULT_COMPUTE_DTYPE = "float32"


_SAFETENSOR_DTYPE = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "U32": np.uint32,
    "I32": np.int32,
    "U64": np.uint64,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _tensor_set_sha256(weights: Mapping[str, mx.array]) -> str:
    """Hash output key/layout metadata without serialising tensor payloads."""

    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        record = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": [int(dim) for dim in value.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def _read_safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Read only the safetensors header (no tensor payload allocation)."""

    with open(path, "rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"invalid safetensors file (short header): {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > 100 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header length {header_len}: {path}")
        raw_header = stream.read(header_len)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    return 8 + header_len, header


def _read_safetensor_tensor(path: Path, key: str, *, header_offset: int, metadata: Mapping[str, Any]) -> mx.array:
    """Read one tensor directly into MLX, including NumPy-less BF16 support.

    ``safetensors.numpy`` cannot represent BF16 on the Python/NumPy versions
    used by this project.  BF16 values are IEEE-754 16-bit bit patterns, so a
    uint16 MLX array can be viewed as ``mx.bfloat16`` without changing bits.
    """

    if key not in metadata or not isinstance(metadata[key], Mapping):
        raise KeyError(f"tensor is absent from safetensors header: {key}")
    spec = metadata[key]
    offsets = spec.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        raise ValueError(f"invalid data_offsets for {key}")
    start, end = int(offsets[0]), int(offsets[1])
    if start < 0 or end < start:
        raise ValueError(f"invalid data range for {key}: {offsets}")
    with open(path, "rb") as stream:
        stream.seek(header_offset + start)
        raw = stream.read(end - start)
    if len(raw) != end - start:
        raise ValueError(f"short tensor payload for {key}: expected {end-start}, got {len(raw)}")

    dtype_name = str(spec.get("dtype", ""))
    shape = tuple(int(dim) for dim in spec.get("shape", ()))
    if dtype_name == "BF16":
        # Safetensors stores little-endian BF16 words.  MLX's view operation
        # preserves those words and changes only the interpretation.
        bits = mx.array(np.frombuffer(raw, dtype="<u2").copy(), dtype=mx.uint16)
        result = mx.view(bits, mx.bfloat16)
    else:
        np_dtype = _SAFETENSOR_DTYPE.get(dtype_name)
        if np_dtype is None:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r} for {key}")
        result = mx.array(np.frombuffer(raw, dtype=np_dtype).copy())
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if result.size != expected:
        raise ValueError(f"payload size mismatch for {key}: {result.size} values, expected {expected}")
    return result.reshape(shape)


def _shard_paths(input_dir: Path, index: Mapping[str, Any]) -> list[Path]:
    names = sorted(set(index.get("weight_map", {}).values()))
    paths = [input_dir / name for name in names]
    if not paths:
        paths = sorted(input_dir.glob("*.safetensors"))
    return paths


def audit_visual_keys(input_dir: str | Path) -> dict[str, Any]:
    """Audit source visual keys without reading tensor payloads.

    This is intentionally separate from :func:`collect_visual_weights`: on an
    Apple laptop, a key audit should never allocate the 2.7GB fourth shard.
    The returned report is suitable for CI and for inspecting a downloaded
    checkpoint before deciding whether to run conversion.
    """

    input_dir = Path(input_dir)
    index_path = input_dir / "model.safetensors.index.json"
    index = _load_json(index_path) if index_path.exists() else {}
    indexed = {
        key: shard
        for key, shard in index.get("weight_map", {}).items()
        if key.startswith((VISION_PREFIX, RESAMPLER_PREFIX))
    }
    shard_keys: dict[str, list[str]] = {}
    for shard in _shard_paths(input_dir, index):
        if not shard.exists():
            continue
        with safe_open(str(shard), framework="numpy") as reader:
            keys = [k for k in reader.keys() if k.startswith((VISION_PREFIX, RESAMPLER_PREFIX))]
        if keys:
            shard_keys[shard.name] = sorted(keys)
    on_disk = {key for keys in shard_keys.values() for key in keys}
    missing = sorted(set(indexed) - on_disk)
    # A single-file checkpoint has no index by design; in that case every
    # visual key discovered on disk is considered indexed implicitly.
    unindexed = sorted(on_disk - set(indexed)) if indexed else []
    mapped = {}
    for key in sorted(on_disk):
        if key.startswith(VISION_PREFIX):
            mapped[key] = _mlx_key(key)
        elif key == "resampler.attn.in_proj_weight":
            mapped[key] = [
                "resampler.attn.q_proj.weight",
                "resampler.attn.k_proj.weight",
                "resampler.attn.v_proj.weight",
            ]
        elif key == "resampler.attn.in_proj_bias":
            mapped[key] = [
                "resampler.attn.q_proj.bias",
                "resampler.attn.k_proj.bias",
                "resampler.attn.v_proj.bias",
            ]
        else:
            mapped[key] = key
    report = {
        "input_dir": str(input_dir),
        "indexed_visual_tensors": len(indexed),
        "on_disk_visual_tensors": len(on_disk),
        "missing_indexed_tensors": missing,
        "unindexed_visual_tensors": unindexed,
        "complete": bool(on_disk) and not missing and not unindexed and (not indexed or len(indexed) == len(on_disk)),
        "source_shards": shard_keys,
        "key_mapping": mapped,
    }
    return report


def _mlx_key(key: str) -> str:
    if key.startswith(VISION_PREFIX):
        return "vision." + key[len(VISION_PREFIX) :]
    return key


def _already_mlx_conv(value: np.ndarray, patch_size: int) -> bool:
    # MLX Conv2d uses [out, kernel_h, kernel_w, in].  Source PyTorch weights
    # are [out, in, kernel_h, kernel_w].  For MiniCPM's 3-channel input the
    # second axis is unambiguously 3 in the source and patch_size in MLX.
    return value.ndim == 4 and value.shape[1] == patch_size and value.shape[-1] == 3


def collect_visual_weights(
    input_dir: str | Path,
    *,
    source_model_id: str = DEFAULT_SOURCE_MODEL_ID,
    source_revision: str = MODEL_REPOSITORY_REVISION,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Read and transform visual tensors without loading unrelated weights."""

    input_dir = Path(input_dir)
    config_path = input_dir / "config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    vision_config = config.get("vision_config", {})
    patch_size = int(vision_config.get("patch_size", config.get("patch_size", 14)))
    index_path = input_dir / "model.safetensors.index.json"
    index = _load_json(index_path) if index_path.exists() else {}
    wanted = {k for k in index.get("weight_map", {}) if k.startswith((VISION_PREFIX, RESAMPLER_PREFIX))}
    result: dict[str, mx.array] = {}
    source_keys: list[str] = []
    found = set()

    for shard in _shard_paths(input_dir, index):
        if not shard.exists():
            raise FileNotFoundError(f"checkpoint shard is missing: {shard}")
        header_offset, header = _read_safetensor_header(shard)
        with safe_open(str(shard), framework="numpy") as reader:
            keys = reader.keys()
            # ``wanted`` is absent when an unindexed single-file checkpoint is
            # supplied, so use the prefix predicate in that case.
            for key in keys:
                if wanted and key not in wanted:
                    continue
                if not key.startswith((VISION_PREFIX, RESAMPLER_PREFIX)):
                    continue
                # Do not call ``reader.get_tensor``: NumPy has no BF16 dtype.
                # Header parsing + direct seek keeps the mixed language/TTS
                # shard lazy and supports BF16 on every supported NumPy build.
                value = _read_safetensor_tensor(shard, key, header_offset=header_offset, metadata=header)
                if key == "resampler.attn.in_proj_weight":
                    q, k, v = mx.split(value, 3, axis=0)
                    result["resampler.attn.q_proj.weight"] = q
                    result["resampler.attn.k_proj.weight"] = k
                    result["resampler.attn.v_proj.weight"] = v
                elif key == "resampler.attn.in_proj_bias":
                    q, k, v = mx.split(value, 3, axis=0)
                    result["resampler.attn.q_proj.bias"] = q
                    result["resampler.attn.k_proj.bias"] = k
                    result["resampler.attn.v_proj.bias"] = v
                else:
                    out_key = _mlx_key(key)
                    if key.endswith("embeddings.patch_embedding.weight"):
                        # Shape inspection is enough; do not convert BF16 to
                        # NumPy just to decide the convolution permutation.
                        if value.ndim == 4 and not (
                            value.shape[1] == patch_size and value.shape[-1] == 3
                        ):
                            value = value.transpose(0, 2, 3, 1)
                    result[out_key] = value
                found.add(key)
                source_keys.append(key)

    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"indexed visual tensors were not found in shards (first 10): {missing[:10]}")
    metadata = {
        "source_model": source_model_id,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "source_basename": input_dir.name,
        "source_model_type": config.get("model_type", "minicpmo"),
        "converter": Path(__file__).name,
        "schema": MANIFEST_SCHEMA,
        "component": COMPONENT_ID,
        "num_tensors": str(len(result)),
        "source_tensors": str(len(source_keys)),
        "vision_config": vision_config,
        "query_num": config.get("query_num", 64),
        "hidden_size": config.get("hidden_size", 4096),
        "compute_dtype": DEFAULT_COMPUTE_DTYPE,
    }
    return result, metadata


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_model_id: str = DEFAULT_SOURCE_MODEL_ID,
    source_revision: str = MODEL_REPOSITORY_REVISION,
) -> dict[str, Any]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Run the header-only audit before allocating any source tensor payloads.
    # This also gives the manifest an authoritative index/shard mapping.
    audit = audit_visual_keys(input_dir)
    if not audit["complete"]:
        raise ValueError(
            "visual source key audit failed: "
            + json.dumps(
                {
                    key: audit[key]
                    for key in (
                        "missing_indexed_tensors",
                        "unindexed_visual_tensors",
                    )
                    if audit[key]
                },
                ensure_ascii=False,
            )
        )
    weights, metadata = collect_visual_weights(
        input_dir,
        source_model_id=source_model_id,
        source_revision=source_revision,
    )
    mx.eval(list(weights.values()))
    weight_path = output_dir / "model.safetensors"
    mx.save_safetensors(
        weight_path,
        weights,
        metadata={k: str(v) for k, v in metadata.items() if k != "vision_config"},
    )

    config = _load_json(input_dir / "config.json") if (input_dir / "config.json").exists() else {}
    bundle_config = {
        "model_type": "minicpmo_vision_mlx",
        "format_version": FORMAT_VERSION,
        "format": MANIFEST_SCHEMA,
        "component_id": COMPONENT_ID,
        "vision_config": config.get("vision_config", {}),
        "query_num": config.get("query_num", 64),
        "hidden_size": config.get("hidden_size", 4096),
        "patch_size": config.get("slice_config", {}).get("patch_size", 14),
        "compute_dtype": DEFAULT_COMPUTE_DTYPE,
        "source_model": source_model_id,
        "source_model_dir_name": input_dir.name,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "weight_file": "model.safetensors",
        "weight_count": len(weights),
    }
    _write_json(output_dir / "config.json", bundle_config)
    # Keep the exact preprocessing settings beside the visual weights.  This
    # avoids coupling runtime loading to the large original HF directory.
    preprocessor = input_dir / "preprocessor_config.json"
    if preprocessor.exists():
        shutil.copy2(preprocessor, output_dir / "preprocessor_config.json")
    source_config_path = input_dir / "config.json"
    source_index_path = input_dir / "model.safetensors.index.json"
    source_shards = []
    for shard_name in sorted(audit.get("source_shards", {})):
        shard_path = input_dir / shard_name
        source_shards.append(_file_record(shard_path))
    source_identity = {
        "model_id": source_model_id,
        "directory_name": input_dir.name,
        "revision": source_revision,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "config": _file_record(source_config_path) if source_config_path.exists() else None,
        "index": _file_record(source_index_path) if source_index_path.exists() else None,
        "shards": source_shards,
        "key_count": audit["on_disk_visual_tensors"],
        "shards_keys": audit.get("source_shards", {}),
    }
    files = {
        "model.safetensors": _file_record(weight_path),
        "config.json": _file_record(output_dir / "config.json"),
    }
    if preprocessor.exists():
        files["preprocessor_config.json"] = _file_record(
            output_dir / "preprocessor_config.json"
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "format_version": FORMAT_VERSION,
        "component_id": COMPONENT_ID,
        "model_type": "minicpmo_vision_mlx",
        "compute_dtype": DEFAULT_COMPUTE_DTYPE,
        "source_model": source_model_id,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "converter": {
            "path": Path(__file__).name,
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "source": source_identity,
        "weights": {
            "path": "model.safetensors",
            "kind": "single_file",
            "tensor_count": len(weights),
            "tensor_set_sha256": _tensor_set_sha256(weights),
        },
        "files": files,
        "strict": True,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return metadata | {
        "output_dir": str(output_dir),
        "manifest": "manifest.json",
        "source_model": source_model_id,
        "source_revision": source_revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL_ID,
        help="Stable source model id recorded in config/manifest (never a local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=MODEL_REPOSITORY_REVISION,
        help=(
            "Model repository revision for source weights (default: "
            f"{MODEL_REPOSITORY_REVISION})"
        ),
    )
    parser.add_argument("--list-only", action="store_true", help="list visual tensors without writing output")
    parser.add_argument("--audit-only", action="store_true", help="audit visual keys without reading tensor payloads")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_visual_keys(args.input_dir), ensure_ascii=False, indent=2))
        return
    if args.list_only:
        weights, metadata = collect_visual_weights(
            args.input_dir,
            source_model_id=args.source_model,
            source_revision=args.source_revision,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        for key, value in sorted(weights.items()):
            print(f"{key}\t{tuple(value.shape)}\t{value.dtype}")
        return
    metadata = convert(
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        source_model_id=args.source_model,
        source_revision=args.source_revision,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
