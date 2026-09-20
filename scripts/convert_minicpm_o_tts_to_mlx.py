#!/usr/bin/env python3
"""Extract the MiniCPM-o semantic TTS path into a small MLX bundle.

MiniCPM-o's Hugging Face checkpoint contains the language model, vision
encoder, audio encoder and TTS decoder in the same four safetensors shards.
The TTS path is independent of the first three paths.  This converter copies
only the ``tts.*`` tensors needed by :class:`minicpm_mlx.MLXMiniCPMTTS` and
``MiniCPMTTSCondition`` into a sibling directory::

    python scripts/convert_minicpm_o_tts_to_mlx.py \
      --input-dir models/MiniCPM-o-4_5 \
      --output-dir models/MiniCPM-o-4_5-tts-mlx

The source is read through safetensors headers and direct byte ranges.  In
particular, it never calls ``mx.load`` on a language-model shard, and a
BF16 source tensor is copied without going through NumPy (which does not
support BF16 on all Python versions used by this project).

The output keeps the official ``tts.`` names.  This deliberately matches the
Python MLX loader and makes parity diagnostics possible without a second key
translation layer.  The two TTS tensor families that are not used by the MLX
decoder (``tts.model.embed_tokens.weight`` and ``tts.projector_spk.*``) are
recorded in the strict key manifest as intentional drops; every other source
TTS key is required and any unknown key fails conversion.  The generated
``manifest.json`` also pins the upstream commit/source config/index/TTS shard
hashes and this converter's own SHA256, so a Swift parity run can be recreated
without trusting mutable ``main`` code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

import mlx.core as mx
import numpy as np
from safetensors import safe_open


FORMAT_VERSION = 1
CONVERTER_VERSION = "1.1.0"
TTS_PREFIX = "tts."

# The Swift semantic decoder is validated against this exact upstream source,
# rather than whichever ``modeling_minicpmo.py`` happens to be installed in a
# developer's virtualenv.  Keep these identities in every generated manifest
# so a parity fixture can be reproduced years later.
# The implementation identity is the pinned Demo checkout.  Keep it distinct
# from the model repository that provides the ``source_model_id`` weights.
UPSTREAM_REPOSITORY = "OpenBMB/MiniCPM-o-Demo"
# ``UPSTREAM_COMMIT`` identifies the Demo implementation.  The checkpoint
# revision is tracked independently so a model artifact can never be mistaken
# for a source-code checkout.
UPSTREAM_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
DEMO_IMPLEMENTATION_COMMIT = UPSTREAM_COMMIT
MODEL_REPOSITORY = "OpenBMB/MiniCPM-o-4_5"
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEFAULT_SOURCE_MODEL_ID = "OpenBMB/MiniCPM-o-4_5"
COMPONENT_ID = "minicpmo-4_5.tts-semantic"
UPSTREAM_MODELING_SHA256 = "877f20cea7282bb0d05684393a1580646d1f264a2299a42f82467fd99f4028af"
UPSTREAM_UNIFIED_MODELING_SHA256 = "9fceb75f969cfafc6381adf2b59255529be7fb34d64ad784b8417d224ffbabb9"

# ``embed_tokens`` is never read by the MLX decoder because all calls supply
# ``input_embeddings``.  ``projector_spk`` is used only by the optional speaker
# conditioning path, which is intentionally outside this semantic TTS bundle.
INTENTIONAL_DROPS = frozenset(
    {
        "tts.model.embed_tokens.weight",
        "tts.projector_spk.linear1.bias",
        "tts.projector_spk.linear1.weight",
        "tts.projector_spk.linear2.bias",
        "tts.projector_spk.linear2.weight",
    }
)

CONDITION_KEYS = (
    "tts.emb_text.weight",
    "tts.projector_semantic.linear1.weight",
    "tts.projector_semantic.linear1.bias",
    "tts.projector_semantic.linear2.weight",
    "tts.projector_semantic.linear2.bias",
)

DECODER_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


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
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Read a safetensors header without touching any tensor payload."""

    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors file (short header): {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100 * 1024 * 1024:
            raise ValueError(
                f"unreasonable safetensors header length {header_length}: {path}"
            )
        raw_header = stream.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return 8 + header_length, header


def _read_safetensor_tensor(
    path: Path,
    key: str,
    *,
    header_offset: int,
    metadata: Mapping[str, Any],
) -> mx.array:
    """Read exactly one source tensor into MLX.

    The direct seek is important for mixed BF16 checkpoints.  ``safe_open``'s
    NumPy reader cannot represent BF16 on older NumPy builds; viewing the
    little-endian uint16 words as ``mx.bfloat16`` preserves the source bits.
    """

    spec = metadata.get(key)
    if not isinstance(spec, Mapping):
        raise KeyError(f"tensor is absent from safetensors header: {key}")
    offsets = spec.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        raise ValueError(f"invalid data_offsets for {key}")
    start, end = int(offsets[0]), int(offsets[1])
    if start < 0 or end < start:
        raise ValueError(f"invalid data range for {key}: {offsets}")
    with path.open("rb") as stream:
        stream.seek(header_offset + start)
        raw = stream.read(end - start)
    if len(raw) != end - start:
        raise ValueError(
            f"short tensor payload for {key}: expected {end - start}, got {len(raw)}"
        )

    dtype_name = str(spec.get("dtype", ""))
    shape = tuple(int(dim) for dim in spec.get("shape", ()))
    if dtype_name == "BF16":
        bits = mx.array(np.frombuffer(raw, dtype="<u2").copy(), dtype=mx.uint16)
        result = mx.view(bits, mx.bfloat16)
    else:
        np_dtype = _SAFETENSOR_DTYPE.get(dtype_name)
        if np_dtype is None:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r} for {key}")
        result = mx.array(np.frombuffer(raw, dtype=np_dtype).copy())
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if int(result.size) != expected:
        raise ValueError(
            f"payload size mismatch for {key}: {result.size} values, expected {expected}"
        )
    return result.reshape(shape)


def _source_shards(input_dir: Path, index: Mapping[str, Any]) -> list[Path]:
    """Return only shards that can contain TTS keys when an index is present."""

    weight_map = index.get("weight_map", {})
    if isinstance(weight_map, Mapping):
        names = sorted(
            {
                str(shard)
                for key, shard in weight_map.items()
                if str(key).startswith(TTS_PREFIX)
            }
        )
        if names:
            return [input_dir / name for name in names]
    candidates = sorted(input_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no safetensors shards found in {input_dir}")
    return candidates


def _expected_keys(tts_config: Mapping[str, Any]) -> list[str]:
    try:
        layers = int(tts_config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tts_config.num_hidden_layers is required") from exc
    if layers <= 0:
        raise ValueError(f"tts_config.num_hidden_layers must be positive, got {layers}")
    result = list(CONDITION_KEYS)
    result += ["tts.emb_code.0.weight"]
    result += [
        "tts.head_code.0.parametrizations.weight.original0",
        "tts.head_code.0.parametrizations.weight.original1",
        "tts.model.norm.weight",
    ]
    for layer in range(layers):
        result.extend(f"tts.model.layers.{layer}.{name}" for name in DECODER_LAYER_KEYS)
    return sorted(result)


def _audit_source(
    input_dir: Path,
    *,
    expected: Iterable[str],
) -> dict[str, Any]:
    """Audit source key names and shard placement without loading payloads."""

    index_path = input_dir / "model.safetensors.index.json"
    index = _load_json(index_path) if index_path.exists() else {}
    expected_set = set(expected)
    indexed_map = index.get("weight_map", {})
    indexed_tts = {
        str(key): str(shard)
        for key, shard in indexed_map.items()
        if str(key).startswith(TTS_PREFIX)
    } if isinstance(indexed_map, Mapping) else {}

    source_shards = _source_shards(input_dir, index)
    shard_keys: dict[str, list[str]] = {}
    on_disk: set[str] = set()
    for shard in source_shards:
        if not shard.exists():
            raise FileNotFoundError(f"checkpoint shard is missing: {shard}")
        with safe_open(str(shard), framework="numpy") as reader:
            keys = sorted(str(key) for key in reader.keys() if str(key).startswith(TTS_PREFIX))
        if keys:
            shard_keys[shard.name] = keys
            on_disk.update(keys)

    if indexed_tts:
        index_missing_on_disk = sorted(set(indexed_tts) - on_disk)
        index_unindexed = sorted(on_disk - set(indexed_tts))
        wrong_shard = sorted(
            key
            for key, shard in indexed_tts.items()
            if key in on_disk and shard not in shard_keys
        )
    else:
        # A single-file checkpoint may omit an index; discovered keys become
        # the source-of-truth in that case.
        index_missing_on_disk = []
        index_unindexed = []
        wrong_shard = []

    unknown = sorted(on_disk - expected_set - INTENTIONAL_DROPS)
    missing = sorted(expected_set - on_disk)
    unexpected_drops = sorted(INTENTIONAL_DROPS.intersection(on_disk) - (on_disk - expected_set))
    # The last expression is intentionally explicit: it documents that an
    # allowlisted drop is acceptable only when it really exists on disk.
    intentional_drops = sorted(INTENTIONAL_DROPS.intersection(on_disk))
    complete = not (
        missing
        or unknown
        or index_missing_on_disk
        or index_unindexed
        or wrong_shard
        or (set(intentional_drops) != INTENTIONAL_DROPS.intersection(on_disk))
    )
    return {
        "source_dir": str(input_dir),
        "source_index": index_path.name if index_path.exists() else None,
        "source_shards": [
            {
                "name": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": None,
            }
            for shard in source_shards
        ],
        "source_tts_shards": shard_keys,
        "indexed_tts_tensors": len(indexed_tts),
        "on_disk_tts_tensors": len(on_disk),
        "missing_keys": missing,
        "unknown_keys": unknown,
        "intentional_drops": intentional_drops,
        "index_missing_on_disk": index_missing_on_disk,
        "index_unindexed": index_unindexed,
        "index_wrong_shard": wrong_shard,
        "complete": bool(complete),
    }


def _collect_weights(
    input_dir: Path,
    *,
    expected: Iterable[str],
    audit: Mapping[str, Any],
) -> dict[str, mx.array]:
    """Read expected tensors from only the audited TTS shards."""

    expected_set = set(expected)
    by_shard = audit.get("source_tts_shards", {})
    if not isinstance(by_shard, Mapping):
        raise ValueError("source audit has invalid source_tts_shards")
    weights: dict[str, mx.array] = {}
    for shard_name, keys in by_shard.items():
        shard = input_dir / str(shard_name)
        header_offset, header = _read_safetensor_header(shard)
        # ``keys`` comes from the header audit, so no unrelated language-model
        # tensor is touched even if a future checkpoint co-locates both paths.
        for key in keys:
            if key not in expected_set:
                continue
            weights[key] = _read_safetensor_tensor(
                shard,
                key,
                header_offset=header_offset,
                metadata=header,
            )
    missing = sorted(expected_set - set(weights))
    if missing:
        raise ValueError(f"expected TTS tensors were not readable (first 10): {missing[:10]}")
    return {key: weights[key] for key in sorted(weights)}


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _converter_record() -> dict[str, Any]:
    """Return the exact converter source identity embedded in a bundle."""

    source = Path(__file__).resolve()
    return {
        "name": source.name,
        "version": CONVERTER_VERSION,
        "sha256": _sha256(source),
    }


def _bundle_paths(bundle_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        bundle_dir / "model.safetensors",
        bundle_dir / "config.json",
        bundle_dir / "keys.json",
        bundle_dir / "checksums.json",
    )


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_model_id: str = DEFAULT_SOURCE_MODEL_ID,
    source_revision: str = MODEL_REPOSITORY_REVISION,
) -> dict[str, Any]:
    """Convert an original MiniCPM-o directory and return the manifest."""

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    source_revision = source_revision or MODEL_REPOSITORY_REVISION
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input checkpoint directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --overwrite"
            )
        # Never remove the source directory accidentally when paths alias.
        if output_dir.resolve() == input_dir.resolve():
            raise ValueError("output directory must differ from input checkpoint directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = input_dir / "config.json"
    source_config = _load_json(source_config_path) if source_config_path.exists() else {}
    tts_config = source_config.get("tts_config", source_config)
    if not isinstance(tts_config, Mapping):
        raise ValueError("source config.tts_config must be a JSON object")
    expected = _expected_keys(tts_config)
    audit = _audit_source(input_dir, expected=expected)
    if not audit["complete"]:
        raise ValueError(
            "TTS source key audit failed: "
            + json.dumps(
                {
                    key: audit[key]
                    for key in (
                        "missing_keys",
                        "unknown_keys",
                        "index_missing_on_disk",
                        "index_unindexed",
                        "index_wrong_shard",
                    )
                    if audit[key]
                },
                ensure_ascii=False,
            )
        )

    weights = _collect_weights(input_dir, expected=expected, audit=audit)
    mx.eval(list(weights.values()))
    weight_path = output_dir / "model.safetensors"
    # Keep safetensors metadata portable: a local checkout path would make a
    # copied bundle machine-specific and would defeat provenance auditing.
    mx.save_safetensors(
        str(weight_path),
        weights,
        metadata={
            "format": "mlx",
            "schema": "minicpmo-tts-mlx/v1",
            "component": COMPONENT_ID,
            "source_model": source_model_id,
            "source_revision": source_revision,
            "revision_kind": "model_repository",
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        },
    )

    # Keep the complete upstream TTS config, not only dimensions used by the
    # current Python implementation.  This allows Swift and future runtimes
    # to validate protocol/token IDs without reopening the 19 GB checkpoint.
    bundle_config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_type": "minicpmo_tts_mlx",
        "source_model_type": source_config.get("model_type", "minicpmo"),
        "source_model": source_model_id,
        "source_model_repository": MODEL_REPOSITORY,
        "source_model_dir_name": input_dir.name,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "weight_file": "model.safetensors",
        "weight_count": len(weights),
        "weight_keys": expected,
        "tts_config": dict(tts_config),
    }
    _write_json(output_dir / "config.json", bundle_config)

    keys_manifest = {
        "format_version": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "source_prefix": TTS_PREFIX,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "selected_keys": expected,
        "output_keys": sorted(weights),
        "intentional_drops": audit["intentional_drops"],
        "source_tts_shards": audit["source_tts_shards"],
        "strict": True,
    }
    _write_json(output_dir / "keys.json", keys_manifest)

    model_record = _file_record(weight_path)
    config_record = _file_record(output_dir / "config.json")
    keys_record = _file_record(output_dir / "keys.json")
    checksums = {
        "algorithm": "sha256",
        "files": {
            "model.safetensors": model_record,
            "config.json": config_record,
            "keys.json": keys_record,
        },
    }
    _write_json(output_dir / "checksums.json", checksums)

    source_index = input_dir / "model.safetensors.index.json"
    # Record only shards that actually contain selected TTS tensors.  Hashing
    # all four 4.9 GB language-model shards would make a no-op verification
    # unnecessarily expensive, while this still fixes the exact source bytes
    # used by the converter.
    tts_shard_names = sorted(audit["source_tts_shards"])
    source_weight_files = []
    for shard_name in tts_shard_names:
        shard_path = input_dir / shard_name
        record = _file_record(shard_path)
        record["name"] = shard_name
        source_weight_files.append(record)
    source_identity = {
        "source_model_id": source_model_id,
        "model_repository": MODEL_REPOSITORY,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_repository": UPSTREAM_REPOSITORY,
        "implementation_revision_kind": "git_commit",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "modeling_sha256": UPSTREAM_MODELING_SHA256,
        "unified_modeling_sha256": UPSTREAM_UNIFIED_MODELING_SHA256,
        "model_dir_name": input_dir.name,
        "config_sha256": _sha256(source_config_path) if source_config_path.exists() else None,
        "index_sha256": _sha256(source_index) if source_index.exists() else None,
        "weight_files": source_weight_files,
        "tts_shards": audit["source_tts_shards"],
    }
    converter_identity = _converter_record()
    manifest = {
        "format_version": FORMAT_VERSION,
        "converter": "convert_minicpm_o_tts_to_mlx.py",
        "converter_version": CONVERTER_VERSION,
        "converter_sha256": converter_identity["sha256"],
        "component_id": COMPONENT_ID,
        "source_model_id": source_model_id,
        "source_model_repository": MODEL_REPOSITORY,
        "model_type": "minicpmo_tts_mlx",
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "revision_kind": "git_commit",
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "modeling_sha256": UPSTREAM_MODELING_SHA256,
            "unified_modeling_sha256": UPSTREAM_UNIFIED_MODELING_SHA256,
        },
        "source": source_identity,
        "selected_key_count": len(expected),
        "selected_keys": expected,
        "intentional_drops": audit["intentional_drops"],
        "files": {
            "model.safetensors": model_record,
            "config.json": config_record,
            "keys.json": keys_record,
            "checksums.json": _file_record(output_dir / "checksums.json"),
        },
        "strict": True,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest | {"output_dir": str(output_dir)}


def verify_bundle(
    bundle_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    source_revision: str = MODEL_REPOSITORY_REVISION,
) -> dict[str, Any]:
    """Verify hashes, config, keys and safetensors header of an MLX bundle.

    When ``source_dir`` is supplied, the fixed source config/index and every
    selected TTS shard recorded in ``manifest.json`` are hashed again.  This
    turns the manifest into a complete provenance check while keeping the
    normal bundle-only verify fast.
    """

    bundle_dir = Path(bundle_dir)
    source_revision = source_revision or MODEL_REPOSITORY_REVISION
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing TTS bundle manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported TTS bundle format: {manifest.get('format_version')}")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("TTS bundle manifest has no fixed upstream identity")
    expected_upstream = {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "revision_kind": "git_commit",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "modeling_sha256": UPSTREAM_MODELING_SHA256,
        "unified_modeling_sha256": UPSTREAM_UNIFIED_MODELING_SHA256,
    }
    if any(upstream.get(key) != value for key, value in expected_upstream.items()):
        raise ValueError(
            "TTS bundle was produced from a different upstream source: "
            + json.dumps(upstream, ensure_ascii=False, sort_keys=True)
        )
    if manifest.get("source_revision") != source_revision:
        raise ValueError("TTS bundle source_revision does not identify the model repository revision")
    if manifest.get("revision_kind") != "model_repository":
        raise ValueError("TTS bundle revision_kind must describe model weights")
    if manifest.get("upstream_commit") != DEMO_IMPLEMENTATION_COMMIT:
        raise ValueError("TTS bundle upstream_commit must identify the Demo implementation")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("weight_files"), list):
        raise ValueError("TTS bundle manifest has no source weight-file identities")
    expected_source_identity = {
        "source_model_id": manifest.get("source_model_id", DEFAULT_SOURCE_MODEL_ID),
        "model_repository": MODEL_REPOSITORY,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_repository": UPSTREAM_REPOSITORY,
        "implementation_revision_kind": "git_commit",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
    }
    if any(source.get(key) != value for key, value in expected_source_identity.items()):
        raise ValueError(
            "TTS source identity mixes model weights and Demo implementation: "
            + json.dumps(dict(source), ensure_ascii=False, sort_keys=True)
        )
    if manifest.get("converter_sha256") is None:
        raise ValueError("TTS bundle manifest has no converter SHA256")
    config_path = bundle_dir / "config.json"
    keys_path = bundle_dir / "keys.json"
    checksums_path = bundle_dir / "checksums.json"
    weight_path = bundle_dir / "model.safetensors"
    for path in (config_path, keys_path, checksums_path, weight_path):
        if not path.exists():
            raise FileNotFoundError(f"TTS bundle is missing {path.name}")

    checksums = _load_json(checksums_path)
    if checksums.get("algorithm") != "sha256" or not isinstance(checksums.get("files"), Mapping):
        raise ValueError("invalid TTS bundle checksums.json")
    for name, record in checksums["files"].items():
        path = bundle_dir / str(name)
        if not path.exists():
            raise FileNotFoundError(f"checksums references missing file {name}")
        actual = _file_record(path)
        if actual != record:
            raise ValueError(f"checksum mismatch for {name}: expected {record}, got {actual}")

    config = _load_json(config_path)
    keys = _load_json(keys_path)
    required = config.get("weight_keys")
    selected = keys.get("selected_keys")
    output = keys.get("output_keys")
    if not isinstance(required, list) or not isinstance(selected, list) or not isinstance(output, list):
        raise ValueError("TTS bundle config/keys manifest has no strict key list")
    if required != selected:
        raise ValueError("config.weight_keys and keys.selected_keys differ")
    if output != sorted(output) or output != sorted(required):
        raise ValueError("TTS bundle output key set/order is not strict")
    header_offset, header = _read_safetensor_header(weight_path)
    del header_offset
    metadata = header.get("__metadata__", {})
    expected_metadata = {
        "format": "mlx",
        "schema": "minicpmo-tts-mlx/v1",
        "component": COMPONENT_ID,
        "source_model": manifest.get("source_model_id", DEFAULT_SOURCE_MODEL_ID),
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
    }
    if metadata != expected_metadata:
        raise ValueError(
            "TTS safetensors metadata mismatch: "
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        )
    tensor_keys = sorted(key for key in header if key != "__metadata__")
    if tensor_keys != sorted(output):
        raise ValueError(
            f"TTS bundle tensor keys differ from manifest: expected {sorted(output)}, got {tensor_keys}"
        )
    source_verified: bool | None = None
    if source_dir is not None:
        source_root = Path(source_dir)
        if not source_root.is_dir():
            raise NotADirectoryError(f"source checkpoint directory does not exist: {source_root}")
        source_verified = True
        source_identity = source
        config_sha = source_identity.get("config_sha256")
        config_path = source_root / "config.json"
        if config_sha is not None and (
            not config_path.exists() or _sha256(config_path) != config_sha
        ):
            source_verified = False
        index_sha = source_identity.get("index_sha256")
        index_path = source_root / "model.safetensors.index.json"
        if index_sha is not None and (
            not index_path.exists() or _sha256(index_path) != index_sha
        ):
            source_verified = False
        for record in source_identity["weight_files"]:
            if not isinstance(record, Mapping):
                source_verified = False
                continue
            name = record.get("name")
            if not isinstance(name, str):
                source_verified = False
                continue
            path = source_root / name
            if not path.exists() or _file_record(path) != {
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }:
                source_verified = False
        if not source_verified:
            raise ValueError("source checkpoint does not match TTS bundle provenance")

    # The manifest's own file records intentionally exclude manifest.json to
    # avoid a self-referential hash.  Check the record shape, but report its
    # hash for callers that want to archive it separately.
    return {
        "bundle_dir": str(bundle_dir),
        "model_type": config.get("model_type"),
        "weight_count": len(tensor_keys),
        "selected_key_count": len(required),
        "manifest_sha256": _sha256(manifest_path),
        "source_verified": source_verified,
        "verified": True,
    }


def audit_source(input_dir: str | Path) -> dict[str, Any]:
    """Public header-only source audit used by CI and diagnostics."""

    input_dir = Path(input_dir)
    config = _load_json(input_dir / "config.json") if (input_dir / "config.json").exists() else {}
    tts_config = config.get("tts_config", config)
    expected = _expected_keys(tts_config)
    return _audit_source(input_dir, expected=expected) | {
        "expected_keys": expected,
        "expected_key_count": len(expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument("--audit-only", action="store_true", help="audit source headers only")
    parser.add_argument("--verify", type=Path, help="verify an existing TTS MLX bundle")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="with --verify, also verify fixed source config/index/TTS shard hashes",
    )
    args = parser.parse_args()
    if args.verify is not None:
        print(
            json.dumps(
                verify_bundle(
                    args.verify,
                    source_dir=args.source_dir,
                    source_revision=args.source_revision,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.input_dir is None:
        parser.error("--input-dir is required unless --verify is used")
    if args.audit_only:
        print(json.dumps(audit_source(args.input_dir), ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --audit-only is used")
    print(
        json.dumps(
            convert(
                args.input_dir,
                args.output_dir,
                overwrite=args.overwrite,
                source_model_id=args.source_model,
                source_revision=args.source_revision,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
