#!/usr/bin/env python3
"""Convert only MiniCPM-o's Qwen3 language backbone to an MLX bundle.

The official MiniCPM-o checkpoint contains audio, vision, TTS, and Qwen3
weights in the same safetensor shards.  This script selects the ``llm.``
namespace, renames it to the MLX Qwen3 namespace, and (only with ``--write``)
applies affine 8-bit/group-64 quantization.  The default action is a dry-run;
``--verify`` validates an already-written bundle and never rewrites it.  Pass
``--bits 0`` to write an independent dense BF16 bundle; this mode refuses to
replace an existing quantized output directory.

The source model and the runtime are deliberately kept separate.  The
generated ``provenance.json`` and ``key_manifest.json`` make the conversion
auditable without depending on a mutable Hugging Face ``main`` branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


# Keep the model artifact and the Demo implementation identities separate.
# The former identifies the checkpoint bytes; the latter identifies the
# Python/Swift parity implementation used to define the conversion contract.
PINNED_MODEL_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
PINNED_DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
# Historical callers imported this name.  It remains an alias for the Demo
# implementation commit, never for a model-weight revision.
PINNED_UPSTREAM_COMMIT = PINNED_DEMO_IMPLEMENTATION_COMMIT
DEFAULT_SOURCE_MODEL = "OpenBMB/MiniCPM-o-4_5"
DEFAULT_BITS = 8
DENSE_BITS = 0
DEFAULT_GROUP_SIZE = 64
QWEN_PREFIX = "llm."
TARGET_PREFIX = "model."


def precision_bits(config: Mapping[str, Any]) -> int:
    """Return the bundle precision encoded in a generated config."""

    value = config.get("quantization")
    if isinstance(value, Mapping) and "bits" in value:
        return int(value["bits"])
    value = config.get("quantization_config")
    if isinstance(value, Mapping) and "bits" in value:
        return int(value["bits"])
    return DEFAULT_BITS


def is_dense_bits(bits: int) -> bool:
    return int(bits) == DENSE_BITS


def precision_format(bits: int) -> str:
    return "minicpm-qwen3-mlx-bf16-v1" if is_dense_bits(bits) else "minicpm-qwen3-mlx-8bit-v1"


def precision_metadata(bits: int) -> dict[str, str]:
    return {
        "architecture": "qwen3",
        "format": "mlx",
        "quantized": "false" if is_dense_bits(bits) else "true",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run/verify MiniCPM-o Qwen3 extraction. Pass --write to "
            "materialize an MLX bundle (8-bit by default, or dense BF16 with --bits 0)."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually convert and write weights (default is dry-run).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing output bundle; never writes or quantizes.",
    )
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help=(
            "Validate an existing output weight file, then atomically write "
            "config/key_manifest/provenance sidecars without touching weights."
        ),
    )
    parser.add_argument(
        "--bits", type=int, choices=(DENSE_BITS, DEFAULT_BITS), default=DEFAULT_BITS,
        help="MLX quantization bits: 8 for the native packed bundle, 0 for dense BF16.",
    )
    parser.add_argument(
        "--group-size", type=int, choices=(32, 64), default=DEFAULT_GROUP_SIZE,
        help="MLX affine group size (pinned to 64).",
    )
    parser.add_argument(
        "--source-revision",
        default=PINNED_MODEL_REVISION,
        help=(
            "Model repository revision for source weights recorded in "
            f"provenance (default: {PINNED_MODEL_REVISION})."
        ),
    )
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Stable model id recorded in provenance (never records a local path).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def source_shards(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        raw = read_json(index)
        names = sorted(set(raw.get("weight_map", {}).values()))
        shards = [source / name for name in names]
    else:
        shards = sorted(source.glob("model-*.safetensors"))
        if not shards:
            single = source / "model.safetensors"
            if single.is_file():
                shards = [single]
    if not shards:
        raise RuntimeError(f"no safetensor shards found under {source}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise RuntimeError("source index references missing shards: " + ", ".join(missing))
    return shards


def source_key_map(source: Path) -> dict[str, str]:
    """Return source tensor name -> shard without loading tensor data."""

    index = source / "model.safetensors.index.json"
    if index.is_file():
        raw = read_json(index)
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"invalid weight_map in {index}")
        mapping = {str(key): str(value) for key, value in weight_map.items()}
    else:
        # A single safetensor file has no index.  Read names with safetensors
        # metadata only; values are not materialized here.
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for unindexed sources") from exc
        shard = source / "model.safetensors"
        if not shard.is_file():
            shards = sorted(source.glob("model-*.safetensors"))
            if len(shards) != 1:
                raise RuntimeError("an index is required for multiple source shards")
            shard = shards[0]
        with safe_open(str(shard), framework="np") as handle:
            mapping = {name: shard.name for name in handle.keys()}

    selected = {name: shard for name, shard in mapping.items() if name.startswith(QWEN_PREFIX)}
    if not selected:
        raise RuntimeError("source checkpoint has no llm.* Qwen3 tensors")
    unexpected = sorted(name for name in selected if not _is_qwen_source_key(name))
    if unexpected:
        raise RuntimeError("unexpected llm tensor(s): " + ", ".join(unexpected[:8]))
    return dict(sorted(selected.items()))


def _is_qwen_source_key(name: str) -> bool:
    target = name.removeprefix(QWEN_PREFIX)
    allowed = (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
        "lm_head.",
    )
    return target.startswith(allowed)


def target_key(source_name: str) -> str:
    if not source_name.startswith(QWEN_PREFIX):
        raise ValueError(source_name)
    return source_name.removeprefix(QWEN_PREFIX)


def load_config(
    source: Path,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    config = read_json(source / "config.json")
    required = (
        "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "max_position_embeddings", "rope_theta",
        "rms_norm_eps", "tie_word_embeddings",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError("source config is missing Qwen fields: " + ", ".join(missing))
    dimensions = {key: config[key] for key in required}
    if any(int(dimensions[key]) <= 0 for key in required if key not in {"rope_theta", "rms_norm_eps", "tie_word_embeddings"}):
        raise RuntimeError("source config contains non-positive Qwen dimensions")
    eos = eos_token_evidence(source)
    if int(bits) not in (DENSE_BITS, DEFAULT_BITS):
        raise RuntimeError(f"unsupported MLX precision bits: {bits}")
    output = {
        **dimensions,
        "format": precision_format(bits),
        "component_id": "minicpmo-4_5.llm-qwen3",
        "source_model": DEFAULT_SOURCE_MODEL,
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "rope_scaling": config.get("rope_scaling"),
        "quantization": {
            "bits": int(bits),
            "group_size": int(group_size),
            "mode": "none" if is_dense_bits(bits) else "affine",
        },
        "quantization_config": {
            "bits": int(bits),
            "group_size": int(group_size),
            "mode": "none" if is_dense_bits(bits) else "affine",
        },
    }
    # MiniCPMLLXConfig accepts this field and uses the official EOS id as a
    # diagnostic/default terminator for chat sessions.  Do not trust a stale
    # output config: the id is cross-checked against tokenizer assets below.
    output["eos_token_id"] = eos["token_id"]
    return output


def _normalise_token_id(value: Any, *, field: str) -> int:
    """Return one token id, rejecting ambiguous/list values."""

    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be an integer token id")
    if isinstance(value, int):
        if value < 0:
            raise RuntimeError(f"{field} must be non-negative")
        return value
    raise RuntimeError(f"{field} must be an integer token id")


def eos_token_evidence(source: Path) -> dict[str, Any]:
    """Cross-check MiniCPM's EOS id in config and tokenizer assets.

    MiniCPM's generation config contains two EOS ids (``im_end`` and BOS),
    whereas the causal-LM config and tokenizer's ``eos_token`` identify the
    actual turn terminator.  We require all three artifacts to agree instead
    of silently copying whichever value happens to be present.
    """

    config_path = source / "config.json"
    tokenizer_config_path = source / "tokenizer_config.json"
    added_tokens_path = source / "added_tokens.json"
    generation_config_path = source / "generation_config.json"
    required = (
        tokenizer_config_path,
        added_tokens_path,
        generation_config_path,
    )
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("source is missing EOS tokenizer evidence: " + ", ".join(missing))

    config = read_json(config_path)
    config_id = _normalise_token_id(config.get("eos_token_id"), field="config.eos_token_id")
    tokenizer_config = read_json(tokenizer_config_path)
    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError("tokenizer_config.eos_token must name the EOS token")
    added_tokens = read_json(added_tokens_path)
    token_id = _normalise_token_id(
        added_tokens.get(eos_token), field=f"added_tokens[{eos_token!r}]"
    )
    generation_config = read_json(generation_config_path)
    generation_ids = generation_config.get("eos_token_id")
    if isinstance(generation_ids, int) and not isinstance(generation_ids, bool):
        generation_ids = [generation_ids]
    if not isinstance(generation_ids, list) or not generation_ids:
        raise RuntimeError("generation_config.eos_token_id must be a non-empty list")
    generation_ids = [
        _normalise_token_id(value, field="generation_config.eos_token_id")
        for value in generation_ids
    ]
    if config_id != token_id or config_id not in generation_ids:
        raise RuntimeError(
            "EOS evidence mismatch: "
            f"config={config_id}, tokenizer={token_id}, generation={generation_ids}"
        )
    return {
        "token": eos_token,
        "token_id": config_id,
        "generation_token_ids": generation_ids,
        "files_sha256": {
            path.name: sha256(path)
            for path in (
                config_path,
                tokenizer_config_path,
                added_tokens_path,
                generation_config_path,
            )
        },
    }


def expected_raw_keys(source_names: Iterable[str]) -> list[str]:
    return [target_key(name) for name in source_names]


def expected_quantized_keys(
    raw_keys: Iterable[str],
    tie_word_embeddings: bool,
    bits: int = DEFAULT_BITS,
) -> list[str]:
    """Compute strict MLX keys produced by ``nn.quantize``.

    MLX quantizes every Linear/Embedding weight into ``weight/scales/biases``;
    norms remain weight-only.  Tied output embeddings do not carry lm_head.
    """

    result: list[str] = []
    if is_dense_bits(bits):
        return sorted(
            key for key in raw_keys
            if not (tie_word_embeddings and key.startswith("lm_head."))
        )
    quantized_suffixes = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    for key in raw_keys:
        if tie_word_embeddings and key.startswith("lm_head."):
            continue
        result.append(key)
        if key == "model.embed_tokens.weight" or key == "lm_head.weight" or key.endswith(quantized_suffixes):
            result.extend([key[:-len(".weight")] + ".scales", key[:-len(".weight")] + ".biases"])
    return sorted(result)


def source_tensor_specs(source: Path, source_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Read source tensor shapes/dtypes from safetensors headers only."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for strict bundle validation") from exc

    by_shard: dict[str, list[str]] = {}
    for name, shard in source_map.items():
        by_shard.setdefault(shard, []).append(name)
    specs: dict[str, dict[str, Any]] = {}
    for shard, names in by_shard.items():
        path = source / shard
        with safe_open(str(path), framework="np") as handle:
            available = set(handle.keys())
            missing = sorted(set(names) - available)
            if missing:
                raise RuntimeError(
                    f"source index/shard mismatch in {shard}: " + ", ".join(missing[:8])
                )
            for name in names:
                view = handle.get_slice(name)
                specs[name] = {
                    "shape": [int(dim) for dim in view.get_shape()],
                    "dtype": str(view.get_dtype()),
                }
    if set(specs) != set(source_map):
        raise RuntimeError("source tensor header set does not match the strict index manifest")
    return specs


def expected_target_specs(
    source_specs: dict[str, dict[str, Any]],
    tie_word_embeddings: bool,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, dict[str, Any]]:
    """Derive every MLX tensor key/shape/dtype from source headers.

    MLX packs four 8-bit values into one U32 word and emits one scale/bias
    pair per affine group.  This catches a bundle that happens to have the
    right key count but was packed with a different layout.
    """

    specs: dict[str, dict[str, Any]] = {}
    quantized_suffixes = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    for source_name in sorted(source_specs):
        raw_name = target_key(source_name)
        if tie_word_embeddings and raw_name.startswith("lm_head."):
            continue
        source_spec = source_specs[source_name]
        shape = source_spec["shape"]
        dtype = source_spec["dtype"]
        is_quantized = (
            not is_dense_bits(bits)
            and (
                raw_name == "model.embed_tokens.weight"
                or raw_name == "lm_head.weight"
                or raw_name.endswith(quantized_suffixes)
            )
        )
        if not is_quantized:
            specs[raw_name] = {"shape": shape, "dtype": dtype}
            continue
        if len(shape) != 2:
            raise RuntimeError(f"quantized source tensor must be rank-2: {source_name}")
        rows, cols = shape
        values_per_word = 32 // DEFAULT_BITS
        if cols <= 0 or cols % values_per_word:
            raise RuntimeError(
                f"quantized source columns are not divisible by {values_per_word}: "
                f"{source_name} shape={shape}"
            )
        packed_shape = [rows, cols // values_per_word]
        groups = (cols + group_size - 1) // group_size
        specs[raw_name] = {"shape": packed_shape, "dtype": "U32"}
        for suffix in ("scales", "biases"):
            specs[f"{raw_name[:-len('.weight')]}.{suffix}"] = {
                "shape": [rows, groups],
                "dtype": dtype,
            }
    return dict(sorted(specs.items()))


def _read_output_specs(output_weights: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for strict bundle validation") from exc
    if not output_weights.is_file():
        raise RuntimeError(f"output weights not found: {output_weights}")
    specs: dict[str, dict[str, Any]] = {}
    with safe_open(str(output_weights), framework="np") as handle:
        metadata = handle.metadata() or {}
        for name in handle.keys():
            view = handle.get_slice(name)
            specs[name] = {
                "shape": [int(dim) for dim in view.get_shape()],
                "dtype": str(view.get_dtype()),
            }
    return dict(sorted(specs.items())), dict(sorted((str(k), str(v)) for k, v in metadata.items()))


# Kept as a compatibility alias for callers that imported the original
# 8-bit metadata constant.  New manifests derive metadata from config bits.
EXPECTED_SAFETENSORS_METADATA = precision_metadata(DEFAULT_BITS)


def validate_output_weights(
    source: Path,
    output: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Fail closed unless output keys, shapes, dtypes and metadata match."""

    source_specs = source_tensor_specs(source, source_map)
    bits = precision_bits(output_config)
    expected = expected_target_specs(
        source_specs, bool(output_config["tie_word_embeddings"]), bits,
        group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)))
    actual, metadata = _read_output_specs(output / "model.safetensors")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "strict output key mismatch; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    for name, expected_spec in expected.items():
        actual_spec = actual[name]
        if actual_spec != expected_spec:
            raise RuntimeError(
                f"output tensor spec mismatch for {name}: "
                f"{actual_spec!r} != {expected_spec!r}"
            )
    expected_metadata = precision_metadata(bits)
    if metadata != expected_metadata:
        raise RuntimeError(
            "output safetensors metadata mismatch: "
            f"{metadata!r} != {expected_metadata!r}"
        )
    return actual, metadata


def manifest(
    source: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
    script_path: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    output: Path | None = None,
    source_revision: str = PINNED_MODEL_REVISION,
) -> dict[str, Any]:
    source_specs = source_tensor_specs(source, source_map)
    raw_keys = expected_raw_keys(source_map)
    bits = precision_bits(output_config)
    target_keys = expected_quantized_keys(
        raw_keys, bool(output_config["tie_word_embeddings"]), bits)
    target_specs = expected_target_specs(
        source_specs, bool(output_config["tie_word_embeddings"]), bits,
        group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)))
    if target_keys != sorted(target_specs):
        raise RuntimeError("internal target key derivation mismatch")
    shard_names = sorted(set(source_map.values()))
    source_shard_hashes = {
        name: sha256(source / name)
        for name in shard_names
    }
    evidence = eos_token_evidence(source)
    data = {
        "format": precision_format(bits),
        "source_namespace": QWEN_PREFIX,
        "source_tensor_count": len(source_map),
        "target_tensor_count": len(target_keys),
        "source_keys": raw_keys,
        "target_keys": target_keys,
        "source_to_target": [
            {"source": source_name, "target": target_key(source_name), "shard": shard}
            for source_name, shard in source_map.items()
        ],
        "strict": True,
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "none" if is_dense_bits(bits) else "affine",
        },
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "upstream_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "implementation_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "source_config_sha256": sha256(source / "config.json"),
        "source_index_sha256": sha256(source / "model.safetensors.index.json")
        if (source / "model.safetensors.index.json").is_file() else None,
        "source_shards_sha256": source_shard_hashes,
        "eos_evidence": evidence,
        "target_specs": target_specs,
        "safetensors_metadata": precision_metadata(bits),
        "converter_sha256": sha256(script_path),
    }
    if output is not None:
        weights = output / "model.safetensors"
        data["target_weights_sha256"] = sha256(weights)
    return data


def print_dry_run(
    source: Path,
    output: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
    script_path: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> None:
    source_revision = str(output_config.get("source_revision", PINNED_MODEL_REVISION))
    bits = precision_bits(output_config)
    info = {
        "action": "dry-run",
        "source": str(source),
        "output": str(output),
        "source_tensor_count": len(source_map),
        "target_tensor_count": len(expected_quantized_keys(
            expected_raw_keys(source_map),
            bool(output_config["tie_word_embeddings"]),
            bits,
        )),
        "format": precision_format(bits),
        "bits": bits,
        "group_size": group_size,
        "mode": "none" if is_dense_bits(bits) else "affine",
        "source_config_sha256": sha256(source / "config.json"),
        "converter_sha256": sha256(script_path),
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "upstream_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "implementation_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "next": "pass --write to materialize weights; pass --verify to audit an existing bundle",
    }
    print(json.dumps(info, indent=2, sort_keys=True))


def verify_bundle(
    source: Path,
    output: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
    script_path: Path,
    *,
    source_revision: str = PINNED_MODEL_REVISION,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> None:
    required = ("config.json", "model.safetensors", "key_manifest.json", "provenance.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError("output bundle is missing: " + ", ".join(missing))
    actual_config = read_json(output / "config.json")
    for key, expected in output_config.items():
        if actual_config.get(key) != expected:
            raise RuntimeError(f"output config mismatch for {key}: {actual.get(key)!r} != {expected!r}")
    validate_output_weights(source, output, source_map, output_config)
    expected = manifest(
        source,
        source_map,
        output_config,
        script_path,
        output=output,
        source_revision=source_revision,
        group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)),
    )
    actual_manifest = read_json(output / "key_manifest.json")
    for key in expected:
        if actual_manifest.get(key) != expected[key]:
            raise RuntimeError(f"manifest mismatch for {key}")
    actual_provenance = read_json(output / "provenance.json")
    expected_provenance = provenance(
        source,
        output,
        expected,
        script_path,
        group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)),
        source_revision=source_revision,
        source_model=source_model,
    )
    for key, value in expected_provenance.items():
        if actual_provenance.get(key) != value:
            raise RuntimeError(f"provenance mismatch for {key}")
    print(json.dumps({"action": "verify", "ok": True, "tensors": expected["target_tensor_count"], "output": str(output)}, indent=2))


def provenance(
    source: Path,
    output: Path,
    data_manifest: dict[str, Any],
    script_path: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    source_revision: str = PINNED_MODEL_REVISION,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> dict[str, Any]:
    """Build a deterministic provenance record for a validated bundle."""

    evidence = data_manifest["eos_evidence"]
    bits = int(data_manifest["quantization"]["bits"])
    return {
        "format": f"{precision_format(bits).removesuffix('-v1')}-provenance-v1",
        "model": "MiniCPM-o-4_5 Qwen3 language backbone",
        # A local filesystem path would make an otherwise reproducible bundle
        # machine-specific and leak the converter's workspace.  Keep only the
        # logical source id plus a harmless basename for operator diagnostics.
        "source_model": source_model,
        "source_basename": source.name,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "upstream_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "implementation_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "converter": script_path.name,
        "converter_sha256": sha256(script_path),
        "source_config_sha256": data_manifest["source_config_sha256"],
        "source_index_sha256": data_manifest["source_index_sha256"],
        "source_shards_sha256": data_manifest["source_shards_sha256"],
        "target_weights_sha256": data_manifest.get("target_weights_sha256"),
        "target_tensor_count": data_manifest["target_tensor_count"],
        "bits": bits,
        "group_size": group_size,
        "mode": "none" if is_dense_bits(bits) else "affine",
        "safetensors_metadata": precision_metadata(bits),
        "eos_evidence": evidence,
    }


def _atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    retain_backup: bool = True,
) -> None:
    """Write a sidecar durably, retaining a backup before replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and retain_backup:
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def conversion_manifest(
    source: Path,
    output: Path,
    *,
    source_model: str,
    source_revision: str,
    script_path: Path,
    tensor_count: int,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    """Build the portable operator-facing bundle manifest.

    Older conversion runs left a ``conversion_manifest.json`` containing
    absolute checkout/output paths.  Keep this sidecar pathless and derive all
    identities from the already audited hashes so it can be safely refreshed
    with ``--write-sidecars`` without touching model payloads.
    """

    def file_record(path: Path) -> dict[str, Any]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    files: dict[str, Any] = {}
    for name in ("model.safetensors", "config.json", "key_manifest.json", "provenance.json"):
        path = output / name
        if path.is_file():
            files[name] = file_record(path)
    tokenizer_assets = [
        name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "tokenization_minicpmo_fast.py",
        )
        if (output / name).is_file()
    ]
    return {
        "format": precision_format(bits),
        "component_id": "minicpmo-4_5.llm-qwen3",
        "source_model": source_model,
        "source_basename": source.name,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "upstream_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "implementation_commit": PINNED_DEMO_IMPLEMENTATION_COMMIT,
        "converter": {
            "path": script_path.name,
            "sha256": sha256(script_path),
        },
        "quantization": {
            "bits": int(bits),
            "group_size": group_size,
            "mode": "none" if is_dense_bits(bits) else "affine",
        },
        "tensor_count": tensor_count,
        "tokenizer_assets": tokenizer_assets,
        "files": files,
    }


def write_sidecars(
    source: Path,
    output: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
    script_path: Path,
    *,
    source_revision: str = PINNED_MODEL_REVISION,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> None:
    """Validate existing weights, then create/update only JSON sidecars."""

    # All validation happens before the first write.  A stale/old bundle can
    # therefore never be made to pass merely by fabricating a manifest.
    validate_output_weights(source, output, source_map, output_config)
    weights = output / "model.safetensors"
    data_manifest = manifest(
        source,
        source_map,
        output_config,
        script_path,
        output=output,
        source_revision=source_revision,
    )
    data_provenance = provenance(
        source,
        output,
        data_manifest,
        script_path,
        source_revision=source_revision,
        source_model=source_model,
    )

    # Preserve non-runtime annotations already present in config.json, while
    # replacing the converter-owned fields (including the audited EOS id).
    config_path = output / "config.json"
    existing_config = read_json(config_path) if config_path.is_file() else {}
    if not isinstance(existing_config, dict):
        raise RuntimeError("output config must be a JSON object")
    for key, expected in output_config.items():
        if key in existing_config and existing_config[key] != expected:
            raise RuntimeError(
                f"output config mismatch for {key}: "
                f"{existing_config[key]!r} != {expected!r}"
            )
    merged_config = {**existing_config, **output_config}
    _atomic_write_json(config_path, merged_config)
    _atomic_write_json(output / "key_manifest.json", data_manifest)
    _atomic_write_json(output / "provenance.json", data_provenance)
    _atomic_write_json(
        output / "conversion_manifest.json",
        conversion_manifest(
            source,
            output,
            source_model=source_model,
            source_revision=source_revision,
            script_path=script_path,
            tensor_count=data_manifest["target_tensor_count"],
            bits=precision_bits(output_config),
            group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)),
        ),
        retain_backup=False,
    )
    print(json.dumps({"action": "write-sidecars", "output": str(output), "tensors": data_manifest["target_tensor_count"]}, indent=2))


def write_bundle(
    source: Path,
    output: Path,
    source_map: dict[str, str],
    output_config: dict[str, Any],
    script_path: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    source_revision: str = PINNED_MODEL_REVISION,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> None:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3 import Model, ModelArgs
    except ImportError as exc:
        raise RuntimeError("--write requires mlx and mlx-lm in the active Python environment") from exc

    bits = precision_bits(output_config)
    if is_dense_bits(bits):
        existing_config_path = output / "config.json"
        if existing_config_path.is_file():
            existing_config = read_json(existing_config_path)
            existing_bits = precision_bits(existing_config)
            if not is_dense_bits(existing_bits):
                raise RuntimeError(
                    "refusing to overwrite an existing quantized bundle with dense BF16; "
                    "choose an independent --output directory"
                )
    output.mkdir(parents=True, exist_ok=True)
    args = ModelArgs(**{
        "model_type": "qwen3",
        "hidden_size": int(output_config["hidden_size"]),
        "num_hidden_layers": int(output_config["num_hidden_layers"]),
        "intermediate_size": int(output_config["intermediate_size"]),
        "num_attention_heads": int(output_config["num_attention_heads"]),
        "rms_norm_eps": float(output_config["rms_norm_eps"]),
        "vocab_size": int(output_config["vocab_size"]),
        "num_key_value_heads": int(output_config["num_key_value_heads"]),
        "max_position_embeddings": int(output_config["max_position_embeddings"]),
        "rope_theta": float(output_config["rope_theta"]),
        "head_dim": int(output_config["head_dim"]),
        "tie_word_embeddings": bool(output_config["tie_word_embeddings"]),
        "rope_scaling": output_config.get("rope_scaling"),
    })
    model = Model(args)
    tensors: dict[str, Any] = {}
    for shard in source_shards(source):
        loaded = mx.load(str(shard))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{shard} does not contain named tensors")
        for name, value in loaded.items():
            if name in source_map:
                tensors[target_key(name)] = value
    if set(tensors) != set(expected_raw_keys(source_map)):
        raise RuntimeError("source tensor extraction did not match the strict index manifest")
    model.load_weights(list(tensors.items()), strict=True)
    if not is_dense_bits(bits):
        nn.quantize(model, group_size=group_size, bits=bits, mode="affine")
    mx.eval(model.parameters())
    flat = dict(tree_flatten(model.parameters()))
    output_config = dict(output_config)
    output_config["format"] = precision_format(bits)
    output_config["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": "none" if is_dense_bits(bits) else "affine",
    }
    output_config["quantization_config"] = dict(output_config["quantization"])
    _atomic_write_json(output / "config.json", output_config)
    mx.save_safetensors(
        str(output / "model.safetensors"),
        flat,
        metadata=precision_metadata(bits),
    )
    data_manifest = manifest(
        source,
        source_map,
        output_config,
        script_path,
        output=output,
        source_revision=source_revision,
    )
    _atomic_write_json(output / "key_manifest.json", data_manifest)
    data_provenance = provenance(
        source,
        output,
        data_manifest,
        script_path,
        source_revision=source_revision,
        source_model=source_model,
    )
    _atomic_write_json(output / "provenance.json", data_provenance)
    _atomic_write_json(
        output / "conversion_manifest.json",
        conversion_manifest(
            source,
            output,
            source_model=source_model,
            source_revision=source_revision,
            script_path=script_path,
            tensor_count=data_manifest["target_tensor_count"],
            bits=bits,
            group_size=int(output_config.get("quantization", {}).get("group_size", DEFAULT_GROUP_SIZE)),
        ),
        retain_backup=False,
    )
    print(json.dumps({"action": "write", "output": str(output), "tensors": len(flat)}, indent=2))


def main() -> int:
    args = parse_args()
    selected_actions = sum(bool(value) for value in (args.write, args.verify, args.write_sidecars))
    if selected_actions > 1:
        raise SystemExit("--write, --verify and --write-sidecars are mutually exclusive")
    source = args.source_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "config.json").is_file():
        raise SystemExit(f"source config.json not found under {source}")
    source_map = source_key_map(source)
    output_config = load_config(source, bits=args.bits, group_size=args.group_size)
    output_config["source_model"] = args.source_model
    output_config["source_revision"] = args.source_revision
    output_config["revision_kind"] = "model_repository"
    output_config["implementation_commit"] = PINNED_DEMO_IMPLEMENTATION_COMMIT
    output_config["upstream_commit"] = PINNED_DEMO_IMPLEMENTATION_COMMIT
    script_path = Path(__file__).resolve()
    if args.verify:
        verify_bundle(
            source,
            output,
            source_map,
            output_config,
            script_path,
            source_revision=args.source_revision,
            source_model=args.source_model,
        )
    elif args.write_sidecars:
        write_sidecars(
            source,
            output,
            source_map,
            output_config,
            script_path,
            source_revision=args.source_revision,
            source_model=args.source_model,
        )
    elif args.write:
        write_bundle(
            source,
            output,
            source_map,
            output_config,
            script_path,
            group_size=args.group_size,
            source_revision=args.source_revision,
            source_model=args.source_model,
        )
    else:
        print_dry_run(source, output, source_map, output_config, script_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
