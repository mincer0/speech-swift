#!/usr/bin/env python3
"""Convert MiniCPM-o's audio weights to the native MLX bundle format.

The conversion is deliberately boring: it renames the official audio
submodules and transposes the two Conv1d kernels.  All source/model
provenance is emitted in ``manifest.json`` so a bundle can never be mistaken
for a conversion from an unrelated model checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


MANIFEST_SCHEMA = "minicpm-o-audio-mlx/v1"
SOURCE_PREFIXES = ("apm.", "audio_projection_layer.")
OUTPUT_WEIGHTS = "audio_encoder.safetensors"
DEFAULT_SOURCE_MODEL_ID = "OpenBMB/MiniCPM-o-4_5"
COMPONENT_ID = "minicpm-o-4_5.audio-encoder"
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Original MiniCPM-o model directory")
    parser.add_argument("output", type=Path, help="Destination audio-only MLX bundle directory")
    parser.add_argument(
        "--source-model-id",
        default=DEFAULT_SOURCE_MODEL_ID,
        help="Stable model identifier recorded in the manifest (no local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=MODEL_REPOSITORY_REVISION,
        help=(
            "Model repository revision for source weights (default: "
            f"{MODEL_REPOSITORY_REVISION}; this is not a code revision)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing bundle's generated files",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
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


def resolve_index_path(source: Path, candidate: str) -> Path:
    relative = Path(candidate)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe safetensors index path: {candidate!r}")
    path = (source / relative).resolve()
    if source.resolve() not in path.parents and path != source.resolve():
        raise RuntimeError(f"safetensors index path escapes model directory: {candidate!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing safetensors shard: {candidate}")
    return path


def source_provenance(source: Path, model_id: str, revision: str | None) -> dict[str, Any]:
    revision_value = revision or "unspecified"
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
    shards = []
    for name in sorted({str(value) for value in weight_map.values()}):
        path = resolve_index_path(source, name)
        shards.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "model_id": model_id,
        "source_revision": revision_value,
        "revision_kind": "model_repository" if revision else "unspecified",
        "tree_sha256": sha256_tree(source),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shards": shards,
    }


def load_selected(source: Path) -> dict:
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
    tensors = {}
    for shard in sorted(set(selected.values())):
        shard_path = resolve_index_path(source, shard)
        names = [name for name, candidate in selected.items() if candidate == shard]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in names:
                if name not in available:
                    raise RuntimeError(f"index maps missing tensor {name!r} in {shard!r}")
                tensors[name] = handle.get_tensor(name)
    return tensors


def convert_name(name: str) -> str:
    if name.startswith("apm."):
        return "encoder." + name.removeprefix("apm.")
    if name.startswith("audio_projection_layer."):
        return "projection." + name.removeprefix("audio_projection_layer.")
    raise ValueError(f"unexpected tensor: {name}")


def tensor_metadata(tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order="C")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": tensor.numel(),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_set_hash(tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        raw_name = name.encode("utf-8")
        raw_dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(raw_name).to_bytes(8, "big"))
        digest.update(raw_name)
        digest.update(len(raw_dtype).to_bytes(8, "big"))
        digest.update(raw_dtype)
        digest.update(len(tensor.shape).to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    weights_path = output / OUTPUT_WEIGHTS
    config_path = output / "config.json"
    manifest_path = output / "manifest.json"
    if output.exists() and not args.overwrite:
        existing = [path for path in (weights_path, config_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite existing MLX bundle; pass --overwrite: {output}"
            )
    source_config = json.loads((source / "config.json").read_text())
    model_id = args.source_model_id or str(
        source_config.get("_name_or_path") or source.name
    )
    if Path(model_id).is_absolute():
        model_id = source.name
    source_revision = args.source_revision or MODEL_REPOSITORY_REVISION
    revision_kind = "model_repository"
    audio = source_config["audio_config"]
    configuration = {
        "format": MANIFEST_SCHEMA,
        "model_type": "minicpmo_audio_mlx",
        "component_id": COMPONENT_ID,
        "source_model": model_id,
        "source_revision": source_revision,
        "revision_kind": revision_kind,
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "hiddenSize": audio["d_model"],
        "intermediateSize": audio["encoder_ffn_dim"],
        "attentionHeads": audio["encoder_attention_heads"],
        "layers": audio["encoder_layers"],
        "melBins": audio["num_mel_bins"],
        "maximumSourcePositions": audio["max_source_positions"],
        "layerNormEpsilon": audio.get("layer_norm_eps", 1e-5),
        "poolStep": source_config["audio_pool_step"],
        "projectionSize": source_config["hidden_size"],
    }

    converted = {}
    for source_name, tensor in load_selected(source).items():
        target_name = convert_name(source_name)
        if target_name in {"encoder.conv1.weight", "encoder.conv2.weight"}:
            # PyTorch [out, in, kernel] -> MLX Conv1d [out, kernel, in].
            tensor = tensor.permute(0, 2, 1)
        if target_name in converted:
            raise RuntimeError(f"duplicate converted tensor name: {target_name}")
        converted[target_name] = tensor.contiguous()

    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
    save_file(
        converted,
        weights_path,
        metadata={
            "format": "mlx",
            "model": "MiniCPM-o-4_5-audio",
            "component": COMPONENT_ID,
            "schema": MANIFEST_SCHEMA,
            "model_id": model_id,
            "source_revision": source_revision,
            "revision_kind": revision_kind,
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            "output_dtype": str(next(iter(converted.values())).dtype).replace("torch.", ""),
        },
    )
    provenance = source_provenance(source, model_id, source_revision)
    converter_path = Path(__file__).resolve()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "component_id": COMPONENT_ID,
        "component": "audio_encoder",
        "model_id": model_id,
        "source_revision": source_revision,
        "revision_kind": revision_kind,
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "source": provenance,
        "converter": {
            "path": converter_path.name,
            "sha256": sha256_file(converter_path),
        },
        "dtype": {
            "source": str(next(iter(converted.values())).dtype).replace("torch.", ""),
            "output": str(next(iter(converted.values())).dtype).replace("torch.", ""),
        },
        "config": configuration,
        "artifact": {
            "weights": {
                "path": OUTPUT_WEIGHTS,
                "size": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            },
            "config": {
                "path": "config.json",
                "size": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "tensor_count": len(converted),
            "tensor_set_sha256": tensor_set_hash(converted),
            "tensors": {
                name: tensor_metadata(tensor)
                for name, tensor in sorted(converted.items())
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total_bytes = sum(t.numel() * t.element_size() for t in converted.values())
    print(
        json.dumps(
            {
                "tensors": len(converted),
                "bytes": total_bytes,
                "output": output.name,
                "manifest": manifest_path.name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
