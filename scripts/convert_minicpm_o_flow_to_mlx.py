#!/usr/bin/env python3
"""Convert MiniCPM-o's CosyVoice2 flow checkpoint to MLX tensor layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


DEFAULT_SOURCE_MODEL = "OpenBMB/MiniCPM-o-4_5"
MANIFEST_SCHEMA = "minicpm-o-token2wav-mlx/v1"
COMPONENT_ID = "minicpm-o-4_5.token2wav.flow"
OUTPUT_WEIGHTS = "flow.safetensors"
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"

# The upstream flow.yaml is intentionally represented in the same compact
# snake_case vocabulary as the Swift configuration.  Serialising it beside
# the converted weights makes the bundle self-describing instead of relying
# on a hard-coded Swift default.
FLOW_CONFIGURATION: dict[str, Any] = {
    "token_vocabulary_size": 6561,
    "token_dimension": 512,
    "mel_dimension": 80,
    "speaker_embedding_dimension": 192,
    "encoder_blocks": 6,
    "encoder_upsample_blocks": 4,
    "encoder_heads": 8,
    "encoder_feed_forward_dimension": 2048,
    "pre_lookahead": 3,
    "upsample_stride": 2,
    "decoder_hidden_dimension": 512,
    "decoder_blocks": 16,
    "decoder_heads": 8,
    "decoder_head_dimension": 64,
    "decoder_mlp_ratio": 4,
    "decoder_input_channels": 320,
    "classifier_free_guidance": 0.7,
    "ode_steps": 10,
    "maximum_streaming_frames": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="MiniCPM token2wav directory")
    parser.add_argument("output", type=Path, help="Destination MLX directory")
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Stable model id recorded in the bundle metadata (never a local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=MODEL_REPOSITORY_REVISION,
        help=(
            "Model repository revision for source weights (default: "
            f"{MODEL_REPOSITORY_REVISION}; not an implementation/code revision)"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_set_sha256(tensors: dict[str, torch.Tensor]) -> str:
    """Hash names/shapes/dtypes without reading tensor payload bytes.

    The converted tensors are already resident for ``save_file``.  A
    structural hash therefore gives the loader a cheap key/layout identity
    while avoiding a second several-hundred-megabyte CPU copy.
    """

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        encoded = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": [int(value) for value in tensor.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_bundle_config(
    output: Path,
    *,
    source_model: str,
    source_revision: str,
    revision_kind: str,
    component: str,
    component_config: dict[str, Any],
) -> None:
    """Merge this component's config into a deterministic root config."""

    config_path = output / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid existing token2wav config: {config_path}") from exc
        if not isinstance(config, dict):
            raise RuntimeError(f"token2wav config must be a JSON object: {config_path}")
    else:
        config = {}
    config.update(
        {
            "format": MANIFEST_SCHEMA,
            "model_type": "minicpmo_token2wav_mlx",
            "source_model": source_model,
            "source_revision": source_revision,
            "revision_kind": revision_kind,
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        }
    )
    components = config.get("components", {})
    if not isinstance(components, dict):
        components = {}
    components[component] = component_config
    config["components"] = components
    _write_json(config_path, config)


def _update_manifest(
    output: Path,
    *,
    source_model: str,
    source_revision: str,
    revision_kind: str,
    component: str,
    component_manifest: dict[str, Any],
) -> None:
    """Merge a component record into ``manifest.json`` without local paths."""

    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid existing token2wav manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError(f"token2wav manifest must be a JSON object: {manifest_path}")
    else:
        manifest = {}
    manifest.update(
        {
            "schema": MANIFEST_SCHEMA,
            "model_type": "minicpmo_token2wav_mlx",
            "source_model": source_model,
            "source_revision": source_revision,
            "revision_kind": revision_kind,
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            # A basename is useful in diagnostics but cannot leak a machine
            # specific checkout path into a portable model bundle.
            "source_basename": source.name,
        }
    )
    components = manifest.get("components", {})
    if not isinstance(components, dict):
        components = {}
    components[component] = component_manifest
    manifest["components"] = components
    _write_json(manifest_path, manifest)


def is_conv1d(name: str) -> bool:
    return name.endswith(".weight") and (
        name.startswith("encoder.pre_lookahead_layer.conv")
        or name == "encoder.up_layer.conv.weight"
        or (".conv.block." in name and name.endswith(".weight")
            and (".block.1.weight" in name or ".block.6.weight" in name))
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_revision = args.source_revision or MODEL_REPOSITORY_REVISION
    revision_kind = "model_repository"
    source_checkpoint = source / "flow.pt"
    source_yaml = source / "flow.yaml"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"missing flow checkpoint: {source_checkpoint}")
    if not source_yaml.is_file():
        raise FileNotFoundError(f"missing flow config: {source_yaml}")
    checkpoint = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in checkpoint.items():
        if is_conv1d(name):
            tensor = tensor.permute(0, 2, 1)
        converted[name] = tensor.contiguous()

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        output / "flow.safetensors",
        metadata={
            "format": "mlx",
            "model": "MiniCPM-o-4_5-CosyVoice2-Flow",
            "component": COMPONENT_ID,
            "source_model": args.source_model,
            "source_revision": source_revision,
            "revision_kind": revision_kind,
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            "schema": MANIFEST_SCHEMA,
        },
    )
    total_bytes = sum(value.numel() * value.element_size() for value in converted.values())
    component_config = dict(FLOW_CONFIGURATION)
    artifact = output / OUTPUT_WEIGHTS
    component_manifest: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component": "flow",
        "source_revision": source_revision,
        "revision_kind": revision_kind,
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "weights": {
            "path": OUTPUT_WEIGHTS,
            "kind": "single_file",
            "size": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "source": {
            "checkpoint": {
                "path": source_checkpoint.name,
                "size": source_checkpoint.stat().st_size,
                "sha256": sha256_file(source_checkpoint),
            },
            "config": {
                "path": source_yaml.name,
                "size": source_yaml.stat().st_size,
                "sha256": sha256_file(source_yaml),
            },
        },
        "converter": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "tensor_count": len(converted),
        "tensor_set_sha256": tensor_set_sha256(converted),
        "config": component_config,
    }
    _update_bundle_config(
        output,
        source_model=args.source_model,
        source_revision=source_revision,
        revision_kind=revision_kind,
        component="flow",
        component_config=component_config,
    )
    _update_manifest(
        output,
        source_model=args.source_model,
        source_revision=source_revision,
        revision_kind=revision_kind,
        component="flow",
        component_manifest=component_manifest,
    )
    print(json.dumps({
        "tensors": len(converted),
        "bytes": total_bytes,
        "output": str(output),
        "manifest": "manifest.json",
    }, indent=2))


if __name__ == "__main__":
    main()
