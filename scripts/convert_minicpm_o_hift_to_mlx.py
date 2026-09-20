#!/usr/bin/env python3
"""Convert MiniCPM-o 4.5's CosyVoice2 HiFT weights to MLX layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn.utils.parametrize import remove_parametrizations

from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator


# Keep model-weight and Demo implementation provenance independent.
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
# Compatibility alias for callers that imported the old name.  It denotes
# the Demo implementation, not the source model revision.
PINNED_UPSTREAM_COMMIT = DEMO_IMPLEMENTATION_COMMIT
DEFAULT_SOURCE_MODEL = "OpenBMB/MiniCPM-o-4_5"
MANIFEST_SCHEMA = "minicpm-o-token2wav-mlx/v1"
COMPONENT_ID = "minicpm-o-4_5.token2wav.hift"
OUTPUT_WEIGHTS = "hift.safetensors"

# This is the configuration consumed by ``MiniCPMHiFTGenerator`` in Swift.
# It is emitted as bundle metadata so the runtime need not silently depend on
# a source checkout's Python class defaults.
HIFT_CONFIGURATION: dict[str, Any] = {
    "input_channels": 80,
    "base_channels": 512,
    "harmonic_count": 8,
    "sample_rate": 24000,
    "sine_amplitude": 0.1,
    "noise_standard_deviation": 0.003,
    "voiced_threshold": 10,
    "upsample_rates": [8, 5, 3],
    "upsample_kernel_sizes": [16, 11, 7],
    "fft_size": 16,
    "fft_hop_length": 4,
    "residual_kernel_sizes": [3, 7, 11],
    "residual_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "source_residual_kernel_sizes": [7, 7, 11],
    "source_residual_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "leaky_relu_slope": 0.1,
    "audio_limit": 0.99,
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
            f"{MODEL_REPOSITORY_REVISION})"
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
    """Hash converted key/layout metadata without a second payload copy."""

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
    component: str,
    component_config: dict[str, Any],
) -> None:
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
            "revision_kind": "model_repository",
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
    component: str,
    component_manifest: dict[str, Any],
) -> None:
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
            "revision_kind": "model_repository",
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        }
    )
    components = manifest.get("components", {})
    if not isinstance(components, dict):
        components = {}
    components[component] = component_manifest
    manifest["components"] = components
    _write_json(manifest_path, manifest)


def normalized_name(name: str) -> str:
    if name.startswith("generator."):
        name = name.removeprefix("generator.")
    if name.startswith("f0_predictor.condnet."):
        pieces = name.split(".")
        pieces[2] = str(int(pieces[2]) // 2)
        name = ".".join(pieces)
    return name


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_revision = args.source_revision or MODEL_REPOSITORY_REVISION

    model = HiFTGenerator()
    raw = torch.load(source / "hift.pt", map_location="cpu", weights_only=True)
    raw = {name.removeprefix("generator."): value for name, value in raw.items()}
    model.load_state_dict(raw, strict=True)
    # This checkpoint uses the modern parametrizations-based weight_norm.
    # Materialize each effective weight instead of calling the legacy
    # remove_weight_norm helper bundled with StepAudio.
    for module in model.modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=True)
    model.eval()

    modules = dict(model.named_modules())
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        module_name, _, field = name.rpartition(".")
        module = modules.get(module_name)
        if field == "weight" and isinstance(module, nn.Conv1d):
            # PyTorch [out, in, kernel] -> MLX [out, kernel, in].
            tensor = tensor.permute(0, 2, 1)
        elif field == "weight" and isinstance(module, nn.ConvTranspose1d):
            # PyTorch [in, out, kernel] -> MLX [out, kernel, in].
            tensor = tensor.permute(1, 2, 0)
        converted[normalized_name(name)] = tensor.detach().contiguous()

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        output / "hift.safetensors",
        metadata={
            "format": "mlx",
            "model": "MiniCPM-o-4_5-CosyVoice2-HiFT",
            "component": COMPONENT_ID,
            "source_model": args.source_model,
            "source_revision": source_revision,
            "revision_kind": "model_repository",
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            "schema": MANIFEST_SCHEMA,
        },
    )
    total_bytes = sum(value.numel() * value.element_size() for value in converted.values())
    source_checkpoint = source / "hift.pt"
    component_config = dict(HIFT_CONFIGURATION)
    artifact = output / OUTPUT_WEIGHTS
    component_manifest: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component": "hift",
        "source_revision": source_revision,
        "revision_kind": "model_repository",
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
        component="hift",
        component_config=component_config,
    )
    _update_manifest(
        output,
        source_model=args.source_model,
        source_revision=source_revision,
        component="hift",
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
