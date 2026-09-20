#!/usr/bin/env python3
"""Export a first-window PyTorch trace for MiniCPM-o's audio encoder.

This is a diagnostic companion to ``export_minicpm_o_audio_golden.py``.  It
executes the pinned upstream encoder one stage at a time and records the
intermediate tensors that identify the first numerical divergence in the
native implementation.  Tensors remain BF16 throughout the artifact: hashes
are computed over the exact uint16 BF16 storage bits, not a float32 rewrite.

The script is intentionally offline and opt-in.  It does not alter the v2
golden oracle and refuses to overwrite an existing trace unless ``--overwrite``
is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


UPSTREAM_REVISION = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
MANIFEST_SCHEMA = "minicpm-o-audio-layer-trace/v2"
REGULAR_CHUNK_SAMPLES = 16_480
TRACE_CHUNK_FRAMES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Original MiniCPM-o model directory")
    parser.add_argument(
        "output",
        type=Path,
        help="Output trace.safetensors path (manifest is written beside it)",
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
        help="Expected upstream git commit",
    )
    parser.add_argument(
        "--source-model-id",
        default=None,
        help="Stable model identifier recorded in the manifest",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Optional model repository revision (not a code revision)",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing trace and manifest",
    )
    return parser.parse_args()


def _load_golden_helpers():
    """Reuse the pinned exporter helpers without importing model code locally."""

    path = Path(__file__).resolve().with_name("export_minicpm_o_audio_golden.py")
    spec = importlib.util.spec_from_file_location("_minicpm_audio_golden_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import exporter helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
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
    relative = Path(candidate)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe safetensors index path: {candidate!r}")
    path = (source / relative).resolve()
    if source.resolve() not in path.parents and path != source.resolve():
        raise RuntimeError(f"safetensors index path escapes model directory: {candidate!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing safetensors shard: {candidate}")
    return path


def _source_provenance(source: Path, model_id: str, revision: str | None) -> dict[str, Any]:
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError("model.safetensors.index.json is required")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("invalid safetensors index: weight_map must be non-empty")
    shards = []
    for name in sorted({str(value) for value in weight_map.values()}):
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


def _load_selected(source: Path) -> dict[str, torch.Tensor]:
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("invalid safetensors index: missing weight_map")
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith(("apm.", "audio_projection_layer."))
    }
    if not selected:
        raise RuntimeError("model index contains no MiniCPM audio tensors")
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(selected.values())):
        shard_path = _relative_source_file(source, shard)
        names = [name for name, candidate in selected.items() if candidate == shard]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in names:
                if name not in available:
                    raise RuntimeError(f"index maps missing tensor {name!r} in {shard!r}")
                tensors[name] = handle.get_tensor(name)
    return tensors


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
        raise RuntimeError(f"upstream revision mismatch: expected {expected}, got {actual}")
    return actual


def _import_upstream(upstream_source: Path, expected_revision: str):
    """Import only the official processor/encoder implementation."""

    upstream_source = upstream_source.expanduser().resolve()
    actual_revision = _assert_upstream_revision(upstream_source, expected_revision)
    # The pinned exporter has a compatibility shim for old torchvision builds.
    helpers = _load_golden_helpers()
    helpers.ensure_torchvision_compat()
    package = "_minicpmo_audio_trace_upstream"
    import types

    root = types.ModuleType(package)
    root.__path__ = [str(upstream_source / "MiniCPMO45")]
    sys.modules[package] = root
    module_root = upstream_source / "MiniCPMO45"
    helpers.load_local_module(
        package,
        "configuration_minicpmo",
        module_root / "configuration_minicpmo.py",
    )
    modeling = helpers.load_local_module(
        package, "modeling_minicpmo", module_root / "modeling_minicpmo.py"
    )
    processing = helpers.load_local_module(
        package, "processing_minicpmo", module_root / "processing_minicpmo.py"
    )
    return modeling, processing, actual_revision


def _bf16_cpu(value: torch.Tensor) -> torch.Tensor:
    """Detach to CPU while retaining the source tensor's BF16 storage bits."""

    return value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _raw_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        # NumPy has no portable bfloat16 dtype; uint16 exposes the exact bits.
        return value.view(torch.uint16).numpy().tobytes(order="C")
    return value.numpy().tobytes(order="C")


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    raw = _raw_bytes(value)
    values = value.float().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "numel": value.numel(),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "max_abs": float(values.abs().max().item()) if values.numel() else 0.0,
        "mean_abs": float(values.abs().mean().item()) if values.numel() else 0.0,
        "finite": bool(torch.isfinite(values).all().item()),
    }


def _validate_artifact(path: Path, manifest: dict[str, Any]) -> None:
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != path.name:
        raise RuntimeError("trace manifest artifact metadata is incomplete")
    tensors = artifact.get("tensors")
    if not isinstance(tensors, dict) or artifact.get("tensor_count") != len(tensors):
        raise RuntimeError("trace manifest tensor_count mismatch")
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        if names != set(tensors):
            raise RuntimeError("trace artifact tensor set does not match manifest")
        for name in sorted(names):
            actual = handle.get_tensor(name).contiguous()
            expected = tensors[name]
            if str(actual.dtype).replace("torch.", "") != expected["dtype"]:
                raise RuntimeError(f"trace tensor {name} dtype mismatch")
            if list(actual.shape) != expected["shape"]:
                raise RuntimeError(f"trace tensor {name} shape mismatch")
            if hashlib.sha256(_raw_bytes(actual)).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"trace tensor {name} hash mismatch")


def _trace_first_chunk(
    encoder: torch.nn.Module,
    modeling: Any,
    processing: Any,
    source: Path,
    *,
    device: torch.device,
    audio_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    processor = processing.MiniCPMAAudioProcessor.from_pretrained(source)
    processor.set_spac_log_norm(log_floor_db=-10)
    waveform = _load_golden_helpers().deterministic_waveform(count=REGULAR_CHUNK_SAMPLES)
    features = processor(
        waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    ).input_features
    # The first 16,480-sample window emits exactly 100 stable mel frames.
    features = features[:, :, :TRACE_CHUNK_FRAMES].to(device=device, dtype=audio_dtype)
    if features.shape != (1, 80, TRACE_CHUNK_FRAMES):
        raise RuntimeError(f"unexpected first-window feature shape: {tuple(features.shape)}")

    tensors: dict[str, torch.Tensor] = {
        "stream_features_nct": _bf16_cpu(features),
    }
    with torch.inference_mode():
        input_features = features.to(dtype=encoder.conv1.weight.dtype, device=encoder.conv1.weight.device)
        conv1 = encoder.conv1(input_features)
        gelu1 = torch.nn.functional.gelu(conv1)
        conv2 = encoder.conv2(gelu1)
        gelu2 = torch.nn.functional.gelu(conv2)
        tensors["conv1_output"] = _bf16_cpu(conv1)
        tensors["conv1_gelu"] = _bf16_cpu(gelu1)
        tensors["conv2_output"] = _bf16_cpu(conv2)
        tensors["conv2_gelu"] = _bf16_cpu(gelu2)

        hidden = gelu2.permute(0, 2, 1)
        positions = encoder.embed_positions.weight[: hidden.shape[1], :]
        tensors["position_embeddings"] = _bf16_cpu(positions.unsqueeze(0))
        hidden = hidden + positions
        tensors["position_added"] = _bf16_cpu(hidden)

        query_length = hidden.shape[1]
        # This is the exact all-zero mask supplied by the pinned streaming
        # exporter.  It broadcasts over attention heads inside WhisperAttention.
        attention_mask = torch.zeros(
            (1, 1, query_length, query_length),
            dtype=audio_dtype,
            device=device,
        )
        for index, layer in enumerate(encoder.layers):
            prefix = f"layer_{index:02d}_"
            pre_attention_norm = layer.self_attn_layer_norm(hidden)
            tensors[prefix + "pre_attention_norm"] = _bf16_cpu(pre_attention_norm)
            residual = hidden
            attention = layer.self_attn
            batch, target_length, _ = pre_attention_norm.shape
            query = attention._shape(
                attention.q_proj(pre_attention_norm) * attention.scaling,
                target_length,
                batch,
            )
            key = attention._shape(
                attention.k_proj(pre_attention_norm), -1, batch
            )
            value = attention._shape(
                attention.v_proj(pre_attention_norm), -1, batch
            )
            scores = torch.matmul(query, key.transpose(2, 3))
            scores = scores + attention_mask[:, :, :, : key.shape[-2]]
            probabilities = torch.nn.functional.softmax(scores, dim=-1)
            context_heads = torch.matmul(probabilities, value)
            context_merged = context_heads.transpose(1, 2).reshape(
                batch, target_length, attention.embed_dim
            )
            attention_output = attention.out_proj(context_merged)
            tensors[prefix + "query"] = _bf16_cpu(query)
            tensors[prefix + "key"] = _bf16_cpu(key)
            tensors[prefix + "value"] = _bf16_cpu(value)
            tensors[prefix + "attention_scores"] = _bf16_cpu(scores)
            tensors[prefix + "attention_probabilities"] = _bf16_cpu(
                probabilities
            )
            tensors[prefix + "attention_context_heads"] = _bf16_cpu(
                context_heads
            )
            tensors[prefix + "attention_context_merged"] = _bf16_cpu(
                context_merged
            )
            tensors[prefix + "attention_output"] = _bf16_cpu(attention_output)
            hidden = residual + attention_output
            tensors[prefix + "post_attention_residual"] = _bf16_cpu(hidden)

            pre_ffn_norm = layer.final_layer_norm(hidden)
            tensors[prefix + "pre_ffn_norm"] = _bf16_cpu(pre_ffn_norm)
            fc1_output = layer.fc1(pre_ffn_norm)
            tensors[prefix + "fc1_output"] = _bf16_cpu(fc1_output)
            ffn_activation = layer.activation_fn(fc1_output)
            tensors[prefix + "ffn_activation"] = _bf16_cpu(ffn_activation)
            ffn_output = layer.fc2(ffn_activation)
            tensors[prefix + "ffn_output"] = _bf16_cpu(ffn_output)
            hidden = hidden + ffn_output
            tensors[prefix + "layer_output"] = _bf16_cpu(hidden)

        tensors["encoder_output"] = _bf16_cpu(encoder.layer_norm(hidden))
    return tensors


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_name(output.stem + ".manifest.json")
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing trace: {output}")

    source_config = json.loads((source / "config.json").read_text())
    model_id = args.source_model_id or str(source_config.get("_name_or_path") or source.name)
    if Path(model_id).is_absolute():
        model_id = source.name
    modeling, processing, upstream_revision = _import_upstream(
        args.upstream_source, args.upstream_revision
    )
    audio_config = modeling.WhisperConfig(**source_config["audio_config"])
    encoder = modeling.MiniCPMWhisperEncoder(audio_config)
    selected = _load_selected(source)
    source_dtypes = {tensor.dtype for tensor in selected.values()}
    if len(source_dtypes) != 1:
        raise RuntimeError(f"audio tensors have mixed dtypes: {sorted(map(str, source_dtypes))}")
    encoder.load_state_dict(
        {name.removeprefix("apm."): value for name, value in selected.items() if name.startswith("apm.")},
        strict=True,
    )
    device = torch.device(args.device)
    source_dtype = next(iter(selected.values())).dtype
    encoder = encoder.to(device=device, dtype=source_dtype).eval()
    tensors = _trace_first_chunk(
        encoder,
        modeling,
        processing,
        source,
        device=device,
        audio_dtype=encoder.conv1.weight.dtype,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            "format": MANIFEST_SCHEMA,
            "oracle_backend": "official_pytorch",
            "upstream_revision": upstream_revision,
            "model_id": model_id,
            "dtype": "bfloat16",
        },
    )
    provenance = _source_provenance(source, model_id, args.source_revision)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "oracle_backend": "official_pytorch",
        "upstream": {
            "repository": "OpenBMB/MiniCPM-o-Demo",
            "revision": upstream_revision,
            "revision_kind": "git_commit",
        },
        "source": provenance,
        "runtime": {
            "device": str(device),
            "source_dtype": str(next(iter(selected.values())).dtype).replace("torch.", ""),
            "trace_dtype": "bfloat16",
            "sample_rate": 16_000,
            "mel_bins": 80,
            "first_chunk_samples": REGULAR_CHUNK_SAMPLES,
            "first_chunk_feature_frames": TRACE_CHUNK_FRAMES,
            "attention_mask_shape": [1, 1, 50, 50],
            "attention_mask": "all_zero_additive_bfloat16_broadcast_over_heads",
            "layer_trace_mapping": {
                "pre_attention_norm": "self_attn_layer_norm(hidden)",
                "attention_output": "self_attn(...)[0]",
                "post_attention_residual": "hidden + attention_output",
                "pre_ffn_norm": "final_layer_norm(hidden)",
                "ffn_activation": "activation_fn(fc1(pre_ffn_norm))",
                "ffn_output": "fc2(ffn_activation)",
                "layer_output": "post_attention_residual + ffn_output",
            },
        },
        "artifact": {
            "path": output.name,
            "sha256": _sha256_file(output),
            "tensor_count": len(tensors),
            "tensors": {
                name: _tensor_metadata(tensor) for name, tensor in sorted(tensors.items())
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _validate_artifact(output, manifest)
    print(
        json.dumps(
            {
                "output": output.name,
                "manifest": manifest_path.name,
                "tensor_count": len(tensors),
                "upstream_revision": upstream_revision,
                "dtype": "bfloat16",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
