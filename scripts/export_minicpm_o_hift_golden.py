#!/usr/bin/env python3
"""Export deterministic intermediate tensors for MiniCPM-o's HiFT vocoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator


UPSTREAM_REVISION = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
MANIFEST_SCHEMA = "minicpm-o-hift-golden/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="MiniCPM token2wav directory")
    parser.add_argument(
        "output", type=Path,
        help="Golden safetensors path (a side-car .manifest.json is written)",
    )
    parser.add_argument(
        "--model-source", type=Path, default=None,
        help=(
            "Original MiniCPM model directory containing config.json, the "
            "safetensors index, and every referenced shard. Defaults to the "
            "parent of <model>/assets/token2wav."
        ),
    )
    parser.add_argument(
        "--upstream-source", type=Path, default=Path("/tmp/MiniCPM-o-Demo"),
        help="Pinned checkout of the official MiniCPM-o implementation",
    )
    parser.add_argument(
        "--upstream-revision", default=UPSTREAM_REVISION,
        help="Expected official upstream git commit",
    )
    parser.add_argument(
        "--source-model-id", default=None,
        help="Stable model identifier recorded in metadata (never a local path)",
    )
    parser.add_argument(
        "--source-revision", default=None,
        help="Optional model repository revision (not the upstream code revision)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing artifact and manifest",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Hash a tree by canonical relative names and file bytes."""

    root = root.resolve()
    ignored = {".git", "__pycache__", ".build", ".venv"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ignored for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        files.append((relative.as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _assert_upstream_revision(source: Path, expected: str) -> str:
    source = source.expanduser().resolve()
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


def _safe_model_file(root: Path, candidate: str) -> Path:
    relative = Path(candidate)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe safetensors index path: {candidate!r}")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(
            f"missing safetensors shard referenced by index: {candidate}"
        )
    return path


def _model_provenance(
    source: Path, model_source: Path, model_id: str, revision: str | None,
) -> dict[str, Any]:
    """Return model and token2wav provenance without leaking local paths."""

    source = source.resolve()
    model_source = model_source.resolve()
    config = model_source / "config.json"
    index_path = model_source / "model.safetensors.index.json"
    for path, label in ((config, "model config"), (index_path, "model safetensors index")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    try:
        source_relative = source.relative_to(model_source).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "token2wav source must live below --model-source so provenance "
            "can be represented without an absolute path"
        ) from exc

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("invalid safetensors index: weight_map must be non-empty")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards = []
    for name in shard_names:
        path = _safe_model_file(model_source, name)
        shards.append({
            "path": path.relative_to(model_source).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })

    token2wav_files = []
    for relative in ("hift.pt", "flow.yaml"):
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing token2wav input: {path}")
        token2wav_files.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return {
        "model_id": model_id,
        "source_revision": revision,
        "revision_kind": "model_repository" if revision else "unspecified",
        "repo_relative_path": source_relative,
        "model_tree_sha256": _sha256_tree(model_source),
        "config_sha256": _sha256_file(config),
        "index_sha256": _sha256_file(index_path),
        "shards": shards,
        "token2wav_tree_sha256": _sha256_tree(source),
        "token2wav_files": token2wav_files,
    }


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order="C")
    values = tensor.float().reshape(-1)
    is_finite = (
        bool(torch.isfinite(values).all().item())
        if tensor.is_floating_point() else True
    )
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
        "finite": is_finite,
    }


def deterministic_mel(frames: int = 12) -> torch.Tensor:
    frequency = torch.arange(80, dtype=torch.float32).view(1, 80, 1)
    time = torch.arange(frames, dtype=torch.float32).view(1, 1, frames)
    return (
        0.22 * torch.sin(frequency * 0.071 + time * 0.31)
        + 0.08 * torch.cos(frequency * 0.037 - time * 0.17)
        - 0.5
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_name(output.stem + ".manifest.json")
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"refusing to overwrite existing fixture; pass --overwrite: {output}"
        )
    if not source.is_dir():
        raise FileNotFoundError(f"missing token2wav directory: {source}")
    model_source = (
        args.model_source.expanduser().resolve()
        if args.model_source is not None
        else source.parent.parent
    )
    config_path = model_source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "could not infer the original MiniCPM model directory; pass "
            f"--model-source (looked at {model_source})"
        )
    source_config = json.loads(config_path.read_text())
    model_id = args.source_model_id or str(
        source_config.get("_name_or_path") or model_source.name
    )
    if Path(model_id).is_absolute() or os.path.sep in model_id:
        model_id = model_source.name
    upstream_source = args.upstream_source.expanduser().resolve()
    upstream_revision = _assert_upstream_revision(
        upstream_source, args.upstream_revision
    )
    raw = torch.load(source / "hift.pt", map_location="cpu", weights_only=True)
    raw = {name.removeprefix("generator."): value for name, value in raw.items()}
    model = HiFTGenerator().eval()
    model.load_state_dict(raw, strict=True)

    # HiFT's SineGen2 uses the CPU RNG for both the initial harmonic phase and
    # the source noise.  Resetting the same explicit seed before each draw is
    # part of the oracle contract and is recorded in the side-car manifest.
    torch.manual_seed(args.seed)
    mel = deterministic_mel()
    tensors: dict[str, torch.Tensor] = {"mel": mel}
    with torch.inference_mode():
        f0 = model.f0_predictor(mel)
        f0_upsampled = model.f0_upsamp(f0[:, None]).transpose(1, 2)

        # SineGen2 draws one uniform [0, 1) cycle per harmonic at the first
        # sample, then pins the fundamental (harmonic zero) to zero.  Capture
        # that draw separately and reset the generator before invoking the
        # reference module so the subsequent randn buffer remains identical.
        initial_phase = torch.rand(
            f0_upsampled.shape[0], model.m_source.l_sin_gen.dim,
            device=f0_upsampled.device)
        initial_phase[:, 0] = 0
        torch.manual_seed(args.seed)
        sine_waves, uv, scaled_noise = model.m_source.l_sin_gen(f0_upsampled)
        noise_amplitude = uv * model.m_source.noise_std + (1 - uv) * model.m_source.sine_amp / 3
        standard_noise = scaled_noise / noise_amplitude
        source_signal = torch.tanh(model.m_source.l_linear(sine_waves)).transpose(1, 2)
        source_real, source_imaginary = model._stft(source_signal.squeeze(1))
        source_stft = torch.cat([source_real, source_imaginary], dim=1)

        tensors["f0"] = f0
        tensors["source"] = source_signal
        tensors["source_noise"] = standard_noise
        tensors["source_initial_phase"] = initial_phase
        tensors["source_stft"] = source_stft

        value = model.conv_pre(mel)
        tensors["conv_pre"] = value
        for stage in range(model.num_upsamples):
            value = F.leaky_relu(value, model.lrelu_slope)
            value = model.ups[stage](value)
            tensors[f"ups.{stage}"] = value
            if stage == model.num_upsamples - 1:
                value = model.reflection_pad(value)
            source_value = model.source_downs[stage](source_stft)
            source_value = model.source_resblocks[stage](source_value)
            value = value + source_value
            fused = None
            for kernel in range(model.num_kernels):
                current = model.resblocks[stage * model.num_kernels + kernel](value)
                fused = current if fused is None else fused + current
            value = fused / model.num_kernels
            tensors[f"stages.{stage}"] = value

        value = F.leaky_relu(value)
        logits = model.conv_post(value)
        tensors["logits"] = logits
        bins = model.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(logits[:, :bins])
        phase = torch.sin(logits[:, bins:])
        waveform = model._istft(magnitude, phase)
        waveform = torch.clamp(waveform, -model.audio_limit, model.audio_limit)
        tensors["waveform"] = waveform

    cpu_tensors = {
        name: value.float().cpu().contiguous() for name, value in tensors.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep local paths out of the safetensors user metadata.  The side-car
    # manifest contains stable IDs and content hashes instead.
    save_file(
        cpu_tensors,
        output,
        metadata={
            "schema": MANIFEST_SCHEMA,
            "oracle_backend": "official_pytorch",
            "upstream_revision": upstream_revision,
            "model_id": model_id,
            "seed": str(args.seed),
            "random_algorithm": "torch_cpu_default_generator",
        },
    )
    model_provenance = _model_provenance(
        source, model_source, model_id, args.source_revision
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "oracle_backend": "official_pytorch",
        "upstream": {
            "repository": "OpenBMB/MiniCPM-o-Demo",
            "revision": upstream_revision,
            "revision_kind": "git_commit",
            "tree_sha256": _sha256_tree(upstream_source),
        },
        "source": model_provenance,
        "converter": {
            "path": Path(__file__).name,
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "output_dtype": "float32",
            "seed": args.seed,
            "random_algorithm": "torch_cpu_default_generator",
            "mel_bins": int(mel.shape[1]),
            "mel_frames": int(mel.shape[2]),
            "sample_rate": int(model.sampling_rate),
            "fft_size": int(model.istft_params["n_fft"]),
            "hop_length": int(model.istft_params["hop_len"]),
            "harmonics": int(model.m_source.l_sin_gen.dim),
        },
        "artifact": {
            "path": output.name,
            "sha256": _sha256_file(output),
            "tensor_count": len(cpu_tensors),
            "tensors": {
                name: _tensor_metadata(value)
                for name, value in sorted(cpu_tensors.items())
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": output.name,
        "manifest": manifest_path.name,
        "tensor_count": len(cpu_tensors),
        "upstream_revision": upstream_revision,
        "model_id": model_id,
    }, indent=2))


if __name__ == "__main__":
    main()
