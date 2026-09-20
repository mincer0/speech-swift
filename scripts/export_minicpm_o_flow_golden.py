#!/usr/bin/env python3
"""Export deterministic encoder, DiT, and CFM tensors for MLX parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import save_file


UPSTREAM_REVISION = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
MANIFEST_SCHEMA = "minicpm-o-flow-golden/v1"


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
    parser.add_argument("--seed", type=int, default=2026)
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
    """Hash a tree by canonical relative names and file bytes.

    VCS and interpreter caches are deliberately excluded.  They are not model
    inputs and would make an otherwise identical golden change after importing
    the pinned Python implementation.
    """

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
    for relative in ("flow.yaml", "flow.pt"):
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


def register_aliases() -> None:
    import stepaudio2.cosyvoice2.flow.decoder_dit as decoder_dit
    import stepaudio2.cosyvoice2.flow.flow as flow
    import stepaudio2.cosyvoice2.flow.flow_matching as flow_matching
    import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as encoder

    root = types.ModuleType("cosyvoice2")
    flow_package = types.ModuleType("cosyvoice2.flow")
    transformer_package = types.ModuleType("cosyvoice2.transformer")
    flow_package.flow = flow
    flow_package.flow_matching = flow_matching
    flow_package.decoder_dit = decoder_dit
    transformer_package.upsample_encoder_v2 = encoder
    root.flow = flow_package
    root.transformer = transformer_package
    sys.modules.update({
        "cosyvoice2": root,
        "cosyvoice2.flow": flow_package,
        "cosyvoice2.flow.flow": flow,
        "cosyvoice2.flow.flow_matching": flow_matching,
        "cosyvoice2.flow.decoder_dit": decoder_dit,
        "cosyvoice2.transformer": transformer_package,
        "cosyvoice2.transformer.upsample_encoder_v2": encoder,
    })


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
    # A local checkout path is never a stable model identifier.
    if Path(model_id).is_absolute() or os.path.sep in model_id:
        model_id = model_source.name
    upstream_source = args.upstream_source.expanduser().resolve()
    upstream_revision = _assert_upstream_revision(
        upstream_source, args.upstream_revision
    )
    register_aliases()
    with (source / "flow.yaml").open() as handle:
        model = load_hyperpyyaml(handle)["flow"]
    model.load_state_dict(torch.load(
        source / "flow.pt", map_location="cpu", weights_only=True, mmap=True), strict=True)
    model.eval()

    # The oracle intentionally uses one explicit CPU RNG stream.  Every
    # stochastic input is captured in the fixture, so MLX never has to guess
    # PyTorch's random state or device-specific generator semantics.
    torch.manual_seed(args.seed)
    tensors: dict[str, torch.Tensor] = {}
    tokens = torch.tensor(
        [[4218, 140, 991, 77, 3021, 9, 4218, 88]], dtype=torch.int32
    )
    token_embeddings = model.input_embedding(tokens)
    lengths = torch.tensor([tokens.shape[1]], dtype=torch.long)
    with torch.inference_mode():
        encoder_output, _ = model.encoder(token_embeddings, lengths)
        projected = model.encoder_proj(encoder_output)
    tensors.update({
        "tokens": tokens,
        "token_embeddings": token_embeddings,
        "encoder_output": encoder_output,
        "encoder_projected": projected,
    })

    # One exact DiT pass, with every block captured to localize parity issues.
    length = projected.shape[1]
    x = torch.randn(2, 80, length)
    mu = torch.randn(2, 80, length) * 0.2
    speakers = torch.randn(2, 80) * 0.1
    condition = torch.randn(2, 80, length) * 0.05
    time = torch.tensor([0.2, 0.7])
    mask = torch.ones(2, 1, length, dtype=torch.float32)
    tensors.update({
        "dit_x": x, "dit_mu": mu, "dit_speakers": speakers,
        "dit_condition": condition, "dit_time": time, "dit_mask": mask,
    })
    hooks = []
    for index, block in enumerate(model.decoder.estimator.blocks):
        hooks.append(block.register_forward_hook(
            lambda _module, _inputs, result, index=index:
                tensors.__setitem__(f"dit_blocks.{index}", result)))
    with torch.inference_mode():
        dit_output = model.decoder.estimator(
            x, mask, mu, time, speakers, condition)
    for hook in hooks:
        hook.remove()
    tensors["dit_output"] = dit_output

    # Two-step CFM keeps the fixture fast while validating CFG and cosine Euler.
    cfm_mu = projected.transpose(1, 2)
    cfm_speaker = model.spk_embed_affine_layer(torch.nn.functional.normalize(
        torch.linspace(-1, 1, 192).view(1, 192), dim=1))
    cfm_condition = torch.zeros_like(cfm_mu)
    cfm_mask = torch.ones(1, 1, cfm_mu.shape[2])
    initial_noise = torch.randn_like(cfm_mu)
    model.decoder.rand_noise[:, :, :cfm_mu.shape[2]] = initial_noise
    with torch.inference_mode():
        cfm_output = model.decoder(
            cfm_mu, cfm_mask, cfm_speaker, cfm_condition, n_timesteps=2)
    tensors.update({
        "cfm_mu": cfm_mu,
        "cfm_speaker": cfm_speaker,
        "cfm_condition": cfm_condition,
        "cfm_mask": cfm_mask,
        "cfm_noise": initial_noise,
        "cfm_output_2_steps": cfm_output,
    })

    # Encoder streaming parity over two chunks (5 stable tokens -> 10 mel frames).
    stream_tokens_1 = torch.tensor(
        [[4218, 140, 991, 77, 3021, 9, 4218, 88]], dtype=torch.int32
    )
    stream_tokens_2 = torch.tensor(
        [[56, 780, 4, 311, 902, 71, 66, 4218]], dtype=torch.int32
    )
    with torch.inference_mode():
        stream_encoder_1, stream_cnn_1, stream_att_1 = model.encoder.forward_chunk(
            model.input_embedding(stream_tokens_1), last_chunk=False)
        stream_encoder_2, stream_cnn_2, stream_att_2 = model.encoder.forward_chunk(
            model.input_embedding(stream_tokens_2), last_chunk=False,
            cnn_cache=stream_cnn_1, att_cache=stream_att_1)
    first_length_1 = stream_att_1.shape[3] // 2
    first_length_2 = stream_att_2.shape[3] // 2
    tensors.update({
        "stream_tokens_1": stream_tokens_1,
        "stream_tokens_2": stream_tokens_2,
        "stream_encoder_1": stream_encoder_1,
        "stream_encoder_2": stream_encoder_2,
        "stream_cnn_1": stream_cnn_1,
        "stream_cnn_2": stream_cnn_2,
        "stream_first_k_1": stream_att_1[0, :, :, :first_length_1, :64],
        "stream_first_v_1": stream_att_1[0, :, :, :first_length_1, 64:],
        "stream_second_k_1": stream_att_1[6, :, :, :, :64],
        "stream_second_v_1": stream_att_1[6, :, :, :, 64:],
        "stream_first_k_2": stream_att_2[0, :, :, :first_length_2, :64],
        "stream_first_v_2": stream_att_2[0, :, :, :first_length_2, 64:],
        "stream_second_k_2": stream_att_2[6, :, :, :, :64],
        "stream_second_v_2": stream_att_2[6, :, :, :, 64:],
    })

    # DiT streaming cache parity independent of the ODE wrapper.
    stream_length = 6
    dit_stream_x_1 = torch.randn(2, 80, stream_length)
    dit_stream_mu_1 = torch.randn(2, 80, stream_length) * 0.2
    dit_stream_condition_1 = torch.randn(2, 80, stream_length) * 0.05
    dit_stream_speakers = torch.randn(2, 80) * 0.1
    dit_stream_time = torch.tensor([0.35, 0.35])
    with torch.inference_mode():
        dit_stream_1, dit_cnn_1, dit_att_1 = model.decoder.estimator.forward_chunk(
            dit_stream_x_1, dit_stream_mu_1, dit_stream_time,
            dit_stream_speakers, dit_stream_condition_1)
        dit_cnn_1 = dit_cnn_1.clone()
        dit_att_1 = dit_att_1.clone()
        dit_stream_x_2 = torch.randn(2, 80, stream_length)
        dit_stream_mu_2 = torch.randn(2, 80, stream_length) * 0.2
        dit_stream_condition_2 = torch.randn(2, 80, stream_length) * 0.05
        dit_stream_2, dit_cnn_2, dit_att_2 = model.decoder.estimator.forward_chunk(
            dit_stream_x_2, dit_stream_mu_2, dit_stream_time,
            dit_stream_speakers, dit_stream_condition_2,
            cnn_cache=dit_cnn_1, att_cache=dit_att_1)
    tensors.update({
        "dit_stream_x_1": dit_stream_x_1,
        "dit_stream_mu_1": dit_stream_mu_1,
        "dit_stream_condition_1": dit_stream_condition_1,
        "dit_stream_x_2": dit_stream_x_2,
        "dit_stream_mu_2": dit_stream_mu_2,
        "dit_stream_condition_2": dit_stream_condition_2,
        "dit_stream_speakers": dit_stream_speakers,
        "dit_stream_time": dit_stream_time,
        "dit_stream_output_1": dit_stream_1,
        "dit_stream_output_2": dit_stream_2,
        "dit_stream_k_1": dit_att_1[0, :, :, :, :64],
        "dit_stream_v_1": dit_att_1[0, :, :, :, 64:],
        "dit_stream_k_2": dit_att_2[0, :, :, :, :64],
        "dit_stream_v_2": dit_att_2[0, :, :, :, 64:],
        "dit_stream_cnn_first_2": dit_cnn_2[0, :, :512, :],
        "dit_stream_cnn_second_2": dit_cnn_2[0, :, 512:, :],
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    cpu = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    # Safetensors metadata is part of the artifact and therefore must remain
    # portable.  In particular, never serialize ``source`` or any resolved
    # local path here; all provenance lives in the side-car manifest below.
    save_file(
        cpu,
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
            "token_dtype": "int32",
            "mask_dtype": "float32",
            "seed": args.seed,
            "random_algorithm": "torch_cpu_default_generator",
            "dit_blocks": len(model.decoder.estimator.blocks),
            "cfm_ode_steps": 2,
        },
        "artifact": {
            "path": output.name,
            "sha256": _sha256_file(output),
            "tensor_count": len(cpu),
            "tensors": {
                name: _tensor_metadata(value)
                for name, value in sorted(cpu.items())
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": output.name,
        "manifest": manifest_path.name,
        "tensor_count": len(cpu),
        "upstream_revision": upstream_revision,
        "model_id": model_id,
    }, indent=2))


if __name__ == "__main__":
    main()
