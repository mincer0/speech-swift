#!/usr/bin/env python3
"""
Feasibility spike: export MiniCPM-o's audio encoder (the `apm` tower) to Core ML
for the Apple Neural Engine.

Why this component: `MiniCPMWhisperEncoder` is a plain feed-forward transformer
encoder - no KV cache, no data-dependent control flow, and it supports a fixed
input length via `cnn_min_length`. That is the shape the ANE handles best, and
unlike the TTS/vocoder components it produces embeddings rather than audio, so
a numerical difference shows up as degraded understanding rather than noise
(verifiable with tools/duplex_understand_test.py).

Run with the venv that has torch + coremltools:
  .venv-minicpmo/bin/python scripts/export_minicpm_o_audio_coreml.py

Outputs:
  <out>/MiniCPM-o-4_5-audio-encoder.mlpackage
and prints a torch-vs-CoreML parity check plus a latency comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path("/Users/mincer/项目/s2s/models/MiniCPM-o-4_5")

_TORCHVISION_COMPAT_LIB = None


def ensure_torchvision_compat() -> None:
    """Declare the optional torchvision schemas before importing Transformers.

    The pinned torchvision in this venv was built against a different torch, so
    importing it raises `operator torchvision::nms does not exist`. Transformers
    only needs the schema to exist, and the duplex runtime already applies the
    same shim (minicpm_duplex/runtime.py).
    """
    global _TORCHVISION_COMPAT_LIB
    if _TORCHVISION_COMPAT_LIB is not None:
        return
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


def load_audio_config(model_dir: Path):
    from transformers import WhisperConfig

    raw = json.loads((model_dir / "config.json").read_text())["audio_config"]
    return WhisperConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})


def load_apm_weights(model_dir: Path, prefix: str = "apm."):
    """Stream only the audio-tower tensors out of the sharded checkpoint."""
    from safetensors import safe_open

    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no checkpoint shards in {model_dir}")
    weights = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    weights[key[len(prefix):]] = handle.get_tensor(key).to(torch.float32)
    return weights


def main():
    ensure_torchvision_compat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100,
                    help="mel frames per chunk (fixed shape; the ANE needs static shapes)")
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/mincer/项目/s2s/models/MiniCPM-o-4_5-audio-ane"))
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    # `modeling_minicpmo.py` uses relative imports (`from .configuration_minicpmo
    # import ...`), so it cannot be imported as a top-level module - and the
    # model directory name (`MiniCPM-o-4_5`) is not a valid package identifier.
    # Build a small package of symlinks instead, which satisfies the relative
    # imports without loading the 17 GB checkpoint the way
    # `AutoModel.from_pretrained(trust_remote_code=True)` would.
    import tempfile

    pkg_root = Path(tempfile.mkdtemp(prefix="minicpm_o45_"))
    pkg = pkg_root / "minicpm_o45"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for source in MODEL_DIR.glob("*.py"):
        (pkg / source.name).symlink_to(source)
    sys.path.insert(0, str(pkg_root))
    from minicpm_o45.modeling_minicpmo import MiniCPMWhisperEncoder

    config = load_audio_config(MODEL_DIR)
    print(f"audio config: layers={config.encoder_layers} d_model={config.d_model} "
          f"heads={config.encoder_attention_heads} mel={config.num_mel_bins} "
          f"max_source_positions={config.max_source_positions}")

    model = MiniCPMWhisperEncoder(config)
    state = load_apm_weights(MODEL_DIR)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  missing sample: {missing[:3]}")
    if unexpected[:3]:
        print(f"  unexpected sample: {unexpected[:3]}")

    model.eval()
    model.requires_grad_(False)
    frames = args.frames
    example = torch.randn(1, config.num_mel_bins, frames)

    with torch.no_grad():
        torch_out = model(example, attention_mask=None, use_cache=False)
        # `BaseModelOutputWithPast.hidden_states` is None unless
        # output_hidden_states=True - the encoder output is last_hidden_state.
        torch_hidden = torch_out.last_hidden_state
    print(f"torch output: {tuple(torch_hidden.shape)}")

    # warm start + time
    with torch.no_grad():
        for _ in range(3):
            model(example, attention_mask=None, use_cache=False)
        t0 = time.perf_counter()
        for _ in range(10):
            model(example, attention_mask=None, use_cache=False)
        torch_ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"torch (CPU) latency: {torch_ms:.1f} ms")
    print("  (compare with the MLX `audio_encoder_ms` ~85 ms on the GPU and the")
    print("   official llama.cpp 42 ms - the ANE number below is the one that matters)")

    if args.skip_convert:
        return

    import coremltools as ct

    args.out.mkdir(parents=True, exist_ok=True)

    # Trace a module, not a lambda: `torch.jit.trace` on a plain function
    # returns a ScriptFunction, which coremltools rejects - it needs a
    # ScriptModule.
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, mel):
            out = self.inner(mel, attention_mask=None, use_cache=False)
            return out.last_hidden_state

    wrapper = EncoderWrapper(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    print("traced ok; converting to mlprogram (this takes a minute) …")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=example.shape)],
        outputs=[ct.TensorType(name="hidden")],
        # coremltools 9.0 does not auto-detect the frontend for a torch 2.8
        # ScriptModule, so name it explicitly.
        source="pytorch",
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = "MiniCPM-o 4.5 audio encoder (apm) exported for the ANE"
    mlmodel.input_description["mel"] = f"[1, {config.num_mel_bins}, {frames}] log-mel"
    package = args.out / "MiniCPM-o-4_5-audio-encoder.mlpackage"
    mlmodel.save(str(package))
    print(f"saved {package}")

    # parity + latency
    cpu_model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    ane_model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    x_np = example.numpy()
    ref = torch_hidden.detach().numpy()
    print(f"  reference activation: mean|.|={np.abs(ref).mean():.4f} max|.|={np.abs(ref).max():.4f}")
    for label, m in (("CPU_ONLY", cpu_model), ("CPU_AND_NE", ane_model)):
        out = m.predict({"mel": x_np})
        key = list(out.keys())[0]
        got = np.asarray(out[key])
        # Raw max|diff| is dominated by fp16 rounding on the largest activations,
        # which is expected for the ANE - report scale-relative numbers plus the
        # cosine similarity the downstream LLM actually cares about.
        max_diff = np.abs(got - ref).max()
        rel = max_diff / (np.abs(ref).max() + 1e-9)
        g2 = got.reshape(-1, got.shape[-1])
        r2 = ref.reshape(-1, ref.shape[-1])
        cos = (g2 * r2).sum(1) / (np.linalg.norm(g2, axis=1) * np.linalg.norm(r2, axis=1) + 1e-9)
        for _ in range(3):
            m.predict({"mel": x_np})
        t0 = time.perf_counter()
        for _ in range(10):
            m.predict({"mel": x_np})
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {label}: max|diff|={max_diff:.3e} ({rel*100:.2f}% of peak)  "
              f"cos={cos.mean():.6f} (min {cos.min():.6f})  latency {ms:.1f} ms  {got.shape}")


if __name__ == "__main__":
    main()
