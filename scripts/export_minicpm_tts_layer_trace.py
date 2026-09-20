#!/usr/bin/env python3
"""Export a layer/operator trace for the pinned MiniCPM-o TTS decoder.

This is a diagnostic companion to ``export_minicpm_tts_semantic_oracle.py``.
It deliberately runs the official PyTorch implementation from the pinned
Demo checkout and records the BF16 values after each Llama decoder operation.
The output is used only to locate a numerical parity break; it is not a new
golden and must never be regenerated to make a failing Swift test pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("condition_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream-repo", type=Path, default=Path("/tmp/MiniCPM-o-Demo"))
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default=None,
        help="override the official Llama attention backend for diagnostics",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Reuse the strict source/weight loading helpers from the official-oracle
    # exporter.  This keeps the diagnostic trace tied to the same provenance
    # checks and tensor shard as the parity gate.
    scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts))
    import export_minicpm_tts_semantic_oracle as oracle

    torch_mod, safe_open, _save_file = oracle.torch_dependencies()
    MiniCPMTTSConfig, MiniCPMTTS = oracle.import_official_tts(args.upstream_repo)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tts_config = dict(config["tts_config"])
    if args.attn_implementation is not None:
        tts_config["attn_implementation"] = args.attn_implementation
    tts = MiniCPMTTS(
        config=MiniCPMTTSConfig(**tts_config),
        audio_tokenizer=None,
    )
    device = torch.device(args.device)
    tts.to(device=device, dtype=torch.bfloat16).eval()

    shard = oracle.select_tts_shard(args.model_dir)
    state: dict[str, Any] = {}
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("tts."):
                state[key.removeprefix("tts.")] = handle.get_tensor(key)
    state = {key: value.to(device=device, dtype=torch.bfloat16) for key, value in state.items()}
    missing, unexpected = tts.load_state_dict(state, strict=False)
    if set(missing) - {"model.embed_tokens.weight"} or unexpected:
        raise RuntimeError(f"official TTS state mismatch: missing={missing} unexpected={unexpected}")
    del state

    from safetensors.torch import load_file, save_file

    fixture = load_file(str(args.condition_dir / "minicpm-tts-condition-oracle.safetensors"))
    condition = fixture["chunk0.expected"].to(device=device, dtype=torch.bfloat16)

    arrays: dict[str, torch.Tensor] = {}
    # Keep the post-layer-norm tensors around so the official RoPE boundary
    # can be exported after the model forward.  Registering a hook directly
    # on LlamaAttention would only expose its fused output; recomputing q/k
    # from the exact BF16 input lets the Swift diagnostic distinguish a RoPE
    # convention mismatch from the subsequent attention implementation.
    input_norm_values: dict[str, torch.Tensor] = {}
    post_attention_norm_values: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "format": 1,
        "oracle_backend": "official_pytorch",
        "oracle_dtype": "bfloat16",
        "device": str(device),
        "condition_key": "chunk0.expected",
        "condition_shape": list(condition.shape),
        "layers": int(tts.config.num_hidden_layers),
        "operations": [],
    }

    def record(name: str, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            if not value:
                return
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return
        # Convert only the saved diagnostic copy to FP32.  The forward path
        # remains BF16, so every saved value is an exact representation of the
        # official BF16 result.
        arrays[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()

    hooks = []
    for index, layer in enumerate(tts.model.layers):
        prefix = f"layer_{index:02d}"
        hooks.append(layer.register_forward_pre_hook(lambda module, inputs, p=prefix: record(p + ".input", inputs[0])))
        def save_input_norm(module, inputs, output, p=prefix):
            record(p + ".input_norm", output)
            input_norm_values[p] = output.detach()

        hooks.append(layer.input_layernorm.register_forward_hook(save_input_norm))
        # Keep the projection/rotation boundary explicit.  The first layer
        # attention mismatch can otherwise be caused by either the linear
        # layout or RoPE, while the higher-level attention hook only shows the
        # combined result after both have already been applied.
        hooks.append(layer.self_attn.q_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".q_proj", output)))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".k_proj", output)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".v_proj", output)))
        hooks.append(layer.self_attn.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".attention", output)))
        def save_post_attention_norm(module, inputs, output, p=prefix):
            record(p + ".post_attention_norm", output)
            post_attention_norm_values[p] = output.detach()

        hooks.append(layer.post_attention_layernorm.register_forward_hook(save_post_attention_norm))
        hooks.append(layer.mlp.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".mlp", output)))
        hooks.append(layer.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".output", output)))

    try:
        with torch.inference_mode():
            output = tts.model(
                inputs_embeds=condition,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    record("embedding_input", condition)
    record("final_output", output.last_hidden_state)
    if output.hidden_states is not None:
        for index, value in enumerate(output.hidden_states):
            record(f"hidden_state_{index:02d}", value)

    # Include the exact official first-step logits so a trace consumer can
    # verify that its final norm/head comparison is using the same probe.
    with torch.inference_mode():
        logits = tts.head_code[0](output.last_hidden_state[:, -1, :]).reshape(-1).float()
    record("first_logits", logits)

    # Export the official HuggingFace Llama RoPE result.  The pinned
    # implementation computes cos/sin in float32, casts them to BF16, then
    # applies the half-split ``rotate_half`` convention.  Keeping this as a
    # separate trace boundary is important: MLX's ``RoPE(traditional:false)``
    # uses an interleaved layout and is not assumed equivalent without a
    # tensor-level comparison.
    position_ids = torch.arange(
        condition.shape[1], device=condition.device, dtype=torch.long
    ).unsqueeze(0)
    cos, sin = tts.model.rotary_emb(condition, position_ids)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        split = value.shape[-1] // 2
        # This is the HuggingFace/Llama split-half convention.  The first
        # output half is the negated *second* input half and the second output
        # half is the original first half.  Keeping this spelling identical
        # to MiniCPMO45.utils.rotate_half is important: MLX's
        # ``RoPE(traditional: false)`` uses an interleaved layout and must not
        # be used as an implicit oracle here.
        first = value[..., :split]
        second = value[..., split:]
        return torch.cat((-second, first), dim=-1)

    with torch.inference_mode():
        for index, layer in enumerate(tts.model.layers):
            prefix = f"layer_{index:02d}"
            normed = input_norm_values[prefix]
            q = layer.self_attn.q_proj(normed).view(
                condition.shape[0], condition.shape[1], -1, layer.self_attn.head_dim
            ).transpose(1, 2)
            k = layer.self_attn.k_proj(normed).view(
                condition.shape[0], condition.shape[1], -1, layer.self_attn.head_dim
            ).transpose(1, 2)
            # ``cos``/``sin`` are [batch, sequence, head_dim]; unsqueeze at 1
            # matches transformers.apply_rotary_pos_emb for [batch, heads, seq,
            # head_dim].
            cos_heads = cos.unsqueeze(1)
            sin_heads = sin.unsqueeze(1)
            record(prefix + ".q_rotated", q * cos_heads + rotate_half(q) * sin_heads)
            record(prefix + ".k_rotated", k * cos_heads + rotate_half(k) * sin_heads)

            # Keep the MLP arithmetic boundary explicit as well.  A single
            # BF16 SiLU/linear outlier can otherwise be amplified by later
            # residual layers and look like an attention or RoPE failure.
            mlp_input = post_attention_norm_values[prefix]
            gate = layer.mlp.gate_proj(mlp_input)
            up = layer.mlp.up_proj(mlp_input)
            activated = layer.mlp.act_fn(gate)
            record(prefix + ".mlp_gate", gate)
            record(prefix + ".mlp_up", up)
            record(prefix + ".mlp_activation", activated)
            record(prefix + ".mlp_down", layer.mlp.down_proj(activated * up))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(args.output), metadata={"trace_format": "minicpm-o-tts-layer-trace/v1"})
    metadata["tensor_keys"] = sorted(arrays)
    metadata["source"] = {
        "upstream_commit": oracle.UPSTREAM_COMMIT,
        "modeling_sha256": oracle.MODELING_SHA256,
        "unified_modeling_sha256": oracle.UNIFIED_MODELING_SHA256,
        "tts_shard": shard.name,
        "tts_shard_sha256": oracle.sha256(shard),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "tensors": len(arrays)}, indent=2))


if __name__ == "__main__":
    main()
