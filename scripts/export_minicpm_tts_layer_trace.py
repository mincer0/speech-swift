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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("condition_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream-repo", type=Path, default=Path("/tmp/MiniCPM-o-Demo"))
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--chunk-index", type=int, default=0)
    p.add_argument("--step-index", type=int, default=0,
                   help="teacher-forced decoder call within the selected chunk (0 is condition prefill)")
    p.add_argument("--include-head-weight", action="store_true",
                   help="also record the reconstructed audio-head weight for a loader parity probe")
    p.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default=None,
        help="override the official Llama attention backend for diagnostics",
    )
    return p.parse_args()


def teacher_forced_plan(metadata: dict[str, Any], chunk_index: int, step_index: int) -> list[dict[str, Any]]:
    """Plan the official cache replay without importing Torch or a model.

    A full chunk samples N codes but returns N-1.  The last decoder call
    consumes returned code N-2; the discarded sample is never fed into the
    next condition.  This is the boundary the layer diagnostic must preserve.
    """
    chunks, trace = metadata["chunks"], metadata["trace"]
    if not 0 <= chunk_index < len(chunks) or len(trace) != len(chunks):
        raise ValueError("chunk index is outside the official trace")
    if not 0 <= step_index < int(trace[chunk_index]["sampled_count"]):
        raise ValueError("step index is outside the official sampled decoder calls")
    plan: list[dict[str, Any]] = []
    offset = 0
    for index in range(chunk_index + 1):
        entry = trace[index]
        if int(entry["text_start"]) != offset:
            raise ValueError("official trace text_start does not match cache replay")
        steps = step_index + 1 if index == chunk_index else int(entry["sampled_count"])
        for step in range(steps):
            length = int(entry["condition_length"]) if step == 0 else 1
            plan.append({
                "chunk_index": index,
                "step_index": step,
                "condition_key": chunks[index]["expected_key"],
                "input_code": None if step == 0 else int(entry["committed"][step - 1]),
                "cache_offset_before": offset,
                "sequence_length": length,
            })
            offset += length
        if index < chunk_index and offset != int(entry["cache_offset"]):
            raise ValueError("official cache offset does not match full-chunk replay")
    return plan


def main() -> None:
    args = parse_args()
    # Reuse the strict source/weight loading helpers from the official-oracle
    # exporter.  This keeps the diagnostic trace tied to the same provenance
    # checks and tensor shard as the parity gate.
    scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts))
    import export_minicpm_tts_semantic_oracle as oracle

    observed = oracle.source_hashes(args.upstream_repo)
    if (observed["upstream_commit"] != oracle.UPSTREAM_COMMIT
            or observed["modeling_sha256"] != oracle.MODELING_SHA256
            or observed["unified_modeling_sha256"] != oracle.UNIFIED_MODELING_SHA256):
        raise SystemExit("layer trace requires the pinned official source checkout")
    fixture_metadata = json.loads((args.condition_dir / "minicpm-tts-condition-oracle.json").read_text())
    plan = teacher_forced_plan(fixture_metadata, args.chunk_index, args.step_index)
    torch, safe_open, _save_file = oracle.torch_dependencies()
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
    condition = fixture[plan[-1]["condition_key"]].to(device=device, dtype=torch.bfloat16)

    arrays: dict[str, torch.Tensor] = {}
    # Keep the post-layer-norm tensors around so the official RoPE boundary
    # can be exported after the model forward.  Registering a hook directly
    # on LlamaAttention would only expose its fused output; recomputing q/k
    # from the exact BF16 input lets the Swift diagnostic distinguish a RoPE
    # convention mismatch from the subsequent attention implementation.
    input_norm_values: dict[str, torch.Tensor] = {}
    post_attention_norm_values: dict[str, torch.Tensor] = {}
    capturing = False
    metadata: dict[str, Any] = {
        "format": 2,
        "oracle_backend": "official_pytorch",
        "oracle_dtype": "bfloat16",
        "device": str(device),
        "condition_key": plan[-1]["condition_key"],
        "condition_shape": list(condition.shape),
        "chunk_index": args.chunk_index,
        "step_index": args.step_index,
        "cache_offset_before": plan[-1]["cache_offset_before"],
        "sequence_length": plan[-1]["sequence_length"],
        "input_code": plan[-1]["input_code"],
        "replay_plan": plan,
        "teacher_forcing": "official committed codes; discarded sampled code is never fed",
        "attention_implementation": tts.model.config._attn_implementation,
        "attention_dropout": float(tts.model.config.attention_dropout),
        "training": bool(tts.training),
        "layers": int(tts.config.num_hidden_layers),
        "operations": [],
    }

    def record(name: str, value: Any) -> None:
        if not capturing:
            return
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
            if capturing:
                input_norm_values[p] = output.detach().clone()

        hooks.append(layer.input_layernorm.register_forward_hook(save_input_norm))
        # Keep the projection/rotation boundary explicit.  The first layer
        # attention mismatch can otherwise be caused by either the linear
        # layout or RoPE, while the higher-level attention hook only shows the
        # combined result after both have already been applied.
        hooks.append(layer.self_attn.q_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".q_proj", output)))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".k_proj", output)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".v_proj", output)))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda module, inputs, p=prefix: record(p + ".attention_context", inputs[0])))
        hooks.append(layer.self_attn.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".attention", output)))
        def save_post_attention_norm(module, inputs, output, p=prefix):
            record(p + ".post_attention_norm", output)
            record(p + ".residual_before_mlp", inputs[0])
            if capturing:
                post_attention_norm_values[p] = output.detach().clone()

        hooks.append(layer.post_attention_layernorm.register_forward_hook(save_post_attention_norm))
        hooks.append(layer.mlp.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".mlp", output)))
        hooks.append(layer.register_forward_hook(lambda module, inputs, output, p=prefix: record(p + ".output", output)))

    try:
        with torch.inference_mode():
            cache = None
            for call_index, call in enumerate(plan):
                is_target = call_index == len(plan) - 1
                capturing = is_target
                input_embeds = (fixture[call["condition_key"]].to(device=device, dtype=torch.bfloat16)
                                if call["input_code"] is None else tts.emb_code[0](torch.tensor(
                                    [[call["input_code"]]], device=device, dtype=torch.long)))
                offset = int(call["cache_offset_before"])
                length = int(call["sequence_length"])
                assert list(input_embeds.shape[:2]) == [1, length]
                if is_target:
                    for layer_index, layer in enumerate(tts.model.layers):
                        prefix = f"layer_{layer_index:02d}"
                        if cache is None:
                            empty = torch.empty((1, tts.model.config.num_key_value_heads, 0,
                                                 layer.self_attn.head_dim), device=device, dtype=torch.bfloat16)
                            record(prefix + ".cache_keys_before", empty)
                            record(prefix + ".cache_values_before", empty)
                        else:
                            keys, values = cache[layer_index]
                            assert int(keys.shape[2]) == offset
                            record(prefix + ".cache_keys_before", keys)
                            record(prefix + ".cache_values_before", values)
                position_ids = torch.arange(offset, offset + length, device=device, dtype=torch.long).unsqueeze(0)
                output = tts.model(
                    inputs_embeds=input_embeds,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=is_target,
                )
                cache = output.past_key_values
                assert int(cache[0][0].shape[2]) == offset + length
            condition = input_embeds
    finally:
        for hook in hooks:
            hook.remove()

    record("embedding_input", condition)
    record("final_output", output.last_hidden_state)
    if output.hidden_states is not None:
        for index, value in enumerate(output.hidden_states):
            record(f"hidden_state_{index:02d}", value)

    # Match generate_chunk's full hidden-state call shape, including T=1.
    with torch.inference_mode():
        logits = tts.head_code[0](output.last_hidden_state)[:, -1, :].reshape(-1).float()
    record("first_logits", logits)
    record("target_logits", logits)
    reference = fixture[f"chunk{args.chunk_index}.teacher_logits"][args.step_index]
    record("reference_teacher_logits", reference)
    metadata["teacher_logit_reference_max_abs"] = float((logits.cpu() - reference.cpu()).abs().max())
    if args.include_head_weight:
        record("audio_head_weight", tts.head_code[0].weight)
    for index in range(len(tts.model.layers)):
        record(f"layer_{index:02d}.cache_keys_after", cache[index][0])
        record(f"layer_{index:02d}.cache_values_after", cache[index][1])

    # Export the official HuggingFace Llama RoPE result.  The pinned
    # implementation computes cos/sin in float32, casts them to BF16, then
    # applies the half-split ``rotate_half`` convention.  Keeping this as a
    # separate trace boundary is important: MLX's ``RoPE(traditional:false)``
    # uses an interleaved layout and is not assumed equivalent without a
    # tensor-level comparison.
    position_ids = torch.arange(
        plan[-1]["cache_offset_before"],
        plan[-1]["cache_offset_before"] + condition.shape[1],
        device=condition.device, dtype=torch.long
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
            record(prefix + ".mlp_product", activated * up)
            record(prefix + ".mlp_down", layer.mlp.down_proj(activated * up))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(args.output), metadata={"trace_format": "minicpm-o-tts-layer-trace/v2"})
    metadata["tensor_keys"] = sorted(arrays)
    metadata["source"] = {
        "upstream_commit": oracle.UPSTREAM_COMMIT,
        "modeling_sha256": oracle.MODELING_SHA256,
        "unified_modeling_sha256": oracle.UNIFIED_MODELING_SHA256,
        "observed_upstream_commit": observed["upstream_commit"],
        "observed_modeling_sha256": observed["modeling_sha256"],
        "observed_unified_modeling_sha256": observed["unified_modeling_sha256"],
        "tts_shard": shard.name,
        "tts_shard_sha256": oracle.sha256(shard),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "tensors": len(arrays)}, indent=2))


if __name__ == "__main__":
    main()
