#!/usr/bin/env python3
"""Export a tiny official-source TTS oracle; no checkpoint weights are loaded.

The pinned MiniCPMTTS.generate_chunk AST runs unchanged over a one-layer
PyTorch Llama with deterministic Float32 weights. Head hooks capture every
step's logits and sampled codes for teacher-forced Swift comparisons across
condition boundaries. Generation, reference logits and weights all come from
PyTorch, independently of the Swift implementation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Union

sys.dont_write_bytecode = True

COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
SOURCE_SHA = "877f20cea7282bb0d05684393a1580646d1f264a2299a42f82467fd99f4028af"
SOURCE_URL = f"https://raw.githubusercontent.com/OpenBMB/MiniCPM-o-Demo/{COMMIT}/MiniCPMO45/modeling_minicpmo.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parents[1] / "Tests/MiniCPMTTSSemanticTests/Fixtures/minicpm-tts-tiny-oracle.safetensors",
    )
    args = parser.parse_args()
    data = (
        args.official_source.read_bytes() if args.official_source
        else urllib.request.urlopen(SOURCE_URL, timeout=20).read()
    )
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA, "official source hash mismatch"
    source = data.decode()
    tree = ast.parse(source)
    tts_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniCPMTTS")
    function = next(n for n in tts_class.body if isinstance(n, ast.FunctionDef) and n.name == "generate_chunk")

    import torch
    from safetensors.torch import save_file

    # Some CPU torchvision builds omit their NMS schemas; declare the schema
    # before Transformers imports that unrelated optional package.
    library = None
    try:
        library = torch.library.Library("torchvision", "DEF")
        for operator in ("nms", "qnms"):
            library.define(f"{operator}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except RuntimeError:
        pass
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel, apply_rotary_pos_emb

    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=1, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, rms_norm_eps=1e-6,
    )
    config._attn_implementation = "sdpa"
    model = LlamaModel(config).eval()
    text = torch.nn.Embedding(10, 8)
    codes = torch.nn.Embedding(7, 8)
    head = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(8, 7, bias=False))
    linear1 = torch.nn.Linear(4, 8)
    linear2 = torch.nn.Linear(8, 8)
    modules = [model, text, codes, head, linear1, linear2]
    with torch.no_grad():
        for module_index, module in enumerate(modules):
            for index, (name, parameter) in enumerate(module.named_parameters()):
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                elif name.endswith("original0"):
                    parameter.fill_(0.8)
                else:
                    values = torch.arange(parameter.numel(), dtype=torch.float32)
                    values = 0.2 * torch.sin(values * 0.73 + (index + 1) * 0.41 + module_index)
                    parameter.copy_(values.reshape_as(parameter))
    scope = {
        "torch": torch, "Union": Union,
        "BaseModelOutputWithPast": SimpleNamespace,
        "P": SimpleNamespace(cached=nullcontext), "F": torch.nn.functional,
        # Repetition=1 produces no processors, and generate_chunk never uses
        # the logits warpers returned by the official helper.
        "gen_logits": lambda **kwargs: ([], []),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), SOURCE_URL, "exec"), scope)
    tts = SimpleNamespace(
        config=SimpleNamespace(num_audio_tokens=7), device=torch.device("cpu"),
        num_audio_tokens=7, num_vq=1, model=model, emb_code=[codes], head_code=[head],
    )
    arrays = {}
    for prefix, module in [
        ("model", model), ("emb_text", text), ("emb_code.0", codes),
        ("head_code.0", head), ("projector_semantic.linear1", linear1),
        ("projector_semantic.linear2", linear2),
    ]:
        for key, value in module.state_dict().items():
            arrays[f"weight.tts.{prefix}.{key}"] = value.detach().clone().contiguous()

    cache = None
    text_start = 0
    trace = []
    with torch.inference_mode():
        for chunk, ids in enumerate([[1, 2, 3], [2, 3, 4, 5], [1, 4]]):
            hidden = torch.cos(torch.arange(len(ids) * 4).reshape(len(ids), 4).float() * 0.3 + chunk)
            projected = linear2(torch.relu(linear1(hidden)))
            merged = text(torch.tensor(ids)) + torch.nn.functional.normalize(projected, dim=-1)
            condition = torch.cat([merged, text(torch.tensor([9]))], dim=0).unsqueeze(0)
            arrays[f"chunk{chunk}.hidden"] = hidden
            arrays[f"chunk{chunk}.condition"] = condition
            captured = []
            hook = head.register_forward_hook(lambda _m, _inputs, output: captured.append(output[:, -1].reshape(-1).clone()))
            returned, cache = scope["generate_chunk"](
                tts, inputs_embeds=condition, temperature=torch.tensor([1e-6]),
                repetition_penalty=1.0, eos_token=torch.tensor(6), force_no_stop=True,
                max_new_token=26, min_new_tokens=0, past_key_values=cache,
                text_start_pos=text_start,
            )
            hook.remove()
            logits = torch.stack(captured)
            masked = logits.clone()
            masked[:, 6] = -torch.inf
            sampled = masked.argmax(dim=-1)
            assert returned[0, :, 0].tolist() == sampled[:-1].tolist()
            arrays[f"chunk{chunk}.logits"] = logits
            arrays[f"chunk{chunk}.sampled"] = sampled.to(torch.int32)
            length = condition.shape[1]
            text_start += length + returned.shape[1]
            cache_length = int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else int(cache[0][0].shape[2])
            assert text_start == cache_length
            trace.append({
                "index": chunk, "tokens": ids, "condition_length": length,
                "committed": returned[0, :, 0].tolist(), "cache_offset": cache_length,
                "next_text_start": text_start,
            })
        assert [row["cache_offset"] for row in trace] == [29, 59, 87]

        normalization_input = torch.tensor(
            [0.275390625, 0.349609375, 0.40625, 0.44140625,
             0.462890625, 0.47265625, 0.4765625, 0.478515625], dtype=torch.bfloat16
        )
        arrays["normalization.input"] = normalization_input
        arrays["normalization.expected"] = torch.nn.functional.normalize(
            normalization_input, dim=-1
        )
        assert not torch.equal(
            arrays["normalization.expected"],
            (normalization_input.float() / normalization_input.float().norm()).bfloat16(),
        ), "normalization fixture must catch division performed in Float32"

        # Module.to casts the non-persistent inv_freq buffer, too. Use the
        # untouched HF implementation at a position that distinguishes that
        # BF16 rounding from permanently retained Float32 frequencies.
        rotary = copy.deepcopy(model.rotary_emb).to(dtype=torch.bfloat16)
        rotary_input = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 1, 4).bfloat16()
        positions = torch.tensor([[53]], dtype=torch.long)
        cos, sin = rotary(rotary_input, positions)
        rotated, _ = apply_rotary_pos_emb(rotary_input, rotary_input, cos, sin)
        arrays["rotary.input"] = rotary_input
        arrays["rotary.expected"] = rotated
        activation_input = torch.linspace(-8, 8, 257).bfloat16()
        arrays["activation.input"] = activation_input
        arrays["activation.expected"] = torch.nn.functional.silu(activation_input)

    config_data = {
        "llm_dim": 4, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "num_audio_tokens": 7, "num_text_tokens": 10, "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6, "audio_bos_token_id": 9, "text_eos_token_id": 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in arrays.items()}, str(args.output))
    metadata = {
        "format": 1, "oracle_backend": "official_pytorch_tiny",
        "upstream_commit": COMMIT, "modeling_sha256": SOURCE_SHA,
        "generate_chunk_sha256": hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest(),
        "source_url": SOURCE_URL, "torch": torch.__version__,
        "dtype": "float32", "config": config_data, "trace": trace,
        "weights_loaded": False,
        "teacher_forcing": "Replay each reference sampled code into the next step; discard the final code before the next condition.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cache_offsets": [r["cache_offset"] for r in trace]}))


if __name__ == "__main__":
    main()
