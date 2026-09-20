#!/usr/bin/env python3
"""Export compact layer-0 intermediates for Swift/PyTorch parity probes.

This is intentionally opt-in and independent from the full golden exporter:
it loads the pinned official PyTorch Qwen3 LLM, evaluates a fixed or explicitly
selected prefill (or an opt-in DynamicCache successor decode), and writes one
little-endian float32 file per operation.  The Swift ``MiniCPMDebugTraceTests``
real-bundle path writes the same names and layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TRACE_TOKEN_IDS = (151643, 198, 108, 151645)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--mlx-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, default=Path("/tmp/MiniCPM-o-Demo"))
    parser.add_argument(
        "--trace-layer", type=int, default=None, metavar="INDEX",
        help=(
            "trace one target layer after sequentially evaluating prior layers; "
            "omit this option for the full decoder layer-output summary (layer 0 is valid)"
        ),
    )
    parser.add_argument(
        "--token-count", type=int, default=len(DEFAULT_TRACE_TOKEN_IDS), metavar="COUNT",
        help=(
            "number of default fixed prefill tokens to trace without a KV cache "
            f"(1-{len(DEFAULT_TRACE_TOKEN_IDS)}, default: {len(DEFAULT_TRACE_TOKEN_IDS)})"
        ),
    )
    parser.add_argument(
        "--token-ids", type=parse_token_ids, default=None, metavar="IDS",
        help=(
            "comma-separated token IDs to trace (overrides --token-count and the "
            "default fixed sequence; for example: 151643,151645,198)"
        ),
    )
    parser.add_argument(
        "--cached-prefix-count", type=int, default=0, metavar="COUNT",
        help=(
            "run a cached decode: prefill COUNT selected tokens, then trace the "
            f"successor token with DynamicCache (1-{len(DEFAULT_TRACE_TOKEN_IDS) - 1}; "
            "default: 0, cache-free)"
        ),
    )
    parser.add_argument(
        "--trace-all-layers", action="store_true",
        help=(
            "in cached mode, also write one layer_NN_layer_out.f32 tensor for "
            "every decoder layer"
        ),
    )
    return parser.parse_args(argv)


def is_targeted_trace(trace_layer: int | None) -> bool:
    """Return whether the CLI explicitly requested detailed layer tracing."""

    return trace_layer is not None


def validate_trace_layer(trace_layer: int | None, layer_count: int) -> None:
    """Validate an explicitly requested layer against the loaded decoder."""

    if trace_layer is not None and (trace_layer < 0 or trace_layer >= layer_count):
        raise ValueError(
            f"--trace-layer must be in [0, {layer_count - 1}], got {trace_layer}"
        )


def parse_token_ids(text: str | list[int] | None) -> list[int] | None:
    """Parse an explicit comma-separated token sequence.

    ``None`` means the option was omitted and keeps the historical fixed
    sequence.  Empty fields and negative/non-integer IDs are rejected so an
    accidentally empty environment/CLI value cannot silently select a
    different trace.
    """

    if text is None:
        return None
    if isinstance(text, list):
        if not text:
            raise ValueError("--token-ids must be a non-empty comma-separated sequence")
        token_ids = [int(token_id) for token_id in text]
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("--token-ids must be non-negative")
        return token_ids
    fields = text.split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("--token-ids must be a non-empty comma-separated sequence")
    token_ids: list[int] = []
    for field in fields:
        token_text = field.strip()
        try:
            token_id = int(token_text)
        except ValueError as exc:
            raise ValueError(
                f"--token-ids contains a non-integer value: {token_text!r}"
            ) from exc
        if token_id < 0:
            raise ValueError(f"--token-ids must be non-negative, got {token_id}")
        token_ids.append(token_id)
    return token_ids


def trace_token_ids(
    token_count: int, token_ids: list[int] | str | None = None,
) -> list[int]:
    """Select explicit IDs or the fixed cache-free prefill prefix."""

    if token_ids is not None:
        parsed = parse_token_ids(token_ids) if isinstance(token_ids, str) else [
            int(token_id) for token_id in token_ids
        ]
        if parsed is None or not parsed:
            raise ValueError("--token-ids must be a non-empty comma-separated sequence")
        if any(token_id < 0 for token_id in parsed):
            raise ValueError("--token-ids must be non-negative")
        return parsed

    if token_count < 1 or token_count > len(DEFAULT_TRACE_TOKEN_IDS):
        raise ValueError(
            f"--token-count must be in [1, {len(DEFAULT_TRACE_TOKEN_IDS)}], got {token_count}"
        )
    return [int(token_id) for token_id in DEFAULT_TRACE_TOKEN_IDS[:token_count]]


def resolve_trace_token_ids(
    token_count: int, token_ids: str | list[int] | None = None,
) -> list[int]:
    """Select explicit IDs when supplied, otherwise use ``--token-count``."""

    if isinstance(token_ids, str) or token_ids is None:
        parsed = parse_token_ids(token_ids)
    else:
        parsed = [int(token_id) for token_id in token_ids]
    if parsed is not None:
        return trace_token_ids(token_count, parsed)
    return trace_token_ids(token_count)


def validate_cached_prefix_count(prefix_count: int, token_count: int) -> None:
    """Validate a cached prefix while keeping the fixed successor available."""

    if prefix_count == 0:
        return
    if token_count < 2 or prefix_count < 1 or prefix_count >= token_count:
        raise ValueError(
            f"--cached-prefix-count must be in [1, {token_count - 1}], got {prefix_count}"
        )


def cached_trace_provenance(
    prefix_token_ids: list[int], current_token_ids: list[int],
    *, trace_layer: int, attn_implementation: str,
    token_ids: list[int] | None = None,
) -> dict[str, object]:
    """Describe the token split and backend used by a cached decode trace."""

    all_token_ids = token_ids if token_ids is not None else prefix_token_ids + current_token_ids
    return {
        "mode": "cached_decode",
        "token_ids": [int(token_id) for token_id in all_token_ids],
        "prefix_token_ids": [int(token_id) for token_id in prefix_token_ids],
        "current_token_ids": [int(token_id) for token_id in current_token_ids],
        "prefix_count": len(prefix_token_ids),
        "current_length": len(current_token_ids),
        "prefix_length": len(prefix_token_ids),
        "input_length": len(all_token_ids),
        "trace_layer": int(trace_layer),
        "attn_implementation": str(attn_implementation),
    }


def trace_input_provenance(token_ids: list[int]) -> dict[str, object]:
    """Describe the fixed token input used by a cache-free trace."""

    return {
        "token_ids": [int(token_id) for token_id in token_ids],
        "input_length": len(token_ids),
    }


def enter_no_grad_inference(torch_module: Any) -> None:
    """Enter and verify the fail-closed inference mode used by this trace."""

    torch_module.set_grad_enabled(False)
    if torch_module.is_grad_enabled():
        raise RuntimeError(
            "layer trace requires torch.no_grad semantics, but gradients remain enabled"
        )


def assert_trace_tensor(torch_module: Any, name: str, value: object) -> None:
    """Reject gradient-tracked tensors before they can enter the trace output."""

    if torch_module.is_grad_enabled():
        raise RuntimeError(
            f"layer trace requires torch.no_grad semantics before recording {name}"
        )
    if torch_module.is_tensor(value) and getattr(value, "requires_grad", False):
        raise RuntimeError(f"trace tensor {name} unexpectedly requires_grad=True")


def trace_provenance(torch_module: Any) -> dict[str, object]:
    """Return provenance only after confirming the process is still no-grad."""

    if torch_module.is_grad_enabled():
        raise RuntimeError(
            "cannot write layer trace provenance while gradients are enabled"
        )
    return {
        "grad_enabled": False,
        "inference_semantics": "torch.no_grad",
    }


def _summary(value: object) -> tuple[list[int], np.ndarray, dict[str, object]]:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype="<f4").copy(order="C")
    flat = array.reshape(-1)
    digest = {
        "shape": [int(item) for item in array.shape],
        "sum": float(np.sum(flat, dtype=np.float64)),
        "sumsq": float(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64)),
        "first8": [float(item) for item in flat[:8]],
        "last8": [float(item) for item in flat[-8:]],
    }
    if array.size in (32 * 4 * 128, 8 * 4 * 128):
        heads = 32 if array.size == 32 * 4 * 128 else 8
        shaped = array.reshape(1, heads, 4, 128)
        digest["position_sums"] = [
            float(np.sum(shaped[:, :, position, :], dtype=np.float64))
            for position in range(4)
        ]
    return [int(item) for item in array.shape], array, digest


def _first_tensor(torch_module: Any, value: object) -> Any:
    """Return the first tensor nested in a module output or hook argument."""

    if torch_module.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(torch_module, item)
            if found is not None:
                return found
    return None


def _looks_like_cache(value: object) -> bool:
    return any(
        hasattr(value, name)
        for name in ("key_cache", "value_cache", "layers", "self_attention_cache")
    )


def _cache_layer_pair(cache: object, layer_index: int) -> tuple[object | None, object | None]:
    """Read one layer from DynamicCache without depending on one transformers API."""

    if cache is None:
        return None, None
    if hasattr(cache, "self_attention_cache"):
        cache = getattr(cache, "self_attention_cache")
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = getattr(cache, "key_cache")
        values = getattr(cache, "value_cache")
        if layer_index >= len(keys) or layer_index >= len(values):
            return None, None
        return keys[layer_index], values[layer_index]
    if hasattr(cache, "layers"):
        layers = getattr(cache, "layers")
        if layer_index >= len(layers):
            return None, None
        layer = layers[layer_index]
        return getattr(layer, "keys", None), getattr(layer, "values", None)
    try:
        pair = cache[layer_index]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None, None
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return pair[0], pair[1]
    return getattr(pair, "keys", None), getattr(pair, "values", None)


def _heads_first(torch_module: Any, value: Any, heads: int) -> Any:
    """Normalize q/k/v projection hook output to [batch, heads, length, dim]."""

    if value.ndim == 4:
        # Qwen3 q_norm/k_norm consume [batch, length, heads, head_dim].
        if int(value.shape[1]) == heads and int(value.shape[2]) != heads:
            return value
        return value.transpose(1, 2)
    if value.ndim != 3:
        raise RuntimeError(f"unexpected Qwen3 head tensor rank: {tuple(value.shape)}")
    batch, length, width = value.shape
    if int(width) % heads:
        raise RuntimeError(
            f"Qwen3 projection width {int(width)} is not divisible by {heads} heads"
        )
    return value.view(batch, length, heads, int(width) // heads).transpose(1, 2)


def _resolve_attention_heads(attention: Any, config: Any, *, key_value: bool) -> int:
    """Resolve Qwen3 head counts across Transformers attention API revisions.

    Transformers 4.51's ``Qwen3Attention`` keeps these values only on the
    model config (``num_attention_heads``/``num_key_value_heads``), whereas
    some later revisions also expose ``num_heads``/``num_key_value_heads`` on
    the attention instance.  Prefer an instance value when available and
    fail clearly if neither API supplies one.
    """

    instance_name = "num_key_value_heads" if key_value else "num_heads"
    config_name = "num_key_value_heads" if key_value else "num_attention_heads"
    value = getattr(attention, instance_name, None)
    if value is None:
        value = getattr(config, config_name, None)
    if value is None:
        raise AttributeError(
            f"Qwen3 attention/config exposes neither {instance_name} nor {config_name}"
        )
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"Qwen3 {config_name} must be positive, got {resolved}")
    return resolved


def _run_cached_trace(
    decoder: Any,
    model: Any,
    token_ids: list[int],
    prefix_count: int,
    trace_layer: int,
    torch_module: Any,
    modeling_qwen3: Any,
    trace_all_layers: bool = False,
) -> tuple[dict[str, dict[str, object]], dict[str, Any], dict[str, object]]:
    """Run an official DynamicCache prefill and capture one real decode layer.

    The prefix and successor both enter ``StreamDecoder.feed`` and therefore
    the official Qwen3 forward/cache/attention backend.  Hooks only observe
    tensors; they never reconstruct attention or replace the model output.
    """

    from transformers.cache_utils import DynamicCache

    prefix_ids = [int(token_id) for token_id in token_ids[:prefix_count]]
    current_ids = [int(token_ids[prefix_count])]
    decoder.cache = DynamicCache()
    decoder.feed(decoder.embed_tokens(prefix_ids))
    if decoder.get_cache_length() != prefix_count:
        raise RuntimeError(
            "official DynamicCache prefix length mismatch: "
            f"expected {prefix_count}, got {decoder.get_cache_length()}"
        )

    layer = model.model.layers[trace_layer]
    attention = layer.self_attn
    num_q_heads = _resolve_attention_heads(attention, model.config, key_value=False)
    num_kv_heads = _resolve_attention_heads(attention, model.config, key_value=True)
    prefix = f"layer_{trace_layer:02d}_"
    trace: dict[str, dict[str, object]] = {}
    arrays: dict[str, Any] = {}

    def record(name: str, value: object) -> Any:
        assert_trace_tensor(torch_module, name, value)
        _shape, array, digest = _summary(value)
        trace[name] = digest
        arrays[name] = array
        return value

    def hook_output(name: str, transform: Any | None = None):
        def hook(_module: Any, _inputs: Any, output: object) -> None:
            value = _first_tensor(torch_module, output)
            if value is None:
                raise RuntimeError(f"Qwen3 hook {name} did not receive a tensor")
            record(name, transform(value) if transform is not None else value)

        return hook

    def hook_input(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            if not inputs:
                raise RuntimeError(f"Qwen3 hook {name} had no positional input")
            record(name, inputs[0])

        return hook

    handles: list[Any] = []
    handles.append(layer.register_forward_pre_hook(hook_input(prefix + "input")))
    handles.append(layer.register_forward_hook(hook_output(prefix + "layer_out")))
    if trace_all_layers:
        for layer_index, traced_layer in enumerate(model.model.layers):
            if layer_index == trace_layer:
                continue
            handles.append(
                traced_layer.register_forward_hook(
                    hook_output(f"layer_{layer_index:02d}_layer_out")
                )
            )
    handles.append(
        layer.input_layernorm.register_forward_hook(hook_output(prefix + "input_norm"))
    )
    handles.append(layer.self_attn.q_proj.register_forward_hook(hook_output(prefix + "q_proj")))
    handles.append(layer.self_attn.k_proj.register_forward_hook(hook_output(prefix + "k_proj")))
    handles.append(layer.self_attn.v_proj.register_forward_hook(hook_output(prefix + "v_proj")))
    handles.append(
        layer.self_attn.q_norm.register_forward_hook(
            hook_output(
                prefix + "q",
                lambda value: _heads_first(
                    torch_module, value, num_q_heads
                ),
            )
        )
    )
    handles.append(
        layer.self_attn.k_norm.register_forward_hook(
            hook_output(
                prefix + "k",
                lambda value: _heads_first(
                    torch_module, value, num_kv_heads
                ),
            )
        )
    )
    handles.append(
        layer.self_attn.v_proj.register_forward_hook(
            hook_output(
                prefix + "v",
                lambda value: _heads_first(
                    torch_module, value, num_kv_heads
                ),
            )
        )
    )
    handles.append(attention.o_proj.register_forward_pre_hook(hook_input(prefix + "sdpa_merge")))
    handles.append(attention.o_proj.register_forward_hook(hook_output(prefix + "o_proj")))
    handles.append(
        layer.post_attention_layernorm.register_forward_pre_hook(
            hook_input(prefix + "attn_residual")
        )
    )
    handles.append(
        layer.post_attention_layernorm.register_forward_hook(hook_output(prefix + "post_norm"))
    )
    handles.append(layer.mlp.gate_proj.register_forward_hook(hook_output(prefix + "gate")))
    handles.append(layer.mlp.up_proj.register_forward_hook(hook_output(prefix + "up")))
    if hasattr(layer.mlp, "act_fn") and hasattr(layer.mlp.act_fn, "register_forward_hook"):
        handles.append(layer.mlp.act_fn.register_forward_hook(hook_output(prefix + "silu_gate")))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook_input(prefix + "mul")))
    handles.append(layer.mlp.down_proj.register_forward_hook(hook_output(prefix + "down")))

    attention_globals = getattr(
        getattr(attention.forward, "__func__", attention.forward), "__globals__", {}
    )
    active_attention = False
    original_attention_forward = attention.forward
    original_rotary = attention_globals.get("apply_rotary_pos_emb")
    original_eager = attention_globals.get("eager_attention_forward")
    attention_functions = attention_globals.get("ALL_ATTENTION_FUNCTIONS")
    attn_implementation = str(
        getattr(model.config, "_attn_implementation", "unknown")
    )
    patched_mapping = False
    old_mapping_value: Any = None

    def capture_rotary(*args: Any, **kwargs: Any) -> Any:
        result = original_rotary(*args, **kwargs)
        if active_attention and isinstance(result, (tuple, list)) and len(result) >= 2:
            if prefix + "q_rope" not in trace:
                record(prefix + "q_rope", result[0])
            if prefix + "k_rope" not in trace:
                record(prefix + "k_rope", result[1])
        return result

    def capture_attention(*args: Any, **kwargs: Any) -> Any:
        tensors = [item for item in args if torch_module.is_tensor(item)]
        query = kwargs.get("query")
        if query is None:
            query = kwargs.get("query_states")
        key = kwargs.get("key")
        if key is None:
            key = kwargs.get("key_states")
        value = kwargs.get("value")
        if value is None:
            value = kwargs.get("value_states")
        query = query if torch_module.is_tensor(query) else (tensors[0] if len(tensors) > 0 else None)
        key = key if torch_module.is_tensor(key) else (tensors[1] if len(tensors) > 1 else None)
        value = value if torch_module.is_tensor(value) else (tensors[2] if len(tensors) > 2 else None)
        if active_attention:
            if query is not None and prefix + "q_rope" not in trace:
                record(prefix + "q_rope", query)
            if key is not None:
                if prefix + "k_rope" not in trace:
                    record(prefix + "k_rope", key[..., -1:, :])
                record(prefix + "cache_k", key)
            if value is not None:
                record(prefix + "cache_v", value)
        return original_attention_interface(*args, **kwargs)

    def wrapped_attention_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal active_attention
        bound_cache = kwargs.get("past_key_values")
        if bound_cache is None:
            bound_cache = kwargs.get("past_key_value")
        if bound_cache is None:
            try:
                import inspect

                bound = inspect.signature(original_attention_forward).bind_partial(*args, **kwargs)
                for name in ("past_key_values", "past_key_value", "cache"):
                    if name in bound.arguments:
                        bound_cache = bound.arguments[name]
                        break
            except (TypeError, ValueError):
                pass
        if bound_cache is None:
            for item in args:
                if _looks_like_cache(item):
                    bound_cache = item
                    break
        keys, values = _cache_layer_pair(bound_cache, trace_layer)
        if keys is not None:
            record(prefix + "prefix_cache_k", keys)
        if values is not None:
            record(prefix + "prefix_cache_v", values)
        active_attention = True
        try:
            return original_attention_forward(*args, **kwargs)
        finally:
            active_attention = False

    original_attention_interface: Any = None
    try:
        attention.forward = wrapped_attention_forward
        if original_rotary is not None:
            attention_globals["apply_rotary_pos_emb"] = capture_rotary
        if original_eager is not None:
            attention_globals["eager_attention_forward"] = capture_attention
        if attention_functions is not None:
            try:
                original_attention_interface = attention_functions[attn_implementation]
                attention_functions[attn_implementation] = capture_attention
                old_mapping_value = original_attention_interface
                patched_mapping = True
            except (KeyError, TypeError, AttributeError):
                pass
        if original_attention_interface is None:
            original_attention_interface = original_eager
        if original_attention_interface is None:
            raise RuntimeError(
                f"cannot locate official Qwen3 attention backend {attn_implementation}"
            )
        decoder.feed(decoder.embed_token(current_ids[0]))
    finally:
        attention.forward = original_attention_forward
        if original_rotary is not None:
            attention_globals["apply_rotary_pos_emb"] = original_rotary
        if original_eager is not None:
            attention_globals["eager_attention_forward"] = original_eager
        if patched_mapping:
            attention_functions[attn_implementation] = old_mapping_value
        for handle in handles:
            handle.remove()

    required = (
        "input", "input_norm", "q_proj", "k_proj", "v_proj", "q", "k", "v",
        "q_rope", "k_rope", "prefix_cache_k", "prefix_cache_v",
        "cache_k", "cache_v", "sdpa_merge", "o_proj",
        "attn_residual", "post_norm", "gate", "up", "silu_gate", "mul",
        "down", "layer_out",
    )
    missing = [prefix + name for name in required if prefix + name not in trace]
    if missing:
        raise RuntimeError("cached Qwen3 trace did not capture: " + ", ".join(missing))
    if trace_all_layers:
        missing_layers = [
            f"layer_{layer_index:02d}_layer_out"
            for layer_index in range(len(model.model.layers))
            if f"layer_{layer_index:02d}_layer_out" not in trace
        ]
        if missing_layers:
            raise RuntimeError(
                "cached Qwen3 trace did not capture all layer outputs: "
                + ", ".join(missing_layers)
            )
    if decoder.get_cache_length() != prefix_count + 1:
        raise RuntimeError(
            "official DynamicCache successor length mismatch: "
            f"expected {prefix_count + 1}, got {decoder.get_cache_length()}"
        )
    provenance = trace_provenance(torch_module)
    provenance.update(
        cached_trace_provenance(
            prefix_ids,
            current_ids,
            trace_layer=trace_layer,
            attn_implementation=attn_implementation,
            token_ids=token_ids,
        )
    )
    return trace, arrays, provenance


def main() -> None:
    args = parse_args()
    token_ids = resolve_trace_token_ids(args.token_count, args.token_ids)
    validate_cached_prefix_count(args.cached_prefix_count, len(token_ids))

    # Keep torchvision's optional fake NMS registration from aborting imports
    # in the pinned CPU-only environment.  This is numerically inert.
    import torch

    # StreamDecoder.feed and the duplex streaming path are no-grad inference
    # entry points.  Set this before loading or executing any embedding/layer
    # operation and fail closed if the process cannot honor that contract.
    enter_no_grad_inference(torch)

    compat = torch.library.Library("torchvision", "DEF")
    for name in ("nms", "qnms"):
        try:
            compat.define(f"{name}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except RuntimeError:
            pass

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_minicpm_o_llm_golden as golden
    from transformers.models.qwen3 import modeling_qwen3

    decoder, _tokenizer, _ids = golden.load_official_torch_decoder(
        args.source_model.expanduser().resolve(),
        args.mlx_bundle.expanduser().resolve(),
        args.upstream_source.expanduser().resolve(),
    )
    model = decoder.m
    validate_trace_layer(args.trace_layer, len(model.model.layers))
    if args.cached_prefix_count:
        cached_layer = 0 if args.trace_layer is None else args.trace_layer
        validate_trace_layer(cached_layer, len(model.model.layers))
        trace, arrays, provenance = _run_cached_trace(
            decoder,
            model,
            token_ids,
            args.cached_prefix_count,
            cached_layer,
            torch,
            modeling_qwen3,
            trace_all_layers=args.trace_all_layers,
        )
        args.output.expanduser().mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            (args.output / f"{name}.f32").write_bytes(array.tobytes(order="C"))
        summary = dict(trace)
        summary["provenance"] = provenance
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "output": str(args.output),
            **provenance,
        }, indent=2))
        return

    provenance = trace_provenance(torch)
    provenance.update(trace_input_provenance(token_ids))
    provenance["mode"] = "prefill"
    provenance["prefix_count"] = 0
    provenance["current_token_ids"] = [int(token_id) for token_id in token_ids]
    provenance["current_length"] = len(token_ids)
    provenance["attn_implementation"] = str(
        getattr(model.config, "_attn_implementation", "unknown")
    )
    # StreamDecoder embeds are already BF16, matching Swift's feed path.
    x = decoder.embed_tokens(token_ids).unsqueeze(0)
    layer = model.model.layers[0]
    attention = layer.self_attn
    trace: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}

    def record(name: str, value: object) -> torch.Tensor:
        assert_trace_tensor(torch, name, value)
        shape, array, digest = _summary(value)
        del shape
        trace[name] = digest
        arrays[name] = array
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value, device=x.device)

    if is_targeted_trace(args.trace_layer):
        # A targeted trace first evaluates the preceding blocks, then keeps
        # every operation in the requested block.  This makes it possible to
        # bisect a layer-output spike without rerunning or serializing all
        # intermediate tensors for the full 36-layer decoder.
        batch, length = x.shape[:2]
        rotary = model.model.rotary_emb
        position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        inv_expanded = rotary.inv_freq[None, :, None].float().expand(1, -1, 1)
        positions_expanded = position_ids[:, None, :].float()
        freqs = (inv_expanded @ positions_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * rotary.attention_scaling).to(dtype=x.dtype)
        sin = (emb.sin() * rotary.attention_scaling).to(dtype=x.dtype)
        cos_broadcast = cos.unsqueeze(1)
        sin_broadcast = sin.unsqueeze(1)

        def run_block(hidden: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
            attention = layer.self_attn
            input_norm = layer.input_layernorm(hidden)
            q_projection = attention.q_proj(input_norm)
            k_projection = attention.k_proj(input_norm)
            v_projection = attention.v_proj(input_norm)
            q_norm = attention.q_norm(
                q_projection.view(
                    batch, length, model.config.num_attention_heads, attention.head_dim
                )
            )
            k_norm = attention.k_norm(
                k_projection.view(
                    batch, length, model.config.num_key_value_heads, attention.head_dim
                )
            )
            v = v_projection.view(
                batch, length, model.config.num_key_value_heads, attention.head_dim
            )
            q = q_norm.transpose(1, 2)
            k = k_norm.transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos_broadcast + modeling_qwen3.rotate_half(q) * sin_broadcast
            k = k * cos_broadcast + modeling_qwen3.rotate_half(k) * sin_broadcast
            repeated_k = modeling_qwen3.repeat_kv(k, attention.num_key_value_groups)
            repeated_v = modeling_qwen3.repeat_kv(v, attention.num_key_value_groups)
            merged_heads = torch.nn.functional.scaled_dot_product_attention(
                q.contiguous(), repeated_k.contiguous(), repeated_v.contiguous(),
                dropout_p=0.0, scale=attention.scaling, is_causal=True,
            )
            attention_merged = merged_heads.transpose(1, 2).contiguous()
            residual = hidden + attention.o_proj(
                attention_merged.reshape(batch, length, -1)
            )
            post_norm = layer.post_attention_layernorm(residual)
            gate = layer.mlp.gate_proj(post_norm)
            up = layer.mlp.up_proj(post_norm)
            down = layer.mlp.down_proj(torch.nn.functional.silu(gate) * up)
            return residual + down

        hidden = x
        for prior_layer in model.model.layers[: args.trace_layer]:
            hidden = run_block(hidden, prior_layer)

        layer_index = args.trace_layer
        layer = model.model.layers[layer_index]
        attention = layer.self_attn
        prefix = f"layer_{layer_index:02d}_"
        record(prefix + "input", hidden)
        input_norm = record(prefix + "input_norm", layer.input_layernorm(hidden))
        q_projection = record(prefix + "q_proj", attention.q_proj(input_norm))
        k_projection = record(prefix + "k_proj", attention.k_proj(input_norm))
        v_projection = record(prefix + "v_proj", attention.v_proj(input_norm))
        q_norm = attention.q_norm(
            q_projection.view(batch, length, model.config.num_attention_heads, attention.head_dim)
        )
        k_norm = attention.k_norm(
            k_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
        )
        v = v_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
        q = record(prefix + "q", q_norm.transpose(1, 2))
        k = record(prefix + "k", k_norm.transpose(1, 2))
        v = record(prefix + "v", v.transpose(1, 2))
        q = record(prefix + "q_rope", q * cos_broadcast + modeling_qwen3.rotate_half(q) * sin_broadcast)
        k = record(prefix + "k_rope", k * cos_broadcast + modeling_qwen3.rotate_half(k) * sin_broadcast)
        repeated_k = modeling_qwen3.repeat_kv(k, attention.num_key_value_groups)
        repeated_v = modeling_qwen3.repeat_kv(v, attention.num_key_value_groups)
        merged_heads = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(), repeated_k.contiguous(), repeated_v.contiguous(),
            dropout_p=0.0, scale=attention.scaling, is_causal=True,
        )
        attention_merged = record(prefix + "sdpa_merge", merged_heads.transpose(1, 2).contiguous())
        output_projection = record(
            prefix + "o_proj", attention.o_proj(attention_merged.reshape(batch, length, -1))
        )
        residual = record(prefix + "attn_residual", hidden + output_projection)
        post_norm = record(prefix + "post_norm", layer.post_attention_layernorm(residual))
        gate = record(prefix + "gate", layer.mlp.gate_proj(post_norm))
        up = record(prefix + "up", layer.mlp.up_proj(post_norm))
        silu_gate = record(prefix + "silu_gate", torch.nn.functional.silu(gate))
        mul = record(prefix + "mul", silu_gate * up)
        down = record(prefix + "down", layer.mlp.down_proj(mul))
        record(prefix + "layer_out", residual + down)

        args.output.expanduser().mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            (args.output / f"{name}.f32").write_bytes(array.tobytes(order="C"))
        summary = dict(trace)
        summary["provenance"] = provenance
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "output": str(args.output),
            "attn_implementation": model.config._attn_implementation,
            "trace_layer": args.trace_layer,
            **provenance,
        }, indent=2))
        return

    record("embed", x)
    input_norm = record("input_norm", layer.input_layernorm(x))
    q_projection = record("q_proj", attention.q_proj(input_norm))
    k_projection = record("k_proj", attention.k_proj(input_norm))
    v_projection = record("v_proj", attention.v_proj(input_norm))
    batch, length = x.shape[:2]
    q_norm = attention.q_norm(
        q_projection.view(batch, length, model.config.num_attention_heads, attention.head_dim)
    )
    k_norm = attention.k_norm(
        k_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
    )
    v = v_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
    record("q_norm", q_norm)
    record("k_norm", k_norm)
    q = q_norm.transpose(1, 2)
    k = k_norm.transpose(1, 2)
    v = v.transpose(1, 2)
    record("q_norm_trans", q)
    record("k_norm_trans", k)
    position_ids = torch.arange(length, device=x.device).unsqueeze(0)
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq
    record("inv_freq", inv_freq)
    record("position_ids", position_ids.float())
    inv_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    positions_expanded = position_ids[:, None, :].float()
    freqs = (inv_expanded @ positions_expanded).transpose(1, 2)
    record("freqs", freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_base = emb.cos() * rotary.attention_scaling
    sin_base = emb.sin() * rotary.attention_scaling
    cos = cos_base.to(dtype=x.dtype)
    sin = sin_base.to(dtype=x.dtype)
    record("cos_bf16", cos)
    record("sin_bf16", sin)
    cos_broadcast = cos.unsqueeze(1)
    sin_broadcast = sin.unsqueeze(1)
    q_rotate = modeling_qwen3.rotate_half(q)
    k_rotate = modeling_qwen3.rotate_half(k)
    record("q_rotate_half", q_rotate)
    record("k_rotate_half", k_rotate)
    q_cos = q * cos_broadcast
    q_sin = q_rotate * sin_broadcast
    k_cos = k * cos_broadcast
    k_sin = k_rotate * sin_broadcast
    record("q_cos", q_cos)
    record("q_sin", q_sin)
    record("k_cos", k_cos)
    record("k_sin", k_sin)
    q = q_cos + q_sin
    k = k_cos + k_sin
    record("q_rope", q)
    record("k_rope", k)

    repeated_k = modeling_qwen3.repeat_kv(k, attention.num_key_value_groups)
    repeated_v = modeling_qwen3.repeat_kv(v, attention.num_key_value_groups)
    # Use the configured Qwen3 attention implementation, exactly as the
    # official model does for a no-cache block.  ``sdpa`` receives is_causal;
    # ``eager`` receives the equivalent additive upper-triangular mask.
    if model.config._attn_implementation == "eager":
        scores = torch.matmul(q, repeated_k.transpose(2, 3)) * attention.scaling
        scores = scores.masked_fill(
            torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1),
            float("-inf"),
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        merged_heads = torch.matmul(probabilities, repeated_v)
    else:
        merged_heads = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(), repeated_k.contiguous(), repeated_v.contiguous(),
            dropout_p=0.0, scale=attention.scaling, is_causal=True,
        )
    attention_merged = merged_heads.transpose(1, 2).contiguous()
    record("sdpa_merge", attention_merged)
    output_projection = record(
        "o_proj", attention.o_proj(attention_merged.reshape(batch, length, -1))
    )
    residual = record("residual", x + output_projection)
    post_norm = record("post_norm", layer.post_attention_layernorm(residual))
    gate = record("gate", layer.mlp.gate_proj(post_norm))
    up = record("up", layer.mlp.up_proj(post_norm))
    down = record("down", layer.mlp.down_proj(torch.nn.functional.silu(gate) * up))
    hidden = record("layer_out", residual + down)
    record("layer_00_out", hidden)

    # Continue the same four-token prefill through every remaining decoder
    # block.  Only the block outputs are retained for this compact drift
    # curve; the detailed layer-0 intermediates above identify the first
    # operation-level mismatch without producing a multi-gigabyte trace.
    for layer_index, layer in enumerate(model.model.layers[1:], start=1):
        attention = layer.self_attn
        input_norm = layer.input_layernorm(hidden)
        q_projection = attention.q_proj(input_norm)
        k_projection = attention.k_proj(input_norm)
        v_projection = attention.v_proj(input_norm)
        q_norm = attention.q_norm(
            q_projection.view(batch, length, model.config.num_attention_heads, attention.head_dim)
        )
        k_norm = attention.k_norm(
            k_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
        )
        v = v_projection.view(batch, length, model.config.num_key_value_heads, attention.head_dim)
        q = q_norm.transpose(1, 2)
        k = k_norm.transpose(1, 2)
        v = v.transpose(1, 2)
        q = (q * cos_broadcast) + (modeling_qwen3.rotate_half(q) * sin_broadcast)
        k = (k * cos_broadcast) + (modeling_qwen3.rotate_half(k) * sin_broadcast)
        repeated_k = modeling_qwen3.repeat_kv(k, attention.num_key_value_groups)
        repeated_v = modeling_qwen3.repeat_kv(v, attention.num_key_value_groups)
        if model.config._attn_implementation == "eager":
            scores = torch.matmul(q, repeated_k.transpose(2, 3)) * attention.scaling
            scores = scores.masked_fill(
                torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
                ),
                float("-inf"),
            )
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            merged_heads = torch.matmul(probabilities, repeated_v)
        else:
            merged_heads = torch.nn.functional.scaled_dot_product_attention(
                q.contiguous(), repeated_k.contiguous(), repeated_v.contiguous(),
                dropout_p=0.0, scale=attention.scaling, is_causal=True,
            )
        attention_merged = merged_heads.transpose(1, 2).contiguous()
        output_projection = attention.o_proj(attention_merged.reshape(batch, length, -1))
        residual = hidden + output_projection
        post_norm = layer.post_attention_layernorm(residual)
        gate = layer.mlp.gate_proj(post_norm)
        up = layer.mlp.up_proj(post_norm)
        down = layer.mlp.down_proj(torch.nn.functional.silu(gate) * up)
        hidden = residual + down
        record(f"layer_{layer_index:02d}_out", hidden)

    args.output.expanduser().mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        (args.output / f"{name}.f32").write_bytes(array.tobytes(order="C"))
    summary = dict(trace)
    summary["provenance"] = provenance
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "attn_implementation": model.config._attn_implementation,
        **provenance,
    }, indent=2))


if __name__ == "__main__":
    main()
