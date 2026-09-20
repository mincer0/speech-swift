#!/usr/bin/env python3
"""Export an official PyTorch MiniCPM-o semantic-TTS parity fixture.

The Swift semantic decoder is tested against the *official* MiniCPM-o TTS
implementation, not another copy of the MLX port.  This script imports the
fixed-source checkout under ``--upstream-repo``, instantiates only the
``MiniCPMTTS`` module, and reads the TTS tensors directly from the final
official safetensors shard.  No LLM, vision, or audio encoder is constructed.

The fixture records deterministic synthetic LLM hidden states, condition
embeddings, first-step logits, and a three-chunk strict trace.  Hidden states
can later be replaced by captured duplex states without changing the protocol
or metadata schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


UPSTREAM_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
MODELING_SHA256 = "877f20cea7282bb0d05684393a1580646d1f264a2299a42f82467fd99f4028af"
UNIFIED_MODELING_SHA256 = (
    "9fceb75f969cfafc6381adf2b59255529be7fb34d64ad784b8417d224ffbabb9"
)

# Keep the operator library alive until torchvision has imported.  Torch removes
# custom schemas when the Library object is garbage-collected on some releases.
_TORCHVISION_LIBRARY: Any = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="official MiniCPM-o directory containing config.json and model.safetensors.index.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "Tests/MiniCPMTTSSemanticTests/Fixtures/minicpm-tts-condition-oracle.safetensors",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=Path("/tmp/MiniCPM-o-Demo"),
        help="fixed-source checkout containing MiniCPMO45/modeling_minicpmo.py",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="official PyTorch device; MPS is preferred on Apple Silicon",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="dtype for official TTS weights and hidden states",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="do not fail when the checkout is not the pinned source",
    )
    return parser.parse_args()


def source_hashes(root: Path) -> dict[str, Any]:
    modeling = root / "MiniCPMO45/modeling_minicpmo.py"
    unified = root / "MiniCPMO45/modeling_minicpmo_unified.py"
    result: dict[str, Any] = {
        "upstream_commit": None,
        "modeling_sha256": sha256(modeling) if modeling.is_file() else None,
        "unified_modeling_sha256": sha256(unified) if unified.is_file() else None,
    }
    try:
        result["upstream_commit"] = (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def torch_dependencies():
    """Import torch and safetensors after argument parsing.

    Transformers imports torchvision on some versions.  The local environment
    carries a CPU-only torchvision build without its custom NMS operators; a
    small schema declaration keeps the official Llama import usable without
    changing model code or installing a second Torch build.
    """

    global _TORCHVISION_LIBRARY
    import torch

    try:
        library = torch.library.Library("torchvision", "DEF")
        for name in ("nms", "qnms"):
            try:
                library.define(
                    f"{name}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
                )
            except RuntimeError:
                pass
        _TORCHVISION_LIBRARY = library
    except Exception:
        # A compatible torchvision or an already-defined namespace needs no
        # workaround.  Import errors below still provide the actionable cause.
        pass
    from safetensors import safe_open
    from safetensors.torch import save_file

    return torch, safe_open, save_file


def import_official_tts(upstream_repo: Path):
    # MiniCPMO45 is a namespace package in the upstream release (no
    # __init__.py), so putting the fixed checkout itself on sys.path preserves
    # all relative imports while selecting it rather than local model copies.
    root = upstream_repo.resolve()
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from MiniCPMO45.configuration_minicpmo import MiniCPMTTSConfig
        from MiniCPMO45.modeling_minicpmo import MiniCPMTTS
    except Exception as exc:  # pragma: no cover - depends on Torch install
        raise SystemExit(
            "unable to import official MiniCPMO45 TTS; use the project "
            "PyTorch environment and verify --upstream-repo"
        ) from exc
    # Never silently import the user's model checkout or a stale installed
    # package.  The oracle is only valid when both source modules come from
    # the pinned checkout requested by this invocation.
    import MiniCPMO45.configuration_minicpmo as official_config
    import MiniCPMO45.modeling_minicpmo as official_modeling

    for module in (official_config, official_modeling):
        module_path = Path(module.__file__).resolve()
        try:
            module_path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                "official TTS import escaped --upstream-repo: "
                + str(module_path)
            ) from exc
    return MiniCPMTTSConfig, MiniCPMTTS


def select_tts_shard(model_dir: Path) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing official safetensors index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = {
        name
        for key, name in index.get("weight_map", {}).items()
        if key.startswith("tts.")
    }
    if len(names) != 1:
        raise SystemExit(
            "expected all official TTS weights in one shard; found "
            + ", ".join(sorted(names))
        )
    shard = model_dir / next(iter(names))
    if not shard.is_file():
        raise SystemExit(f"missing TTS shard: {shard}")
    return shard


def load_official_tts(
    model_dir: Path,
    upstream_repo: Path,
    device: Any,
    dtype: Any,
    torch: Any,
    safe_open: Any,
):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tts_config = config.get("tts_config", {})
    MiniCPMTTSConfig, MiniCPMTTS = import_official_tts(upstream_repo)
    tts = MiniCPMTTS(
        config=MiniCPMTTSConfig(**tts_config),
        audio_tokenizer=None,
    )
    tts.to(device=device, dtype=dtype)

    shard = select_tts_shard(model_dir)
    state: dict[str, Any] = {}
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("tts."):
                # MiniCPMTTS is the module under the upstream MiniCPMO root;
                # strip that root before load_state_dict.
                state[key.removeprefix("tts.")] = handle.get_tensor(key)
    if not state:
        raise SystemExit(f"no tts.* tensors found in {shard}")
    state = {key: value.to(device=device, dtype=dtype) for key, value in state.items()}
    missing, unexpected = tts.load_state_dict(state, strict=False)
    # embed_tokens is unused by inputs_embeds generation and may be omitted by
    # converted bundles, but every other missing key indicates a bad shard.
    allowed_missing = {"model.embed_tokens.weight"}
    missing_set = set(missing) - allowed_missing
    if missing_set or unexpected:
        raise RuntimeError(
            "official TTS state mismatch: missing="
            + repr(sorted(missing_set))
            + " unexpected="
            + repr(sorted(unexpected))
        )
    del state
    tts.eval()
    return tts, shard, config


def build_condition(tts: Any, hidden: Any, token_ids: list[int], torch: Any) -> Any:
    ids = torch.tensor(token_ids, dtype=torch.long, device=hidden.device)
    text = tts.emb_text(ids)
    projected = tts.projector_semantic(hidden)
    projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
    audio_bos = tts.emb_text(
        torch.tensor([tts.audio_bos_token_id], dtype=torch.long, device=hidden.device)
    )
    return torch.cat([text + projected, audio_bos], dim=0).unsqueeze(0)


def save_fixture(output: Path, metadata_path: Path, arrays: dict[str, Any], metadata: dict[str, Any], save_file: Any, torch: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_arrays = {
        key: value.detach().to(device="cpu").contiguous() for key, value in arrays.items()
    }
    save_file(
        cpu_arrays,
        str(output),
        metadata={
            "fixture_format": str(metadata.get("format", 2)),
            "oracle_backend": "official_pytorch",
            "upstream_commit": UPSTREAM_COMMIT,
            "tts_bundle_sha256": str(metadata["source"].get("tts_bundle_sha256", "")),
        },
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    upstream_repo = args.upstream_repo.expanduser().resolve()
    config_file = model_dir / "config.json"
    if not model_dir.is_dir() or not config_file.is_file():
        raise SystemExit(f"official MiniCPM-o model directory is incomplete: {model_dir}")

    observed = source_hashes(upstream_repo)
    source_matches = (
        observed["upstream_commit"] == UPSTREAM_COMMIT
        and observed["modeling_sha256"] == MODELING_SHA256
        and observed["unified_modeling_sha256"] == UNIFIED_MODELING_SHA256
    )
    if not source_matches and not args.allow_source_mismatch:
        raise SystemExit(
            "upstream source is not the pinned MiniCPM-o revision; pass "
            "--allow-source-mismatch only for an intentional regeneration"
        )

    torch, safe_open, save_file = torch_dependencies()
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    tts, shard, config = load_official_tts(
        model_dir, upstream_repo, device, dtype, torch, safe_open
    )
    arrays: dict[str, Any] = {}
    audio_bos = tts.emb_text(
        torch.tensor([tts.audio_bos_token_id], dtype=torch.long, device=device)
    )
    arrays["empty.expected"] = audio_bos.unsqueeze(0)

    cases = [
        ("filtered", [22, 33, 44], [151706, 22, 33, 44], 3),
        ("control", [55, 151717], [55, 151717], 2),
        ("chunk0", [22, 33, 44], [22, 33, 44], 3),
        ("chunk1", [55, 66, 77, 88], [55, 66, 77, 88], 4),
        ("chunk2", [99, 151717], [99, 151717], 2),
    ]
    case_metadata: dict[str, Any] = {}
    with torch.inference_mode():
        for name, tokens, raw_tokens, hidden_count in cases:
            values = np_rng.standard_normal((hidden_count, 4096)).astype(np.float32)
            hidden = torch.from_numpy(values).to(device=device, dtype=dtype)
            condition = build_condition(tts, hidden, tokens, torch)
            arrays[f"{name}.hidden"] = hidden
            arrays[f"{name}.expected"] = condition
            case_metadata[name] = {
                "tokens": tokens,
                "raw_tokens": raw_tokens,
                "hidden_key": f"{name}.hidden",
                "expected_key": f"{name}.expected",
                "condition_shape": list(condition.shape),
                "dtype": str(condition.dtype),
            }

        # Numerical first-step oracle.  This bypasses sampling and validates
        # the Llama attention/RoPE path before an argmax tie can hide a bug.
        first_condition = arrays["chunk0.expected"]
        first_output = tts.model(
            inputs_embeds=first_condition,
            use_cache=True,
            return_dict=True,
        )
        arrays["chunk0.first_logits"] = tts.head_code[0](
            first_output.last_hidden_state[:, -1, :]
        ).reshape(-1).float()

        chunk_definitions = [
            {
                "name": name,
                "tokens": tokens,
                "hidden_key": f"{name}.hidden",
                "expected_key": f"{name}.expected",
            }
            for name, tokens, _raw, _count in cases[-3:]
        ]
        cache = None
        trace: list[dict[str, Any]] = []
        text_start = 0
        # Official generate_chunk uses multinomial sampling.  A tiny positive
        # temperature is the numerically stable equivalent of greedy decoding
        # and keeps the source loop untouched; manual_seed makes ties replayable.
        temperature = torch.tensor([1e-6], dtype=torch.float32, device=device)
        eos = torch.tensor(tts.num_audio_tokens - 1, dtype=torch.long, device=device)
        for index, definition in enumerate(chunk_definitions):
            condition = arrays[definition["expected_key"]]
            returned, cache = tts.generate_chunk(
                inputs_embeds=condition,
                temperature=temperature,
                repetition_penalty=1.0,
                eos_token=eos,
                force_no_stop=True,
                max_new_token=26,
                min_new_tokens=0 if index in (0, 2) else 26,
                past_key_values=cache,
                text_start_pos=text_start,
            )
            committed = returned[0, :, 0].detach().to("cpu").tolist()
            committed = [int(value) for value in committed]
            condition_length = int(condition.shape[1])
            trace.append(
                {
                    "index": index,
                    "condition_length": condition_length,
                    "text_start": text_start,
                    "next_text_start": text_start + condition_length + len(committed),
                    "sampled_count": 26,
                    "returned_count": len(committed),
                    "stable_prefix": committed[:4] if index == 0 else [],
                    "committed": committed,
                    "pending": None,
                    "strict_pending": None,
                    "cache_offset": int(cache[0][0].shape[2]),
                    "is_first": index == 0,
                    "is_final": index == len(chunk_definitions) - 1,
                    "force_flush": True,
                    "ended_by_eos": False,
                }
            )
            text_start = trace[-1]["next_text_start"]

    source = {
        "model_id": model_dir.name,
        "tts_shard": shard.name,
        "config_sha256": sha256(config_file),
        "tts_bundle_sha256": sha256(shard),
        "upstream_commit": UPSTREAM_COMMIT,
        "modeling_sha256": MODELING_SHA256,
        "unified_modeling_sha256": UNIFIED_MODELING_SHA256,
        "observed_upstream_commit": observed["upstream_commit"],
        "observed_modeling_sha256": observed["modeling_sha256"],
        "observed_unified_modeling_sha256": observed["unified_modeling_sha256"],
        "condition_helper": "MiniCPMO._convert_results_to_tts_input",
        "condition_verified_with_fixed_source": source_matches,
        "oracle_backend": "official_pytorch",
        "oracle_dtype": args.dtype,
        "device": str(device),
        "temperature": 1e-6,
        "torch": torch.__version__,
    }
    metadata = {
        "format": 2,
        "source": source,
        "condition_file": args.output.name,
        "cases": case_metadata,
        "chunks": chunk_definitions,
        "trace": trace,
        "trace_contract": {
            "max_new_token": 26,
            "middle_min_new_tokens": 26,
            "sampled_per_full_chunk": 26,
            "returned_per_full_chunk": 25,
            "committed_per_full_chunk": 25,
            "pending_per_full_chunk": 0,
            "token_identity": "stable_prefix_only_after_bf16_argmax_tie",
            "logits_oracle_file": args.output.name,
            "logits_oracle_key": "chunk0.first_logits",
            "first_protocol_token_filtered": True,
            "condition_special_tokens": ["audio_bos"],
            "optional_condition_special_tokens": ["text_eos"],
            "force_flush_cadence": "first middle final",
            "eos_token": int(tts.num_audio_tokens - 1),
            "reset_after_final": "explicit session.finishTurn() after token2wav flush",
            "force_no_stop": True,
            "repetition_penalty": 1.0,
        },
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    save_fixture(args.output, metadata_path, arrays, metadata, save_file, torch)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_path),
                "oracle_backend": "official_pytorch",
                "device": str(device),
                "arrays": len(arrays),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
