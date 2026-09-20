#!/usr/bin/env python3
"""Convert MiniCPM's S3Tokenizer-v2 ONNX initializers to MLX safetensors.

The converter is intentionally an offline, ONNX-only tool.  It resolves
initializer inputs by the semantic node scopes embedded in the exported graph
(``/blocks.i/attn/query/MatMul`` etc.) and transposes ONNX MatMul weights from
``[in, out]`` to the MLX/PyTorch Linear layout ``[out, in]``.  Runtime Swift
code never imports this script, Torch, or ONNX Runtime.

The output key schema is the one consumed by ``CosyVoiceWeightLoader``'s
``SpeechTokenizerModel`` loader.  Use ``--dry-run`` first to verify that all
102 source initializers are mapped before writing the ~495 MiB bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


MANIFEST_SCHEMA = "minicpm-o-token2wav-mlx/v1"
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
# Compatibility alias: this name historically identified the Demo commit.
PINNED_UPSTREAM_COMMIT = DEMO_IMPLEMENTATION_COMMIT
DEFAULT_SOURCE_MODEL = "OpenBMB/MiniCPM-o-4_5"
COMPONENT_ID = "minicpm-o-4_5.token2wav.speech-tokenizer"

# This matches ``SpeechTokenizerConfig.miniCPMV2`` in the Swift runtime.  It
# is emitted in the combined token2wav config so a future loader does not
# have to infer the ONNX variant from a filename.
TOKENIZER_CONFIGURATION: dict[str, Any] = {
    "variant": "s3tokenizer-v2",
    "n_mels": 128,
    "n_audio_state": 1280,
    "n_audio_head": 20,
    "n_audio_layer": 6,
    "codebook_levels": 3,
    "codebook_channels": 8,
    "fsmn_kernel_size": 31,
    "subsample_stride1": 2,
    "subsample_stride2": 2,
    "rope_base": 10000,
    "rope_max_seq_len": 2048,
    "attention_layer_norm_epsilon": 1e-6,
    "fsq_uses_affine_mapping": True,
    "fsq_clamp_epsilon": 1e-6,
}


def _normal(path: str) -> str:
    return path.strip("/")


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_set_sha256(tensors: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        record = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": [int(dim) for dim in value.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _update_bundle_config(
    output_dir: Path,
    *,
    source_model: str,
    source_revision: str,
) -> None:
    config_path = output_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing token2wav config: {config_path}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"token2wav config must be a JSON object: {config_path}")
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
    components["speech_tokenizer"] = dict(TOKENIZER_CONFIGURATION)
    config["components"] = components
    _write_json(config_path, config)


def _update_manifest(
    output_dir: Path,
    *,
    source_basename: str,
    source_model: str,
    source_revision: str,
    component_manifest: Mapping[str, Any],
) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing token2wav manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"token2wav manifest must be a JSON object: {manifest_path}")
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
            "source_basename": source_basename,
        }
    )
    components = manifest.get("components", {})
    if not isinstance(components, dict):
        components = {}
    components["speech_tokenizer"] = dict(component_manifest)
    manifest["components"] = components
    _write_json(manifest_path, manifest)


def _find_node(graph: onnx.GraphProto, suffix: str) -> onnx.NodeProto:
    matches = [node for node in graph.node if _normal(node.name).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one node ending {suffix!r}, got {len(matches)}")
    return matches[0]


def _input_initializer(
    node: onnx.NodeProto, initializers: dict[str, np.ndarray], *, index: int | None = None
) -> tuple[str, np.ndarray]:
    inputs: Iterable[str]
    if index is None:
        inputs = node.input
    else:
        inputs = [node.input[index]]
    matches = [(name, initializers[name]) for name in inputs if name in initializers]
    if len(matches) != 1:
        raise ValueError(
            f"node {node.name!r} expected one initializer input, got "
            f"{[name for name, _ in matches]}"
        )
    return matches[0]


def convert(path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {
        tensor.name: numpy_helper.to_array(tensor).astype(np.float32, copy=False)
        for tensor in graph.initializer
    }
    target: dict[str, np.ndarray] = {}

    # The two subsampling convolutions are named directly in the graph.
    for name in ("conv1", "conv2"):
        conv = _find_node(graph, f"{name}/Conv")
        _, weight = _input_initializer(conv, initializers, index=1)
        _, bias = _input_initializer(conv, initializers, index=2)
        target[f"encoder.{name}.weight"] = weight.copy()
        target[f"encoder.{name}.bias"] = bias.copy()

    for block in range(6):
        prefix = f"blocks.{block}"
        graphPrefix = f"blocks.{block}"

        fsmn = _find_node(graph, f"{graphPrefix}/attn/fsmn_block/Conv")
        _, fsmnWeight = _input_initializer(fsmn, initializers, index=1)
        target[f"encoder.{prefix}.attn.fsmn_block.weight"] = fsmnWeight.copy()

        for ln in ("attn_ln", "mlp_ln"):
            node = _find_node(graph, f"{graphPrefix}/{ln}/LayerNormalization")
            _, scale = _input_initializer(node, initializers, index=1)
            _, bias = _input_initializer(node, initializers, index=2)
            target[f"encoder.{prefix}.{ln}.weight"] = scale.copy()
            target[f"encoder.{prefix}.{ln}.bias"] = bias.copy()

        for projection in ("query", "key", "value", "out"):
            scope = f"{graphPrefix}/attn/{projection}"
            matmul = _find_node(graph, f"{scope}/MatMul")
            _, weight = _input_initializer(matmul, initializers, index=1)
            target[f"encoder.{prefix}.attn.{projection}.weight"] = weight.T.copy()
            # Key is the only bias-free attention projection in S3Tokenizer v2.
            if projection != "key":
                add = _find_node(graph, f"{scope}/Add")
                _, bias = _input_initializer(add, initializers)
                target[f"encoder.{prefix}.attn.{projection}.bias"] = bias.copy()

        for source, destination in (("mlp/mlp.0", "0"), ("mlp/mlp.2", "2")):
            scope = f"{graphPrefix}/{source}"
            matmul = _find_node(graph, f"{scope}/MatMul")
            _, weight = _input_initializer(matmul, initializers, index=1)
            add = _find_node(graph, f"{scope}/Add")
            _, bias = _input_initializer(add, initializers)
            target[f"encoder.{prefix}.mlp.{destination}.weight"] = weight.T.copy()
            target[f"encoder.{prefix}.mlp.{destination}.bias"] = bias.copy()

    # The ONNX exporter calls this projection `project_in`; the MLX FSQ
    # implementation follows the source module's `_codebook.project_down`.
    project = _find_node(graph, "quantizer/project_in/MatMul")
    _, projectWeight = _input_initializer(project, initializers, index=1)
    target["quantizer._codebook.project_down.weight"] = projectWeight.T.copy()
    # This Add is scoped to quantizer.project_in in the exported graph.
    projectAdd = _find_node(graph, "quantizer/project_in/Add")
    _, projectBias = _input_initializer(projectAdd, initializers)
    target["quantizer._codebook.project_down.bias"] = projectBias.copy()

    if len(target) != 102:
        raise ValueError(f"mapped {len(target)} tensors; expected all 102 ONNX initializers")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Stable source model id recorded in metadata (never a local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=MODEL_REPOSITORY_REVISION,
        help=(
            "Model repository revision for source weights (default: "
            f"{MODEL_REPOSITORY_REVISION})"
        ),
    )
    args = parser.parse_args()
    source_revision = args.source_revision or MODEL_REPOSITORY_REVISION
    onnx_path = args.onnx_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    tensors = convert(onnx_path)
    component_config = dict(TOKENIZER_CONFIGURATION)
    component_manifest = {
        "component_id": COMPONENT_ID,
        "component": "speech_tokenizer",
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "weights": {
            "path": output_path.name,
            "kind": "single_file",
        },
        "source": {
            "onnx": {
                "path": onnx_path.name,
                "bytes": onnx_path.stat().st_size,
                "sha256": _sha256_file(onnx_path),
            },
        },
        "converter": {
            "path": Path(__file__).name,
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "tensor_count": len(tensors),
        "tensor_set_sha256": _tensor_set_sha256(tensors),
        "config": component_config,
    }
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(output_path),
            metadata={
                "format": "mlx",
                "schema": MANIFEST_SCHEMA,
                "component": COMPONENT_ID,
                "source_model": args.source_model,
                "source_revision": source_revision,
                "revision_kind": "model_repository",
                "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
                "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            },
        )
        component_manifest["weights"].update(
            {
                "bytes": output_path.stat().st_size,
                "sha256": _sha256_file(output_path),
            }
        )
        _update_bundle_config(
            output_path.parent,
            source_model=args.source_model,
            source_revision=source_revision,
        )
        _update_manifest(
            output_path.parent,
            source_basename=onnx_path.parent.name,
            source_model=args.source_model,
            source_revision=source_revision,
            component_manifest=component_manifest,
        )
        print(
            json.dumps(
                {
                    "component_id": COMPONENT_ID,
                    "tensor_count": len(tensors),
                    "output": output_path.name,
                    "manifest": "manifest.json",
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        # Dry-run output intentionally contains no local path.  It is safe to
        # paste into CI logs while still exposing the source/artifact identity.
        print(
            json.dumps(
                {
                    "component_id": COMPONENT_ID,
                    "source_model": args.source_model,
                    "source_revision": source_revision,
                    "revision_kind": "model_repository",
                    "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
                    "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
                    "source": {
                        "path": onnx_path.name,
                        "bytes": onnx_path.stat().st_size,
                        "sha256": _sha256_file(onnx_path),
                    },
                    "tensor_count": len(tensors),
                    "tensor_set_sha256": component_manifest["tensor_set_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
