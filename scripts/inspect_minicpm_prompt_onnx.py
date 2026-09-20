#!/usr/bin/env python3
"""Deterministically inspect MiniCPM-o Token2Wav prompt ONNX graphs.

This utility intentionally depends only on ``onnx`` and the Python standard
library.  It does not import Torch, onnx2torch, or ONNX Runtime.  The JSON
output is the migration manifest for the native MLX path: graph I/O shapes,
every operator/attribute, and every initializer's exact dtype/shape/byte
checksum are retained so a converted safetensors bundle can be audited
against the source ONNX file.

Example (the model's bundled assets):

  python scripts/inspect_minicpm_prompt_onnx.py \
    models/MiniCPM-o-4_5/assets/token2wav/speech_tokenizer_v2_25hz.onnx \
    models/MiniCPM-o-4_5/assets/token2wav/campplus.onnx \
    --json-out docs/models/minicpm-prompt-onnx-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, helper


def _dim(value: Any) -> Any:
    if value.dim_value:
        return int(value.dim_value)
    if value.dim_param:
        return value.dim_param
    return "?"


def _value_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
    return {
        "name": value.name,
        "elem_type": int(tensor.elem_type),
        "elem_type_name": dtype.name,
        "shape": [_dim(dim) for dim in tensor.shape.dim],
    }


def _attribute(value: onnx.AttributeProto) -> dict[str, Any]:
    result: dict[str, Any] = {"name": value.name, "type": int(value.type)}
    kind = value.type
    if kind == AttributeProto.INT:
        result["value"] = int(value.i)
    elif kind == AttributeProto.FLOAT:
        result["value"] = float(value.f)
    elif kind == AttributeProto.STRING:
        result["value"] = value.s.decode("utf-8", errors="replace")
    elif kind == AttributeProto.INTS:
        result["value"] = [int(item) for item in value.ints]
    elif kind == AttributeProto.FLOATS:
        result["value"] = [float(item) for item in value.floats]
    elif kind == AttributeProto.STRINGS:
        result["value"] = [item.decode("utf-8", errors="replace") for item in value.strings]
    elif kind == AttributeProto.TENSOR:
        result["tensor_name"] = value.t.name
        result["tensor_dims"] = [int(dim) for dim in value.t.dims]
        result["tensor_data_type"] = int(value.t.data_type)
    else:
        # Graph/sparse/tensor attributes are uncommon in these two files.  Do
        # not silently discard them if a future export starts using one.
        result["summary"] = str(value)
    return result


def _initializer(value: onnx.TensorProto) -> dict[str, Any]:
    # ``raw_data`` is canonical for these exports; serialized_content also
    # handles packed typed fields if an ONNX exporter omitted raw_data.
    payload = value.raw_data or value.SerializeToString()
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(value.data_type))
    return {
        "name": value.name,
        "data_type": int(value.data_type),
        "data_type_name": dtype.name,
        "dims": [int(dim) for dim in value.dims],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def inspect(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    op_counts: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(graph.node):
        op = f"{node.domain + ':' if node.domain else ''}{node.op_type}"
        op_counts[op] = op_counts.get(op, 0) + 1
        nodes.append(
            {
                "index": index,
                "name": node.name,
                "op_type": node.op_type,
                "domain": node.domain,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": [_attribute(attribute) for attribute in node.attribute],
            }
        )

    return {
        "path": str(path.resolve()),
        "file_bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ir_version": int(model.ir_version),
        "opset_import": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
        "producer": {
            "name": model.producer_name,
            "version": model.producer_version,
            "domain": model.domain,
        },
        "inputs": [_value_info(value) for value in graph.input],
        "outputs": [_value_info(value) for value in graph.output],
        "value_info": [_value_info(value) for value in graph.value_info],
        "node_count": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "initializer_count": len(graph.initializer),
        "initializer_bytes": sum(len(value.raw_data) for value in graph.initializer),
        "initializers": [_initializer(value) for value in graph.initializer],
        "nodes": nodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="ONNX files to inspect")
    parser.add_argument("--json-out", type=Path, help="write one JSON manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [inspect(path) for path in args.paths]
    payload: Any = reports[0] if len(reports) == 1 else {"models": reports}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
