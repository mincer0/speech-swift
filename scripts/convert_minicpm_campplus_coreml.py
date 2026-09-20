#!/usr/bin/env python3
"""Convert MiniCPM-o's CAM++ ONNX front-end to a fixed-shape CoreML bundle.

The official ``campplus.onnx`` graph has dynamic masking/pooling shape nodes
that the modern coremltools ONNX importer no longer exposes.  We first import
it through onnx2torch (which preserves the ONNX weights/ops), freeze the
production input shape [1, 500, 80], trace to TorchScript, and then ask
coremltools to emit an ML Program.  The generated bundle remains directly
compatible with ``MiniCPMCoreMLCamPlusPlusEncoder``.

The output is a sidecar in the parent Token2Wav bundle.  ``config.json`` and
``manifest.json`` are merged rather than replaced so conversion can be run
after the flow, HiFT, and S3Tokenizer converters.  Provenance records contain
only stable identifiers and basenames; the converter never serialises a local
checkout path into a portable bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Optional, Sequence


MANIFEST_SCHEMA = "minicpm-o-token2wav-mlx/v1"
DEFAULT_SOURCE_MODEL = "OpenBMB/MiniCPM-o-4_5"
COMPONENT_ID = "minicpm-o-4_5.token2wav.campplus"
COMPONENT_KEY = "speaker_encoder"
COMPONENT_NAME = "campplus"
MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
INPUT_SHAPE = (1, 500, 80)
OUTPUT_SHAPE = (1, 192)
# The fixed CAM++ export reduces the 500-frame fbank to 250 frames before
# each of its 52 attentive statistics pooling blocks.  The upstream ONNX
# graph expresses the final partial window through a dynamic Pad followed by
# AveragePool(ceil_mode=1).  onnx2torch does not evaluate that dynamic Pad for
# the frozen shape, so relying on the imported module leaves the last window
# with a denominator of 50 instead of the ORT/official denominator of 100.
CAMPPLUS_POOL_COUNT = 52
CAMPPLUS_POOL_KERNEL = 100
CAMPPLUS_POOL_STRIDE = 100
CAMPPLUS_INTERNAL_FRAMES = 250

# ``torch.library.Library`` unregisters its operators when the last Python
# reference is released.  Keep the compatibility declaration alive for the
# whole conversion, including the lazy ``onnx2torch`` import and CoreML trace.
_TORCHVISION_COMPAT_LIBRARY: Any = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse converter arguments without importing model-conversion packages."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="MiniCPM-o campplus.onnx source graph")
    parser.add_argument(
        "output",
        type=Path,
        help="CoreML sidecar path (for example MiniCPM-CamPlusPlus.mlpackage)",
    )
    # coremltools 6.x exposes iOS target constants only.  An ML Program
    # generated for iOS15 is also loadable by macOS 15 Core ML, which is the
    # runtime used by the Swift adapter.
    parser.add_argument(
        "--deployment-target",
        default="iOS15",
        help="coremltools target constant (default: iOS15)",
    )
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Stable model identifier recorded in config/manifest (never a local path)",
    )
    parser.add_argument(
        "--source-revision",
        default=MODEL_REPOSITORY_REVISION,
        help=(
            "Model repository revision for source weights (default: "
            f"{MODEL_REPOSITORY_REVISION}; not a Demo/code revision)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this converter's existing CoreML sidecar/component record",
    )
    return parser.parse_args(argv)


def _configure_protobuf_compatibility() -> None:
    """Use protobuf's Python implementation unless the caller chose one.

    Some coremltools releases still construct descriptors through APIs removed
    from newer protobuf C++ runtimes.  The environment variable is scoped to
    this process and ``setdefault`` intentionally preserves an explicit user
    choice (including a value that is not ``"python"``).
    """

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _register_torchvision_compat(torch_module: Any) -> Any:
    """Define optional torchvision NMS schemas before importing onnx2torch.

    CPU-only torchvision wheels can omit the compiled NMS operators while
    their Python package unconditionally registers fake kernels for them.  A
    schema-only ``DEF`` library is numerically inert and lets that import
    complete.  Returning/retaining the library is important: torch removes
    these definitions when its last ``Library`` reference is collected.
    """

    global _TORCHVISION_COMPAT_LIBRARY

    library_factory = getattr(getattr(torch_module, "library", None), "Library", None)
    if library_factory is None:
        return _TORCHVISION_COMPAT_LIBRARY
    try:
        library = library_factory("torchvision", "DEF")
    except (RuntimeError, TypeError):
        # A full torchvision build may already own the namespace.  In that
        # case its real operators (or an earlier compatibility declaration)
        # are sufficient and no second DEF namespace can be created.
        return _TORCHVISION_COMPAT_LIBRARY

    for name in ("nms", "qnms"):
        try:
            library.define(
                f"{name}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except RuntimeError:
            # The operator can already exist when this helper is called more
            # than once in a long-lived Python process.
            pass
    _TORCHVISION_COMPAT_LIBRARY = library
    return library


def _load_conversion_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load heavy conversion dependencies after all path checks have passed."""

    # This must happen immediately before coremltools is imported.  In
    # particular, importing the script or requesting ``--help`` must not touch
    # protobuf/coremltools at all.
    _configure_protobuf_compatibility()
    import coremltools as ct
    import numpy as np
    import torch

    # onnx2torch imports torchvision on several supported versions.  Prefer a
    # normal import when the wheel already ships its compiled operators; only
    # install schema-only compatibility declarations for the CPU-only wheel
    # failure mode.  Registering first is unsafe with a full torchvision
    # wheel: its native extension then attempts to define the same schema and
    # aborts the process in C++ before Python can catch the duplicate.
    try:
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        if "torchvision::nms" not in str(exc):
            raise
        _register_torchvision_compat(torch)
        import torchvision  # noqa: F401
    from onnx2torch import convert as onnx2torch_convert

    return ct, np, torch, onnx2torch_convert


def _pool_right_padding(
    input_length: int, *, kernel_size: int, stride: int
) -> int:
    """Return the static right pad needed for a ceil-mode final window.

    CAM++ uses a kernel equal to its stride.  The official graph therefore
    has one window for every ``ceil(input_length / stride)`` block, and the
    incomplete final block is averaged with the full kernel denominator after
    zero padding.  Keeping this arithmetic separate makes the conversion
    contract testable without importing torch/coremltools.
    """

    if input_length <= 0:
        raise ValueError("pool input length must be positive")
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("pool kernel and stride must be positive")
    if kernel_size != stride:
        raise ValueError("static CAM++ pool replacement requires kernel == stride")
    return (-input_length) % stride


def _replace_campplus_avg_pools(model: Any, torch_module: Any) -> int:
    """Make all imported CAM++ partial windows use the official denominator.

    ``onnx2torch`` stores this graph's modules under flat slash-separated
    names, so replacing the entries in ``_modules`` also updates the
    ``call_module`` targets in its ``GraphModule``.  The strict count and
    parameter checks prevent a converter/library change from silently
    producing a numerically different sidecar.
    """

    nn = torch_module.nn
    right_padding = _pool_right_padding(
        CAMPPLUS_INTERNAL_FRAMES,
        kernel_size=CAMPPLUS_POOL_KERNEL,
        stride=CAMPPLUS_POOL_STRIDE,
    )

    class StaticRightPaddedAveragePool1d(nn.Module):
        """AveragePool with an explicit zero-padded final stride window."""

        def __init__(self, old_pool: Any) -> None:
            super().__init__()
            self.right_pad = nn.ConstantPad1d((0, right_padding), 0.0)
            self.pool = nn.AvgPool1d(
                kernel_size=old_pool.kernel_size,
                stride=old_pool.stride,
                padding=old_pool.padding,
                ceil_mode=False,
                count_include_pad=True,
            )

        def forward(self, values: Any) -> Any:
            return self.pool(self.right_pad(values))

    replaced = 0
    for name, module in list(model._modules.items()):
        if not isinstance(module, nn.AvgPool1d):
            continue
        kernel = tuple(module.kernel_size)
        stride = tuple(module.stride)
        padding = tuple(module.padding)
        if (
            kernel != (CAMPPLUS_POOL_KERNEL,)
            or stride != (CAMPPLUS_POOL_STRIDE,)
            or padding != (0,)
            or not module.ceil_mode
            or module.count_include_pad
        ):
            raise RuntimeError(
                "unexpected CAM++ AvgPool1d parameters at "
                f"{name!r}: kernel={kernel}, stride={stride}, "
                f"padding={padding}, ceil_mode={module.ceil_mode}, "
                f"count_include_pad={module.count_include_pad}"
            )
        model._modules[name] = StaticRightPaddedAveragePool1d(module)
        replaced += 1

    if replaced != CAMPPLUS_POOL_COUNT:
        raise RuntimeError(
            "unexpected CAM++ AvgPool1d count: "
            f"expected {CAMPPLUS_POOL_COUNT}, found {replaced}"
        )
    return replaced


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a CoreML file/package without including its machine-local root."""

    if root.is_file():
        return _sha256_file(root, chunk_size=chunk_size)
    if not root.is_dir():
        raise FileNotFoundError(f"CoreML output does not exist: {root}")
    digest = hashlib.sha256()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    ]
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid existing {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"existing {label} must be a JSON object: {path}")
    return value


def _portable_model_id(value: str) -> str:
    """Reject a model argument that would make provenance machine-specific."""

    model_id = str(value).strip()
    if not model_id:
        raise ValueError("--source-model must not be empty")
    # PureWindowsPath catches drive-letter/UNC paths even when conversion is
    # run on macOS.  Hugging Face ids such as ``OpenBMB/MiniCPM-o-4_5`` remain
    # valid stable identifiers.
    if Path(model_id).is_absolute() or PureWindowsPath(model_id).is_absolute():
        raise ValueError("--source-model must be a stable model id, not an absolute path")
    if model_id.startswith("~"):
        raise ValueError("--source-model must be a stable model id, not a home path")
    return model_id


def _component_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.components must be a JSON object")
    return dict(value)


def _component_config(
    *,
    output_name: str,
    output_tree_sha256: str,
    deployment_target: str,
    compute_precision: str,
) -> dict[str, Any]:
    return {
        "component": COMPONENT_NAME,
        "component_id": COMPONENT_ID,
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "deployment_target": deployment_target,
        "compute_precision": compute_precision,
        "weights": {
            "path": output_name,
            "kind": "coreml_bundle",
            "tree_sha256": output_tree_sha256,
        },
    }


def _check_component_conflicts(bundle_dir: Path, *, overwrite: bool) -> None:
    """Fail before model conversion if an existing parent record would change."""

    for filename, label in (
        ("config.json", "Token2Wav config"),
        ("manifest.json", "Token2Wav manifest"),
    ):
        path = bundle_dir / filename
        if not path.exists():
            continue
        value = _load_object(path, label=label)
        components = _component_mapping(value.get("components"), label=label)
        if COMPONENT_KEY in components and not overwrite:
            raise FileExistsError(
                f"{label} already has component {COMPONENT_KEY!r}; pass --overwrite"
            )


def _update_bundle_config(
    bundle_dir: Path,
    *,
    source_model: str,
    source_revision: str,
    component_config: Mapping[str, Any],
    overwrite: bool,
) -> None:
    config_path = bundle_dir / "config.json"
    config = _load_object(config_path, label="Token2Wav config") if config_path.exists() else {}
    components = _component_mapping(config.get("components"), label="Token2Wav config")
    if COMPONENT_KEY in components and not overwrite:
        raise FileExistsError(
            f"Token2Wav config already has component {COMPONENT_KEY!r}; pass --overwrite"
        )
    config.update(
        {
            "format": MANIFEST_SCHEMA,
            "model_type": "minicpmo_token2wav_mlx",
            "source_model": source_model,
            "source_revision": source_revision,
            "revision_kind": "model_repository",
            # This is the Demo implementation identity, never the model
            # repository revision above.
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        }
    )
    components[COMPONENT_KEY] = dict(component_config)
    config["components"] = components
    _write_json(config_path, config)


def _update_manifest(
    bundle_dir: Path,
    *,
    source_basename: str,
    source_model: str,
    source_revision: str,
    component_manifest: Mapping[str, Any],
    overwrite: bool,
) -> None:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _load_object(manifest_path, label="Token2Wav manifest") if manifest_path.exists() else {}
    components = _component_mapping(manifest.get("components"), label="Token2Wav manifest")
    if COMPONENT_KEY in components and not overwrite:
        raise FileExistsError(
            f"Token2Wav manifest already has component {COMPONENT_KEY!r}; pass --overwrite"
        )
    manifest.update(
        {
            "schema": MANIFEST_SCHEMA,
            "model_type": "minicpmo_token2wav_mlx",
            "source_model": source_model,
            "source_revision": source_revision,
            "revision_kind": "model_repository",
            "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
            # A basename is useful for diagnostics but cannot leak the source
            # checkout path into a copied bundle.
            "source_basename": source_basename,
        }
    )
    components[COMPONENT_KEY] = dict(component_manifest)
    manifest["components"] = components
    _write_json(manifest_path, manifest)


def _remove_existing_output(output_path: Path) -> None:
    """Remove only an explicitly selected sidecar, never its parent bundle."""

    if output_path.is_dir() and not output_path.is_symlink():
        # A caller accidentally passing the Token2Wav parent would otherwise
        # delete flow/HiFT/tokenizer artifacts when --overwrite is supplied.
        protected = {"config.json", "manifest.json", "flow.safetensors", "hift.safetensors"}
        if any((output_path / name).exists() for name in protected):
            raise ValueError(
                "output appears to be the Token2Wav bundle directory; "
                "choose a CoreML sidecar path inside it"
            )
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source_model = _portable_model_id(args.source_model)
    source_revision = args.source_revision or MODEL_REPOSITORY_REVISION
    onnx_path = args.onnx.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"CAM++ ONNX source does not exist: {onnx_path}")
    if output_path == onnx_path:
        raise ValueError("CoreML output path must differ from the ONNX source path")
    # Never allow --overwrite to target the source directory (or an ancestor
    # containing it): removing an existing package there would delete the
    # ONNX input and potentially the rest of the downloaded model.
    try:
        onnx_path.relative_to(output_path)
    except ValueError:
        pass
    else:
        raise ValueError("CoreML output path must not contain the ONNX source path")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing CoreML sidecar; pass --overwrite: {output_path}"
            )
        _remove_existing_output(output_path)
    bundle_dir = output_path.parent
    _check_component_conflicts(bundle_dir, overwrite=args.overwrite)

    # Delay all model-conversion imports until argument parsing, path safety,
    # overwrite handling, and existing-manifest conflict checks are complete.
    ct, np, torch, onnx2torch_convert = _load_conversion_dependencies()

    # Preserve the numerical ONNX -> onnx2torch -> TorchScript -> CoreML path.
    # The fixed-shape replacement below is part of the numerical path: it
    # makes the dynamic ONNX right-padding contract explicit before tracing.
    source_onnx_sha256 = _sha256_file(onnx_path)
    model = onnx2torch_convert(str(onnx_path)).eval()
    replaced_pool_count = _replace_campplus_avg_pools(model, torch)
    example = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        eager = model(example).detach().cpu().numpy()
    if eager.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"unexpected ONNX-import output shape: {eager.shape}")

    # TorchScript removes Python-side graph wrappers inserted by onnx2torch;
    # strict=False permits harmless shape helper constants in old exports.
    traced = torch.jit.trace(model, example, strict=False)
    traced = torch.jit.freeze(traced.eval())
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="mel_features", shape=INPUT_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=getattr(ct.target, args.deployment_target),
        # CoreML's default ML Program precision is FP16.  On this graph the
        # 52 repeated attention pools amplify FP16 reduction error enough to
        # move the 192-d embedding (the default export reaches ~0.38 max
        # error on the fixed prompt cases).  The production sidecar is a
        # fixed-shape parity artifact, so retain Float32 accumulation here;
        # the caller can still select CPU/GPU/ANE at model load time.
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.author = "MiniCPM-o native runtime"
    mlmodel.short_description = "CAM++ 192-d speaker embedding for MiniCPM-o Token2Wav"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    output_tree_sha256 = _sha256_tree(output_path)
    converter_path = Path(__file__).resolve()
    component_config = _component_config(
        output_name=output_path.name,
        output_tree_sha256=output_tree_sha256,
        deployment_target=args.deployment_target,
        compute_precision="FLOAT32",
    )
    component_manifest: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component": COMPONENT_NAME,
        "component_key": COMPONENT_KEY,
        "source_revision": source_revision,
        "revision_kind": "model_repository",
        "implementation_commit": DEMO_IMPLEMENTATION_COMMIT,
        "upstream_commit": DEMO_IMPLEMENTATION_COMMIT,
        "weights": {
            "path": output_path.name,
            "kind": "coreml_bundle",
            "tree_sha256": output_tree_sha256,
        },
        # Keep a direct artifact record as well as the peer-converter
        # ``weights`` record; both intentionally use a basename only.
        "artifact": {
            "path": output_path.name,
            "kind": "coreml_bundle",
            "tree_sha256": output_tree_sha256,
        },
        "source": {
            "onnx": {
                "path": onnx_path.name,
                "bytes": onnx_path.stat().st_size,
                "sha256": source_onnx_sha256,
            },
        },
        "converter": {
            "path": converter_path.name,
            "sha256": _sha256_file(converter_path),
        },
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "deployment_target": args.deployment_target,
        "compute_precision": "FLOAT32",
        "config": component_config,
    }
    _update_bundle_config(
        bundle_dir,
        source_model=source_model,
        source_revision=source_revision,
        component_config=component_config,
        overwrite=args.overwrite,
    )
    _update_manifest(
        bundle_dir,
        source_basename=onnx_path.parent.name,
        source_model=source_model,
        source_revision=source_revision,
        component_manifest=component_manifest,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "component_id": COMPONENT_ID,
                "tensor_shape": list(eager.shape),
                "replaced_avg_pool_count": replaced_pool_count,
                "right_pad_frames": _pool_right_padding(
                    CAMPPLUS_INTERNAL_FRAMES,
                    kernel_size=CAMPPLUS_POOL_KERNEL,
                    stride=CAMPPLUS_POOL_STRIDE,
                ),
                "output": output_path.name,
                "manifest": "manifest.json",
                "output_tree_sha256": output_tree_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
