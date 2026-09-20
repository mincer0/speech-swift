#!/usr/bin/env python3
"""Export a small, fixed-source MiniCPM-o vision parity fixture.

This exporter deliberately lives in the native Swift package rather than
importing another checkout's helper script.  It imports the pinned upstream
MiniCPM image processor, SigLIP tower, and Resampler from ``--source`` and
records that source revision in ``manifest.json``.  Inputs are deterministic
PNG round trips generated in this file, so no user media or hidden assets are
part of a fixture.

The fixture contains the no-slice path, the official 1000x500 -> 2x1 slice
path, a multi-image ordering case, and independent per-frame slice limits.
It is intentionally small enough to regenerate; model outputs are written as
little-endian float32 even when the PyTorch oracle computes in bfloat16.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_SOURCE_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
PROCESSOR_FILE = "MiniCPMO45/processing_minicpmo.py"
VISION_FILE = "MiniCPMO45/modeling_navit_siglip.py"
RESAMPLER_FILE = "MiniCPMO45/modeling_minicpmo.py"
SCHEMA_VERSION = 1
_TORCHVISION_COMPAT_LIB = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    if root.is_file():
        return sha256_file(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve oracle source commit under {root}: {exc}") from exc


def import_fixed_sources(source_root: Path):
    """Import the three upstream modules as one namespace package.

    The upstream directory intentionally has no ``__init__.py``.  Adding its
    parent (rather than the package directory) to ``sys.path`` lets Python's
    namespace-package machinery resolve the relative imports used by
    ``modeling_minicpmo.py``.
    """

    package = source_root / "MiniCPMO45"
    if not package.is_dir():
        raise FileNotFoundError(f"missing fixed source package: {package}")
    sys.path.insert(0, str(source_root))
    try:
        processor = importlib.import_module("MiniCPMO45.processing_minicpmo")
        vision = importlib.import_module("MiniCPMO45.modeling_navit_siglip")
        model = importlib.import_module("MiniCPMO45.modeling_minicpmo")
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass
    return processor, vision, model


def ensure_torchvision_compat() -> None:
    """Work around CPU-only torchvision wheels missing optional NMS ops.

    Some environments import ``transformers`` while exporting the oracle but
    ship a torchvision wheel whose custom C++ operators were not built. The
    upstream processor does not call these ops; defining their schemas keeps
    torchvision's fake-kernel registration from aborting the import. A global
    reference keeps the temporary operator library alive for the process.
    """

    global _TORCHVISION_COMPAT_LIB
    try:
        import torch
    except Exception:
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
        # A normal torchvision installation already owns the namespace.
        _TORCHVISION_COMPAT_LIB = None


def checkpoint_shards(model_root: Path) -> list[Path]:
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        paths = [model_root / str(name) for name in names]
    else:
        paths = sorted(model_root.glob("*.safetensors"))
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise FileNotFoundError(f"no safetensors shards under {model_root}")
    return paths


def load_visual_weights(model_root: Path, safe_open):
    vision: dict[str, Any] = {}
    resampler: dict[str, Any] = {}
    source_shards: list[str] = []
    for shard in checkpoint_shards(model_root):
        found = False
        with safe_open(str(shard), framework="pt") as reader:
            for key in reader.keys():
                if key.startswith("vpm."):
                    vision[key[len("vpm.") :]] = reader.get_tensor(key)
                    found = True
                elif key.startswith("resampler."):
                    resampler[key[len("resampler.") :]] = reader.get_tensor(key)
                    found = True
        if found:
            source_shards.append(shard.name)
    if not vision or not resampler:
        raise ValueError("checkpoint lacks vpm.* or resampler.* tensors")
    return vision, resampler, source_shards


def synthetic_png(size: tuple[int, int], seed: int, pattern: str = "gradient"):
    """Create and decode a deterministic RGB/RGBA PNG (width, height).

    ``quadrants`` is deliberately asymmetric so a vertically reversed crop is
    obvious. ``rgba_alpha`` keeps distinct RGB values under alpha 0/64/128;
    Pillow's ``convert("RGB")`` drops alpha, while a premultiplied CoreGraphics
    context would incorrectly darken those colors.
    """

    import numpy as np
    from PIL import Image

    width, height = size
    y, x = np.indices((height, width), dtype=np.uint32)
    if pattern == "gradient":
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = (3 * x + 5 * y + 17 * seed) % 256
        pixels[..., 1] = (7 * x + 11 * y + 29 * seed) % 256
        pixels[..., 2] = (13 * x + 2 * y + 43 * seed) % 256
        image = Image.fromarray(pixels, mode="RGB")
    elif pattern in {"quadrants", "rgba_alpha"}:
        # Use four non-symmetric colors and a small coordinate ramp in each
        # quadrant. The ramp prevents an accidental 180-degree rotation from
        # looking like a valid crop when a quadrant happens to be uniform.
        colors = np.array(
            [[236, 31, 47], [24, 190, 84], [41, 96, 214], [245, 201, 35]],
            dtype=np.uint8,
        )
        quadrant = (y >= height // 2).astype(np.int32) * 2 + (x >= width // 2).astype(np.int32)
        rgb = colors[quadrant]
        ramp = ((x + 3 * y + seed) % 17).astype(np.uint8)
        pixels = (rgb.astype(np.uint16) + ramp[..., None]) % 256
        pixels = pixels.astype(np.uint8)
        if pattern == "rgba_alpha":
            alpha_values = np.array([255, 128, 64, 0], dtype=np.uint8)
            alpha = alpha_values[quadrant][..., None]
            image = Image.fromarray(np.concatenate([pixels, alpha], axis=-1), mode="RGBA")
        else:
            image = Image.fromarray(pixels, mode="RGB")
    else:
        raise ValueError(f"unknown synthetic image pattern: {pattern}")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    output.seek(0)
    with Image.open(output) as image:
        # Keep RGBA inputs intact here; the fixed upstream processor performs
        # the authoritative ``convert("RGB")`` step itself.
        return image.copy()


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kind: str
    images: tuple[tuple[int, int], ...]
    limits: int | tuple[int, ...]
    seeds: tuple[int, ...]
    pattern: str = "gradient"


def case_specs() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("overview_square_max1", "single_overview", ((224, 224),), 1, (1,)),
        CaseSpec("wide_1000x500_max9", "hd_slices", ((1000, 500),), 9, (2,)),
        CaseSpec(
            "multi_image_order",
            "multi_image",
            ((1000, 500), (224, 224)),
            9,
            (3, 4),
        ),
        CaseSpec(
            "video_frame_order_per_frame_limits",
            "video_frames_per_frame_limits",
            ((224, 224), (1000, 500), (224, 224)),
            (1, 9, 1),
            (5, 6, 7),
        ),
        CaseSpec(
            "processor_quadrants_rgb",
            "processor_rgb_quadrants",
            ((1000, 500),),
            9,
            (8,),
            "quadrants",
        ),
        CaseSpec(
            "processor_alpha_drop",
            "processor_rgba_alpha_drop",
            ((224, 224),),
            1,
            (9,),
            "rgba_alpha",
        ),
    )


def as_int_list(value: Sequence[Any]) -> list[int]:
    return [int(item) for item in value]


def process_case(processor, spec: CaseSpec):
    images = [synthetic_png(size, seed, spec.pattern) for size, seed in zip(spec.images, spec.seeds)]
    pixels: list[Any] = []
    targets: list[Any] = []
    groups: list[dict[str, Any]] = []
    if isinstance(spec.limits, int):
        prepared = processor.preprocess(images, max_slice_nums=spec.limits)
        pixels = list(prepared["pixel_values"][0])
        targets = list(prepared["tgt_sizes"][0])
        offset = 0
        for index, image in enumerate(images):
            sliced = processor.get_sliced_images(image, spec.limits)
            grid = processor.get_sliced_grid(image.size, spec.limits)
            count = len(sliced)
            groups.append(
                {
                    "source_index": index,
                    "frame_index": None,
                    "original_size": list(map(int, image.size)),
                    "max_slice_nums": int(spec.limits),
                    "grid": None if grid is None else as_int_list(grid),
                    "part_start": offset,
                    "part_count": count,
                }
            )
            offset += count
    else:
        for index, (image, limit) in enumerate(zip(images, spec.limits)):
            prepared = processor.preprocess([image], max_slice_nums=int(limit))
            current_pixels = list(prepared["pixel_values"][0])
            current_targets = list(prepared["tgt_sizes"][0])
            offset = len(pixels)
            pixels.extend(current_pixels)
            targets.extend(current_targets)
            grid = processor.get_sliced_grid(image.size, int(limit))
            groups.append(
                {
                    "source_index": index,
                    "frame_index": index,
                    "original_size": list(map(int, image.size)),
                    "max_slice_nums": int(limit),
                    "grid": None if grid is None else as_int_list(grid),
                    "part_start": offset,
                    "part_count": len(current_pixels),
                }
            )
    if len(pixels) != len(targets):
        raise AssertionError("processor returned mismatched pixel/target counts")
    return images, pixels, targets, groups


def configure_vision_config(vision_config):
    """Apply the exporter's deterministic vision-model settings.

    ``use_return_dict`` is a read-only convenience property in current
    Transformers releases.  Its backing setting is ``return_dict`` (while
    ``torchscript`` may still force the derived property off), so configure
    that field instead of assigning to the property.  Keep the assertion here
    to make an unexpected output-mode change explicit rather than silently
    producing tuple outputs that would diverge from the golden contract.
    """

    vision_config._attn_implementation = "eager"
    vision_config.output_attentions = False
    vision_config.output_hidden_states = False
    vision_config.return_dict = True
    if not vision_config.use_return_dict:
        raise AssertionError(
            "vision config must expose return_dict outputs (check torchscript setting)"
        )
    return vision_config


def load_models(source_model: Path, vision_module, model_module, torch, safe_open, dtype_name: str):
    root_config = json.loads((source_model / "config.json").read_text(encoding="utf-8"))
    vision_config = configure_vision_config(
        vision_module.SiglipVisionConfig(**root_config["vision_config"])
    )
    vision = vision_module.SiglipVisionTransformer(vision_config).eval()
    hidden_size = int(root_config.get("hidden_size", 4096))
    query_num = int(root_config.get("query_num", 64))
    resampler = model_module.Resampler(
        num_queries=query_num,
        embed_dim=hidden_size,
        num_heads=max(1, hidden_size // 128),
        kv_dim=int(root_config["vision_config"]["hidden_size"]),
        max_size=(70, 70),
    ).eval()
    vision_state, resampler_state, source_shards = load_visual_weights(source_model, safe_open)
    vision.load_state_dict(vision_state, strict=True)
    resampler.load_state_dict(resampler_state, strict=True)
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    return (
        root_config,
        vision.to(dtype=dtype).eval(),
        resampler.to(dtype=dtype).eval(),
        source_shards,
        dtype,
    )


def pad_packed_pixels(pixels: Sequence[Any], targets: Sequence[Any], np):
    arrays = [np.asarray(value, dtype=np.float32) for value in pixels]
    if not arrays:
        raise ValueError("cannot batch an empty processor result")
    patch = int(arrays[0].shape[1])
    counts = [int(np.asarray(target).reshape(-1)[0] * np.asarray(target).reshape(-1)[1]) for target in targets]
    max_count = max(counts)
    batch = np.zeros((len(arrays), 3, patch, max_count * patch), dtype=np.float32)
    mask = np.zeros((len(arrays), 1, max_count), dtype=np.bool_)
    for index, (array, count) in enumerate(zip(arrays, counts)):
        if array.shape != (3, patch, count * patch):
            raise ValueError(f"part {index} has shape {array.shape}, expected {(3, patch, count * patch)}")
        batch[index, :, :, : count * patch] = array
        mask[index, 0, :count] = True
    return batch, mask


def write_f32(path: Path, value, np) -> str:
    data = np.asarray(value, dtype="<f4").tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def key_values(value, np) -> list[dict[str, Any]]:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return []
    indexes = sorted(set([0, min(1, flat.size - 1), min(2, flat.size - 1), flat.size // 2, flat.size - 1]))
    return [{"flat_index": int(index), "value": float(flat[index])} for index in indexes]


STAGE_NAMES = (
    "processor",
    "patch_embedding",
    "positional",
    "encoder",
    "projector",
    "final_embedding",
)


def _as_tensor(value):
    """Extract a tensor from a PyTorch/Transformers module output."""

    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value


def _as_numpy(value, np):
    """Convert a captured tensor/array to a CPU NumPy view."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _stage_summary(value, np, *, reference=None) -> dict[str, Any]:
    """Return compact, JSON-safe statistics for one intermediate tensor.

    Full final embeddings remain in binary files for the numerical gate.  The
    intermediate stages are intentionally represented by shape/statistics so
    regenerating a fixture does not multiply its size by six.  ``cosine`` is
    measured against the float32 oracle when a second precision pass is
    available; otherwise it is explicitly the self-cosine (1.0).
    """

    array = _as_numpy(value, np).astype(np.float32, copy=False)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {
            "shape": [int(item) for item in array.shape],
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "cosine": 1.0,
            "cosine_reference": "self",
            "key_values": [],
        }
    cosine = 1.0
    cosine_reference = "self"
    if reference is not None:
        other = _as_numpy(reference, np).astype(np.float32, copy=False).reshape(-1)
        if other.shape != flat.shape:
            raise ValueError(
                f"stage reference shape mismatch: {array.shape} vs {_as_numpy(reference, np).shape}"
            )
        denominator = float(np.linalg.norm(flat) * np.linalg.norm(other))
        cosine = float(np.dot(flat, other) / max(denominator, 1e-12))
        cosine_reference = "float32_oracle"
    return {
        "shape": [int(item) for item in array.shape],
        "min": float(flat.min()),
        "max": float(flat.max()),
        "max_abs": float(np.abs(flat).max()),
        "mean": float(flat.mean()),
        "mean_abs": float(np.abs(flat).mean()),
        "cosine": cosine,
        "cosine_reference": cosine_reference,
        "key_values": key_values(flat, np),
    }


def _capture_stage_hooks(vision, resampler, *, torch):
    """Capture official intermediate tensors without changing computation."""

    captured: dict[str, Any] = {}

    def save(name, transform=lambda value: value):
        def hook(_module, _inputs, output):
            value = transform(_as_tensor(output))
            captured[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return hook

    handles = []
    handles.append(vision.embeddings.patch_embedding.register_forward_hook(
        save("patch_embedding", lambda value: value.flatten(2).transpose(1, 2))))
    handles.append(vision.embeddings.position_embedding.register_forward_hook(
        save("positional")))
    handles.append(vision.encoder.register_forward_hook(save("encoder")))
    handles.append(resampler.kv_proj.register_forward_hook(save("projector")))
    handles.append(resampler.register_forward_hook(save("final_embedding")))
    return captured, handles


def _run_visual_batch(
    vision,
    resampler,
    padded,
    patch_mask,
    target_batch,
    *,
    dtype,
    torch,
    np,
):
    """Run one official PyTorch batch and return final + intermediate stages."""

    captured, hooks = _capture_stage_hooks(vision, resampler, torch=torch)
    try:
        with torch.inference_mode():
            pixel_batch = torch.from_numpy(padded).to(dtype=dtype)
            target = torch.from_numpy(target_batch).to(dtype=torch.int32)
            hidden = vision(
                pixel_values=pixel_batch,
                patch_attention_mask=torch.from_numpy(patch_mask),
                tgt_sizes=target,
            ).last_hidden_state
            embedding = resampler(hidden, target)
            # The hook captures the same final tensor. Keep a direct copy as a
            # defensive fallback for model variants that bypass module hooks.
            final = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
    finally:
        for handle in hooks:
            handle.remove()
    captured.setdefault("final_embedding", final)
    # Processor output is the pre-model stage and is always float32 here.
    captured["processor"] = np.asarray(padded, dtype=np.float32)
    return captured


def _trim_stage(value, index: int, valid_count: int, *, stage: str):
    """Trim padded token axes while preserving batch/feature dimensions."""

    if stage in {"patch_embedding", "positional", "encoder", "projector"}:
        return value[index : index + 1, :valid_count, ...]
    if stage == "processor":
        return value[index : index + 1, ...]
    return value[index : index + 1, ...]


def run(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    actual_commit = git_commit(source)
    if actual_commit != args.source_commit:
        raise RuntimeError(f"source commit mismatch: expected {args.source_commit}, got {actual_commit}")
    for relative in (PROCESSOR_FILE, VISION_FILE, RESAMPLER_FILE):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)

    try:
        import numpy as np
        import torch
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("export requires numpy, torch, safetensors, and Pillow") from exc
    ensure_torchvision_compat()
    processor_module, vision_module, model_module = import_fixed_sources(source)
    processor = processor_module.MiniCPMVImageProcessor(
        max_slice_nums=9, scale_resolution=448, patch_size=14
    )
    root_config, vision, resampler, source_shards, dtype = load_models(
        args.source_model.resolve(), vision_module, model_module, torch, safe_open, args.dtype
    )
    # A bfloat16 fixture needs a float32 oracle for stage-level diagnostics.
    # Keep this second model local to the exporter; generated fixtures contain
    # only compact statistics, never a second copy of intermediate tensors.
    reference_vision = None
    reference_resampler = None
    if args.dtype == "bfloat16":
        _, reference_vision, reference_resampler, _, _ = load_models(
            args.source_model.resolve(),
            vision_module,
            model_module,
            torch,
            safe_open,
            "float32",
        )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for spec in case_specs():
        images, pixels, targets, groups = process_case(processor, spec)
        input_files: list[str] = []
        for image_index, image in enumerate(images):
            input_path = output / "inputs" / f"{spec.name}-image-{image_index:03d}.png"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(input_path, format="PNG", optimize=False)
            input_files.append(str(input_path.relative_to(output)))
        padded, patch_mask = pad_packed_pixels(pixels, targets, np)
        target_batch = np.asarray(targets, dtype=np.int32)
        captured = _run_visual_batch(
            vision,
            resampler,
            padded,
            patch_mask,
            target_batch,
            dtype=dtype,
            torch=torch,
            np=np,
        )
        reference = None
        if reference_vision is not None and reference_resampler is not None:
            reference = _run_visual_batch(
                reference_vision,
                reference_resampler,
                padded,
                patch_mask,
                target_batch,
                dtype=torch.float32,
                torch=torch,
                np=np,
            )
        embedding = _as_numpy(captured["final_embedding"], np).astype(
            np.float32, copy=False
        )
        case_root = output / "cases" / spec.name
        parts: list[dict[str, Any]] = []
        part_index = 0
        for group in groups:
            sliced = processor.get_sliced_images(images[group["source_index"]], group["max_slice_nums"])
            for local_index, image in enumerate(sliced):
                pixels_array = np.asarray(pixels[part_index], dtype=np.float32)
                target = np.asarray(targets[part_index], dtype=np.int32).reshape(-1)
                valid_count = int(target[0] * target[1])
                stage_statistics: dict[str, dict[str, Any]] = {}
                for stage in STAGE_NAMES:
                    if stage == "processor":
                        stage_value = pixels_array
                        reference_value = None
                    else:
                        stage_value = _trim_stage(
                            captured[stage],
                            part_index,
                            valid_count,
                            stage=stage,
                        )
                        reference_value = (
                            _trim_stage(
                                reference[stage],
                                part_index,
                                valid_count,
                                stage=stage,
                            )
                            if reference is not None
                            else None
                        )
                    stage_statistics[stage] = _stage_summary(
                        stage_value,
                        np,
                        reference=reference_value,
                    )
                prefix = f"part-{part_index:03d}"
                pixels_file = case_root / f"{prefix}.pixels.f32"
                embedding_file = case_root / f"{prefix}.embedding.f32"
                parts.append(
                    {
                        "order": part_index,
                        "source_index": group["source_index"],
                        "frame_index": group.get("frame_index"),
                        "role": "overview" if local_index == 0 else "slice",
                        "original_size": group["original_size"],
                        "resized_size": [int(image.size[0]), int(image.size[1])],
                        "target_size": [int(target[0]), int(target[1])],
                        "pixels_file": str(pixels_file.relative_to(output)),
                        "pixels_shape": list(map(int, pixels_array.shape)),
                        "pixels_dtype": "float32-le",
                        "pixels_sha256": write_f32(pixels_file, pixels_array, np),
                        "embedding_file": str(embedding_file.relative_to(output)),
                        "embedding_shape": list(map(int, embedding[part_index].shape)),
                        "embedding_dtype": "float32-le",
                        "embedding_sha256": write_f32(embedding_file, embedding[part_index], np),
                        "key_values": key_values(embedding[part_index], np),
                        "stage_statistics": stage_statistics,
                    }
                )
                part_index += 1
        cases.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "input_image_sizes": [list(size) for size in spec.images],
                "input_patterns": [spec.pattern for _ in spec.images],
                "max_slice_nums": spec.limits if isinstance(spec.limits, int) else None,
                "per_frame_max_slice_nums": list(spec.limits) if not isinstance(spec.limits, int) else None,
                "input_files": input_files,
                "groups": groups,
                "part_count": len(parts),
                "parts": parts,
            }
        )

    source_model = args.source_model.resolve()
    bundle = args.mlx_bundle.resolve() if args.mlx_bundle else source_model
    parity = {
        "max_abs": 0.25 if args.dtype == "bfloat16" else 0.05,
        "mean_abs": 0.02 if args.dtype == "bfloat16" else 0.002,
        "cosine": 0.999 if args.dtype == "bfloat16" else 0.99999,
    }
    # Paths are intentionally represented as stable provenance labels rather
    # than machine-specific absolute paths.  Content hashes above remain the
    # authoritative way to identify the source/model/bundle.
    source_label = f"fixed-upstream/{source.name}"
    source_model_label = f"source-model/{source_model.name}"
    bundle_label = f"mlx-bundle/{bundle.name}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": "minicpm_o_vision_siglip_resampler_oracle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "oracle": {
            "source_root": source_label,
            "git_commit": actual_commit,
            "processor_file": PROCESSOR_FILE,
            "vision_file": VISION_FILE,
            "resampler_file": RESAMPLER_FILE,
            "source_file_sha256": {relative: sha256_file(source / relative) for relative in (PROCESSOR_FILE, VISION_FILE, RESAMPLER_FILE)},
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "compute_dtype": args.dtype,
            "serialized_dtype": "float32-le",
        },
        "source_model": {
            "path": source_model_label,
            "tree_sha256": sha256_tree(source_model),
            "config_sha256": sha256_file(source_model / "config.json"),
            "visual_source_shards": source_shards,
            "vision_config": root_config.get("vision_config", {}),
            "query_num": int(root_config.get("query_num", 64)),
            "hidden_size": int(root_config.get("hidden_size", 4096)),
        },
        "bundle": {
            "path": bundle_label,
            "tree_sha256": sha256_tree(bundle),
            "bundle_kind": "mlx_vision" if args.mlx_bundle else "source_model",
        },
        "parity": {
            "embedding_shape": [int(root_config.get("query_num", 64)), int(root_config.get("hidden_size", 4096))],
            "recommended_max_abs": parity["max_abs"],
            "recommended_mean_abs": parity["mean_abs"],
            "recommended_cosine": parity["cosine"],
            "comparison": "Swift/PyTorch outputs are compared after float32 serialization; thresholds are acceptance gates.",
        },
        "processor_parity": {
            "recommended_max_abs": 0.2,
            "recommended_mean_abs": 0.02,
            "comparison": "Swift CoreGraphics processor pixels are compared with the fixed-source PIL processor; interpolation may differ by a small amount.",
        },
        "cases": cases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/MiniCPM-o-Demo"))
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--source-model", type=Path, default=Path("models/MiniCPM-o-4_5"))
    parser.add_argument("--mlx-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
