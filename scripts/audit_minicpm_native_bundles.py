#!/usr/bin/env python3
"""Audit the portable MiniCPM-o native model bundle set.

This tool is deliberately read-only and uses only Python's standard library.
It checks the five native bundle directories (LLM, audio, vision, semantic
TTS, and Token2Wav) without importing MLX, Torch, CoreML, or safetensors.  A
successful report proves that required artifacts exist, manifest/provenance
identities use the pinned model and Demo revisions, local absolute paths are
absent, and every in-bundle file/hash reference still matches the bytes on
disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping


MODEL_REPOSITORY_REVISION = "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
DEMO_IMPLEMENTATION_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
DEFAULT_LLM_BUNDLE = "MiniCPM-o-4_5-llm-mlx-8bit"
DEFAULT_BUNDLES = {
    "audio": "MiniCPM-o-4_5-audio-mlx",
    "vision": "MiniCPM-o-4_5-vision-mlx",
    "tts": "MiniCPM-o-4_5-tts-mlx",
    "token2wav": "MiniCPM-o-4_5-token2wav-mlx",
}
VISION_COMPUTE_DTYPES = frozenset({"float32", "bfloat16"})

REQUIRED_FILES = {
    "audio": ("config.json", "manifest.json", "audio_encoder.safetensors"),
    "vision": ("config.json", "manifest.json", "model.safetensors"),
    "tts": (
        "config.json",
        "manifest.json",
        "model.safetensors",
        "keys.json",
        "checksums.json",
    ),
    "llm": ("config.json", "model.safetensors", "key_manifest.json", "provenance.json"),
    "token2wav": (
        "config.json",
        "manifest.json",
        "flow.safetensors",
        "hift.safetensors",
        "speech_tokenizer.safetensors",
    ),
}

ABSOLUTE_WINDOWS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
SOURCE_CONTEXT_KEYS = frozenset(
    {
        "source",
        "sources",
        "source_files",
        "source_shards",
        "source_tts_shards",
        "source_to_target",
        "upstream",
        "converter",
    }
)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 of one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file/package deterministically without including its root path."""

    if path.is_file():
        return sha256_file(path, chunk_size=chunk_size)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and ".git" not in candidate.relative_to(path).parts
    ]
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _is_absolute_string(value: str) -> bool:
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    return bool(ABSOLUTE_WINDOWS.match(value))


def _walk_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_strings(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")


def _safe_relative(bundle: Path, raw: Any, *, label: str, errors: list[str]) -> Path | None:
    if not isinstance(raw, str) or not raw:
        errors.append(f"{label}: expected a relative path string")
        return None
    if _is_absolute_string(raw):
        errors.append(f"{label}: absolute path is not portable: {raw}")
        return None
    relative = Path(raw)
    if ".." in relative.parts:
        errors.append(f"{label}: path escapes bundle: {raw}")
        return None
    resolved = (bundle / relative).resolve()
    try:
        resolved.relative_to(bundle.resolve())
    except ValueError:
        errors.append(f"{label}: path escapes bundle: {raw}")
        return None
    return resolved


def _check_no_absolute_paths(value: Any, *, label: str, errors: list[str]) -> None:
    for path, text in _walk_strings(value):
        if _is_absolute_string(text):
            errors.append(f"{label}.{path}: absolute local path is not allowed: {text}")


def _check_identity(value: Mapping[str, Any], *, label: str, errors: list[str]) -> None:
    """Require separate model-repository and Demo implementation identities."""

    source_revision = value.get("source_revision")
    if source_revision != MODEL_REPOSITORY_REVISION:
        errors.append(
            f"{label}.source_revision: expected model revision "
            f"{MODEL_REPOSITORY_REVISION}, got {source_revision!r}"
        )
    if source_revision == DEMO_IMPLEMENTATION_COMMIT:
        errors.append(f"{label}.source_revision aliases the Demo implementation commit")
    for key in ("implementation_commit", "upstream_commit"):
        if value.get(key) != DEMO_IMPLEMENTATION_COMMIT:
            errors.append(
                f"{label}.{key}: expected Demo commit "
                f"{DEMO_IMPLEMENTATION_COMMIT}, got {value.get(key)!r}"
            )
    upstream = value.get("upstream")
    if isinstance(upstream, Mapping):
        commit = upstream.get("commit")
        if commit != DEMO_IMPLEMENTATION_COMMIT:
            errors.append(
                f"{label}.upstream.commit: expected Demo commit "
                f"{DEMO_IMPLEMENTATION_COMMIT}, got {commit!r}"
            )
        if upstream.get("revision_kind") not in (None, "git_commit"):
            errors.append(f"{label}.upstream.revision_kind is not git_commit")


def _record_hash(
    bundle: Path,
    raw_path: Any,
    record: Mapping[str, Any] | None,
    *,
    label: str,
    errors: list[str],
) -> None:
    """Check one manifest/config file reference and its optional hash record."""

    target = _safe_relative(bundle, raw_path, label=label, errors=errors)
    if target is None or not target.exists():
        if target is not None:
            errors.append(f"{label}: referenced path does not exist: {raw_path}")
        return
    if record is None:
        return
    if not isinstance(record, Mapping):
        errors.append(f"{label}: hash record is not an object")
        return
    expected_size = record.get("bytes", record.get("size"))
    if expected_size is not None:
        try:
            actual_size = target.stat().st_size if target.is_file() else sum(
                child.stat().st_size
                for child in target.rglob("*")
                if child.is_file()
            )
            if int(expected_size) != actual_size:
                errors.append(
                    f"{label}: size mismatch (expected {expected_size}, got {actual_size})"
                )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{label}: unable to check size: {exc}")
    expected_sha = record.get("sha256")
    if expected_sha is not None:
        try:
            actual_sha = sha256_file(target) if target.is_file() else sha256_tree(target)
        except OSError as exc:
            errors.append(f"{label}: unable to hash path: {exc}")
        else:
            if actual_sha != expected_sha:
                errors.append(
                    f"{label}: SHA256 mismatch (expected {expected_sha}, got {actual_sha})"
                )
    expected_tree = record.get("tree_sha256")
    if expected_tree is not None:
        if not target.is_dir():
            errors.append(f"{label}: tree_sha256 requires a directory/package")
        else:
            try:
                actual_tree = sha256_tree(target)
            except OSError as exc:
                errors.append(f"{label}: unable to hash tree: {exc}")
            else:
                if actual_tree != expected_tree:
                    errors.append(
                        f"{label}: tree SHA256 mismatch "
                        f"(expected {expected_tree}, got {actual_tree})"
                    )


def _iter_path_records(value: Any, *, context: str = "") -> Iterable[tuple[str, Any, Mapping[str, Any] | None]]:
    """Yield in-bundle path references from a manifest/config object.

    ``source`` and ``converter`` records intentionally identify files outside
    the generated bundle.  Their strings are still scanned for absolute paths,
    but they are not treated as local artifact references here.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_context = f"{context}.{key_text}" if context else key_text
            if key_text in SOURCE_CONTEXT_KEYS:
                continue
            if key_text in {"path", "weight_file", "weights_file", "model_file", "file"}:
                if isinstance(child, str):
                    yield child_context, child, None
                elif isinstance(child, Mapping):
                    yield from _iter_path_records(child, context=child_context)
                continue
            if key_text in {"files", "checksums"} and isinstance(child, Mapping):
                for name, record in child.items():
                    if isinstance(name, str):
                        record_path = record.get("path", name) if isinstance(record, Mapping) else name
                        yield f"{child_context}.{name}", record_path, record if isinstance(record, Mapping) else None
                    yield from _iter_path_records(record, context=f"{child_context}.{name}")
                continue
            yield from _iter_path_records(child, context=child_context)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_path_records(child, context=f"{context}[{index}]")


def _check_references(
    bundle: Path,
    value: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    for path_label, raw_path, record in _iter_path_records(value):
        key = (path_label, str(raw_path))
        if key in seen:
            continue
        seen.add(key)
        _record_hash(
            bundle,
            raw_path,
            record,
            label=f"{label}.{path_label}",
            errors=errors,
        )


def _check_checksums_file(bundle: Path, *, errors: list[str]) -> None:
    path = bundle / "checksums.json"
    if not path.is_file():
        return
    try:
        value = _read_json(path)
    except ValueError as exc:
        errors.append(f"checksums.json: {exc}")
        return
    if value.get("algorithm") != "sha256":
        errors.append("checksums.json.algorithm must be sha256")
    files = value.get("files")
    if not isinstance(files, Mapping):
        errors.append("checksums.json.files must be an object")
        return
    _check_references(bundle, {"files": files}, label="checksums", errors=errors)


def _check_vision_compute_dtype(
    config: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    *,
    errors: list[str],
) -> None:
    """Require an explicit, matching dtype declaration in both Vision records."""

    values: dict[str, Any] = {}
    for label, value in (("config", config), ("manifest", manifest)):
        if value is None:
            continue
        field = f"vision.{label}.compute_dtype"
        if "compute_dtype" not in value:
            errors.append(f"{field} is missing; expected float32 or bfloat16")
            continue
        dtype = value["compute_dtype"]
        values[label] = dtype
        if not isinstance(dtype, str) or dtype not in VISION_COMPUTE_DTYPES:
            errors.append(
                f"{field} must be one of float32 or bfloat16, got {dtype!r}"
            )
    if "config" in values and "manifest" in values:
        if values["config"] != values["manifest"]:
            errors.append(
                "vision config and manifest compute_dtype conflict: "
                f"config={values['config']!r}, manifest={values['manifest']!r}"
            )


def _required_files(bundle: Path, kind: str, *, errors: list[str]) -> None:
    for name in REQUIRED_FILES[kind]:
        path = bundle / name
        if not path.is_file():
            errors.append(f"missing required {kind} artifact: {name}")


def _audit_llm(bundle: Path, *, errors: list[str], warnings: list[str]) -> None:
    provenance_path = bundle / "provenance.json"
    key_manifest_path = bundle / "key_manifest.json"
    conversion_manifest_path = bundle / "conversion_manifest.json"
    provenance = _read_json(provenance_path)
    key_manifest = _read_json(key_manifest_path)
    _check_identity(provenance, label="llm.provenance", errors=errors)
    _check_identity(key_manifest, label="llm.key_manifest", errors=errors)
    _check_no_absolute_paths(provenance, label="llm.provenance", errors=errors)
    _check_no_absolute_paths(key_manifest, label="llm.key_manifest", errors=errors)
    _check_references(bundle, provenance, label="llm.provenance", errors=errors)
    _check_references(bundle, key_manifest, label="llm.key_manifest", errors=errors)
    bits = provenance.get("bits")
    if bits not in (0, 8):
        errors.append(f"llm.provenance.bits must be 0 (BF16) or 8, got {bits!r}")
    target_hash = provenance.get("target_weights_sha256")
    if not isinstance(target_hash, str):
        errors.append("llm.provenance.target_weights_sha256 is missing")
    else:
        weights = bundle / "model.safetensors"
        if weights.is_file() and sha256_file(weights) != target_hash:
            errors.append("llm.provenance.target_weights_sha256 does not match model.safetensors")
    if conversion_manifest_path.is_file():
        try:
            conversion_manifest = _read_json(conversion_manifest_path)
        except ValueError as exc:
            errors.append(f"llm.conversion_manifest: {exc}")
        else:
            _check_identity(conversion_manifest, label="llm.conversion_manifest", errors=errors)
            _check_no_absolute_paths(
                conversion_manifest,
                label="llm.conversion_manifest",
                errors=errors,
            )
            _check_references(
                bundle,
                conversion_manifest,
                label="llm.conversion_manifest",
                errors=errors,
            )
    else:
        warnings.append("llm conversion_manifest.json is absent (legacy bundle)")


def _audit_token2wav(bundle: Path, *, errors: list[str], warnings: list[str]) -> None:
    manifest = _read_json(bundle / "manifest.json")
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        errors.append("token2wav.manifest.components must be an object")
        return
    required_components = {"flow", "hift", "speech_tokenizer", "speaker_encoder"}
    missing = sorted(required_components - set(str(key) for key in components))
    if missing:
        errors.append("token2wav manifest is missing components: " + ", ".join(missing))
    speaker = components.get("speaker_encoder")
    if isinstance(speaker, Mapping) and speaker.get("component") not in (None, "campplus"):
        errors.append("token2wav speaker_encoder component is not campplus")
    for name, component in components.items():
        if not isinstance(component, Mapping):
            errors.append(f"token2wav.components.{name} must be an object")
            continue
        _check_identity(component, label=f"token2wav.components.{name}", errors=errors)
    config = _read_json(bundle / "config.json")
    config_components = config.get("components")
    if not isinstance(config_components, Mapping):
        errors.append("token2wav.config.components must be an object")
    elif "speaker_encoder" not in config_components:
        errors.append("token2wav.config is missing components.speaker_encoder")
    _check_no_absolute_paths(manifest, label="token2wav.manifest", errors=errors)
    _check_no_absolute_paths(config, label="token2wav.config", errors=errors)
    _check_references(bundle, manifest, label="token2wav.manifest", errors=errors)
    _check_references(bundle, config, label="token2wav.config", errors=errors)
    _check_checksums_file(bundle, errors=errors)
    if not any(
        (bundle / candidate).exists()
        for candidate in (
            "MiniCPM-CamPlusPlus.mlmodelc",
            "MiniCPM-CamPlusPlus.mlpackage",
            "campplus.mlmodelc",
            "campplus.mlpackage",
            "CamPlusPlus.mlmodelc",
            "CamPlusPlus.mlpackage",
        )
    ):
        speaker_path = None
        if isinstance(speaker, Mapping):
            weights = speaker.get("weights")
            if isinstance(weights, Mapping):
                speaker_path = weights.get("path")
            if speaker_path is None:
                artifact = speaker.get("artifact")
                if isinstance(artifact, Mapping):
                    speaker_path = artifact.get("path")
        if not isinstance(speaker_path, str) or not (bundle / speaker_path).exists():
            errors.append("token2wav speaker_encoder CoreML sidecar is missing")


def audit_bundle(bundle: Path, kind: str) -> dict[str, Any]:
    """Audit one bundle and return a JSON-serialisable result."""

    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "kind": kind,
        "path": str(bundle),
        "required_files": list(REQUIRED_FILES[kind]),
        "errors": errors,
        "warnings": warnings,
    }
    if not bundle.is_dir():
        errors.append(f"bundle directory does not exist: {bundle}")
        result["ok"] = False
        return result
    _required_files(bundle, kind, errors=errors)
    manifest: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    manifest_path = bundle / "manifest.json"
    config_path = bundle / "config.json"
    if manifest_path.is_file():
        try:
            manifest = _read_json(manifest_path)
        except ValueError as exc:
            errors.append(f"{kind}.manifest: {exc}")
    if config_path.is_file():
        try:
            config = _read_json(config_path)
        except ValueError as exc:
            errors.append(f"{kind}.config: {exc}")
    if manifest is not None:
        _check_identity(manifest, label=f"{kind}.manifest", errors=errors)
        _check_no_absolute_paths(manifest, label=f"{kind}.manifest", errors=errors)
        _check_references(bundle, manifest, label=f"{kind}.manifest", errors=errors)
    if config is not None:
        # Configs are not the primary provenance record, but checking their
        # identities catches a stale bundle before a runtime loader sees it.
        if any(key in config for key in ("source_revision", "implementation_commit", "upstream_commit")):
            _check_identity(config, label=f"{kind}.config", errors=errors)
        _check_no_absolute_paths(config, label=f"{kind}.config", errors=errors)
        _check_references(bundle, config, label=f"{kind}.config", errors=errors)
    if kind == "vision":
        _check_vision_compute_dtype(config, manifest, errors=errors)
    if kind == "llm" and all((bundle / name).is_file() for name in REQUIRED_FILES[kind]):
        try:
            _audit_llm(bundle, errors=errors, warnings=warnings)
        except ValueError as exc:
            errors.append(str(exc))
    elif kind == "tts":
        _check_checksums_file(bundle, errors=errors)
    elif kind == "token2wav" and manifest is not None and config is not None:
        try:
            _audit_token2wav(bundle, errors=errors, warnings=warnings)
        except ValueError as exc:
            errors.append(str(exc))
    result["ok"] = not errors
    return result


def audit_model_root(
    root: str | Path,
    *,
    llm_bundle: str | Path | None = None,
    audio_bundle: str | Path | None = None,
    vision_bundle: str | Path | None = None,
    tts_bundle: str | Path | None = None,
    token2wav_bundle: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all five bundles, allowing each path to be overridden explicitly."""

    root_path = Path(root).expanduser().resolve()
    overrides = {
        "llm": llm_bundle,
        "audio": audio_bundle,
        "vision": vision_bundle,
        "tts": tts_bundle,
        "token2wav": token2wav_bundle,
    }
    paths = {
        kind: Path(value).expanduser().resolve()
        if value is not None
        else root_path / (DEFAULT_LLM_BUNDLE if kind == "llm" else DEFAULT_BUNDLES[kind])
        for kind, value in overrides.items()
    }
    bundles = {kind: audit_bundle(path, kind) for kind, path in paths.items()}
    return {
        "ok": all(bool(item.get("ok")) for item in bundles.values()),
        "model_revision": MODEL_REPOSITORY_REVISION,
        "demo_commit": DEMO_IMPLEMENTATION_COMMIT,
        "root": str(root_path),
        "bundles": bundles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path, help="Parent directory containing native MiniCPM bundles")
    parser.add_argument("--llm-bundle", type=Path, help="Explicit 8-bit or dense BF16 LLM bundle directory")
    parser.add_argument("--audio-bundle", type=Path, help="Explicit audio bundle directory")
    parser.add_argument("--vision-bundle", type=Path, help="Explicit vision bundle directory")
    parser.add_argument("--tts-bundle", type=Path, help="Explicit semantic TTS bundle directory")
    parser.add_argument("--token2wav-bundle", type=Path, help="Explicit Token2Wav bundle directory")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    report = audit_model_root(
        args.model_root,
        llm_bundle=args.llm_bundle,
        audio_bundle=args.audio_bundle,
        vision_bundle=args.vision_bundle,
        tts_bundle=args.tts_bundle,
        token2wav_bundle=args.token2wav_bundle,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    # Keep a single parser definition while making ``main(argv)`` convenient
    # for the tempfile regression tests.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--llm-bundle", type=Path)
    parser.add_argument("--audio-bundle", type=Path)
    parser.add_argument("--vision-bundle", type=Path)
    parser.add_argument("--tts-bundle", type=Path)
    parser.add_argument("--token2wav-bundle", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
