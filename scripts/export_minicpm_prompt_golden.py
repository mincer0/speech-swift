#!/usr/bin/env python3
"""Export MiniCPM-o Token2Wav prompt-front-end parity vectors.

The native Swift prompt path has three independent frontends and each one has
different framing semantics:

* S3Tokenizer-v2: 16 kHz, 400/160 STFT, 128 mel bins, 25 Hz codes;
* Flow prompt mel: 24 kHz, 1920/480 STFT (owned by ``PromptPreparation``);
* CAM++: Kaldi fbank, variable raw frame count adapted to ``[1, 500, 80]``.

This fixture deliberately stores PCM plus every intermediate S3/CAM++ tensor.
It can therefore catch an error in framing or resampling before a complete
Token2Wav model is allocated.  The ONNX graphs are used only as a reference
oracle; runtime Swift uses the converted S3 safetensors and a compiled CAM++
CoreML model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


S3_ONNX_SHA256 = "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71"
CAMPPLUS_ONNX_SHA256 = "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"
# Canonical prompt preparation is supplied by the pinned MiniCPM-o Demo
# dependency, not by the optional long-audio extension below.  Keep the
# package and wrapper hash in the fixture so a future exporter cannot silently
# change framing or padding semantics.
MINICPMO_UTILS_PACKAGE = "minicpmo-utils"
MINICPMO_UTILS_VERSION = "1.0.6"
STEPAUDIO2_TOKEN2WAV_SHA256 = "0598c25ad9883f75d6d5ca1e9276c03adbfa0357a29442164198ea6931a06efb"
MINICPMO_DEMO_COMMIT = "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
SAMPLE_RATE = 16_000
TARGET_CAM_FRAMES = 500
FLOW_SAMPLE_RATE = 24_000
FLOW_NFFT = 1_920
FLOW_HOP = 480
FLOW_NMELS = 80
S3_SEGMENT_SECONDS = 30.0
S3_SEGMENT_OVERLAP_SECONDS = 4.0
S3_TOKEN_RATE = 25
PROMPT_GOLDEN_SCHEMA = "minicpm-o-prompt-golden/v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash stable source bytes without VCS/interpreter cache entries."""

    root = root.resolve()
    ignored = {".git", "__pycache__", ".build", ".venv"}
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ignored for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        files.append((relative.as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def assert_upstream_revision(source: Path, expected: str) -> tuple[str, str]:
    """Bind the oracle to the exact official Demo checkout and tree."""

    source = source.expanduser().resolve()
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"upstream source must be a git checkout at {expected}: {source}"
        ) from exc
    if actual != expected:
        raise SystemExit(
            "upstream MiniCPM-o Demo revision mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual, sha256_tree(source)


def tensor_metadata(array: np.ndarray) -> dict[str, Any]:
    """Describe the exact bytes written to the safetensors artifact."""

    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    values = contiguous.astype(np.float32, copy=False).reshape(-1)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "numel": int(contiguous.size),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sum": float(values.sum()) if values.size else 0.0,
        "sum_sq": float(np.square(values).sum()) if values.size else 0.0,
        "max_abs": float(np.abs(values).max()) if values.size else 0.0,
        "mean_abs": float(np.abs(values).mean()) if values.size else 0.0,
        "finite": bool(np.isfinite(values).all()),
    }


def assert_no_absolute_paths(value: Any) -> None:
    """Golden metadata must be portable across machines."""

    if isinstance(value, str) and value.startswith("/"):
        raise RuntimeError("prompt golden metadata contains an absolute path")
    if isinstance(value, dict):
        for child in value.values():
            assert_no_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_absolute_paths(child)


def validate_stepaudio2_provenance(*, allow_mismatch: bool = False) -> dict[str, Any]:
    """Validate the pinned minicpmo-utils wrapper without importing it.

    ``Token2wav._prepare_prompt`` is part of the Python distribution rather
    than an ONNX graph, so a fixture must bind both its package version and
    exact wrapper bytes.  We inspect distribution metadata and hash the file
    in place; no stepaudio2 module (and therefore no CUDA initialization) is
    imported.  Returned metadata intentionally contains no local filesystem
    path.
    """

    expected_version = MINICPMO_UTILS_VERSION
    expected_hash = STEPAUDIO2_TOKEN2WAV_SHA256
    try:
        distribution = importlib.metadata.distribution(MINICPMO_UTILS_PACKAGE)
    except importlib.metadata.PackageNotFoundError as exc:
        if not allow_mismatch:
            raise SystemExit(
                f"required {MINICPMO_UTILS_PACKAGE}=={expected_version} "
                "distribution is not installed"
            ) from exc
        return {
            "package": MINICPMO_UTILS_PACKAGE,
            "version_expected": expected_version,
            "version_actual": None,
            "wrapper_sha256_expected": expected_hash,
            "wrapper_sha256_actual": None,
            "override": True,
            "status": "missing_distribution_override",
        }

    actual_version = distribution.version
    wrapper_path: Path | None = None
    for file in distribution.files or ():
        if file.as_posix() == "stepaudio2/token2wav.py":
            wrapper_path = Path(distribution.locate_file(file))
            break
    actual_hash = sha256(wrapper_path) if wrapper_path is not None and wrapper_path.is_file() else None
    mismatch = actual_version != expected_version or actual_hash != expected_hash
    if mismatch and not allow_mismatch:
        details = (
            f"{MINICPMO_UTILS_PACKAGE} provenance mismatch: "
            f"version {actual_version!r} (expected {expected_version!r}), "
            f"token2wav.py sha256 {actual_hash!r} (expected {expected_hash!r})"
        )
        raise SystemExit(details)
    return {
        "package": MINICPMO_UTILS_PACKAGE,
        "version_expected": expected_version,
        "version_actual": actual_version,
        "wrapper_sha256_expected": expected_hash,
        "wrapper_sha256_actual": actual_hash,
        "override": bool(mismatch),
        "status": "ok" if not mismatch else "mismatch_override",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="original MiniCPM-o bundle containing assets/token2wav/*.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "Tests/MiniCPMToken2WavTests/Fixtures/minicpm-prompt-golden.safetensors",
    )
    parser.add_argument("--s3-onnx", type=Path, default=None)
    parser.add_argument("--campplus-onnx", type=Path, default=None)
    parser.add_argument(
        "--durations",
        type=str,
        default="short=0.25,one_second=1.0,five_seconds=5.0,exact=5.015",
        help="comma-separated name=seconds entries (canonical prompt cases are <=30 s)",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--frequency", type=float, default=220.0)
    parser.add_argument(
        "--s3-pad-from-head",
        action="store_true",
        help=(
            "pad a short final >30 s S3 segment with samples from the start; "
            "matches the optional s3tokenizer-long pad_from_head=True path"
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-long-extension",
        action="store_true",
        help="allow the separately labelled external 30 s/overlap S3 extension",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="allow a non-pinned ONNX graph for deliberate experiments",
    )
    parser.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="allow a non-pinned minicpmo-utils wrapper for deliberate experiments",
    )
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=Path("/tmp/MiniCPM-o-Demo"),
        help="pinned OpenBMB/MiniCPM-o-Demo checkout",
    )
    parser.add_argument(
        "--upstream-revision",
        default=MINICPMO_DEMO_COMMIT,
        help="expected official Demo git commit",
    )
    return parser.parse_args()


def parse_durations(spec: str) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    names: set[str] = set()
    for item in spec.split(","):
        name, separator, raw_seconds = item.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid duration entry {item!r}; expected name=seconds")
        if name in names:
            raise ValueError(f"duplicate duration case name {name!r}")
        seconds = float(raw_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"duration for {name!r} must be positive")
        names.add(name)
        result.append((name, seconds))
    if not result:
        raise ValueError("at least one duration case is required")
    return result


def load_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import onnxruntime
        import s3tokenizer
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This exporter requires onnxruntime, s3tokenizer, torch, and torchaudio"
        ) from exc
    return onnxruntime, s3tokenizer, torch, torchaudio, kaldi


def deterministic_pcm(seconds: float, *, seed: int, frequency: float) -> np.ndarray:
    count = int(round(seconds * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    # A quiet second harmonic and a deterministic noise bed exercise both
    # low-energy mel floors and the CAM++ CMN path without requiring a wav file.
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count).astype(np.float32) * 0.002
    pcm = (
        0.15 * np.sin(2 * np.pi * frequency * time)
        + 0.04 * np.sin(2 * np.pi * frequency * 2.0 * time + 0.37)
        + noise
    )
    return pcm.astype(np.float32, copy=False)


def s3_mel(audio: np.ndarray, s3tokenizer: Any, torch: Any) -> np.ndarray:
    tensor = torch.from_numpy(audio)
    mel = s3tokenizer.log_mel_spectrogram(tensor, n_mels=128)
    return mel.detach().cpu().numpy().astype(np.float32, copy=False)


def speech_tokens(
    session: Any,
    mel: np.ndarray,
    s3tokenizer: Any,
    torch: Any,
) -> np.ndarray:
    # Match stepaudio2.Token2wav._prepare_prompt exactly: padding() returns
    # [B, 128, T] plus the unmodified frame length.  For one clip this is a
    # no-op in shape, but it is semantically important and makes the fixture
    # fail if a future batch/padding implementation changes.
    mel_tensor = torch.from_numpy(mel.astype(np.float32, copy=False))
    padded, lengths = s3tokenizer.padding([mel_tensor])
    output = session.run(
        None,
        {
            "feats": padded.detach().cpu().numpy().astype(np.float32, copy=False),
            "feats_length": lengths.detach().cpu().numpy().astype(np.int32, copy=False),
        },
    )
    if not output:
        raise RuntimeError("S3Tokenizer ONNX returned no outputs")
    return np.asarray(output[0], dtype=np.int32)


def raw_campplus_fbank(audio: np.ndarray, torch: Any, kaldi: Any) -> np.ndarray:
    waveform = torch.from_numpy(audio).unsqueeze(0)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
    )
    # The original MiniCPM bridge performs per-bin CMN before CAMPPlus.
    features = features - features.mean(dim=0, keepdim=True)
    return features.detach().cpu().numpy().astype(np.float32, copy=False)


def _slaney_filterbank(
    *, sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    """Build the same area-normalized Slaney bank used by Swift."""

    def hz_to_mel(hz: float) -> float:
        f_sp = 200.0 / 3.0
        min_log_hz = 1_000.0
        min_log_mel = min_log_hz / f_sp
        log_step = np.log(6.4) / 27.0
        return hz / f_sp if hz < min_log_hz else min_log_mel + np.log(hz / min_log_hz) / log_step

    def mel_to_hz(mel: float) -> float:
        f_sp = 200.0 / 3.0
        min_log_mel = 15.0
        log_step = np.log(6.4) / 27.0
        return f_sp * mel if mel < min_log_mel else 1_000.0 * np.exp(log_step * (mel - min_log_mel))

    low = hz_to_mel(f_min)
    high = hz_to_mel(f_max)
    points = np.asarray(
        [mel_to_hz(low + i / (n_mels + 1) * (high - low)) for i in range(n_mels + 2)],
        dtype=np.float32,
    )
    bins = n_fft // 2 + 1
    bank = np.zeros((n_mels, bins), dtype=np.float32)
    for mel in range(n_mels):
        left, center, right = points[mel : mel + 3]
        norm = 2.0 / max(float(right - left), 1e-12)
        frequencies = np.arange(bins, dtype=np.float32) * sample_rate / n_fft
        rising = (frequencies - left) / max(float(center - left), 1e-12)
        falling = (right - frequencies) / max(float(right - center), 1e-12)
        bank[mel] = np.maximum(0.0, np.minimum(rising, falling)) * norm
    return bank


def flow_prompt_mel(audio: np.ndarray, torch: Any) -> np.ndarray:
    """Export the 24 kHz/80-bin magnitude mel used by Token2Wav Flow."""

    waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
    window = torch.hann_window(FLOW_NFFT, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=FLOW_NFFT,
        hop_length=FLOW_HOP,
        win_length=FLOW_NFFT,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)
    bank = torch.from_numpy(
        _slaney_filterbank(
            sample_rate=FLOW_SAMPLE_RATE,
            n_fft=FLOW_NFFT,
            n_mels=FLOW_NMELS,
            f_min=0.0,
            f_max=FLOW_SAMPLE_RATE / 2,
        )
    )
    mel = bank @ magnitude
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.detach().cpu().numpy().astype(np.float32, copy=False)[None, :, :]


def align_flow_mel(mel: np.ndarray, target_frames: int) -> np.ndarray:
    """Match MiniCPMToken2WavPromptPreparer's copy/last-frame alignment."""

    if mel.ndim != 3 or mel.shape[0] != 1:
        raise ValueError(f"flow mel must have shape [1, channels, frames], got {mel.shape}")
    channels, source_frames = mel.shape[1], mel.shape[2]
    output = np.zeros((1, channels, target_frames), dtype=np.float32)
    copied = min(source_frames, target_frames)
    if copied:
        output[:, :, :copied] = mel[:, :, :copied]
        if copied < target_frames:
            output[:, :, copied:] = mel[:, :, copied - 1 : copied]
    return output


def resample_audio(audio: np.ndarray, torch: Any, torchaudio: Any, target_rate: int) -> np.ndarray:
    """Use torchaudio's default sinc/Hann resampler, like the Swift bridge."""

    if target_rate == SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    tensor = torch.from_numpy(audio.astype(np.float32, copy=False))
    result = torchaudio.functional.resample(tensor, SAMPLE_RATE, target_rate)
    return result.detach().cpu().numpy().astype(np.float32, copy=False)


def s3_long_segments(
    pcm: np.ndarray,
    *,
    session: Any,
    s3tokenizer: Any,
    torch: Any,
    arrays: dict[str, np.ndarray],
    name: str,
    pad_from_head: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run the external s3tokenizer-long compatible segment/merge path.

    The fixed MiniCPM-o Demo itself chunks its *Whisper* audio features every
    30 seconds without overlap.  This helper is specifically for the separate
    S3 speech-token prompt path and mirrors ``s3tokenizer-long``: 30-second
    windows, 4-second overlap, and optional ``pad_from_head`` for the short
    final window.
    """

    segment_samples = int(round(S3_SEGMENT_SECONDS * SAMPLE_RATE))
    overlap_samples = int(round(S3_SEGMENT_OVERLAP_SECONDS * SAMPLE_RATE))
    step_samples = segment_samples - overlap_samples
    if segment_samples <= overlap_samples:
        raise ValueError("S3 segment overlap must be shorter than the segment")

    segments: list[dict[str, Any]] = []
    token_lists: list[list[int]] = []
    start = 0
    segment_index = 0
    while start < pcm.shape[0]:
        end = min(start + segment_samples, pcm.shape[0])
        segment_pcm = pcm[start:end]
        actual_sample_count = int(segment_pcm.shape[0])
        padded_sample_count = 0
        if pad_from_head and actual_sample_count < segment_samples:
            padded_sample_count = segment_samples - actual_sample_count
            # The reference helper uses waveform[:pad_length] (not zeros),
            # intentionally repeating the beginning of the clip at the tail.
            segment_pcm = np.concatenate(
                [segment_pcm, pcm[:padded_sample_count]], axis=0
            )
        segment_mel = s3_mel(segment_pcm, s3tokenizer, torch)
        segment_tokens = speech_tokens(session, segment_mel, s3tokenizer, torch)
        segment_tokens = np.asarray(segment_tokens, dtype=np.int32).reshape(1, -1)
        mel_key = f"{name}.segment{segment_index}.s3_mel"
        token_key = f"{name}.segment{segment_index}.s3_tokens"
        arrays[mel_key] = segment_mel[None, :, :]
        arrays[token_key] = segment_tokens
        token_lists.append(segment_tokens[0].tolist())
        segments.append(
            {
                "index": segment_index,
                "sample_start": int(start),
                "sample_end": int(end),
                "actual_sample_count": actual_sample_count,
                "padded_from_head": bool(padded_sample_count > 0),
                "padded_sample_count": padded_sample_count,
                "model_sample_count": int(segment_pcm.shape[0]),
                "duration_seconds": float(end - start) / SAMPLE_RATE,
                "s3_mel_key": mel_key,
                "s3_mel_shape": list(arrays[mel_key].shape),
                "s3_mel_length": int(segment_mel.shape[1]),
                "s3_token_key": token_key,
                "s3_token_shape": list(segment_tokens.shape),
                "s3_token_length": int(segment_tokens.shape[1]),
            }
        )
        segment_index += 1
        if end >= pcm.shape[0]:
            break
        start += step_samples

    merged = s3tokenizer.merge_tokenized_segments(
        token_lists,
        overlap=int(S3_SEGMENT_OVERLAP_SECONDS),
        token_rate=S3_TOKEN_RATE,
    )
    return np.asarray([merged], dtype=np.int32), segments


def adapt_campplus(raw: np.ndarray, target_frames: int = TARGET_CAM_FRAMES) -> tuple[np.ndarray, str]:
    frames, bins = raw.shape
    if frames == 0:
        return np.zeros((target_frames, bins), dtype=np.float32), "emptyPadded"
    if frames < target_frames:
        indices = np.arange(target_frames, dtype=np.int64) % frames
        return raw[indices].astype(np.float32, copy=False), "tiledShort"
    if frames > target_frames:
        offset = (frames - target_frames) // 2
        return raw[offset : offset + target_frames].astype(np.float32, copy=False), "centerCroppedLong"
    return raw.astype(np.float32, copy=False), "exact"


def main() -> None:
    args = parse_args()
    # Reject a non-canonical long-audio request before loading any ONNX graph
    # or Python model.  The fixed Demo does not define the overlap/merge path.
    duration_cases = parse_durations(args.durations)
    has_long_case = any(
        seconds > S3_SEGMENT_SECONDS for _, seconds in duration_cases
    )
    if has_long_case and not args.allow_noncanonical_long_extension:
        raise SystemExit(
            "canonical MiniCPM-o Token2wav prompt export does not define an "
            ">30 s overlap/merge path; pass --allow-noncanonical-long-extension "
            "to run the explicitly non-Demo extension"
        )
    upstream_revision, upstream_tree = assert_upstream_revision(
        args.upstream_source, args.upstream_revision
    )
    provenance = validate_stepaudio2_provenance(
        allow_mismatch=args.allow_provenance_mismatch
    )
    model_dir = args.model_dir.expanduser().resolve()
    token2wav = model_dir / "assets/token2wav"
    s3_path = (args.s3_onnx or token2wav / "speech_tokenizer_v2_25hz.onnx").expanduser().resolve()
    camp_path = (args.campplus_onnx or token2wav / "campplus.onnx").expanduser().resolve()
    for path in (s3_path, camp_path):
        if not path.is_file():
            raise SystemExit(f"missing ONNX source graph: {path}")

    source_hashes = {
        "s3_tokenizer_onnx": sha256(s3_path),
        "campplus_onnx": sha256(camp_path),
    }
    if not args.allow_hash_mismatch:
        if source_hashes["s3_tokenizer_onnx"] != S3_ONNX_SHA256:
            raise SystemExit("S3Tokenizer ONNX hash does not match the pinned graph")
        if source_hashes["campplus_onnx"] != CAMPPLUS_ONNX_SHA256:
            raise SystemExit("CAM++ ONNX hash does not match the pinned graph")
    onnx_hash_override = any(
        source_hashes[name] != expected
        for name, expected in (
            ("s3_tokenizer_onnx", S3_ONNX_SHA256),
            ("campplus_onnx", CAMPPLUS_ONNX_SHA256),
        )
    )

    onnxruntime, s3tokenizer, torch, torchaudio, kaldi = load_dependencies()
    providers = ["CPUExecutionProvider"]
    s3_session = onnxruntime.InferenceSession(str(s3_path), providers=providers)
    camp_session = onnxruntime.InferenceSession(str(camp_path), providers=providers)
    s3_input_names = {entry.name for entry in s3_session.get_inputs()}
    camp_input_name = camp_session.get_inputs()[0].name
    camp_output_name = camp_session.get_outputs()[0].name
    if not {"feats", "feats_length"}.issubset(s3_input_names):
        raise SystemExit("unexpected S3Tokenizer ONNX input names")

    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for case_index, (name, seconds) in enumerate(duration_cases):
        pcm = deterministic_pcm(
            seconds,
            seed=args.seed + case_index,
            frequency=args.frequency + case_index * 17.0,
        )
        mel = s3_mel(pcm, s3tokenizer, torch)
        is_long = seconds > S3_SEGMENT_SECONDS
        segments: list[dict[str, Any]] = []
        if is_long:
            # This is intentionally not part of canonical Demo parity.  It is
            # an opt-in external s3tokenizer-long-compatible extension.
            tokens, segments = s3_long_segments(
                pcm,
                session=s3_session,
                s3tokenizer=s3tokenizer,
                torch=torch,
                arrays=arrays,
                name=name,
                pad_from_head=args.s3_pad_from_head,
            )
        else:
            tokens = speech_tokens(s3_session, mel, s3tokenizer, torch)
            tokens = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        merged_token_key = f"{name}.s3_tokens_merged"
        direct_token_key = f"{name}.s3_tokens_direct"
        raw = raw_campplus_fbank(pcm, torch, kaldi)
        fixed, adaptation = adapt_campplus(raw)
        speaker = camp_session.run(
            [camp_output_name],
            {camp_input_name: fixed[None, :, :].astype(np.float32, copy=False)},
        )[0]
        speaker = np.asarray(speaker, dtype=np.float32)

        flow_audio = resample_audio(pcm, torch, torchaudio, FLOW_SAMPLE_RATE)
        flow_mel = flow_prompt_mel(flow_audio, torch)
        flow_prompt = align_flow_mel(flow_mel, int(tokens.shape[1]) * 2)
        expected_flow_frames = int(tokens.shape[1]) * 2
        if flow_prompt.shape[2] != expected_flow_frames:
            raise RuntimeError(
                f"{name}: Flow prompt frames {flow_prompt.shape[2]} "
                f"do not equal 2x speech-token count {expected_flow_frames}"
            )

        arrays[f"{name}.pcm"] = pcm
        arrays[f"{name}.s3_mel"] = mel[None, :, :]
        arrays[f"{name}.s3_mel_length"] = np.asarray([mel.shape[1]], dtype=np.int32)
        arrays[f"{name}.s3_tokens"] = tokens
        arrays[merged_token_key] = tokens
        # Keep the direct one-pass output for short cases as a diagnostic.
        # A full-window ONNX invocation is not a valid long-audio reference,
        # so long cases intentionally omit this key.
        if not is_long:
            arrays[direct_token_key] = tokens
        arrays[f"{name}.s3_token_length"] = np.asarray([tokens.shape[1]], dtype=np.int32)
        arrays[f"{name}.flow_mel"] = flow_mel
        arrays[f"{name}.flow_prompt_mel"] = flow_prompt
        arrays[f"{name}.cam_raw_fbank"] = raw[None, :, :]
        arrays[f"{name}.cam_fixed_fbank"] = fixed[None, :, :]
        arrays[f"{name}.cam_embedding"] = speaker
        cases.append(
            {
                "name": name,
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(pcm.shape[0]),
                "duration_seconds": seconds,
                "pcm_key": f"{name}.pcm",
                "s3_mel_key": f"{name}.s3_mel",
                "s3_mel_length_key": f"{name}.s3_mel_length",
                "s3_token_key": f"{name}.s3_tokens",
                "s3_token_length_key": f"{name}.s3_token_length",
                "s3_tokens_merged_key": merged_token_key,
                "s3_tokens_direct_key": None if is_long else direct_token_key,
                "cam_raw_fbank_key": f"{name}.cam_raw_fbank",
                "cam_fixed_fbank_key": f"{name}.cam_fixed_fbank",
                "cam_embedding_key": f"{name}.cam_embedding",
                "flow_mel_key": f"{name}.flow_mel",
                "flow_prompt_mel_key": f"{name}.flow_prompt_mel",
                "s3_mel_shape": list(arrays[f"{name}.s3_mel"].shape),
                "s3_token_shape": list(tokens.shape),
                "s3_mel_length": int(mel.shape[1]),
                "s3_token_length": int(tokens.shape[1]),
                "flow_mel_shape": list(flow_mel.shape),
                "flow_mel_length": int(flow_mel.shape[2]),
                "flow_prompt_mel_shape": list(flow_prompt.shape),
                "flow_prompt_mel_length": int(flow_prompt.shape[2]),
                "flow_prompt_alignment": "speech_token_count_times_2",
                "cam_raw_frame_count": int(raw.shape[0]),
                "cam_fixed_shape": [1, TARGET_CAM_FRAMES, 80],
                "cam_adaptation": adaptation,
                "s3_long_segmented": is_long,
                "s3_long_backend": (
                    "external_s3tokenizer_long_compatible_non_demo_extension"
                    if is_long else None
                ),
                "s3_segment_seconds": S3_SEGMENT_SECONDS if is_long else None,
                "s3_segment_overlap_seconds": S3_SEGMENT_OVERLAP_SECONDS if is_long else None,
                "s3_segments": segments,
            }
        )

    metadata = {
        "schema": PROMPT_GOLDEN_SCHEMA,
        "format": 3,
        "oracle_backend": "official_onnx_and_pytorch_frontends",
        "upstream": {
            "repository": "OpenBMB/MiniCPM-o-Demo",
            "revision": upstream_revision,
            "revision_kind": "git_commit",
            "tree_sha256": upstream_tree,
        },
        "source": {
            "model_id": model_dir.name,
            "canonical_prompt_oracle": "stepaudio2.Token2wav._prepare_prompt",
            "canonical_dependency": provenance["package"],
            # Backward-compatible expected-value aliases consumed by the
            # existing parity test; the explicit actual/expected fields below
            # are authoritative for provenance auditing.
            "canonical_dependency_version": provenance["version_expected"],
            "canonical_dependency_version_expected": provenance["version_expected"],
            "canonical_dependency_version_actual": provenance["version_actual"],
            "canonical_token2wav_wrapper_sha256": provenance[
                "wrapper_sha256_expected"
            ],
            "canonical_token2wav_wrapper_sha256_expected": provenance[
                "wrapper_sha256_expected"
            ],
            "canonical_token2wav_wrapper_sha256_actual": provenance[
                "wrapper_sha256_actual"
            ],
            "canonical_provenance_status": provenance["status"],
            "canonical_provenance_override": provenance["override"],
            "canonical_demo_commit": MINICPMO_DEMO_COMMIT,
            "canonical_upstream_tree_sha256": upstream_tree,
            "s3_tokenizer_onnx": s3_path.name,
            "campplus_onnx": camp_path.name,
            "s3_tokenizer_onnx_sha256": source_hashes["s3_tokenizer_onnx"],
            "campplus_onnx_sha256": source_hashes["campplus_onnx"],
            "expected_s3_tokenizer_onnx_sha256": S3_ONNX_SHA256,
            "expected_campplus_onnx_sha256": CAMPPLUS_ONNX_SHA256,
            "onnx_hash_override": onnx_hash_override,
            "s3_frontend": "s3tokenizer.log_mel_spectrogram(n_mels=128)",
            "s3_sample_rate": SAMPLE_RATE,
            "s3_fft": 400,
            "s3_hop": 160,
            "s3_frame_rate_hz": 100,
            "s3_token_rate_hz": 25,
            "s3_long_segment_seconds": (
                S3_SEGMENT_SECONDS if args.allow_noncanonical_long_extension else None
            ),
            "s3_long_segment_overlap_seconds": (
                S3_SEGMENT_OVERLAP_SECONDS if args.allow_noncanonical_long_extension else None
            ),
            "s3_long_extension_enabled": bool(args.allow_noncanonical_long_extension),
            "s3_long_backend": (
                "external_s3tokenizer_long_compatible_non_demo_extension"
                if args.allow_noncanonical_long_extension else None
            ),
            "s3_long_pad_from_head": (
                bool(args.s3_pad_from_head) if args.allow_noncanonical_long_extension else None
            ),
            "s3_long_merge": (
                "s3tokenizer.merge_tokenized_segments(overlap=4, token_rate=25)"
                if args.allow_noncanonical_long_extension else None
            ),
            "flow_frontend": "24 kHz STFT magnitude + Slaney mel",
            "flow_sample_rate": FLOW_SAMPLE_RATE,
            "flow_fft": FLOW_NFFT,
            "flow_hop": FLOW_HOP,
            "flow_frame_rate_hz": FLOW_SAMPLE_RATE / FLOW_HOP,
            "flow_mel_frames_per_speech_token": 2,
            "campplus_frontend": "torchaudio.compliance.kaldi.fbank + per-bin CMN",
            "campplus_input_shape": [1, TARGET_CAM_FRAMES, 80],
            "campplus_adaptation": "tile short; center crop long",
            "seed": args.seed,
            "frequency": args.frequency,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("install safetensors to write the prompt fixture") from exc
    arrays = {
        name: np.ascontiguousarray(value)
        for name, value in arrays.items()
    }
    save_file(
        arrays,
        str(args.output),
        metadata={
            "schema": PROMPT_GOLDEN_SCHEMA,
            "fixture_format": "3",
            "oracle_backend": "official_onnx_and_pytorch_frontends",
            "upstream_revision": upstream_revision,
            "s3_tokenizer_onnx_sha256": source_hashes["s3_tokenizer_onnx"],
            "campplus_onnx_sha256": source_hashes["campplus_onnx"],
        },
    )
    metadata["artifact"] = {
        "path": args.output.name,
        "sha256": sha256(args.output),
        "tensor_count": len(arrays),
        "tensors": {
            name: tensor_metadata(value)
            for name, value in sorted(arrays.items())
        },
    }
    # The side-car is intentionally portable.  Absolute paths are useful in a
    # log but are not a valid golden contract and are therefore rejected here.
    assert_no_absolute_paths(metadata)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "metadata": str(metadata_path), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
