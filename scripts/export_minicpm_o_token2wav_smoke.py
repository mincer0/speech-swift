#!/usr/bin/env python3
"""Export a real MiniCPM prompt and valid S3 speech tokens for Swift smoke tests.

This intentionally loads only the S3Tokenizer and CAMPPlus prompt frontends.
CosyVoice Flow and HiFT remain unloaded so their first full inference happens
in the native MLX Swift executable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import s3tokenizer
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from safetensors.numpy import save_file
from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompt_wav", type=Path)
    parser.add_argument("content_wav", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--content-seconds", type=float, default=3.2)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def device_from_argument(value: str) -> torch.device:
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(value)


def load_mono(path: Path, sample_rate: int, seconds: float | None = None) -> torch.Tensor:
    audio, source_rate = torchaudio.load(str(path), backend="soundfile")
    audio = audio.mean(dim=0)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if seconds is not None:
        audio = audio[: int(round(seconds * sample_rate))]
    return audio.contiguous()


@torch.inference_mode()
def speech_tokens(
    tokenizer: torch.nn.Module,
    audio_16k: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    mel = s3tokenizer.log_mel_spectrogram(audio_16k)
    mel, mel_lengths = s3tokenizer.padding([mel])
    tokens, lengths = tokenizer.quantize(
        mel.to(device), mel_lengths.to(device)
    )
    length = int(lengths.reshape(-1)[0].item())
    return tokens[:, :length].to(dtype=torch.int32, device="cpu")


def speaker_embedding(model_path: Path, audio_16k: torch.Tensor) -> np.ndarray:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    features = kaldi.fbank(
        audio_16k.unsqueeze(0),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16_000,
    )
    features = features - features.mean(dim=0, keepdim=True)
    output = session.run(
        None,
        {session.get_inputs()[0].name: features.unsqueeze(0).numpy()},
    )[0]
    return np.asarray(output, dtype=np.float32)


def prompt_mel(audio_24k: torch.Tensor, target_frames: int) -> torch.Tensor:
    mel = mel_spectrogram(audio_24k.unsqueeze(0)).transpose(1, 2)
    if mel.shape[1] < target_frames:
        padding = mel[:, -1:, :].expand(-1, target_frames - mel.shape[1], -1)
        mel = torch.cat([mel, padding], dim=1)
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames, :]
    return mel.transpose(1, 2).contiguous().to(dtype=torch.float32, device="cpu")


def main() -> None:
    args = parse_args()
    token2wav = args.model_dir / "assets" / "token2wav"
    tokenizer_path = token2wav / "speech_tokenizer_v2_25hz.onnx"
    campplus_path = token2wav / "campplus.onnx"
    for path in (tokenizer_path, campplus_path, args.prompt_wav, args.content_wav):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = device_from_argument(args.device)
    prompt_16k = load_mono(args.prompt_wav, 16_000)
    content_16k = load_mono(args.content_wav, 16_000, args.content_seconds)

    tokenizer = s3tokenizer.load_model(str(tokenizer_path)).to(device).eval()
    prompt_tokens = speech_tokens(tokenizer, prompt_16k, device)
    generated_tokens = speech_tokens(tokenizer, content_16k, device)
    del tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    prompt_24k = load_mono(args.prompt_wav, 24_000)
    mel = prompt_mel(prompt_24k, target_frames=prompt_tokens.shape[1] * 2)
    speaker = speaker_embedding(campplus_path, prompt_16k)

    arrays = {
        "prompt_tokens": prompt_tokens.numpy(),
        "prompt_mel": mel.numpy(),
        "speaker_embedding": speaker,
        "generated_tokens": generated_tokens.numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        arrays,
        str(args.output),
        metadata={
            "prompt_wav": str(args.prompt_wav),
            "content_wav": str(args.content_wav),
            "token_rate_hz": "25",
            "mel_rate_hz": "50",
        },
    )
    print(
        {
            "output": str(args.output),
            "device": str(device),
            "prompt_tokens": int(prompt_tokens.shape[1]),
            "prompt_mel_frames": int(mel.shape[2]),
            "generated_tokens": int(generated_tokens.shape[1]),
        }
    )


if __name__ == "__main__":
    main()
