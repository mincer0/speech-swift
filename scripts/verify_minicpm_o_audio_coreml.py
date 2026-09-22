#!/usr/bin/env python3
"""
Validate the Core ML audio-encoder export before committing to the Swift work.

Two questions, both answered without touching Swift:

1. Cold start - how long does loading the 583 MB mlpackage take the first time
   (when Core ML must compile it for the ANE) versus a warm load? If the first
   load costs tens of seconds it would slow the service start-up unacceptably.
2. Numerics on real speech - run the torch reference and the ANE model over the
   mel of a real utterance (tools/fixtures/understand_test.wav, macOS `say`) and
   report per-position cosine similarity, not just max|diff|. Also evaluate the
   projection layer, because that is the tensor the LLM actually consumes.

  .venv-minicpmo/bin/python scripts/verify_minicpm_o_audio_coreml.py
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/Users/mincer/项目/s2s/models/MiniCPM-o-4_5")
PACKAGE = Path("/Users/mincer/项目/s2s/models/MiniCPM-o-4_5-audio-ane/MiniCPM-o-4_5-audio-encoder.mlpackage")
SPEECH = Path("/Users/mincer/项目/s2s/tools/fixtures/understand_test.wav")

sys.path.insert(0, str(ROOT / "scripts"))
from export_minicpm_o_audio_coreml import (  # noqa: E402
    ensure_torchvision_compat,
    load_apm_weights,
    load_audio_config,
)

_TORCHVISION_COMPAT_LIB = None


def load_encoder():
    ensure_torchvision_compat()
    import tempfile

    pkg_root = Path(tempfile.mkdtemp(prefix="minicpm_o45_verify_"))
    pkg = pkg_root / "minicpm_o45"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for source in MODEL_DIR.glob("*.py"):
        (pkg / source.name).symlink_to(source)
    sys.path.insert(0, str(pkg_root))
    from minicpm_o45.modeling_minicpmo import MiniCPMWhisperEncoder

    config = load_audio_config(MODEL_DIR)
    model = MiniCPMWhisperEncoder(config)
    state = load_apm_weights(MODEL_DIR)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.requires_grad_(False)
    return model, config


def mel_from_wav(path: Path, config) -> np.ndarray:
    """Log-mel matching the encoder's expected [1, n_mels, T] layout.

    The audio tower is whisper-medium (`_name_or_path` in audio_config), so the
    mel front-end uses Whisper's standard constants. 16000 Hz / hop 160 gives
    exactly 100 frames per second, which is the input the export was traced
    with and matches the 50 positions per unit the pipeline reports (conv2
    halves it).
    """
    sampling_rate = 16_000
    n_fft = 400
    hop_length = 160
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    target = sampling_rate
    if sr != target:  # linear resample
        ratio = sr / target
        idx = np.arange(int(len(samples) / ratio)) * ratio
        lo = idx.astype(int)
        hi = np.minimum(lo + 1, len(samples) - 1)
        frac = idx - lo
        samples = samples[lo] * (1 - frac) + samples[hi] * frac
    import librosa

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=target,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=config.num_mel_bins,
    )
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel.astype(np.float32)


def cos_per_position(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    return (a2 * b2).sum(1) / (np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1) + 1e-9)


def main():
    import coremltools as ct

    print("== 1. 冷启动 / 加载耗时 ==")
    t0 = time.perf_counter()
    ane = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_AND_NE)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_AND_NE)
    warm = time.perf_counter() - t0
    print(f"  首次加载（含 ANE 编译）: {cold:.2f} s")
    print(f"  二次加载           : {warm:.2f} s")

    print("\n== 2. 真实语音数值验证 ==")
    model, config = load_encoder()
    frames = 100
    mel = mel_from_wav(SPEECH, config)
    print(f"  梅尔: {mel.shape}（取前 {frames} 帧，不足补零）")
    if mel.shape[1] < frames:
        mel = np.pad(mel, ((0, 0), (0, frames - mel.shape[1])))
    mel = mel[:, :frames][None, ...]

    with torch.no_grad():
        ref = model(torch.from_numpy(mel), attention_mask=None, use_cache=False).last_hidden_state.numpy()

    t_first0 = time.perf_counter()
    got = np.asarray(list(ane.predict({"mel": mel}).values())[0])
    first_ms = (time.perf_counter() - t_first0) * 1000
    warmup = time.perf_counter()
    for _ in range(3):
        ane.predict({"mel": mel})
    print(f"  预热 3 次共 {time.perf_counter() - warmup:.2f} s")
    t0 = time.perf_counter()
    for _ in range(20):
        ane.predict({"mel": mel})
    ane_ms = (time.perf_counter() - t0) / 20 * 1000

    c = cos_per_position(got, ref)
    print(f"  编码器输出: cos 平均 {c.mean():.6f}  最低 {c.min():.6f}  最差位置 #{int(c.argmin())}")
    print(f"  相对误差  : max|diff|/max|ref| = {np.abs(got - ref).max()/ (np.abs(ref).max()+1e-9)*100:.2f}%")
    print(f"  首次推理: {first_ms:.1f} ms（含 ANE 首次编译/预热）")
    print(f"  稳态推理: {ane_ms:.1f} ms（20 次平均）")

    low = int((c < 0.99).sum())
    print(f"  cos < 0.99 的位置: {low}/{len(c)}")


if __name__ == "__main__":
    main()
