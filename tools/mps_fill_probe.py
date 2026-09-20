#!/usr/bin/env python3
"""
Measure the intra-block speech fill ratio with the **official orchestration**.

The duplex loop here is `models/MiniCPM-o-4_5/modeling_minicpmo.py`
(`streaming_generate`), i.e. the upstream PyTorch implementation. Note that
`DuplexConfig` delegates the TTS / LLM / token2wav / audio encoder to the same
Swift/MLX components the native server uses, so this isolates the **duplex
orchestration** (how a unit is chunked, how many speech codes are requested)
rather than the acoustic models themselves:

    use_mlx_tts / use_mlx_llm / use_swift_token2wav / use_swift_audio_encoder = True

Usage:
  .venv-minicpmo/bin/python tools/mps_fill_probe.py [--chunks 20]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicpm_duplex.runtime import MiniCPMDuplexRuntime  # noqa: E402


def voiced_seconds(w, sr, thresh=0.01):
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return 0.0, 0.0
    win = int(sr * 0.02)
    if win < 1:
        return 0.0, 0.0
    n = (len(w) // win) * win
    frames = w[:n].reshape(-1, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return (rms > thresh).sum() * 0.02, len(w) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=20)
    args = ap.parse_args()

    print("loading runtime …", flush=True)
    rt = MiniCPMDuplexRuntime()
    rt.load()
    print("loaded", flush=True)

    prompt = "请给我讲一个很长很长的故事，从头讲到尾，不要停下来。"
    try:
        r = rt.process_text(prompt)
        try:
            rt.accept_pending_result(r)
        except Exception:
            pass
        print(f"text turn ok (text={r.get('text','')[:40]!r})", flush=True)
    except Exception as e:
        print(f"process_text failed: {type(e).__name__}: {e}", flush=True)

    fills, durs, texts = [], [], []
    silence = np.zeros(16_000, dtype=np.float32)
    for i in range(args.chunks):
        t0 = time.time()
        try:
            res = rt.process_audio_chunk(silence, allow_speak=True)
        except Exception as e:
            print(f"  chunk {i} failed: {type(e).__name__}: {e}", flush=True)
            break
        try:
            rt.accept_pending_result(res)
        except Exception:
            pass
        w = res.get("audio_waveform")
        txt = res.get("text") or ""
        if w is not None and len(np.asarray(w).reshape(-1)) > 0:
            v, d = voiced_seconds(w, 24_000)
            fills.append(v)
            durs.append(d)
            texts.append(txt)
            print(f"  unit {i:>2}: {d:.2f}s  语音 {v:.2f}s  填充 {v/d*100 if d else 0:>3.0f}%  "
                  f"({time.time()-t0:.2f}s)  {txt[:28]!r}", flush=True)
        else:
            print(f"  unit {i:>2}: (listen/no audio)  {time.time()-t0:.2f}s", flush=True)

    if fills:
        import statistics
        print(f"\n=== 官方编排 + 我们的组件 ===")
        print(f"  发声 unit: {len(fills)}")
        print(f"  块长平均:  {statistics.mean(durs):.2f}s")
        print(f"  语音平均:  {statistics.mean(fills):.2f}s")
        print(f"  → 填充率:  {statistics.mean(fills)/statistics.mean(durs)*100:.0f}%")
        print(f"  (Swift 原生服务器实测填充率 ≈ 60-70%)")


if __name__ == "__main__":
    main()
