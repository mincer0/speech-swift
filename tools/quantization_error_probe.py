#!/usr/bin/env python3
"""
Offline check: how much numerical error would 8-bit affine quantization add to
the MiniCPM-o speech components?

Answers "is proposal (1) worth the Swift-side work?" without touching the Swift
loaders: quantize the safetensors with mlx, dequantize, and report the relative
error of each weight matrix plus the resulting error of a simulated linear
layer. Run with the venv that has mlx:

  .venv-minicpmo/bin/python tools/quantization_error_probe.py
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx

MODELS = Path("/Users/mincer/项目/s2s/models")
COMPONENTS = {
    "tts": MODELS / "MiniCPM-o-4_5-tts-mlx",
    "token2wav": MODELS / "MiniCPM-o-4_5-token2wav-mlx",
}


def load_weights(root: Path):
    import json
    files = sorted(root.glob("*.safetensors"))
    if not files:
        return {}
    weights = {}
    for f in files:
        try:
            weights.update(mx.load(str(f)))
        except Exception as e:
            print(f"  ! failed to load {f.name}: {e}")
    return weights


def quant_error(weight: mx.array, bits: int, group_size: int):
    """Return (relative weight error, relative linear-output error)."""
    w = weight.astype(mx.float32)
    original_norm = float(mx.sqrt(mx.sum(w * w)))
    if original_norm == 0:
        return 0.0, 0.0
    try:
        qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        dq = mx.dequantize(qw, scales, biases, group_size=group_size, bits=bits)
    except Exception as e:
        return float("nan"), float("nan")
    err = float(mx.sqrt(mx.sum((dq - w) * (dq - w)))) / original_norm

    # Simulated linear: y = x @ W^T with x ~ N(0, 1)
    out_dim, in_dim = w.shape
    x = mx.random.normal((4, in_dim))
    y = x @ w.T
    yq = x @ dq.T
    y_norm = float(mx.sqrt(mx.sum(y * y)))
    out_err = (float(mx.sqrt(mx.sum((yq - y) * (yq - y)))) / y_norm) if y_norm else 0.0
    return err, out_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--top", type=int, default=12, help="show the worst layers")
    args = ap.parse_args()

    for name, root in COMPONENTS.items():
        if not root.exists():
            print(f"{name}: {root} missing")
            continue
        weights = load_weights(root)
        total_bytes = sum(v.nbytes for v in weights.values())
        rows = []
        for key, value in weights.items():
            if value.ndim != 2 or min(value.shape) < 64:
                continue
            err, out_err = quant_error(value, args.bits, args.group_size)
            if err != err:  # NaN
                continue
            rows.append((key, value.shape, value.nbytes, err, out_err))
        if not rows:
            print(f"{name}: no 2-D weight matrices found")
            continue

        quant_bytes = sum(r[2] for r in rows) * (args.bits / 32) + \
            sum(r[2] for r in rows) * (32 / 32) / args.group_size
        rel = [r[3] for r in rows]
        out = [r[4] for r in rows]
        mean_rel = sum(rel) / len(rel)
        max_rel = max(rel)
        mean_out = sum(out) / len(out)
        max_out = max(out)

        print(f"\n=== {name}  ({len(rows)} 个 2-D 权重矩阵, {total_bytes/1e6:.0f} MB) ===")
        print(f"  {args.bits}-bit / group={args.group_size}")
        print(f"  权重相对误差:  平均 {mean_rel*100:.3f}%   最大 {max_rel*100:.3f}%")
        print(f"  线性层输出误差: 平均 {mean_out*100:.3f}%   最大 {max_out*100:.3f}%")
        print(f"  估算权重大小:  {total_bytes/1e6:.0f} MB → {quant_bytes/1e6:.0f} MB "
              f"({quant_bytes/total_bytes*100:.0f}%)")
        print(f"  误差最大的层:")
        for key, shape, nbytes, err, out_err in sorted(rows, key=lambda r: -r[3])[:args.top]:
            print(f"    {err*100:7.3f}%  out={out_err*100:7.3f}%  {str(shape):<16} {key[:70]}")


if __name__ == "__main__":
    main()
