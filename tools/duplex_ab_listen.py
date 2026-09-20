#!/usr/bin/env python3
"""
A/B the duplex `listen_prob_scale` knob over the raw WebSocket protocol.

Scenario: 3 s of real speech (so the model has something to answer) followed by
N seconds of silence (the user has gone quiet). While the user is silent the
model is expected to keep talking; every unit it spends on `<|listen|>` instead
is a 1-second hole in the answer - audible as stop-and-go.

Usage:
  python3 tools/duplex_ab_listen.py [--chunks 22] [--runs 0.5,1.0]
"""

import argparse
import base64
import json
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import websocket

WS_URL = "ws://127.0.0.1:7861/v1/realtime"
REF_WAV = Path("/Users/mincer/项目/s2s/data-native/assets/ref_audio/builtin_default.wav")
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 1.0
FEED_INTERVAL = 1.05  # seconds between input chunks, mirrors the browser client


def load_wav_mono(path):
    with wave.open(str(path), "rb") as w:
        n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        s = [v / 32768.0 for v in struct.unpack(f"<{n}h", raw)]
    elif sw == 4:
        s = [v / 2147483648.0 for v in struct.unpack(f"<{n}i", raw)]
    else:
        raise SystemExit(f"unsupported sample width {sw}")
    if ch > 1:
        s = s[ch - 1 :: ch]
    return s, sr


def resample(samples, src_rate, dst_rate):
    if src_rate == dst_rate:
        return samples
    ratio = src_rate / dst_rate
    out_len = int(len(samples) / ratio)
    out = []
    for i in range(out_len):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(samples[lo] * (1 - frac) + samples[hi] * frac)
    return out


def b64_f32(samples):
    data = b"".join(struct.pack("<f", max(-1.0, min(1.0, v))) for v in samples)
    return base64.b64encode(data).decode("ascii")


def run_once(scale, speech_chunks, silence_chunks):
    counts = {"audio": 0, "listen": 0, "text": 0, "other": 0}
    text_parts = []
    listen_units = []
    seen = {"stop": False}
    lock = threading.Lock()
    # Aggregate per-stage timings so a performance change can be verified
    # without asking anyone to listen.
    stage_samples = {}
    STAGE_KEYS = ("wall_clock_ms", "audio_encoder_ms", "audio_bridge_ms",
                  "llm_prefill_ms", "llm_decode_ms", "semantic_tts_ms",
                  "token2wav_ms", "flow_ms", "hift_ms")

    ws = websocket.WebSocket()
    ws.connect(WS_URL, timeout=30)

    def recv_loop():
        while True:
            try:
                raw = ws.recv()
            except Exception:
                return
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t == "response.output.delta":
                kind = msg.get("kind")
                m = msg.get("metrics") or {}
                with lock:
                    for k in STAGE_KEYS:
                        v = m.get(k)
                        if isinstance(v, (int, float)):
                            stage_samples.setdefault(k, []).append(float(v))
                    if kind in ("audio", "listen", "text"):
                        counts[kind] += 1
                        if kind == "listen":
                            listen_units.append(m.get("unit_index"))
                        if kind == "text" and msg.get("text"):
                            text_parts.append(msg["text"])
                    else:
                        counts["other"] += 1
            elif t in ("response.done", "session.close"):
                with lock:
                    seen["stop"] = True

    recv = threading.Thread(target=recv_loop, daemon=True)
    recv.start()

    ref_prompt = speech_pool[: SAMPLE_RATE_IN * 2]  # 2 s of the prompt as voice ref

    payload = {
        "system_prompt": "你是酸枣岭的语音助手，请用中文自然地长篇讲述。",
        "config": {
            "length_penalty": 1.1,
            "force_listen_count": 0,
            "top_k": 20,
            "greedy_listen_speak_decision": False,
            "listen_prob_scale": scale,
        },
        "use_tts": True,
        # The backend rejects WAV containers for the voice prompt: it wants raw
        # little-endian Float32 PCM at 16 kHz.
        "ref_audio_base64": b64_f32(ref_prompt),
    }
    ws.send(json.dumps({"type": "session.init", "payload": payload}))

    # wait for session to become ready
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception:
            break
        if not isinstance(raw, str):
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue  # keep-alive / non-JSON frame
        mt = msg.get("type")
        if mt in ("session.ready", "session.active", "session.created",
                  "session.state", "input.accepted"):
            ready = True
            break
    if not ready:
        print("  session did not become ready")
        return None

    total = speech_chunks + silence_chunks
    for i in range(total):
        body = {}
        if i == 0:
            # A text turn is the reliable way to ask for a long answer; a 3 s
            # voice clip does not constitute a request at all.
            body = {
                "text": "请给我讲一个很长很长的故事，从头讲到尾，不要停下来。",
                "input_id": f"ab_{scale}_0000",
            }
        elif i < speech_chunks + 1:
            lo = (i - 1) * SAMPLE_RATE_IN
            samples = speech_pool[lo : lo + SAMPLE_RATE_IN]
            if len(samples) < SAMPLE_RATE_IN:
                samples = samples + [0.0] * (SAMPLE_RATE_IN - len(samples))
            body = {
                "audio": b64_f32(samples),
                "sample_rate": SAMPLE_RATE_IN,
                "format": "pcm_f32le",
                "speech_active": active,
            }
        else:
            body = {
                "audio": b64_f32([0.0] * SAMPLE_RATE_IN),
                "sample_rate": SAMPLE_RATE_IN,
                "format": "pcm_f32le",
                "speech_active": False,
            }
        body["input_id"] = f"ab_{scale}_{i:04d}"
        ws.send(json.dumps({
            "type": "input.append",
            "message_id": body["input_id"],
            "input": body,
        }))
        # keep the receive loop moving while we pace the feed
        deadline2 = time.time() + FEED_INTERVAL
        while time.time() < deadline2:
            time.sleep(0.05)
        if seen["stop"]:
            break

    # drain the tail: the server needs ~0.8 s per unit in real time, and units
    # keep arriving after the feed ends. Stop after a stretch of silence.
    quiet = 0.0
    drained = 0.0
    while drained < 45.0:
        step = 0.25
        time.sleep(step)
        drained += step
        with lock:
            before = sum(counts.values())
        quiet = quiet + step if sum(counts.values()) == before else 0.0
        if quiet >= 6.0:
            break
    try:
        ws.close()
    except Exception:
        pass
    time.sleep(0.5)

    with lock:
        stats = {
            k: (sum(v) / len(v), max(v))
            for k, v in stage_samples.items() if v
        }
        return dict(counts), "".join(text_parts), list(listen_units), stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=22, help="silence seconds after the speech prompt")
    ap.add_argument("--speech", type=int, default=3, help="speech seconds used as the prompt")
    ap.add_argument("--runs", type=str, default="0.5,1.0")
    args = ap.parse_args()

    raw, sr = load_wav_mono(REF_WAV)
    speech_pool = resample(raw, sr, SAMPLE_RATE_IN)
    print(f"voice prompt: {len(speech_pool)/SAMPLE_RATE_IN:.1f}s @ {SAMPLE_RATE_IN}Hz")

    scales = [float(x) for x in args.runs.split(",") if x.strip()]
    results = {}
    for scale in scales:
        print(f"\n=== listen_prob_scale = {scale} ===")
        out = run_once(scale, args.speech, args.chunks)
        if out is None:
            continue
        counts, text, listen_units, stats = out
        total_units = counts["audio"] + counts["listen"]
        ratio = (counts["listen"] / total_units * 100) if total_units else 0.0
        results[scale] = (counts, ratio, stats)
        print(f"  audio={counts['audio']}  listen={counts['listen']}  → listen 占比 {ratio:.0f}%")
        if stats:
            print("  阶段耗时 (平均 / 最大):")
            for k in ("wall_clock_ms", "audio_encoder_ms", "audio_bridge_ms",
                      "llm_prefill_ms", "llm_decode_ms", "semantic_tts_ms",
                      "token2wav_ms", "flow_ms", "hift_ms"):
                if k in stats:
                    avg, mx = stats[k]
                    print(f"    {k:<18} {avg:7.1f} / {mx:7.0f} ms")
        print(f"  文本({len(text)}字): {text[:160]}")
        print(f"  listen unit: {listen_units[:20]}")

    print("\n=== 对比 ===")
    if len(results) == 2:
        (s1, (c1, r1)), (s2, (c2, r2)) = results.items()
        print(f"  scale={s1}: listen {c1['listen']}/{c1['audio']+c1['listen']} = {r1:.0f}%")
        print(f"  scale={s2}: listen {c2['listen']}/{c2['audio']+c2['listen']} = {r2:.0f}%")
