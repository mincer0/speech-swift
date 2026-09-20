#!/usr/bin/env python3
"""
Speech-understanding probe for the duplex server.

Feeds a KNOWN spoken request (`tools/fixtures/understand_test.wav`,
macOS `say`, 16 kHz) followed by silence, then prints the model's text
reply. Used to verify that the audio-encoder fast path did not break the
model's ability to understand speech: a healthy reply mentions the prompt
content (astronaut / Mars / potato); a broken one says "没听清楚" or
blurts a canned opener.

Usage:
  python3 tools/duplex_understand_test.py [--chunks 8]
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

ROOT = Path(__file__).resolve().parents[1]
WS_URL = "ws://127.0.0.1:7861/v1/realtime"
SPEECH_WAV = ROOT / "tools/fixtures/understand_test.wav"
REF_WAV = Path("/Users/mincer/项目/s2s/data-native/assets/ref_audio/builtin_default.wav")
SAMPLE_RATE_IN = 16_000


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
    out = []
    for i in range(int(len(samples) / ratio)):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(samples[lo] * (1 - frac) + samples[hi] * frac)
    return out


def b64_f32(samples):
    data = b"".join(struct.pack("<f", max(-1.0, min(1.0, v))) for v in samples)
    return base64.b64encode(data).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8, help="silence seconds after the speech")
    args = ap.parse_args()

    speech, sr = load_wav_mono(SPEECH_WAV)
    speech = resample(speech, sr, SAMPLE_RATE_IN)
    # pad to whole 1 s chunks
    while len(speech) % SAMPLE_RATE_IN:
        speech.append(0.0)
    speech_chunks = len(speech) // SAMPLE_RATE_IN
    print(f"prompt: {speech_chunks} chunks ({speech_chunks:.0f}s) + {args.chunks} silence")

    reply = []
    done = {"stop": False}
    lock = threading.Lock()

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
                m = msg.get("metrics") or {}
                if msg.get("kind") == "text" and msg.get("text"):
                    with lock:
                        reply.append((m.get("unit_index"), msg["text"]))
            elif t in ("response.done", "session.close"):
                with lock:
                    done["stop"] = True

    threading.Thread(target=recv_loop, daemon=True).start()

    ref = resample(*load_wav_mono(REF_WAV), SAMPLE_RATE_IN)[: SAMPLE_RATE_IN * 2]
    ws.send(json.dumps({"type": "session.init", "payload": {
        "system_prompt": "你是语音助手，请认真听用户说话并回应。",
        "config": {"length_penalty": 1.1, "force_listen_count": 0, "top_k": 20,
                   "greedy_listen_speak_decision": False},
        "use_tts": True,
        "ref_audio_base64": b64_f32(ref),
    }}))

    # wait for ready/active
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline and not ready:
        try:
            raw = ws.recv()
        except Exception:
            continue
        if not isinstance(raw, str):
            continue
        try:
            mt = json.loads(raw).get("type")
        except Exception:
            continue
        if mt in ("session.created", "session.active", "session.state", "session.queue_done"):
            ready = True
    if not ready:
        print("session did not become ready")
        return

    total = speech_chunks + args.chunks
    for i in range(total):
        if i < speech_chunks:
            samples = speech[i * SAMPLE_RATE_IN : (i + 1) * SAMPLE_RATE_IN]
            active = True
        else:
            samples = [0.0] * SAMPLE_RATE_IN
            active = False
        ws.send(json.dumps({
            "type": "input.append", "message_id": f"ut_{i:04d}",
            "input": {"audio": b64_f32(samples), "input_id": f"ut_{i:04d}",
                      "sample_rate": SAMPLE_RATE_IN, "format": "pcm_f32le",
                      "speech_active": active},
        }))
        time.sleep(1.05)
        with lock:
            if done["stop"]:
                break

    quiet = drained = 0.0
    while drained < 40.0:
        time.sleep(0.25); drained += 0.25
        with lock:
            before = len(reply)
        quiet = quiet + 0.25 if len(reply) == before else 0.0
        if quiet >= 5.0:
            break
    try:
        ws.close()
    except Exception:
        pass
    time.sleep(0.3)

    with lock:
        text = "".join(t for _, t in reply)
    print("\n=== 模型回复 ===")
    print(text or "(无文本)")
    keywords = ["宇航员", "火星", "土豆", "科幻", "太空"]
    hit = [k for k in keywords if k in text]
    negative = [k for k in ("没听清", "听不见", "没听清", "没明白", "再说一遍") if k in text]
    print(f"\n命中关键词: {hit or '无'}   负面信号: {negative or '无'}")
    verdict = "✓ 理解正常" if hit and not negative else ("✗ 理解受损" if negative else "? 不确定")
    print(verdict)


if __name__ == "__main__":
    main()
