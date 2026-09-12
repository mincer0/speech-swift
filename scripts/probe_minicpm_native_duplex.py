#!/usr/bin/env python3
"""Probe the MiniCPM native realtime WebSocket without a microphone.

The native Swift demo speaks the small official wire protocol directly:
``session.queue_done`` is sent before ``session.init`` and all media travels
inside JSON ``input.append`` envelopes as raw little-endian Float32 PCM.  This
probe intentionally keeps the transport and its assertions visible so a
failed run leaves enough evidence to diagnose turn boundaries, stale output,
and a sliding KV cache.

The optional network dependency is imported only when the probe is run.  The
helpers in this file are standard-library-only and are used by the portable
regression tests; those tests never start a server or load a model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
INPUT_CHANNELS = 1
INPUT_SAMPLE_BYTES = 4
INPUT_PACKET_SAMPLES = INPUT_RATE
INPUT_PACKET_BYTES = INPUT_PACKET_SAMPLES * INPUT_SAMPLE_BYTES
RESPONSE_WINDOW_PACKETS = 3
SILENCE_PACKET = b"\0" * INPUT_PACKET_BYTES
OUTPUT_FORMAT = "pcm_f32"

DEFAULT_URL = "ws://127.0.0.1:7860/v1/realtime?mode=audio"
DEFAULT_OUTPUT = Path("test-output/minicpm-native-duplex")
DEFAULT_SAY_VOICE = "Tingting"

NORMAL_QUESTIONS = (
    "你好，请用一句话介绍你自己。",
    "二加三等于多少？只回答数字。",
    "请说出三种常见水果。",
    "上一题里你提到的第二种水果是什么？",
    "最后请用一句简短的话和我告别。",
)
STORY_QUESTIONS = (
    "请记住：小猫把一颗红球放在窗台上。请简短复述这个事实。",
    "小猫把什么颜色的球放在哪里？只回答颜色和位置。",
    "现在请用一句话继续这个故事，并提到那颗球。",
)
BARGE_IN_QUESTIONS = (
    "请从一数到五十，每个数字都清楚地说出来。",
    "请停止数数，只回答七加八等于多少。",
    "第三轮请只回答九减四等于多少。",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_text(text: str) -> str:
    """Normalize text for duplicate/acknowledgement checks, not display."""

    return "".join(text.split()).casefold().strip("。，！？,.!?：:；;")


def normalize_repetition_text(text: str) -> str:
    """Remove separators that can split otherwise identical spoken phrases."""

    return "".join(
        character.casefold()
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
    )


def find_tandem_repeat(text: str, *, minimum_unit_length: int = 4) -> dict[str, Any] | None:
    """Return the first adjacent repeated span in accumulated turn text."""

    normalized = normalize_repetition_text(text)
    if minimum_unit_length <= 0:
        raise ValueError("minimum_unit_length must be positive")
    for start in range(len(normalized)):
        maximum = (len(normalized) - start) // 2
        for unit_length in range(maximum, minimum_unit_length - 1, -1):
            unit = normalized[start : start + unit_length]
            if unit == normalized[start + unit_length : start + 2 * unit_length]:
                return {
                    "kind": "adjacent_tandem",
                    "normalized": normalized,
                    "repeated_unit": unit,
                    "start": start,
                    "unit_length": unit_length,
                }
    return None


def decode_float32_pcm(payload: bytes) -> tuple[float, ...]:
    """Decode and validate raw little-endian Float32 PCM bytes."""

    if not payload or len(payload) % INPUT_SAMPLE_BYTES:
        raise ValueError("Float32 PCM must be non-empty and divisible by four bytes")
    samples = struct.unpack(f"<{len(payload) // INPUT_SAMPLE_BYTES}f", payload)
    if not all(math.isfinite(sample) for sample in samples):
        raise ValueError("Float32 PCM contains NaN or infinity")
    return samples


def encode_float32_pcm(samples: Iterable[float]) -> bytes:
    values = tuple(float(value) for value in samples)
    if not values:
        raise ValueError("Float32 PCM must contain at least one sample")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Float32 PCM contains NaN or infinity")
    return struct.pack(f"<{len(values)}f", *values)


def chunk_float32_pcm(payload: bytes, packet_samples: int = INPUT_PACKET_SAMPLES) -> list[bytes]:
    """Split PCM into exact one-second packets, padding only the final packet."""

    if packet_samples <= 0:
        raise ValueError("packet_samples must be positive")
    decode_float32_pcm(payload)
    packet_bytes = packet_samples * INPUT_SAMPLE_BYTES
    chunks = [payload[offset : offset + packet_bytes] for offset in range(0, len(payload), packet_bytes)]
    if not chunks:
        chunks = [b""]
    final_size = packet_bytes - len(chunks[-1])
    if final_size:
        chunks[-1] += b"\0" * final_size
    return chunks


def build_session_init(
    system_prompt: str | None = None,
    *,
    listen_probability_scale: float | None = None,
    temperature: float | None = None,
    sliding_window_high: int | None = None,
    sliding_window_low: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": "full_duplex"}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    config: dict[str, Any] = {
        "force_listen_count": 0,
        "top_k": 20,
        "greedy_listen_speak_decision": False,
    }
    if listen_probability_scale is not None:
        config["listen_prob_scale"] = listen_probability_scale
    if temperature is not None:
        config["temperature"] = temperature
    if sliding_window_high is not None:
        config["sliding_window_mode"] = "basic"
        config["basic_window_high_tokens"] = sliding_window_high
        config["basic_window_low_tokens"] = (
            sliding_window_low if sliding_window_low is not None else sliding_window_high
        )
    if config:
        payload["config"] = config
    return {"type": "session.init", "payload": payload}


def build_input_append(
    payload: bytes,
    *,
    input_id: str | None = None,
    speech_active: bool | None = None,
    allow_sampled_decision: bool | None = None,
    continuation_tick: bool = False,
) -> dict[str, Any]:
    """Build the strict official input envelope for one Float32 packet."""

    samples = decode_float32_pcm(payload)
    if len(samples) != INPUT_PACKET_SAMPLES:
        raise ValueError(
            f"input packets must contain exactly {INPUT_PACKET_SAMPLES} samples; "
            f"got {len(samples)}"
        )
    input_object: dict[str, Any] = {
        "audio": base64.b64encode(payload).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate": INPUT_RATE,
    }
    if input_id:
        input_object["input_id"] = input_id
    if speech_active is not None:
        input_object["speech_active"] = speech_active
    if allow_sampled_decision is not None:
        input_object["allow_sampled_listen_speak_decision"] = allow_sampled_decision
    if continuation_tick:
        input_object["continuation_tick"] = True
    return {"type": "input.append", "input": input_object}


def write_float32_wav(path: Path, payload: bytes, sample_rate: int = OUTPUT_RATE) -> None:
    """Write a standards-compliant IEEE-float WAV (format code 3)."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    decode_float32_pcm(payload)
    block_align = INPUT_SAMPLE_BYTES
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        1,
        sample_rate,
        byte_rate,
        block_align,
        32,
    )
    # IEEE-float WAV files carry a ``fact`` sample-count chunk in addition to
    # the fmt/data chunks.  Keeping it here makes the saved answers readable
    # by stricter audio tooling while preserving the raw Float32 payload.
    fact = struct.pack("<I", len(payload) // INPUT_SAMPLE_BYTES)
    riff_size = 4 + 8 + len(fmt) + 8 + len(fact) + 8 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        target.write(b"RIFF")
        target.write(struct.pack("<I", riff_size))
        target.write(b"WAVEfmt ")
        target.write(struct.pack("<I", len(fmt)))
        target.write(fmt)
        target.write(b"fact")
        target.write(struct.pack("<I", len(fact)))
        target.write(fact)
        target.write(b"data")
        target.write(struct.pack("<I", len(payload)))
        target.write(payload)


def _float32_from_wav(path: Path) -> bytes:
    # ``wave`` in the system Python rejects WAVE_FORMAT_IEEE_FLOAT (format
    # code 3), which is exactly what ffmpeg emits for ``pcm_f32le``.  Parse
    # the tiny RIFF container ourselves and accept only mono 16 kHz input.
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: expected a RIFF/WAVE container")
    offset = 12
    format_code: int | None = None
    channels: int | None = None
    sample_rate: int | None = None
    bits: int | None = None
    frames: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        if end > len(data):
            raise ValueError(f"{path}: truncated WAV chunk {chunk_id!r}")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError(f"{path}: truncated fmt chunk")
            format_code, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, start
            )
        elif chunk_id == b"data":
            frames = data[start:end]
        offset = end + (chunk_size & 1)
    if format_code is None or channels is None or sample_rate is None or bits is None or frames is None:
        raise ValueError(f"{path}: WAV is missing fmt or data")
    if channels != INPUT_CHANNELS or sample_rate != INPUT_RATE:
        raise ValueError(
            f"{path}: expected mono {INPUT_RATE} Hz WAV, got {channels}ch/{sample_rate}Hz"
        )
    if format_code == 3 and bits == 32:
        decode_float32_pcm(frames)
        return frames
    if format_code == 1 and bits == 16:
        if len(frames) % 2:
            raise ValueError(f"{path}: odd PCM16 byte count")
        pcm16 = struct.unpack(f"<{len(frames) // 2}h", frames)
        return encode_float32_pcm(sample / 32768.0 for sample in pcm16)
    raise ValueError(
        f"{path}: expected PCM16 or IEEE Float32 WAV, got format={format_code} bits={bits}"
    )


def _run_checked(command: Sequence[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown failure"
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")


def synthesize_questions(output: Path, questions: Sequence[str], *, voice: str = DEFAULT_SAY_VOICE) -> list[Path]:
    """Use macOS ``say`` and ffmpeg to create 16 kHz Float32 question WAVs."""

    say = shutil.which("say")
    ffmpeg = shutil.which("ffmpeg")
    if say is None or ffmpeg is None:
        raise RuntimeError("question synthesis requires macOS 'say' and 'ffmpeg'")
    question_dir = output / "questions"
    question_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, question in enumerate(questions, start=1):
        aiff = question_dir / f"question-{index:02d}.aiff"
        wav = question_dir / f"question-{index:02d}.wav"
        _run_checked([say, "-v", voice, "-r", "185", "-o", str(aiff), question])
        _run_checked(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(aiff),
                "-ar",
                str(INPUT_RATE),
                "-ac",
                str(INPUT_CHANNELS),
                "-c:a",
                "pcm_f32le",
                str(wav),
            ]
        )
        aiff.unlink(missing_ok=True)
        _float32_from_wav(wav)
        paths.append(wav)
    return paths


@dataclass
class TurnResult:
    number: int
    question: str
    started_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    text_parts: list[str] = field(default_factory=list)
    audio_parts: list[bytes] = field(default_factory=list)
    listen_boundaries: list[dict[str, Any]] = field(default_factory=list)
    end_boundaries: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    kv_cache_lengths: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    repeated_text: list[dict[str, Any]] = field(default_factory=list)
    saw_audio: bool = False
    saw_listen: bool = False
    saw_end_boundary: bool = False
    saw_interrupt: bool = False
    completion_reason: str | None = None
    error: str | None = None
    audio_path: str | None = None
    first_text_at: float | None = None
    first_audio_at: float | None = None
    audio_started: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    _seen_text: set[str] = field(default_factory=set, repr=False)
    _reported_repetitions: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    @property
    def text(self) -> str:
        return "".join(self.text_parts).strip()

    @property
    def audio(self) -> bytes:
        return b"".join(self.audio_parts)

    @property
    def output_audio_seconds(self) -> float:
        return len(self.audio) / (OUTPUT_RATE * INPUT_SAMPLE_BYTES)

    def unique_generation_seconds(self) -> float:
        intervals: set[tuple[int, int]] = set()
        for metrics in self.metrics:
            started = metrics.get("generation_started_at_ns")
            exited = metrics.get("generation_exited_at_ns")
            if (
                isinstance(started, int)
                and not isinstance(started, bool)
                and isinstance(exited, int)
                and not isinstance(exited, bool)
                and exited >= started
            ):
                intervals.add((started, exited))
        return sum(exited - started for started, exited in intervals) / 1_000_000_000

    def add_text(self, text: str, *, event: Mapping[str, Any] | None = None) -> None:
        if not text:
            return
        normalized = normalize_text(text)
        if normalized and normalized in self._seen_text:
            signature = ("duplicate_chunk", normalized)
            if signature not in self._reported_repetitions:
                self.repeated_text.append(
                    {
                        "kind": "duplicate_chunk",
                        "text": text,
                        "normalized": normalized,
                        "event": dict(event or {}),
                    }
                )
                self._reported_repetitions.add(signature)
        if normalized:
            self._seen_text.add(normalized)
        self.text_parts.append(text)
        tandem = find_tandem_repeat(self.text)
        if tandem is not None:
            signature = (
                tandem["kind"],
                tandem["start"],
                tandem["unit_length"],
                tandem["repeated_unit"],
            )
            if signature not in self._reported_repetitions:
                tandem["text"] = self.text
                tandem["event"] = dict(event or {})
                self.repeated_text.append(tandem)
                self._reported_repetitions.add(signature)

    def summary(self) -> dict[str, Any]:
        duration = None if self.completed_at is None else self.completed_at - self.started_at
        audio_seconds = self.output_audio_seconds
        generation_seconds = self.unique_generation_seconds()
        return {
            "number": self.number,
            "question": self.question,
            "duration_seconds": None if duration is None else round(duration, 3),
            "first_text_latency_seconds": None
            if self.first_text_at is None
            else round(self.first_text_at - self.started_at, 3),
            "first_audio_latency_seconds": None
            if self.first_audio_at is None
            else round(self.first_audio_at - self.started_at, 3),
            "output_audio_seconds": round(audio_seconds, 3),
            "model_generation_seconds": round(generation_seconds, 3),
            "model_generation_rtf": None
            if audio_seconds <= 0
            else round(generation_seconds / audio_seconds, 3),
            "turn_wall_rtf": None
            if duration is None or audio_seconds <= 0
            else round(duration / audio_seconds, 3),
            "completion_reason": self.completion_reason,
            "error": self.error,
            "text": self.text,
            "text_parts": self.text_parts,
            "audio_bytes": len(self.audio),
            "audio_sample_rate": OUTPUT_RATE,
            "audio_format": OUTPUT_FORMAT,
            "audio_path": self.audio_path,
            "saw_audio": self.saw_audio,
            "saw_listen": self.saw_listen,
            "saw_end_boundary": self.saw_end_boundary,
            "saw_interrupt": self.saw_interrupt,
            "listen_boundaries": self.listen_boundaries,
            "end_boundaries": self.end_boundaries,
            "metrics": self.metrics,
            "kv_cache_lengths": self.kv_cache_lengths,
            "repeated_text": self.repeated_text,
            "events": self.events,
        }


def _safe_event_snapshot(event: Mapping[str, Any]) -> dict[str, Any]:
    def redact(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for child_key, child_value in value.items():
                result[str(child_key)] = redact(child_value, str(child_key))
            return result
        if isinstance(value, list):
            return [redact(item, key) for item in value]
        if key == "audio" and isinstance(value, str):
            try:
                decoded = base64.b64decode(value, validate=True)
                return {"base64": "<base64 omitted>", "audio_bytes": len(decoded)}
            except (ValueError, binascii.Error):
                return {"base64": "<base64 omitted>", "audio_bytes": None}
        return value

    return redact(event)


class EventObserver:
    """Protocol/event state machine shared by the network runner and tests."""

    def __init__(self) -> None:
        self.timeline: list[dict[str, Any]] = []
        self.current: TurnResult | None = None
        self.session_id: str | None = None
        self.server_mode: str | None = None
        self.queue_done = asyncio.Event()
        self.session_created = asyncio.Event()
        self.session_closed = asyncio.Event()
        self.session_reset = asyncio.Event()
        self.fatal_error: str | None = None
        self.unsolicited_outputs: list[dict[str, Any]] = []
        self.repeated_text: list[dict[str, Any]] = []
        self.all_metrics: list[dict[str, Any]] = []
        self.all_kv_cache_lengths: list[int] = []
        self.interrupt_traces: list[dict[str, Any]] = []
        # The realtime endpoint deliberately does not expose the internal
        # ``input.accepted``/``input.committed`` events.  Every official
        # response delta still carries the input_id that caused it, so the
        # probe can use the first terminal output for that input as a bounded
        # application-level acknowledgement.  This keeps a slow model from
        # being flooded by one-second silence packets while preserving the
        # server's explicit queue-full failure as evidence.
        self._input_ack_events: dict[str, asyncio.Event] = {}
        self._acknowledged_input_ids: set[str] = set()
        self.input_acknowledgements: list[dict[str, Any]] = []
        self.input_pacing_fallbacks: list[dict[str, Any]] = []
        # Full-duplex reality: the model can close the *previous* response
        # while this window's question is still streaming, and it may keep
        # speaking (or stream server-driven continuation units) just after a
        # boundary.  ``last_completed_turn`` lets those deltas attach to the
        # conversation instead of being misread as stale unsolicited output.
        self.last_completed_turn: TurnResult | None = None

    def register_input(self, input_id: str) -> asyncio.Event:
        """Register an input before its frame is sent.

        Registration happens before ``send`` because a local WebSocket test
        double (and, in practice, a fast server) may deliver the matching
        response before the awaiter is installed.
        """

        flag = asyncio.Event()
        if input_id in self._acknowledged_input_ids:
            flag.set()
        else:
            self._input_ack_events[input_id] = flag
        return flag

    def _acknowledge_input(self, event: Mapping[str, Any]) -> None:
        input_id = event.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            return
        event_type = str(event.get("type") or "")
        kind = str(event.get("kind") or "")
        # ``audio`` is the last normal output delta for the native engine;
        # ``listen`` is the only delta for a buffered/idle packet and the
        # explicit end marker for a completed turn.  response.done covers
        # turn-based adapters.  A text-only adapter has no later marker, so
        # retain text as a fallback acknowledgement as well.
        if event_type == "input.committed":
            self._acknowledged_input_ids.add(input_id)
            flag = self._input_ack_events.pop(input_id, None)
            if flag is not None:
                flag.set()
            self.input_acknowledgements.append(
                {
                    "input_id": input_id,
                    "event_type": event_type,
                    "kind": None,
                    "monotonic": round(time.monotonic(), 6),
                }
            )
        elif event_type == "response.done" or event_type == "response.output.delta":
            if kind in {"audio", "listen", "text"} or event_type == "response.done":
                self._acknowledged_input_ids.add(input_id)
                flag = self._input_ack_events.pop(input_id, None)
                if flag is not None:
                    flag.set()
                self.input_acknowledgements.append(
                    {
                        "input_id": input_id,
                        "event_type": event_type,
                        "kind": kind or None,
                        "monotonic": round(time.monotonic(), 6),
                    }
                )

    def log(self, direction: str, event: str, **fields: Any) -> None:
        self.timeline.append(
            {
                "wall_time": utc_now(),
                "monotonic": round(time.monotonic(), 6),
                "direction": direction,
                "event": event,
                **fields,
            }
        )

    def set_current(self, turn: TurnResult | None) -> None:
        self.current = turn
        if turn is not None:
            self.log("probe", "turn_started", number=turn.number, question=turn.question)

    def _record_metrics(self, event: Mapping[str, Any], turn: TurnResult | None) -> None:
        metrics = event.get("metrics")
        if isinstance(metrics, Mapping):
            record = dict(metrics)
            self.all_metrics.append(record)
            if turn is not None:
                turn.metrics.append(record)
            kv = record.get("kv_cache_length")
            if isinstance(kv, (int, float)) and not isinstance(kv, bool):
                self._record_kv(int(kv), turn)
        kv = event.get("kv_cache_length")
        if isinstance(kv, (int, float)) and not isinstance(kv, bool):
            self._record_kv(int(kv), turn)

    def _record_kv(self, value: int, turn: TurnResult | None) -> None:
        self.all_kv_cache_lengths.append(value)
        if turn is not None:
            turn.kv_cache_lengths.append(value)

    def handle_event(self, event: Mapping[str, Any]) -> None:
        """Consume one decoded server event; no network or model is touched."""

        event_type = str(event.get("type") or "")
        snapshot = _safe_event_snapshot(event)
        self.log("server", event_type or "unknown", payload=snapshot)
        if event_type == "session.queue_done":
            self.queue_done.set()
            return
        if event_type == "session.created":
            self.session_id = str(event.get("session_id") or "") or None
            self.server_mode = str(event.get("mode") or "") or None
            self.session_created.set()
            return
        if event_type == "session.closed":
            self.session_closed.set()
            if self.current is not None and not self.current.done.is_set():
                self.current.completion_reason = "session_closed"
                self.current.completed_at = time.monotonic()
                self.current.done.set()
            return
        if event_type == "error":
            error = event.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message") or error)
            else:
                message = str(event.get("message") or error or "server error")
            if self.current is not None:
                self.current.error = message
                self.current.completion_reason = "server_error"
                self.current.completed_at = time.monotonic()
                self.current.done.set()
            else:
                self.fatal_error = message
            return
        # The in-process backend emits ``backend.state``.  The official
        # realtime WebSocket translation deliberately exposes that diagnostic
        # event as ``session.state`` and carries the same scalar trace under
        # ``metrics``.  Observe both names so the live probe validates the
        # wire protocol rather than an internal event spelling.
        if event_type in {"backend.state", "session.state"}:
            if str(event.get("reason") or "") == "reset":
                self.session_reset.set()
            if self.current is not None and str(event.get("reason") or "") == "interrupt":
                self.current.saw_interrupt = True
            if str(event.get("reason") or "") == "interrupt":
                trace = event.get("trace")
                if not isinstance(trace, Mapping):
                    trace = event.get("metrics")
                if isinstance(trace, Mapping):
                    self.interrupt_traces.append(dict(trace))
            self._record_metrics(event, self.current)
            return
        if event_type == "response.done":
            self._acknowledge_input(event)
            turn = self.current
            if turn is None:
                self.unsolicited_outputs.append({"kind": "done", "event": snapshot})
                return
            self._record_metrics(event, turn)
            turn.end_boundaries.append(snapshot)
            turn.saw_end_boundary = True
            turn.completion_reason = "response_done"
            turn.completed_at = time.monotonic()
            turn.done.set()
            self.last_completed_turn = turn
            return
        if event_type != "response.output.delta":
            return

        self._acknowledge_input(event)

        turn = self.current
        kind = str(event.get("kind") or "")
        if turn is None:
            if kind in {"text", "audio"}:
                if self.last_completed_turn is not None:
                    # Trailing model speech after a completed boundary (and
                    # server-driven cont_* continuation units) is normal
                    # full-duplex output, not stale/cross-turn leakage.
                    # Genuine stale repeats are still caught by the
                    # repeated-text audit below and by the barge-in turns.
                    last = self.last_completed_turn
                    last.events.append(snapshot)
                    if kind == "text" and str(event.get("text") or ""):
                        last.add_text(str(event.get("text")), event=snapshot)
                    elif kind == "audio":
                        try:
                            last.audio_parts.append(self._decode_output_audio(event))
                        except ValueError:
                            pass
                    return
                item = {"kind": kind, "event": snapshot}
                if kind == "audio":
                    try:
                        item["audio_bytes"] = len(self._decode_output_audio(event))
                    except ValueError as exc:
                        item["error"] = str(exc)
                self.unsolicited_outputs.append(item)
            return

        turn.events.append(snapshot)
        self._record_metrics(event, turn)
        if kind == "text":
            text = str(event.get("text") or "")
            if text and turn.first_text_at is None:
                turn.first_text_at = time.monotonic()
            before = len(turn.repeated_text)
            turn.add_text(text, event=snapshot)
            if len(turn.repeated_text) > before:
                self.repeated_text.extend(turn.repeated_text[before:])
        elif kind == "audio":
            try:
                audio = self._decode_output_audio(event)
            except ValueError as exc:
                turn.error = str(exc)
                turn.completion_reason = "invalid_audio"
                turn.completed_at = time.monotonic()
                turn.done.set()
                return
            turn.audio_parts.append(audio)
            if turn.first_audio_at is None:
                turn.first_audio_at = time.monotonic()
            turn.saw_audio = True
            turn.audio_started.set()
        elif kind == "listen":
            turn.saw_listen = True
            turn.listen_boundaries.append(snapshot)
            reason = str(event.get("reason") or "").casefold()
            # Generic listen markers are state updates, not turn boundaries.
            # The native duplex backend uses ``listen``/``model_listen`` while
            # it remains ready for more input; only an explicit end marker
            # closes the current response.  A response.done event is handled
            # separately above.
            if bool(event.get("end_of_turn")) or reason in {
                "turn_end",
                "end_of_turn",
                "response_done",
            }:
                if turn.first_text_at is None and turn.first_audio_at is None:
                    # The boundary closes the *previous* response, which
                    # overlapped into this window while the question was
                    # still streaming (server-driven full duplex runs ahead
                    # of the client's question cadence).  This window has
                    # not produced any model output yet, so keep it open
                    # until its own answer arrives instead of completing it
                    # on the old response's boundary.
                    return
                turn.saw_end_boundary = True
                turn.end_boundaries.append(snapshot)
                turn.completion_reason = "listen_end"
                turn.completed_at = time.monotonic()
                turn.done.set()
                self.last_completed_turn = turn

    @staticmethod
    def _decode_output_audio(event: Mapping[str, Any]) -> bytes:
        encoded = event.get("audio")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("audio output delta has no base64 audio")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("audio output is not valid base64") from exc
        if len(payload) % INPUT_SAMPLE_BYTES:
            raise ValueError("audio output Float32 byte length is not divisible by four")
        try:
            decode_float32_pcm(payload)
        except ValueError as exc:
            raise ValueError(f"invalid output Float32 PCM: {exc}") from exc
        sample_rate = event.get("sample_rate", OUTPUT_RATE)
        if sample_rate != OUTPUT_RATE:
            raise ValueError(f"audio output sample_rate must be {OUTPUT_RATE}, got {sample_rate!r}")
        fmt = str(event.get("format") or "")
        if fmt and "f32" not in fmt.casefold() and "float" not in fmt.casefold():
            raise ValueError(f"audio output format is not Float32: {fmt}")
        return payload


def _acknowledgement_only(text: str) -> bool:
    return normalize_text(text) in {"好的", "好", "收到", "ok", "okay", "明白"}


def validate_report(
    mode: str,
    questions: Sequence[str],
    turns: Sequence[TurnResult],
    observer: EventObserver,
) -> list[str]:
    """Return explicit failures; an empty list means the probe passed."""

    failures: list[str] = []
    if observer.fatal_error:
        failures.append(f"receiver: {observer.fatal_error}")
    if len(turns) != len(questions):
        failures.append(f"completed {len(turns)} of {len(questions)} requested turns")
    for turn in turns:
        if turn.error:
            failures.append(f"turn {turn.number}: {turn.error}")
        if not turn.saw_audio or not turn.audio:
            failures.append(f"turn {turn.number}: no valid Float32/24k audio output")
        interrupted_first_barge = (
            mode == "barge-in"
            and turn.number == 1
            and turn.completion_reason == "barge_in"
        )
        if not turn.saw_end_boundary and not interrupted_first_barge:
            failures.append(f"turn {turn.number}: no listen/end boundary")
        if not turn.text:
            failures.append(f"turn {turn.number}: no text output")

    if mode == "story":
        all_text = "".join(turn.text for turn in turns)
        if not all_text or _acknowledgement_only(all_text):
            failures.append("story produced only an acknowledgement instead of continuing")
        if len(turns) < 2:
            failures.append("story stopped before a follow-up turn")
    elif mode == "barge-in":
        if len(turns) != 3:
            failures.append("barge-in requires exactly three turns")
        elif not turns[0].saw_audio or turns[0].completion_reason != "barge_in":
            failures.append("first barge-in turn did not stop after audio onset")
        elif not turns[1].saw_audio or not turns[1].saw_end_boundary:
            failures.append("second barge-in turn did not complete")
        elif not turns[2].saw_audio or not turns[2].saw_end_boundary:
            failures.append("third barge-in turn did not complete")
        else:
            second_text = normalize_text(turns[1].text)
            third_text = normalize_text(turns[2].text)
            if "15" not in second_text and "十五" not in second_text:
                failures.append(
                    f"second barge-in answer does not contain 15/十五: {turns[1].text!r}"
                )
            if "5" not in third_text and "五" not in third_text:
                failures.append(
                    f"third barge-in answer does not contain 5/五: {turns[2].text!r}"
                )
            first_text = normalize_text(turns[0].text)
            if len(first_text) >= 6 and (
                first_text in second_text or first_text in third_text
            ):
                failures.append("first barge-in response text crossed into a later turn")

        if not observer.interrupt_traces:
            failures.append("barge-in has no structured interrupt trace")
        else:
            trace = observer.interrupt_traces[-1]
            required = {
                "epoch",
                "unit_index",
                "current_turn_ended",
                "kv_cache_length",
                "pending_token_count",
                "prefetched",
                "needs_finalize",
                "tts_context_present",
                "token2wav_prepared",
                "stale_output_discard_count",
                "cancel_requested_at_ns",
                "cancel_returned_at_ns",
            }
            missing = sorted(key for key in required if key not in trace)
            if missing:
                failures.append(
                    "interrupt trace missing fields: " + ", ".join(missing)
                )
            generation_exit = trace.get("generation_exited_at_ns")
            cancel_return = trace.get("cancel_returned_at_ns")
            if not isinstance(generation_exit, (int, float)):
                failures.append("interrupt trace has no generation exit timestamp")
            if not isinstance(cancel_return, (int, float)):
                failures.append("interrupt trace has no cancel return timestamp")

        for item in observer.timeline:
            serialized = json.dumps(item, ensure_ascii=False).casefold()
            if "input queue is full" in serialized or "queue is full" in serialized:
                failures.append("barge-in encountered input queue is full")
                break

    if observer.unsolicited_outputs:
        failures.append(
            f"{len(observer.unsolicited_outputs)} unsolicited text/audio outputs during silence"
        )
    repeated_turns = [turn.number for turn in turns if turn.repeated_text]
    if repeated_turns or observer.repeated_text:
        numbers = ", ".join(str(number) for number in repeated_turns) or "unknown"
        failures.append(f"repeated response text detected in turn(s): {numbers}")
    return failures


class NativeDuplexProbe:
    def __init__(
        self,
        websocket: Any,
        output: Path,
        *,
        packet_seconds: float = 1.0,
        acknowledgement_timeout: float = 5.0,
    ) -> None:
        self.websocket = websocket
        self.output = output
        if packet_seconds <= 0:
            raise ValueError("packet_seconds must be positive")
        self.packet_seconds = packet_seconds
        self.acknowledgement_timeout = max(packet_seconds, acknowledgement_timeout)
        self.observer = EventObserver()
        self.receiver_error: str | None = None
        self._next_packet_at = 0.0
        self._recent_speech_packets = 0

    def _allow_sampled_decision(self, speech_active: bool) -> bool:
        if speech_active:
            self._recent_speech_packets = RESPONSE_WINDOW_PACKETS
            return True
        allowed = self._recent_speech_packets > 0
        if allowed:
            self._recent_speech_packets -= 1
        return allowed

    async def receive_loop(self) -> None:
        try:
            async for raw in self.websocket:
                if isinstance(raw, bytes):
                    self.receiver_error = "official realtime transport returned a binary frame"
                    self.observer.fatal_error = self.receiver_error
                    continue
                if not isinstance(raw, str):
                    self.receiver_error = "official realtime transport returned a non-text frame"
                    self.observer.fatal_error = self.receiver_error
                    continue
                try:
                    event = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    self.receiver_error = f"invalid server JSON: {exc}"
                    self.observer.fatal_error = self.receiver_error
                    continue
                if not isinstance(event, Mapping):
                    self.receiver_error = "server event must be a JSON object"
                    self.observer.fatal_error = self.receiver_error
                    continue
                self.observer.handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - exercised by live server
            self.receiver_error = f"{type(exc).__name__}: {exc}"
            self.observer.fatal_error = self.receiver_error

    async def send_json(self, event: Mapping[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.observer.log(
            "client",
            str(event.get("type") or "unknown"),
            payload=_safe_event_snapshot(event),
        )
        await self.websocket.send(payload)

    async def _pace_next_packet(self) -> None:
        now = time.monotonic()
        if self._next_packet_at > now:
            await asyncio.sleep(self._next_packet_at - now)
        self._next_packet_at = time.monotonic() + self.packet_seconds

    async def send_input_packet(
        self,
        payload: bytes,
        *,
        input_id: str,
        timeout: float | None = None,
        speech_active: bool = False,
        continuation_tick: bool = False,
    ) -> None:
        """Send one packet and wait until the server has consumed it.

        The wire protocol has no separate input acknowledgement.  A matching
        official output delta is the observable completion point for one
        backend transaction.  We wait for it for one packet interval, but a
        legal empty unit may produce no output at all; that case advances only
        after the same bounded interval and is recorded in the report.
        """

        await self._pace_next_packet()
        acknowledgement = self.observer.register_input(input_id)
        allow_sampled_decision = self._allow_sampled_decision(speech_active)
        await self.send_json(
            build_input_append(
                payload,
                input_id=input_id,
                speech_active=speech_active,
                allow_sampled_decision=allow_sampled_decision,
                continuation_tick=continuation_tick,
            )
        )
        if timeout is None:
            timeout = max(5.0, self.packet_seconds * 4.0)
        # A spoken MiniCPM unit routinely takes longer than one packet period
        # because semantic TTS and Token2Wav run before the audio delta is
        # emitted.  Advancing after only ``packet_seconds`` can leave multiple
        # silence packets queued behind that unit; those stale packets then
        # cross a turn_end boundary and become the first inputs of the next
        # turn.  Keep the wait bounded by the turn deadline, but otherwise use
        # the existing service budget rather than the transport cadence.
        ack_budget = min(timeout, self.acknowledgement_timeout)
        try:
            await self.wait_for(
                acknowledgement,
                ack_budget,
                f"input consumption ({input_id})",
            )
        except TimeoutError:
            # If the overall turn still has time left, this can be an empty
            # model unit rather than a transport failure.  Do not send faster
            # than the measured service budget, and retain the distinction in
            # the evidence.  A deadline-sized wait means the turn itself has
            # no budget left and remains a hard failure.
            if self.observer.fatal_error:
                raise RuntimeError(self.observer.fatal_error)
            if timeout <= self.packet_seconds + 1e-6:
                self.observer.log(
                    "probe",
                    "input_backpressure_timeout",
                    input_id=input_id,
                    timeout_seconds=timeout,
                )
                raise
            fallback = {
                "input_id": input_id,
                "wait_seconds": self.packet_seconds,
                "monotonic": round(time.monotonic(), 6),
            }
            self.observer.input_pacing_fallbacks.append(fallback)
            self.observer.log("probe", "input_pacing_fallback", **fallback)
        # The server sends the final audio delta immediately before the
        # explicit turn_end listen marker.  Give the receiver task one small
        # settlement window so finish_turn observes that boundary before it
        # decides to enqueue another silence packet.
        await asyncio.sleep(min(0.05, max(0.0, timeout)))
        self.observer.log(
            "probe",
            "input_acknowledged",
            input_id=input_id,
        )

    async def wait_for(self, flag: asyncio.Event, timeout: float, label: str) -> None:
        try:
            await asyncio.wait_for(flag.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"timed out waiting for {label}") from exc
        if self.observer.fatal_error:
            raise RuntimeError(self.observer.fatal_error)

    async def stream_pcm(
        self,
        payload: bytes,
        *,
        label: str,
        deadline: float | None = None,
        speech_active: bool = True,
    ) -> None:
        chunks = chunk_float32_pcm(payload)
        for index, chunk in enumerate(chunks, start=1):
            input_id = f"probe_{label}_{index:04d}"
            timeout = None if deadline is None else max(0.01, deadline - time.monotonic())
            await self.send_input_packet(
                chunk,
                input_id=input_id,
                timeout=timeout,
                speech_active=speech_active,
            )

    async def finish_turn(
        self,
        turn: TurnResult,
        *,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> None:
        started = time.monotonic()
        if deadline is None:
            if timeout is None:
                raise ValueError("finish_turn requires timeout or deadline")
            deadline = time.monotonic() + timeout
        packet_index = 0
        while not turn.done.is_set() and time.monotonic() < deadline:
            packet_index += 1
            remaining = max(0.01, deadline - time.monotonic())
            input_id = f"turn_{turn.number:02d}_silence_{packet_index:04d}"
            try:
                await self.send_input_packet(
                    SILENCE_PACKET,
                    input_id=input_id,
                    timeout=remaining,
                    continuation_tick=turn.audio_started.is_set(),
                )
            except TimeoutError as exc:
                turn.error = str(exc)
                turn.completion_reason = "timeout"
                turn.completed_at = time.monotonic()
                turn.done.set()
                break
        if not turn.done.is_set():
            elapsed = max(0.0, time.monotonic() - started)
            turn.error = f"turn {turn.number} timed out after {elapsed:.1f}s"
            turn.completion_reason = "timeout"
            turn.completed_at = time.monotonic()
            turn.done.set()
        self.last_completed_turn = turn
        if self.observer.fatal_error and not turn.error:
            turn.error = self.observer.fatal_error
            turn.completion_reason = "receiver_error"
        await asyncio.sleep(0.15)
        answer = self.output / f"answer-{turn.number:02d}.wav"
        write_float32_wav(answer, turn.audio or encode_float32_pcm([0.0]), OUTPUT_RATE)
        turn.audio_path = str(answer)
        summary = turn.summary()
        (self.output / f"turn-{turn.number:02d}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.observer.log("probe", "turn_finished", summary=summary)

    async def run_turn(self, number: int, question: str, payload: bytes, *, timeout: float) -> TurnResult:
        turn = TurnResult(number, question)
        self.observer.set_current(turn)
        deadline = time.monotonic() + timeout
        try:
            await self.stream_pcm(
                payload,
                label=f"turn_{number:02d}",
                deadline=deadline,
            )
            await self.finish_turn(turn, deadline=deadline)
        except TimeoutError as exc:
            turn.error = str(exc)
            turn.completion_reason = "timeout"
            turn.completed_at = time.monotonic()
            turn.done.set()
            await self.finish_turn(turn, deadline=time.monotonic())
        self.observer.set_current(None)
        return turn

    async def run_barge_in(
        self,
        questions: Sequence[str],
        payloads: Sequence[bytes],
        *,
        timeout: float,
    ) -> list[TurnResult]:
        first = TurnResult(1, questions[0])
        self.observer.set_current(first)
        await self.stream_pcm(payloads[0], label="barge_first")
        onset_deadline = time.monotonic() + timeout
        packet_index = 0
        while not first.audio_started.is_set() and time.monotonic() < onset_deadline:
            packet_index += 1
            await self.send_json(
                build_input_append(
                    SILENCE_PACKET,
                    input_id=f"barge_first_silence_{packet_index:04d}",
                    speech_active=False,
                    allow_sampled_decision=self._allow_sampled_decision(False),
                )
            )
            remaining = max(0.0, min(self.packet_seconds, onset_deadline - time.monotonic()))
            if remaining:
                try:
                    await asyncio.wait_for(first.audio_started.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
        if not first.audio_started.is_set():
            exc = TimeoutError("timed out waiting for first speech audio onset")
            first.error = str(exc)
            first.completion_reason = "barge_in_not_reached"
            first.completed_at = time.monotonic()
            first.done.set()
            await self.finish_turn(first, timeout=0.1)
            self.observer.set_current(None)
            return [first]

        self.observer.log("probe", "barge_in_started", from_turn=1, to_turn=2)
        await self.send_json({"type": "control", "payload": {"type": "response.cancel"}})
        first.saw_interrupt = True
        first.completion_reason = "barge_in"
        first.completed_at = time.monotonic()
        first.done.set()
        await self.finish_turn(first, timeout=0.1)

        second = TurnResult(2, questions[1])
        self.observer.set_current(second)
        await self.stream_pcm(payloads[1], label="barge_second")
        await self.finish_turn(second, timeout=timeout)
        third = await self.run_turn(3, questions[2], payloads[2], timeout=timeout)
        self.observer.set_current(None)
        return [first, second, third]


def _questions_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    selected = [bool(args.story), bool(args.barge_in)]
    if sum(selected) > 1:
        raise ValueError("--story and --barge-in are mutually exclusive")
    if args.question:
        if args.story or args.barge_in:
            raise ValueError("--question cannot be combined with --story or --barge-in")
        return tuple(args.question)
    if args.story:
        return STORY_QUESTIONS
    if args.barge_in:
        return BARGE_IN_QUESTIONS
    return NORMAL_QUESTIONS


def compact_terminal_summary(output: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep stdout useful without duplicating the full on-disk audit trace."""

    kv_lengths = report.get("kv_cache_lengths") or []
    return {
        "event": "summary",
        "output": str(output),
        "mode": report.get("mode"),
        "session_id": report.get("session_id"),
        "duration_seconds": report.get("duration_seconds"),
        "requested_turns": report.get("requested_turns"),
        "completed_turns": report.get("completed_turns"),
        "passed_turns": report.get("passed_turns"),
        "failed_turns": report.get("failed_turns"),
        "failures": report.get("failures"),
        "receiver_error": report.get("receiver_error"),
        "unsolicited_output_count": len(report.get("unsolicited_outputs") or []),
        "repeated_text_count": len(report.get("repeated_text") or []),
        "input_pacing_fallback_count": len(report.get("input_pacing_fallbacks") or []),
        "metrics_seen": report.get("metrics_seen"),
        "kv_cache": {
            "first": kv_lengths[0] if kv_lengths else None,
            "peak": max(kv_lengths) if kv_lengths else None,
            "final": kv_lengths[-1] if kv_lengths else None,
        },
        "reset_confirmed": report.get("reset_confirmed"),
        "turns": [
            {
                "number": turn.get("number"),
                "text": turn.get("text"),
                "error": turn.get("error"),
                "completion_reason": turn.get("completion_reason"),
                "first_audio_latency_seconds": turn.get("first_audio_latency_seconds"),
                "output_audio_seconds": turn.get("output_audio_seconds"),
                "model_generation_rtf": turn.get("model_generation_rtf"),
            }
            for turn in report.get("turns") or []
        ],
    }


async def run_probe(args: argparse.Namespace) -> int:
    questions = _questions_for_args(args)
    mode = "barge-in" if args.barge_in else "story" if args.story else "normal"
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output = args.output / stamp
    output.mkdir(parents=True, exist_ok=False)
    probe: NativeDuplexProbe | None = NativeDuplexProbe(
        None,
        output,
        packet_seconds=args.packet_seconds,
        acknowledgement_timeout=args.acknowledgement_timeout,
    )
    turns: list[TurnResult] = []
    started = time.monotonic()

    try:
        wav_paths = synthesize_questions(output, questions, voice=args.say_voice)
        payloads = [_float32_from_wav(path) for path in wav_paths]
        try:
            from websockets.asyncio.client import connect  # type: ignore[import-not-found]
        except ImportError:
            try:
                from websockets import connect  # type: ignore[import-not-found,no-redef]
            except ImportError as exc:
                raise RuntimeError(
                    "network probing requires the Python 'websockets' package"
                ) from exc

        async with connect(args.url, max_size=None, ping_timeout=30) as websocket:
            probe = NativeDuplexProbe(
                websocket,
                output,
                packet_seconds=args.packet_seconds,
                acknowledgement_timeout=args.acknowledgement_timeout,
            )
            receiver = asyncio.create_task(probe.receive_loop())
            try:
                await probe.wait_for(probe.observer.queue_done, args.connect_timeout, "session.queue_done")
                await probe.send_json(build_session_init(
                    args.system_prompt,
                    listen_probability_scale=args.listen_probability_scale,
                    temperature=args.temperature,
                    sliding_window_high=args.sliding_window_high,
                    sliding_window_low=args.sliding_window_low,
                ))
                await probe.wait_for(
                    probe.observer.session_created,
                    args.connect_timeout,
                    "session.created",
                )
                if args.silence_seconds > 0:
                    await probe.stream_pcm(
                        encode_float32_pcm([0.0] * int(INPUT_RATE * args.silence_seconds)),
                        label="initial_silence",
                        speech_active=False,
                    )
                if args.barge_in:
                    turns.extend(
                        await probe.run_barge_in(
                            questions, payloads, timeout=args.turn_timeout
                        )
                    )
                else:
                    for index, (question, payload) in enumerate(zip(questions, payloads), start=1):
                        turn = await probe.run_turn(
                            index, question, payload, timeout=args.turn_timeout
                        )
                        turns.append(turn)
                        if turn.error:
                            break
                        if args.reset_after_turn == index:
                            probe.observer.session_reset.clear()
                            await probe.send_json({
                                "type": "control",
                                "payload": {"type": "session.reset"},
                            })
                            await probe.wait_for(
                                probe.observer.session_reset,
                                args.connect_timeout,
                                "session reset confirmation",
                            )
                        await asyncio.sleep(args.turn_gap_seconds)
                if args.post_silence_seconds > 0:
                    await probe.stream_pcm(
                        encode_float32_pcm([0.0] * int(INPUT_RATE * args.post_silence_seconds)),
                        label="post_silence",
                        speech_active=False,
                    )
                await asyncio.sleep(0.25)
                await probe.send_json({"type": "session.close", "reason": "probe_complete"})
                try:
                    await probe.wait_for(probe.observer.session_closed, 5.0, "session.closed")
                except (TimeoutError, RuntimeError):
                    # The report records a missing close event and the explicit
                    # validation below turns it into a non-zero exit code.
                    pass
            finally:
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
    except Exception as exc:
        assert probe is not None
        probe.observer.fatal_error = str(exc)
    finally:
        assert probe is not None
        failures = validate_report(mode, questions, turns, probe.observer)
        if not probe.observer.session_closed.is_set():
            failures.append("missing session.closed")
        report = {
            "protocol": "MiniCPM native official realtime",
            "url": args.url,
            "mode": mode,
            "session_id": probe.observer.session_id,
            "server_mode": probe.observer.server_mode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "requested_turns": len(questions),
            "completed_turns": len(turns),
            "passed_turns": len(turns) - sum(1 for turn in turns if turn.error),
            "failed_turns": [turn.number for turn in turns if turn.error],
            "failures": failures,
            "receiver_error": probe.receiver_error,
            "unsolicited_outputs": probe.observer.unsolicited_outputs,
            "repeated_text": probe.observer.repeated_text,
            "metrics_seen": len(probe.observer.all_metrics),
            "kv_cache_lengths": probe.observer.all_kv_cache_lengths,
            "interrupt_traces": probe.observer.interrupt_traces,
            "input_acknowledgements": probe.observer.input_acknowledgements,
            "input_pacing_fallbacks": probe.observer.input_pacing_fallbacks,
            "packet_seconds_minimum": probe.packet_seconds,
            "initial_silence_seconds": args.silence_seconds,
            "post_silence_seconds": args.post_silence_seconds,
            "reset_after_turn": args.reset_after_turn,
            "reset_confirmed": probe.observer.session_reset.is_set(),
            "turns": [turn.summary() for turn in turns],
        }
        output.joinpath("timeline.json").write_text(
            json.dumps(probe.observer.timeline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.joinpath("summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(compact_terminal_summary(output, report), ensure_ascii=False))
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--question", action="append", help="override normal test questions")
    parser.add_argument("--story", action="store_true", help="run context-dependent story turns")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="cancel after turn one audio onset, then verify turns two and three",
    )
    parser.add_argument("--say-voice", default=DEFAULT_SAY_VOICE)
    parser.add_argument("--system-prompt")
    parser.add_argument("--listen-probability-scale", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--sliding-window-high", type=int)
    parser.add_argument("--sliding-window-low", type=int)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--turn-timeout", type=float, default=50.0)
    parser.add_argument("--turn-gap-seconds", type=float, default=0.35)
    parser.add_argument(
        "--packet-seconds",
        type=float,
        default=1.6,
        help="minimum spacing between packets; each packet also waits for its output acknowledgement",
    )
    parser.add_argument(
        "--acknowledgement-timeout",
        type=float,
        default=5.0,
        help="maximum wait for one input result before a paced fallback",
    )
    parser.add_argument("--silence-seconds", type=float, default=1.0)
    parser.add_argument("--post-silence-seconds", type=float, default=1.0)
    parser.add_argument("--reset-after-turn", type=int)
    args = parser.parse_args(argv)
    if args.silence_seconds < 0 or args.post_silence_seconds < 0:
        parser.error("silence durations must be non-negative")
    if args.turn_timeout <= 0 or args.connect_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.turn_gap_seconds < 0:
        parser.error("turn gap must be non-negative")
    if args.packet_seconds <= 0:
        parser.error("packet spacing must be positive")
    if args.acknowledgement_timeout <= 0:
        parser.error("acknowledgement timeout must be positive")
    if args.reset_after_turn is not None and args.reset_after_turn <= 0:
        parser.error("reset-after-turn must be positive")
    if args.listen_probability_scale is not None and args.listen_probability_scale < 0:
        parser.error("listen probability scale must be non-negative")
    if args.temperature is not None and args.temperature < 0:
        parser.error("temperature must be non-negative")
    if args.sliding_window_high is not None and args.sliding_window_high <= 0:
        parser.error("sliding window high watermark must be positive")
    if args.sliding_window_low is not None and args.sliding_window_high is None:
        parser.error("sliding window low watermark requires a high watermark")
    if args.sliding_window_low is not None and not (0 < args.sliding_window_low <= args.sliding_window_high):
        parser.error("sliding window low watermark must be positive and at most high")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return asyncio.run(run_probe(args))
    except (RuntimeError, ValueError, TimeoutError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
