#!/usr/bin/env python3
"""JSONL smoke test for the native MiniCPM helper processes.

This script intentionally does not use SwiftPM/XCTest.  The package's full
test target currently contains unrelated optional MLX modules, while these
checks only need the already-built helper executables.  No model process is
started unless its executable *and* model paths are supplied explicitly.

Examples (run serially, one helper at a time when using large models)::

    python3 Tests/MiniCPMAudioTests/helper_process_smoke.py \
      --audio-helper .build/arm64-apple-macosx/debug/minicpm-audio-helper \
      --audio-model /path/to/MiniCPM-o-4_5-audio-mlx

    python3 Tests/MiniCPMAudioTests/helper_process_smoke.py \
      --token2wav-helper .build/arm64-apple-macosx/debug/minicpm-token2wav-helper \
      --token2wav-model /path/to/MiniCPM-o-4_5-token2wav-mlx \
      --token2wav-prompt /path/to/prompt.safetensors

The audio helper is exercised with 16 kHz little-endian Float32 PCM.  The
token2wav helper receives a 25+3 silence window followed by a final token so
the test covers both stable-window emission and final flush.  Set
``--skip-audio`` or ``--skip-token2wav`` to run only one side.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
from typing import Any


class SmokeFailure(RuntimeError):
    """A helper violated the expected JSONL protocol."""


class JSONLProcess:
    def __init__(self, command: list[str], timeout: float) -> None:
        self.command = command
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.request_id = 0

    def read(self) -> dict[str, Any]:
        process = self.process
        if process.stdout is None:
            raise SmokeFailure("helper stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise SmokeFailure(
                    f"helper response timed out after {self.timeout:.1f}s"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise SmokeFailure(
                f"helper exited with code {process.poll()}: {stderr}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SmokeFailure(f"helper emitted non-JSON: {line!r}") from exc
        if not response.get("ok"):
            raise SmokeFailure(response.get("error") or "helper request failed")
        return response

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        process = self.process
        if process.stdin is None:
            raise SmokeFailure("helper stdin is unavailable")
        self.request_id += 1
        request_id = str(self.request_id)
        process.stdin.write(
            json.dumps(
                {"id": request_id, "op": op, **payload},
                separators=(",", ":"),
            )
            + "\n"
        )
        process.stdin.flush()
        response = self.read()
        if response.get("id") != request_id:
            raise SmokeFailure(
                f"response id mismatch: {response.get('id')!r} != {request_id!r}"
            )
        return response

    def close(self) -> None:
        process = self.process
        if process.poll() is None:
            try:
                self.request("shutdown")
            except Exception:
                process.kill()
        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.timeout)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _float32_pcm_base64(count: int, value: float = 0.0) -> str:
    return base64.b64encode(struct.pack(f"<{count}f", *([value] * count))).decode(
        "ascii"
    )


def _require_file(path: str | None, label: str) -> Path:
    if not path:
        raise SmokeFailure(f"{label} was not supplied")
    value = Path(path).expanduser()
    if not value.is_file():
        raise SmokeFailure(f"{label} does not exist: {value}")
    return value


def _require_directory(path: str | None, label: str) -> Path:
    if not path:
        raise SmokeFailure(f"{label} was not supplied")
    value = Path(path).expanduser()
    if not value.is_dir():
        raise SmokeFailure(f"{label} does not exist: {value}")
    return value


def smoke_audio(args: argparse.Namespace) -> None:
    helper = _require_file(args.audio_helper, "--audio-helper")
    model = _require_directory(args.audio_model, "--audio-model")
    client = JSONLProcess([str(helper), str(model)], args.timeout)
    try:
        ready = client.read()
        if ready.get("event") != "ready":
            raise SmokeFailure(f"unexpected audio startup event: {ready}")

        reset = client.request("reset")
        if reset.get("cacheLength") != 0:
            raise SmokeFailure(f"audio reset retained cache: {reset}")

        # The first streaming chunk is 16,480 samples (100 stable mel frames).
        streaming = client.request(
            "accept_audio",
            pcmBase64=_float32_pcm_base64(16_480),
            samples=16_480,
            sampleRate=16_000,
            isFinal=False,
        )
        if not streaming.get("embeddingsBase64"):
            raise SmokeFailure(f"audio accept_audio emitted no embedding: {streaming}")
        live_cache_length = int(streaming.get("cacheLength") or 0)
        if live_cache_length <= 0:
            raise SmokeFailure(f"audio streaming cache did not grow: {streaming}")

        # A reference clip must use a no-cache path.  Verify that the live
        # stream cache remains exactly where it was before this request.
        reference = client.request(
            "encode_reference",
            pcmBase64=_float32_pcm_base64(1_600),
            samples=1_600,
            sampleRate=16_000,
        )
        if not reference.get("embeddingsBase64"):
            raise SmokeFailure(f"audio reference emitted no embedding: {reference}")
        if int(reference.get("cacheLength") or 0) != live_cache_length:
            raise SmokeFailure(
                "reference encoding polluted live cache: "
                f"before={live_cache_length}, response={reference}"
            )

        stats = client.request("stats")
        if int(stats.get("cacheLength") or 0) != live_cache_length:
            raise SmokeFailure(f"audio stats changed cache length: {stats}")
        client.request("reset")
        after_reset = client.request("stats")
        if int(after_reset.get("cacheLength") or 0) != 0:
            raise SmokeFailure(f"audio reset did not clear cache: {after_reset}")
        print("audio helper: PASS (accept_audio, reference isolation, reset/stats)")
    finally:
        client.close()


def smoke_token2wav(args: argparse.Namespace) -> None:
    helper = _require_file(args.token2wav_helper, "--token2wav-helper")
    model = _require_directory(args.token2wav_model, "--token2wav-model")
    prompt = _require_file(args.token2wav_prompt, "--token2wav-prompt")
    command = [str(helper), str(model), str(prompt), str(args.ode_steps)]
    client = JSONLProcess(command, args.timeout)
    try:
        ready = client.read()
        if ready.get("event") != "ready":
            raise SmokeFailure(f"unexpected token2wav startup event: {ready}")
        initial_stats = client.request("stats")
        context = initial_stats.get("context") or {}
        if int(context.get("pendingTokens") or 0) != 3:
            raise SmokeFailure(
                "token2wav prompt did not install the three-token lookahead: "
                f"{initial_stats}"
            )

        stable = client.request(
            "append",
            tokens=[4_218] * 28,
            forceFlush=False,
            isFinal=False,
        )
        chunks = stable.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise SmokeFailure(f"token2wav append did not emit one stable chunk: {stable}")
        if chunks[0].get("isFinal"):
            raise SmokeFailure(f"stable token2wav chunk was marked final: {stable}")

        final = client.request(
            "append",
            tokens=[4_218],
            forceFlush=False,
            isFinal=True,
        )
        final_chunks = final.get("chunks")
        if not isinstance(final_chunks, list) or len(final_chunks) != 1:
            raise SmokeFailure(f"token2wav final append did not flush: {final}")
        if not final_chunks[0].get("isFinal"):
            raise SmokeFailure(f"token2wav final chunk was not marked final: {final}")

        # A completed turn is closed; interrupt must reset it to a prompt-based
        # turn without retaining generated speech tokens.
        interrupted = client.request("interrupt")
        context = interrupted.get("context") or {}
        if int(context.get("pendingTokens") or 0) != 3:
            raise SmokeFailure(f"token2wav interrupt did not restore lookahead: {interrupted}")
        stats = client.request("stats")
        if int((stats.get("context") or {}).get("pendingTokens") or 0) != 3:
            raise SmokeFailure(f"token2wav stats mismatch after interrupt: {stats}")
        print("token2wav helper: PASS (append, final flush, interrupt/stats)")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-helper",
        default=os.environ.get("MINICPM_AUDIO_HELPER"),
        help="path to minicpm-audio-helper (or MINICPM_AUDIO_HELPER)",
    )
    parser.add_argument(
        "--audio-model",
        default=os.environ.get("MINICPM_AUDIO_MODEL"),
        help="converted MiniCPM audio model directory (or MINICPM_AUDIO_MODEL)",
    )
    parser.add_argument(
        "--token2wav-helper",
        default=os.environ.get("MINICPM_TOKEN2WAV_HELPER"),
        help="path to minicpm-token2wav-helper (or MINICPM_TOKEN2WAV_HELPER)",
    )
    parser.add_argument(
        "--token2wav-model",
        default=os.environ.get("MINICPM_TOKEN2WAV_MODEL"),
        help="converted MiniCPM token2wav model directory (or MINICPM_TOKEN2WAV_MODEL)",
    )
    parser.add_argument(
        "--token2wav-prompt",
        default=os.environ.get("MINICPM_TOKEN2WAV_PROMPT"),
        help="prepared prompt.safetensors fixture (or MINICPM_TOKEN2WAV_PROMPT)",
    )
    parser.add_argument("--ode-steps", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-token2wav", action="store_true")
    args = parser.parse_args()
    if args.skip_audio and args.skip_token2wav:
        parser.error("at least one helper smoke must be enabled")
    if args.ode_steps <= 0:
        parser.error("--ode-steps must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        # Keep the default order deterministic.  Callers should still run this
        # script only when no other model-heavy Swift process is active.
        if not args.skip_audio:
            smoke_audio(args)
        if not args.skip_token2wav:
            smoke_token2wav(args)
    except SmokeFailure as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
