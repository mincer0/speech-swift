#!/usr/bin/env python3
"""Subprocess regression for the MiniCPM realtime WebSocket close path.

This is intentionally dependency-free: it implements only the small RFC 6455
client surface needed by the fake server.  Run it after a release build with:

    python3 scripts/tests/test_minicpm_websocket_close.py

The test starts ``minicpm-mlx-server --fake``, performs five queue/init/close
handshakes, responds to the server close frame, and then checks health after a
short survival window.  It never loads a model.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ProtocolError(RuntimeError):
    """The server did not complete the expected WebSocket exchange."""


class WebSocketClient:
    """Minimal masked WebSocket client for the close regression."""

    def __init__(self, host: str, port: int, path: str, timeout: float = 10.0) -> None:
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.buffer = bytearray()
        self.close_sent = False
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        self.socket.sendall(request)
        header = self._read_until(b"\r\n\r\n")
        header_bytes, _, trailing = header.partition(b"\r\n\r\n")
        self.buffer.clear()
        self.buffer.extend(trailing)
        lines = header_bytes.decode("latin1").split("\r\n")
        if not lines or not lines[0].startswith("HTTP/1.1 101 "):
            raise ProtocolError(f"WebSocket upgrade failed: {lines[:1]}")
        fields = {
            name.strip().lower(): value.strip()
            for line in lines[1:]
            if ":" in line
            for name, value in [line.split(":", 1)]
        }
        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if fields.get("sec-websocket-accept") != accept:
            raise ProtocolError("WebSocket upgrade returned an invalid accept key")

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self.buffer:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ProtocolError("connection closed during WebSocket upgrade")
            self.buffer.extend(chunk)
        return bytes(self.buffer)

    def _read_exact(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.socket.recv(max(4096, count - len(self.buffer)))
            if not chunk:
                raise ProtocolError("connection closed in a WebSocket frame")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        if len(payload) < 126:
            header = bytes((0x80 | opcode, 0x80 | len(payload)))
        elif len(payload) <= 0xFFFF:
            header = bytes((0x80 | opcode, 0x80 | 126)) + len(payload).to_bytes(2, "big")
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + len(payload).to_bytes(8, "big")
        mask = secrets.token_bytes(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(header + mask + masked)

    def send_json(self, value: dict[str, Any]) -> None:
        self._send_frame(0x1, json.dumps(value, separators=(",", ":")).encode("utf-8"))

    def send_close(self, payload: bytes = b"") -> None:
        if not self.close_sent:
            self._send_frame(0x8, payload)
            self.close_sent = True

    def recv_frame(self) -> tuple[int, bytes]:
        first, second = self._read_exact(2)
        opcode = first & 0x0F
        if not first & 0x80:
            raise ProtocolError("fragmented WebSocket frames are not supported by the smoke client")
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(self._read_exact(2), "big")
        elif length == 127:
            length = int.from_bytes(self._read_exact(8), "big")
        masked = bool(second & 0x80)
        mask = self._read_exact(4) if masked else b""
        payload = bytearray(self._read_exact(length))
        if masked:
            for index in range(length):
                payload[index] ^= mask[index % 4]
        return opcode, bytes(payload)

    def receive_event(self, expected: str, timeout: float = 10.0) -> dict[str, Any]:
        self.socket.settimeout(timeout)
        while True:
            opcode, payload = self.recv_frame()
            if opcode == 0x9:  # ping
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode == 0x8:
                self.send_close(payload)
                raise ProtocolError(f"server closed before {expected}: {payload!r}")
            if opcode != 0x1:
                raise ProtocolError(f"unexpected WebSocket opcode {opcode}")
            event = json.loads(payload.decode("utf-8"))
            if event.get("type") == expected:
                return event

    def complete_close(self) -> None:
        """Send the client close and acknowledge the server close frame."""

        self.send_close()
        self.socket.settimeout(10.0)
        while True:
            opcode, payload = self.recv_frame()
            if opcode == 0x9:
                self._send_frame(0xA, payload)
            elif opcode == 0x8:
                if not self.close_sent:
                    self.send_close(payload)
                return

    def close(self) -> None:
        self.socket.close()


def _free_port(host: str) -> int:
    with socket.socket() as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _health(host: str, port: int) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
        return response.status, response.read().decode("utf-8")


def _wait_for_health(process: subprocess.Popen[bytes], host: str, port: int) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with {process.returncode}")
        try:
            status, _ = _health(host, port)
            if status == 200:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError("server health endpoint did not become ready")


def _run_round(host: str, port: int, round_number: int) -> list[str]:
    client = WebSocketClient(host, port, "/v1/realtime?mode=audio")
    seen: list[str] = []
    try:
        event = client.receive_event("session.queue_done")
        seen.append(str(event["type"]))
        client.send_json({"type": "session.init", "payload": {"mode": "full_duplex"}})
        event = client.receive_event("session.created")
        seen.append(str(event["type"]))
        client.send_json({"type": "session.close", "reason": f"close_round_{round_number}"})
        closed = client.receive_event("session.closed")
        seen.append(str(closed["type"]))
        if closed.get("reason") != f"close_round_{round_number}":
            raise ProtocolError(f"round {round_number}: wrong close reason {closed}")
        client.complete_close()
        return seen
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--executable", type=Path, default=root / ".build/release/minicpm-mlx-server")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--survival-seconds", type=float, default=3.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("--rounds must be at least 5")
    executable = args.executable.expanduser().resolve()
    if not executable.is_file():
        raise SystemExit(f"missing release executable: {executable}")
    port = args.port or _free_port(args.host)
    with tempfile.TemporaryDirectory(prefix="minicpm-websocket-close-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(executable), "--fake", "--host", args.host, "--port", str(port), "--data-dir", str(Path(directory) / "data")],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_for_health(process, args.host, port)
                for round_number in range(1, args.rounds + 1):
                    seen = _run_round(args.host, port, round_number)
                    if process.poll() is not None:
                        raise RuntimeError(f"round {round_number}: server exited with {process.returncode}")
                    print(f"round {round_number}: {seen}; server alive")
                time.sleep(args.survival_seconds)
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during survival window with {process.returncode}")
                status, body = _health(args.host, port)
                print(f"health after {args.rounds} rounds + {args.survival_seconds:g}s: {status} {body}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                log.flush()
                output = log_path.read_text("utf-8", errors="replace")
                if process.returncode not in (0, -15):
                    raise RuntimeError(f"server returncode {process.returncode}; log:\n{output}")
                if output.strip():
                    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
