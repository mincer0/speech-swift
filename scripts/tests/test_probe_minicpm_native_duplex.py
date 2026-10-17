#!/usr/bin/env python3
"""No-model protocol, PCM, and decision tests for the native duplex probe."""

from __future__ import annotations

import base64
import asyncio
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "probe_minicpm_native_duplex.py"
SPEC = importlib.util.spec_from_file_location("probe_minicpm_native_duplex", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"unable to import probe from {SCRIPT_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def event(kind: str, **fields: object) -> dict[str, object]:
    return {"type": "response.output.delta", "kind": kind, **fields}


class FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def stream():
            for message in self.messages:
                yield message

        return stream()


class AckingWebSocket:
    """Minimal send-side transport that acknowledges one input at a time."""

    def __init__(self, observer: object) -> None:
        self.observer = observer
        self.sent: list[dict[str, object]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def send(self, payload: str) -> None:
        message = json.loads(payload)
        self.sent.append(message)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        input_object = message["input"]
        input_id = input_object["input_id"]
        self.observer.handle_event(
            {
                "type": "response.output.delta",
                "kind": "audio",
                "input_id": input_id,
                "audio": base64.b64encode(PROBE.encode_float32_pcm([0.0])).decode("ascii"),
                "format": "pcm_f32",
                "sample_rate": 24_000,
            }
        )
        self.in_flight -= 1


class NonAcknowledgingWebSocket:
    async def send(self, payload: str) -> None:
        del payload


class AudioThenTurnEndWebSocket:
    """Emit the server's real final-delta ordering with a short boundary lag."""

    def __init__(self, observer: object) -> None:
        self.observer = observer
        self.sent: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        message = json.loads(payload)
        self.sent.append(message)
        input_id = message["input"]["input_id"]
        audio = base64.b64encode(PROBE.encode_float32_pcm([0.0])).decode("ascii")
        self.observer.handle_event(
            event(
                "audio",
                input_id=input_id,
                audio=audio,
                format="pcm_f32",
                sample_rate=24_000,
            )
        )

        async def finish_boundary() -> None:
            await asyncio.sleep(0.01)
            self.observer.handle_event(
                event("listen", input_id=input_id, reason="turn_end")
            )

        asyncio.create_task(finish_boundary())


class ProbeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._compat_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls._compat_loop)

    def tearDown(self) -> None:
        # Python 3.9 closes the loop created by asyncio.run; restore a loop so
        # the synchronous observer tests can construct asyncio.Event fields.
        asyncio.set_event_loop(self._compat_loop)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._compat_loop.close()
        asyncio.set_event_loop(None)

    def test_receive_loop_parses_text_queue_and_created_events(self) -> None:
        websocket = FakeWebSocket(
            [
                json.dumps({"type": "session.queue_done", "ticket_id": "t"}),
                json.dumps({"type": "session.created", "session_id": "s"}),
            ]
        )
        probe = PROBE.NativeDuplexProbe(websocket, Path("/tmp/probe-test"))
        asyncio.run(probe.receive_loop())
        self.assertTrue(probe.observer.queue_done.is_set())
        self.assertTrue(probe.observer.session_created.is_set())
        self.assertEqual(probe.observer.session_id, "s")
        self.assertIsNone(probe.receiver_error)

    def test_receive_loop_rejects_binary_frames_without_blocking_text_events(self) -> None:
        websocket = FakeWebSocket(
            [
                b"not-json",
                json.dumps({"type": "session.queue_done", "ticket_id": "t"}),
            ]
        )
        probe = PROBE.NativeDuplexProbe(websocket, Path("/tmp/probe-test"))
        asyncio.run(probe.receive_loop())
        self.assertEqual(
            probe.receiver_error,
            "official realtime transport returned a binary frame",
        )
        self.assertTrue(probe.observer.queue_done.is_set())

    def test_packet_stream_waits_for_each_matching_input_ack(self) -> None:
        async def exercise() -> tuple[list[dict[str, object]], int, list[dict[str, object]]]:
            probe = PROBE.NativeDuplexProbe(
                None,
                Path("/tmp/probe-test"),
                packet_seconds=0.001,
            )
            websocket = AckingWebSocket(probe.observer)
            probe.websocket = websocket
            payload = PROBE.encode_float32_pcm(
                [0.25] * (PROBE.INPUT_PACKET_SAMPLES * 2)
            )
            await probe.stream_pcm(payload, label="paced")
            return websocket.sent, websocket.max_in_flight, probe.observer.input_acknowledgements

        sent, max_in_flight, acknowledgements = asyncio.run(exercise())
        self.assertEqual(
            [message["input"]["input_id"] for message in sent],
            ["probe_paced_0001", "probe_paced_0002"],
        )
        self.assertEqual(max_in_flight, 1)
        self.assertEqual(
            [item["input_id"] for item in acknowledgements],
            ["probe_paced_0001", "probe_paced_0002"],
        )

    def test_packet_ack_timeout_is_audited(self) -> None:
        async def exercise_with_probe() -> list[dict[str, object]]:
            probe = PROBE.NativeDuplexProbe(
                NonAcknowledgingWebSocket(),
                Path("/tmp/probe-test"),
                packet_seconds=0.001,
            )
            with self.assertRaises(TimeoutError):
                await probe.send_input_packet(
                    PROBE.SILENCE_PACKET,
                    input_id="no_ack",
                    timeout=0.001,
                )
            return probe.observer.timeline

        timeline = asyncio.run(exercise_with_probe())
        self.assertTrue(
            any(item.get("event") == "input_backpressure_timeout" for item in timeline)
        )

    def test_startup_silence_requires_explicit_input_commit(self) -> None:
        observer = PROBE.EventObserver()
        acknowledgement = observer.register_input(
            "startup", phase="initial_silence", require_committed=True
        )
        observer.handle_event(event("text", input_id="startup", text="抢答"))
        observer.handle_event(event("listen", input_id="startup", reason="turn_end"))
        self.assertFalse(acknowledgement.is_set())
        observer.handle_event({"type": "input.committed", "input_id": "startup"})
        self.assertTrue(acknowledgement.is_set())
        self.assertEqual(observer.input_acknowledgements[-1]["event_type"], "input.committed")

    def test_text_is_not_an_input_completion_ack(self) -> None:
        observer = PROBE.EventObserver()
        acknowledgement = observer.register_input("question")
        observer.handle_event(event("text", input_id="question", text="第一块"))
        self.assertFalse(acknowledgement.is_set())
        observer.handle_event(event("listen", input_id="question", reason="turn_end"))
        self.assertTrue(acknowledgement.is_set())

    def test_startup_silence_timeout_never_uses_paced_fallback(self) -> None:
        async def exercise() -> object:
            probe = PROBE.NativeDuplexProbe(
                NonAcknowledgingWebSocket(),
                Path("/tmp/probe-test"),
                packet_seconds=0.001,
                acknowledgement_timeout=0.001,
            )
            with self.assertRaises(TimeoutError):
                await probe.stream_pcm(
                    PROBE.SILENCE_PACKET,
                    label="initial_silence",
                    speech_active=False,
                    deadline=PROBE.time.monotonic() + 0.01,
                )
            return probe.observer

        observer = asyncio.run(exercise())
        self.assertEqual(observer.input_pacing_fallbacks, [])
        failures = [
            item for item in observer.timeline
            if item.get("event") == "input_backpressure_timeout"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["required_event"], "input.committed")
        self.assertFalse(any(item.get("event") == "input_acknowledged" for item in observer.timeline))

    def test_startup_silence_waits_through_text_until_commit(self) -> None:
        async def exercise() -> object:
            probe = PROBE.NativeDuplexProbe(
                None, Path("/tmp/probe-test"), packet_seconds=0.001,
                acknowledgement_timeout=0.001,
            )

            class DelayedCommitWebSocket:
                async def send(self, payload: str) -> None:
                    input_id = json.loads(payload)["input"]["input_id"]
                    probe.observer.handle_event(event("text", input_id=input_id, text="抢答"))

                    async def commit() -> None:
                        await asyncio.sleep(0.01)
                        probe.observer.handle_event({"type": "input.committed", "input_id": input_id})

                    asyncio.create_task(commit())

            probe.websocket = DelayedCommitWebSocket()
            await probe.stream_pcm(
                PROBE.SILENCE_PACKET, label="initial_silence", speech_active=False,
                deadline=PROBE.time.monotonic() + 0.5,
            )
            return probe.observer

        observer = asyncio.run(exercise())
        self.assertEqual(observer.input_pacing_fallbacks, [])
        self.assertEqual(len(observer.input_acknowledgements), 1)
        self.assertEqual(observer.input_acknowledgements[0]["event_type"], "input.committed")

    def test_finish_turn_does_not_send_past_audio_then_turn_end_boundary(self) -> None:
        async def exercise(directory: Path) -> list[dict[str, object]]:
            probe = PROBE.NativeDuplexProbe(
                None,
                directory,
                packet_seconds=0.001,
            )
            websocket = AudioThenTurnEndWebSocket(probe.observer)
            probe.websocket = websocket
            turn = PROBE.TurnResult(1, "question")
            probe.observer.set_current(turn)
            await probe.finish_turn(turn, timeout=0.5)
            return websocket.sent

        with tempfile.TemporaryDirectory() as directory:
            sent = asyncio.run(exercise(Path(directory)))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["input"]["input_id"], "turn_01_silence_0001")

    def test_session_init_and_one_second_float32_input_envelope(self) -> None:
        init = PROBE.build_session_init("test")
        self.assertEqual(init["type"], "session.init")
        self.assertEqual(init["payload"]["mode"], "full_duplex")
        self.assertEqual(init["payload"]["system_prompt"], "test")
        self.assertEqual(
            init["payload"]["config"],
            {
                "force_listen_count": 0,
                "top_k": 20,
                "greedy_listen_speak_decision": False,
            },
        )
        calibrated = PROBE.build_session_init(
            listen_probability_scale=0.8,
            temperature=0.0,
            sliding_window_high=500,
            sliding_window_low=400,
        )
        self.assertEqual(
            calibrated["payload"]["config"],
            {
                "force_listen_count": 0,
                "top_k": 20,
                "greedy_listen_speak_decision": False,
                "listen_prob_scale": 0.8,
                "temperature": 0.0,
                "sliding_window_mode": "basic",
                "basic_window_high_tokens": 500,
                "basic_window_low_tokens": 400,
            },
        )
        payload = PROBE.encode_float32_pcm([0.25] * PROBE.INPUT_PACKET_SAMPLES)
        request = PROBE.build_input_append(
            payload,
            input_id="one",
            speech_active=True,
            allow_sampled_decision=True,
            continuation_tick=True,
        )
        self.assertEqual(request["type"], "input.append")
        input_object = request["input"]
        self.assertEqual(input_object["format"], "pcm_f32le")
        self.assertEqual(input_object["sample_rate"], 16_000)
        self.assertEqual(input_object["speech_active"], True)
        self.assertEqual(input_object["allow_sampled_listen_speak_decision"], True)
        self.assertEqual(input_object["continuation_tick"], True)
        decoded = base64.b64decode(input_object["audio"], validate=True)
        self.assertEqual(decoded, payload)
        self.assertEqual(struct.unpack("<f", decoded[:4])[0], 0.25)

    def test_chunking_pads_only_final_packet(self) -> None:
        payload = PROBE.encode_float32_pcm([0.5] * (PROBE.INPUT_PACKET_SAMPLES + 3))
        chunks = PROBE.chunk_float32_pcm(payload)
        self.assertEqual([len(chunk) for chunk in chunks], [PROBE.INPUT_PACKET_BYTES] * 2)
        self.assertEqual(chunks[0][-4:], struct.pack("<f", 0.5))
        self.assertEqual(chunks[1][:12], struct.pack("<3f", 0.5, 0.5, 0.5))
        self.assertEqual(chunks[1][12:], b"\0" * (PROBE.INPUT_PACKET_BYTES - 12))

    def test_float32_wav_uses_ieee_float_header(self) -> None:
        payload = PROBE.encode_float32_pcm([0.0, 1.0, -1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.wav"
            PROBE.write_float32_wav(path, payload, sample_rate=16_000)
            data = path.read_bytes()
            decoded = PROBE._float32_from_wav(path)
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(data[8:12], b"WAVE")
        self.assertEqual(struct.unpack_from("<H", data, 20)[0], 3)
        self.assertEqual(struct.unpack_from("<H", data, 22)[0], 1)
        self.assertEqual(struct.unpack_from("<I", data, 24)[0], 16_000)
        self.assertEqual(decoded, payload)

    def test_failed_turn_artifacts_can_be_rewritten_without_empty_pcm_error(self) -> None:
        turn = PROBE.TurnResult(1, "question")
        turn.error = "timed out"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            PROBE._persist_turn_artifacts(turn, output)
            PROBE._persist_turn_artifacts(turn, output)
            wav = Path(turn.audio_path).read_bytes()
            summary = json.loads((output / "turn-01.json").read_text())
        self.assertEqual(wav[-4:], PROBE.encode_float32_pcm([0.0]))
        self.assertEqual(summary["error"], "timed out")
        self.assertEqual(summary["output_audio_seconds"], 0.0)
        self.assertEqual(turn.audio, b"")

    def test_observer_tracks_queue_session_audio_listen_metrics_and_duplicate_text(self) -> None:
        observer = PROBE.EventObserver()
        observer.handle_event({"type": "session.queue_done", "ticket_id": "t"})
        observer.handle_event({"type": "session.created", "session_id": "s", "mode": "full_duplex"})
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.handle_event(event("text", text="好的"))
        observer.handle_event(event("text", text="好的", metrics={"kv_cache_length": 8}))
        audio = PROBE.encode_float32_pcm([0.1, 0.2])
        observer.handle_event(
            event(
                "audio",
                audio=base64.b64encode(audio).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
                metrics={"kv_cache_length": 9},
            )
        )
        observer.handle_event(event("listen", reason="turn_end"))
        self.assertTrue(observer.queue_done.is_set())
        self.assertTrue(observer.session_created.is_set())
        self.assertEqual(observer.session_id, "s")
        self.assertEqual(turn.text, "好的好的")
        self.assertEqual(len(turn.repeated_text), 1)
        self.assertEqual(turn.audio, audio)
        self.assertTrue(turn.audio_started.is_set())
        self.assertTrue(turn.saw_end_boundary)
        self.assertTrue(turn.done.is_set())
        self.assertEqual(turn.kv_cache_lengths, [8, 9])

    def test_turn_detects_adjacent_sentence_repeat_split_across_chunks(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)

        observer.handle_event(event("text", text="七加八等于十"))
        observer.handle_event(event("text", text="五。七加八"))
        observer.handle_event(event("text", text="等于十五。"))

        self.assertEqual(len(turn.repeated_text), 1)
        self.assertEqual(turn.repeated_text[0]["kind"], "adjacent_tandem")
        self.assertEqual(turn.repeated_text[0]["repeated_unit"], "七加八等于十五")
        self.assertEqual(observer.repeated_text, turn.repeated_text)

    def test_tandem_detector_ignores_short_natural_repetition(self) -> None:
        self.assertIsNone(PROBE.find_tandem_repeat("哈哈"))
        self.assertIsNone(PROBE.find_tandem_repeat("好的，我来回答"))

    def test_tandem_detector_rejects_invalid_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            PROBE.find_tandem_repeat("测试", minimum_unit_length=0)

    def test_generic_listen_markers_wait_for_explicit_turn_boundary(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.handle_event(event("listen", reason="listen"))
        observer.handle_event(event("listen", reason="model_listen"))
        observer.handle_event(event("listen", reason="awaiting_audio"))
        self.assertTrue(turn.saw_listen)
        self.assertFalse(turn.saw_end_boundary)
        self.assertFalse(turn.done.is_set())

        observer.handle_event(event("text", text="继续回答"))
        audio = PROBE.encode_float32_pcm([0.0, 0.0])
        observer.handle_event(
            event(
                "audio",
                audio=base64.b64encode(audio).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
            )
        )
        observer.handle_event(event("listen", reason="model_listen"))
        self.assertFalse(turn.done.is_set())
        observer.handle_event(event("listen", reason="turn_end"))
        self.assertTrue(turn.saw_end_boundary)
        self.assertTrue(turn.done.is_set())

    def test_official_session_state_interrupt_collects_metrics_trace(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        trace = {
            "epoch": 2,
            "unit_index": 6,
            "current_turn_ended": True,
            "kv_cache_length": 142,
            "pending_token_count": 0,
            "prefetched": False,
            "needs_finalize": False,
            "tts_context_present": False,
            "token2wav_prepared": True,
            "stale_output_discard_count": 0,
            "cancel_requested_at_ns": 10,
            "cancel_returned_at_ns": 20,
            "generation_exited_at_ns": 15,
        }
        observer.handle_event(
            {"type": "session.state", "reason": "interrupt", "metrics": trace}
        )
        self.assertTrue(turn.saw_interrupt)
        self.assertEqual(observer.interrupt_traces, [trace])
        self.assertEqual(observer.all_metrics, [trace])

        observer.handle_event(
            {"type": "session.state", "reason": "reset"}
        )
        self.assertTrue(observer.session_reset.is_set())

    def test_unrequested_text_and_audio_are_recorded(self) -> None:
        observer = PROBE.EventObserver()
        observer.handle_event(event("listen", reason="listen"))
        self.assertEqual(observer.unsolicited_outputs, [])
        audio = PROBE.encode_float32_pcm([0.0])
        observer.handle_event(event("text", text="proactive"))
        observer.handle_event(
            event(
                "audio",
                audio=base64.b64encode(audio).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
            )
        )
        self.assertEqual(len(observer.unsolicited_outputs), 2)
        self.assertEqual(observer.unsolicited_outputs[1]["audio_bytes"], len(audio))

    def test_silent_audio_is_validated_but_not_recorded_or_flagged(self) -> None:
        observer = PROBE.EventObserver()
        silent = PROBE.encode_float32_pcm([0.0, 0.0])
        delta = event(
            "audio",
            silent=True,
            audio=base64.b64encode(silent).decode("ascii"),
            format="pcm_f32",
            sample_rate=24_000,
            input_id="silent_input",
        )
        observer.handle_event(delta)
        self.assertEqual(observer.unsolicited_outputs, [])
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.handle_event(delta)
        self.assertEqual(turn.audio, b"")
        self.assertIsNone(turn.first_audio_at)
        self.assertFalse(turn.saw_audio)
        self.assertFalse(turn.audio_started.is_set())
        self.assertIn("silent_input", observer._acknowledged_input_ids)

    def test_silent_marker_cannot_hide_nonzero_audio(self) -> None:
        observer = PROBE.EventObserver()
        observer.handle_event(
            event(
                "audio",
                silent=True,
                audio=base64.b64encode(PROBE.encode_float32_pcm([0.25])).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
            )
        )
        self.assertEqual(len(observer.unsolicited_outputs), 1)
        self.assertIn("nonzero PCM", observer.unsolicited_outputs[0]["error"])

    def test_post_silence_speech_is_not_attached_to_last_answer(self) -> None:
        observer = PROBE.EventObserver()
        observer.last_completed_turn = PROBE.TurnResult(1, "question")
        observer.handle_event(
            event(
                "audio",
                input_id="probe_post_silence_0001",
                audio=base64.b64encode(PROBE.encode_float32_pcm([0.25])).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
            )
        )
        self.assertEqual(len(observer.unsolicited_outputs), 1)
        self.assertEqual(observer.last_completed_turn.audio, b"")

    def test_delayed_startup_output_cannot_become_first_question_answer(self) -> None:
        observer = PROBE.EventObserver()
        observer.register_input("startup", phase="initial_silence", require_committed=True)
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.register_input("question", phase="turn_01")
        observer.handle_event(event("text", input_id="startup", response_id="startup_resp", text="启动抢答"))
        observer.handle_event(event(
            "audio", input_id="startup", response_id="startup_resp",
            audio=base64.b64encode(PROBE.encode_float32_pcm([0.25])).decode("ascii"),
            format="pcm_f32", sample_rate=24_000,
        ))
        # A newly submitted question can advance the same startup response.
        observer.handle_event(event(
            "text", input_id="question", response_id="startup_resp", text="继续抢答",
            metrics={"kv_cache_length": 10},
        ))
        observer.handle_event(event("listen", input_id="startup", reason="turn_end"))
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.audio, b"")
        self.assertEqual(turn.metrics, [])
        self.assertFalse(turn.done.is_set())
        self.assertEqual(len(observer.unsolicited_outputs), 3)
        self.assertTrue(all(item["phase"] == "initial_silence" for item in observer.unsolicited_outputs))

    def test_late_previous_turn_output_and_boundary_keep_their_owner(self) -> None:
        observer = PROBE.EventObserver()
        first = PROBE.TurnResult(1, "first")
        observer.set_current(first)
        observer.register_input("first_input", phase="turn_01")
        observer.handle_event(event("text", input_id="first_input", response_id="first_resp", text="第一轮"))
        observer.handle_event(event("listen", input_id="first_input", response_id="first_resp", reason="turn_end"))
        second = PROBE.TurnResult(2, "second")
        observer.set_current(second)
        observer.register_input("second_input", phase="turn_02")
        observer.handle_event(event("text", input_id="first_input", text="延迟尾音"))
        observer.handle_event(event("text", input_id="cont_0001", response_id="first_resp", text="延迟续块"))
        observer.handle_event({"type": "response.done", "input_id": "first_input"})
        self.assertEqual(first.text, "第一轮延迟尾音延迟续块")
        self.assertEqual(second.text, "")
        self.assertFalse(second.done.is_set())
        self.assertEqual(first.completion_reason, "listen_end")
        self.assertEqual(observer.unsolicited_outputs, [])
        observer.handle_event(event("text", input_id="second_input", response_id="second_resp", text="第二轮"))
        self.assertEqual(second.text, "第二轮")

    def test_post_silence_output_keeps_silence_owner_during_next_turn(self) -> None:
        observer = PROBE.EventObserver()
        observer.register_input("post", phase="post_silence")
        turn = PROBE.TurnResult(2, "next")
        observer.set_current(turn)
        observer.handle_event(event("text", input_id="post", text="无请求语音"))
        observer.handle_event({"type": "response.done", "input_id": "post"})
        self.assertEqual(turn.text, "")
        self.assertFalse(turn.done.is_set())
        self.assertEqual([item["phase"] for item in observer.unsolicited_outputs], ["post_silence", "post_silence"])

    def test_unknown_explicit_input_and_continuation_do_not_inherit_current_turn(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.handle_event(event("text", input_id="unknown", text="未知来源"))
        observer.handle_event(event("text", input_id="cont_0001", text="未知续块"))
        self.assertEqual(turn.text, "")
        self.assertEqual(len(observer.unsolicited_outputs), 2)

    def test_commit_metrics_keep_registered_input_owner(self) -> None:
        observer = PROBE.EventObserver()
        first = PROBE.TurnResult(1, "first")
        observer.set_current(first)
        acknowledgement = observer.register_input("first_input", phase="turn_01")
        second = PROBE.TurnResult(2, "second")
        observer.set_current(second)
        observer.handle_event({
            "type": "input.committed", "input_id": "first_input",
            "metrics": {"kv_cache_length": 10},
        })
        self.assertTrue(acknowledgement.is_set())
        self.assertEqual(first.kv_cache_lengths, [10])
        self.assertEqual(second.kv_cache_lengths, [])

    def test_audio_padding_is_removed_from_recorded_answer(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        payload = PROBE.encode_float32_pcm([0.0, 0.25, 0.5])
        observer.handle_event(
            event(
                "audio",
                audio=base64.b64encode(payload).decode("ascii"),
                format="pcm_f32",
                sample_rate=24_000,
                metrics={"audio_padding_samples": 1},
            )
        )
        self.assertEqual(turn.audio, PROBE.encode_float32_pcm([0.25, 0.5]))

    def test_padding_only_audio_does_not_start_speech(self) -> None:
        observer = PROBE.EventObserver()
        turn = PROBE.TurnResult(1, "question")
        observer.set_current(turn)
        observer.handle_event(
            event(
                "audio",
                audio=base64.b64encode(PROBE.encode_float32_pcm([0.0])).decode("ascii"),
                format="pcm_f32", sample_rate=24_000,
                metrics={"audio_padding_samples": 1},
            )
        )
        self.assertEqual(turn.audio, b"")
        self.assertFalse(turn.saw_audio)
        self.assertIsNone(turn.first_audio_at)
        self.assertFalse(turn.audio_started.is_set())

    def test_terminal_summary_omits_large_trace_arrays(self) -> None:
        report = {
            "mode": "normal",
            "session_id": "session-one",
            "duration_seconds": 12.5,
            "requested_turns": 1,
            "completed_turns": 1,
            "passed_turns": 1,
            "failed_turns": [],
            "failures": [],
            "receiver_error": None,
            "unsolicited_outputs": [{"text": "unexpected"}],
            "repeated_text": [],
            "input_pacing_fallbacks": [{"input_id": "one"}],
            "input_acknowledgements": [{"input_id": "one", "payload": "large"}],
            "interrupt_traces": [{"metrics": "large"}],
            "metrics_seen": 3,
            "kv_cache_lengths": [10, 30, 20],
            "reset_confirmed": False,
            "turns": [{
                "number": 1,
                "text": "answer",
                "error": None,
                "completion_reason": "listen_end",
                "first_audio_latency_seconds": 2.5,
                "output_audio_seconds": 1.5,
                "model_generation_rtf": 1.2,
                "metrics": [{"large": True}],
                "events": [{"large": True}],
            }],
        }

        compact = PROBE.compact_terminal_summary(Path("/tmp/report"), report)

        self.assertEqual(compact["kv_cache"], {"first": 10, "peak": 30, "final": 20})
        self.assertEqual(compact["unsolicited_output_count"], 1)
        self.assertEqual(compact["input_pacing_fallback_count"], 1)
        self.assertNotIn("input_acknowledgements", compact)
        self.assertNotIn("interrupt_traces", compact)
        self.assertNotIn("metrics", compact["turns"][0])
        self.assertNotIn("events", compact["turns"][0])
        self.assertEqual(compact["turns"][0]["first_audio_latency_seconds"], 2.5)

    def test_turn_summary_reports_latency_audio_duration_and_deduplicated_rtf(self) -> None:
        turn = PROBE.TurnResult(1, "question", started_at=10.0)
        turn.first_text_at = 11.0
        turn.first_audio_at = 12.0
        turn.completed_at = 14.0
        turn.audio_parts.append(b"\0" * (PROBE.OUTPUT_RATE * PROBE.INPUT_SAMPLE_BYTES * 2))
        metrics = {"generation_started_at_ns": 1_000, "generation_exited_at_ns": 1_000_001_000}
        turn.metrics.extend([metrics, dict(metrics)])

        summary = turn.summary()

        self.assertEqual(summary["first_text_latency_seconds"], 1.0)
        self.assertEqual(summary["first_audio_latency_seconds"], 2.0)
        self.assertEqual(summary["output_audio_seconds"], 2.0)
        self.assertEqual(summary["model_generation_seconds"], 1.0)
        self.assertEqual(summary["model_generation_rtf"], 0.5)
        self.assertEqual(summary["turn_wall_rtf"], 2.0)


class ProbeDecisionTests(unittest.TestCase):
    @staticmethod
    def _turn(number: int, text: str = "答案", *, reason: str = "listen_end") -> object:
        turn = PROBE.TurnResult(number, "question")
        turn.add_text(text)
        turn.audio_parts.append(PROBE.encode_float32_pcm([0.0]))
        turn.saw_audio = True
        turn.saw_listen = True
        turn.saw_end_boundary = True
        turn.completion_reason = reason
        return turn

    def test_story_rejects_single_acknowledgement(self) -> None:
        observer = PROBE.EventObserver()
        turn = self._turn(1, "好的")
        failures = PROBE.validate_report("story", ["q1", "q2"], [turn], observer)
        self.assertTrue(any("acknowledgement" in failure for failure in failures))
        self.assertTrue(any("follow-up" in failure for failure in failures))

    def test_story_accepts_multiple_context_turns(self) -> None:
        observer = PROBE.EventObserver()
        turns = [self._turn(1, "红球在窗台"), self._turn(2, "红色，在窗台")]
        self.assertEqual(PROBE.validate_report("story", ["q1", "q2"], turns, observer), [])

    def test_report_rejects_cross_chunk_tandem_repeat(self) -> None:
        observer = PROBE.EventObserver()
        turn = self._turn(1, "")
        turn.add_text("七加八等于十")
        turn.add_text("五。七加八")
        turn.add_text("等于十五。")
        failures = PROBE.validate_report("normal", ["q1"], [turn], observer)
        self.assertTrue(any("repeated response text" in failure for failure in failures))

    def test_barge_in_requires_first_audio_onset_and_three_turn_boundaries(self) -> None:
        observer = PROBE.EventObserver()
        first = self._turn(1, "一二三", reason="barge_in")
        second = self._turn(2, "十五")
        third = self._turn(3, "五")
        observer.interrupt_traces.append(
            {
                "epoch": 1,
                "unit_index": 0,
                "current_turn_ended": True,
                "kv_cache_length": 8,
                "pending_token_count": 0,
                "prefetched": False,
                "needs_finalize": False,
                "tts_context_present": False,
                "token2wav_prepared": True,
                "stale_output_discard_count": 1,
                "cancel_requested_at_ns": 10,
                "cancel_returned_at_ns": 20,
                "generation_exited_at_ns": 15,
            }
        )
        self.assertEqual(
            PROBE.validate_report(
                "barge-in", ["q1", "q2", "q3"], [first, second, third], observer
            ),
            [],
        )
        first.completion_reason = "listen_end"
        failures = PROBE.validate_report(
            "barge-in", ["q1", "q2", "q3"], [first, second, third], observer
        )
        self.assertTrue(any("first barge-in" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
