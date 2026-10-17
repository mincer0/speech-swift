"""Actual Swift/HF PCM parity without MLX, SwiftPM, or model weights.

Run in the pinned PyTorch environment, with MINICPM_AUDIO_SOURCE pointing to
the original model config/code directory when it is not at the local default.
Only the frontend config is loaded. The two original audio classes are taken
unchanged from their AST to avoid importing unrelated vision dependencies.
"""

from __future__ import annotations

import ast
import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

import numpy as np
import torch
from transformers.audio_utils import spectrogram, window_function
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor


PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get(
    "MINICPM_AUDIO_SOURCE",
    str(PACKAGE.parents[1] / "models" / "MiniCPM-o-4_5"),
))


def original_audio_classes() -> types.SimpleNamespace:
    tree = ast.parse((SOURCE / "processing_minicpmo.py").read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"MiniCPMAAudioProcessor", "StreamingMelProcessorExact"}]
    if len(classes) != 2:
        raise RuntimeError("original audio processor classes are missing")
    namespace = {
        "copy": copy, "np": np, "torch": torch,
        "WhisperFeatureExtractor": WhisperFeatureExtractor,
        "spectrogram": spectrogram, "window_function": window_function,
    }
    exec("from __future__ import annotations\n" + "\n".join(ast.unparse(c) for c in classes),
         namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in (
        "MiniCPMAAudioProcessor", "StreamingMelProcessorExact")})


def original_frontend(processing):
    return processing.StreamingMelProcessorExact(
        processing.MiniCPMAAudioProcessor.from_pretrained(SOURCE),
        chunk_ms=1000, first_chunk_ms=1035, cnn_redundancy_ms=20,
        enable_sliding_window=True, slide_trigger_seconds=30.0, slide_stride_seconds=10.0,
    )


class SwiftProbe:
    def __init__(self, executable: Path):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )

    def request(self, **payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError(self.process.stderr.read())
        result = json.loads(line)
        if not result.get("ok"):
            raise AssertionError(result)
        return result

    def accept(self, samples, *, final=False):
        return self.request(samplesBase64=base64.b64encode(
            np.asarray(samples, dtype="<f4").tobytes()).decode(), isFinal=final)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()
        self.process.stderr.close()


class AudioStreamingOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir():
            raise unittest.SkipTest("set MINICPM_AUDIO_SOURCE to the original audio source/config")
        if not shutil.which("swiftc"):
            raise unittest.SkipTest("standalone Swift DSP verification requires swiftc on macOS")
        cls.processing = original_audio_classes()
        cls.temporary = tempfile.TemporaryDirectory(prefix="minicpm-audio-dsp-")
        cls.executable = Path(cls.temporary.name) / "frontend"
        subprocess.run([
            "swiftc", "-parse-as-library",
            str(PACKAGE / "Sources/MiniCPMAudio/Configuration.swift"),
            str(PACKAGE / "Sources/MiniCPMAudio/WhisperFeatureExtractor.swift"),
            str(PACKAGE / "scripts/tests/audio_frontend_probe.swift"),
            "-o", str(cls.executable),
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.probe = SwiftProbe(self.executable)

    def tearDown(self):
        self.probe.close()

    def assert_features(self, response, expected):
        values = np.frombuffer(base64.b64decode(response["featuresBase64"]), dtype="<f4")
        actual = values.reshape(response["frames"], 80)
        oracle = expected.squeeze(0).transpose(0, 1).numpy()
        self.assertEqual(actual.shape, oracle.shape)
        error = np.abs(actual - oracle)
        self.assertLess(float(error.max()), 0.002)
        self.assertLess(float(error.mean()), 0.00005)
        return actual

    def test_actual_pcm_context_matches_original_frontend(self):
        frontend = original_frontend(self.processing)
        offset = 0
        for index, size in enumerate((16480, 16000, 16000)):
            samples = np.sin(np.arange(offset, offset + size, dtype=np.float32) * 0.017) * 0.2
            expected, _ = frontend.process(samples)
            actual = self.probe.accept(samples)
            self.assert_features(actual, expected)
            self.assertEqual(actual["frames"], 102 if index == 0 else 104)
            self.assertEqual(actual["encoderPositions"], 50)
            self.assertEqual(actual["audioTokens"], 10)
            self.assertEqual(actual["prefixExtraFrames"], 0 if index == 0 else 2)
            self.assertEqual(actual["suffixExtraFrames"], 2)
            offset += size

    def test_packet_boundary_flush_and_snapshot_restore(self):
        first = np.sin(np.arange(16480, dtype=np.float32) * 0.017) * 0.2
        self.assertNotIn("frames", self.probe.accept(first[:1000]))
        checkpoint = self.probe.request(op="snapshot", id="partial")
        self.assertNotIn("frames", self.probe.accept(first[1000:-1]))
        self.assertEqual(self.probe.accept(first[-1:])["frames"], 102)
        self.probe.request(op="restore", id="partial")
        self.assertEqual(self.probe.accept(first[1000:])["frames"], 102)
        self.assertEqual(checkpoint["nextChunkSamples"], 15480)
        self.probe.request(op="snapshot", id="before-flush")
        tail = np.cos(np.arange(4321, dtype=np.float32) * 0.021) * 0.1
        flushed = self.probe.accept(tail, final=True)
        self.assertEqual(flushed["prefixExtraFrames"], 2)
        self.assertEqual(flushed["suffixExtraFrames"], 0)
        extractor = self.processing.MiniCPMAAudioProcessor.from_pretrained(SOURCE)
        extractor.set_spac_log_norm(log_floor_db=-10)
        full = extractor(np.concatenate([first, tail]), sampling_rate=16000,
                         return_tensors="pt", padding=False).input_features
        self.assert_features(flushed, full[:, :, 98:])
        self.probe.request(op="restore", id="before-flush")
        self.assertEqual(self.probe.accept(tail, final=True), flushed)
        # A checkpoint after the reflected final frame is valid too.
        self.probe.request(op="snapshot", id="after-flush")
        self.probe.request(op="reset")
        self.probe.request(op="restore", id="after-flush")
        self.assertNotIn("frames", self.probe.accept([], final=True))

    def test_sliding_normalization_drops_old_loud_peak_and_replays(self):
        frontend = original_frontend(self.processing)
        offset = 0
        for index in range(31):
            size = 16480 if index == 0 else 16000
            samples = np.sin(np.arange(offset, offset + size, dtype=np.float32) * 0.017) * 1e-5
            if index == 0:
                samples[:16000] *= 50000
            if index == 29:
                self.probe.request(op="snapshot", id="before-slide")
            expected, info = frontend.process(samples)
            actual = self.probe.accept(samples)
            values = self.assert_features(actual, expected)
            self.assertEqual(actual["bufferStartSample"], info["left_samples_dropped"])
            self.assertLess(actual["bufferedSamples"], 480000)
            if index == 29:
                self.assertEqual(actual["bufferStartSample"], 160000)
                self.assertLess(float(values.min()), -1.49)
                self.probe.request(op="restore", id="before-slide")
                self.assertEqual(self.probe.accept(samples), actual)
            offset += size

    def test_golden_exporter_uses_real_official_pcm_context(self):
        path = PACKAGE / "scripts/export_minicpm_o_audio_golden.py"
        spec = importlib.util.spec_from_file_location("audio_golden_exporter", path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        waveform = np.sin(np.arange(48480, dtype=np.float32) * 0.017) * 0.2
        chunks, pcm, info = exporter.official_duplex_feature_chunks(
            self.processing, SOURCE, waveform)
        self.assertEqual([chunk.shape[-1] for chunk in chunks], [102, 104, 104])
        self.assertEqual([len(chunk) for chunk in pcm], [16480, 16000, 16000])
        self.assertEqual([item["core_start"] for item in info], [0, 100, 200])
        for samples, expected in zip(pcm, chunks):
            self.assert_features(self.probe.accept(samples), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
