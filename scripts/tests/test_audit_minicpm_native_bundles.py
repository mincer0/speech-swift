#!/usr/bin/env python3
"""Regression tests for ``audit_minicpm_native_bundles.py``.

The fixtures intentionally contain tiny placeholder weight files.  The audit
tool validates bundle shape, provenance and hashes without loading model
weights, so a small standard-library-only fixture keeps these tests fast and
portable.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "audit_minicpm_native_bundles.py"
SPEC = importlib.util.spec_from_file_location("audit_minicpm_native_bundles", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"unable to import auditor from {SCRIPT_PATH}")
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDITOR
SPEC.loader.exec_module(AUDITOR)


class NativeBundleAuditTests(unittest.TestCase):
    """Exercise the portable bundle checks with a minimal complete model root."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self._create_valid_fixture()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    @staticmethod
    def _identity() -> dict[str, str]:
        return {
            "source_revision": AUDITOR.MODEL_REPOSITORY_REVISION,
            "implementation_commit": AUDITOR.DEMO_IMPLEMENTATION_COMMIT,
            "upstream_commit": AUDITOR.DEMO_IMPLEMENTATION_COMMIT,
        }

    def _create_valid_fixture(self) -> None:
        """Create all five bundle kinds accepted by the auditor."""

        # LLM: the specialized audit requires a matching target hash and bits.
        llm = self.root / AUDITOR.DEFAULT_LLM_BUNDLE
        llm_weights = b"minimal llm weights\n"
        (llm / "model.safetensors").parent.mkdir(parents=True)
        (llm / "model.safetensors").write_bytes(llm_weights)
        self._write_json(llm / "config.json", {"format_version": 1})
        provenance = self._identity()
        provenance.update(
            {
                "bits": 8,
                "target_weights_sha256": hashlib.sha256(llm_weights).hexdigest(),
            }
        )
        self._write_json(llm / "provenance.json", provenance)
        self._write_json(llm / "key_manifest.json", self._identity())

        # Audio and vision use the generic manifest identity checks.
        audio = self.root / AUDITOR.DEFAULT_BUNDLES["audio"]
        (audio / "audio_encoder.safetensors").parent.mkdir(parents=True)
        (audio / "audio_encoder.safetensors").write_bytes(b"minimal audio weights\n")
        self._write_json(audio / "config.json", {"format_version": 1})
        self._write_json(audio / "manifest.json", self._identity())

        vision = self.root / AUDITOR.DEFAULT_BUNDLES["vision"]
        (vision / "model.safetensors").parent.mkdir(parents=True)
        (vision / "model.safetensors").write_bytes(b"minimal vision weights\n")
        self._write_json(vision / "config.json", {"format_version": 1, "compute_dtype": "float32"})
        vision_manifest = self._identity()
        vision_manifest["compute_dtype"] = "float32"
        self._write_json(vision / "manifest.json", vision_manifest)

        # TTS additionally validates checksums.json references.
        tts = self.root / AUDITOR.DEFAULT_BUNDLES["tts"]
        (tts / "model.safetensors").parent.mkdir(parents=True)
        (tts / "model.safetensors").write_bytes(b"minimal tts weights\n")
        self._write_json(tts / "config.json", {"format_version": 1})
        self._write_json(tts / "keys.json", {"format_version": 1})
        self._write_json(tts / "manifest.json", self._identity())
        checksums: dict[str, dict[str, Any]] = {}
        for name in ("config.json", "keys.json", "model.safetensors"):
            path = tts / name
            checksums[name] = {
                "bytes": path.stat().st_size,
                "sha256": AUDITOR.sha256_file(path),
            }
        self._write_json(
            tts / "checksums.json",
            {"algorithm": "sha256", "files": checksums},
        )

        # Token2Wav validates component identities and a speaker sidecar.
        token2wav = self.root / AUDITOR.DEFAULT_BUNDLES["token2wav"]
        token2wav.mkdir(parents=True)
        for name in (
            "flow.safetensors",
            "hift.safetensors",
            "speech_tokenizer.safetensors",
        ):
            (token2wav / name).write_bytes(f"minimal {name}\n".encode("ascii"))
        self._write_json(
            token2wav / "config.json",
            {"format_version": 1, "components": {"speaker_encoder": {}}},
        )
        components = {
            name: self._identity()
            for name in ("flow", "hift", "speech_tokenizer")
        }
        components["speaker_encoder"] = {
            **self._identity(),
            "component": "campplus",
        }
        manifest = self._identity()
        manifest["components"] = components
        self._write_json(token2wav / "manifest.json", manifest)
        (token2wav / "campplus.mlmodelc").write_bytes(b"minimal CoreML sidecar\n")

    def _audit(self) -> dict[str, Any]:
        return AUDITOR.audit_model_root(self.root)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _update_json(self, path: Path, update: Callable[[dict[str, Any]], None]) -> None:
        value = self._read_json(path)
        update(value)
        self._write_json(path, value)

    def test_complete_legal_bundles_pass_via_cli(self) -> None:
        """A complete fixture passes both the CLI return code and JSON report."""

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return_code = AUDITOR.main([str(self.root)])

        self.assertEqual(return_code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["ok"], report)
        self.assertEqual(set(report["bundles"]), {"llm", "audio", "vision", "tts", "token2wav"})
        for bundle in report["bundles"].values():
            self.assertTrue(bundle["ok"], bundle)

    def test_model_revision_error_is_reported(self) -> None:
        provenance = self.root / AUDITOR.DEFAULT_LLM_BUNDLE / "provenance.json"
        self._update_json(provenance, lambda value: value.update({"source_revision": "wrong-revision"}))

        report = self._audit()
        errors = report["bundles"]["llm"]["errors"]
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("llm.provenance.source_revision: expected model revision" in error for error in errors),
            errors,
        )

    def test_manifest_absolute_path_is_rejected(self) -> None:
        manifest = self.root / AUDITOR.DEFAULT_BUNDLES["audio"] / "manifest.json"
        leaked_path = str(self.root / "source-checkout" / "audio_encoder.safetensors")
        self._update_json(manifest, lambda value: value.update({"source": leaked_path}))

        report = self._audit()
        errors = report["bundles"]["audio"]["errors"]
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("audio.manifest.source: absolute local path is not allowed" in error for error in errors),
            errors,
        )

    def test_vision_compute_dtype_missing_is_rejected(self) -> None:
        config = self.root / AUDITOR.DEFAULT_BUNDLES["vision"] / "config.json"
        self._update_json(config, lambda value: value.pop("compute_dtype"))

        report = self._audit()
        errors = report["bundles"]["vision"]["errors"]
        self.assertFalse(report["ok"])
        self.assertIn(
            "vision.config.compute_dtype is missing; expected float32 or bfloat16",
            errors,
        )

    def test_vision_compute_dtype_invalid_is_rejected(self) -> None:
        manifest = self.root / AUDITOR.DEFAULT_BUNDLES["vision"] / "manifest.json"
        self._update_json(manifest, lambda value: value.update({"compute_dtype": "float16"}))

        report = self._audit()
        errors = report["bundles"]["vision"]["errors"]
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "vision.manifest.compute_dtype must be one of float32 or bfloat16" in error
                for error in errors
            ),
            errors,
        )

    def test_vision_compute_dtype_conflict_is_rejected(self) -> None:
        manifest = self.root / AUDITOR.DEFAULT_BUNDLES["vision"] / "manifest.json"
        self._update_json(manifest, lambda value: value.update({"compute_dtype": "bfloat16"}))

        report = self._audit()
        errors = report["bundles"]["vision"]["errors"]
        self.assertFalse(report["ok"])
        self.assertIn(
            "vision config and manifest compute_dtype conflict: config='float32', manifest='bfloat16'",
            errors,
        )

    def test_token2wav_missing_component_is_rejected(self) -> None:
        manifest = self.root / AUDITOR.DEFAULT_BUNDLES["token2wav"] / "manifest.json"

        def remove_speaker(value: dict[str, Any]) -> None:
            value["components"].pop("speaker_encoder")

        self._update_json(manifest, remove_speaker)

        report = self._audit()
        errors = report["bundles"]["token2wav"]["errors"]
        self.assertFalse(report["ok"])
        self.assertIn("token2wav manifest is missing components: speaker_encoder", errors)

    def test_file_hash_mismatch_is_rejected(self) -> None:
        model = self.root / AUDITOR.DEFAULT_BUNDLES["tts"] / "model.safetensors"
        model.write_bytes(b"tampered tts weights\n")

        report = self._audit()
        errors = report["bundles"]["tts"]["errors"]
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("model.safetensors" in error and "SHA256 mismatch" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
