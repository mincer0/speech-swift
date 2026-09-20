#!/usr/bin/env python3
"""Startup and compatibility regressions for the CAM++ CoreML converter.

These tests intentionally never import a real ML stack or load an ONNX/CoreML
model.  The subprocess check catches accidental top-level imports, while the
small fakes exercise the compatibility declarations in isolation.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "convert_minicpm_campplus_coreml.py"
SPEC = importlib.util.spec_from_file_location("convert_minicpm_campplus_coreml", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"unable to import converter from {SCRIPT_PATH}")
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class CampPlusConverterStartupTests(unittest.TestCase):
    """Keep --help and dependency setup independent from model conversion."""

    def test_help_subprocess_succeeds_without_heavy_imports(self) -> None:
        """A clean interpreter can render help even when ML packages are absent."""

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--deployment-target", result.stdout)

    def test_help_subprocess_import_guard_blocks_every_heavy_dependency(self) -> None:
        """The full entrypoint must not import any conversion dependency for help."""

        guard = r'''
import builtins
import runpy
import sys

blocked = {"coremltools", "numpy", "torch", "torchvision", "onnx2torch"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("heavy dependency imported during --help: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    raise SystemExit(exc.code)
'''
        result = subprocess.run(
            [sys.executable, "-c", guard, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("campplus.onnx source graph", result.stdout)

    def test_parse_args_accepts_explicit_argv_without_importing_ml_stack(self) -> None:
        with mock.patch.object(
            builtins,
            "__import__",
            side_effect=self._reject_heavy_imports(builtins.__import__),
        ):
            parsed = CONVERTER.parse_args(["source.onnx", "bundle.mlpackage", "--overwrite"])
        self.assertEqual(parsed.onnx, Path("source.onnx"))
        self.assertEqual(parsed.output, Path("bundle.mlpackage"))
        self.assertTrue(parsed.overwrite)

    def test_help_parser_exits_cleanly_without_importing_ml_stack(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            builtins,
            "__import__",
            side_effect=self._reject_heavy_imports(builtins.__import__),
        ):
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    CONVERTER.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--source-model", output.getvalue())

    def test_protobuf_compatibility_defaults_but_respects_user_value(self) -> None:
        variable = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(variable, None)
            CONVERTER._configure_protobuf_compatibility()
            self.assertEqual(os.environ[variable], "python")

        with mock.patch.dict(os.environ, {variable: "cpp"}, clear=False):
            CONVERTER._configure_protobuf_compatibility()
            self.assertEqual(os.environ[variable], "cpp")

    def test_static_pool_padding_matches_fixed_camplus_tail_window(self) -> None:
        self.assertEqual(
            CONVERTER._pool_right_padding(
                CONVERTER.CAMPPLUS_INTERNAL_FRAMES,
                kernel_size=CONVERTER.CAMPPLUS_POOL_KERNEL,
                stride=CONVERTER.CAMPPLUS_POOL_STRIDE,
            ),
            50,
        )
        self.assertEqual(
            CONVERTER._pool_right_padding(
                200,
                kernel_size=CONVERTER.CAMPPLUS_POOL_KERNEL,
                stride=CONVERTER.CAMPPLUS_POOL_STRIDE,
            ),
            0,
        )

    def test_static_pool_padding_rejects_non_stride_pool(self) -> None:
        with self.assertRaises(ValueError):
            CONVERTER._pool_right_padding(250, kernel_size=80, stride=100)

    def test_torchvision_compat_defines_nms_and_qnms_and_keeps_library(self) -> None:
        class FakeLibrary:
            instances = []

            def __init__(self, namespace: str, kind: str) -> None:
                self.namespace = namespace
                self.kind = kind
                self.schemas = []
                self.instances.append(self)

            def define(self, schema: str) -> None:
                self.schemas.append(schema)

        fake_torch = SimpleNamespace(library=SimpleNamespace(Library=FakeLibrary))
        library = CONVERTER._register_torchvision_compat(fake_torch)

        self.assertIs(library, FakeLibrary.instances[-1])
        self.assertIs(CONVERTER._TORCHVISION_COMPAT_LIBRARY, library)
        self.assertEqual(library.namespace, "torchvision")
        self.assertEqual(library.kind, "DEF")
        self.assertEqual(
            library.schemas,
            [
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
                "qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
            ],
        )

    def test_path_validation_finishes_before_dependency_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.onnx"
            output = missing.parent / "campplus.mlpackage"
            with mock.patch.object(
                CONVERTER,
                "_load_conversion_dependencies",
                side_effect=AssertionError("dependency loader ran too early"),
            ):
                with self.assertRaises(FileNotFoundError):
                    CONVERTER.main([str(missing), str(output)])

    @staticmethod
    def _reject_heavy_imports(real_import):
        blocked = {"coremltools", "numpy", "torch", "torchvision", "onnx2torch"}

        def guarded(name, *args, **kwargs):
            if name.split(".", 1)[0] in blocked:
                raise AssertionError(f"heavy dependency imported: {name}")
            return real_import(name, *args, **kwargs)

        return guarded


if __name__ == "__main__":
    unittest.main()
