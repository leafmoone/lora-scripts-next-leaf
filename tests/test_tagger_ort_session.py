from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mikazuki.tagger.ort_session import MIN_ONNX_BYTES, resolve_onnx_providers


class TaggerOrtSessionTests(unittest.TestCase):
    def test_resolve_providers_cpu_mode(self):
        fake_ort = mock.Mock()
        fake_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]
        with mock.patch.dict(os.environ, {"MIKAZUKI_TAGGER_ORT_PROVIDERS": "cpu"}, clear=False):
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                providers = resolve_onnx_providers()
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_validate_rejects_tiny_onnx(self):
        from mikazuki.tagger.ort_session import _validate_model_file

        with tempfile.TemporaryDirectory() as td:
            tiny = Path(td) / "model.onnx"
            tiny.write_bytes(b"x" * 1024)
            with self.assertRaises(ValueError) as ctx:
                _validate_model_file(tiny)
            self.assertIn("incomplete", str(ctx.exception).lower())

    def test_min_onnx_bytes_sane(self):
        self.assertGreater(MIN_ONNX_BYTES, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
