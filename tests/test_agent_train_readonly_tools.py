from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class AgentTrainReadOnlyToolTests(unittest.TestCase):
    def test_inspect_dataset_counts_images_and_captions(self):
        from mikazuki.agent_train.readonly_tools import inspect_dataset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.png").write_bytes(b"png")
            (root / "a.txt").write_text("tag", encoding="utf-8")
            (root / "b.jpg").write_bytes(b"jpg")
            (root / "note.md").write_text("x", encoding="utf-8")

            result = inspect_dataset(str(root))

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["exists"])
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["caption_count"], 1)
        self.assertEqual(result["missing_caption_count"], 1)

    def test_inspect_dataset_handles_missing_path(self):
        from mikazuki.agent_train.readonly_tools import inspect_dataset

        result = inspect_dataset("/path/not/exist")

        self.assertEqual(result["status"], "warning")
        self.assertFalse(result["exists"])

    def test_inspect_model_file_reports_presence(self):
        from mikazuki.agent_train.readonly_tools import inspect_model_file

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as fh:
            result = inspect_model_file(fh.name)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["exists"])
        self.assertTrue(result["is_file"])


if __name__ == "__main__":
    unittest.main()
