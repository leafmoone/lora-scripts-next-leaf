import json
import os
import re
import tempfile
import traceback
import unittest
from unittest import mock


def infer_step_from_state_dir(state_dir: str) -> int:
    """Mirror scripts/stable/library/train_util._infer_step_from_state_dir."""
    match = re.search(r"-step(\d+)-state$", os.path.basename(os.path.normpath(state_dir)))
    if match:
        return int(match.group(1))
    train_state_file = os.path.join(state_dir, "train_state.json")
    if os.path.isfile(train_state_file):
        try:
            with open(train_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "current_step" in data:
                return int(data["current_step"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0


def is_accelerate_missing_step_error(exc: KeyError) -> bool:
    if exc.args != ("step",):
        return False
    for frame in traceback.extract_tb(exc.__traceback__):
        filename = frame.filename.replace("\\", "/")
        if filename.endswith("accelerate/accelerator.py") and frame.name == "load_state":
            return True
    return False


def load_accelerate_state_with_step_fallback(accelerator, state_dir: str):
    try:
        accelerator.load_state(state_dir)
    except KeyError as exc:
        if not is_accelerate_missing_step_error(exc):
            raise
        accelerator.step = infer_step_from_state_dir(state_dir)


class AccelerateResumeFallbackTests(unittest.TestCase):
    def test_infers_step_from_step_state_dir(self):
        self.assertEqual(
            infer_step_from_state_dir("output/model-step00000800-state"),
            800,
        )

    def test_non_step_state_dir_defaults_to_zero(self):
        self.assertEqual(infer_step_from_state_dir("output/model-000001-state"), 0)

    def test_infers_step_from_train_state_json(self):
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "custom-state")
            os.makedirs(state_dir)
            with open(os.path.join(state_dir, "train_state.json"), "w", encoding="utf-8") as f:
                json.dump({"current_epoch": 10, "current_step": 3350}, f)
            self.assertEqual(infer_step_from_state_dir(state_dir), 3350)

    def test_missing_step_key_fallback_sets_accelerator_step(self):
        class FakeAccelerator:
            step = -1

            def load_state(self, _state_dir):
                raise KeyError("step")

        accelerator = FakeAccelerator()
        with mock.patch(__name__ + ".is_accelerate_missing_step_error", return_value=True):
            load_accelerate_state_with_step_fallback(accelerator, "output/model-step00000800-state")
        self.assertEqual(accelerator.step, 800)

    def test_non_accelerate_key_error_is_not_swallowed(self):
        class FakeAccelerator:
            def load_state(self, _state_dir):
                raise KeyError("other")

        with self.assertRaises(KeyError):
            load_accelerate_state_with_step_fallback(FakeAccelerator(), "output/model-step00000800-state")


if __name__ == "__main__":
    unittest.main()
