import sys
import unittest
from pathlib import Path
from unittest import mock


VENDOR_SD_SCRIPTS = Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"
if str(VENDOR_SD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(VENDOR_SD_SCRIPTS))

from library import train_util  # noqa: E402


class AccelerateResumeFallbackTests(unittest.TestCase):
    def test_infers_step_from_step_state_dir(self):
        self.assertEqual(
            train_util._infer_step_from_state_dir("output/model-step00000800-state"),
            800,
        )

    def test_non_step_state_dir_defaults_to_zero(self):
        self.assertEqual(train_util._infer_step_from_state_dir("output/model-000001-state"), 0)

    def test_missing_step_key_fallback_sets_accelerator_step(self):
        class FakeAccelerator:
            step = -1

            def load_state(self, _state_dir):
                raise KeyError("step")

        accelerator = FakeAccelerator()

        with mock.patch.object(train_util, "_is_accelerate_missing_step_error", return_value=True):
            train_util._load_accelerate_state_with_step_fallback(
                accelerator,
                "output/model-step00000800-state",
            )

        self.assertEqual(accelerator.step, 800)

    def test_non_accelerate_key_error_is_not_swallowed(self):
        class FakeAccelerator:
            def load_state(self, _state_dir):
                raise KeyError("step")

        with mock.patch.object(train_util, "_is_accelerate_missing_step_error", return_value=False):
            with self.assertRaises(KeyError):
                train_util._load_accelerate_state_with_step_fallback(
                    FakeAccelerator(),
                    "output/model-step00000800-state",
                )


if __name__ == "__main__":
    unittest.main()
