import unittest
from unittest import mock

from train_monitor import server


class TrainMonitorStatusTests(unittest.TestCase):
    def test_gui_api_failure_is_non_blocking_warning(self):
        with mock.patch.object(server, "newest_preview_images", return_value=[]), \
                mock.patch.object(server, "_training_output_dir", return_value=None), \
                mock.patch.object(server, "latest_training_config", return_value={}), \
                mock.patch.object(server, "build_model_outputs", return_value={}), \
                mock.patch.object(server, "_extract_train_params", return_value=[]), \
                mock.patch.object(server, "tensorboard_loss_scalars", return_value=[{"tag": "loss/average"}]), \
                mock.patch.object(server, "gpu_info", return_value={}), \
                mock.patch.object(server, "fetch_gui_json", side_effect=OSError("HTTP Error 404: Not Found")):
            status = server.collect_status()

        self.assertNotIn("error", status)
        self.assertIn("gui_warning", status)
        self.assertEqual(status["state"], "GUI 离线")
        self.assertEqual(status["tensorboard_loss"], [{"tag": "loss/average"}])


if __name__ == "__main__":
    unittest.main()
