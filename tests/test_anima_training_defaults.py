import unittest

from mikazuki.app.api import apply_anima_training_defaults


class AnimaTrainingDefaultsTests(unittest.TestCase):
    def test_anima_does_not_auto_enable_full_bf16(self):
        config = {
            "mixed_precision": "bf16",
            "optimizer_type": "AdamW8bit",
            "unet_lr": "5e-5",
            "attn_mode": "torch",
        }

        apply_anima_training_defaults(config, "anima-lora")

        self.assertNotIn("full_bf16", config)
        self.assertEqual(config["unet_lr"], 5e-5)

    def test_anima_disables_full_bf16_for_came(self):
        config = {
            "mixed_precision": "bf16",
            "full_bf16": True,
            "optimizer_type": "pytorch_optimizer.CAME",
            "unet_lr": "2e-5",
            "attn_mode": "torch",
        }

        apply_anima_training_defaults(config, "anima-lora")

        self.assertNotIn("full_bf16", config)
        self.assertEqual(config["unet_lr"], 2e-5)

    def test_anima_disables_full_bf16_for_automagic(self):
        config = {
            "mixed_precision": "bf16",
            "full_bf16": True,
            "optimizer_type": "Automagic",
            "unet_lr": "1e-6",
            "attn_mode": "torch",
        }

        apply_anima_training_defaults(config, "anima-lora")

        self.assertNotIn("full_bf16", config)
        self.assertEqual(config["unet_lr"], 1e-6)


if __name__ == "__main__":
    unittest.main()
