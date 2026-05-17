import unittest

from mikazuki.anima_backend.adapter import adapt_anima_config


class AnimaBackendAdapterTests(unittest.TestCase):
    def test_adapter_keeps_supported_anima_fields(self):
        config = {
            "model_train_type": "anima-lora",
            "pretrained_model_name_or_path": "./sd-models/anima/anima-preview3-base.safetensors",
            "vae": "./sd-models/anima/qwen_image_vae.safetensors",
            "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
            "network_module": "networks.lora_anima",
            "network_dim": 16,
            "network_alpha": 16,
            "enable_preview": True,
            "sample_width": 1024,
            "sample_height": 1024,
        }

        adapted, warnings = adapt_anima_config(config)

        self.assertTrue(adapted["pretrained_model_name_or_path"].endswith("anima-preview3-base.safetensors"))
        self.assertTrue(adapted["vae"].endswith("qwen_image_vae.safetensors"))
        self.assertTrue(adapted["qwen3"].endswith("qwen_3_06b_base.safetensors"))
        self.assertEqual(adapted["network_module"], "networks.lora_anima")
        self.assertEqual(adapted["network_dim"], 16)
        self.assertNotIn("model_train_type", adapted)
        self.assertNotIn("enable_preview", adapted)
        self.assertEqual(warnings, [])

    def test_adapter_warns_for_unsupported_debug_fields(self):
        adapted, warnings = adapt_anima_config(
            {
                "pretrained_model_name_or_path": "model.safetensors",
                "anima_debug_mode": True,
                "anima_rope_mismatch_mode": "resample",
            }
        )

        self.assertEqual(adapted["pretrained_model_name_or_path"], "model.safetensors")
        self.assertNotIn("anima_debug_mode", adapted)
        self.assertNotIn("anima_rope_mismatch_mode", adapted)
        self.assertIn("Unsupported Anima field ignored: anima_debug_mode", warnings)
        self.assertIn("Unsupported Anima field ignored: anima_rope_mismatch_mode", warnings)

    def test_network_args_custom_becomes_network_args(self):
        adapted, warnings = adapt_anima_config(
            {
                "network_args_custom": ["train_llm_adapter=True", "verbose=True"],
            }
        )

        self.assertEqual(adapted["network_args"], ["train_llm_adapter=True", "verbose=True"])
        self.assertNotIn("network_args_custom", adapted)
        self.assertEqual(warnings, [])

    def test_adapter_warns_when_unknown_field_is_passed_through(self):
        adapted, warnings = adapt_anima_config({"future_sd_scripts_option": "enabled"})

        self.assertEqual(adapted["future_sd_scripts_option"], "enabled")
        self.assertIn("Unknown field passed through to sd-scripts: future_sd_scripts_option", warnings)


if __name__ == "__main__":
    unittest.main()
