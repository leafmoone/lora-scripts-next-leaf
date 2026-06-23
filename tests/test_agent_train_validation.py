from __future__ import annotations

import unittest


class AgentTrainValidationTests(unittest.TestCase):
    def test_tagger_blocks_empty_input_dir(self):
        from mikazuki.agent_train.validation import validate_skill_payload

        result = validate_skill_payload("tagger_leaf_batch_caption", {"input_dir": ""})

        self.assertFalse(result.can_execute)
        self.assertIn("input_dir", result.errors[0])

    def test_tagger_warns_for_missing_path_without_blocking(self):
        from mikazuki.agent_train.validation import validate_skill_payload

        result = validate_skill_payload("tagger_leaf_batch_caption", {"input_dir": "/path/not/exist"})

        self.assertTrue(result.can_execute)
        self.assertTrue(any("/path/not/exist" in warning for warning in result.warnings))

    def test_anima_fast_blocks_wrong_train_type(self):
        from mikazuki.agent_train.validation import validate_skill_payload

        result = validate_skill_payload(
            "anima_fast_train",
            {
                "model_train_type": "anima-lora",
                "train_data_dir": "/data/train",
                "pretrained_model_name_or_path": "/models/anima.safetensors",
                "vae": "/models/vae.safetensors",
                "qwen3": "/models/qwen3.safetensors",
                "max_train_epochs": 1,
                "learning_rate": "1e-4",
            },
        )

        self.assertFalse(result.can_execute)
        self.assertTrue(any("model_train_type" in error for error in result.errors))

    def test_training_blocks_invalid_numeric_values(self):
        from mikazuki.agent_train.validation import validate_skill_payload

        result = validate_skill_payload(
            "anima_lora_train",
            {
                "model_train_type": "anima-lora",
                "train_data_dir": "/data/train",
                "pretrained_model_name_or_path": "/models/anima.safetensors",
                "vae": "/models/vae.safetensors",
                "qwen3": "/models/qwen3.safetensors",
                "max_train_epochs": 0,
                "network_dim": 0,
                "learning_rate": "-1",
            },
        )

        self.assertFalse(result.can_execute)
        self.assertTrue(any("max_train_epochs" in error for error in result.errors))
        self.assertTrue(any("network_dim" in error for error in result.errors))
        self.assertTrue(any("learning_rate" in error for error in result.errors))

    def test_differential_requires_pair_and_model_paths(self):
        from mikazuki.agent_train.validation import validate_skill_payload

        result = validate_skill_payload(
            "differential_lora_train",
            {
                "model_train_type": "differential-lora",
                "folder_a": "/data/a",
                "folder_b": "",
                "pretrained_model_name_or_path": "/models/anima.safetensors",
                "vae": "/models/vae.safetensors",
                "qwen3": "/models/qwen3.safetensors",
                "lora_rank": 32,
                "num_epochs": 5,
                "learning_rate": "1e-4",
            },
        )

        self.assertFalse(result.can_execute)
        self.assertTrue(any("folder_b" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
