from __future__ import annotations

import unittest


class AgentTrainSlotTests(unittest.TestCase):
    def test_tagger_with_path_can_execute(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result("tagger_leaf_batch_caption", "请给 /data/images 打标", {})

        self.assertEqual(result["slots"]["path"], "/data/images")
        self.assertEqual(result["missing_slots"], [])
        self.assertTrue(result["can_execute"])

    def test_tagger_path_strips_trailing_chinese_action_word(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result(
            "tagger_leaf_batch_caption",
            "给/root/autodl-tmp/lora-scripts-next-leaf/dataset/test打标",
            {},
        )

        self.assertEqual(result["slots"]["path"], "/root/autodl-tmp/lora-scripts-next-leaf/dataset/test")

    def test_tagger_without_path_asks_for_path(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result("tagger_leaf_batch_caption", "帮我打标", {})

        self.assertFalse(result["can_execute"])
        self.assertEqual(result["missing_slots"], ["path"])
        self.assertIn("图片目录", result["assistant_message"])

    def test_differential_requires_folder_b(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result("differential_lora_train", "对 /data/a 做 differential 差分训练", {})

        self.assertEqual(result["slots"]["folder_a"], "/data/a")
        self.assertIn("folder_b", result["missing_slots"])
        self.assertFalse(result["can_execute"])

    def test_followup_updates_existing_differential_slots(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result(
            "differential_lora_train",
            "folder_b 是 /data/b",
            {"folder_a": "/data/a"},
        )

        self.assertEqual(result["slots"]["folder_a"], "/data/a")
        self.assertEqual(result["slots"]["folder_b"], "/data/b")
        self.assertNotIn("folder_b", result["missing_slots"])

    def test_training_requires_model_paths(self):
        from mikazuki.agent_train.slots import build_slot_result

        result = build_slot_result("anima_fast_train", "使用 /data/train 做 Anima Fast", {})

        self.assertIn("pretrained_model_name_or_path", result["missing_slots"])
        self.assertIn("vae", result["missing_slots"])
        self.assertIn("qwen3", result["missing_slots"])
        self.assertFalse(result["can_execute"])

    def test_payload_merge_uses_slots(self):
        from mikazuki.agent_train.slots import apply_slots_to_payload

        payload = {
            "train_data_dir": "",
            "source_image_dir": "",
            "pretrained_model_name_or_path": "",
            "vae": "",
            "qwen3": "",
        }
        slots = {
            "path": "/data/train",
            "pretrained_model_name_or_path": "/models/anima.safetensors",
            "vae": "/models/vae.safetensors",
            "qwen3": "/models/qwen3.safetensors",
        }

        merged = apply_slots_to_payload("anima_fast_train", payload, slots)

        self.assertEqual(merged["train_data_dir"], "/data/train")
        self.assertEqual(merged["source_image_dir"], "/data/train")
        self.assertEqual(merged["pretrained_model_name_or_path"], "/models/anima.safetensors")
        self.assertEqual(merged["vae"], "/models/vae.safetensors")
        self.assertEqual(merged["qwen3"], "/models/qwen3.safetensors")


if __name__ == "__main__":
    unittest.main()
