from __future__ import annotations

import unittest


class AgentTrainSkillRegistryTests(unittest.TestCase):
    def test_builtin_skills_have_unique_ids_and_side_effect_levels(self):
        from mikazuki.agent_train.skills import BUILTIN_SKILLS, get_skill

        ids = [skill.id for skill in BUILTIN_SKILLS]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids),
            {
                "tagger_leaf_batch_caption",
                "differential_lora_train",
                "anima_lora_train",
                "anima_fast_train",
            },
        )
        for skill in BUILTIN_SKILLS:
            self.assertIn(skill.side_effect_level, {"read_only", "writes_files", "gpu_training"})
            self.assertIs(get_skill(skill.id), skill)

    def test_skill_action_payload_includes_skill_metadata(self):
        from mikazuki.agent_train.skills import make_skill_plan

        plan, action = make_skill_plan("tagger_leaf_batch_caption", "/data/images")

        self.assertEqual(plan["skill_id"], "tagger_leaf_batch_caption")
        self.assertEqual(plan["side_effect_level"], "writes_files")
        self.assertEqual(action["skill_id"], "tagger_leaf_batch_caption")
        self.assertEqual(action["kind"], "run_tagger")
        self.assertEqual(action["payload"]["input_dir"], "/data/images")

    def test_anima_lora_payload_uses_full_schema_defaults(self):
        from mikazuki.agent_train.skills import make_skill_plan

        _, action = make_skill_plan("anima_lora_train", "/data/train")
        payload = action["payload"]

        self.assertGreaterEqual(len(payload), 45)
        self.assertEqual(payload["model_train_type"], "anima-lora")
        self.assertEqual(payload["train_data_dir"], "/data/train")
        self.assertEqual(payload["pretrained_model_name_or_path"], "./sd-models/anima/anima-base-v1.0.safetensors")
        self.assertEqual(payload["vae"], "./sd-models/anima/qwen_image_vae.safetensors")
        self.assertEqual(payload["qwen3"], "./sd-models/anima/qwen_3_06b_base.safetensors")
        self.assertEqual(payload["resolution"], "1024,1024")
        self.assertEqual(payload["max_train_epochs"], 10)
        self.assertEqual(payload["network_module"], "networks.lora_anima")
        self.assertEqual(payload["network_alpha"], 16)
        self.assertEqual(payload["optimizer_type"], "AdamW8bit")
        self.assertEqual(payload["logging_dir"], "./logs")
        self.assertEqual(payload["caption_extension"], ".txt")

    def test_anima_fast_payload_uses_full_schema_defaults(self):
        from mikazuki.agent_train.skills import make_skill_plan

        _, action = make_skill_plan("anima_fast_train", "/data/train")
        payload = action["payload"]

        self.assertGreaterEqual(len(payload), 40)
        self.assertEqual(payload["model_train_type"], "anima-lora-fast")
        self.assertEqual(payload["train_data_dir"], "/data/train")
        self.assertEqual(payload["source_image_dir"], "/data/train")
        self.assertEqual(payload["max_train_epochs"], 1)
        self.assertEqual(payload["torch_compile"], True)
        self.assertEqual(payload["compile_mode"], "blocks")
        self.assertEqual(payload["logging_dir"], "./logs/anima_fast")
        self.assertEqual(payload["network_module"], "networks.lora_anima")

    def test_differential_payload_uses_full_schema_defaults(self):
        from mikazuki.agent_train.skills import make_skill_plan

        _, action = make_skill_plan("differential_lora_train", "/data/a")
        payload = action["payload"]

        self.assertGreaterEqual(len(payload), 35)
        self.assertEqual(payload["model_train_type"], "differential-lora")
        self.assertEqual(payload["folder_a"], "/data/a")
        self.assertEqual(payload["tag_dir"], "/data/a")
        self.assertEqual(payload["lora_rank"], 32)
        self.assertEqual(payload["dataset_repeat"], 1000)
        self.assertEqual(payload["postprocess_comfyui"], True)
        self.assertEqual(payload["postprocess_svd"], True)

    def test_match_skill_uses_keywords(self):
        from mikazuki.agent_train.skills import match_skill

        self.assertEqual(match_skill("请给 /data/images 打标").id, "tagger_leaf_batch_caption")
        self.assertEqual(match_skill("跑 differential 差分训练").id, "differential_lora_train")
        self.assertEqual(match_skill("准备 Anima Fast").id, "anima_fast_train")
        self.assertEqual(match_skill("训练 anima lora").id, "anima_lora_train")
        self.assertIsNone(match_skill("你好"))


if __name__ == "__main__":
    unittest.main()
