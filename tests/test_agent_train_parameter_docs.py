from __future__ import annotations

import unittest


class AgentTrainParameterDocsTests(unittest.TestCase):
    def test_docs_cover_every_default_payload_field(self):
        from mikazuki.agent_train.parameter_docs import load_parameter_docs
        from mikazuki.agent_train.skills import BUILTIN_SKILLS

        for skill in BUILTIN_SKILLS:
            with self.subTest(skill_id=skill.id):
                payload = skill.default_payload("/data/images")
                docs = load_parameter_docs(skill.id)
                documented = {str(item.get("field")) for item in docs}

                self.assertEqual(set(payload), documented)
                for item in docs:
                    self.assertIn(item.get("type"), {"boolean", "integer", "number", "string", "array"})
                    self.assertTrue(str(item.get("description") or "").strip())
                    self.assertIsInstance(item.get("llm_editable"), bool)

    def test_retrieves_tagger_vlm_and_additional_tags_docs(self):
        from mikazuki.agent_train.parameter_docs import retrieve_parameter_docs

        docs = retrieve_parameter_docs(
            "tagger_leaf_batch_caption",
            "不使用 vllm/vlm 打标，触发词为 kisaragi_yuuna",
            {"use_vlm": True, "additional_tags": ""},
        )

        fields = [item["field"] for item in docs]
        self.assertIn("use_vlm", fields)
        self.assertIn("additional_tags", fields)

    def test_retrieves_training_rank_lr_epoch_and_preview_docs(self):
        from mikazuki.agent_train.parameter_docs import retrieve_parameter_docs

        docs = retrieve_parameter_docs(
            "anima_fast_train",
            "rank 32、学习率 5e-5、训练 8 轮，开启预览图",
            {"network_dim": 16, "learning_rate": "1e-4", "max_train_epochs": 1, "enable_preview": False},
        )

        fields = [item["field"] for item in docs]
        self.assertIn("network_dim", fields)
        self.assertIn("learning_rate", fields)
        self.assertIn("max_train_epochs", fields)
        self.assertIn("enable_preview", fields)

    def test_apply_payload_patch_allows_only_editable_known_typed_fields(self):
        from mikazuki.agent_train.parameter_docs import apply_payload_patch

        payload = {"use_vlm": True, "threshold": 0.35, "input_dir": "/data/images"}
        merged, report = apply_payload_patch(
            "tagger_leaf_batch_caption",
            payload,
            {"use_vlm": "false", "threshold": "bad", "input_dir": "/tmp/other", "unknown": 1},
        )

        self.assertEqual(merged["use_vlm"], False)
        self.assertEqual(merged["threshold"], 0.35)
        self.assertEqual(merged["input_dir"], "/data/images")
        self.assertEqual(report["applied"], {"use_vlm": False})
        self.assertIn("threshold", report["rejected"])
        self.assertIn("input_dir", report["rejected"])
        self.assertIn("unknown", report["rejected"])


if __name__ == "__main__":
    unittest.main()
