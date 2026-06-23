from __future__ import annotations

import unittest
from unittest import mock


class AgentTrainPlannerTests(unittest.TestCase):
    def test_rule_planner_selects_skill_and_slots(self):
        from mikazuki.agent_train.planner import plan_user_message

        decision = plan_user_message(
            "使用 /data/train 做 Anima Fast base=/models/anima.safetensors",
            model_config={},
            existing_slots={},
            workflow={},
        )

        self.assertEqual(decision.skill_id, "anima_fast_train")
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.slots["path"], "/data/train")
        self.assertEqual(decision.slots["pretrained_model_name_or_path"], "/models/anima.safetensors")
        self.assertGreaterEqual(decision.confidence, 0.5)

    def test_llm_planner_accepts_valid_json_decision(self):
        from mikazuki.agent_train.planner import parse_planner_json

        decision = parse_planner_json(
            """
            {"skill_id":"tagger_leaf_batch_caption","slots":{"path":"/data/images"},"payload_patch":{"use_vlm":false},"confidence":0.92,"reason":"用户要求打标","questions":[]}
            """,
            fallback_reason="unused",
        )

        self.assertEqual(decision.skill_id, "tagger_leaf_batch_caption")
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.slots["path"], "/data/images")
        self.assertEqual(decision.payload_patch["use_vlm"], False)
        self.assertEqual(decision.reason, "用户要求打标")

    def test_llm_planner_ignores_non_object_payload_patch(self):
        from mikazuki.agent_train.planner import parse_planner_json

        decision = parse_planner_json(
            '{"skill_id":"tagger_leaf_batch_caption","slots":{},"payload_patch":["bad"],"confidence":0.9,"reason":"x"}',
            fallback_reason="unused",
        )

        self.assertEqual(decision.payload_patch, {})

    def test_llm_planner_rejects_invalid_json(self):
        from mikazuki.agent_train.planner import parse_planner_json

        decision = parse_planner_json("不是 JSON", fallback_reason="invalid planner json")

        self.assertIsNone(decision.skill_id)
        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.fallback_reason, "invalid planner json")

    def test_llm_planner_rejects_unknown_skill(self):
        from mikazuki.agent_train.planner import parse_planner_json

        decision = parse_planner_json(
            '{"skill_id":"unknown_skill","slots":{},"confidence":0.9,"reason":"x"}',
            fallback_reason="unknown skill",
        )

        self.assertIsNone(decision.skill_id)
        self.assertEqual(decision.source, "fallback")
        self.assertIn("unknown", decision.fallback_reason)

    def test_plan_user_message_falls_back_when_llm_fails(self):
        from mikazuki.agent_train import planner

        with mock.patch.object(planner, "_plan_with_llm", return_value=None):
            decision = planner.plan_user_message(
                "请给 /data/images 打标",
                model_config={"base_url": "http://127.0.0.1:8000/v1", "model": "local"},
                existing_slots={},
                workflow={},
            )

        self.assertEqual(decision.skill_id, "tagger_leaf_batch_caption")
        self.assertEqual(decision.source, "rule")
        self.assertEqual(decision.fallback_reason, "llm planner unavailable")

    def test_rule_planner_extracts_common_tagger_payload_patch(self):
        from mikazuki.agent_train import planner

        with mock.patch.object(planner, "_plan_with_llm", return_value=None):
            decision = planner.plan_user_message(
                "给/root/autodl-tmp/lora-scripts-next-leaf/dataset/test打标，触发词为kisaragi_yuuna，不进行vllm打标",
                model_config={"base_url": "https://api.example.com/v1", "model": "model"},
                existing_slots={},
                workflow={},
            )

        self.assertEqual(decision.skill_id, "tagger_leaf_batch_caption")
        self.assertEqual(decision.slots["path"], "/root/autodl-tmp/lora-scripts-next-leaf/dataset/test")
        self.assertEqual(decision.payload_patch["additional_tags"], "kisaragi_yuuna")
        self.assertEqual(decision.payload_patch["use_vlm"], False)


if __name__ == "__main__":
    unittest.main()
