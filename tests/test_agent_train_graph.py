from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

from mikazuki.agent_train.graph import run_agent_turn
from mikazuki.agent_train.planner import PlannerDecision


class AgentTrainGraphTests(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_tagger_request_creates_pending_action(self):
        result = run_agent_turn("请给 /data/images 打标", session_id="s1")

        self.assertEqual(result["plan"]["skill_id"], "tagger_leaf_batch_caption")
        self.assertEqual(result["pending_action"]["skill_id"], "tagger_leaf_batch_caption")
        self.assertEqual(result["pending_action"]["side_effect_level"], "writes_files")
        self.assertEqual(result["pending_action"]["kind"], "run_tagger")
        self.assertEqual(result["pending_action"]["payload"]["input_dir"], "/data/images")
        self.assertEqual(result["decision"]["source"], "rule")
        self.assertTrue(result["validation"]["can_execute"])
        self.assertIn("validation", result["plan"])
        self.assertIn("确认", result["assistant_message"])
        if find_spec("langgraph.checkpoint.sqlite"):
            self.assertTrue(Path(".cache/agent_train/checkpoints.sqlite").is_file())

    def test_tagger_without_path_collects_missing_slot(self):
        result = run_agent_turn("帮我打标", session_id="s1")

        self.assertEqual(result["plan"]["skill_id"], "tagger_leaf_batch_caption")
        self.assertIsNone(result["pending_action"])
        self.assertEqual(result["missing_slots"], ["path"])

    def test_differential_request_collects_missing_folder_b_and_models(self):
        result = run_agent_turn("对 /data/a 做 differential 差分训练", session_id="s1")

        self.assertEqual(result["plan"]["type"], "differential_lora")
        self.assertEqual(result["plan"]["skill_id"], "differential_lora_train")
        self.assertIsNone(result["pending_action"])
        self.assertIn("folder_b", result["missing_slots"])
        self.assertIn("pretrained_model_name_or_path", result["missing_slots"])

    def test_anima_fast_request_collects_model_slots(self):
        result = run_agent_turn("使用 /data/train 做 Anima Fast", session_id="s1")

        self.assertEqual(result["plan"]["type"], "anima_fast")
        self.assertEqual(result["plan"]["skill_id"], "anima_fast_train")
        self.assertIsNone(result["pending_action"])
        self.assertIn("pretrained_model_name_or_path", result["missing_slots"])
        self.assertIn("can_execute", result["validation"])

    def test_anima_fast_request_creates_action_when_slots_are_complete(self):
        result = run_agent_turn(
            "使用 /data/train 做 Anima Fast base=/models/anima.safetensors vae=/models/vae.safetensors qwen3=/models/qwen3.safetensors",
            session_id="s1",
        )

        self.assertEqual(result["pending_action"]["kind"], "run_anima_fast_train")
        self.assertEqual(result["pending_action"]["payload"]["pretrained_model_name_or_path"], "/models/anima.safetensors")
        self.assertTrue(result["pending_action"]["validation"]["can_execute"])

    def test_llm_payload_patch_is_applied_after_slots(self):
        with mock.patch(
            "mikazuki.agent_train.graph.plan_user_message",
            return_value=PlannerDecision(
                skill_id="tagger_leaf_batch_caption",
                slots={"path": "/data/images"},
                payload_patch={"use_vlm": "false", "additional_tags": "kisaragi_yuuna", "input_dir": "/tmp/other"},
                confidence=0.9,
                reason="用户要求打标并调整参数",
                source="llm",
            ),
        ):
            result = run_agent_turn("给 /data/images 打标，不使用 vlm，触发词 kisaragi_yuuna", session_id="s1")

        payload = result["pending_action"]["payload"]
        self.assertEqual(payload["input_dir"], "/data/images")
        self.assertEqual(payload["use_vlm"], False)
        self.assertEqual(payload["additional_tags"], "kisaragi_yuuna")
        self.assertEqual(result["plan"]["payload_patch"]["applied"]["use_vlm"], False)
        self.assertIn("input_dir", result["plan"]["payload_patch"]["rejected"])

    def test_unknown_request_asks_for_missing_context(self):
        result = run_agent_turn("你好", session_id="s1")

        self.assertIsNone(result["pending_action"])
        self.assertIn("图片目录", result["assistant_message"])

    def test_checkpoint_does_not_store_raw_api_key(self):
        if not find_spec("langgraph.checkpoint.sqlite"):
            self.skipTest("sqlite checkpointer not installed")

        secret = "sk-agent-train-secret"
        result = run_agent_turn(
            "请给 /data/images 打标",
            session_id="s-secret",
            model_config={
                "base_url": "http://127.0.0.1:9/v1",
                "model": "local-model",
                "api_key": secret,
            },
        )

        self.assertEqual(result["pending_action"]["kind"], "run_tagger")
        checkpoint = Path(".cache/agent_train/checkpoints.sqlite")
        if checkpoint.is_file():
            self.assertNotIn(secret.encode("utf-8"), checkpoint.read_bytes())


if __name__ == "__main__":
    unittest.main()
