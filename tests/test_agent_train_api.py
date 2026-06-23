from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starlette.requests import Request


def make_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/api/test", "headers": []}, receive)


class AgentTrainApiTests(unittest.TestCase):
    def test_create_session_persists_refreshable_state(self):
        from mikazuki.agent_train.sessions import AgentSessionStore

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            loaded = store.get_session(session["session_id"])

        self.assertEqual(loaded["session_id"], session["session_id"])
        self.assertEqual(loaded["messages"], [])
        self.assertIsNone(loaded["pending_action"])
        self.assertEqual(loaded["decision_log"], [])
        self.assertIn("can_execute", loaded["validation"])

    def test_chat_does_not_persist_api_key(self):
        from mikazuki.agent_train.sessions import AgentSessionStore

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            updated = store.record_user_message(
                session["session_id"],
                "给 /data/images 打标",
                model_config={
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "local-model",
                    "api_key": "secret-value",
                },
            )
            raw = (Path(td) / f"{session['session_id']}.json").read_text(encoding="utf-8")

        self.assertNotIn("secret-value", raw)
        self.assertEqual(updated["model_config"]["api_key"], "")
        self.assertEqual(updated["model_config"]["model"], "local-model")

    def test_pending_side_effect_requires_approval(self):
        from mikazuki.agent_train.sessions import AgentSessionStore

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            updated = store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "title": "标注图片目录",
                    "payload": {"input_dir": "/data/images"},
                    "summary": ["将写入 .txt caption 文件"],
                    "validation": {"errors": [], "warnings": ["check"], "can_execute": True},
                },
            )

        self.assertEqual(updated["pending_action"]["kind"], "run_tagger")
        self.assertEqual(updated["pending_action"]["status"], "pending")
        self.assertEqual(updated["pending_action"]["validation"]["warnings"], ["check"])

    def test_decision_log_persists_without_api_key(self):
        from mikazuki.agent_train.sessions import AgentSessionStore

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.append_decision_log(
                session["session_id"],
                user_message="给 /data/images 打标",
                decision={
                    "source": "rule",
                    "skill_id": "tagger_leaf_batch_caption",
                    "confidence": 0.7,
                    "reason": "matched",
                    "slots": {"path": "/data/images"},
                },
                missing_slots=[],
                validation={"errors": [], "warnings": [], "can_execute": True},
                pending_action={"kind": "run_tagger"},
            )
            raw = (Path(td) / f"{session['session_id']}.json").read_text(encoding="utf-8")
            loaded = store.get_session(session["session_id"])

        self.assertNotIn("api_key", raw)
        self.assertEqual(loaded["decision_log"][0]["planner_source"], "rule")
        self.assertTrue(loaded["decision_log"][0]["pending_action"])

    def test_approve_executes_pending_action_once(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.agent_train.tools import AgentToolExecutor

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "title": "标注图片目录",
                    "payload": {"input_dir": "/data/images"},
                    "summary": [],
                },
            )
            executor = AgentToolExecutor()
            with mock.patch.object(
                executor,
                "run_tagger",
                return_value={"status": "success", "data": {"task": "tagger"}},
            ) as runner:
                updated = asyncio.run(store.approve_pending_action(session["session_id"], executor))

        runner.assert_called_once_with({"input_dir": "/data/images"})
        self.assertIsNone(updated["pending_action"])
        self.assertEqual(updated["last_tool_result"]["data"]["task"], "tagger")

    def test_approve_rejects_invalid_payload_override_before_execution(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.agent_train.tools import AgentToolExecutor

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "skill_id": "tagger_leaf_batch_caption",
                    "title": "标注图片目录",
                    "payload": {"input_dir": "/data/images"},
                    "summary": [],
                },
            )
            executor = AgentToolExecutor()
            with mock.patch.object(executor, "run_tagger", return_value={"status": "success"}) as runner:
                updated = asyncio.run(
                    store.approve_pending_action(
                        session["session_id"],
                        executor,
                        payload_override={"input_dir": "", "threshold": 2},
                    )
                )

        runner.assert_not_called()
        self.assertEqual(updated["pending_action"]["kind"], "run_tagger")
        self.assertEqual(updated["last_tool_result"]["status"], "fail")
        self.assertIn("参数校验失败", updated["messages"][-1]["content"])

    def test_failed_tool_response_keeps_pending_action(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.agent_train.tools import AgentToolExecutor

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "title": "标注图片目录",
                    "payload": {"input_dir": "/missing"},
                    "summary": [],
                },
            )
            executor = AgentToolExecutor()
            with mock.patch.object(
                executor,
                "run_tagger",
                return_value={"status": "fail", "message": "目录不存在", "data": {}},
            ):
                updated = asyncio.run(store.approve_pending_action(session["session_id"], executor))

        self.assertEqual(updated["pending_action"]["kind"], "run_tagger")
        self.assertEqual(updated["last_tool_result"]["status"], "fail")
        self.assertIn("执行失败", updated["messages"][-1]["content"])

    def test_agent_train_router_session_endpoint(self):
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", autospec=True) as store_cls:
                store = store_cls.return_value
                store.create_session.return_value = {"session_id": "session-1", "messages": []}
                response = asyncio.run(agent_train_api.create_session())

        self.assertEqual(response.status, "success")
        self.assertEqual(response.data["session_id"], "session-1")

    def test_chat_endpoint_records_plan_and_pending_action(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        graph_result = {
            "assistant_message": "请确认打标计划。",
            "plan": {"type": "tagger", "payload": {"input_dir": "/data/images"}},
            "pending_action": {
                "kind": "run_tagger",
                "title": "执行打标",
                "payload": {"input_dir": "/data/images"},
                "summary": ["写入 caption"],
            },
            "decision": {"source": "rule", "skill_id": "tagger_leaf_batch_caption", "slots": {"path": "/data/images"}},
            "validation": {"errors": [], "warnings": [], "can_execute": True},
        }

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                with mock.patch("mikazuki.app.agent_train_api.run_agent_turn", return_value=graph_result) as runner:
                    response = asyncio.run(agent_train_api.chat(make_request({
                        "message": "给 /data/images 打标",
                    })))

        runner.assert_called_once()
        self.assertEqual(response.status, "success")
        self.assertEqual(response.data["messages"][0]["role"], "user")
        self.assertEqual(response.data["messages"][1]["role"], "assistant")
        self.assertEqual(response.data["plan"]["type"], "tagger")
        self.assertEqual(response.data["pending_action"]["kind"], "run_tagger")
        self.assertEqual(response.data["decision_log"][0]["planner_source"], "rule")
        self.assertTrue(response.data["validation"]["can_execute"])

    def test_chat_endpoint_clears_stale_plan_and_pending_action(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        graph_result = {
            "assistant_message": "请补充图片目录和训练类型。",
            "plan": None,
            "pending_action": None,
        }

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.set_plan(session["session_id"], {"type": "tagger"})
            store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "title": "执行打标",
                    "payload": {"input_dir": "/data/images"},
                    "summary": [],
                },
            )
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                with mock.patch("mikazuki.app.agent_train_api.run_agent_turn", return_value=graph_result):
                    response = asyncio.run(agent_train_api.chat(make_request({
                        "session_id": session["session_id"],
                        "message": "你好",
                    })))

        self.assertEqual(response.status, "success")
        self.assertIsNone(response.data["plan"])
        self.assertIsNone(response.data["pending_action"])

    def test_chat_endpoint_rejects_invalid_session_id(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                response = asyncio.run(agent_train_api.chat(make_request({
                    "session_id": "../bad",
                    "message": "你好",
                })))

        self.assertEqual(response.status, "fail")
        self.assertIn("invalid session_id", response.message)

    def test_chat_endpoint_persists_slots_between_turns(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                first = asyncio.run(agent_train_api.chat(make_request({
                    "message": "使用 /data/train 做 Anima Fast",
                })))
                second = asyncio.run(agent_train_api.chat(make_request({
                    "session_id": first.data["session_id"],
                    "message": "base=/models/anima.safetensors vae=/models/vae.safetensors qwen3=/models/qwen3.safetensors",
                })))

        self.assertEqual(first.status, "success")
        self.assertIsNone(first.data["pending_action"])
        self.assertIn("pretrained_model_name_or_path", first.data["missing_slots"])
        self.assertEqual(second.status, "success")
        self.assertEqual(second.data["slots"]["path"], "/data/train")
        self.assertEqual(second.data["pending_action"]["kind"], "run_anima_fast_train")
        self.assertEqual(
            second.data["pending_action"]["payload"]["pretrained_model_name_or_path"],
            "/models/anima.safetensors",
        )
        self.assertEqual(second.data["workflow"]["steps"][0]["skill_id"], "anima_fast_train")

    def test_chat_confirmation_approves_existing_pending_action(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            session = store.create_session()
            store.set_pending_action(
                session["session_id"],
                {
                    "kind": "run_tagger",
                    "skill_id": "tagger_leaf_batch_caption",
                    "title": "标注图片目录",
                    "payload": {"input_dir": "/data/images"},
                    "summary": [],
                },
            )
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                with mock.patch(
                    "mikazuki.agent_train.tools.AgentToolExecutor.run_tagger",
                    return_value={"status": "success", "data": {"task": "tagger"}},
                ) as runner:
                    response = asyncio.run(agent_train_api.chat(make_request({
                        "session_id": session["session_id"],
                        "message": "确认",
                    })))

        runner.assert_called_once_with({"input_dir": "/data/images"})
        self.assertEqual(response.status, "success")
        self.assertIsNone(response.data["pending_action"])
        self.assertEqual(response.data["last_tool_result"]["data"]["task"], "tagger")
        self.assertIn("已执行", response.data["messages"][-1]["content"])

    def test_approve_tagger_advances_to_next_training_step(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                first = asyncio.run(agent_train_api.chat(make_request({
                    "message": "给 /data/train 打标，然后做 Anima Fast",
                })))
                with mock.patch(
                    "mikazuki.agent_train.tools.AgentToolExecutor.run_tagger",
                    return_value={"status": "success", "data": {"task": "tagger"}},
                ):
                    approved = asyncio.run(agent_train_api.approve(make_request({
                        "session_id": first.data["session_id"],
                        "decision": "approve",
                    })))

        self.assertEqual(approved.status, "success")
        self.assertIsNone(approved.data["pending_action"])
        self.assertEqual(approved.data["workflow"]["current_skill_id"], "anima_fast_train")
        self.assertEqual(approved.data["workflow"]["status"], "collecting_slots")
        self.assertEqual([step["skill_id"] for step in approved.data["workflow"]["steps"]], [
            "tagger_leaf_batch_caption",
            "anima_fast_train",
        ])
        self.assertEqual(approved.data["workflow"]["steps"][0]["status"], "completed")
        self.assertEqual(approved.data["workflow"]["steps"][1]["status"], "collecting_slots")
        self.assertEqual(approved.data["plan"]["skill_id"], "anima_fast_train")
        self.assertIn("pretrained_model_name_or_path", approved.data["missing_slots"])

    def test_status_endpoint_rejects_invalid_session_id(self):
        from mikazuki.agent_train.sessions import AgentSessionStore
        from mikazuki.app import agent_train_api

        with tempfile.TemporaryDirectory() as td:
            store = AgentSessionStore(Path(td))
            with mock.patch("mikazuki.app.agent_train_api.AgentSessionStore", return_value=store):
                response = asyncio.run(agent_train_api.status("../bad"))

        self.assertEqual(response.status, "fail")
        self.assertIn("invalid session_id", response.message)


if __name__ == "__main__":
    unittest.main()
