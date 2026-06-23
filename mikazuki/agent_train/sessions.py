from __future__ import annotations

import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from mikazuki.agent_train.skills import BUILTIN_SKILLS
from mikazuki.agent_train.validation import validate_skill_payload

SIDE_EFFECT_ACTIONS = {skill.action_kind for skill in BUILTIN_SKILLS}
ACTION_KIND_TO_SKILL_ID = {skill.action_kind: skill.id for skill in BUILTIN_SKILLS}


def default_session_dir() -> Path:
    root = os.environ.get("AGENT_TRAIN_STATE_DIR")
    if root:
        return Path(root).expanduser() / "sessions"
    return Path.cwd() / ".cache" / "agent_train" / "sessions"


def redact_model_config(config: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = dict(config or {})
    if "api_key" in sanitized:
        sanitized["api_key"] = ""
    return sanitized


class AgentSessionStore:
    """Small JSON-backed session store for browser-refresh recovery."""

    def __init__(self, root: Path | None = None):
        self.root = root or default_session_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not session_id or "/" in session_id or "\\" in session_id:
            raise ValueError("invalid session_id")
        return self.root / f"{session_id}.json"

    def _write(self, session: dict[str, Any]) -> dict[str, Any]:
        session["updated_at"] = int(time.time())
        path = self._path(session["session_id"])
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        return copy.deepcopy(session)

    def create_session(self) -> dict[str, Any]:
        session = {
            "session_id": f"agent-{uuid.uuid4().hex[:12]}",
            "messages": [],
            "model_config": {},
            "plan": None,
            "pending_action": None,
            "last_tool_result": None,
            "linked_task_ids": [],
            "slots": {},
            "missing_slots": [],
            "workflow": {},
            "validation": {"errors": [], "warnings": [], "can_execute": False},
            "decision_log": [],
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        return self._write(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"session not found: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def record_user_message(
        self,
        session_id: str,
        content: str,
        *,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["messages"].append({
            "role": "user",
            "content": str(content or ""),
            "created_at": int(time.time()),
        })
        if model_config:
            session["model_config"] = redact_model_config(model_config)
        return self._write(session)

    def record_assistant_message(self, session_id: str, content: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["messages"].append({
            "role": "assistant",
            "content": str(content or ""),
            "created_at": int(time.time()),
        })
        return self._write(session)

    def set_plan(self, session_id: str, plan: dict[str, Any] | None) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["plan"] = plan
        return self._write(session)

    def set_agent_state(
        self,
        session_id: str,
        *,
        slots: dict[str, Any] | None = None,
        missing_slots: list[str] | None = None,
        workflow: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if slots is not None:
            session["slots"] = slots
        if missing_slots is not None:
            session["missing_slots"] = missing_slots
        if workflow is not None:
            session["workflow"] = workflow
        if validation is not None:
            session["validation"] = validation
        return self._write(session)

    def append_decision_log(
        self,
        session_id: str,
        *,
        user_message: str,
        decision: dict[str, Any] | None,
        missing_slots: list[str] | None,
        validation: dict[str, Any] | None,
        pending_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        log = session.get("decision_log")
        if not isinstance(log, list):
            log = []
        decision = decision or {}
        validation = validation or {"errors": [], "warnings": [], "can_execute": False}
        log.append({
            "created_at": int(time.time()),
            "user_message": str(user_message or ""),
            "planner_source": decision.get("source") or "",
            "skill_id": decision.get("skill_id") or "",
            "confidence": decision.get("confidence") or 0,
            "reason": decision.get("reason") or "",
            "fallback_reason": decision.get("fallback_reason") or "",
            "slots": decision.get("slots") if isinstance(decision.get("slots"), dict) else {},
            "missing_slots": list(missing_slots or []),
            "validation": validation,
            "pending_action": bool(pending_action),
        })
        session["decision_log"] = log[-50:]
        return self._write(session)

    def set_pending_action(self, session_id: str, action: dict[str, Any] | None) -> dict[str, Any]:
        session = self.get_session(session_id)
        if action is None:
            session["pending_action"] = None
            return self._write(session)

        kind = action.get("kind")
        if kind not in SIDE_EFFECT_ACTIONS:
            raise ValueError(f"unsupported side-effect action: {kind}")
        session["pending_action"] = {
            "kind": kind,
            "skill_id": action.get("skill_id") or ACTION_KIND_TO_SKILL_ID.get(kind, ""),
            "title": action.get("title") or kind,
            "payload": action.get("payload") or {},
            "summary": action.get("summary") or [],
            "side_effect_level": action.get("side_effect_level") or "writes_files",
            "validation": action.get("validation") or {"errors": [], "warnings": [], "can_execute": True},
            "status": "pending",
            "created_at": int(time.time()),
        }
        return self._write(session)

    def reject_pending_action(self, session_id: str, reason: str = "") -> dict[str, Any]:
        session = self.get_session(session_id)
        if session.get("pending_action"):
            session["messages"].append({
                "role": "assistant",
                "content": f"已取消待执行动作。{reason}".strip(),
                "created_at": int(time.time()),
            })
        session["pending_action"] = None
        return self._write(session)

    async def approve_pending_action(self, session_id: str, executor, payload_override: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self.get_session(session_id)
        pending = session.get("pending_action")
        if not pending:
            raise ValueError("no pending action")

        payload = payload_override if payload_override is not None else pending.get("payload") or {}
        skill_id = str(pending.get("skill_id") or ACTION_KIND_TO_SKILL_ID.get(pending["kind"], ""))
        validation = validate_skill_payload(skill_id, payload).as_dict()
        if not validation["can_execute"]:
            pending["validation"] = validation
            session["pending_action"] = pending
            session["last_tool_result"] = {
                "status": "fail",
                "message": "参数校验失败",
                "data": {"validation": validation},
            }
            session["messages"].append({
                "role": "assistant",
                "content": "参数校验失败，请修正后再执行。",
                "created_at": int(time.time()),
            })
            return self._write(session)

        result = await executor.execute(pending["kind"], payload)
        if isinstance(result, dict) and result.get("status") == "fail":
            session["last_tool_result"] = result
            session["messages"].append({
                "role": "assistant",
                "content": f"执行失败：{result.get('message') or pending.get('title')}",
                "created_at": int(time.time()),
            })
            return self._write(session)

        session["pending_action"] = None
        session["last_tool_result"] = result
        data = result.get("data") if isinstance(result, dict) else None
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if task_id and task_id not in session["linked_task_ids"]:
            session["linked_task_ids"].append(task_id)
        session["messages"].append({
            "role": "assistant",
            "content": f"已执行：{pending.get('title')}",
            "created_at": int(time.time()),
        })
        return self._write(session)
