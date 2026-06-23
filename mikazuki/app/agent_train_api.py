from __future__ import annotations

from fastapi import APIRouter, Request

from mikazuki.agent_train.graph import run_agent_turn
from mikazuki.agent_train.readonly_tools import inspect_payload
from mikazuki.agent_train.sessions import AgentSessionStore
from mikazuki.agent_train.skills import make_skill_plan
from mikazuki.agent_train.slots import apply_slots_to_payload, build_slot_result
from mikazuki.agent_train.tools import AgentToolExecutor
from mikazuki.agent_train.validation import validate_skill_payload
from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.log import log

router = APIRouter(prefix="/api/agent-train")


def _store() -> AgentSessionStore:
    return AgentSessionStore()


def _is_confirm_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return normalized in {"确认", "批准", "执行", "开始", "同意", "approve", "yes", "y", "ok"}


def _advance_workflow_after_success(store: AgentSessionStore, session: dict) -> dict:
    workflow = session.get("workflow") or {}
    next_skill_id = str(workflow.get("next_skill_id") or "")
    if not next_skill_id:
        return session

    result = session.get("last_tool_result")
    if isinstance(result, dict) and result.get("status") == "fail":
        return session

    slots = session.get("slots") or {}
    slot_result = build_slot_result(next_skill_id, "", slots)
    plan, action = make_skill_plan(next_skill_id, str(slots.get("path") or slots.get("folder_a") or ""))
    plan["slots"] = slot_result["slots"]
    plan["missing_slots"] = slot_result["missing_slots"]
    plan["payload"] = apply_slots_to_payload(next_skill_id, plan["payload"], slot_result["slots"])
    action["payload"] = apply_slots_to_payload(next_skill_id, action["payload"], slot_result["slots"])
    decision = {
        "skill_id": next_skill_id,
        "slots": slot_result["slots"],
        "confidence": 1.0,
        "reason": "上一步执行成功后进入 workflow 下一步。",
        "questions": [],
        "source": "workflow",
        "fallback_reason": "",
    }
    validation = validate_skill_payload(next_skill_id, action["payload"]).as_dict()
    readonly_checks = inspect_payload(next_skill_id, action["payload"])
    plan["planner"] = decision
    plan["validation"] = validation
    plan["readonly_checks"] = readonly_checks
    action["validation"] = validation

    store.set_plan(session["session_id"], plan)
    if slot_result["can_execute"] and validation["can_execute"]:
        store.set_pending_action(session["session_id"], action)
        status = "awaiting_confirmation"
    else:
        store.set_pending_action(session["session_id"], None)
        status = "collecting_slots"
    steps = []
    for step in workflow.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step = dict(step)
        if step.get("skill_id") == workflow.get("current_skill_id"):
            step["status"] = "completed"
        elif step.get("skill_id") == next_skill_id:
            step["status"] = status
        steps.append(step)
    if not any(step.get("skill_id") == next_skill_id for step in steps):
        steps.append({"skill_id": next_skill_id, "status": status})

    store.set_agent_state(
        session["session_id"],
        slots=slot_result["slots"],
        missing_slots=slot_result["missing_slots"],
        validation=validation,
        workflow={"current_skill_id": next_skill_id, "next_skill_id": "", "status": status, "steps": steps},
    )
    store.append_decision_log(
        session["session_id"],
        user_message="",
        decision=decision,
        missing_slots=slot_result["missing_slots"],
        validation=validation,
        pending_action=action if status == "awaiting_confirmation" else None,
    )
    store.record_assistant_message(
        session["session_id"],
        "上一步已完成。我已生成下一步计划，请先补齐关键参数并确认后再执行。",
    )
    return store.get_session(session["session_id"])


@router.post("/sessions")
async def create_session():
    return APIResponseSuccess(data=_store().create_session())


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        return APIResponseSuccess(data=_store().get_session(session_id))
    except FileNotFoundError:
        return APIResponseFail(message="Agent session not found")
    except ValueError as exc:
        return APIResponseFail(message=str(exc))


@router.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON 请求体")

    store = _store()
    session_id = data.get("session_id") or ""
    session = None
    if session_id:
        try:
            session = store.get_session(session_id)
        except FileNotFoundError:
            return APIResponseFail(message="Agent session not found")
        except ValueError as exc:
            return APIResponseFail(message=str(exc))
    else:
        session = store.create_session()
        session_id = session["session_id"]

    message = str(data.get("message") or "").strip()
    if not message:
        return APIResponseFail(message="请输入对话内容")

    model_config = data.get("model_config") if isinstance(data.get("model_config"), dict) else {}
    store.record_user_message(session_id, message, model_config=model_config)

    if isinstance(session, dict) and session.get("pending_action") and _is_confirm_message(message):
        try:
            approved = await store.approve_pending_action(session_id, AgentToolExecutor())
            approved = _advance_workflow_after_success(store, approved)
            return APIResponseSuccess(data=approved)
        except Exception as exc:  # noqa: BLE001 - keep chat failures structured
            log.error(f"[AgentTrain] chat approve failed: {exc}")
            return APIResponseFail(message=f"执行失败: {exc}")

    try:
        result = run_agent_turn(
            message,
            session_id=session_id,
            model_config=model_config,
            slots=session.get("slots") if isinstance(session, dict) else {},
            workflow=session.get("workflow") if isinstance(session, dict) else {},
        )
    except Exception as exc:  # noqa: BLE001 - return structured API failure
        log.error(f"[AgentTrain] graph failed: {exc}")
        return APIResponseFail(message=f"Agent 运行失败: {exc}")

    store.record_assistant_message(session_id, result.get("assistant_message", ""))
    store.set_plan(session_id, result.get("plan"))
    store.set_pending_action(session_id, result.get("pending_action"))
    store.set_agent_state(
        session_id,
        slots=result.get("slots"),
        missing_slots=result.get("missing_slots"),
        validation=result.get("validation"),
        workflow=result.get("workflow"),
    )
    store.append_decision_log(
        session_id,
        user_message=message,
        decision=result.get("decision"),
        missing_slots=result.get("missing_slots"),
        validation=result.get("validation"),
        pending_action=result.get("pending_action"),
    )

    return APIResponseSuccess(data=store.get_session(session_id))


@router.post("/approve")
async def approve(request: Request):
    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON 请求体")

    session_id = str(data.get("session_id") or "")
    decision = str(data.get("decision") or "approve")
    store = _store()

    try:
        if decision == "reject":
            session = store.reject_pending_action(session_id, str(data.get("reason") or ""))
        elif decision in {"approve", "edit"}:
            override = data.get("payload") if decision == "edit" and isinstance(data.get("payload"), dict) else None
            session = await store.approve_pending_action(session_id, AgentToolExecutor(), payload_override=override)
            session = _advance_workflow_after_success(store, session)
        else:
            return APIResponseFail(message=f"不支持的确认动作: {decision}")
    except FileNotFoundError:
        return APIResponseFail(message="Agent session not found")
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception as exc:  # noqa: BLE001 - keep API failures structured
        log.error(f"[AgentTrain] approve failed: {exc}")
        return APIResponseFail(message=f"执行失败: {exc}")

    return APIResponseSuccess(data=session)


@router.get("/status/{session_id}")
async def status(session_id: str):
    try:
        session = _store().get_session(session_id)
    except FileNotFoundError:
        return APIResponseFail(message="Agent session not found")
    except ValueError as exc:
        return APIResponseFail(message=str(exc))

    data = {"session": session, "train_tasks": []}
    try:
        from mikazuki.tasks import tm

        data["train_tasks"] = tm.dump()
    except Exception:
        pass
    try:
        from mikazuki.app.tag_edit_leaf_api import _task_state as tagger_state

        data["tagger"] = dict(tagger_state)
    except Exception:
        pass
    return APIResponseSuccess(data=data)
