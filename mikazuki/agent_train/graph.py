from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mikazuki.agent_train.planner import PlannerDecision, plan_user_message
from mikazuki.agent_train.parameter_docs import apply_payload_patch
from mikazuki.agent_train.readonly_tools import inspect_payload
from mikazuki.agent_train.skills import get_skill, make_skill_plan
from mikazuki.agent_train.slots import apply_slots_to_payload, build_slot_result
from mikazuki.agent_train.validation import validate_skill_payload


class AgentTrainState(TypedDict, total=False):
    session_id: str
    user_message: str
    assistant_message: str
    plan: dict[str, Any] | None
    pending_action: dict[str, Any] | None
    model_config: dict[str, Any]
    slots: dict[str, Any]
    missing_slots: list[str]
    workflow: dict[str, Any]
    decision: dict[str, Any]
    validation: dict[str, Any]


IMAGE_DIR_RE = re.compile(r"(/[^ \n\r\t，。；;]+)")


def default_checkpoint_path() -> Path:
    root = os.environ.get("AGENT_TRAIN_STATE_DIR")
    path = (Path(root).expanduser() if root else Path.cwd() / ".cache" / "agent_train") / "checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _extract_path(text: str) -> str:
    match = IMAGE_DIR_RE.search(text)
    return match.group(1) if match else ""


def _next_workflow_skill_id(message: str, current_skill_id: str) -> str:
    if current_skill_id != "tagger_leaf_batch_caption":
        return ""
    lower = message.lower()
    if "anima fast" in lower or "anima-fast" in lower:
        return "anima_fast_train"
    if "differential" in lower or "差分" in message:
        return "differential_lora_train"
    if "anima" in lower or "训练" in message:
        return "anima_lora_train"
    return ""


def _decision_to_dict(decision: PlannerDecision) -> dict[str, Any]:
    return {
        "skill_id": decision.skill_id or "",
        "slots": dict(decision.slots),
        "payload_patch_fields": sorted(str(key) for key in decision.payload_patch),
        "confidence": decision.confidence,
        "reason": decision.reason,
        "questions": list(decision.questions),
        "source": decision.source,
        "fallback_reason": decision.fallback_reason,
    }


def _workflow_steps(
    workflow: dict[str, Any],
    *,
    current_skill_id: str,
    current_status: str,
    next_skill_id: str = "",
) -> list[dict[str, Any]]:
    existing = workflow.get("steps")
    if isinstance(existing, list) and existing:
        steps = [dict(step) for step in existing if isinstance(step, dict)]
    else:
        steps = []

    if not steps or steps[0].get("skill_id") != current_skill_id:
        steps = [{"skill_id": current_skill_id, "status": current_status}]
    else:
        steps[0]["status"] = current_status

    if next_skill_id:
        if len(steps) == 1:
            steps.append({"skill_id": next_skill_id, "status": "pending"})
        elif steps[1].get("skill_id") != next_skill_id:
            steps = steps[:1] + [{"skill_id": next_skill_id, "status": "pending"}]
    return steps


def _workflow_state(
    workflow: dict[str, Any],
    *,
    current_skill_id: str,
    status: str,
    next_skill_id: str = "",
) -> dict[str, Any]:
    return {
        "current_skill_id": current_skill_id,
        "next_skill_id": next_skill_id,
        "status": status,
        "steps": _workflow_steps(
            workflow,
            current_skill_id=current_skill_id,
            current_status=status,
            next_skill_id=next_skill_id,
        ),
    }


def _planner_node(state: AgentTrainState) -> AgentTrainState:
    message = state.get("user_message", "")
    path = _extract_path(message)
    workflow = state.get("workflow") or {}
    decision = plan_user_message(
        message,
        model_config=state.get("model_config") or {},
        existing_slots=state.get("slots") or {},
        workflow=workflow,
    )
    decision_data = _decision_to_dict(decision)

    if decision.skill_id:
        skill = get_skill(decision.skill_id)
        plan, action = make_skill_plan(decision.skill_id, path)
        slot_result = build_slot_result(decision.skill_id, message, decision.slots)
        slots = slot_result["slots"]
        next_skill_id = _next_workflow_skill_id(message, decision.skill_id)
        plan["slots"] = slots
        plan["missing_slots"] = slot_result["missing_slots"]
        plan["planner"] = decision_data
        plan_payload = apply_slots_to_payload(decision.skill_id, plan["payload"], slots)
        action_payload = apply_slots_to_payload(decision.skill_id, action["payload"], slots)
        plan_payload, patch_report = apply_payload_patch(decision.skill_id, plan_payload, decision.payload_patch)
        action_payload, patch_report = apply_payload_patch(decision.skill_id, action_payload, decision.payload_patch)
        plan["payload"] = plan_payload
        action["payload"] = action_payload
        plan["payload_patch"] = patch_report
        action["payload_patch"] = patch_report
        decision_data["payload_patch_result"] = {
            "applied_fields": sorted(patch_report["applied"]),
            "rejected_fields": sorted(patch_report["rejected"]),
        }
        validation = validate_skill_payload(decision.skill_id, action["payload"]).as_dict()
        readonly_checks = inspect_payload(decision.skill_id, action["payload"])
        plan["validation"] = validation
        plan["readonly_checks"] = readonly_checks
        action["validation"] = validation

        if not slot_result["can_execute"]:
            return {
                **state,
                "assistant_message": slot_result["assistant_message"],
                "plan": plan,
                "pending_action": None,
                "slots": slots,
                "missing_slots": slot_result["missing_slots"],
                "decision": decision_data,
                "validation": validation,
                "workflow": _workflow_state(
                    workflow,
                    current_skill_id=decision.skill_id,
                    next_skill_id=next_skill_id,
                    status="collecting_slots",
                ),
            }

        if not validation["can_execute"]:
            return {
                **state,
                "assistant_message": "计划已生成，但参数校验未通过。请先修正错误后再执行。",
                "plan": plan,
                "pending_action": None,
                "slots": slots,
                "missing_slots": [],
                "decision": decision_data,
                "validation": validation,
                "workflow": _workflow_state(
                    workflow,
                    current_skill_id=decision.skill_id,
                    next_skill_id=next_skill_id,
                    status="collecting_slots",
                ),
            }

        return {
            **state,
            "assistant_message": skill.assistant_message if skill else "我已生成计划。请检查参数，确认后我再执行。",
            "plan": plan,
            "pending_action": action,
            "slots": slots,
            "missing_slots": [],
            "decision": decision_data,
            "validation": validation,
            "workflow": _workflow_state(
                workflow,
                current_skill_id=decision.skill_id,
                next_skill_id=next_skill_id,
                status="awaiting_confirmation",
            ),
        }

    return {
        **state,
        "assistant_message": "请告诉我图片目录、想先打标还是直接训练，以及训练类型：Differential LoRA、Anima LoRA 或 Anima Fast。",
        "plan": None,
        "pending_action": None,
        "missing_slots": [],
        "decision": decision_data,
        "validation": {"errors": [], "warnings": [], "can_execute": False},
    }


def _llm_node(state: AgentTrainState) -> AgentTrainState:
    decision = state.get("decision") or {}
    if decision.get("source") == "llm":
        return state

    config = state.get("model_config") or {}
    model = str(config.get("model") or "").strip()
    base_url = str(config.get("base_url") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    if not model or (not api_key and not base_url):
        return state

    plan = state.get("plan")
    pending = state.get("pending_action")
    system = (
        "你是 LoRA 训练编排助手。你可以帮助用户规划 Tag-Edit-Leaf 打标、"
        "Differential LoRA、Anima LoRA 和 Anima Fast 训练。"
        "任何会写文件、启动打标或启动训练的动作，都必须等待用户在界面确认；"
        "不要声称已经执行未确认的动作。回答要简洁，指出还缺哪些参数。"
    )
    user = (
        f"用户输入：{state.get('user_message', '')}\n"
        f"本地规则生成的回复：{state.get('assistant_message', '')}\n"
        f"计划：{plan or '无'}\n"
        f"待确认动作：{pending or '无'}\n"
        "请基于以上内容生成给用户的下一条回复。"
    )

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key or "not-needed",
            base_url=base_url or None,
            timeout=20,
            max_retries=0,
        )
        response = llm.invoke([
            ("system", system),
            ("user", user),
        ])
    except Exception:
        return state

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    content = str(content or "").strip()
    if not content:
        return state
    return {**state, "assistant_message": content}


def build_agent_graph(*, checkpointer: Any | None = None):
    graph = StateGraph(AgentTrainState)
    graph.add_node("planner", _planner_node)
    graph.add_node("llm", _llm_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "llm")
    graph.add_edge("llm", END)
    return graph.compile(checkpointer=checkpointer)


def run_agent_turn(
    user_message: str,
    *,
    session_id: str,
    model_config: dict[str, Any] | None = None,
    slots: dict[str, Any] | None = None,
    workflow: dict[str, Any] | None = None,
) -> AgentTrainState:
    safe_model_config = dict(model_config or {})
    secret_api_key = str(safe_model_config.get("api_key") or "")
    state = {
        "session_id": session_id,
        "user_message": user_message,
        "model_config": safe_model_config,
        "slots": slots or {},
        "workflow": workflow or {},
    }
    config = {"configurable": {"thread_id": session_id}}

    if secret_api_key:
        graph = build_agent_graph()
        return graph.invoke(state, config=config)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        graph = build_agent_graph()
        return graph.invoke(state, config=config)

    checkpoint_path = default_checkpoint_path()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_agent_graph(checkpointer=checkpointer)
        return graph.invoke(state, config=config)
