from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mikazuki.log import log
from mikazuki.agent_train.parameter_docs import format_parameter_docs_for_prompt, retrieve_parameter_docs
from mikazuki.agent_train.skills import get_skill, match_skill
from mikazuki.agent_train.slots import extract_slots


@dataclass(frozen=True)
class PlannerDecision:
    skill_id: str | None
    slots: dict[str, Any] = field(default_factory=dict)
    payload_patch: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    questions: list[str] = field(default_factory=list)
    source: str = "rule"
    fallback_reason: str = ""


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TRIGGER_RE = re.compile(r"(?:触发词|trigger(?: word)?)\s*(?:为|是|=|:|：)?\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
DISABLE_VLM_RE = re.compile(r"(?:不|不要|不用|关闭|禁用|不进行)[^，。；;\\n\\r]*(?:vlm|vllm|视觉语言模型|大模型打标)", re.IGNORECASE)
ENABLE_VLM_RE = re.compile(r"(?:使用|启用|开启)[^，。；;\\n\\r]*(?:vlm|vllm|视觉语言模型|大模型打标)", re.IGNORECASE)


def _model_config_ready(model_config: dict[str, Any] | None) -> bool:
    config = model_config or {}
    model = str(config.get("model") or "").strip()
    base_url = str(config.get("base_url") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    return bool(model and (api_key or base_url))


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _extract_rule_payload_patch(skill_id: str, message: str) -> dict[str, Any]:
    if skill_id != "tagger_leaf_batch_caption":
        return {}

    patch: dict[str, Any] = {}
    trigger = TRIGGER_RE.search(message)
    if trigger:
        patch["additional_tags"] = trigger.group(1).strip()
    if DISABLE_VLM_RE.search(message):
        patch["use_vlm"] = False
    elif ENABLE_VLM_RE.search(message):
        patch["use_vlm"] = True
    return patch


def parse_planner_json(
    content: str,
    *,
    fallback_reason: str = "invalid planner json",
    log_rejection: bool = False,
) -> PlannerDecision:
    match = JSON_OBJECT_RE.search(str(content or ""))
    if not match:
        if log_rejection:
            log.info(f"[AgentTrain] LLM planner rejected response: {fallback_reason}; no JSON object found")
        return PlannerDecision(skill_id=None, source="fallback", fallback_reason=fallback_reason)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        if log_rejection:
            log.info(f"[AgentTrain] LLM planner rejected response: {fallback_reason}; invalid JSON")
        return PlannerDecision(skill_id=None, source="fallback", fallback_reason=fallback_reason)

    if not isinstance(data, dict):
        if log_rejection:
            log.info(f"[AgentTrain] LLM planner rejected response: {fallback_reason}; JSON root is not an object")
        return PlannerDecision(skill_id=None, source="fallback", fallback_reason=fallback_reason)

    skill_id = str(data.get("skill_id") or "").strip()
    if skill_id and get_skill(skill_id) is None:
        if log_rejection:
            log.info(f"[AgentTrain] LLM planner rejected response: unknown skill_id={skill_id}")
        return PlannerDecision(skill_id=None, source="fallback", fallback_reason=f"unknown skill: {skill_id}")

    raw_slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
    slots = {str(key): value for key, value in raw_slots.items() if value not in (None, "")}
    raw_patch = data.get("payload_patch") if isinstance(data.get("payload_patch"), dict) else {}
    payload_patch = {str(key): value for key, value in raw_patch.items() if value is not None}
    questions = data.get("questions") if isinstance(data.get("questions"), list) else []
    return PlannerDecision(
        skill_id=skill_id or None,
        slots=slots,
        payload_patch=payload_patch,
        confidence=_coerce_confidence(data.get("confidence")),
        reason=str(data.get("reason") or ""),
        questions=[str(question) for question in questions if str(question).strip()],
        source="llm",
    )


def _rule_decision(
    message: str,
    *,
    existing_slots: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    fallback_reason: str = "",
) -> PlannerDecision:
    workflow = workflow or {}
    current_skill = get_skill(str(workflow.get("current_skill_id") or ""))
    skill = match_skill(message)
    if current_skill and workflow.get("status") == "collecting_slots" and any(token in message for token in ("=", ":", "：", "是")):
        skill = current_skill
    elif skill is None and current_skill:
        skill = current_skill

    if not skill:
        return PlannerDecision(
            skill_id=None,
            slots=dict(existing_slots or {}),
            confidence=0.0,
            reason="未匹配到可执行 skill。",
            source="rule",
            fallback_reason=fallback_reason,
        )

    slots = extract_slots(skill.id, message, existing_slots)
    return PlannerDecision(
        skill_id=skill.id,
        slots=slots,
        payload_patch=_extract_rule_payload_patch(skill.id, message),
        confidence=0.7,
        reason="根据关键词和当前 workflow 选择 skill。",
        source="rule",
        fallback_reason=fallback_reason,
    )


def _plan_with_llm(
    message: str,
    *,
    model_config: dict[str, Any],
    existing_slots: dict[str, Any],
    workflow: dict[str, Any],
) -> PlannerDecision | None:
    if not _model_config_ready(model_config):
        return None

    model = str(model_config.get("model") or "").strip()
    base_url = str(model_config.get("base_url") or "").strip()
    api_key = str(model_config.get("api_key") or "").strip()
    candidate_skill = match_skill(message) or get_skill(str(workflow.get("current_skill_id") or ""))
    docs_text = "[]"
    if candidate_skill:
        current_payload = candidate_skill.default_payload(str(existing_slots.get("path") or existing_slots.get("folder_a") or ""))
        docs = retrieve_parameter_docs(candidate_skill.id, message, current_payload)
        docs_text = format_parameter_docs_for_prompt(docs)

    system = (
        "你是 LoRA 训练编排 planner。只输出一个 JSON 对象，不要输出 markdown。"
        "可选 skill_id: tagger_leaf_batch_caption, differential_lora_train, anima_lora_train, anima_fast_train。"
        "字段必须包含 skill_id, slots, payload_patch, confidence, reason, questions。"
        "slots 只填用户明确给出的值；不确定时留空并在 questions 里提问。"
        "payload_patch 只填参数文档中 llm_editable=true 且用户明确要求修改的字段；不要填路径、模型路径等 slots 管理字段。"
        "不要编造参数字段；不确定时不要写入 payload_patch。"
    )
    user = (
        f"用户输入: {message}\n"
        f"已有 slots: {json.dumps(existing_slots, ensure_ascii=False)}\n"
        f"workflow: {json.dumps(workflow, ensure_ascii=False)}\n"
        f"当前 skill 相关参数文档片段: {docs_text}"
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
        response = llm.invoke([("system", system), ("user", user)])
    except Exception as exc:
        log.error(
            f"[AgentTrain] LLM planner call failed: {type(exc).__name__}: {exc}; "
            f"model={model}, base_url={base_url or '<default>'}"
        )
        return None

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    content = str(content or "").strip()
    if not content:
        log.info(f"[AgentTrain] LLM planner returned empty response; model={model}, base_url={base_url or '<default>'}")
        return None

    log.info(f"[AgentTrain] LLM planner raw response: {content[:1000]!r}")
    decision = parse_planner_json(content, fallback_reason="invalid llm planner response", log_rejection=True)
    if decision.source != "llm" or not decision.skill_id:
        log.info(
            f"[AgentTrain] LLM planner unavailable after parsing; "
            f"source={decision.source}, skill_id={decision.skill_id or ''}, reason={decision.fallback_reason}"
        )
        return None
    if decision.confidence < 0.45:
        log.info(
            f"[AgentTrain] LLM planner rejected response: confidence too low "
            f"({decision.confidence}); skill_id={decision.skill_id}"
        )
        return None
    return decision


def plan_user_message(
    message: str,
    *,
    model_config: dict[str, Any] | None,
    existing_slots: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
) -> PlannerDecision:
    existing = dict(existing_slots or {})
    flow = dict(workflow or {})
    llm_decision = _plan_with_llm(
        message,
        model_config=model_config or {},
        existing_slots=existing,
        workflow=flow,
    )
    if llm_decision and llm_decision.skill_id:
        merged_slots = extract_slots(llm_decision.skill_id, message, existing)
        merged_slots.update({key: value for key, value in llm_decision.slots.items() if value not in (None, "")})
        return PlannerDecision(
            skill_id=llm_decision.skill_id,
            slots=merged_slots,
            payload_patch=dict(llm_decision.payload_patch),
            confidence=llm_decision.confidence,
            reason=llm_decision.reason,
            questions=llm_decision.questions,
            source="llm",
        )
    fallback_reason = "llm planner unavailable" if _model_config_ready(model_config) else ""
    return _rule_decision(message, existing_slots=existing, workflow=flow, fallback_reason=fallback_reason)
