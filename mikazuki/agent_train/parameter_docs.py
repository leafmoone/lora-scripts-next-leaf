from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DOCS_DIR = Path(__file__).with_name("rag_docs")
TOKEN_RE = re.compile(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(text or ""))}


@lru_cache(maxsize=16)
def load_parameter_docs(skill_id: str) -> tuple[dict[str, Any], ...]:
    path = DOCS_DIR / f"{skill_id}.json"
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(data, list):
        return ()
    return tuple(item for item in data if isinstance(item, dict) and str(item.get("field") or "").strip())


def retrieve_parameter_docs(
    skill_id: str,
    message: str,
    current_payload: dict[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    message_lower = str(message or "").lower()
    message_tokens = _tokens(message_lower)
    payload_fields = set(current_payload)
    scored: list[tuple[int, dict[str, Any]]] = []

    for doc in load_parameter_docs(skill_id):
        field = str(doc.get("field") or "").strip()
        if field not in payload_fields:
            continue
        score = 0
        if field.lower() in message_lower:
            score += 10
        for key, weight in (("aliases", 6), ("intents", 5)):
            values = doc.get(key) if isinstance(doc.get(key), list) else []
            for value in values:
                value_lower = str(value).lower()
                if value_lower and value_lower in message_lower:
                    score += weight
                score += len(_tokens(value_lower) & message_tokens) * 2
        description = str(doc.get("description") or "").lower()
        score += len(_tokens(description) & message_tokens)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: (-item[0], str(item[1].get("field") or "")))
    return [dict(doc) for _, doc in scored[:limit]]


def format_parameter_docs_for_prompt(docs: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for doc in docs:
        rows.append(
            {
                "field": doc.get("field"),
                "type": doc.get("type"),
                "default": doc.get("default"),
                "description": doc.get("description"),
                "aliases": doc.get("aliases", []),
                "intents": doc.get("intents", []),
                "llm_editable": bool(doc.get("llm_editable")),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def _coerce_bool(value: Any) -> tuple[bool, Any]:
    if isinstance(value, bool):
        return True, value
    if isinstance(value, (int, float)) and value in (0, 1):
        return True, bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on", "enable", "enabled", "开启", "启用", "使用", "是"}:
        return True, True
    if text in {"false", "0", "no", "n", "off", "disable", "disabled", "关闭", "禁用", "不使用", "不用", "否"}:
        return True, False
    return False, value


def _coerce_value(value: Any, expected_type: str) -> tuple[bool, Any]:
    if expected_type == "boolean":
        return _coerce_bool(value)
    if expected_type == "integer":
        if isinstance(value, bool):
            return False, value
        try:
            converted = int(value)
        except (TypeError, ValueError):
            return False, value
        return True, converted
    if expected_type == "number":
        if isinstance(value, bool):
            return False, value
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return False, value
        return True, converted
    if expected_type == "string":
        if isinstance(value, (dict, list)):
            return False, value
        return True, str(value)
    return False, value


def apply_payload_patch(
    skill_id: str,
    payload: dict[str, Any],
    payload_patch: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(payload)
    patch = payload_patch if isinstance(payload_patch, dict) else {}
    docs_by_field = {str(doc.get("field")): doc for doc in load_parameter_docs(skill_id)}
    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    for raw_key, raw_value in patch.items():
        field = str(raw_key)
        doc = docs_by_field.get(field)
        if field not in merged:
            rejected[field] = "unknown field"
            continue
        if not doc:
            rejected[field] = "field is not documented"
            continue
        if not bool(doc.get("llm_editable")):
            rejected[field] = "field is not llm editable"
            continue
        ok, value = _coerce_value(raw_value, str(doc.get("type") or ""))
        if not ok:
            rejected[field] = f"type mismatch: expected {doc.get('type')}"
            continue
        merged[field] = value
        applied[field] = value

    return merged, {"applied": applied, "rejected": rejected}
