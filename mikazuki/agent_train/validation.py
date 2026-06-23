from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def can_execute(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "can_execute": self.can_execute,
        }


EXPECTED_TRAIN_TYPES = {
    "differential_lora_train": "differential-lora",
    "anima_lora_train": "anima-lora",
    "anima_fast_train": "anima-lora-fast",
}


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _require(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if _is_blank(payload.get(key)):
        errors.append(f"{key} 是必填参数。")


def _warn_missing_path(payload: dict[str, Any], key: str, warnings: list[str]) -> None:
    value = str(payload.get(key) or "").strip()
    if not value:
        return
    if value.startswith(("./", "../")):
        return
    if value.startswith("/") and not Path(value).exists():
        warnings.append(f"{key} 路径不存在或当前环境不可访问：{value}")


def _positive_int(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in payload or _is_blank(payload.get(key)):
        return
    try:
        value = int(payload[key])
    except (TypeError, ValueError):
        errors.append(f"{key} 必须是正整数。")
        return
    if value <= 0:
        errors.append(f"{key} 必须大于 0。")


def _positive_float(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in payload or _is_blank(payload.get(key)):
        return
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        errors.append(f"{key} 必须是正数。")
        return
    if value <= 0:
        errors.append(f"{key} 必须大于 0。")


def _validate_common_training(skill_id: str, payload: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    expected = EXPECTED_TRAIN_TYPES.get(skill_id)
    if expected and payload.get("model_train_type") != expected:
        errors.append(f"model_train_type 必须是 {expected}。")

    for key in ("pretrained_model_name_or_path", "vae", "qwen3"):
        _require(payload, key, errors)
        _warn_missing_path(payload, key, warnings)

    _positive_float(payload, "learning_rate", errors)
    _positive_int(payload, "train_batch_size", errors)
    _positive_int(payload, "gradient_accumulation_steps", errors)


def validate_skill_payload(skill_id: str, payload: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if skill_id == "tagger_leaf_batch_caption":
        _require(payload, "input_dir", errors)
        _warn_missing_path(payload, "input_dir", warnings)
        threshold = payload.get("threshold")
        if not _is_blank(threshold):
            try:
                value = float(threshold)
            except (TypeError, ValueError):
                errors.append("threshold 必须是 0 到 1 之间的数字。")
            else:
                if value < 0 or value > 1:
                    errors.append("threshold 必须在 0 到 1 之间。")
        return ValidationResult(errors=errors, warnings=warnings)

    if skill_id == "differential_lora_train":
        for key in ("folder_a", "folder_b"):
            _require(payload, key, errors)
            _warn_missing_path(payload, key, warnings)
        _validate_common_training(skill_id, payload, errors, warnings)
        _positive_int(payload, "lora_rank", errors)
        _positive_int(payload, "num_epochs", errors)
        _positive_int(payload, "dataset_repeat", errors)
        return ValidationResult(errors=errors, warnings=warnings)

    if skill_id in {"anima_lora_train", "anima_fast_train"}:
        _require(payload, "train_data_dir", errors)
        _warn_missing_path(payload, "train_data_dir", warnings)
        _validate_common_training(skill_id, payload, errors, warnings)
        epoch_key = "max_train_epochs"
        _positive_int(payload, epoch_key, errors)
        _positive_int(payload, "network_dim", errors)
        return ValidationResult(errors=errors, warnings=warnings)

    errors.append(f"未知 skill：{skill_id}")
    return ValidationResult(errors=errors, warnings=warnings)
