"""
Differential LoRA TOML 配置适配器

将前端 UI 配置转换为 Kohya sd-scripts 兼容的 TOML 配置，
用于 Step 1 和 Step 2 的训练。
"""

import ast
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional


def _try_parse_value(v: str) -> Any:
    """Try to parse a string as Python literal (int/float/bool/list); fallback to string."""
    v = v.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    return v


def build_step1_toml(
    config: dict,
    dataset_dir: str,
    output_dir: str,
) -> dict:
    """
    构建 Step 1 的 TOML 配置。

    Step 1: 在图A上训练标准 LoRA，使 LoRA 过拟合到图A的风格。

    Args:
        config: 前端传入的完整配置
        dataset_dir: 临时数据集目录（单图 + .txt caption）
        output_dir: LoRA 输出目录

    Returns:
        Kohya-compatible TOML 字典
    """
    toml: dict[str, Any] = {}

    # ── 基础模型 ──
    toml["pretrained_model_name_or_path"] = config.get(
        "pretrained_model_name_or_path",
        "./sd-models/anima/anima-base-v1.0.safetensors",
    )

    # Anima 特有模型路径
    for anm_key in ("vae", "qwen3"):
        if config.get(anm_key):
            toml[anm_key] = config[anm_key]

    # ── 数据集 ──
    toml["train_data_dir"] = dataset_dir
    toml["resolution"] = config.get("resolution", "1024,1024")
    toml["enable_bucket"] = config.get("enable_bucket", True)
    toml["caption_extension"] = config.get("caption_extension", ".txt")

    # ── 输出 ──
    toml["output_dir"] = output_dir
    toml["output_name"] = config.get("output_name", "differential_lora_step1")
    toml["save_precision"] = config.get("save_precision", "fp16")
    toml["save_every_n_epochs"] = config.get("num_epochs", 5)
    toml["save_model_as"] = "safetensors"

    # ── LoRA 网络（Anima DiT 必须用 lora_anima，networks.lora 是 SD 专用且导致 empty param list）──
    toml["network_module"] = "networks.lora_anima"
    lora_rank = config.get("lora_rank", 32)
    toml["network_dim"] = lora_rank
    toml["network_alpha"] = lora_rank  # 1:1 ratio for overfitting

    conv_dim = config.get("conv_dim", 0)
    conv_alpha = config.get("conv_alpha", 1)
    network_args = [r"exclude_patterns=[r'.*llm_adapter.*']"]
    if conv_dim:
        network_args.append(f"conv_dim={conv_dim}")
        network_args.append(f"conv_alpha={conv_alpha}")
    if config.get("lora_exclude_modules"):
        network_args.append(f"exclude_modules={config['lora_exclude_modules']}")
    if network_args:
        toml["network_args"] = network_args

    # ── 训练超参 ──
    toml["learning_rate"] = float(config.get("learning_rate", "1e-4"))
    toml["train_batch_size"] = 1
    toml["max_train_epochs"] = config.get("num_epochs", 5)

    toml["optimizer_type"] = config.get("optimizer_type", "AdamW8bit")
    toml["lr_scheduler"] = config.get("lr_scheduler", "constant")
    toml["lr_warmup_steps"] = config.get("lr_warmup_steps", 0)
    toml["mixed_precision"] = config.get("mixed_precision", "bf16")

    # ── 梯度相关 ──
    toml["gradient_accumulation_steps"] = config.get("gradient_accumulation_steps", 1)
    toml["gradient_checkpointing"] = config.get("gradient_checkpointing", False)

    # ── 日志 ──
    toml["logging_dir"] = config.get("logging_dir", "./logs/differential_lora")

    # ── 采样 ──
    if config.get("enable_sample"):
        sample_every = config.get("sample_every", 10000)
        if sample_every > 0:
            toml["sample_every_n_steps"] = sample_every
        if config.get("sample_prompts"):
            toml["sample_prompts"] = config["sample_prompts"]
        toml["sample_at_first"] = config.get("sample_at_first", False)

    # ── 自定义 TOML 参数（前端 textarea，逐行 key=value）──
    custom_params = config.get("custom_params", "")
    if custom_params:
        for line in custom_params.strip().split("\n"):
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            # Try to parse as literal Python value (int, float, list, str)
            toml[key] = _try_parse_value(value)

    data_enhancement = config.get("data_enhancement", [])
    if isinstance(data_enhancement, str):
        data_enhancement = [data_enhancement]
    if data_enhancement:
        toml["data_enhancement"] = data_enhancement

    # ── 种子与杂项 ──
    toml["seed"] = config.get("seed", 42)

    # Anima 特定参数
    if config.get("attn_mode"):
        toml["attn_mode"] = config["attn_mode"]
    if config.get("discrete_flow_shift"):
        toml["discrete_flow_shift"] = config["discrete_flow_shift"]

    # ── 清理 None/空值 ──
    toml = {k: v for k, v in toml.items() if v is not None and v != ""}

    return toml


def build_step2_toml(
    config: dict,
    dataset_dir: str,
    output_dir: str,
    merged_model_path: str,
) -> dict:
    """
    构建 Step 2 的 TOML 配置。

    Step 2: 在图B+触发词上训练差分 LoRA，以 Step1 融合后模型为底模。

    Args:
        config: 前端传入的完整配置
        dataset_dir: 临时数据集目录（图B + 触发词 prompt）
        output_dir: 差分 LoRA 输出目录
        merged_model_path: Step1 LoRA 融合后的模型路径

    Returns:
        Kohya-compatible TOML 字典
    """
    toml = build_step1_toml(config, dataset_dir, output_dir)

    # 关键: 使用融合了 LoRA1 的模型作为底模
    toml["pretrained_model_name_or_path"] = merged_model_path
    toml["output_name"] = config.get("output_name", "differential_lora_step2")

    # 不需要 preset_lora，因为已融入底模
    toml.pop("preset_lora_path", None)

    return toml


def _build_default_config() -> dict:
    """构建默认 Differential LoRA 配置。"""
    return {
        "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
        "vae": "./sd-models/anima/qwen_image_vae.safetensors",
        "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
        "resolution": "1024,1024",
        "enable_bucket": True,
        "lora_rank": 32,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "dataset_repeat": 1000,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "constant",
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "save_precision": "fp16",
        "seed": 42,
        "logging_dir": "./logs/differential_lora",
        "output_dir": "./models/differential_lora",
    }
