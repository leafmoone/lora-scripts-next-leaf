from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


PayloadFactory = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class AgentSkill:
    id: str
    title: str
    description: str
    trigger_keywords: tuple[str, ...]
    input_schema: dict[str, Any]
    default_payload: PayloadFactory
    plan_type: str
    action_kind: str
    side_effect_level: str
    summary: tuple[str, ...]
    assistant_message: str


def _tagger_payload(path: str) -> dict[str, Any]:
    return {
        "input_dir": path,
        "output_dir": path,
        "mode": "smart",
        "model": "wd-eva02-large-tagger-v3",
        "threshold": 0.35,
        "char_threshold": 0.85,
        "max_tags": 0,
        "blacklist": "",
        "additional_tags": "",
        "exclude_tags": "",
        "purpose": "character",
        "use_vlm": True,
        "use_cpu": False,
        "save_captions": True,
        "recursive": False,
    }


def _differential_payload(path: str) -> dict[str, Any]:
    return {
        "model_train_type": "differential-lora",
        "training_method": "differential-lora",
        "folder_a": path,
        "folder_b": "",
        "output_dir": "./models/differential_lora",
        "tag_dir": path,
        "trigger_word": "Character_Splitting",
        "remove_tokens": "",
        "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
        "vae": "./sd-models/anima/qwen_image_vae.safetensors",
        "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
        "attn_mode": "",
        "discrete_flow_shift": 3.0,
        "lora_rank": 32,
        "conv_dim": 0,
        "conv_alpha": 1,
        "lora_exclude_modules": "",
        "learning_rate": "1e-4",
        "num_epochs": 5,
        "dataset_repeat": 1000,
        "resolution": "1024,1024",
        "enable_bucket": True,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "mixed_precision": "bf16",
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "constant",
        "max_grad_norm": 1.0,
        "seed": 42,
        "save_precision": "fp16",
        "auto_tag": False,
        "tagger_mode": "smart",
        "tagger_use_vlm": True,
        "tagger_use_cpu": False,
        "tagger_recursive": False,
        "tagger_model": "wd-eva02-large-tagger-v3",
        "tagger_threshold": 0.35,
        "tagger_char_threshold": 0.85,
        "tagger_max_tags": 0,
        "tagger_blacklist": "",
        "tagger_purpose": "character",
        "logging_dir": "./logs/differential_lora",
        "sample_every": 10000,
        "sample_prompts": "",
        "sample_sampler": "euler",
        "noise_offset": 0,
        "data_enhancement": "",
        "postprocess_comfyui": True,
        "postprocess_svd": True,
        "keep_temp": False,
    }


def _anima_payload(path: str) -> dict[str, Any]:
    return {
        "model_train_type": "anima-lora",
        "lora_type": "lora",
        "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
        "vae": "./sd-models/anima/qwen_image_vae.safetensors",
        "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
        "llm_adapter_path": "",
        "t5_tokenizer_path": "",
        "resume": "",
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "timestep_sampling": "shift",
        "sigmoid_scale": 1.0,
        "discrete_flow_shift": 3.0,
        "weighting_scheme": "uniform",
        "logit_mean": "",
        "logit_std": "",
        "mode_scale": "",
        "attn_mode": "",
        "split_attn": False,
        "vae_chunk_size": "",
        "vae_disable_cache": False,
        "unsloth_offload_checkpointing": False,
        "train_data_dir": path,
        "reg_data_dir": "",
        "prior_loss_weight": 1.0,
        "resolution": "1024,1024",
        "enable_bucket": True,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": True,
        "output_name": "aki",
        "output_dir": "./output",
        "save_model_as": "safetensors",
        "save_precision": "fp16",
        "save_every_n_epochs": 2,
        "save_every_n_steps": "",
        "save_state": False,
        "max_train_epochs": 10,
        "train_batch_size": 1,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 1,
        "network_train_unet_only": True,
        "network_train_text_encoder_only": False,
        "learning_rate": "1e-4",
        "unet_lr": "1e-4",
        "text_encoder_lr": "1e-5",
        "lr_scheduler": "cosine_with_restarts",
        "lr_warmup_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "loss_type": "",
        "optimizer_type": "AdamW8bit",
        "min_snr_gamma": "",
        "optimizer_args_custom": [],
        "network_weights": "",
        "network_module": "networks.lora_anima",
        "network_dim": 16,
        "network_alpha": 16,
        "dim_from_weights": False,
        "scale_weight_norms": "",
        "train_norm": False,
        "conv_dim": "",
        "conv_alpha": "",
        "network_args_custom": [],
        "enable_base_weight": False,
        "network_dropout": 0,
        "pissa_init": False,
        "lycoris_algo": "",
        "lokr_factor": "",
        "dropout": "",
        "enable_preview": False,
        "positive_prompts": "",
        "negative_prompts": "",
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_cfg": 4.5,
        "sample_seed": 42,
        "sample_steps": 40,
        "sample_sampler": "euler",
        "sample_scheduler": "simple",
        "sample_at_first": True,
        "sample_every_n_epochs": 2,
        "log_with": "tensorboard",
        "log_prefix": "",
        "log_tracker_name": "",
        "logging_dir": "./logs",
        "caption_extension": ".txt",
        "shuffle_caption": False,
        "weighted_captions": False,
        "keep_tokens": 0,
        "keep_tokens_separator": "",
        "max_token_length": 255,
        "caption_dropout_rate": "",
        "caption_dropout_every_n_epochs": "",
        "caption_tag_dropout_rate": "",
        "mixed_precision": "bf16",
        "full_fp16": False,
        "full_bf16": False,
        "no_half_vae": False,
        "xformers": False,
        "sdpa": True,
        "lowram": False,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": False,
        "cache_text_encoder_outputs_to_disk": False,
        "persistent_data_loader_workers": True,
        "vae_batch_size": "",
        "seed": 1337,
    }


def _anima_fast_payload(path: str) -> dict[str, Any]:
    return {
        "model_train_type": "anima-lora-fast",
        "lora_type": "lora",
        "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
        "vae": "./sd-models/anima/qwen_image_vae.safetensors",
        "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
        "resume": "",
        "method": "lora",
        "methods_subdir": "gui-methods",
        "qwen3_max_token_length": 512,
        "timestep_sampling": "shift",
        "discrete_flow_shift": 3.0,
        "attn_mode": "",
        "torch_compile": True,
        "static_token_count": 4096,
        "compile_mode": "blocks",
        "train_data_dir": path,
        "source_image_dir": path,
        "resized_image_dir": "",
        "lora_cache_dir": "",
        "cache_latents": False,
        "cache_latents_to_disk": False,
        "cache_text_encoder_outputs": False,
        "cache_text_encoder_outputs_to_disk": False,
        "skip_cache_check": False,
        "caption_extension": ".txt",
        "resolution": "1024,1024",
        "enable_bucket": True,
        "output_name": "aki",
        "output_dir": "./output/anima_fast",
        "save_model_as": "safetensors",
        "save_precision": "fp16",
        "save_every_n_epochs": 2,
        "save_every_n_steps": "",
        "save_state": False,
        "logging_dir": "./logs/anima_fast",
        "progress_jsonl": "",
        "max_train_epochs": 1,
        "max_train_steps": "",
        "train_batch_size": 1,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 1,
        "seed": 42,
        "learning_rate": "1e-4",
        "unet_lr": "1e-4",
        "text_encoder_lr": "1e-5",
        "lr_scheduler": "cosine_with_restarts",
        "lr_warmup_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "loss_type": "",
        "optimizer_type": "AdamW8bit",
        "min_snr_gamma": "",
        "optimizer_args_custom": [],
        "enable_preview": False,
        "randomly_choice_prompt": False,
        "prompt_file": "",
        "positive_prompts": "",
        "negative_prompts": "",
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_cfg": 4.5,
        "sample_seed": 42,
        "sample_steps": 40,
        "sample_sampler": "euler",
        "sample_at_first": False,
        "sample_every_n_epochs": 2,
        "network_module": "networks.lora_anima",
        "network_weights": "",
        "network_dim": 16,
        "network_alpha": 16,
        "network_dropout": 0,
        "network_args_custom": [],
    }


PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute image folder path parsed from the user message.",
        },
    },
    "required": ["path"],
}


BUILTIN_SKILLS = (
    AgentSkill(
        id="tagger_leaf_batch_caption",
        title="图片文件夹打标",
        description="使用 Tag-Edit-Leaf 对图片目录批量生成 caption 文件。",
        trigger_keywords=("tagger", "tag", "打标", "标注"),
        input_schema=PATH_SCHEMA,
        default_payload=_tagger_payload,
        plan_type="tagger",
        action_kind="run_tagger",
        side_effect_level="writes_files",
        summary=("使用 Smart Tag", "写入 .txt caption 文件", "输出 results.json"),
        assistant_message="我已生成打标计划。请检查路径和参数，确认后我再执行。",
    ),
    AgentSkill(
        id="differential_lora_train",
        title="Differential LoRA 训练",
        description="通过差分训练入口启动 Differential LoRA 训练任务。",
        trigger_keywords=("differential", "差分"),
        input_schema=PATH_SCHEMA,
        default_payload=_differential_payload,
        plan_type="differential_lora",
        action_kind="run_differential_lora",
        side_effect_level="gpu_training",
        summary=("需要 folder_a/folder_b 配对目录", "训练会占用 GPU", "输出到 ./models/differential_lora"),
        assistant_message="我已生成 Differential LoRA 训练草案。还需要确认 folder_b 和关键训练参数。",
    ),
    AgentSkill(
        id="anima_fast_train",
        title="Anima Fast 训练",
        description="通过现有 Anima Fast 入口启动快速 LoRA 训练。",
        trigger_keywords=("anima fast", "anima-fast"),
        input_schema=PATH_SCHEMA,
        default_payload=_anima_fast_payload,
        plan_type="anima_fast",
        action_kind="run_anima_fast_train",
        side_effect_level="gpu_training",
        summary=("需要已就绪的 Anima Fast 环境", "训练会占用 GPU", "启动前会走现有预检查"),
        assistant_message="我已生成 Anima Fast 训练草案。请补齐底模路径并确认参数。",
    ),
    AgentSkill(
        id="anima_lora_train",
        title="Anima LoRA 训练",
        description="通过现有训练入口启动 Anima LoRA 训练。",
        trigger_keywords=("anima", "训练"),
        input_schema=PATH_SCHEMA,
        default_payload=_anima_payload,
        plan_type="anima_lora",
        action_kind="run_anima_train",
        side_effect_level="gpu_training",
        summary=("需要图片目录和底模路径", "训练会占用 GPU", "走现有 /api/run 训练入口"),
        assistant_message="我已生成 Anima LoRA 训练草案。请补齐底模路径并确认参数。",
    ),
)


def get_skill(skill_id: str) -> AgentSkill | None:
    for skill in BUILTIN_SKILLS:
        if skill.id == skill_id:
            return skill
    return None


def match_skill(message: str) -> AgentSkill | None:
    lower = message.lower()
    for skill in BUILTIN_SKILLS:
        if any(keyword in lower for keyword in skill.trigger_keywords):
            return skill
    return None


def make_skill_plan(skill_id: str, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    skill = get_skill(skill_id)
    if skill is None:
        raise ValueError(f"unknown agent skill: {skill_id}")

    payload = skill.default_payload(path)
    summary = list(skill.summary)
    plan = {
        "type": skill.plan_type,
        "skill_id": skill.id,
        "title": skill.title,
        "summary": summary,
        "payload": payload,
        "side_effect_level": skill.side_effect_level,
        "input_schema": skill.input_schema,
    }
    action = {
        "kind": skill.action_kind,
        "skill_id": skill.id,
        "title": skill.title,
        "payload": payload,
        "summary": summary,
        "side_effect_level": skill.side_effect_level,
    }
    return plan, action
