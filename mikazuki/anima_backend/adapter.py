from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_FIELDS = {
    "pretrained_model_name_or_path",
    "vae",
    "qwen3",
    "llm_adapter_path",
    "t5_tokenizer_path",
    "resume",
    "qwen3_max_token_length",
    "t5_max_token_length",
    "timestep_sampling",
    "sigmoid_scale",
    "discrete_flow_shift",
    "weighting_scheme",
    "logit_mean",
    "logit_std",
    "mode_scale",
    "attn_mode",
    "split_attn",
    "vae_chunk_size",
    "vae_disable_cache",
    "unsloth_offload_checkpointing",
    "train_data_dir",
    "reg_data_dir",
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "output_dir",
    "output_name",
    "save_model_as",
    "save_precision",
    "save_every_n_epochs",
    "max_train_epochs",
    "max_train_steps",
    "train_batch_size",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "optimizer_type",
    "optimizer_args",
    "lr_scheduler",
    "lr_warmup_steps",
    "network_module",
    "network_weights",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "dim_from_weights",
    "scale_weight_norms",
    "train_norm",
    "lycoris_algo",
    "lokr_factor",
    "dropout",
    "pissa_init",
    "pissa_method",
    "pissa_niter",
    "pissa_oversample",
    "pissa_apply_conv2d",
    "pissa_export_mode",
    "sample_prompts",
    "sample_at_first",
    "sample_every_n_epochs",
    "caption_extension",
    "shuffle_caption",
    "keep_tokens",
    "caption_tag_dropout_rate",
    "prefer_json_caption",
    "noise_offset",
    "multires_noise_iterations",
    "multires_noise_discount",
    "fp8_base",
    "fp8_base_unet",
    "cache_latents",
    "cache_latents_to_disk",
    "cache_text_encoder_outputs",
    "cache_text_encoder_outputs_to_disk",
    "persistent_data_loader_workers",
    "max_data_loader_n_workers",
    "text_encoder_batch_size",
    "disable_mmap_load_safetensors",
    "blocks_to_swap",
    "cpu_offload_checkpointing",
    "mixed_precision",
    "seed",
    "logging_dir",
    "log_with",
}

UI_ONLY_FIELDS = {
    "model_train_type",
    "enable_preview",
    "positive_prompts",
    "negative_prompts",
    "sample_width",
    "sample_height",
    "sample_cfg",
    "sample_seed",
    "sample_steps",
    "sample_sampler",
    "sample_scheduler",
    "randomly_choice_prompt",
    "prompt_file",
    "enable_debug_options",
    "json_caption_hint",
}


def adapt_anima_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = deepcopy(config)
    adapted: dict[str, Any] = {}
    warnings: list[str] = []

    custom_network_args = source.pop("network_args_custom", None)
    if custom_network_args:
        source["network_args"] = custom_network_args

    for key, value in source.items():
        if key in UI_ONLY_FIELDS:
            continue
        if key in SUPPORTED_FIELDS:
            # skip empty string for optional fields that expect None or a real value
            if key == "attn_mode" and value in ("", None):
                continue
            adapted[key] = value
            continue
        if key.startswith("anima_"):
            warnings.append(f"Unsupported Anima field ignored: {key}")
            continue
        warnings.append(f"Unknown field passed through to sd-scripts: {key}")
        adapted[key] = value

    return adapted, warnings
