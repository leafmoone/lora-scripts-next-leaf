#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "sd-models"
QWEN3_CONFIG_DIR = REPO / "scripts/dev/configs/qwen3_06b"
QWEN3_TOKENIZER_SOURCE = Path("/.autodl/Qwen/Qwen3-4B")
T5_CONFIG_DIR = REPO / "scripts/dev/configs/t5_old"
T5_TOKENIZER_SOURCES = (
    Path("/root/models/t5_old"),
    Path("/root/models/t5-v1_1-xxl"),
    Path("/.autodl/black-forest-labs/FLUX.1-dev/tokenizer_2"),
    Path("/.autodl/google/t5-v1_1-xxl"),
)
SAFE_SAMPLE_POSITIVE = (
    "1girl, solo, smile, japanese clothes, kimono, blue eyes, closed mouth, upper body, "
    "looking at viewer, hair ornament, long hair, yellow kimono, black hair, anime coloring, "
    "yukata, choker, split mouth, side ponytail, bow, brown hair"
)
SAFE_SAMPLE_NEGATIVE = (
    "nsfw, explicit, sexual content, nude, naked, nipples, areola, genitals, "
    "cleavage, breasts, ass, buttocks, thighs, underwear, lingerie, bikini, swimsuit, "
    "erotic, suggestive, lewd, spread legs, close-up body, transparent clothes, "
    "worst quality, low quality, score_1, score_2, score_3, artist name, jpeg artifacts"
)

QWEN3_06B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

LINKS: dict[str, tuple[Path, ...]] = {
    "anima/anima-preview.safetensors": (
        Path("/root/models/anima/anima-preview.safetensors"),
        Path("/.autodl/circlestone-labs/Anima/split_files/diffusion_models/anima-preview.safetensors"),
    ),
    "anima/anima-preview2.safetensors": (
        Path("/root/models/anima/anima-preview2.safetensors"),
        Path("/.autodl/circlestone-labs/Anima/split_files/diffusion_models/anima-preview2.safetensors"),
    ),
    "anima/anima-preview3-base.safetensors": (
        Path("/root/models/anima/anima-preview3-base.safetensors"),
        Path("/.autodl/circlestone-labs/Anima/split_files/diffusion_models/anima-preview3-base.safetensors"),
    ),
    "anima/qwen_3_06b_base.safetensors": (
        Path("/root/models/anima/qwen_3_06b_base.safetensors"),
        Path("/root/qwen_3_06b_base.safetensors"),
        Path("/.autodl/circlestone-labs/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors"),
    ),
    "anima/qwen_image_vae.safetensors": (
        Path("/root/models/anima/qwen_image_vae.safetensors"),
        Path("/.autodl/circlestone-labs/Anima/split_files/vae/qwen_image_vae.safetensors"),
    ),
    "flux/flux1-dev-fp8.safetensors": (
        Path("/root/models/flux/flux1-dev-fp8.safetensors"),
        Path("/.autodl/23/73/8c/23738c26b548113ea2d392abd91d3fd0"),
    ),
    "flux/ae.safetensors": (
        Path("/root/models/flux/ae.safetensors"),
        Path("/.autodl/black-forest-labs/FLUX.1-dev/ae.safetensors"),
    ),
    "flux/t5xxl_fp8_e4m3fn.safetensors": (
        Path("/root/models/flux/t5xxl_fp8_e4m3fn.safetensors"),
        Path("/.autodl/a3/e7/20/a3e720ed91f439ecc3dfd15e56b137bc"),
    ),
    "flux/clip_l.safetensors": (
        Path("/root/models/flux/clip_l.safetensors"),
        Path("/.autodl/comfyanonymous/flux_text_encoders/clip_l.safetensors"),
    ),
    "sdxl/eps/ChenkinNoob-XL-V0.5.safetensors": (
        Path("/root/models/sdxl/eps/ChenkinNoob-XL-V0.5.safetensors"),
        Path("/.autodl/5c/18/f8/5c18f83c804c06e10e24c191b422f02d"),
    ),
    "sdxl/eps/ChenkinNoob-XL-V0.2.safetensors": (
        Path("/root/models/sdxl/eps/ChenkinNoob-XL-V0.2.safetensors"),
        Path("/.autodl/ChenkinNoob/ChenkinNoob-XL-V0.2/ChenkinNoob-XL-V0.2.safetensors"),
    ),
    "sdxl/rectified_flow/ChenkinNoob-XL-v0.3-Rectified-Flow.safetensors": (
        Path("/root/models/sdxl/rectified_flow/ChenkinNoob-XL-v0.3-Rectified-Flow.safetensors"),
        Path("/.autodl/ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow/ChenkinNoob-XL-v0.3-Rectified-Flow.safetensors"),
    ),
    "sdxl/eps/illustriousXL_v01.safetensors": (
        Path("/root/models/sdxl/eps/illustriousXL_v01.safetensors"),
        Path("/.autodl/a8/94/d5/a894d5ef78c0b284d0ec1e22c4bb56dc"),
    ),
    "sdxl/v_prediction/noobaiXLNAIXL_vPred10Version.safetensors": (
        Path("/root/models/sdxl/v_prediction/noobaiXLNAIXL_vPred10Version.safetensors"),
        Path("/.autodl/a0/81/5e/a0815ef81f4a91a830dcc35dc8e06ac1"),
    ),
}

SCHEMA_DEFAULTS = {
    REPO / "mikazuki/schema/sd3-lora.ts": {
        "pretrained_model_name_or_path": "./sd-models/anima/anima-preview3-base.safetensors",
        "vae": "./sd-models/anima/qwen_image_vae.safetensors",
        "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
        "positive_prompts": SAFE_SAMPLE_POSITIVE,
        "negative_prompts": SAFE_SAMPLE_NEGATIVE,
    },
    REPO / "mikazuki/schema/flux-lora.ts": {
        "pretrained_model_name_or_path": "./sd-models/flux/flux1-dev-fp8.safetensors",
        "ae": "./sd-models/flux/ae.safetensors",
        "clip_l": "./sd-models/flux/clip_l.safetensors",
        "t5xxl": "./sd-models/flux/t5xxl_fp8_e4m3fn.safetensors",
    },
    REPO / "mikazuki/schema/lora-master.ts": {
        "pretrained_model_name_or_path": "./sd-models/sdxl/eps/ChenkinNoob-XL-V0.5.safetensors",
    },
}


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_links() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, candidates in LINKS.items():
        target = MODEL_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = first_existing(candidates)
        if source is None:
            print(f"missing source model for {target}: {', '.join(str(p) for p in candidates)}")
            continue

        if target.is_symlink():
            if Path(os.readlink(target)) == source:
                continue
            target.unlink()
        elif target.exists():
            print(f"skip existing non-symlink: {target}")
            continue

        target.symlink_to(source)
        print(f"linked {target} -> {source}")


def ensure_qwen3_config() -> None:
    QWEN3_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config_file = QWEN3_CONFIG_DIR / "config.json"
    wanted = json.dumps(QWEN3_06B_CONFIG, ensure_ascii=False, indent=2) + "\n"
    if not config_file.exists() or config_file.read_text(encoding="utf-8") != wanted:
        config_file.write_text(wanted, encoding="utf-8")
        print(f"patched Qwen3 0.6B config: {config_file}")

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        source = QWEN3_TOKENIZER_SOURCE / name
        target = QWEN3_CONFIG_DIR / name
        if target.exists():
            continue
        if not source.exists():
            print(f"missing Qwen3 tokenizer source: {source}")
            continue
        shutil.copy2(source, target)
        print(f"copied Qwen3 tokenizer file: {target}")

    for optional_name in ("generation_config.json", "configuration.json"):
        source = QWEN3_TOKENIZER_SOURCE / optional_name
        target = QWEN3_CONFIG_DIR / optional_name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            print(f"copied Qwen3 optional config file: {target}")


def ensure_t5_config() -> None:
    T5_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = first_existing(
        source for source in T5_TOKENIZER_SOURCES if (source / "spiece.model").exists()
    )
    if source_dir is None:
        print(f"missing T5 tokenizer source: {', '.join(str(p) for p in T5_TOKENIZER_SOURCES)}")
        return

    for name in ("spiece.model", "tokenizer.json"):
        source = source_dir / name
        target = T5_CONFIG_DIR / name
        if target.exists():
            continue
        if not source.exists():
            print(f"missing T5 tokenizer file: {source}")
            continue
        shutil.copy2(source, target)
        print(f"copied T5 tokenizer file: {target}")

    for optional_name in ("tokenizer_config.json", "special_tokens_map.json"):
        source = source_dir / optional_name
        target = T5_CONFIG_DIR / optional_name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            print(f"copied T5 optional tokenizer file: {target}")


def set_schema_default(line: str, field: str, value: str) -> str:
    if not line.lstrip().startswith(f"{field}:"):
        return line

    default_expr = f'.default("{value}")'
    if ".default(" in line:
        return re.sub(r'\.default\(".*?"\)', default_expr, line)

    if ".description(" in line:
        return line.replace(".description(", f"{default_expr}.description(", 1)

    return line.rstrip() + f"{default_expr}\n"


def ensure_schema_defaults(schema: Path, defaults: dict[str, str]) -> None:
    text = schema.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = False
    patched = []

    for line in lines:
        new_line = line
        for field, value in defaults.items():
            new_line = set_schema_default(new_line, field, value)
        if new_line != line:
            changed = True
        patched.append(new_line)

    if changed:
        schema.write_text("".join(patched), encoding="utf-8")
        print(f"patched schema defaults: {schema}")
    else:
        print(f"schema defaults already patched: {schema}")


def ensure_xformers_default() -> None:
    """If xformers is not installed, flip schema defaults so the WebUI doesn't ask for it.

    `mikazuki/schema/shared.ts` ships with `xformers: default(true)`. When the conda env lacks
    xformers (e.g. RTX 5090 / cu128 wheels not yet stable), kicking off any training will crash
    with `ImportError: No module named 'xformers'`. We patch the defaults to prefer SDPA instead.
    """
    schema = REPO / "mikazuki/schema/shared.ts"
    if not schema.exists():
        return

    has_xformers = importlib.util.find_spec("xformers") is not None
    text = schema.read_text(encoding="utf-8")

    if has_xformers:
        new_text = re.sub(
            r'xformers: Schema\.boolean\(\)(?:\.default\([^)]*\))?\.description\(',
            'xformers: Schema.boolean().default(true).description(',
            text,
        )
        new_text = re.sub(
            r'sdpa: Schema\.boolean\(\)(?:\.default\([^)]*\))?\.description\(',
            'sdpa: Schema.boolean().description(',
            new_text,
        )
    else:
        new_text = re.sub(
            r'xformers: Schema\.boolean\(\)(?:\.default\([^)]*\))?\.description\(',
            'xformers: Schema.boolean().default(false).description(',
            text,
        )
        new_text = re.sub(
            r'sdpa: Schema\.boolean\(\)(?:\.default\([^)]*\))?\.description\(',
            'sdpa: Schema.boolean().default(true).description(',
            new_text,
        )

    if new_text != text:
        schema.write_text(new_text, encoding="utf-8")
        state = "available" if has_xformers else "missing"
        print(f"patched xformers schema defaults ({state}): {schema}")
    else:
        state = "available" if has_xformers else "missing"
        print(f"xformers schema defaults already aligned ({state}): {schema}")


def ensure_anima_schema_extras() -> None:
    schema = REPO / "mikazuki/schema/sd3-lora.ts"
    text = schema.read_text(encoding="utf-8")
    patched = text

    sample_at_first = (
        '                sample_at_first: Schema.boolean().default(true).description("训练开始前生成 step 0 预览图，'
        '用作未训练基线对照。建议开启"),'
    )
    patched = re.sub(
        r'\n\s*sample_at_first: Schema\.boolean\(\)\.default\(true\)\.description\("[^"]*"\),',
        "",
        patched,
    )
    if "sample_at_first:" not in patched:
        patched = patched.replace(
            '                sample_scheduler: Schema.union(["simple"]).default("simple").description("Anima 预览调度器"),',
            '                sample_scheduler: Schema.union(["simple"]).default("simple").description("Anima 预览调度器"),\n'
            + sample_at_first,
        )

    if patched != text:
        schema.write_text(patched, encoding="utf-8")
        print(f"patched Anima schema extras: {schema}")
    else:
        print(f"Anima schema extras already patched: {schema}")


def ensure_api_sample_defaults() -> None:
    api = REPO / "mikazuki/app/api.py"
    text = api.read_text(encoding="utf-8")

    def py_string_constant(name: str, value: str, width: int = 88) -> str:
        chunks = [value[i:i + width] for i in range(0, len(value), width)]
        lines = [f"{name} = ("]
        lines.extend(f'    "{chunk}"' for chunk in chunks)
        lines.append(")")
        return "\n".join(lines) + "\n"

    positive = py_string_constant("ANIMA_DEFAULT_SAMPLE_POSITIVE", SAFE_SAMPLE_POSITIVE)
    negative = py_string_constant("ANIMA_DEFAULT_SAMPLE_NEGATIVE", SAFE_SAMPLE_NEGATIVE)
    patched = re.sub(
        r'ANIMA_DEFAULT_SAMPLE_POSITIVE = \(\n(?:    ".*"\n)+\)\n',
        positive,
        text,
        count=1,
    )
    patched = re.sub(
        r'ANIMA_DEFAULT_SAMPLE_NEGATIVE = \(\n(?:    ".*"\n)+\)\n',
        negative,
        patched,
        count=1,
    )
    patched = re.sub(
        r'ANIMA_DEFAULT_UNET_LR = "5e-5"\n',
        'ANIMA_DEFAULT_UNET_LR = 5e-5\n',
        patched,
        count=1,
    )
    patched = patched.replace(
        'use_anima_defaults = model_train_type in ANIMA_TRAIN_TYPES and config.get("enable_preview") is True',
        'use_anima_defaults = model_train_type in ANIMA_TRAIN_TYPES and is_preview_enabled(config)',
    )
    patched = patched.replace(
        '    if config.get("enable_preview") is True:\n        config["sample_at_first"] = True\n',
        '    if is_preview_enabled(config) or config.get("sample_prompts"):\n        config["sample_at_first"] = True\n',
    )
    patched = patched.replace(
        '    if str(config.get("unet_lr", "")).strip() in ANIMA_LEGACY_UNET_LR:\n'
        '        config["unet_lr"] = ANIMA_DEFAULT_UNET_LR\n\n'
        '    if is_preview_enabled(config):\n',
        '    if str(config.get("unet_lr", "")).strip() in ANIMA_LEGACY_UNET_LR:\n'
        '        config["unet_lr"] = ANIMA_DEFAULT_UNET_LR\n'
        '    elif isinstance(config.get("unet_lr"), str):\n'
        '        config["unet_lr"] = float(config["unet_lr"])\n\n'
        '    if is_preview_enabled(config) or config.get("sample_prompts"):\n',
    )
    patched = patched.replace(
        '    if is_preview_enabled(config):\n        config["sample_at_first"] = True\n',
        '    if is_preview_enabled(config) or config.get("sample_prompts"):\n        config["sample_at_first"] = True\n',
    )
    if "def is_preview_enabled" not in patched:
        patched = patched.replace(
            "\n\ndef apply_anima_training_defaults(config: dict, model_train_type: str):",
            '\n\ndef is_preview_enabled(config: dict) -> bool:\n'
            '    return config.get("enable_preview") in (True, "true", "True", "1", 1)\n'
            '\n\ndef apply_anima_training_defaults(config: dict, model_train_type: str):',
            1,
        )
    if "ANIMA_DEFAULT_UNET_LR" not in patched:
        patched = patched.replace(
            "avaliable_scripts = [",
            'ANIMA_DEFAULT_UNET_LR = 5e-5\n'
            'ANIMA_LEGACY_UNET_LR = {"0.0001", "1e-4", "1E-4"}\n\n'
            "avaliable_scripts = [",
            1,
        )
    if "def apply_anima_training_defaults" not in patched:
        patched = patched.replace(
            "\n\n@router.post(\"/run\")",
            '\n\ndef is_preview_enabled(config: dict) -> bool:\n'
            '    return config.get("enable_preview") in (True, "true", "True", "1", 1)\n\n\n'
            'def apply_anima_training_defaults(config: dict, model_train_type: str):\n'
            '    if model_train_type not in ANIMA_TRAIN_TYPES:\n'
            '        return\n\n'
            '    if str(config.get("unet_lr", "")).strip() in ANIMA_LEGACY_UNET_LR:\n'
            '        config["unet_lr"] = ANIMA_DEFAULT_UNET_LR\n'
            '    elif isinstance(config.get("unet_lr"), str):\n'
            '        config["unet_lr"] = float(config["unet_lr"])\n\n'
            '    if is_preview_enabled(config) or config.get("sample_prompts"):\n'
            '        config["sample_at_first"] = True\n'
            '\n\n@router.post("/run")',
            1,
        )
    if "apply_anima_training_defaults(config, model_train_type)\n\n    with open(toml_file" not in patched:
        patched = patched.replace(
            '    with open(toml_file, "w", encoding="utf-8") as f:\n',
            '    apply_anima_training_defaults(config, model_train_type)\n\n'
            '    with open(toml_file, "w", encoding="utf-8") as f:\n',
            1,
        )
    if "apply_anima_training_defaults(config, model_train_type)" not in patched:
        patched = patched.replace(
            "    apply_sdxl_prediction_type(config, model_train_type)\n",
            "    apply_sdxl_prediction_type(config, model_train_type)\n"
            "    apply_anima_training_defaults(config, model_train_type)\n",
            1,
        )
    if patched != text:
        api.write_text(patched, encoding="utf-8")
        print(f"patched Anima API sample defaults: {api}")
    else:
        print(f"Anima API sample defaults already patched: {api}")


def main() -> None:
    ensure_links()
    ensure_qwen3_config()
    ensure_t5_config()
    ensure_api_sample_defaults()
    for schema, defaults in SCHEMA_DEFAULTS.items():
        ensure_schema_defaults(schema, defaults)
    ensure_anima_schema_extras()
    ensure_xformers_default()


if __name__ == "__main__":
    main()
