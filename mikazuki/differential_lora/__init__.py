"""
Differential LoRA 训练模块

提供差分 LoRA（Differential LoRA）训练的完整后端支持：
- 图片配对与数据集预处理
- UI 配置 → Kohya TOML 转换
- 双步训练编排（Step1 过拟合 → 合并 → Step2 差分）
"""

from mikazuki.differential_lora.preprocess import (
    pair_images,
    create_single_image_dataset,
    build_step2_prompt,
)
from mikazuki.differential_lora.adapter import (
    build_step1_toml,
    build_step2_toml,
)
from mikazuki.differential_lora.task_runner import (
    DIFFERENTIAL_TRAIN_TYPE,
    run_differential_lora,
)

__all__ = [
    "pair_images",
    "create_single_image_dataset",
    "build_step2_prompt",
    "build_step1_toml",
    "build_step2_toml",
    "DIFFERENTIAL_TRAIN_TYPE",
    "run_differential_lora",
]
