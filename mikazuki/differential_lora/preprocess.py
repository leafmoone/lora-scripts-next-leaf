"""
Differential LoRA 数据集预处理模块

提供图片配对、单图临时数据集构建、提示词处理等功能。
"""

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def pair_images(folder_a: str, folder_b: str) -> list[Tuple[str, str, str]]:
    """
    匹配两个文件夹中同名的图片文件。

    Args:
        folder_a: 原风格图片目录
        folder_b: 目标风格图片目录

    Returns:
        [(filename, img_a_path, img_b_path), ...]
    """
    pairs = []
    folder_a_path = Path(folder_a)
    folder_b_path = Path(folder_b)

    if not folder_a_path.is_dir():
        raise FileNotFoundError(f"folder_a 不存在: {folder_a}")
    if not folder_b_path.is_dir():
        raise FileNotFoundError(f"folder_b 不存在: {folder_b}")

    for entry in sorted(folder_a_path.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue

        fname = entry.name
        img_b = folder_b_path / fname
        if img_b.is_file():
            pairs.append((fname, str(entry), str(img_b)))

    return pairs


def read_base_tags(tag_dir: str, img_filename: str) -> str:
    """
    读取与图片同名的 .txt 标签文件。

    若标签文件不存在，回退使用文件名（不含扩展名）作为标签。

    Args:
        tag_dir: 标签文件目录
        img_filename: 图片文件名（如 "cat.jpg"）

    Returns:
        标签文本（逗号分隔的标签字符串）
    """
    base_name = os.path.splitext(img_filename)[0]
    tag_file = os.path.join(tag_dir, f"{base_name}.txt")

    if os.path.isfile(tag_file):
        with open(tag_file, "r", encoding="utf-8") as f:
            # 合并所有行，去除首尾空白
            tags = " ".join(line.strip() for line in f if line.strip())
            return tags.strip()
    else:
        return base_name


def remove_tokens(tags: str, to_remove: str) -> str:
    """
    从逗号分隔的标签串中删除指定 token。

    Args:
        tags: 原始标签串，如 "1girl, boots, blue sky"
        to_remove: 要删除的 token 列表（逗号分隔），如 "boots,gloves"

    Returns:
        清理后的标签串
    """
    if not to_remove or not tags:
        return tags

    remove_set = {t.strip().lower() for t in to_remove.split(",") if t.strip()}
    if not remove_set:
        return tags

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    filtered = [t for t in tag_list if t.lower() not in remove_set]
    return ", ".join(filtered)


def build_step2_prompt(
    base_tags: str,
    trigger_word: str,
    remove_tokens_str: str = "",
) -> str:
    """
    构建 Step 2 的训练提示词。

    规则: TRIGGER_WORD + (基础标签 - 要删除的 token)

    Args:
        base_tags: Step 1 使用的基础标签
        trigger_word: 差分触发词
        remove_tokens_str: 要从基础标签中删除的 token（逗号分隔）

    Returns:
        Step 2 的训练提示词
    """
    cleaned = remove_tokens(base_tags, remove_tokens_str)
    if cleaned:
        return f"{trigger_word}, {cleaned}"
    else:
        return trigger_word


def create_single_image_dataset(
    image_path: str,
    prompt: str,
    output_dir: Optional[str] = None,
    safe_name: Optional[str] = None,
    repeat: int = 1,
) -> str:
    """
    创建单图训练数据集（符合 Kohya Dataset 格式）。

    生成结构:
      <output_dir>/
      └── <repeat>_<safe_name>/
          ├── <image_filename>
          └── <image_filename_no_ext>.txt

    Args:
        repeat: Kohya repeats——文件夹名前缀数字，如 1000_xxx/ 表示每 epoch 重复 1000 次。

    同时生成 metadata.csv（兼容 DiffSynth 格式）:
      image,prompt
      <image_filename>,<prompt>

    Args:
        image_path: 图片路径
        prompt: 训练提示词
        output_dir: 数据集输出目录（默认创建临时目录）
        safe_name: 安全目录名

    Returns:
        dataset_dir 路径
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="difflora_ds_")

    img_filename = os.path.basename(image_path)
    base_name = os.path.splitext(img_filename)[0]

    if safe_name is None:
        safe_name = base_name

    # 创建 Kohya 格式的 repeats 子目录
    repeat_dir = os.path.join(output_dir, f"{repeat}_{safe_name}")
    os.makedirs(repeat_dir, exist_ok=True)

    # 复制图片
    dst_img = os.path.join(repeat_dir, img_filename)
    shutil.copy2(image_path, dst_img)

    # 写入标签 .txt
    txt_path = os.path.join(repeat_dir, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    # 写入 metadata.csv（兼容 DiffSynth 格式）
    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prompt"])
        writer.writerow([img_filename, prompt])

    return output_dir
