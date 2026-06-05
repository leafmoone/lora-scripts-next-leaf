"""
Differential LoRA 训练任务编排器

核心逻辑:
  对每组配对图片:
    1. Step 1: 标准 Kohya LoRA 训练 (过拟合到图A)
    2. 合并 LoRA1 到底模 (merge_lora_to_base)
    3. Step 2: 差分 LoRA 训练 (以合并模型为底模, 图B + 触发词)
  后处理:
    - ComfyUI 格式转换
    - SVD 合并所有差分 LoRA
"""

import asyncio
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mikazuki.differential_lora.adapter import build_step1_toml, build_step2_toml
from mikazuki.differential_lora.preprocess import (
    build_step2_prompt,
    create_single_image_dataset,
    pair_images,
    read_base_tags,
)
from mikazuki.log import log
from mikazuki.tasks import tm, TaskStatus

DIFFERENTIAL_TRAIN_TYPE = "differential-lora"

# Paths relative to project root
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
MERGE_TOOL = str(PROJECT_ROOT / "tools" / "merge_lora_to_base.py")
AVERAGE_TOOL = str(PROJECT_ROOT / "tools" / "average_lora.py")
COMFYUI_TOOL = str(PROJECT_ROOT / "tools" / "convert_differential_to_comfyui.py")


def _get_project_python() -> str:
    """获取当前运行使用的 Python 解释器路径。"""
    return sys.executable


def _run_subprocess(cmd: list[str], desc: str = "") -> tuple[int, str, str]:
    """
    同步执行子进程，返回 (returncode, stdout, stderr)。
    """
    log.info(f"[DiffLoRA] 执行: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            log.error(f"[DiffLoRA] {desc} 失败 (exit {proc.returncode}): {proc.stderr[:500]}")
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        log.error(f"[DiffLoRA] {desc} 超时")
        return -1, "", "Timeout"
    except Exception as e:
        log.error(f"[DiffLoRA] {desc} 异常: {e}")
        return -1, "", str(e)


def _run_kohya_training(
    toml_config: dict,
    step_label: str,
    gpu_ids: Optional[list[str]] = None,
) -> Optional[str]:
    """
    运行标准 Kohya 训练，返回输出目录中最终的 checkpoint 路径。

    使用 mikazuki.process 中的现有流程。
    """
    from mikazuki.process import build_accelerate_train_command

    # 写入 TOML
    autosave_dir = PROJECT_ROOT / "config" / "autosave"
    autosave_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_path = autosave_dir / f"{timestamp}-differential-{step_label}.toml"

    import toml as toml_lib

    toml_path.write_text(toml_lib.dumps(toml_config), encoding="utf-8")

    # 构建 accelerate 命令
    trainer_file = "./scripts/dev/anima_train_network.py"
    args, env, _ = build_accelerate_train_command(
        trainer_file=trainer_file,
        toml_path=str(toml_path),
        cpu_threads=2,
        gpu_ids=gpu_ids,
    )

    output_dir = toml_config.get("output_dir", "")
    log.info(f"[DiffLoRA] {step_label} 开始训练, output_dir={output_dir}")

    # 使用 TaskManager 同步执行
    task = tm.create_task(args, env)
    if task is None:
        log.error(f"[DiffLoRA] {step_label} 无法创建任务（可能已有任务在运行）")
        return None

    task.execute()
    task.wait()

    if task.returncode != 0:
        log.error(f"[DiffLoRA] {step_label} 训练失败 (exit {task.returncode})")
        return None

    # 找到最终 checkpoint
    if not output_dir or not os.path.isdir(output_dir):
        log.error(f"[DiffLoRA] {step_label} 输出目录不存在: {output_dir}")
        return None

    ckpts = sorted(glob.glob(os.path.join(output_dir, "*.safetensors")))
    if not ckpts:
        log.error(f"[DiffLoRA] {step_label} 未生成 checkpoint")
        return None

    final_ckpt = ckpts[-1]
    log.info(f"[DiffLoRA] {step_label} 完成, checkpoint: {final_ckpt}")
    return final_ckpt


def _merge_lora_to_base(
    base_path: str,
    lora_path: str,
    output_path: str,
    scale: float = 1.0,
    dtype: str = "fp16",
) -> bool:
    """调用 merge_lora_to_base.py 进行融合。"""
    python = _get_project_python()
    cmd = [
        python,
        MERGE_TOOL,
        "--base", base_path,
        "--lora", lora_path,
        "--output", output_path,
        "--scale", str(scale),
        "--dtype", dtype,
        "--verbose",
    ]
    rc, stdout, stderr = _run_subprocess(cmd, "LoRA 融合")
    return rc == 0


def _train_single_pair(
    config: dict,
    img_filename: str,
    img_a_path: str,
    img_b_path: str,
    safe_name: str,
    gpu_ids: Optional[list[str]],
    output_dir: str,
    base_model_path: str,
) -> Optional[str]:
    """
    训练一组差分 LoRA 配对。

    Returns:
        差分 LoRA safetensors 路径, 或 None (失败)
    """
    base_tags = read_base_tags(config.get("tag_dir", config["folder_a"]), img_filename)
    trigger_word = config.get("trigger_word", "Character_Splitting")
    remove_tokens_str = config.get("remove_tokens", "")
    step2_prompt = build_step2_prompt(base_tags, trigger_word, remove_tokens_str)

    log.info(f"[DiffLoRA] 配对: {img_filename}")
    log.info(f"[DiffLoRA]   Step1 prompt: {base_tags[:100]}")
    log.info(f"[DiffLoRA]   Step2 prompt: {step2_prompt[:100]}")

    pair_dir = os.path.join(output_dir, safe_name)
    os.makedirs(pair_dir, exist_ok=True)

    # ── Step 1: 训练 LoRA1 ──
    lora1_output = os.path.join(output_dir, f".tmp_lora1_{safe_name}")
    os.makedirs(lora1_output, exist_ok=True)

    step1_dataset = create_single_image_dataset(
        image_path=img_a_path,
        prompt=base_tags,
        safe_name=safe_name,
    )

    step1_toml = build_step1_toml(config, step1_dataset, lora1_output)
    step1_toml["output_name"] = f"lora1_{safe_name}"
    # 确保使用正确的底模
    step1_toml["pretrained_model_name_or_path"] = base_model_path

    lora1_ckpt = _run_kohya_training(step1_toml, f"step1_{safe_name}", gpu_ids)
    if not lora1_ckpt:
        log.error(f"[DiffLoRA] Step1 失败: {img_filename}")
        return None

    # ── 合并 LoRA1 → 底模 ──
    merged_model = os.path.join(output_dir, f".tmp_merged_{safe_name}.safetensors")
    success = _merge_lora_to_base(
        base_path=base_model_path,
        lora_path=lora1_ckpt,
        output_path=merged_model,
        scale=1.0,
        dtype=config.get("save_precision", "fp16"),
    )
    if not success:
        log.error(f"[DiffLoRA] 合并失败: {img_filename}")
        return None

    # ── Step 2: 训练差分 LoRA ──
    lora2_output = os.path.join(output_dir, f".tmp_lora2_{safe_name}")
    os.makedirs(lora2_output, exist_ok=True)

    step2_dataset = create_single_image_dataset(
        image_path=img_b_path,
        prompt=step2_prompt,
        safe_name=safe_name,
    )

    step2_toml = build_step2_toml(config, step2_dataset, lora2_output, merged_model)
    step2_toml["output_name"] = f"lora2_{safe_name}"

    lora2_ckpt = _run_kohya_training(step2_toml, f"step2_{safe_name}", gpu_ids)
    if not lora2_ckpt:
        log.error(f"[DiffLoRA] Step2 失败: {img_filename}")
        return None

    # ── 保存最终差分 LoRA ──
    final_path = os.path.join(pair_dir, f"{safe_name}_differential_lora.safetensors")
    import shutil

    shutil.copy2(lora2_ckpt, final_path)
    log.info(f"[DiffLoRA] 差分 LoRA 已保存: {final_path}")

    # ── 保存 prompt map ──
    prompt_map_path = os.path.join(output_dir, ".prompt_map.json")
    prompt_map = {}
    if os.path.isfile(prompt_map_path):
        try:
            with open(prompt_map_path, "r") as f:
                prompt_map = json.load(f)
        except Exception:
            pass
    prompt_map[safe_name] = step2_prompt
    with open(prompt_map_path, "w") as f:
        json.dump(prompt_map, f, ensure_ascii=False, indent=2)

    # ── 清理临时文件 ──
    if not config.get("keep_temp", False):
        _safe_rmtree(lora1_output)
        _safe_rmtree(lora2_output)
        _safe_rmfile(merged_model)

    return final_path


def _safe_rmtree(path: str) -> None:
    """安全删除目录。"""
    try:
        import shutil
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _safe_rmfile(path: str) -> None:
    """安全删除文件。"""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _postprocess(output_dir: str, config: dict) -> dict:
    """后处理: ComfyUI 转换 + SVD 合并。"""
    results = {
        "comfyui_converted": False,
        "svd_merged": False,
        "comfyui_count": 0,
        "merged_path": "",
    }

    python = _get_project_python()

    # 1. ComfyUI 格式转换
    if config.get("postprocess_comfyui", True):
        rc, stdout, stderr = _run_subprocess(
            [python, COMFYUI_TOOL, output_dir],
            "ComfyUI 转换",
        )
        if rc == 0:
            comfy_files = sorted(glob.glob(
                os.path.join(output_dir, "**", "*_comfyui.safetensors"), recursive=True
            ))
            results["comfyui_converted"] = True
            results["comfyui_count"] = len(comfy_files)
            log.info(f"[DiffLoRA] ComfyUI 转换完成: {len(comfy_files)} 个文件")

    # 2. SVD 合并
    if config.get("postprocess_svd", True):
        comfy_count = len(glob.glob(
            os.path.join(output_dir, "**", "*_comfyui.safetensors"), recursive=True
        ))
        if comfy_count >= 2:
            rc, stdout, stderr = _run_subprocess(
                [
                    python, AVERAGE_TOOL, output_dir,
                    "--method", "svd",
                    "--rank", str(config.get("lora_rank", 32)),
                ],
                "SVD 合并",
            )
            if rc == 0:
                merged_path = os.path.join(output_dir, "merged_lora.safetensors")
                if os.path.isfile(merged_path):
                    results["svd_merged"] = True
                    results["merged_path"] = merged_path
                    log.info(f"[DiffLoRA] SVD 合并完成: {merged_path}")
        else:
            log.info(f"[DiffLoRA] 仅 {comfy_count} 个文件，跳过 SVD 合并")

    return results


def run_differential_lora(config: dict) -> dict:
    """
    执行完整 Differential LoRA 训练流程。

    此函数应该在后台线程中调用。

    Args:
        config: 前端传入的完整配置（已通过 schema 解析）

    Returns:
        包含 results, errors, postprocess 的字典
    """
    folder_a = config.get("folder_a", "")
    folder_b = config.get("folder_b", "")
    output_dir = config.get("output_dir", "./models/differential_lora")
    base_model = config.get(
        "pretrained_model_name_or_path",
        "./sd-models/anima/anima-base-v1.0.safetensors",
    )
    gpu_ids = config.get("gpu_ids")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 配对图片
    pairs = pair_images(folder_a, folder_b)
    if not pairs:
        return {
            "status": "error",
            "message": f"未找到配对图片。folder_a={folder_a}, folder_b={folder_b}",
            "pairs": [],
        }

    log.info(f"[DiffLoRA] 找到 {len(pairs)} 组配对")

    # 2. 逐组训练
    results = []
    errors = []

    for idx, (filename, img_a, img_b) in enumerate(pairs):
        safe_name = os.path.splitext(filename)[0]
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_name)

        log.info(f"[DiffLoRA] [{idx + 1}/{len(pairs)}] {filename}")

        try:
            result_path = _train_single_pair(
                config=config,
                img_filename=filename,
                img_a_path=img_a,
                img_b_path=img_b,
                safe_name=safe_name,
                gpu_ids=gpu_ids,
                output_dir=output_dir,
                base_model_path=base_model,
            )
            if result_path:
                results.append({
                    "filename": filename,
                    "safe_name": safe_name,
                    "result_path": result_path,
                })
            else:
                errors.append({"filename": filename, "error": "训练失败"})
        except Exception as e:
            log.error(f"[DiffLoRA] {filename} 异常: {e}")
            traceback.print_exc()
            errors.append({"filename": filename, "error": str(e)})

    # 3. 后处理
    postprocess_results = {"comfyui_converted": False, "svd_merged": False}
    if results:
        try:
            postprocess_results = _postprocess(output_dir, config)
        except Exception as e:
            log.error(f"[DiffLoRA] 后处理异常: {e}")
            postprocess_results["error"] = str(e)

    return {
        "status": "success" if results else "error",
        "message": f"完成 {len(results)}/{len(pairs)} 组, 失败 {len(errors)} 组",
        "pairs": len(pairs),
        "success_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
        "output_dir": output_dir,
        "postprocess": postprocess_results,
        "prompt_map_path": os.path.join(output_dir, ".prompt_map.json"),
    }


def run_in_background(config: dict, result_holder: dict) -> None:
    """在后台线程中执行差分训练，结果写入 result_holder。"""
    try:
        result = run_differential_lora(config)
        result_holder["result"] = result
    except Exception as e:
        log.error(f"[DiffLoRA] 后台训练异常: {e}")
        traceback.print_exc()
        result_holder["error"] = str(e)
