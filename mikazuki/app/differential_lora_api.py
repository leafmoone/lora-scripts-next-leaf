"""
Differential LoRA REST API 路由

端点:
  POST /api/differential-lora/pairs     - 预览图片配对
  POST /api/differential-lora/run       - 启动差分训练
  GET  /api/differential-lora/status    - 查询训练状态
  POST /api/differential-lora/tag       - 自动打标
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, Request

from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess
from mikazuki.differential_lora.preprocess import pair_images
from mikazuki.differential_lora.task_runner import (
    DIFFERENTIAL_TRAIN_TYPE,
    run_differential_lora,
)
from mikazuki.log import log
from mikazuki.utils.tagger_cmd import build_tagger_cmd, default_tagger_dir

router = APIRouter(prefix="/api/differential-lora")

# 用 dict 保存最近一次任务的运行状态，简单不需要持久化
_task_states: dict[str, dict] = {}


def _current_task() -> dict | None:
    """获取最近一次差分训练任务状态。"""
    return _task_states.get("current")


@router.post("/pairs")
async def preview_pairs(request: Request):
    """预览图片配对结果。"""
    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON 请求体")

    folder_a = data.get("folder_a", "")
    folder_b = data.get("folder_b", "")

    if not folder_a or not folder_b:
        return APIResponseFail(message="请指定 folder_a 和 folder_b")

    try:
        pairs = pair_images(folder_a, folder_b)
        return APIResponseSuccess(data={
            "count": len(pairs),
            "pairs": [
                {"filename": fname, "img_a": a, "img_b": b}
                for fname, a, b in pairs
            ],
        })
    except FileNotFoundError as e:
        return APIResponseFail(message=str(e))
    except Exception as e:
        log.error(f"[DiffLoRA API] pairs 异常: {e}")
        return APIResponseFail(message=str(e))


@router.post("/run")
async def run_differential_training(request: Request):
    """启动 Differential LoRA 训练。"""
    try:
        config = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON 请求体")

    # 标记训练类型
    config["model_train_type"] = DIFFERENTIAL_TRAIN_TYPE

    # 验证必要字段
    folder_a = config.get("folder_a", "")
    folder_b = config.get("folder_b", "")
    if not folder_a or not folder_b:
        return APIResponseFail(message="请指定 folder_a 和 folder_b")

    if not os.path.isdir(folder_a):
        return APIResponseFail(message=f"folder_a 不存在: {folder_a}")
    if not os.path.isdir(folder_b):
        return APIResponseFail(message=f"folder_b 不存在: {folder_b}")

    # 先预览配对数量
    pairs = pair_images(folder_a, folder_b)
    if not pairs:
        return APIResponseFail(message=f"未找到配对图片。folder_a={folder_a}, folder_b={folder_b}")

    # 设置默认输出目录
    config.setdefault("output_dir", "./models/differential_lora")

    # 初始化任务状态
    task_id = f"diff-lora-{int(__import__('time').time())}"
    _task_states["current"] = {
        "task_id": task_id,
        "status": "running",
        "total_pairs": len(pairs),
        "completed_pairs": 0,
        "results": [],
        "errors": [],
        "start_time": __import__('time').time(),
    }

    # 在后台线程中执行训练
    result_holder = {}

    def _background_train():
        try:
            # 先执行自动标注（如果启用）
            if config.get("auto_tag", False):
                _run_auto_tagging(folder_a, config)

            # 执行训练
            result = run_differential_lora(config)
            result_holder["result"] = result

            # 更新任务状态
            current = _task_states.get("current", {})
            current["status"] = result.get("status", "error")
            current["results"] = result.get("results", [])
            current["errors"] = result.get("errors", [])
            current["completed_pairs"] = result.get("success_count", 0)
            current["postprocess"] = result.get("postprocess", {})
            _task_states["current"] = current
        except Exception as e:
            log.error(f"[DiffLoRA API] 后台训练异常: {e}")
            import traceback
            traceback.print_exc()
            current = _task_states.get("current", {})
            current["status"] = "error"
            current["error_message"] = str(e)
            _task_states["current"] = current

    thread = threading.Thread(target=_background_train, daemon=True)
    thread.start()

    return APIResponseSuccess(
        message=f"差分训练已启动，共 {len(pairs)} 组配对",
        data={
            "task_id": task_id,
            "total_pairs": len(pairs),
            "pairs": [{"filename": fname} for fname, _, _ in pairs],
        },
    )


def _run_auto_tagging(folder_a: str, config: dict) -> None:
    """调用 tools/differential_tagger/main.py 打标，直接生成 .txt 到 folder_a。"""
    import subprocess

    tagger_dir = default_tagger_dir()
    main_py = os.path.join(tagger_dir, "main.py")

    if not os.path.isfile(main_py):
        log.error(f"[DiffLoRA] 标注器脚本不存在: {main_py}")
        return

    cmd = build_tagger_cmd({
        "input_dir": folder_a,
        "output_dir": folder_a,
        "mode": config.get("tagger_mode", "smart"),
        "save_captions": True,
        "model": config.get("tagger_model", "wd-eva02-large-tagger-v3"),
        "threshold": config.get("tagger_threshold", 0.35),
        "char_threshold": config.get("tagger_char_threshold", 0.85),
        "use_vlm": config.get("tagger_use_vlm", True),
        "use_cpu": config.get("tagger_use_cpu", False),
        "recursive": config.get("tagger_recursive", False),
        "resume": config.get("tagger_resume", False),
        "purpose": config.get("tagger_purpose", "character"),
        "taggers": config.get("tagger_taggers", ""),
        "consensus": config.get("tagger_consensus", 2),
        "max_tags": config.get("tagger_max_tags", 0),
        "blacklist": config.get("tagger_blacklist", ""),
        "wd14_batch": config.get("tagger_wd14_batch", 8),
        "vlm_batch": config.get("tagger_vlm_batch", 4),
        "vlm_backend": config.get("tagger_vlm_backend", "transformers"),
        "vllm_api_url": config.get("tagger_vllm_api_url", ""),
        "vllm_model": config.get("tagger_vllm_model", ""),
        "data_dir": config.get("tagger_data_dir", ""),
    }, python_executable=sys.executable, tagger_dir=tagger_dir)

    log.info(f"[DiffLoRA] 自动打标: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=600, cwd=tagger_dir)
        if result.returncode == 0:
            log.info(f"[DiffLoRA] 自动打标完成: {folder_a}")
        else:
            log.error(f"[DiffLoRA] 打标失败 (exit {result.returncode})")
    except subprocess.TimeoutExpired:
        log.error("[DiffLoRA] 打标超时")
    except Exception as e:
        log.error(f"[DiffLoRA] 打标异常: {e}")


@router.get("/status")
async def get_training_status():
    """查询差分训练状态。"""
    current = _task_states.get("current")
    if not current:
        return APIResponseSuccess(data={"status": "idle", "message": "没有正在运行的差分训练任务"})

    elapsed = __import__('time').time() - current.get("start_time", __import__('time').time())
    return APIResponseSuccess(data={
        "status": current.get("status", "idle"),
        "task_id": current.get("task_id"),
        "total_pairs": current.get("total_pairs", 0),
        "completed_pairs": current.get("completed_pairs", 0),
        "results_count": len(current.get("results", [])),
        "errors_count": len(current.get("errors", [])),
        "postprocess": current.get("postprocess", {}),
        "elapsed_seconds": int(elapsed),
        "error_message": current.get("error_message", ""),
    })
