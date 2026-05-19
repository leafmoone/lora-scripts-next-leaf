
import asyncio
import os
import sys
import webbrowser
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path
from mikazuki.portable_utils import train_env_overrides


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_train_log_urls(task_id: str) -> dict:
    """Construct full http(s) URLs for the train-log viewer + SSE stream.

    Reads host/port from ``MIKAZUKI_HOST`` / ``MIKAZUKI_PORT`` (set in
    ``mikazuki/app/application.py`` at boot). ``0.0.0.0`` is normalized to
    ``127.0.0.1`` so the printed URL is actually clickable from the host
    machine.
    """

    host = os.environ.get("MIKAZUKI_HOST", "127.0.0.1") or "127.0.0.1"
    port = os.environ.get("MIKAZUKI_PORT", "28000") or "28000"
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    base = f"http://{display_host}:{port}"
    return {
        "base": base,
        "viewer": f"{base}/train-log?task_id={task_id}",
        "stream": f"{base}/api/train/log/stream/{task_id}",
    }


def _announce_train_log(task_id: str, urls: dict) -> None:
    """Print a prominent, clickable banner pointing at the live log viewer."""

    viewer = urls["viewer"]
    stream = urls["stream"]
    banner = (
        "\n"
        "  Train log viewer (open in browser):\n"
        f"    {viewer}\n"
        f"    SSE stream: {stream}\n"
        f"    task_id   : {task_id}\n"
    )
    log.info(banner)

    if _truthy_env("MIKAZUKI_AUTO_OPEN_TRAIN_LOG"):
        try:
            from mikazuki.app.application import _resolve_browser
            _resolve_browser().open(viewer)
        except Exception as exc:  # noqa: BLE001 — best-effort UX nicety
            log.warning(f"Failed to auto-open train log in browser: {exc}")


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env.update(train_env_overrides())
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    urls = build_train_log_urls(task.task_id)
    _announce_train_log(task.task_id, urls)

    def _run():
        try:
            task.execute()
            task.wait()
            rc = task.process.returncode if task.process else -1
            if rc != 0:
                log.error(f"Training failed / 训练失败 (exit {rc})")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(
        status="success",
        message=f"Training started / 训练开始 ID: {task.task_id}",
        data={
            "task_id": task.task_id,
            "train_log_path": "/train-log",
            "train_log_query": f"task_id={task.task_id}",
            "train_log_stream": f"/api/train/log/stream/{task.task_id}",
            # Full clickable URLs (new in this release).
            "train_log_url": urls["viewer"],
            "train_log_stream_url": urls["stream"],
        },
    )
