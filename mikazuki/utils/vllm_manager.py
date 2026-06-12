"""Manage local vLLM subprocesses for Anima Train / Tag-Edit-Leaf."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "anima_caption_models.json"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_log_lines: list[str] = []
_state: dict[str, Any] = {
    "vlm_model": "",
    "status": "idle",
    "pid": None,
    "port": None,
    "api_url": "",
    "served_name": "",
    "message": "",
}


def _default_presets() -> dict[str, dict[str, Any]]:
    return {
        "toriigate-0.5": {
            "display_name": "ToriiGate 0.5",
            "default_api_url": "http://127.0.0.1:18901/v1/chat/completions",
            "default_served_name": "toriigate-0.5",
            "local_model_dir": "models/toriigate/toriigate-0.5",
            "port": 18901,
        },
        "gemma-4-e4b": {
            "display_name": "Gemma-4-E4B (vLLM)",
            "default_api_url": "http://127.0.0.1:9002/v1/chat/completions",
            "default_served_name": "spawner-gemma-4-e4b-it",
            "local_model_dir": "models/gemma-4-E3B-it",
            "modelscope_id": "spawner/spawner-gemma-4-E4B-it",
            "port": 9002,
        },
    }


def load_vlm_presets() -> dict[str, dict[str, Any]]:
    presets = _default_presets()
    try:
        if CONFIG_PATH.is_file():
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for model_id, preset in (raw.get("vlm_models") or {}).items():
                merged = dict(presets.get(model_id, {}))
                merged.update(preset or {})
                if "port" not in merged and merged.get("default_api_url"):
                    merged["port"] = parse_port(str(merged["default_api_url"]))
                presets[model_id] = merged
    except Exception:
        pass
    return presets


def get_vlm_preset(vlm_model: str) -> dict[str, Any]:
    key = str(vlm_model or "").strip().lower()
    presets = load_vlm_presets()
    if key in presets:
        return presets[key]
    aliases = {
        "toriigate": "toriigate-0.5",
        "gemma": "gemma-4-e4b",
        "spawner-gemma-4-e4b-it": "gemma-4-e4b",
    }
    return presets.get(aliases.get(key, key), presets["toriigate-0.5"])


def parse_port(api_url: str) -> int:
    parsed = urlparse(str(api_url or "").strip())
    if parsed.port:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    return 80


def models_endpoint(api_url: str) -> str:
    base = str(api_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def check_vllm_health(api_url: str, served_name: str = "", timeout: float = 5.0) -> dict[str, Any]:
    url = models_endpoint(api_url)
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return {
                "ready": False,
                "message": f"vLLM 未就绪 (HTTP {response.status_code})",
                "models_url": url,
            }
        payload = response.json()
        model_ids = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
        if served_name and model_ids and served_name not in model_ids:
            return {
                "ready": True,
                "message": f"vLLM 已启动，served models: {', '.join(model_ids)}",
                "models": model_ids,
                "models_url": url,
            }
        return {
            "ready": True,
            "message": "vLLM 已就绪",
            "models": model_ids,
            "models_url": url,
        }
    except Exception as exc:
        return {
            "ready": False,
            "message": str(exc),
            "models_url": url,
        }


def _cuda_library_paths() -> list[str]:
    site_packages = Path(sys.executable).resolve().parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    candidates = [
        site_packages / "nvidia" / "cu13" / "lib",
        site_packages / "nvidia" / "cuda_runtime" / "lib",
        Path("/usr/local/cuda/lib64"),
    ]
    return [str(path) for path in candidates if path.is_dir()]


def _build_vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = _cuda_library_paths()
    if lib_paths:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([existing] if existing else []))
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    return env


def _validate_model_dir(vlm_model: str, preset: dict[str, Any], *, auto_download_gemma: bool) -> Path:
    rel_dir = preset.get("local_model_dir")
    if not rel_dir:
        raise FileNotFoundError(f"{vlm_model} 未配置 local_model_dir")
    model_dir = (PROJECT_ROOT / str(rel_dir)).resolve()
    config_path = model_dir / "config.json"
    if config_path.is_file():
        return model_dir

    if vlm_model in {"gemma-4-e4b", "gemma"}:
        tools_root = PROJECT_ROOT / "tools"
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))
        from anima_caption_pipeline.model_resolver import ensure_gemma_model

        return ensure_gemma_model(PROJECT_ROOT, auto_download=auto_download_gemma)

    raise FileNotFoundError(f"模型目录无效: {model_dir}")


def _read_process_output(proc: subprocess.Popen) -> None:
    global _log_lines
    if not proc.stdout:
        return
    for line in proc.stdout:
        text = line.rstrip("\r\n")
        if not text:
            continue
        with _lock:
            _log_lines.append(text)
            if len(_log_lines) > 200:
                _log_lines = _log_lines[-200:]


def _stop_managed_process() -> None:
    global _proc
    with _lock:
        proc = _proc
        _proc = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def get_vllm_status(vlm_model: str = "") -> dict[str, Any]:
    preset = get_vlm_preset(vlm_model) if vlm_model else {}
    api_url = str(preset.get("default_api_url", "")) if preset else ""
    served_name = str(preset.get("default_served_name", "")) if preset else ""
    port = int(preset.get("port") or (parse_port(api_url) if api_url else 0))

    with _lock:
        snapshot = dict(_state)
        log_tail = "\n".join(_log_lines[-20:])

    health = check_vllm_health(api_url, served_name) if api_url else {"ready": False, "message": "未配置 API URL"}
    managed_running = bool(snapshot.get("pid")) and snapshot.get("status") in {"starting", "running"}

    if health.get("ready"):
        status = "running"
        message = health.get("message", "vLLM 已就绪")
    elif managed_running:
        status = snapshot.get("status", "starting")
        message = snapshot.get("message") or "vLLM 启动中..."
    elif is_port_open(port) and not health.get("ready"):
        status = "busy"
        message = f"端口 {port} 已被占用，但 vLLM /v1/models 不可用"
    else:
        status = snapshot.get("status") if snapshot.get("status") != "running" else "idle"
        if status not in {"starting", "error"}:
            status = "stopped"
        message = snapshot.get("message") or "vLLM 未运行"

    return {
        "vlm_model": vlm_model or snapshot.get("vlm_model", ""),
        "status": status,
        "ready": bool(health.get("ready")),
        "message": message,
        "pid": snapshot.get("pid"),
        "port": port,
        "api_url": api_url,
        "served_name": served_name,
        "local_model_dir": preset.get("local_model_dir"),
        "managed": managed_running,
        "models": health.get("models", []),
        "log_tail": log_tail,
    }


def start_vllm(
    vlm_model: str,
    *,
    auto_download_gemma: bool = True,
    wait_ready: bool = True,
    timeout: float = 900.0,
) -> dict[str, Any]:
    global _proc, _log_lines

    preset = get_vlm_preset(vlm_model)
    api_url = str(preset["default_api_url"])
    served_name = str(preset["default_served_name"])
    port = int(preset.get("port") or parse_port(api_url))

    existing = check_vllm_health(api_url, served_name, timeout=3.0)
    if existing.get("ready"):
        with _lock:
            _state.update({
                "vlm_model": vlm_model,
                "status": "running",
                "pid": _state.get("pid"),
                "port": port,
                "api_url": api_url,
                "served_name": served_name,
                "message": existing.get("message", "vLLM 已就绪"),
            })
        return get_vllm_status(vlm_model)

    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        raise RuntimeError("未找到 vllm 命令，请先安装: pip install vllm")

    model_dir = _validate_model_dir(vlm_model, preset, auto_download_gemma=auto_download_gemma)

    with _lock:
        if _proc and _proc.poll() is None and _state.get("vlm_model") not in ("", vlm_model):
            _stop_managed_process()

    if is_port_open(port):
        health = check_vllm_health(api_url, served_name, timeout=2.0)
        if not health.get("ready"):
            raise RuntimeError(
                f"端口 {port} 已被占用，但 {models_endpoint(api_url)} 不可用。"
                "请先停止占用该端口的进程。"
            )

    cmd = [
        vllm_bin,
        "serve",
        str(model_dir),
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--max-model-len",
        "8192",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
    ]

    with _lock:
        _log_lines = []
        _state.update({
            "vlm_model": vlm_model,
            "status": "starting",
            "pid": None,
            "port": port,
            "api_url": api_url,
            "served_name": served_name,
            "message": "正在启动 vLLM...",
        })

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=_build_vllm_env(),
    )

    with _lock:
        _proc = proc
        _state["pid"] = proc.pid

    threading.Thread(target=_read_process_output, args=(proc,), daemon=True).start()

    if not wait_ready:
        return get_vllm_status(vlm_model)

    deadline = time.time() + max(30.0, float(timeout))
    last_message = "vLLM 启动中..."
    while time.time() < deadline:
        if proc.poll() is not None:
            with _lock:
                tail = "\n".join(_log_lines[-20:])
                _state["status"] = "error"
                _state["message"] = f"vLLM 进程已退出 (code={proc.returncode})"
            raise RuntimeError(f"vLLM 启动失败 (exit {proc.returncode})\n{tail}")

        health = check_vllm_health(api_url, served_name, timeout=5.0)
        if health.get("ready"):
            with _lock:
                _state["status"] = "running"
                _state["message"] = health.get("message", "vLLM 已就绪")
            return get_vllm_status(vlm_model)

        with _lock:
            if _log_lines:
                for line in reversed(_log_lines[-10:]):
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        last_message = line
                        break
                else:
                    last_message = _log_lines[-1]
            _state["message"] = last_message
        time.sleep(2.0)

    with _lock:
        _state["status"] = "error"
        _state["message"] = f"vLLM 启动超时 ({int(timeout)}s)"
        tail = "\n".join(_log_lines[-20:])
    raise RuntimeError(f"vLLM 启动超时\n{tail}")


def ensure_vllm_ready(
    vlm_model: str,
    *,
    auto_download_gemma: bool = True,
    timeout: float = 900.0,
) -> dict[str, Any]:
    status = get_vllm_status(vlm_model)
    if status.get("ready"):
        return status
    return start_vllm(
        vlm_model,
        auto_download_gemma=auto_download_gemma,
        wait_ready=True,
        timeout=timeout,
    )


def stop_vllm(vlm_model: str = "") -> dict[str, Any]:
    with _lock:
        current_model = _state.get("vlm_model", "")
    if vlm_model and current_model and vlm_model != current_model:
        return get_vllm_status(vlm_model)
    _stop_managed_process()
    with _lock:
        _state.update({
            "status": "stopped",
            "pid": None,
            "message": "vLLM 已停止",
        })
    return get_vllm_status(vlm_model or current_model)
