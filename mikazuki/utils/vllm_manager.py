"""Manage local vLLM subprocesses for Anima Train / Tag-Edit-Leaf."""

from __future__ import annotations

import json
import importlib.util
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
            "vllm_serve": {
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.35,
                "max_num_seqs": 16,
            },
        },
        "gemma-4-e4b": {
            "display_name": "Gemma-4-E4B (vLLM)",
            "default_api_url": "http://127.0.0.1:9002/v1/chat/completions",
            "default_served_name": "spawner-gemma-4-e4b-it",
            "local_model_dir": "models/gemma-4-E3B-it",
            "modelscope_id": "spawner/spawner-gemma-4-E4B-it",
            "port": 9002,
            "vllm_serve": {
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.42,
                "max_num_seqs": 8,
            },
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


def _site_packages_dirs() -> list[Path]:
    """Collect likely site-packages roots for vLLM CUDA runtime wheels."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        dirs.append(resolved)

    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _add(PROJECT_ROOT / ".venv" / "lib" / py_tag / "site-packages")

    try:
        import vllm

        _add(Path(vllm.__file__).resolve().parent.parent)
    except Exception:
        pass

    vllm_bin = shutil.which("vllm")
    if vllm_bin:
        _add(Path(vllm_bin).resolve().parent.parent / "lib" / py_tag / "site-packages")

    _add(Path(sys.executable).resolve().parent.parent / "lib" / py_tag / "site-packages")
    return dirs


def _cuda_library_paths() -> list[str]:
    lib_paths: list[str] = []
    seen: set[str] = set()
    subdirs = (
        ("nvidia", "cu13", "lib"),
        ("nvidia", "cuda_runtime", "lib"),
    )
    for site_packages in _site_packages_dirs():
        for parts in subdirs:
            path = site_packages.joinpath(*parts)
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            lib_paths.append(resolved)

    system_cuda = Path("/usr/local/cuda/lib64")
    if system_cuda.is_dir():
        lib_paths.append(str(system_cuda))
    return lib_paths


def _build_vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = _cuda_library_paths()
    if lib_paths:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([existing] if existing else []))
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    return env


def _append_vllm_serve_options(cmd: list[str], preset: dict[str, Any]) -> None:
    """Append memory-safe vLLM serve flags for caption workloads."""
    serve = dict(preset.get("vllm_serve") or {})
    max_model_len = int(serve.get("max_model_len", 4096))
    gpu_memory_utilization = float(serve.get("gpu_memory_utilization", 0.42))
    max_num_seqs = int(serve.get("max_num_seqs", 8))
    enable_custom_ops = bool(serve.get("enable_custom_ops", False))

    cmd.extend(
        [
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-num-seqs",
            str(max_num_seqs),
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--generation-config",
            "vllm",
        ]
    )
    if not enable_custom_ops:
        # CUDA 12.8 driver cannot run vLLM 0.22 _C kernels built for CUDA 13.
        # Set vllm_serve.enable_custom_ops=true after upgrading to a CUDA-13-capable driver.
        cmd.extend(["-cc.custom_ops", '["none"]'])


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


def _parse_vllm_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for piece in str(version or "0").split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _check_vllm_runtime_deps() -> None:
    """Fail fast when optional deps break vLLM engine startup."""
    if importlib.util.find_spec("flash_attn") is None:
        return
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "检测到 flash-attn 与当前 PyTorch/vLLM 不兼容，会导致 vLLM 引擎启动失败。"
            "请执行: uv pip uninstall flash-attn"
            f"（{exc}）"
        ) from exc


def _check_vllm_runtime_compat(vlm_model: str, model_dir: Path) -> None:
    """Fail fast when the installed vLLM cannot serve the selected VLM."""
    gemma_keys = {"gemma-4-e4b", "gemma", "spawner-gemma-4-e4b-it"}
    if vlm_model not in gemma_keys:
        return

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type != "gemma4":
        return

    try:
        import vllm

        version = getattr(vllm, "__version__", "0")
    except Exception:
        version = "0"

    if _parse_vllm_version(version) < (0, 22, 0):
        raise RuntimeError(
            f"Gemma 4 需要 vLLM >= 0.22，当前为 vLLM {version}。"
            "请改用 ToriiGate (toriigate-0.5)，或在独立环境中部署新版 vLLM 服务并通过 API URL 连接。"
            "（Gemma 4 使用 rope_parameters 新格式，vLLM 0.9.x 无法加载。）"
        )


def _is_gemma_vllm_model(vlm_model: str) -> bool:
    return str(vlm_model or "").strip().lower() in {"gemma-4-e4b", "gemma", "spawner-gemma-4-e4b-it"}


def _probe_gemma_vllm_or_raise(api_url: str, served_name: str) -> None:
    """Fail when vLLM is reachable but Gemma generation is empty/pad-only."""
    tools_root = PROJECT_ROOT / "tools"
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from anima_caption_pipeline.vlm_client import (
        GemmaVllmUnavailableError,
        gemma_vllm_unavailable_message,
        probe_vllm_generation,
    )

    if not probe_vllm_generation(api_url=api_url, model_name=served_name, request_timeout=120.0):
        raise GemmaVllmUnavailableError(gemma_vllm_unavailable_message(api_url, served_name))


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
        if _is_gemma_vllm_model(vlm_model):
            _probe_gemma_vllm_or_raise(api_url, served_name)
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
    _check_vllm_runtime_deps()
    _check_vllm_runtime_compat(vlm_model, model_dir)

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
    ]
    _append_vllm_serve_options(cmd, preset)
    if vlm_model in {"gemma-4-e4b", "gemma", "spawner-gemma-4-e4b-it"}:
        cmd.extend(["--limit-mm-per-prompt", '{"image": 1}'])

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
                tail = "\n".join(_log_lines[-40:])
                _state["status"] = "error"
                _state["message"] = f"vLLM 进程已退出 (code={proc.returncode})"
            raise RuntimeError(f"vLLM 启动失败 (exit {proc.returncode})\n{tail}")

        health = check_vllm_health(api_url, served_name, timeout=5.0)
        if health.get("ready"):
            try:
                if _is_gemma_vllm_model(vlm_model):
                    _probe_gemma_vllm_or_raise(api_url, served_name)
            except Exception as exc:
                with _lock:
                    tail = "\n".join(_log_lines[-40:])
                    _state["status"] = "error"
                    _state["message"] = str(exc)
                raise RuntimeError(f"{exc}\n{tail}") from exc
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
