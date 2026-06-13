"""
DiffSynth Tag Editor (Tag-Edit-Leaf) REST API

独立图片标注器，调用 tools/differential_tagger/main.py
支持 simple 和 smart 两种模式，适用于任意图片文件夹。
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import APIRouter, Request

from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.log import log
from mikazuki.utils.tag_edit_leaf_helpers import (
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_WORKERS,
    MAX_API_WORKERS,
    parse_progress_line,
    prepare_image_for_api,
)
from mikazuki.utils.tagger_cmd import build_tagger_cmd, default_tagger_dir
from mikazuki.utils.vllm_manager import (
    ensure_vllm_ready,
    get_vllm_status,
    get_vlm_preset,
    start_vllm,
    stop_vllm,
)

router = APIRouter(prefix="/api/tag-edit-leaf")

TAGGER_DIR = default_tagger_dir()

AVAILABLE_MODELS = [
    "wd-eva02-large-tagger-v3",
    "wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3",
    "wd-vit-tagger-v3",
    "wd-vit-large-tagger-v3",
    "camie-tagger-v2",
    "pixai-tagger-v0.9",
    "oppai-oracle-v1.1",
]

AVAILABLE_PURPOSES = ["character", "style", "general", "concept"]

_task_state: dict = {
    "status": "idle",
    "progress": 0,
    "phase": "",
    "message": "",
    "output_dir": "",
    "results": [],
    "failed_count": 0,
    "last_error": "",
}


@router.post("/scan")
async def scan_directory(request: Request):
    """Scan directory and return image count + preview list."""
    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON")

    path = data.get("path", "")
    recursive = data.get("recursive", False)

    if not path or not os.path.isdir(path):
        return APIResponseFail(message=f"目录不存在: {path}")

    p = Path(path)
    pattern = "**/*" if recursive else "*"
    images = []
    for f in sorted(p.glob(pattern)):
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}:
            caption_path = f.with_suffix(".txt")
            caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else ""
            images.append({
                "name": f.name,
                "path": str(f),
                "relative_path": str(f.relative_to(p)),
                "caption": caption,
                "caption_exists": caption_path.is_file(),
            })

    return APIResponseSuccess(data={
        "path": path,
        "count": len(images),
        "preview": images[:50],
    })


@router.get("/image")
async def serve_image(root: str, img: str):
    """Serve an image file from the dataset directory."""
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    try:
        root_path = Path(root).expanduser().resolve()
        img_path = (root_path / img).resolve()
        img_path.relative_to(root_path)
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="image path is outside root")
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(img_path))



@router.post("/run")
async def run_tagger(request: Request):
    """Start tagging job."""
    global _task_state

    if _task_state["status"] == "running":
        return APIResponseFail(message="标注任务正在运行中")

    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON")

    input_dir = data.get("input_dir", "")
    if not input_dir or not os.path.isdir(input_dir):
        return APIResponseFail(message=f"输入目录不存在: {input_dir}")

    output_dir = data.get("output_dir", "") or input_dir
    mode = data.get("mode", "smart")
    model = data.get("model", "wd-eva02-large-tagger-v3")
    threshold = float(data.get("threshold", 0.35))
    char_threshold = float(data.get("char_threshold", 0.85))
    use_cpu = data.get("use_cpu", False)
    recursive = data.get("recursive", False)
    purpose = data.get("purpose", "character")
    trigger = data.get("trigger", "")
    use_wd14 = data.get("use_wd14", True)
    use_vlm = data.get("use_vlm", True)
    vlm_prompt_mode = data.get("vlm_prompt_mode", "lora")
    inject_wd14_tags = data.get("inject_wd14_tags", True)
    taggers = data.get("taggers", [])
    consensus = int(data.get("consensus", 2))
    max_tags = int(data.get("max_tags", 0))
    blacklist = data.get("blacklist", [])
    save_captions = data.get("save_captions", True)
    verbose = data.get("verbose", False)
    data_dir = data.get("data_dir") or os.path.join(os.path.dirname(TAGGER_DIR), "..", "models")
    data_dir = os.path.abspath(data_dir)  # resolve relative to project root, not tagger cwd

    # API mode: call third-party API instead of local tagger
    if data.get("api_mode"):
        return _run_api_tagging(data)

    if mode == "anima_train":
        return _run_anima_train_tagging(data)

    if mode == "smart":
        use_wd14_bool = data.get("use_wd14", True)
        use_vlm_bool = data.get("use_vlm", True)
        if isinstance(use_wd14_bool, str):
            use_wd14_bool = use_wd14_bool.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(use_vlm_bool, str):
            use_vlm_bool = use_vlm_bool.strip().lower() in {"1", "true", "yes", "on"}
        if not use_wd14_bool and not use_vlm_bool:
            return APIResponseFail(message="Smart 模式至少需要启用 WD14 或 ToriiGate VLM 之一")

    cmd = build_tagger_cmd({
        "input_dir": input_dir,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "mode": mode,
        "model": model,
        "threshold": threshold,
        "char_threshold": char_threshold,
        "use_cpu": use_cpu,
        "recursive": recursive,
        "save_captions": save_captions,
        "resume": data.get("resume", False),
        "verbose": verbose,
        "trigger": trigger,
        "wd14_batch": data.get("wd14_batch", 8),
        "vlm_batch": data.get("vlm_batch", 4),
        "vlm_backend": data.get("vlm_backend", "transformers"),
        "vllm_api_url": data.get("vllm_api_url", ""),
        "vllm_model": data.get("vllm_model", ""),
        "purpose": purpose,
        "use_wd14": use_wd14,
        "use_vlm": use_vlm,
        "vlm_prompt_mode": vlm_prompt_mode,
        "inject_wd14_tags": inject_wd14_tags,
        "taggers": taggers,
        "consensus": consensus,
        "max_tags": max_tags,
        "blacklist": blacklist,
    }, python_executable=sys.executable, tagger_dir=TAGGER_DIR)

    log.info(f"[TagLeaf] {' '.join(cmd)}")

    _task_state.update({
        "status": "running",
        "progress": 0,
        "phase": "starting",
        "message": "正在扫描图片...",
        "output_dir": output_dir,
        "results": [],
        "failed_count": 0,
        "last_error": "",
        "start_time": time.time(),
    })

    def _run():
        global _task_state
        captured_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=TAGGER_DIR,
            )
            for line in proc.stdout:
                stripped = line.rstrip("\r\n")
                captured_lines.append(line)
                if not stripped:
                    continue
                # Always log output (skip pure carriage-return tqdm lines to avoid noise)
                if not stripped.startswith("\r") and stripped:
                    log.info(f"[TagLeaf] {stripped}")
                if not parse_progress_line(stripped, _task_state):
                    _task_state["message"] = stripped[-200:] if len(stripped) > 200 else stripped
            proc.wait()

            if proc.returncode == 0:
                results_json = os.path.join(output_dir, "results.json")
                results = []
                if os.path.isfile(results_json):
                    import json
                    try:
                        with open(results_json, "r") as f:
                            results = json.load(f)
                    except Exception:
                        pass

                _task_state["status"] = "done"
                _task_state["progress"] = 100
                _task_state["message"] = "标注完成"
                _task_state["results"] = results
                log.info(f"[TagLeaf] 标注完成: {output_dir}")
            else:
                tail = "".join(captured_lines[-30:]) if captured_lines else "(无输出)"
                error_summary = f"标注失败 (exit {proc.returncode})"
                log.error(f"[TagLeaf] {error_summary}")
                log.error(f"[TagLeaf] ── 子进程输出（最后 30 行）──\n{tail}\n── 结束 ──")
                _task_state["status"] = "error"
                _task_state["message"] = error_summary
                _task_state["error_log"] = tail
        except Exception as e:
            tail = "".join(captured_lines[-30:]) if captured_lines else "(无输出)"
            log.error(f"[TagLeaf] 子进程异常: {e}")
            if captured_lines:
                log.error(f"[TagLeaf] ── 子进程输出（最后 30 行）──\n{tail}\n── 结束 ──")
            _task_state["status"] = "error"
            _task_state["message"] = str(e)
            _task_state["error_log"] = tail

    threading.Thread(target=_run, daemon=True).start()

    return APIResponseSuccess(
        message="标注任务已启动",
        data={"output_dir": output_dir},
    )


@router.get("/status")
async def get_status():
    """Get current tagging job status."""
    elapsed = 0
    if _task_state.get("start_time"):
        elapsed = int(time.time() - _task_state["start_time"])

    return APIResponseSuccess(data={
        "status": _task_state["status"],
        "progress": _task_state["progress"],
        "phase": _task_state.get("phase", ""),
        "message": _task_state["message"],
        "output_dir": _task_state.get("output_dir", ""),
        "results_count": len(_task_state.get("results", [])),
        "results": _task_state.get("results", [])[:20],
        "failed_count": _task_state.get("failed_count", 0),
        "last_error": _task_state.get("last_error", ""),
        "elapsed_seconds": elapsed,
        "single_task": True,
    })


def _run_api_tagging(data: dict) -> dict:
    """后台线程：扫描图片 → base64 编码 → 调用 OpenAI/Anthropic API → 写 .txt + results.json"""
    global _task_state

    input_dir = data.get("input_dir", "")
    output_dir = data.get("output_dir", "") or input_dir
    save_captions = data.get("save_captions", True)
    recursive = data.get("recursive", False)
    trigger = data.get("trigger", "")

    api_provider = data.get("api_provider", "openai")
    api_base_url = data.get("api_base_url", "https://api.openai.com/v1").rstrip("/")
    api_key = data.get("api_key", "")
    api_model = data.get("api_model", "gpt-4o")
    api_system_prompt = data.get("api_system_prompt", "You are an AI image tagging assistant.")
    api_max_tokens = int(data.get("api_max_tokens", 256))
    api_temperature = float(data.get("api_temperature", 0.3))
    api_workers = min(max(1, int(data.get("api_workers", DEFAULT_API_WORKERS))), MAX_API_WORKERS)
    api_max_retries = max(0, int(data.get("api_max_retries", DEFAULT_API_MAX_RETRIES)))
    api_timeout = max(10, int(data.get("api_timeout", DEFAULT_API_TIMEOUT)))

    _task_state.update({
        "status": "running",
        "progress": 0,
        "phase": "scanning",
        "message": "API 标注: 扫描图片...",
        "output_dir": output_dir,
        "results": [],
        "failed_count": 0,
        "last_error": "",
        "start_time": time.time(),
    })

    def _worker():
        global _task_state
        try:
            import requests

            p = Path(input_dir)
            pattern = "**/*" if recursive else "*"
            image_files = sorted(
                f
                for f in p.glob(pattern)
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
            )
            if not image_files:
                _task_state["status"] = "error"
                _task_state["message"] = "未找到图片文件"
                return

            is_openai = api_provider == "openai"
            if is_openai:
                url = f"{api_base_url}/chat/completions"
            else:
                url = f"{api_base_url}/v1/messages"

            headers = {"Content-Type": "application/json"}
            if is_openai:
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"

            system_msg = api_system_prompt
            if trigger:
                system_msg += f'\nAlways include the trigger word "{trigger}" at the beginning.'

            total = len(image_files)
            results: list[dict] = []
            failed_count = 0
            last_error = ""
            completed = 0
            lock = threading.Lock()

            def tag_one(img_path: Path) -> dict:
                img_bytes, subtype = prepare_image_for_api(img_path)
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                data_uri = f"data:image/{subtype};base64,{img_b64}"

                if is_openai:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {"type": "text", "text": "Tag this image:"},
                            ],
                        },
                    ]
                    payload = {
                        "model": api_model,
                        "messages": messages,
                        "max_tokens": api_max_tokens,
                        "temperature": api_temperature,
                    }
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": f"image/{subtype}",
                                        "data": img_b64,
                                    },
                                },
                                {"type": "text", "text": "Tag this image:"},
                            ],
                        },
                    ]
                    payload = {
                        "model": api_model,
                        "messages": messages,
                        "max_tokens": api_max_tokens,
                        "temperature": api_temperature,
                        "system": system_msg,
                    }

                last_exc: Exception | None = None
                for attempt in range(api_max_retries + 1):
                    try:
                        resp = requests.post(url, json=payload, headers=headers, timeout=api_timeout)
                        if resp.status_code == 429:
                            wait = min(30, 2 ** attempt)
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        body = resp.json()
                        if is_openai:
                            content = body["choices"][0]["message"]["content"]
                        else:
                            content = body["content"][0]["text"]
                        tags = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
                        if trigger and tags and trigger not in tags[0]:
                            tags.insert(0, trigger)
                        if save_captions:
                            img_path.with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
                        return {
                            "image_path": str(img_path),
                            "caption": content,
                            "all_tags": [{"tag": t, "category": "general"} for t in tags],
                        }
                    except Exception as exc:
                        last_exc = exc
                        if attempt < api_max_retries:
                            time.sleep(min(10, 2 ** attempt))
                raise last_exc or RuntimeError("API request failed")

            _task_state["phase"] = "tagging"
            with ThreadPoolExecutor(max_workers=api_workers) as pool:
                futures = {pool.submit(tag_one, img_path): img_path for img_path in image_files}
                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        failed_count += 1
                        last_error = str(exc)
                        log.warning(f"[TagLeaf API] {img_path.name} 请求失败: {exc}")
                        results.append({"image_path": str(img_path), "error": str(exc), "all_tags": []})
                    with lock:
                        completed += 1
                        _task_state["progress"] = round(completed / total * 100)
                        _task_state["message"] = f"API 标注: [{completed}/{total}] {img_path.name}"
                        _task_state["failed_count"] = failed_count
                        _task_state["last_error"] = last_error

            results.sort(key=lambda item: item.get("image_path", ""))
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            ok_count = total - failed_count
            _task_state["status"] = "done" if failed_count == 0 else "done"
            _task_state["progress"] = 100
            _task_state["phase"] = "done"
            if failed_count:
                _task_state["message"] = f"API 标注完成: 成功 {ok_count}/{total}，失败 {failed_count}"
            else:
                _task_state["message"] = f"API 标注完成: {total} 张"
            _task_state["results"] = results
            _task_state["failed_count"] = failed_count
            _task_state["last_error"] = last_error
            log.info(f"[TagLeaf API] 完成: {ok_count}/{total} 成功, {failed_count} 失败")

        except Exception as exc:
            log.error(f"[TagLeaf API] 异常: {exc}")
            _task_state["status"] = "error"
            _task_state["message"] = str(exc)
            _task_state["last_error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()

    return APIResponseSuccess(
        message="API 标注已启动",
        data={"output_dir": output_dir},
    )


def _run_anima_train_tagging(data: dict) -> dict:
    """后台线程：WD14 batch → 两步 VLM (vLLM HTTP) → anima_train_v1 .txt"""
    global _task_state

    input_dir = data.get("input_dir", "")
    output_dir = data.get("output_dir", "") or input_dir
    save_captions = data.get("save_captions", True)
    recursive = data.get("recursive", False)
    resume = data.get("resume", False)
    trigger = data.get("trigger", "")
    purpose = data.get("purpose", "character")
    style_hint = data.get("style_hint", "")
    wd14_model = data.get("wd14_model") or data.get("model", "wd-eva02-large-tagger-v3")
    threshold = float(data.get("threshold", 0.35))
    char_threshold = float(data.get("char_threshold", 0.85))
    wd14_batch = int(data.get("wd14_batch", 8))
    use_cpu = data.get("use_cpu", False)
    data_dir = data.get("data_dir") or os.path.join(os.path.dirname(TAGGER_DIR), "..", "models")
    data_dir = os.path.abspath(data_dir)

    vlm_model = data.get("vlm_model", "toriigate-0.5")
    vllm_api_url = data.get("vllm_api_url", "")
    vllm_model = data.get("vllm_model", "")
    vlm_workers = min(max(1, int(data.get("vlm_workers", data.get("vlm_batch", 4)))), 32)
    vlm_max_tokens = int(data.get("vlm_max_tokens", 2048))
    temperature = float(data.get("temperature", data.get("vlm_temperature", 0.2)))
    use_alias_index = data.get("use_alias_index", True)
    auto_download_gemma = data.get("auto_download_gemma", True)
    auto_start_vllm = data.get("auto_start_vllm", False)
    gemma_vlm_backend = data.get("gemma_vlm_backend", "")

    preset = get_vlm_preset(vlm_model)
    if not vllm_api_url:
        vllm_api_url = str(preset.get("default_api_url", ""))
    if not vllm_model:
        vllm_model = str(preset.get("default_served_name", ""))

    project_root = Path(__file__).resolve().parents[2]
    pipeline_root = project_root / "tools"
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))

    _task_state.update({
        "status": "running",
        "progress": 0,
        "phase": "starting",
        "message": "Anima Train: 准备中...",
        "output_dir": output_dir,
        "results": [],
        "failed_count": 0,
        "last_error": "",
        "start_time": time.time(),
    })

    def _progress_callback(payload: dict) -> None:
        current = int(payload.get("current", 0))
        total = int(payload.get("total", 0))
        phase = str(payload.get("phase", ""))
        message = str(payload.get("message", ""))
        if total > 0:
            _task_state["progress"] = round(current / total * 100)
        _task_state["phase"] = phase
        _task_state["message"] = message

    def _worker():
        global _task_state
        try:
            from anima_caption_pipeline.runner import run_anima_train_batch

            results = run_anima_train_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=recursive,
                save_captions=save_captions,
                wd14_model=wd14_model,
                threshold=threshold,
                char_threshold=char_threshold,
                wd14_batch=wd14_batch,
                use_gpu=not use_cpu,
                data_dir=data_dir,
                vlm_model=vlm_model,
                vllm_api_url=vllm_api_url,
                vllm_model=vllm_model,
                vlm_workers=vlm_workers,
                vlm_max_tokens=vlm_max_tokens,
                temperature=temperature,
                trigger=trigger,
                purpose=purpose,
                style_hint=style_hint,
                use_alias_index=use_alias_index,
                auto_download_gemma=auto_download_gemma,
                auto_start_vllm=auto_start_vllm,
                gemma_vlm_backend=gemma_vlm_backend,
                vlm_preset=preset,
                resume=resume,
                progress_callback=_progress_callback,
                project_root=project_root,
            )

            failed_count = sum(1 for item in results if item.get("error"))
            _task_state["status"] = "done"
            _task_state["progress"] = 100
            _task_state["phase"] = "done"
            _task_state["results"] = results
            _task_state["failed_count"] = failed_count
            if failed_count:
                _task_state["message"] = f"Anima Train 完成: 成功 {len(results) - failed_count}/{len(results)}，失败 {failed_count}"
            else:
                _task_state["message"] = f"Anima Train 完成: {len(results)} 张"
            log.info("[TagLeaf Anima] 完成: %s 张, %s 失败", len(results), failed_count)
        except Exception as exc:
            log.error("[TagLeaf Anima] 异常: %s", exc)
            _task_state["status"] = "error"
            _task_state["message"] = str(exc)
            _task_state["last_error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()

    return APIResponseSuccess(
        message="Anima Train 标注已启动",
        data={"output_dir": output_dir},
    )


def _load_anima_train_vlm_models() -> list[dict]:
    config_path = Path(__file__).resolve().parents[2] / "config" / "anima_caption_models.json"
    models: list[dict] = []
    try:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for model_id, preset in (raw.get("vlm_models") or {}).items():
                models.append(
                    {
                        "id": model_id,
                        "label": preset.get("display_name", model_id),
                        "default_api_url": preset.get("default_api_url", ""),
                        "default_served_name": preset.get("default_served_name", ""),
                        "local_model_dir": preset.get("local_model_dir"),
                        "modelscope_id": preset.get("modelscope_id"),
                        "port": preset.get("port"),
                    }
                )
    except Exception as exc:
        log.warning("[TagLeaf] Failed to load anima_caption_models.json: %s", exc)
    if not models:
        models = [
            {
                "id": "toriigate-0.5",
                "label": "ToriiGate 0.5",
                "default_api_url": "http://127.0.0.1:18901/v1/chat/completions",
                "default_served_name": "toriigate-0.5",
                "local_model_dir": None,
            },
            {
                "id": "gemma-4-e4b",
                "label": "Gemma-4-E4B (vLLM)",
                "default_api_url": "http://127.0.0.1:9002/v1/chat/completions",
                "default_served_name": "spawner-gemma-4-e4b-it",
                "local_model_dir": "models/gemma-4-E3B-it",
                "modelscope_id": "spawner/spawner-gemma-4-E4B-it",
            },
        ]
    return models


@router.get("/vllm/status")
async def vllm_status(vlm_model: str = "toriigate-0.5"):
    """Return local vLLM health for the selected Anima Train VLM preset."""
    try:
        return APIResponseSuccess(data=get_vllm_status(vlm_model))
    except Exception as exc:
        return APIResponseFail(message=str(exc))


@router.post("/vllm/start")
async def vllm_start(request: Request):
    """Start vLLM for the selected Anima Train VLM model and wait until ready."""
    try:
        data = await request.json()
    except Exception:
        return APIResponseFail(message="无效的 JSON")

    vlm_model = data.get("vlm_model", "toriigate-0.5")
    auto_download_gemma = data.get("auto_download_gemma", True)
    wait_ready = data.get("wait_ready", True)
    timeout = float(data.get("timeout", 900))

    try:
        if wait_ready:
            status = ensure_vllm_ready(
                vlm_model,
                auto_download_gemma=auto_download_gemma,
                timeout=timeout,
            )
        else:
            status = start_vllm(
                vlm_model,
                auto_download_gemma=auto_download_gemma,
                wait_ready=False,
            )
        preset = get_vlm_preset(vlm_model)
        return APIResponseSuccess(
            message=status.get("message", "vLLM 已启动"),
            data={
                **status,
                "default_api_url": preset.get("default_api_url", ""),
                "default_served_name": preset.get("default_served_name", ""),
                "local_model_dir": preset.get("local_model_dir"),
            },
        )
    except Exception as exc:
        log.error("[TagLeaf] vLLM start failed: %s", exc)
        return APIResponseFail(message=str(exc))


@router.post("/vllm/stop")
async def vllm_stop(request: Request):
    """Stop the managed vLLM subprocess started by this API."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    vlm_model = data.get("vlm_model", "")
    try:
        status = stop_vllm(vlm_model)
        return APIResponseSuccess(message="vLLM 已停止", data=status)
    except Exception as exc:
        return APIResponseFail(message=str(exc))


@router.get("/models")
async def get_models():
    """List available tagger models."""
    return APIResponseSuccess(data={
        "models": AVAILABLE_MODELS,
        "purposes": AVAILABLE_PURPOSES,
        "vlm_backends": [
            {"id": "transformers", "label": "本地 Transformers（默认）"},
            {"id": "vllm", "label": "vLLM 服务（OpenAI API）"},
        ],
        "vlm_prompt_modes": [
            {"id": "lora", "label": "LoRA 训练目的模板", "needs_purpose": True},
            {"id": "official_short", "label": "ToriiGate 官方 · short", "needs_purpose": False},
            {"id": "official_long", "label": "ToriiGate 官方 · long", "needs_purpose": False},
            {
                "id": "official_min_structured_md",
                "label": "ToriiGate 官方 · min_structured_md",
                "needs_purpose": False,
            },
        ],
        "anima_train": {
            "vlm_models": _load_anima_train_vlm_models(),
        },
    })
