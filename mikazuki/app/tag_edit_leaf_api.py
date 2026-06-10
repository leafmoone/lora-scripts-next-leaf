"""
DiffSynth Tag Editor (Tag-Edit-Leaf) REST API

独立图片标注器，调用 tools/differential_tagger/main.py
支持 simple 和 smart 两种模式，适用于任意图片文件夹。
"""

import json
import os
import base64
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request

from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.log import log

router = APIRouter(prefix="/api/tag-edit-leaf")

TAGGER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools", "differential_tagger")
)

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


_task_state: dict = {"status": "idle", "progress": 0, "message": "", "output_dir": "", "results": []}


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
    img_path = Path(root) / img
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
    use_vlm = data.get("use_vlm", True)
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

    # Build command
    cmd = [sys.executable, os.path.join(TAGGER_DIR, "main.py"),
           "--data-dir", data_dir,
           "--input", input_dir, "--output", output_dir]

    if mode == "smart":
        cmd.append("--smart")
    else:
        cmd.append("--simple")

    cmd.extend(["--model", model])
    cmd.extend(["--threshold", str(threshold)])
    cmd.extend(["--character-threshold", str(char_threshold)])

    if use_cpu:
        cmd.append("--cpu")
    if recursive:
        cmd.append("--recursive")
    if save_captions:
        cmd.append("--save-captions")
    if trigger:
        cmd.extend(["--trigger", trigger])
    if data.get("wd14_batch", 1) > 1:
        cmd.extend(["--wd14-batch", str(max(1, int(data["wd14_batch"])))])
    if data.get("vlm_workers", 1) > 1:
        cmd.extend(["--vlm-workers", str(max(1, int(data["vlm_workers"])))])
    if data.get("resume", False):
        cmd.append("--resume")

    if mode == "smart":
        cmd.extend(["--purpose", purpose])
        if use_vlm:
            cmd.append("--vlm")
        else:
            cmd.append("--no-vlm")

        if len(taggers) >= 2:
            cmd.append("--taggers")
            cmd.extend(taggers)
            cmd.extend(["--consensus", str(consensus)])

    if max_tags > 0:
        cmd.extend(["--max-tags", str(max_tags)])
    if blacklist:
        cmd.extend(["--blacklist"] + [t.strip() for t in blacklist if isinstance(t, str)])
    if verbose:
        cmd.append("--verbose")

    log.info(f"[TagLeaf] {' '.join(cmd)}")

    _task_state.update({
        "status": "running",
        "progress": 0,
        "message": "正在扫描图片...",
        "output_dir": output_dir,
        "results": [],
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
                # Update frontend message
                _task_state["message"] = stripped[-200:] if len(stripped) > 200 else stripped

                # Parse [N/M] tagging progress (highest authority)
                import re
                m = re.search(r'\[(\d+)\s*/\s*(\d+)\]', stripped)
                if m:
                    try:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        if tot > 0:
                            _task_state["progress"] = round(cur / tot * 100)
                    except (ValueError, IndexError):
                        pass
                    continue

                # Parse download progress from any % line
                m2 = re.search(r'(?:^|\r)\s*(\d+)%', stripped)
                if m2:
                    try:
                        _task_state["progress"] = min(int(m2.group(1)), 99)
                    except (ValueError, IndexError):
                        pass
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
        "message": _task_state["message"],
        "output_dir": _task_state.get("output_dir", ""),
        "results_count": len(_task_state.get("results", [])),
        "results": _task_state.get("results", [])[:20],
        "elapsed_seconds": elapsed,
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
    api_max_tokens = data.get("api_max_tokens", 256)
    api_temperature = data.get("api_temperature", 0.3)

    _task_state.update({
        "status": "running", "progress": 0,
        "message": "API 标注: 扫描图片...",
        "output_dir": output_dir, "results": [],
        "start_time": time.time(),
    })

    def _worker():
        global _task_state
        try:
            import requests

            # Scan images
            p = Path(input_dir)
            pattern = "**/*" if recursive else "*"
            image_files = sorted(f for f in p.glob(pattern) if f.suffix.lower() in
                {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"})
            if not image_files:
                _task_state["status"] = "error"
                _task_state["message"] = "未找到图片文件"
                return

            results = []
            # Build OpenAI-compatible request (Anthropic via dedicated endpoint)
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

            # Prepare system prompt
            system_msg = api_system_prompt
            if trigger:
                system_msg += f"\nAlways include the trigger word \"{trigger}\" at the beginning."

            for idx, img_path in enumerate(image_files):
                base_name = img_path.stem
                _task_state["progress"] = round(idx / len(image_files) * 100)
                _task_state["message"] = f"API 标注: [{idx+1}/{len(image_files)}] {img_path.name}"

                # Base64 encode image
                img_bytes = img_path.read_bytes()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                ext = img_path.suffix.lower().lstrip(".")
                data_uri = f"data:image/{ext};base64,{img_b64}"

                # Build messages
                if is_openai:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": "Tag this image:"},
                        ]},
                    ]
                    payload = {"model": api_model, "messages": messages,
                               "max_tokens": api_max_tokens, "temperature": api_temperature}
                else:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": f"image/{ext}", "data": img_b64}},
                            {"type": "text", "text": "Tag this image:"},
                        ]},
                    ]
                    payload = {"model": api_model, "messages": messages,
                               "max_tokens": api_max_tokens, "temperature": api_temperature,
                               "system": system_msg}

                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    resp.raise_for_status()
                    body = resp.json()
                    if is_openai:
                        content = body["choices"][0]["message"]["content"]
                    else:
                        content = body["content"][0]["text"]
                except Exception as exc:
                    log.warning(f"[TagLeaf API] {img_path.name} 请求失败: {exc}")
                    results.append({"image_path": str(img_path), "error": str(exc), "all_tags": []})
                    continue

                tags = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
                if trigger and tags and trigger not in tags[0]:
                    tags.insert(0, trigger)

                # Save .txt
                if save_captions:
                    txt_path = img_path.with_suffix(".txt")
                    txt_path.write_text(", ".join(tags), encoding="utf-8")

                results.append({
                    "image_path": str(img_path),
                    "caption": content,
                    "all_tags": [{"tag": t, "category": "general"} for t in tags],
                })

            # Save results.json
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            _task_state["status"] = "done"
            _task_state["progress"] = 100
            _task_state["message"] = f"API 标注完成: {len(results)} 张"
            _task_state["results"] = results
            log.info(f"[TagLeaf API] 完成: {len(results)} 张图片")

        except Exception as exc:
            log.error(f"[TagLeaf API] 异常: {exc}")
            _task_state["status"] = "error"
            _task_state["message"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()

    return APIResponseSuccess(
        message="API 标注已启动",
        data={"output_dir": output_dir},
    )


@router.get("/models")
async def get_models():
    """List available tagger models."""
    return APIResponseSuccess(data={
        "models": AVAILABLE_MODELS,
        "purposes": AVAILABLE_PURPOSES,
    })
