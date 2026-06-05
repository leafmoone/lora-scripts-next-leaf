"""
DiffSynth Tag Editor (Tag-Edit-Leaf) REST API

独立图片标注器，调用 tools/differential_tagger/main.py
支持 simple 和 smart 两种模式，适用于任意图片文件夹。
"""

import asyncio
import os
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

    # Quick scan via ls/find
    p = Path(path)
    pattern = "**/*" if recursive else "*"
    images = []
    for f in sorted(p.glob(pattern)):
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}:
            images.append({"name": f.name, "path": str(f)})

    return APIResponseSuccess(data={
        "path": path,
        "count": len(images),
        "preview": images[:50],
    })


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

    # Build command
    python_exe = sys.executable
    main_py = os.path.join(TAGGER_DIR, "main.py")

    cmd = [python_exe, main_py, "--input", input_dir, "--output", output_dir]

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

    _task_state = {
        "status": "running",
        "progress": 0,
        "message": "正在扫描图片...",
        "output_dir": output_dir,
        "results": [],
        "start_time": time.time(),
    }

    def _run():
        global _task_state
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=TAGGER_DIR,
            )
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                # Parse progress like: [5/100] 5.0% — Tagging image_005.jpg
                if "[" in line and "/" in line and "%" in line:
                    try:
                        bracket = line[line.index("["):line.index("]") + 1]
                        parts = bracket.strip("[]").split("/")
                        current = int(parts[0])
                        total = int(parts[1])
                        pct = round(current / total * 100) if total > 0 else 0
                        _task_state["progress"] = pct
                        _task_state["message"] = line
                    except Exception:
                        pass
            proc.wait()

            if proc.returncode == 0:
                # Collect results
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
                _task_state["status"] = "error"
                _task_state["message"] = f"标注失败 (exit {proc.returncode})"
                log.error(f"[TagLeaf] 标注失败 (exit {proc.returncode})")
        except Exception as e:
            _task_state["status"] = "error"
            _task_state["message"] = str(e)
            log.error(f"[TagLeaf] 异常: {e}")

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
        "results": _task_state.get("results", [])[:20],  # First 20 for preview
        "elapsed_seconds": elapsed,
    })


@router.get("/models")
async def get_models():
    """List available tagger models."""
    return APIResponseSuccess(data={
        "models": AVAILABLE_MODELS,
        "purposes": AVAILABLE_PURPOSES,
    })
