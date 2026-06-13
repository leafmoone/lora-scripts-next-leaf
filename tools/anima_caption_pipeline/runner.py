"""Batch runner: WD14 → two-step VLM → anima_train_v1 captions."""

from __future__ import annotations

import gc
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .alias_index import AliasIndex
from .config import IMAGE_EXTENSIONS, PROJECT_ROOT, project_root_from
from .model_resolver import resolve_vlm_runtime, should_start_vllm_for_gemma
from .gemma_local_client import LocalGemmaVlmClient
from .pipeline import run_single_image_pipeline
from .vlm_client import create_vlm_client, is_gemma_vlm_model

logger = logging.getLogger(__name__)

TAGGER_DIR = PROJECT_ROOT / "tools" / "differential_tagger"


def _ensure_tagger_path() -> None:
    tagger_path = str(TAGGER_DIR)
    if tagger_path not in sys.path:
        sys.path.insert(0, tagger_path)


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    current: int,
    total: int,
    message: str,
) -> None:
    payload = {
        "type": "progress",
        "phase": phase,
        "current": current,
        "total": total,
        "message": message,
    }
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    if callback:
        callback(payload)


def _scan_images(input_dir: str, recursive: bool) -> list[Path]:
    root = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def _wd14_tags_to_string(wd14_row: dict[str, Any]) -> str:
    tags: list[str] = []
    for item in wd14_row.get("all_tags") or []:
        if isinstance(item, dict):
            tag = str(item.get("tag", "")).strip()
        else:
            tag = str(item).strip()
        if tag:
            tags.append(tag)
    return ", ".join(tags)


def _run_wd14_batch(
    image_paths: list[str],
    *,
    model_name: str,
    threshold: float,
    char_threshold: float,
    use_gpu: bool,
    wd14_batch: int,
    data_dir: str,
) -> list[dict[str, Any]]:
    _ensure_tagger_path()
    from main import run_simple  # type: ignore

    if data_dir:
        import os

        os.environ.setdefault("SD_TAGGER_DATA_DIR", data_dir)

    return run_simple(
        image_paths,
        model_name=model_name,
        threshold=threshold,
        character_threshold=char_threshold,
        use_gpu=use_gpu,
        wd14_batch_size=max(1, int(wd14_batch)),
    )


def _release_wd14_runtime() -> None:
    _ensure_tagger_path()
    try:
        from tagger import get_tagger  # type: ignore

        tagger = get_tagger(force_reload=False)
        if hasattr(tagger, "unload"):
            tagger.unload()
        elif hasattr(tagger, "close"):
            tagger.close()
    except Exception as exc:
        logger.debug("WD14 unload skipped: %s", exc)
    gc.collect()


def _stop_managed_vllm_before_wd14(
    vlm_model: str,
    *,
    project_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Free GPU for WD14 when a previous vLLM session is still running."""
    root_str = str(project_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from mikazuki.utils.vllm_manager import get_vllm_status, stop_vllm

    status = get_vllm_status(vlm_model)
    if not (status.get("ready") or status.get("managed") or status.get("status") in {"running", "starting"}):
        return

    _emit_progress(
        progress_callback,
        phase="wd14",
        current=0,
        total=0,
        message="检测到 vLLM 占用显存，正在停止以便 WD14 打标...",
    )
    stop_vllm(vlm_model)
    gc.collect()


def _ensure_vllm_if_needed(
    *,
    vlm_model: str,
    auto_download_gemma: bool,
    auto_start_vllm: bool,
    project_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    if not auto_start_vllm:
        return

    root_str = str(project_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from mikazuki.utils.vllm_manager import ensure_vllm_ready

    _emit_progress(
        progress_callback,
        phase="vllm",
        current=0,
        total=0,
        message="WD14 完成，正在启动 vLLM...",
    )
    ensure_vllm_ready(vlm_model, auto_download_gemma=auto_download_gemma)


def _stop_vllm_if_local_gemma(
    *,
    vlm_model: str,
    client: Any,
    project_root: Path,
) -> None:
    if not isinstance(client, LocalGemmaVlmClient):
        return
    if not is_gemma_vlm_model(vlm_model):
        return

    root_str = str(project_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from mikazuki.utils.vllm_manager import stop_vllm

    logger.info("Stopping managed vLLM to free GPU for local Gemma inference")
    stop_vllm(vlm_model)
    gc.collect()


def run_anima_train_batch(
    *,
    input_dir: str,
    output_dir: str | None = None,
    recursive: bool = False,
    save_captions: bool = True,
    wd14_model: str = "wd-eva02-large-tagger-v3",
    threshold: float = 0.35,
    char_threshold: float = 0.85,
    wd14_batch: int = 8,
    use_gpu: bool = True,
    data_dir: str = "",
    vlm_model: str = "toriigate-0.5",
    vllm_api_url: str = "",
    vllm_model: str = "",
    vlm_workers: int = 4,
    vlm_max_tokens: int = 2048,
    temperature: float = 0.2,
    trigger: str = "",
    purpose: str = "character",
    style_hint: str = "",
    use_alias_index: bool = True,
    auto_download_gemma: bool = True,
    auto_start_vllm: bool = False,
    gemma_vlm_backend: str = "",
    vlm_preset: dict[str, Any] | None = None,
    resume: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = project_root_from(project_root)
    out_dir = Path(output_dir or input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _scan_images(input_dir, recursive)
    if resume:
        image_paths = [path for path in image_paths if not path.with_suffix(".txt").is_file()]
    if not image_paths:
        _emit_progress(progress_callback, phase="done", current=0, total=0, message="未找到待处理图片")
        return []

    total = len(image_paths)
    path_strings = [str(path) for path in image_paths]

    if auto_start_vllm and use_gpu:
        _stop_managed_vllm_before_wd14(
            vlm_model,
            project_root=root,
            progress_callback=progress_callback,
        )

    _emit_progress(progress_callback, phase="wd14", current=0, total=total, message=f"WD14 打标 0/{total}")
    wd14_results = _run_wd14_batch(
        path_strings,
        model_name=wd14_model,
        threshold=threshold,
        char_threshold=char_threshold,
        use_gpu=use_gpu,
        wd14_batch=wd14_batch,
        data_dir=data_dir,
    )
    wd14_by_path = {str(row.get("image_path", "")): row for row in wd14_results}
    _emit_progress(progress_callback, phase="wd14", current=total, total=total, message=f"WD14 完成 {total}/{total}")

    _release_wd14_runtime()

    runtime = resolve_vlm_runtime(
        vlm_model,
        user_url=vllm_api_url,
        user_served_name=vllm_model,
        project_root=root,
        auto_download_gemma=auto_download_gemma,
        gemma_vlm_backend=gemma_vlm_backend,
        preset=vlm_preset,
    )
    backend = str(runtime.get("gemma_vlm_backend") or "auto")
    if auto_start_vllm and should_start_vllm_for_gemma(backend):
        _ensure_vllm_if_needed(
            vlm_model=vlm_model,
            auto_download_gemma=auto_download_gemma,
            auto_start_vllm=True,
            project_root=root,
            progress_callback=progress_callback,
        )
    elif auto_start_vllm and backend == "transformers":
        _emit_progress(
            progress_callback,
            phase="vlm",
            current=0,
            total=0,
            message="Gemma 使用本地 transformers，跳过 vLLM 启动",
        )

    client = create_vlm_client(
        vlm_model=vlm_model,
        api_url=str(runtime["api_url"]),
        model_name=str(runtime["served_name"]),
        local_model_dir=runtime.get("local_model_dir"),
        max_tokens=vlm_max_tokens,
        temperature=temperature,
        gemma_vlm_backend=backend,
    )
    _stop_vllm_if_local_gemma(vlm_model=vlm_model, client=client, project_root=root)
    alias_index = AliasIndex(enabled=use_alias_index)

    results: list[dict[str, Any]] = []
    completed = 0
    failed_count = 0
    lock = threading.Lock()

    def process_one(image_path: Path) -> dict[str, Any]:
        path_str = str(image_path)
        wd14_row = wd14_by_path.get(path_str, {})
        if wd14_row.get("error"):
            raise RuntimeError(str(wd14_row["error"]))
        raw_tags = _wd14_tags_to_string(wd14_row)
        enriched = run_single_image_pipeline(
            client,
            path_str,
            raw_tags=raw_tags,
            purpose=purpose,
            style_hint=style_hint,
            trigger=trigger,
            alias_index=alias_index,
        )
        training_text = str(enriched.get("formatted_training_text", "")).strip()
        if trigger and training_text:
            lines = training_text.split("\n\n", 1)
            tag_line = lines[0]
            caption_line = lines[1] if len(lines) > 1 else ""
            if trigger.lower() not in tag_line.lower():
                tag_line = f"{trigger}, {tag_line}" if tag_line else trigger
            training_text = tag_line + ("\n\n" + caption_line if caption_line else "")
            enriched["formatted_training_text"] = training_text
            enriched["formatted_prompt"] = training_text

        if save_captions and training_text:
            image_path.with_suffix(".txt").write_text(training_text, encoding="utf-8")

        return {
            "image_path": path_str,
            "wd14_raw_tags": raw_tags,
            "caption": training_text,
            "formatted_training_text": training_text,
            "all_tags": wd14_row.get("all_tags", []),
            "vlm_result": enriched,
        }

    _emit_progress(progress_callback, phase="vlm", current=0, total=total, message=f"VLM 两步链 0/{total}")
    workers = 1 if isinstance(client, LocalGemmaVlmClient) else max(1, int(vlm_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, image_path): image_path for image_path in image_paths}
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                failed_count += 1
                logger.warning("Anima Train failed for %s: %s", image_path, exc)
                results.append(
                    {
                        "image_path": str(image_path),
                        "error": str(exc),
                        "all_tags": wd14_by_path.get(str(image_path), {}).get("all_tags", []),
                    }
                )
            with lock:
                completed += 1
                _emit_progress(
                    progress_callback,
                    phase="vlm",
                    current=completed,
                    total=total,
                    message=f"VLM 两步链 [{completed}/{total}] {image_path.name}",
                )

    results.sort(key=lambda item: item.get("image_path", ""))
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    ok_count = total - failed_count
    _emit_progress(
        progress_callback,
        phase="save",
        current=total,
        total=total,
        message=f"保存完成: 成功 {ok_count}/{total}，失败 {failed_count}",
    )
    return results
