#!/usr/bin/env python3
"""
Standalone AI Image Tagger — CLI entry point.

Two modes:
  simple   — WD14 ONNX booru tagger only
  smart    — Full pipeline: WD14 tagging → noise strip → ToriiGate VLM → caption

Usage:
  # Simple mode: WD14 tagging only
  python main.py --input ./images/ --simple

  # Smart Tag mode with ToriiGate VLM
  python main.py --input ./images/ --smart --vlm --purpose general

  # Smart Tag with multi-tagger consensus
  python main.py --input ./images/ --smart --vlm --taggers wd-swinv2-tagger-v3 wd-vit-tagger-v3

  # CPU only
  python main.py --input ./images/ --simple --cpu

  # Custom output directory
  python main.py --input ./images/ --output ./results/ --smart --vlm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the package directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pick up --data-dir early (before config.py reads STANDALONE_TAGGER_DATA_DIR at import).
# Default download path is ./models/tagger; pass an explicit directory to reuse cached models.
_data_dir_candidates = ["--data-dir"]
_data_dir = None
for _i, _arg in enumerate(sys.argv):
    if _arg in _data_dir_candidates and _i + 1 < len(sys.argv):
        _data_dir = os.path.abspath(sys.argv[_i + 1])
        break
os.environ["STANDALONE_TAGGER_DATA_DIR"] = (
    _data_dir if _data_dir is not None else os.path.abspath("./models")
)

from config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DEFAULT_TAGGER_MODEL,
    TAGGER_MODELS,
    TAGGER_GENERAL_THRESHOLD,
    TAGGER_CHARACTER_THRESHOLD,
    TAGGER_USE_GPU,
    get_wd14_model_dir,
)

logger = logging.getLogger("standalone-tagger")


def discover_images(paths: List[str], recursive: bool = False) -> List[str]:
    """Find all image files from a list of dirs/files."""
    discovered: List[str] = []
    seen: set = set()

    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            s = str(p)
            if s not in seen:
                seen.add(s)
                discovered.append(s)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in sorted(p.glob(pattern)):
                if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    s = str(f)
                    if s not in seen:
                        seen.add(s)
                        discovered.append(s)

    return discovered


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a tagging result for JSON output."""
    out: Dict[str, Any] = {
        "image_path": result.get("image_path", ""),
        "rating": result.get("rating", "unknown"),
    }
    if "error" in result and result["error"]:
        out["error"] = result["error"]
        return out

    out["general_tags"] = result.get("general_tags", [])
    out["character_tags"] = result.get("character_tags", [])
    out["copyright_tags"] = result.get("copyright_tags", [])
    out["all_tags"] = result.get("all_tags", [])

    if "ai_caption" in result:
        out["ai_caption"] = result["ai_caption"]
    if "noise_stripped_count" in result:
        out["noise_stripped_count"] = result["noise_stripped_count"]
    if "nl_text" in result and result["nl_text"]:
        out["nl_text"] = result["nl_text"]

    return out


def _format_simple_result(
    raw: Dict[str, Any],
    *,
    threshold: float,
    character_threshold: float,
    blacklist: Optional[set] = None,
    max_tags: int = 0,
) -> Dict[str, Any]:
    """Normalize a single WD14 ``tag()`` / ``tag_batch()`` row for ``run_simple``."""
    all_tags = list(raw.get("all_tags") or [])
    if blacklist:
        all_tags = [
            t for t in all_tags
            if str(t.get("tag", "")).strip().lower() not in blacklist
        ]
    if max_tags > 0 and len(all_tags) > max_tags:
        all_tags.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        all_tags = all_tags[:max_tags]

    result: Dict[str, Any] = {
        "image_path": raw.get("image_path", ""),
        "rating": raw.get("rating", "unknown"),
        "rating_confidences": raw.get("rating_confidences", {}),
        "general_tags": raw.get("general_tags", []),
        "character_tags": raw.get("character_tags", []),
        "copyright_tags": raw.get("copyright_tags", []),
        "all_tags": all_tags,
    }
    if raw.get("error"):
        result["error"] = raw["error"]
    return result


def run_simple(
    image_paths: List[str],
    *,
    model_name: str = DEFAULT_TAGGER_MODEL,
    threshold: float = TAGGER_GENERAL_THRESHOLD,
    character_threshold: float = TAGGER_CHARACTER_THRESHOLD,
    use_gpu: bool = TAGGER_USE_GPU,
    blacklist: Optional[List[str]] = None,
    max_tags: int = 0,
    progress_callback=None,
    wd14_batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Run simple WD14 tagging on a list of images."""
    from tagger import get_tagger

    tagger = get_tagger(
        model_name=model_name,
        threshold=threshold,
        character_threshold=character_threshold,
        use_gpu=use_gpu,
        force_reload=True,
    )
    if hasattr(tagger, "load"):
        tagger.load()

    blacklist_lower = set()
    if blacklist:
        blacklist_lower = {t.strip().lower() for t in blacklist if t.strip()}

    results = []
    total = len(image_paths)

    # ── Batch path ──────────────────────────────────────────
    if wd14_batch_size > 1 and hasattr(tagger, "tag_batch"):
        batch_results = tagger.tag_batch(
            image_paths,
            preferred_batch_size=wd14_batch_size,
            threshold=threshold,
            character_threshold=character_threshold,
        )
        for path, raw in zip(image_paths, batch_results):
            if raw is None:
                raw = {"error": "empty batch result"}
            else:
                raw = dict(raw)
            raw.setdefault("image_path", path)
            out = _format_simple_result(
                raw,
                threshold=threshold,
                character_threshold=character_threshold,
                blacklist=blacklist_lower or None,
                max_tags=max_tags,
            )
            results.append(out)
        return results

    # ── Sequential fallback ─────────────────────────────────
    for idx, path in enumerate(image_paths):
        if progress_callback:
            progress_callback(idx + 1, total, f"Tagging {idx + 1}/{total}")

        try:
            raw = tagger.tag(
                path,
                threshold=threshold,
                character_threshold=character_threshold,
            )
            raw.setdefault("image_path", path)
        except Exception as exc:
            logger.error("Failed to tag %s: %s", path, exc)
            results.append({"image_path": path, "error": str(exc)})
            continue

        results.append(
            _format_simple_result(
                raw,
                threshold=threshold,
                character_threshold=character_threshold,
                blacklist=blacklist_lower or None,
                max_tags=max_tags,
            )
        )

    return results


def run_smart(
    image_paths: List[str],
    *,
    training_purpose: str = "general",
    trigger_word: str = "",
    tagger_model: str = DEFAULT_TAGGER_MODEL,
    use_gpu: bool = TAGGER_USE_GPU,
    general_threshold: float = TAGGER_GENERAL_THRESHOLD,
    character_threshold: float = TAGGER_CHARACTER_THRESHOLD,
    max_tags: int = 0,
    enable_vlm: bool = True,
    enable_wd14: bool = True,
    taggers: Optional[List[Dict[str, Any]]] = None,
    consensus_min: int = 2,
    progress_callback=None,
    wd14_batch_size: int = 8,
    vlm_batch_size: int = 4,
    vlm_backend: str = "transformers",
    vllm_api_url: str = "",
    vllm_model: str = "",
    vlm_prompt_mode: str = "lora",
    inject_wd14_tags: bool = True,
    blacklist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run Smart Tag pipeline."""
    from smart_tag import run_smart_tag_pipeline

    smart_results = run_smart_tag_pipeline(
        image_paths=image_paths,
        training_purpose=training_purpose,
        trigger_word=trigger_word,
        auto_strip_noise=True,
        enable_wd14=enable_wd14,
        enable_vlm=enable_vlm,
        tagger_model=tagger_model,
        use_gpu=use_gpu,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        max_tags_per_image=max_tags,
        taggers=taggers,
        consensus_min=consensus_min,
        progress_callback=progress_callback,
        wd14_batch_size=wd14_batch_size,
        vlm_batch_size=vlm_batch_size,
        vlm_backend=vlm_backend,
        vllm_api_url=vllm_api_url,
        vllm_model=vllm_model,
        vlm_prompt_mode=vlm_prompt_mode,
        inject_wd14_tags=inject_wd14_tags,
        blacklist=blacklist,
    )

    results: List[Dict[str, Any]] = []
    for r in smart_results:
        entry: Dict[str, Any] = {
            "image_path": r.image_path,
            "rating": r.rating or "unknown",
            "general_tags": [
                {"tag": t} for t in r.general_tags
            ],
            "copyright_tags": [
                {"tag": t} for t in r.copyright_tags
            ],
            "character_tags": [
                {"tag": t} for t in r.character_tags
            ],
            "ai_caption": r.caption,
            "nl_text": r.nl_text,
            "noise_stripped_count": r.noise_stripped_count,
        }
        if r.error:
            entry["error"] = r.error
        else:
            # Build all_tags from categories
            all_tags: List[Dict[str, Any]] = []
            for t in r.character_tags:
                all_tags.append({"tag": t, "category": "character"})
            for t in r.general_tags:
                all_tags.append({"tag": t, "category": "general"})
            for t in r.copyright_tags:
                all_tags.append({"tag": t, "category": "copyright"})
            entry["all_tags"] = all_tags

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone AI Image Tagger — WD14 booru tags + ToriiGate VLM captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, nargs="+",
        help="Input image directories or files",
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively scan input directories",
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--simple", action="store_true", default=True,
        help="Simple mode: WD14 tagging only (default)",
    )
    mode.add_argument(
        "--smart", action="store_true",
        help="Smart mode: WD14 + noise-strip + ToriiGate VLM + caption assembly",
    )

    # Tagger options
    parser.add_argument(
        "--model", "-m", default=DEFAULT_TAGGER_MODEL,
        help=f"WD14 model name (default: {DEFAULT_TAGGER_MODEL})",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=TAGGER_GENERAL_THRESHOLD,
        help=f"Confidence threshold (default: {TAGGER_GENERAL_THRESHOLD})",
    )
    parser.add_argument(
        "--character-threshold", type=float, default=TAGGER_CHARACTER_THRESHOLD,
        help=f"Character tag threshold (default: {TAGGER_CHARACTER_THRESHOLD})",
    )

    # GPU / CPU
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use CPU only (default: use GPU if available)",
    )

    # Smart Tag options
    parser.add_argument(
        "--purpose", default="general",
        choices=["style", "character", "general", "concept"],
        help="Training purpose for VLM caption (default: general)",
    )
    parser.add_argument(
        "--trigger", default="",
        help="Trigger word to inject at the start of captions",
    )
    parser.add_argument(
        "--vlm", action="store_true", default=False,
        help="Enable ToriiGate VLM for natural language captions (smart mode)",
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Disable VLM even in smart mode",
    )
    parser.add_argument(
        "--no-wd14", action="store_true",
        help="Disable WD14 booru tagging in smart mode (VLM-only)",
    )
    parser.add_argument(
        "--vlm-prompt-mode",
        default="lora",
        choices=["lora", "official_short", "official_long", "official_min_structured_md"],
        help="ToriiGate VLM user prompt template (default: lora purpose presets)",
    )
    parser.add_argument(
        "--no-inject-wd14-tags",
        action="store_true",
        help="Do not inject WD14 tags into ToriiGate user prompt",
    )

    # Multi-tagger consensus
    parser.add_argument(
        "--taggers", nargs="+", default=[],
        help="Multiple tagger model names for consensus (smart mode)",
    )
    parser.add_argument(
        "--consensus", type=int, default=2,
        help="Min tagger weight sum for consensus (default: 2)",
    )

    # Filters
    parser.add_argument(
        "--max-tags", type=int, default=0,
        help="Max tags per image (0 = unlimited)",
    )
    parser.add_argument(
        "--blacklist", nargs="+", default=[],
        help="Tags to filter out (e.g., 'watermark' 'signature')",
    )

    # Output
    parser.add_argument(
        "--save-captions", action="store_true",
        help="Save individual .txt caption files alongside results.json",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    # Parallelism
    parser.add_argument(
        "--wd14-batch", type=int, default=8,
        help="Batch size for WD14 ONNX inference (default: 8). "
             "Higher values trade VRAM for speed."
    )
    parser.add_argument(
        "--vlm-batch", type=int, default=4,
        help="Batch size for local Transformers VLM, or HTTP concurrency for vLLM (default: 4). "
             "WD14 completes first, then VLM runs with a shared prompt template."
    )
    parser.add_argument(
        "--vlm-backend",
        default="transformers",
        choices=["transformers", "vllm", "toriigate"],
        help="ToriiGate runtime: local HuggingFace transformers (default) or remote vLLM server",
    )
    parser.add_argument(
        "--vllm-api-url",
        default="",
        help="vLLM OpenAI API URL (default: SD_TORIIGATE_VLLM_API_URL or http://127.0.0.1:18901/v1/chat/completions)",
    )
    parser.add_argument(
        "--vllm-model",
        default="",
        help="Model name exposed by vLLM server (default: SD_TORIIGATE_VLLM_MODEL or toriigate-0.5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images that already have a non-empty .txt caption file",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override models/data directory (default: tools/differential_tagger/data/). "
             "Set to a custom path to reuse pre-downloaded models.",
    )

    args = parser.parse_args()

    # Apply data-dir override before config loads defaults
    if args.data_dir:
        os.environ["STANDALONE_TAGGER_DATA_DIR"] = os.path.abspath(args.data_dir)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discover images
    print(f"Scanning for images in: {', '.join(args.input)}")
    image_paths = discover_images(args.input, recursive=args.recursive)

    if not image_paths:
        print("ERROR: No images found.", file=sys.stderr)
        sys.exit(1)

    # ── Resume mode: skip already-tagged images ────────────
    if args.resume:
        original = len(image_paths)
        pending = []
        for p in image_paths:
            txt = os.path.splitext(p)[0] + ".txt"
            if os.path.isfile(txt) and os.path.getsize(txt) > 10:
                continue
            pending.append(p)
        skipped = original - len(pending)
        print(f"Resume mode: {skipped}/{original} images already tagged, {len(pending)} remaining")
        if not pending:
            print("All images already tagged. Nothing to do.")
            sys.exit(0)
        image_paths = pending

    print(f"Found {len(image_paths)} image(s)")

    # Prepare output
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = not args.cpu

    # List available models
    available_models = list(TAGGER_MODELS.keys())
    print(f"Available tagger models: {', '.join(available_models[:7])}...")

    def progress(current, total, message):
        pct = current / total * 100 if total > 0 else 0
        text = f"  [{current}/{total}] {pct:5.1f}% - {message}"
        print(
            json.dumps(
                {
                    "type": "progress",
                    "phase": "tagging",
                    "current": current,
                    "total": total,
                    "message": message,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if sys.stdout.isatty():
            print(f"\r{text}", end="", flush=True)
        else:
            print(text, flush=True)

    start_time = time.time()

    if args.smart:
        enable_vlm = args.vlm and not args.no_vlm
        enable_wd14 = not args.no_wd14
        print(
            f"Smart Tag mode: purpose={args.purpose}, wd14={'on' if enable_wd14 else 'off'}, "
            f"vlm={'ToriiGate' if enable_vlm else 'disabled'}, "
            f"vlm_prompt={args.vlm_prompt_mode}, "
            f"inject_wd14_tags={not args.no_inject_wd14_tags}"
        )
        if args.taggers:
            print(f"Multi-tagger consensus: {args.taggers} (min votes: {args.consensus})")

        tagger_configs = None
        if args.taggers:
            from smart_tag import _tagger_defaults
            tagger_configs = []
            for name in args.taggers:
                defaults = _tagger_defaults(name)
                tagger_configs.append({
                    "model": name,
                    "weight": 1.0,
                    "general_threshold": args.threshold,
                    "character_threshold": args.character_threshold,
                    "copyright_threshold": args.threshold,
                })

        if not enable_wd14 and not enable_vlm:
            print("Error: smart mode requires at least WD14 (--no-wd14 off) or VLM (--vlm).")
            sys.exit(1)

        results = run_smart(
            image_paths=image_paths,
            training_purpose=args.purpose,
            trigger_word=args.trigger,
            tagger_model=args.model,
            use_gpu=use_gpu,
            general_threshold=args.threshold,
            character_threshold=args.character_threshold,
            max_tags=args.max_tags,
            enable_vlm=enable_vlm,
            enable_wd14=enable_wd14,
            taggers=tagger_configs,
            consensus_min=args.consensus,
            progress_callback=progress,
            wd14_batch_size=args.wd14_batch,
            vlm_batch_size=args.vlm_batch,
            vlm_backend=args.vlm_backend,
            vllm_api_url=args.vllm_api_url,
            vllm_model=args.vllm_model,
            vlm_prompt_mode=args.vlm_prompt_mode,
            inject_wd14_tags=not args.no_inject_wd14_tags,
            blacklist=args.blacklist if args.blacklist else None,
        )
    else:
        print(f"Simple mode: model={args.model}, threshold={args.threshold}")
        results = run_simple(
            image_paths=image_paths,
            model_name=args.model,
            threshold=args.threshold,
            character_threshold=args.character_threshold,
            use_gpu=use_gpu,
            blacklist=args.blacklist if args.blacklist else None,
            max_tags=args.max_tags,
            progress_callback=progress,
            wd14_batch_size=args.wd14_batch,
        )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Processed {len(results)} image(s).")

    # Save results.json
    formatted = [format_result(r) for r in results]
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(formatted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to: {results_path}")

    # Save individual caption files alongside original images
    if args.save_captions:
        saved = 0
        skipped = 0
        for r in formatted:
            if "error" in r:
                continue
            raw_path = (r.get("image_path") or "").strip()
            if not raw_path:
                logger.warning("Skipping caption save: result has no image_path")
                skipped += 1
                continue
            image_path = Path(raw_path)
            caption_file = image_path.with_suffix(".txt")
            caption = r.get("ai_caption", "") or ", ".join(
                t.get("tag", "") for t in r.get("all_tags", [])
            )
            caption_file.write_text(caption, encoding="utf-8")
            saved += 1
        print(f"Saved {saved} caption .txt files alongside images")
        if skipped:
            print(f"Skipped {skipped} result(s) with missing image_path")

    # Summary
    errors = [r for r in formatted if "error" in r]
    if errors:
        print(f"\n{len(errors)} image(s) failed:")
        for e in errors[:5]:
            print(f"  - {e['image_path']}: {e['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def auto_tag_images(
    input_path: str,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    trigger: str = "",
    use_vlm: bool = False,
    purpose: str = "general",
    max_tags: int = 0,
    recursive: bool = False,
    verbose: bool = False,
):
    """为目录中所有图片自动生成 .txt 标签文件。

    每张图片 img.png 会生成 img.txt，内容为逗号分隔的标签列表。
    如果 .txt 已存在且内容大于 10 字节则跳过，避免重复标注。
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    input_list = [input_path]
    image_paths = discover_images(input_list, recursive=recursive)
    if not image_paths:
        print(f"[tagger] 未找到图片: {input_path}")
        return

    # 跳过已有标签文件的图片
    pending = []
    skipped = 0
    for p in image_paths:
        tag_file = os.path.splitext(p)[0] + ".txt"
        if os.path.isfile(tag_file) and os.path.getsize(tag_file) > 10:
            skipped += 1
            continue
        pending.append(p)

    if skipped:
        print(f"[tagger] 跳过 {skipped} 张已有标签的图片")
    if not pending:
        print(f"[tagger] 所有 {len(image_paths)} 张图片均已有标签，无需标注")
        return

    print(f"[tagger] 开始标注 {len(pending)} 张图片... (threshold={general_threshold})")

    if use_vlm:
        results = run_smart(
            image_paths=pending,
            training_purpose=purpose,
            trigger_word=trigger,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            max_tags=max_tags,
            enable_vlm=True,
        )
        for r in results:
            img_path = r.get("image_path", "")
            if "error" in r:
                continue
            caption = r.get("ai_caption", "")
            tag_file = os.path.splitext(img_path)[0] + ".txt"
            with open(tag_file, "w", encoding="utf-8") as f:
                f.write(caption)
    else:
        results = run_simple(
            image_paths=pending,
            threshold=general_threshold,
            character_threshold=character_threshold,
            max_tags=max_tags,
        )
        for r in results:
            img_path = r.get("image_path", "")
            if "error" in r:
                continue
            tags = [t["tag"] for t in r.get("all_tags", [])]
            if trigger:
                tags.insert(0, trigger)
            tag_file = os.path.splitext(img_path)[0] + ".txt"
            with open(tag_file, "w", encoding="utf-8") as f:
                f.write(", ".join(tags))

    tagged = sum(1 for r in results if "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    print(f"[tagger] 完成: {tagged} 张标注成功" + (f", {errors} 张失败" if errors else ""))


if __name__ == "__main__":
    main()
