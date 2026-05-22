#!/usr/bin/env python3
"""Download the default WD tagger into HF_HOME cache (source install + portable build)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mikazuki.tagger.defaults import (  # noqa: E402
    DEFAULT_TAGGER_FILES,
    DEFAULT_TAGGER_KEY,
    DEFAULT_TAGGER_REPO_ID,
    DEFAULT_TAGGER_REVISION,
)


def _resolve_hf_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env).resolve()
    return (REPO_ROOT / "huggingface").resolve()


def is_default_tagger_cached(hf_home: Path | None = None) -> bool:
    hf_home = _resolve_hf_home(str(hf_home) if hf_home else None)
    os.environ["HF_HOME"] = str(hf_home)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    for filename in DEFAULT_TAGGER_FILES:
        try:
            hf_hub_download(
                repo_id=DEFAULT_TAGGER_REPO_ID,
                filename=filename,
                revision=DEFAULT_TAGGER_REVISION,
                local_files_only=True,
            )
        except Exception:
            return False
    return True


def ensure_default_tagger_model(
    hf_home: Path | None = None,
    *,
    use_china_mirror: bool = True,
    force: bool = False,
) -> Path:
    hf_home = _resolve_hf_home(str(hf_home) if hf_home else None)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    if use_china_mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if not force and is_default_tagger_cached(hf_home):
        print(f"[tagger] Default model already present ({DEFAULT_TAGGER_KEY})")
        print(f"         cache: {hf_home / 'hub'}")
        return hf_home

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run install-cn.ps1 / pip install -r requirements.txt first."
        ) from exc

    print(f"[tagger] Downloading default WD tagger: {DEFAULT_TAGGER_REPO_ID}")
    print(f"         revision={DEFAULT_TAGGER_REVISION} (~388 MB, please wait)")
    print(f"         HF_HOME={hf_home}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"         HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    for filename in DEFAULT_TAGGER_FILES:
        print(f"[tagger]   - {filename}")
        path = hf_hub_download(
            repo_id=DEFAULT_TAGGER_REPO_ID,
            filename=filename,
            revision=DEFAULT_TAGGER_REVISION,
        )
        print(f"[tagger]     -> {path}")

    print(f"[tagger] Done. WebUI tagging can use '{DEFAULT_TAGGER_KEY}' offline.")
    return hf_home


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        help="HF cache root (default: ./huggingface or HF_HOME env)",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Exit 0 without downloading when the model is already cached",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not set HF_ENDPOINT to hf-mirror.com",
    )
    args = parser.parse_args()

    hf_home = _resolve_hf_home(args.hf_home)

    if args.if_missing and is_default_tagger_cached(hf_home) and not args.force:
        return 0

    ensure_default_tagger_model(
        hf_home,
        use_china_mirror=not args.no_mirror,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
