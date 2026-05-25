"""Download interrogator assets from Hugging Face with progress reporting."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import tqdm as hf_hub_tqdm

from mikazuki.tagger.progress import TaggerCancelled, tagger_progress

if TYPE_CHECKING:
    from mikazuki.tagger.interrogators.base import Interrogator


def _asset_filenames(interrogator: "Interrogator") -> list[str]:
    model_path = getattr(interrogator, "model_path", None)
    tags_path = getattr(interrogator, "tags_path", None)
    mapping_path = getattr(interrogator, "tag_mapping_path", None)
    files: list[str] = []
    if model_path:
        files.append(str(model_path))
    if tags_path:
        files.append(str(tags_path))
    if mapping_path:
        files.append(str(mapping_path))
    return files


def _hf_kwargs(interrogator: "Interrogator") -> dict:
    kwargs = dict(getattr(interrogator, "kwargs", {}) or {})
    if not kwargs.get("repo_id"):
        raise ValueError("interrogator 未配置 Hugging Face repo_id")
    return kwargs


def _file_cached(kwargs: dict, filename: str) -> bool:
    revision = kwargs.get("revision")
    repo_id = kwargs["repo_id"]
    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    if cached is not None:
        return True
    try:
        hf_hub_download(**kwargs, filename=filename, local_files_only=True)
        return True
    except Exception:
        return False


def interrogator_assets_ready(interrogator: "Interrogator") -> bool:
    """Return True when all HF files for this interrogator are already in the local cache."""
    kwargs = _hf_kwargs(interrogator)
    files = _asset_filenames(interrogator)
    if not files:
        return False
    return all(_file_cached(kwargs, filename) for filename in files)


class _TaggerDownloadTqdm(hf_hub_tqdm):
    """Bridge Hugging Face hub tqdm bytes to tagger_progress API."""

    _file_index: int = 1
    _file_total: int = 1
    _filename: str = ""

    def update(self, n=1):
        if tagger_progress.is_cancel_requested():
            raise TaggerCancelled()
        result = super().update(n)
        total = int(getattr(self, "total", 0) or 0)
        current = int(getattr(self, "n", 0) or 0)
        tagger_progress.set_download_bytes(
            file_index=self._file_index,
            file_total=self._file_total,
            filename=self._filename,
            bytes_current=current,
            bytes_total=total,
        )
        return result


@contextmanager
def _hub_download_progress(file_index: int, file_total: int, filename: str) -> Iterator[None]:
    from huggingface_hub.utils import tqdm as tqdm_module

    class _BoundTaggerDownloadTqdm(_TaggerDownloadTqdm):
        pass

    _BoundTaggerDownloadTqdm._file_index = file_index
    _BoundTaggerDownloadTqdm._file_total = file_total
    _BoundTaggerDownloadTqdm._filename = filename

    original = tqdm_module.tqdm
    tqdm_module.tqdm = _BoundTaggerDownloadTqdm
    try:
        yield
    finally:
        tqdm_module.tqdm = original


def download_interrogator_assets(
    model_key: str,
    interrogator: "Interrogator",
    *,
    continue_to_tagging: bool = False,
) -> None:
    """
    Download missing files with WebUI progress updates.

    continue_to_tagging=False: prefetch finished → phase done + release busy.
    continue_to_tagging=True: keep busy, switch to tagging phase for follow-up job.
    """
    kwargs = _hf_kwargs(interrogator)
    files = _asset_filenames(interrogator)
    if not files:
        raise ValueError(f"模型 {model_key} 无可用下载文件列表")

    tagger_progress.begin_download(model_key, len(files), message="正在下载模型…")

    for index, filename in enumerate(files, start=1):
        tagger_progress.check_cancelled()
        tagger_progress.set_download(index, len(files), filename)
        with _hub_download_progress(index, len(files), filename):
            hf_hub_download(**kwargs, filename=filename)
        tagger_progress.set_download_bytes(
            file_index=index,
            file_total=len(files),
            filename=filename,
            bytes_current=0,
            bytes_total=0,
        )

    if continue_to_tagging:
        tagger_progress.complete_download_for_tagging(
            model_key,
            f"模型 {model_key} 已就绪，开始打标…",
        )
    else:
        tagger_progress.finish_download_success(f"模型 {model_key} 已就绪")


def ensure_interrogator_assets(model_key: str, interrogator: "Interrogator") -> bool:
    """
    Ensure model files exist locally before tagging.

    Returns True if a download was performed (caller should expect brief load after).
    """
    if interrogator_assets_ready(interrogator):
        return False
    download_interrogator_assets(model_key, interrogator, continue_to_tagging=True)
    return True


def apply_hf_mirror_from_env() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
