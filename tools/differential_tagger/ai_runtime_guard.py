"""Shared guard for exclusive local AI model runtimes (prevents OOM crashes)."""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)

AI_RUNTIME_LOCK_DISABLED = os.environ.get(
    "STANDALONE_TAGGER_DISABLE_AI_RUNTIME_LOCK", "false",
).lower() in {"1", "true", "yes"}

_process_lock = threading.RLock()
_lease_depth = 0


class AiRuntimeLease:
    """Exclusive lease for heavy model load/inference sections."""

    def __init__(self, label: str) -> None:
        self.label = str(label or "ai-runtime")
        self._handle: Optional[BinaryIO] = None
        self._acquired = False
        self._nested = False

    def acquire(self) -> "AiRuntimeLease":
        if self._acquired:
            return self

        global _lease_depth

        _process_lock.acquire()
        if _lease_depth > 0:
            _lease_depth += 1
            self._nested = True
            self._acquired = True
            return self

        if AI_RUNTIME_LOCK_DISABLED:
            _lease_depth += 1
            self._acquired = True
            return self

        lock_path = Path(tempfile.gettempdir()) / "standalone-tagger-ai-runtime.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            _lock_file(handle)
            handle.seek(0)
            handle.truncate()
            handle.write(
                f"pid={os.getpid()} label={self.label}\n".encode("utf-8", errors="ignore")
            )
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            handle.close()
            _process_lock.release()
            raise

        self._handle = handle
        self._acquired = True
        _lease_depth += 1
        logger.debug("Acquired AI runtime lease: %s", self.label)
        return self

    def release(self) -> None:
        if not self._acquired:
            return

        global _lease_depth

        try:
            _lease_depth = max(0, _lease_depth - 1)
            if self._nested:
                self._nested = False
            elif self._handle is not None:
                try:
                    self._handle.seek(0)
                    self._handle.truncate()
                    self._handle.flush()
                    _unlock_file(self._handle)
                finally:
                    self._handle.close()
                    self._handle = None
        finally:
            self._acquired = False
            _process_lock.release()
            logger.debug("Released AI runtime lease: %s", self.label)

    def __enter__(self) -> "AiRuntimeLease":
        return self.acquire()

    def __exit__(self, *_args) -> bool:
        self.release()
        return False


def exclusive_ai_runtime(label: str) -> AiRuntimeLease:
    """Context manager for exclusive heavy-runtime work."""
    return AiRuntimeLease(label)


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            logger.debug("AI runtime Windows file unlock failed", exc_info=True)
        return

    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.debug("AI runtime POSIX file unlock failed", exc_info=True)


def clear_torch_cuda_cache(torch_module=None) -> None:
    """Best-effort CUDA cache release without importing torch unless needed."""
    try:
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except Exception:
        logger.debug("CUDA cache clear failed", exc_info=True)
