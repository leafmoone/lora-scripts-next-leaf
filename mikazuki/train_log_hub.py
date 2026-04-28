"""
Buffers training subprocess stdout per task_id for SSE streaming and optional UI.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Tuple

_MAX_LINES = 15000


class TrainLogHub:
    """Thread-safe line ring buffer per training task."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: Dict[str, Deque[str]] = {}
        self._done: Dict[str, bool] = {}

    def start_task(self, task_id: str) -> None:
        with self._lock:
            self._lines[task_id] = deque(maxlen=_MAX_LINES)
            self._done[task_id] = False

    def append_line(self, task_id: str, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text and line == "":
            return
        with self._lock:
            dq = self._lines.get(task_id)
            if dq is None:
                dq = deque(maxlen=_MAX_LINES)
                self._lines[task_id] = dq
            dq.append(text)

    def mark_done(self, task_id: str) -> None:
        with self._lock:
            self._done[task_id] = True

    def is_done(self, task_id: str) -> bool:
        with self._lock:
            return self._done.get(task_id, False)

    def snapshot_from(self, task_id: str, start_idx: int) -> Tuple[List[str], int, bool]:
        """Return new lines since start_idx, total line count, and whether task finished."""
        with self._lock:
            dq = self._lines.get(task_id)
            done = self._done.get(task_id, False)
            if dq is None:
                return [], 0, done
            lst = list(dq)
        total = len(lst)
        return lst[start_idx:], total, done


hub = TrainLogHub()
