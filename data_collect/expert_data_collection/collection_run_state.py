"""Crash-safe run metadata and single-writer locking for expert data collection."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path


def atomic_write_json(path: str | Path, data: dict) -> None:
    """Atomically replace a JSON file after flushing its contents to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)

        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class ExclusiveRunLock:
    """Hold a non-blocking process lock for one collection run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / ".quest_policy_collect.lock"
        self._file = None

    def acquire(self) -> "ExclusiveRunLock":
        if self._file is not None:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "unknown owner"
            lock_file.close()
            raise RuntimeError(
                f"Collection run is already active: {self.path.parent} ({owner})"
            ) from exc

        lock_file.seek(0)
        lock_file.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            lock_file,
            ensure_ascii=False,
        )
        lock_file.flush()
        os.fsync(lock_file.fileno())
        self._file = lock_file
        return self

    def close(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ExclusiveRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
