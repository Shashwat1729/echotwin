"""Persisted background job runner shared by the API, CLI and UI.

Job state lives in the workspace Store, so it survives restarts and every adapter
sees the same list. Functions run on a small thread pool; "heavy" jobs (model
loads, training, synthesis) are serialised by a semaphore. Cancellation is
cooperative: the job function polls ``ctx.cancelled()``.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from ..workspace import get_store

TERMINAL = {"completed", "failed", "cancelled"}
_MAX_ERROR = 200

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="echotwin-job")
_heavy = threading.Semaphore(1)
_cancel: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobContext:
    def __init__(self, job_id: str, event: threading.Event) -> None:
        self.job_id = job_id
        self._event = event

    def progress(self, frac: float, msg: str = "") -> None:
        get_store().job_update(self.job_id, progress=max(0.0, min(1.0, float(frac))), message=msg)

    def cancelled(self) -> bool:
        return self._event.is_set()


def submit(kind: str, fn: Callable[[JobContext], dict | None], *, heavy: bool = False) -> dict:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "message": "",
        "created_at": _now(),
    }
    event = threading.Event()
    store = get_store()
    store.job_insert(job)
    with _lock:
        _cancel[job["job_id"]] = event
    _executor.submit(_run, job["job_id"], fn, heavy, event)
    return store.job_get(job["job_id"]) or job


def _run(
    job_id: str, fn: Callable[[JobContext], dict | None], heavy: bool, event: threading.Event
) -> None:
    store = get_store()
    try:
        if heavy:
            _heavy.acquire()
        try:
            if event.is_set():
                _finish(job_id, "cancelled")
                return
            store.job_update(job_id, status="running", started_at=_now())
            try:
                result = fn(JobContext(job_id, event))
            except Exception as exc:
                _finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR])
                return
            if event.is_set():
                _finish(job_id, "cancelled")
            else:
                _finish(job_id, "completed", progress=1.0, result=result)
        finally:
            if heavy:
                _heavy.release()
    finally:
        with _lock:
            _cancel.pop(job_id, None)


def _finish(job_id: str, status: str, **fields) -> None:
    get_store().job_update(job_id, status=status, finished_at=_now(), **fields)


def get(job_id: str) -> dict | None:
    return get_store().job_get(job_id)


def list_jobs() -> list[dict]:
    return get_store().job_list()


def cancel(job_id: str) -> bool:
    """Request cancellation. True if the job was still queued or running."""
    job = get_store().job_get(job_id)
    if job is None or job["status"] in TERMINAL:
        return False
    with _lock:
        event = _cancel.get(job_id)
    if event is None:
        return False
    event.set()
    if job["status"] == "queued":
        _finish(job_id, "cancelled")
    return True


def wait(job_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        job = get_store().job_get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] in TERMINAL or time.monotonic() >= deadline:
            return job
        time.sleep(0.02)


def startup_cleanup() -> int:
    """Mark jobs left 'running' by a crashed process as failed (interrupted)."""
    return get_store().fail_interrupted()
