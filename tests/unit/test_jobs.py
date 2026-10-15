"""Persisted job runner: success, failure, cancel, restart cleanup, heavy exclusion."""

from __future__ import annotations

import threading
import time

import pytest

from echotwin.services import jobs
from echotwin.workspace import get_store, reset_store


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_store()


def test_success_stores_result_and_progress():
    def fn(ctx):
        ctx.progress(0.5, "half")
        return {"n": 3}

    job = jobs.submit("demo", fn)
    assert job["status"] in {"queued", "running", "completed"}
    done = jobs.wait(job["job_id"])
    assert done["status"] == "completed"
    assert done["progress"] == 1.0
    assert done["result"] == {"n": 3}
    assert done["finished_at"] and done["error"] is None
    assert jobs.get(job["job_id"])["status"] == "completed"
    assert job["job_id"] in {j["job_id"] for j in jobs.list_jobs()}


def test_raising_fn_fails_with_short_error_and_no_traceback():
    def fn(ctx):
        raise ValueError("x" * 500)

    done = jobs.wait(jobs.submit("demo", fn)["job_id"])
    assert done["status"] == "failed"
    assert done["error"].startswith("ValueError: xxx")
    assert len(done["error"]) <= 200
    assert "Traceback" not in done["error"]


def test_cancel_before_start():
    gate = threading.Event()
    ran: list[int] = []
    blockers = [jobs.submit("block", lambda ctx: gate.wait(10) and None) for _ in range(2)]
    time.sleep(0.1)  # let both pool workers pick up the blockers
    queued = jobs.submit("late", lambda ctx: ran.append(1))
    assert jobs.cancel(queued["job_id"]) is True
    gate.set()
    for b in blockers:
        jobs.wait(b["job_id"])
    assert jobs.wait(queued["job_id"])["status"] == "cancelled"
    assert ran == []


def test_cooperative_cancel_mid_run():
    started = threading.Event()

    def fn(ctx):
        started.set()
        while not ctx.cancelled():
            time.sleep(0.01)

    job = jobs.submit("loop", fn)
    assert started.wait(5)
    assert jobs.cancel(job["job_id"]) is True
    assert jobs.wait(job["job_id"])["status"] == "cancelled"


def test_cancel_finished_job_is_false():
    job = jobs.submit("demo", lambda ctx: None)
    jobs.wait(job["job_id"])
    assert jobs.cancel(job["job_id"]) is False
    assert jobs.cancel("nope") is False


def test_startup_cleanup_marks_stale_running_failed():
    store = get_store()
    store.job_insert(
        {"job_id": "stale", "kind": "x", "status": "running", "created_at": "2026-01-01T00:00:00Z"}
    )
    jobs.startup_cleanup()
    job = jobs.get("stale")
    assert job["status"] == "failed" and job["error"] == "interrupted"


def test_heavy_jobs_never_overlap():
    active = 0
    peak = 0
    lock = threading.Lock()

    def fn(ctx):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with lock:
            active -= 1

    ids = [jobs.submit("heavy", fn, heavy=True)["job_id"] for _ in range(2)]
    for i in ids:
        assert jobs.wait(i)["status"] == "completed"
    assert peak == 1
