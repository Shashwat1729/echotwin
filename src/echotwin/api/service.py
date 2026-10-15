"""FastAPI service: validated schemas, localhost by default, sandboxed paths."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..settings import get_settings
from ..storage.paths import ensure_within, unique_path, validate_upload


class Job(BaseModel):
    job_id: str
    kind: str
    status: str = "queued"  # queued | running | completed | failed
    progress: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=2)

    def submit(self, kind: str, fn, *args, **kwargs) -> Job:
        job = Job(job_id=f"job-{uuid.uuid4().hex[:12]}", kind=kind)
        self.jobs[job.job_id] = job

        def _run() -> None:
            job.status = "running"
            try:
                result = fn(*args, **kwargs)
                job.result = result
                job.status = "completed"
            except Exception as e:
                job.error = str(e)
                job.status = "failed"
            finally:
                job.finished_at = datetime.now(UTC).isoformat()
                job.progress = 1.0

        self.pool.submit(_run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)


JOBS = JobManager()


def save_upload(filename: str, data: bytes) -> Path:
    """Validate + persist an upload inside ECHOTWIN_HOME/exports."""
    validate_upload(filename, len(data))
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem[:40] or "upload"
    target = unique_path(settings.exports_dir, stem, suffix)
    target.write_bytes(data)
    return target


def safe_download_path(relative: str) -> Path:
    """Resolve a user-provided relative path strictly inside the artifacts dir."""
    settings = get_settings()
    return ensure_within(settings.artifacts_dir, relative)
