"""FastAPI application: thin adapters over echotwin.services."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..exceptions import (
    ArtifactNotFoundError,
    BackendNotAvailableError,
    ConsentRequiredError,
    EchoTwinError,
    ModelNotAvailableError,
)
from ..services import jobs
from ..settings import get_settings
from ..storage import db
from .routes_data import router as data_router
from .routes_models import router as models_router
from .routes_voice import router as voice_router


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    jobs.startup_cleanup()
    yield


app = FastAPI(
    title="EchoTwin API",
    version=__version__,
    description="Local-first personalization API. Binds to localhost by default.",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:7860"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _status_for(exc: EchoTwinError) -> int:
    if isinstance(exc, ArtifactNotFoundError):
        return 404
    if isinstance(exc, ConsentRequiredError):
        return 403
    if isinstance(exc, (ModelNotAvailableError, BackendNotAvailableError)):
        return 409
    return 400


@app.exception_handler(EchoTwinError)
async def echotwin_error_handler(_request: Request, exc: EchoTwinError):
    return JSONResponse(status_code=_status_for(exc), content={"error": str(exc), "hint": exc.hint})


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "host_binding": settings.api_host,
        "home": str(settings.home),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    from ..voice.registry import all_backend_statuses

    return {
        "voice_backends": all_backend_statuses(),
        "jobs": jobs.list_jobs()[:10],
        "artifacts": db.list_artifacts()[:10],
    }


app.include_router(data_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(voice_router, prefix="/api/v1")


@app.get("/api/v1/jobs")
def list_jobs() -> dict:
    return {"jobs": jobs.list_jobs()}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id} not found")
    return {"job_id": job_id, "cancel_requested": jobs.cancel(job_id)}
