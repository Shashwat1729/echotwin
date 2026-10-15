"""FastAPI application and routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..exceptions import EchoTwinError
from ..settings import get_settings
from ..storage import db
from . import service
from .routes_data import router as data_router
from .routes_models import router as models_router
from .routes_voice import router as voice_router

app = FastAPI(
    title="EchoTwin API",
    version="0.1.0",
    description="Local-first personalization API. Binds to localhost by default.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:7860"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EchoTwinError)
async def echotwin_error_handler(_request: Request, exc: EchoTwinError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "hint": exc.hint},
    )


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "host_binding": settings.api_host,
        "home": str(settings.home),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    from ..voice.registry import all_backend_statuses
    from .service import JOBS

    return {
        "voice_backends": all_backend_statuses(),
        "jobs": [j.model_dump() for j in JOBS.list()[:10]],
        "artifacts": db.list_artifacts()[:10],
    }


app.include_router(data_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(voice_router, prefix="/api/v1")


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = service.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job.model_dump()
