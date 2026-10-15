"""Sandboxed filesystem helpers.

All API/UI file operations resolve paths inside configured roots. Anything
escaping its root raises PathTraversalError instead of touching the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..exceptions import PathTraversalError, UploadValidationError
from ..settings import get_settings


def ensure_within(root: Path | str, candidate: Path | str) -> Path:
    root_p = Path(root).resolve()
    cand_p = Path(candidate)
    if not cand_p.is_absolute():
        cand_p = root_p / cand_p
    cand_res = cand_p.resolve()
    if not (cand_res == root_p or root_p in cand_res.parents):
        raise PathTraversalError(f"Path escapes its sandbox root: {cand_p}")
    return cand_res


def validate_upload(filename: str, size: int) -> str:
    """Validate an uploaded file's extension and size against settings."""
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_upload_extensions:
        raise UploadValidationError(
            f"Unsupported upload type '{ext or filename}'. Allowed: {settings.allowed_upload_extensions}"
        )
    if size > settings.max_upload_bytes:
        raise UploadValidationError(
            f"Upload exceeds limit ({size} > {settings.max_upload_bytes} bytes). "
            "Raise ECHOTWIN_MAX_UPLOAD_MB if intended."
        )
    return ext


def safe_bytesize(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        candidate = directory / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def home_summary() -> dict:
    settings = get_settings()
    try:
        writable = os.access(settings.home, os.W_OK)
    except OSError:
        writable = False
    return {
        "home": str(settings.home),
        "exists": settings.home.exists(),
        "writable": writable,
        "db": str(settings.db_path),
    }
