"""Upload and download path helpers for the API (localhost, sandboxed paths)."""

from __future__ import annotations

from pathlib import Path

from ..settings import get_settings
from ..storage.paths import ensure_within, unique_path, validate_upload


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
