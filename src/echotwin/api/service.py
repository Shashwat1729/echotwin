"""Upload and download path helpers for the API (localhost, sandboxed paths)."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import UploadValidationError
from ..settings import get_settings
from ..storage.paths import ensure_within, unique_path, validate_upload


async def save_upload_stream(file, filename: str | None = None) -> Path:
    """Persist an uploaded file without buffering it all in memory.

    The size limit is enforced *while streaming* (and up-front when the client
    announced a length), so a huge upload is rejected instead of being read
    into RAM first.
    """
    settings = get_settings()
    name = filename or getattr(file, "filename", None) or "upload"
    limit = settings.max_upload_bytes

    declared = getattr(file, "size", None)
    if isinstance(declared, int) and declared > limit:
        raise UploadValidationError(
            f"Upload is {declared} bytes, over the {limit} byte limit. "
            "Raise ECHOTWIN_MAX_UPLOAD_MB if you intend a larger file."
        )

    suffix = Path(name).suffix.lower()
    stem = Path(name).stem[:40] or "upload"
    validate_upload(name, 0)  # extension check before we write anything
    target = unique_path(settings.exports_dir, stem, suffix)
    written = 0
    with target.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                fh.close()
                target.unlink(missing_ok=True)
                raise UploadValidationError(
                    f"Upload exceeds the {limit} byte limit; nothing was stored."
                )
            fh.write(chunk)
    if written == 0:
        target.unlink(missing_ok=True)
        raise UploadValidationError("Upload was empty.")
    return target


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
