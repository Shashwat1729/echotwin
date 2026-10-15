"""Global settings for EchoTwin.

All runtime state lives under a single local ``ECHOTWIN_HOME`` directory
(default ``~/.echotwin``; override with the environment variable). Nothing here
touches the network; that only happens when a user explicitly configures it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HOME = Path(os.environ.get("ECHOTWIN_HOME", Path.home() / ".echotwin"))


class Settings(BaseSettings):
    """Runtime settings. Overridable via environment variables prefixed ``ECHOTWIN_``."""

    model_config = SettingsConfigDict(
        env_prefix="ECHOTWIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    home: Annotated[Path, Field(description="Root directory for all local state")] = DEFAULT_HOME

    # --- API/UI ---
    api_host: Annotated[str, Field(description="Bind address; localhost by default")] = "127.0.0.1"
    api_port: Annotated[int, Field(ge=1, le=65535)] = 8765
    ui_host: Annotated[str, Field(description="Gradio bind address")] = "127.0.0.1"
    ui_port: Annotated[int, Field(ge=1, le=65535)] = 7860

    # --- upload limits / safety ---
    max_upload_mb: Annotated[int, Field(ge=1, le=4096)] = 50
    allowed_upload_extensions: Annotated[
        list[str], Field(description="Lower-case extensions accepted for uploads")
    ] = [".txt", ".zip", ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"]

    # --- ingestion / dataset defaults ---
    language_detector: Annotated[
        str, Field(description="heuristic | langdetect (langdetect requires the [lang] extra)")
    ] = "heuristic"
    dayfirst: Annotated[bool, Field(description="Interpret ambiguous dates DD/MM before MM/DD")] = (
        True
    )
    redact_pii: Annotated[
        bool, Field(description="Redact phone numbers/emails in processed views")
    ] = False

    # --- voice defaults ---
    default_voice_backend: Annotated[
        str, Field(description="Registry key of the default backend")
    ] = "voxcpm"
    target_sample_rate: Annotated[
        int, Field(description="Canonical sample rate for derived audio")
    ] = 16000

    # --- privacy / retention ---
    retention_days: Annotated[int, Field(ge=0, description="0 = keep until the user deletes")] = 0
    log_raw_messages: Annotated[bool, Field(description="NEVER enable: logs raw message text")] = (
        False
    )

    @field_validator("allowed_upload_extensions")
    @classmethod
    def _norm_exts(cls, v: list[str]) -> list[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in v]

    # --- derived directories ---
    @property
    def db_path(self) -> Path:
        return self.home / "echotwin.sqlite3"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def recordings_dir(self) -> Path:
        return self.home / "recordings"

    @property
    def exports_dir(self) -> Path:
        return self.home / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (
            self.home,
            self.artifacts_dir,
            self.recordings_dir,
            self.exports_dir,
            self.cache_dir,
            self.artifacts_dir / "datasets",
            self.artifacts_dir / "profiles",
            self.artifacts_dir / "experiments",
            self.artifacts_dir / "evaluations",
            self.artifacts_dir / "voice_profiles",
            self.artifacts_dir / "audio",
            self.artifacts_dir / "models",
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton (respects ECHOTWIN_HOME for tests/demo)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Test helper: drop the cached singleton so ECHOTWIN_HOME changes take effect."""
    global _settings
    _settings = None
