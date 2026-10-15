"""EchoTwin exception hierarchy.

Every error that user actions can trigger should derive from EchoTwinError so the
CLI/API can render actionable messages instead of tracebacks.
"""

from __future__ import annotations


class EchoTwinError(Exception):
    """Base class for all EchoTwin errors."""

    exit_code = 1
    hint: str | None = None


class ConfigurationError(EchoTwinError):
    """Invalid or missing configuration."""

    exit_code = 2


class ExportValidationError(EchoTwinError):
    """A WhatsApp export could not be validated (empty, unreadable, wrong format)."""

    exit_code = 3
    hint = (
        "Expected a WhatsApp chat export: a '_chat.txt' file, or a .zip containing one, "
        "exported via chat > Export chat."
    )


class ParseError(EchoTwinError):
    """A specific line/segment could not be parsed (non-fatal per-record errors are
    collected on the ParseResult instead of being raised)."""

    exit_code = 4


class ConsentRequiredError(EchoTwinError):
    """A consent/authorization gate blocked the requested operation."""

    exit_code = 5
    hint = "Re-run with explicit consent; see docs/privacy-and-security.md."


class DatasetError(EchoTwinError):
    """Dataset construction, validation, or split failure."""

    exit_code = 6


class InsufficientDataError(DatasetError):
    """Not enough usable data for the requested operation."""

    exit_code = 7


class ModelNotAvailableError(EchoTwinError):
    """A model or optional dependency is not installed/downloaded."""

    exit_code = 8
    hint = "Install the relevant extra (e.g. uv pip install -e '.[train]') or run scripts/download_models.py."


class UnsupportedCapabilityError(EchoTwinError):
    """A backend does not support the requested capability; we never silently substitute."""

    exit_code = 9


class BackendNotAvailableError(EchoTwinError):
    """A voice backend package or model weights are unavailable."""

    exit_code = 10


class PrivacyError(EchoTwinError):
    """A privacy control blocked the operation (e.g. raw text in logs)."""

    exit_code = 11


class UploadValidationError(EchoTwinError):
    """An uploaded file failed validation (size, type, content)."""

    exit_code = 12


class PathTraversalError(PrivacyError):
    """A path escaped its allowed sandbox."""

    exit_code = 13


class TrainingError(EchoTwinError):
    """Training pipeline failure."""

    exit_code = 14


class EvaluationError(EchoTwinError):
    """Evaluation pipeline failure."""

    exit_code = 15


class ArtifactNotFoundError(EchoTwinError):
    """A requested artifact does not exist."""

    exit_code = 16
