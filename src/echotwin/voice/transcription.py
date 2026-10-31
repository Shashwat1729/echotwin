"""Speech-to-text for uploaded voice and live conversation (optional dependency).

Uses ``faster-whisper`` when installed. Nothing is sent to a network service; the
model weights are downloaded once by faster-whisper itself on first use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..exceptions import BackendNotAvailableError

DEFAULT_MODEL = "small"  # multilingual; handles Hindi / English / Hinglish reasonably


@lru_cache(maxsize=2)
def _model(size: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Speech-to-text needs faster-whisper: pip install faster-whisper"
        ) from exc
    return WhisperModel(size, compute_type="int8")


def transcribe(
    path: str | Path, language: str | None = None, model_size: str = DEFAULT_MODEL
) -> str:
    """Return the transcript of an audio file ("" for silence)."""
    segments, _info = _model(model_size).transcribe(
        str(path), language=language, vad_filter=True, beam_size=5
    )
    return " ".join(s.text.strip() for s in segments).strip()
