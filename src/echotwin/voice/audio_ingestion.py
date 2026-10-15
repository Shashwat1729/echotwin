"""Audio ingestion: read/validate recordings via soundfile (stdlib wave fallback).

Validation contract:

* WAV is always readable (stdlib); FLAC/OGG/etc. require the ``[voice]`` extra.
* Corrupt/truncated headers raise ``AudioFormatError`` with actionable messages.
* Sample rates outside backend requirements are reported (warnings), never rejected
  silently — preprocessing handles conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exceptions import UploadValidationError

SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
LIBSUPPORTED = {".wav", ".flac", ".ogg"}  # via soundfile


@dataclass
class AudioClip:
    samples: np.ndarray  # float32 mono or (n, channels)
    sample_rate: int
    path: str | None = None
    channels: int = 1
    subtype: str = "PCM_16"


class AudioFormatError(UploadValidationError):
    pass


def read_audio(path: str | Path, *, mono: bool = False) -> AudioClip:
    p = Path(path)
    if not p.exists():
        raise AudioFormatError(f"Audio file does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix and suffix not in SUPPORTED_SUFFIXES:
        raise AudioFormatError(
            f"Unsupported audio format '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix and suffix not in LIBSUPPORTED:
        # mp3/m4a/webm: need soundfile>=0.12 with libsndfile support; be explicit
        try:
            import soundfile as sf  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise AudioFormatError(
                f"Reading '{suffix}' requires the [voice] extra (soundfile): uv pip install "
                f"'-e .[voice]'. WAV files always work."
            ) from e

    if suffix == ".wav" or suffix == "":
        clip = _read_wave_stdlib(p)
        if clip is None:
            clip = _read_soundfile(p)
    else:
        clip = _read_soundfile(p)

    if clip is None:
        raise AudioFormatError(f"Could not read audio file: {p}")

    if mono and clip.samples.ndim > 1:
        clip.samples = clip.samples.mean(axis=1)
        clip.channels = 1
    return clip


def _read_wave_stdlib(p: Path) -> AudioClip | None:
    import wave

    try:
        with wave.open(str(p), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
    except (wave.Error, EOFError) as e:
        raise AudioFormatError(
            f"WAV file is corrupt or unreadable: {p} ({e}). Re-export as 16-bit PCM WAV."
        ) from e

    if sampwidth not in (1, 2, 3, 4):
        raise AudioFormatError(f"Unsupported WAV sample width: {sampwidth * 8} bits.")
    samples: np.ndarray
    if sampwidth == 3:  # 24-bit packed ints
        packed = np.frombuffer(raw, dtype=np.uint8)
        if len(packed) % 3:
            raise AudioFormatError("Truncated 24-bit PCM data (file may be truncated).")
        a = np.zeros((len(packed) // 3, 3), dtype=np.uint8)
        a[:, 0] = packed[0::3]
        a[:, 1] = packed[1::3]
        a[:, 2] = packed[2::3]
        vals = (
            (a[:, 0].astype(np.int32) << 8)
            | (a[:, 1].astype(np.int32) << 16)
            | (a[:, 2].astype(np.int32) << 24)
        )
        samples = (vals >> 8).astype(np.float32) / 2**23
    else:
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
        data = np.frombuffer(raw, dtype=dtype)
        if sampwidth == 1:
            samples = (data.astype(np.float32) - 128.0) / 128.0
        else:
            scale = float(2 ** (sampwidth * 8 - 1))
            samples = data.astype(np.float32) / scale
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return AudioClip(
        samples=samples,
        sample_rate=sr,
        path=str(p),
        channels=n_channels,
        subtype=f"PCM_{sampwidth * 8}",
    )


def _read_soundfile(p: Path) -> AudioClip | None:
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise AudioFormatError(
            f"Reading {p.name} requires the [voice] extra: uv pip install '-e .[voice]'"
        ) from e
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as e:
        raise AudioFormatError(f"Could not decode audio file {p}: {e}") from e
    channels = 1 if data.ndim == 1 else data.shape[1]
    return AudioClip(
        samples=data, sample_rate=int(sr), path=str(p), channels=channels, subtype="float32"
    )


def validate_recording(path: str | Path, *, max_seconds: float | None = None) -> dict:
    """Lightweight validation for uploads: format + duration bounds."""
    clip = read_audio(path)
    duration = len(clip.samples) / clip.sample_rate if clip.sample_rate else 0.0
    ext_ok = Path(path).suffix.lower() in SUPPORTED_SUFFIXES
    return {
        "path": str(path),
        "sample_rate": clip.sample_rate,
        "channels": clip.channels,
        "duration_seconds": round(duration, 3),
        "format_ok": ext_ok,
        "too_long": bool(max_seconds and duration > max_seconds),
    }
