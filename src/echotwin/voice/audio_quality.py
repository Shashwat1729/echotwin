"""Audio quality analysis (no ML required).

Pure signal checks with documented, configurable thresholds. Automated noise
estimates are approximate: the report says so on every run. Feedback messages
are actionable and never moralizing: a quiet recording may still be someone's
natural voice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class QualityThresholds(BaseModel):
    min_seconds: float = 1.0
    max_seconds: float = 120.0
    warn_min_seconds: float = 2.0
    warn_max_seconds: float = 60.0
    peak_too_quiet: float = 0.02  # normalized peak
    peak_too_loud: float = 1.0  # >= this is clipping risk at float scale
    rms_floor: float = 0.003
    clip_fraction_warn: float = 0.001  # 0.1% of samples clipped
    silence_fraction_warn: float = 0.6
    frame_ms: int = 25
    hop_ms: int = 10
    energy_silence_db: float = -45.0
    truncation_tail_db: float = -55.0
    snr_warn_db: float = 15.0


class QualityIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = Field(
        default="info", description="info | warning | error"
    )
    message: str = ""
    hint: str = ""

    def __init__(
        self, code: str, severity: str = "info", message: str = "", hint: str = "", **data
    ):
        # Explicit positional construction: this is a tiny internal record type
        # used heavily in analysis code; keyword construction also works.
        super().__init__(code=code, severity=severity, message=message, hint=hint, **data)


class QualityReport(BaseModel):
    path: str | None = None
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    frames: int = 0
    peak: float = 0.0
    peak_db: float = -120.0
    rms: float = 0.0
    rms_db: float = -120.0
    clipping_fraction: float = 0.0
    silence_fraction: float = 0.0
    speech_fraction: float = 0.0
    snr_estimate_db: float | None = None
    tail_energy_db: float | None = None
    likely_truncated: bool = False
    issues: list[QualityIssue] = Field(default_factory=list)
    verdict: str = Field(description="ok | warning | fail")

    def issue_codes(self) -> list[str]:
        return [i.code for i in self.issues]


def frame_signal(samples: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    # `as_strided` assumes a unit stride. A non-contiguous input (e.g. base[::2])
    # made the last frame read past the end of the buffer, silently corrupting
    # every RMS-derived measurement. Force contiguity first.
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    if len(samples) < frame_len:
        return np.array([float(np.sqrt(np.mean(samples**2)))]) if len(samples) else np.array([])
    n_frames = 1 + (len(samples) - frame_len) // hop_len
    strided = np.lib.stride_tricks.as_strided
    frames = strided(
        samples,
        shape=(n_frames, frame_len),
        strides=(samples.strides[0] * hop_len, samples.strides[0]),
    )
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def analyze_wave_array(
    samples: np.ndarray, sr: int, *, path: str | None = None, thr: QualityThresholds | None = None
) -> QualityReport:
    thr = thr or QualityThresholds()
    channels = samples.shape[1] if samples.ndim > 1 else 1
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    mono = mono.astype(np.float32, copy=False)
    duration = len(mono) / sr if sr else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    peak_db = 20 * np.log10(peak + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)

    clipping_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0

    frame_rms = frame_signal(mono, sr, thr.frame_ms, thr.hop_ms)
    if len(frame_rms):
        frame_db = 20 * np.log10(frame_rms + 1e-12)
        speech_mask = frame_db > thr.energy_silence_db
        silence_fraction = float(1.0 - np.mean(speech_mask))
        speech_fraction = float(np.mean(speech_mask))
        speech_db = frame_db[speech_mask] if speech_mask.any() else np.array([])
        noise_db = frame_db[~speech_mask] if (~speech_mask).any() else np.array([])
        snr = None
        if len(speech_db) and len(noise_db):
            snr = float(np.mean(speech_db) - np.mean(noise_db))
        elif len(speech_db):
            snr = None  # cannot estimate without a noise floor; do not fabricate
        tail = frame_db[-3:].mean() if len(frame_db) >= 3 else None
    else:
        silence_fraction = 1.0
        speech_fraction = 0.0
        snr = None
        tail = None

    likely_truncated = bool(
        tail is not None and tail < thr.truncation_tail_db and speech_fraction < 0.35
    )

    issues: list[QualityIssue] = []
    if duration < thr.min_seconds:
        issues.append(
            QualityIssue(
                "duration_too_short",
                "error",
                "The recording is too short to be useful.",
                f"Record at least {thr.min_seconds:.0f}s.",
            )
        )
    elif duration < thr.warn_min_seconds:
        issues.append(
            QualityIssue(
                "duration_short",
                "warning",
                "The recording is shorter than recommended.",
                f"Aim for {thr.warn_min_seconds:.0f}+ seconds.",
            )
        )
    if duration > thr.max_seconds:
        issues.append(
            QualityIssue(
                "duration_too_long",
                "error",
                "The recording is very long.",
                f"Keep references under {thr.max_seconds:.0f}s.",
            )
        )
    elif duration > thr.warn_max_seconds:
        issues.append(
            QualityIssue(
                code="duration_long",
                severity="warning",
                message="The recording is longer than recommended.",
                hint="Split long scripts into separate segments.",
            )
        )
    if peak < thr.peak_too_quiet:
        issues.append(
            QualityIssue(
                code="too_quiet",
                severity="warning",
                message="The recording is too quiet.",
                hint="Move closer to the microphone or raise input gain.",
            )
        )
    if rms < thr.rms_floor and peak > 0:
        issues.append(
            QualityIssue(
                code="low_rms",
                severity="warning",
                message="The recording level is low.",
                hint="Check the recording level; note your natural speaking volume may be low.",
            )
        )
    if clipping_fraction >= thr.clip_fraction_warn or peak >= thr.peak_too_loud:
        issues.append(
            QualityIssue(
                code="clipping",
                severity="error",
                message="The signal appears clipped.",
                hint="Re-record with lower input gain to avoid distortion.",
            )
        )
    if silence_fraction > thr.silence_fraction_warn:
        issues.append(
            QualityIssue(
                code="long_silence",
                severity="warning",
                message="There is a long silent section.",
                hint="Trim leading/trailing silence or re-record.",
            )
        )
    if snr is not None and snr < thr.snr_warn_db:
        issues.append(
            QualityIssue(
                code="low_snr_estimate",
                severity="warning",
                message="Estimated SNR is low; background noise may be present.",
                hint="Record in a quieter room if available. Automated SNR is only an estimate.",
            )
        )
    if likely_truncated:
        issues.append(
            QualityIssue(
                code="possible_truncation",
                severity="warning",
                message="The final word may have been cut off.",
                hint="Stop the recording only after you finish speaking.",
            )
        )

    severities = [i.severity for i in issues]
    if "error" in severities:
        verdict = "fail"
    elif "warning" in severities:
        verdict = "warning"
    else:
        verdict = "ok"

    return QualityReport(
        path=path,
        duration_seconds=round(duration, 3),
        sample_rate=int(sr),
        channels=int(channels),
        frames=len(mono),
        peak=round(peak, 5),
        peak_db=round(float(peak_db), 2),
        rms=round(rms, 5),
        rms_db=round(float(rms_db), 2),
        clipping_fraction=round(clipping_fraction, 6),
        silence_fraction=round(silence_fraction, 4),
        speech_fraction=round(speech_fraction, 4),
        snr_estimate_db=round(snr, 2) if snr is not None else None,
        tail_energy_db=round(float(tail), 2) if tail is not None else None,
        likely_truncated=likely_truncated,
        issues=issues,
        verdict=verdict,
    )


def analyze_audio_file(path: str | Path, *, thr: QualityThresholds | None = None) -> QualityReport:
    """Read wav/flac/ogg via soundfile (or stdlib wave) and analyze."""
    from ..voice.audio_ingestion import read_audio

    clip = read_audio(path)
    return analyze_wave_array(clip.samples, clip.sample_rate, path=str(path), thr=thr)


def summary_line(report: QualityReport) -> str:
    if report.verdict == "ok":
        return "The recording quality is acceptable for the configured backend."
    if report.verdict == "warning":
        return "The recording is usable but could be improved: " + "; ".join(
            i.message for i in report.issues
        )
    return "The recording has problems: " + "; ".join(i.message for i in report.issues)
