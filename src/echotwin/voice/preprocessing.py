"""Audio preprocessing: canonical format conversion without altering content.

Rules:

* Resampling is only applied when the backend's supported rate differs from the
  source, and the derived file is stored separately (source preserved).
* Optional peak normalization and silence trimming (trimming is OFF by default
  so natural room tone/pauses are kept).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exceptions import UnsupportedCapabilityError
from ..voice.audio_ingestion import AudioClip, read_audio


@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    peak_normalize: bool = False
    trim_silence: bool = False
    trim_db: float = -45.0
    mono: bool = True


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    try:
        from scipy.signal import resample_poly

        g = np.gcd(int(orig_sr), int(target_sr))
        return resample_poly(samples, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError as e:  # pragma: no cover
        raise UnsupportedCapabilityError(
            "Resampling requires scipy (part of the [voice] extra)."
        ) from e


def to_canonical(clip: AudioClip, cfg: PreprocessConfig, out_path: str | Path) -> Path:
    samples = clip.samples
    if cfg.mono and samples.ndim > 1:
        samples = samples.mean(axis=1)
    if clip.sample_rate != cfg.target_sr:
        samples = resample(samples, clip.sample_rate, cfg.target_sr)
    if cfg.trim_silence:
        samples = trim_silence(samples, cfg.target_sr, cfg.trim_db)
    if cfg.peak_normalize:
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 1e-6:
            samples = samples / peak * 0.95
    return write_wav(out_path, samples, cfg.target_sr)


def trim_silence(samples: np.ndarray, sr: int, trim_db: float) -> np.ndarray:
    if not len(samples):
        return samples
    frame = max(1, int(sr * 0.02))
    n_frames = len(samples) // frame
    if n_frames < 2:
        return samples
    trim_lin = 10 ** (trim_db / 20)
    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = np.where(rms > trim_lin)[0]
    if len(active) == 0:
        return samples
    start = max(0, int(active[0]) * frame - int(sr * 0.05))
    end = min(len(samples), (int(active[-1]) + 1) * frame + int(sr * 0.05))
    return samples[start:end]


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(p), samples.astype(np.float32), sr, subtype="PCM_16")
        return p
    except ImportError:
        import wave

        data = np.clip(samples, -1.0, 1.0)
        pcm = (data * 32767.0).astype("<i2")
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return p


def preprocess_file(path: str | Path, cfg: PreprocessConfig, out_path: str | Path) -> Path:
    clip = read_audio(path, mono=cfg.mono)
    return to_canonical(clip, cfg, out_path)
