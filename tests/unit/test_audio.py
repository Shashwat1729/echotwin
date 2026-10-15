"""Unit tests: audio ingestion + quality analysis on synthetic signals."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from echotwin.voice.audio_ingestion import AudioFormatError, read_audio, validate_recording
from echotwin.voice.audio_quality import (
    QualityThresholds,
    analyze_audio_file,
    analyze_wave_array,
    summary_line,
)


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def speech_like(tmp_path) -> Path:
    sr = 16000
    t = np.arange(sr * 5) / sr
    env = np.clip(np.sin(2 * np.pi * 0.5 * t), 0, 1)  # amplitude envelope
    tone = 0.3 * np.sin(2 * np.pi * 180 * t) * env
    return write_wav(tmp_path / "speech.wav", tone, sr)


@pytest.fixture
def quiet_clip(tmp_path) -> Path:
    sr = 16000
    t = np.arange(sr * 3) / sr
    tone = 0.001 * np.sin(2 * np.pi * 180 * t)
    return write_wav(tmp_path / "quiet.wav", tone, sr)


@pytest.fixture
def clipped_clip(tmp_path) -> Path:
    sr = 16000
    t = np.arange(sr * 3) / sr
    tone = 1.2 * np.sin(2 * np.pi * 180 * t)
    return write_wav(tmp_path / "clipped.wav", np.clip(tone, -1, 1), sr)


@pytest.fixture
def silent_tail(tmp_path) -> Path:
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = 0.3 * np.sin(2 * np.pi * 180 * t)
    tail = np.zeros(sr * 4)
    return write_wav(tmp_path / "tail.wav", np.concatenate([tone, tail]), sr)


class TestIngestion:
    def test_read_wav(self, speech_like):
        clip = read_audio(speech_like)
        assert clip.sample_rate == 16000
        assert clip.channels == 1
        assert len(clip.samples) > 16000 * 4

    def test_read_mono_conversion(self, tmp_path):
        sr = 16000
        stereo = np.stack(
            [0.3 * np.sin(np.linspace(0, 100, sr)), 0.2 * np.sin(np.linspace(0, 80, sr))], axis=1
        )
        pcm = (stereo * 32767).astype("<i2")
        with wave.open(str(tmp_path / "s.wav"), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.reshape(-1).tobytes())
        clip = read_audio(tmp_path / "s.wav", mono=True)
        assert clip.channels == 1

    def test_corrupt_wav_raises(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"RIFFnot really a wave file")
        with pytest.raises(AudioFormatError):
            read_audio(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AudioFormatError):
            read_audio(tmp_path / "nope.wav")

    def test_validate_recording(self, speech_like):
        info = validate_recording(speech_like, max_seconds=120)
        assert info["format_ok"]
        assert not info["too_long"]
        assert 4.9 < info["duration_seconds"] < 5.1

    def test_validate_rejects_unsupported(self, tmp_path):
        f = tmp_path / "x.xyz"
        f.write_bytes(b"data")
        with pytest.raises(AudioFormatError):
            read_audio(f)


class TestQuality:
    def test_good_recording_ok(self, speech_like):
        report = analyze_audio_file(speech_like)
        assert report.verdict in {"ok", "warning"}
        assert report.duration_seconds > 4
        assert report.peak > 0.1

    def test_too_quiet_detected(self, quiet_clip):
        report = analyze_audio_file(quiet_clip)
        assert "too_quiet" in report.issue_codes()

    def test_clipping_detected(self, clipped_clip):
        report = analyze_audio_file(clipped_clip)
        assert "clipping" in report.issue_codes()
        assert report.verdict == "fail"

    def test_long_silence_detected(self, silent_tail):
        report = analyze_audio_file(silent_tail)
        assert "long_silence" in report.issue_codes()

    def test_summary_line_positive_when_ok(self, speech_like):
        report = analyze_audio_file(speech_like)
        line = summary_line(report)
        assert isinstance(line, str) and line

    def test_thresholds_configurable(self, quiet_clip):
        strict = QualityThresholds(peak_too_quiet=0.1)
        report = analyze_audio_file(quiet_clip, thr=strict)
        assert report.verdict != "ok"

    def test_array_analysis(self):
        sr = 16000
        samples = (0.3 * np.sin(np.linspace(0, 500, sr))).astype(np.float32)
        report = analyze_wave_array(samples, sr)
        assert report.duration_seconds == pytest.approx(1.0, abs=0.01)
        assert report.peak > 0.2

    def test_snr_not_fabricated(self):
        # pure tone with no silence floor -> snr is None, not invented
        sr = 16000
        t = np.arange(sr * 2) / sr
        samples = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        report = analyze_wave_array(samples, sr)
        assert report.snr_estimate_db is None
