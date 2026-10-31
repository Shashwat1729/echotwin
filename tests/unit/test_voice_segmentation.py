"""Long-take segmentation, script alignment and the clipping rule."""

from __future__ import annotations

import numpy as np

from echotwin.voice.audio_quality import analyze_wave_array
from echotwin.voice.segmentation import (
    assign_transcripts,
    normalize_peak,
    split_on_pauses,
    spoken_sentences,
)

SR = 16000


def _speech(seconds: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (
        0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)


def _take(chunks: list[float], pause: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(0)
    gap = (0.0005 * rng.standard_normal(int(pause * SR))).astype(np.float32)
    parts: list[np.ndarray] = []
    for c in chunks:
        parts += [_speech(c, rng), gap]
    return np.concatenate(parts)


def test_split_cuts_in_pauses_not_in_speech():
    x = _take([8, 9, 7, 10, 8, 9])  # ~51 s of speech with 0.5 s pauses
    segs = split_on_pauses(x, SR)
    assert 2 <= len(segs) <= 6
    assert all(6 <= s.seconds(SR) <= 27 for s in segs)
    # a cut inside speech would leave loud samples right at a segment edge
    for s in segs[:-1]:
        assert np.abs(x[s.end - 200 : s.end]).max() < 0.1


def test_short_audio_is_one_segment():
    segs = split_on_pauses(_take([5]), SR)
    assert len(segs) == 1


def test_spoken_sentences_drop_mood_cues():
    text = "[excited]\nAre you coming? Great!\n\n[calm, warm]\nसब ठीक है। Okay."
    assert spoken_sentences(text) == ["Are you coming?", "Great!", "सब ठीक है।", "Okay."]


def test_transcripts_follow_speech_proportion():
    x = _take([10, 10, 10, 10])
    segs = split_on_pauses(x, SR, max_seconds=22.0)
    script = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    texts = assign_transcripts(segs, script)
    assert len(texts) == len(segs)
    assert " ".join(texts).split() == script.split()  # nothing lost, nothing duplicated, order kept


def test_normalize_only_attenuates():
    quiet = np.full(10, 0.3, dtype=np.float32)
    assert np.allclose(normalize_peak(quiet), quiet)
    hot = np.array([1.0056, -0.5], dtype=np.float32)
    assert abs(float(np.abs(normalize_peak(hot)).max()) - 0.9) < 1e-6


def test_codec_overshoot_is_not_clipping():
    rng = np.random.default_rng(1)
    x = _speech(5.0, rng) * 0.9
    x[1000] = 1.0056  # one decoded-Opus-style overshoot sample
    report = analyze_wave_array(x, SR)
    codes = {i.code for i in report.issues}
    assert "clipping" not in codes


def test_real_clipping_is_still_flagged():
    rng = np.random.default_rng(2)
    x = np.clip(_speech(5.0, rng) * 6.0, -1.0, 1.0)
    report = analyze_wave_array(x, SR)
    assert "clipping" in {i.code for i in report.issues}
