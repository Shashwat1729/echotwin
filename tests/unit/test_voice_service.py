"""Voice use-cases: recording intake, consent gate, synthesis provenance."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from echotwin.exceptions import ArtifactNotFoundError, ConsentRequiredError, EchoTwinError
from echotwin.services import voice
from echotwin.voice.metadata import load_profile, save_profile
from echotwin.workspace import get_store, reset_store


@pytest.fixture(autouse=True)
def _clean():
    yield
    reset_store()


def _sine_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> Path:
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(n)
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


class FakeBackend:
    name = "fake"
    model_id = "fake-1"

    def capabilities(self) -> dict:
        return {"version": "t"}

    def synthesize(self, text, reference_audio, transcript=None, output_path=None, seed=None, **kw):
        Path(output_path).write_bytes(b"RIFFfake")
        return Path(output_path)


@pytest.fixture()
def recorded(tmp_path):
    res = voice.save_recording(_sine_wav(tmp_path / "clip.wav"), "en", "")
    prof = load_profile(res.profile_id)
    prof.reference_selection = {"selected": [{"recording_id": res.recording_id}]}
    save_profile(prof)
    return res


def test_no_profile_yet():
    assert voice.current_profile() is None
    with pytest.raises(ArtifactNotFoundError):
        voice.analyze()


def test_save_recording_sets_pointer_and_verdict(tmp_path):
    res = voice.save_recording(_sine_wav(tmp_path / "clip.wav"), "en", "")
    assert res.verdict and res.summary
    assert get_store().pointer("current_voice_profile") == res.profile_id
    prof = voice.current_profile()
    assert [r.recording_id for r in prof.recordings] == [res.recording_id]
    assert Path(prof.recordings[0].path).exists()


def test_save_recording_rejects_unreadable_file(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(EchoTwinError):
        voice.save_recording(bad, "en", "")
    assert voice.current_profile() is None


def test_analyze_and_transcript_and_coverage(recorded):
    rows = voice.analyze()
    assert rows and rows[0]["recording_id"] == recorded.recording_id
    check = voice.set_transcript(recorded.recording_id, "hello there how are you")
    assert check.verdict
    assert voice.current_profile().recordings[0].has_transcript
    assert voice.coverage() is not None
    with pytest.raises(EchoTwinError):
        voice.set_transcript("missing-id", "x")


def test_synthesize_requires_consent(recorded):
    assert voice.consent_status().granted is False
    with pytest.raises(ConsentRequiredError):
        voice.synthesize("hello", None)


def test_synthesize_with_consent_writes_wav_and_provenance(recorded, monkeypatch):
    monkeypatch.setattr("echotwin.voice.synthesis.get_backend", lambda name, **kw: FakeBackend())
    voice.grant_consent("tester")
    assert voice.consent_status().granted is True
    result = voice.synthesize("hello", 7)
    out = Path(result.output_path)
    assert out.exists()
    assert Path(str(out) + ".provenance.json").exists()
    assert result.seed == 7


def test_revoke_blocks_again(recorded, monkeypatch):
    monkeypatch.setattr("echotwin.voice.synthesis.get_backend", lambda name, **kw: FakeBackend())
    voice.grant_consent("tester")
    voice.revoke_consent()
    assert voice.consent_status().granted is False
    with pytest.raises(ConsentRequiredError):
        voice.synthesize("hello", None)


def test_plan_lists_tasks():
    plan = voice.plan("voxcpm", ["en"], 5)
    assert plan.backend and plan.tasks
