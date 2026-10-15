"""The studio builds, and its handlers drive the pipeline end to end on a temp home."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from echotwin.services import chat as chat_service
from echotwin.services import jobs
from echotwin.ui.app import build_app
from echotwin.ui.pages import chat, dataset, evaluate, home, import_, profile, settings, voice
from echotwin.workspace import reset_store


@pytest.fixture(autouse=True)
def _clean():
    reset_store()
    yield
    reset_store()


def test_build_app_constructs():
    assert build_app() is not None


def test_empty_workspace_chat_shows_callout_not_fake_text():
    history, session = [], chat_service.ChatSession()
    _box, history, session, _why, status = chat.send("hello", history, session, "ngram", "plain")
    assert history == [] and session.history == []
    assert "et-callout" in status


def test_pipeline_through_handlers():
    status, *_ = home.load_sample()
    assert "et-callout" in status
    result = import_.do_select_author("Aarav")
    assert "Aarav" in result[0]
    cards, _warnings, _manifest, rail = dataset.do_prepare(4, 0.2)
    assert "et-card" in cards and "et-rail" in rail
    cards, fragment, _profile_json, rail = profile.do_build()
    assert fragment and "et-card" in cards

    history, session = [], chat_service.ChatSession()
    _box, history, session, why, status = chat.send(
        "kal milte hain?", history, session, "ngram", "plain"
    )
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"]
    assert "ngram" in why


def test_evaluate_job_populates_tables():
    home.load_sample()
    status = evaluate.start_eval("ngram", "plain", 20)
    assert "et-callout" in status
    job = next(j for j in jobs.list_jobs() if j["kind"] == "evaluation")
    jobs.wait(job["job_id"], timeout=60)
    _status, cards, metrics, examples, _marker = evaluate.eval_poll("")
    assert "et-card" in cards
    assert not metrics.empty and not examples.empty


def test_evaluate_rejects_unavailable_engine():
    home.load_sample()
    status = evaluate.start_eval("hf:no-such-model", "plain", 20)
    assert "et-error" in status or "et-warn" in status
    assert not [j for j in jobs.list_jobs() if j["kind"] == "evaluation"]


def test_method_choices_follow_engine_capabilities():
    assert evaluate.methods_for("ngram") == ["plain"]
    assert "style_fewshot" in evaluate.methods_for("openai")


def _sine_wav(path: Path) -> Path:
    rate = 16000
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
        for i in range(rate * 3)
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


def test_voice_flow_blocks_synthesis_until_consent(tmp_path):
    badge, table, *_ = voice.save_rec(str(_sine_wav(tmp_path / "c.wav")), "en", "")
    assert "et-badge" in badge and not table.empty
    voice.do_select_refs(False)
    status = voice.synth_start("hello there", "")
    assert "consent" in status.lower()
    voice.grant("tester", True)
    assert voice.consent_badge().count("granted") >= 1
    voice.revoke()
    assert "not granted" in voice.consent_badge().lower()


def test_settings_rejects_remote_openai_endpoint():
    status = settings.save_openai("http://example.com:8000/v1", "m", "")
    assert "et-warn" in status or "et-error" in status


def test_home_helpers_render():
    assert "et-callout" in home.next_step_html()
    assert list(home.engines_df().columns) == ["Engine", "Available", "Note"]
