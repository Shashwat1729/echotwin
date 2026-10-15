"""Core services end to end on the synthetic exports (temp ECHOTWIN_HOME per test)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echotwin.engines import registry
from echotwin.exceptions import EchoTwinError, ModelNotAvailableError
from echotwin.services import chat, datasets, evaluation, imports, models, profiles
from echotwin.workspace import get_store, reset_store

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
MULTI = EXAMPLES / "synthetic_multi.zip"
SINGLE = EXAMPLES / "synthetic_whatsapp.txt"


@pytest.fixture(autouse=True)
def _clean():
    yield
    registry.reset()
    reset_store()


@pytest.fixture()
def prepared():
    imports.import_export(MULTI)
    imports.select_author("Aarav")
    return datasets.prepare()


def test_import_lists_participants_and_sets_pointer():
    res = imports.import_export(MULTI)
    assert "Aarav" in [p["sender_name"] for p in res.participants]
    assert len(res.conversations) >= 6 and res.n_messages > 50
    assert get_store().pointer("current_import") == res.import_id
    msgs, import_id = imports.load_messages()
    assert import_id == res.import_id and len(msgs) == res.n_messages


def test_select_author_sets_pointer_and_rejects_unknown():
    imports.import_export(MULTI)
    sid = imports.select_author("aarav")
    assert get_store().pointer("current_author") == sid
    with pytest.raises(EchoTwinError):
        imports.select_author("Nobody Here")


def test_new_import_clears_author_and_dataset_pointers(prepared):
    imports.import_export(MULTI)
    ptr = get_store().pointers()
    assert ptr["current_author"] is None and ptr["current_dataset"] is None


def test_prepare_requires_author():
    imports.import_export(MULTI)
    with pytest.raises(EchoTwinError):
        datasets.prepare()


def test_prepare_has_heldout_split_and_manifest(prepared):
    assert prepared.n_test > 0 and prepared.n_train > 0
    assert prepared.leakage_check_passed
    assert get_store().pointer("current_dataset") == prepared.dataset_id
    ds = datasets.load()
    assert len(ds.train) == prepared.n_train and len(ds.test) == prepared.n_test
    assert len(ds.all) == prepared.n_examples
    assert datasets.manifest()["dataset_id"] == prepared.dataset_id


def test_profile_build_and_current_fragment(prepared):
    assert profiles.current_fragment() is None
    res = profiles.build()
    assert Path(res.path).exists() and res.fragment
    assert profiles.current_fragment() == res.fragment


@pytest.mark.parametrize("method", ["plain", "style_rag"])
def test_evaluation_runs_with_ngram(prepared, method):
    profiles.build()
    seen: list[float] = []
    res = evaluation.run(method, "ngram", max_prompts=10, progress=lambda f, m="": seen.append(f))
    assert Path(res.predictions_path).exists()
    assert res.report is not None and seen and seen[-1] == 1.0


def test_evaluation_on_empty_test_split_raises():
    imports.import_export(SINGLE)
    imports.select_author("Aarav")
    ds = datasets.prepare()
    assert ds.n_test == 0
    with pytest.raises(EchoTwinError):
        evaluation.run("plain", "ngram")


def test_chat_keeps_history_and_builds_index_once(prepared, monkeypatch):
    calls = {"n": 0}
    real = chat.build_index_from_examples

    def counting(examples):
        calls["n"] += 1
        return real(examples)

    monkeypatch.setattr(chat, "build_index_from_examples", counting)
    session = chat.ChatSession(history=[])
    r1 = chat.reply(session, "kal milte hain?", "ngram", "style_rag")
    r2 = chat.reply(session, "kitne baje?", "ngram", "style_rag")
    assert r1.error is None and r1.text and r2.text
    assert r1.engine and r1.method == "style_rag"
    assert len(session.history) == 4
    assert calls["n"] == 1


def test_chat_without_dataset_raises_not_fakes():
    with pytest.raises(EchoTwinError):
        chat.reply(chat.ChatSession(history=[]), "hi", "ngram", "plain")


def test_chat_unavailable_engine_returns_error_and_keeps_history(prepared):
    session = chat.ChatSession(history=[])
    r = chat.reply(session, "hi", "openai", "plain")
    assert r.text == "" and r.error
    assert session.history == []


def test_models_configure_openai_loopback_only(monkeypatch):
    monkeypatch.delenv("ECHOTWIN_ALLOW_REMOTE_ENGINE", raising=False)
    with pytest.raises(ModelNotAvailableError):
        models.configure_openai("https://api.example.com/v1", "m")
    models.configure_openai("http://127.0.0.1:11434/v1", "llama3", None)
    infos = {i.id: i for i in models.list_engines()}
    assert infos["openai:llama3"].available
    models.clear_openai()
    assert "openai:llama3" not in {i.id for i in models.list_engines()}
