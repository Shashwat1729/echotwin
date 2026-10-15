"""Text engines: n-gram, mock gate, OpenAI-compatible client, registry."""

from __future__ import annotations

import http.server
import json
import threading
from typing import ClassVar

import pytest

from echotwin.dataset.builder import ChatTurn, ExampleMetadata, TrainingExample, save_jsonl
from echotwin.engines import (
    BaseEngine,
    EngineCapabilities,
    Generation,
    MockEngine,
    NGramEngine,
    OpenAICompatEngine,
    mock_allowed,
    registry,
)
from echotwin.exceptions import ModelNotAvailableError
from echotwin.models.generation import GeneratorConfig
from echotwin.models.ngram import NGramModel
from echotwin.settings import get_settings
from echotwin.workspace import get_store, reset_store

TEXTS = [
    "haan bro kal milte hain 6 bje",
    "haan bro kal milte hain library mein",
    "theek hai done, kal baat karte hain",
    "morning! ummeed hai aaj ka din achha jayega",
]
CFG = GeneratorConfig(seed=3, temperature=0.9, max_new_tokens=32)
MSGS = [{"role": "user", "content": "kal milte hain?"}]


def _write_dataset(ds_id: str = "ds-t") -> None:
    exs = [
        TrainingExample(
            messages=[ChatTurn(role="user", content="ping"), ChatTurn(role="assistant", content=t)],
            metadata=ExampleMetadata(conversation_id=f"c{i}", language="hinglish", example_type="natural_reply"),
        )
        for i, t in enumerate(TEXTS)
    ]
    d = get_settings().artifacts_dir / "datasets" / ds_id
    save_jsonl(exs, d / "train.jsonl")
    save_jsonl(exs, d / "examples.jsonl")


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    registry.reset()
    reset_store()


def test_ngram_engine_deterministic_per_seed():
    eng = NGramEngine(NGramModel(order=3).train_texts(TEXTS))
    a = eng.generate_full(MSGS, CFG)
    b = eng.generate_full(MSGS, CFG)
    assert a.error is None and a.text and a.text == b.text
    assert a.engine == eng.name


def test_failing_engine_returns_error_not_raise():
    class Boom(BaseEngine):
        kind = "mock"
        capabilities = EngineCapabilities(False, False, False)

        def __init__(self) -> None:
            self.name = "boom"

        def _generate(self, messages, cfg):
            raise RuntimeError("kaput")

    g = Boom().generate_full(MSGS, CFG)
    assert isinstance(g, Generation)
    assert g.text == "" and "kaput" in (g.error or "")
    with pytest.raises(ModelNotAvailableError):
        Boom().generate(MSGS, CFG)


def test_mock_gate():
    assert mock_allowed({"ECHOTWIN_ENABLE_MOCK": "1"})
    assert mock_allowed({"PYTEST_CURRENT_TEST": "x"})
    assert not mock_allowed({})
    assert MockEngine(["a"], seed=1).generate(MSGS, CFG)


class _Handler(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        out = json.dumps({"choices": [{"message": {"content": "hi there"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def stub_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_compat_roundtrip(stub_server):
    eng = OpenAICompatEngine(stub_server, "m1")
    g = eng.generate_full(MSGS, CFG)
    assert g.text == "hi there" and g.error is None
    assert _Handler.seen[-1]["model"] == "m1"
    assert _Handler.seen[-1]["messages"] == MSGS


def test_openai_compat_unreachable_is_error():
    g = OpenAICompatEngine("http://127.0.0.1:9/v1", "m", timeout=1).generate_full(MSGS, CFG)
    assert g.text == "" and g.error


def test_openai_compat_rejects_remote(monkeypatch):
    monkeypatch.delenv("ECHOTWIN_ALLOW_REMOTE_ENGINE", raising=False)
    with pytest.raises(ModelNotAvailableError):
        OpenAICompatEngine("https://api.example.com/v1", "m")
    monkeypatch.setenv("ECHOTWIN_ALLOW_REMOTE_ENGINE", "1")
    OpenAICompatEngine("https://api.example.com/v1", "m")


def test_registry_lists_ngram_availability():
    infos = {i.id: i for i in registry.list_engines()}
    assert not infos["ngram"].available and "dataset" in infos["ngram"].reason_unavailable.lower()
    _write_dataset()
    get_store().set_pointer("current_dataset", "ds-t")
    infos = {i.id: i for i in registry.list_engines()}
    assert infos["ngram"].available and infos["ngram"].reason_unavailable is None
    hf = [i for i in infos.values() if i.kind == "local_hf"]
    assert hf and all(not i.available and i.reason_unavailable for i in hf)
    assert "mock" not in infos or mock_allowed()


def test_registry_ngram_for_dataset_trains_and_caches():
    _write_dataset()
    e1 = registry.get_engine("ngram", dataset_id="ds-t")
    assert e1.generate(MSGS, CFG)
    assert (get_settings().artifacts_dir / "models" / "ngram-ds-t.json").exists()


def test_registry_reuses_and_closes_on_switch():
    closed: list[str] = []

    class Fake(BaseEngine):
        kind = "local_hf"
        capabilities = EngineCapabilities(True, True, False)

        def __init__(self, n: str) -> None:
            self.name = n

        def _generate(self, messages, cfg):
            return self.name

        def close(self) -> None:
            closed.append(self.name)

    registry.register_factory("fake", lambda ident, ds: Fake(ident), heavy=True)
    a1 = registry.get_engine("fake:a")
    a2 = registry.get_engine("fake:a")
    assert a1 is a2
    registry.get_engine("fake:b")
    assert closed == ["fake:a"]


def test_registry_unknown_engine():
    with pytest.raises(ModelNotAvailableError):
        registry.get_engine("nope:x")


def test_plain_method_sends_prompt_only():
    from echotwin.evaluation.baseline import METHODS, build_messages

    prompt = {"messages": [{"role": "user", "content": "hi"}]}
    assert "plain" in METHODS
    assert build_messages("plain", prompt, "STYLE", [{"context": "a", "response": "b"}]) == prompt["messages"]
