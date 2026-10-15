"""Chat with the clone: real history, cached retrieval, any engine.

Never fabricates a reply: with no dataset this raises; with an unavailable engine
the reply carries ``error`` and the session history is left untouched.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..dataset.retrieval import RetrievalConfig, build_index_from_examples, retrieve_exemplars
from ..engines import registry
from ..evaluation.baseline import RETRIEVAL_METHODS, build_messages
from ..exceptions import ConfigurationError, DatasetError, ModelNotAvailableError
from ..models.generation import GeneratorConfig
from ..settings import get_settings
from ..workspace import get_store
from . import datasets, profiles

CHAT_METHODS = ("plain", "zero_shot", "style_profile", "few_shot", "style_fewshot", "style_rag")
HISTORY_TURNS = 6
FEW_SHOT_N = 5
RAG_K = 3


@dataclass
class ChatSession:
    history: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ChatReply:
    text: str
    engine: str
    method: str
    latency_s: float
    exemplars: list[dict]
    error: str | None = None


@dataclass(frozen=True)
class _Context:
    index: object
    few_shot: list[dict]
    fragment: str | None


_cache: dict[tuple, _Context] = {}
_cache_lock = threading.Lock()


def _context(dataset_id: str) -> _Context:
    """Retrieval index, few-shot pairs and profile fragment, cached until files change."""
    base = get_settings().artifacts_dir / "datasets" / dataset_id
    prof = profiles.current_path()
    key = (
        dataset_id,
        (base / "examples.jsonl").stat().st_mtime_ns,
        str(prof),
        prof.stat().st_mtime_ns if prof else None,
    )
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    ds = datasets.load(dataset_id)
    ctx = _Context(
        build_index_from_examples(ds.all),
        datasets.few_shot_pairs(ds.train, FEW_SHOT_N),
        profiles.current_fragment(),
    )
    with _cache_lock:
        _cache.clear()  # one dataset resident is enough for a personal tool
        _cache[key] = ctx
    return ctx


def reply(session: ChatSession, text: str, engine_id: str, method: str) -> ChatReply:
    if method not in CHAT_METHODS:
        raise ConfigurationError(f"Unknown chat method {method!r}; expected one of {CHAT_METHODS}")
    dataset_id = get_store().pointer("current_dataset")
    if not dataset_id:
        raise DatasetError("No dataset yet. Prepare a dataset before chatting.")
    ctx = _context(dataset_id)

    turns = [*session.history, {"role": "user", "content": text}][-HISTORY_TURNS:]
    exemplars: list[dict] = []
    if method in RETRIEVAL_METHODS:
        exemplars = retrieve_exemplars(
            ctx.index, [{"role": "user", "content": text}], cfg=RetrievalConfig(k=RAG_K)
        )
    messages = build_messages(method, {"messages": turns}, ctx.fragment, ctx.few_shot, exemplars)

    try:
        engine = registry.get_engine(engine_id, dataset_id=dataset_id)
    except ModelNotAvailableError as exc:
        return ChatReply("", engine_id, method, 0.0, exemplars, error=str(exc))
    gen = engine.generate_full(messages, GeneratorConfig())
    if gen.error is None:
        session.history.extend(
            [{"role": "user", "content": text}, {"role": "assistant", "content": gen.text}]
        )
    return ChatReply(gen.text, gen.engine, method, gen.latency_s, exemplars, gen.error)
