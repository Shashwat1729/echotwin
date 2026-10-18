"""Engine discovery and lifecycle.

Engine ids: ``ngram``, ``hf:<spec>``, ``openai:<model>``, ``mock``. One heavy
engine (local weights) stays resident; switching closes the previous one.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ModelNotAvailableError
from ..models.registry import list_models
from ..settings import get_settings
from ..workspace import get_store
from .base import BaseEngine, EngineCapabilities
from .local_hf import LocalHFEngine
from .mock import MockEngine, mock_allowed
from .ngram import NGramEngine
from .openai_compat import OpenAICompatEngine

Factory = Callable[[str, str | None], BaseEngine]
OPENAI_KV = "engine_openai"


@dataclass(frozen=True)
class EngineInfo:
    id: str
    kind: str
    label: str
    available: bool
    reason_unavailable: str | None
    capabilities: EngineCapabilities


_lock = threading.RLock()
_factories: dict[str, tuple[Factory, bool]] = {}
_cache: dict[tuple[str, str | None], BaseEngine] = {}
_heavy_key: tuple[str, str | None] | None = None


def register_factory(prefix: str, factory: Factory, heavy: bool = False) -> None:
    with _lock:
        _factories[prefix] = (factory, heavy)


def _openai_config() -> dict | None:
    cfg = get_store().kv_get(OPENAI_KV)
    return cfg if isinstance(cfg, dict) and cfg.get("base_url") and cfg.get("model") else None


def _openai_factory(_ident: str, _dataset: str | None) -> BaseEngine:
    cfg = _openai_config()
    if cfg is None:
        raise ModelNotAvailableError(
            "No OpenAI-compatible endpoint configured. Add one in Settings."
        )
    return OpenAICompatEngine(cfg["base_url"], cfg["model"], cfg.get("api_key"))


def _ngram_factory(_ident: str, dataset: str | None) -> BaseEngine:
    ds = dataset or get_store().pointer("current_dataset")
    if not ds:
        raise ModelNotAvailableError("Prepare a dataset first.")
    try:
        return NGramEngine.for_dataset(ds)
    except ValueError as exc:
        raise ModelNotAvailableError(str(exc)) from exc


def _install_defaults() -> None:
    _factories.clear()
    _factories["ngram"] = (_ngram_factory, False)
    _factories["hf"] = (lambda ident, _d: LocalHFEngine(ident.split(":", 1)[1]), True)
    _factories["openai"] = (_openai_factory, False)
    _factories["mock"] = (lambda _i, _d: MockEngine(), False)


_install_defaults()


def _hf_unavailable(spec) -> str | None:
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return "transformers/torch not installed (uv pip install -e '.[train]')"
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    root = Path(os.environ.get("HF_HUB_CACHE") or hf_home / "hub")
    if not (root / ("models--" + spec.hf_model_id.replace("/", "--"))).exists():
        return "Weights not downloaded (scripts/download_models.py)"
    return None


def list_engines(dataset_id: str | None = None) -> list[EngineInfo]:
    ds = dataset_id or get_store().pointer("current_dataset")
    train = get_settings().artifacts_dir / "datasets" / ds / "train.jsonl" if ds else None
    ngram_ok = bool(train and train.exists() and train.stat().st_size > 0)
    out = [
        EngineInfo(
            "ngram",
            "ngram",
            "N-gram (instant, CPU)",
            ngram_ok,
            None if ngram_ok else "Prepare a dataset first",
            NGramEngine.capabilities,
        )
    ]
    for spec in list_models():
        reason = _hf_unavailable(spec)
        out.append(
            EngineInfo(
                f"hf:{spec.name}",
                "local_hf",
                spec.hf_model_id,
                reason is None,
                reason,
                LocalHFEngine.capabilities,
            )
        )
    cfg = _openai_config()
    out.append(
        EngineInfo(
            f"openai:{cfg['model']}" if cfg else "openai",
            "openai",
            f"OpenAI-compatible ({cfg['model']})" if cfg else "OpenAI-compatible endpoint",
            cfg is not None,
            None if cfg else "Not configured (Settings)",
            OpenAICompatEngine.capabilities,
        )
    )
    if mock_allowed():
        out.append(
            EngineInfo("mock", "mock", "Mock (tests only)", True, None, MockEngine.capabilities)
        )
    return out


def get_engine(engine_id: str, dataset_id: str | None = None) -> BaseEngine:
    global _heavy_key
    prefix = engine_id.split(":", 1)[0]
    with _lock:
        entry = _factories.get(prefix)
        if entry is None:
            raise ModelNotAvailableError(f"Unknown engine {engine_id!r}.")
        factory, heavy = entry
        key = (engine_id, dataset_id if prefix == "ngram" else None)
        if key in _cache:
            return _cache[key]
        engine = factory(engine_id, dataset_id)
        if heavy:
            if _heavy_key is not None and _heavy_key in _cache:
                _cache.pop(_heavy_key).close()
            _heavy_key = key
        _cache[key] = engine
        return engine


def reset() -> None:
    """Close every cached engine and restore the default factories (tests)."""
    global _heavy_key
    with _lock:
        for eng in _cache.values():
            eng.close()
        _cache.clear()
        _heavy_key = None
        _install_defaults()
