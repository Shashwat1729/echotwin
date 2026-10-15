"""Engine discovery and the optional OpenAI-compatible endpoint setting."""

from __future__ import annotations

from ..engines import EngineInfo, OpenAICompatEngine
from ..engines import list_engines as _list_engines
from ..engines.registry import OPENAI_KV
from ..workspace import get_store


def list_engines(dataset_id: str | None = None) -> list[EngineInfo]:
    return _list_engines(dataset_id)


def configure_openai(base_url: str, model: str, api_key: str | None = None) -> None:
    """Save an OpenAI-compatible endpoint. Raises for non-loopback hosts unless allowed."""
    OpenAICompatEngine(base_url, model, api_key)  # validates the host before saving
    get_store().kv_set(OPENAI_KV, {"base_url": base_url, "model": model, "api_key": api_key})


def clear_openai() -> None:
    get_store().kv_set(OPENAI_KV, None)
