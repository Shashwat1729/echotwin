"""Pluggable text engines behind one contract."""

from . import registry
from .base import BaseEngine, EngineCapabilities, Generation
from .local_hf import LocalHFEngine
from .mock import MockEngine, mock_allowed
from .ngram import NGramEngine
from .openai_compat import OpenAICompatEngine
from .registry import EngineInfo, get_engine, list_engines

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "EngineInfo",
    "Generation",
    "LocalHFEngine",
    "MockEngine",
    "NGramEngine",
    "OpenAICompatEngine",
    "get_engine",
    "list_engines",
    "mock_allowed",
    "registry",
]
