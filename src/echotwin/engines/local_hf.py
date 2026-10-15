"""Local Hugging Face engine (transformers). Weights load lazily on first use."""

from __future__ import annotations

import gc
from collections.abc import Callable

from ..models.generation import GeneratorConfig, TransformersGenerator
from ..models.registry import get_model_spec
from .base import BaseEngine, EngineCapabilities


class LocalHFEngine(BaseEngine):
    kind = "local_hf"
    capabilities = EngineCapabilities(uses_prompt_context=True, supports_seed=True, needs_network=False)

    def __init__(self, spec_name: str, loader: Callable | None = None) -> None:
        self.spec = get_model_spec(spec_name)
        self.name = f"hf:{self.spec.name}"
        self._loader = loader
        self._gen: TransformersGenerator | None = None

    def _ensure(self) -> TransformersGenerator:
        if self._gen is None:
            loader = self._loader
            if loader is None:
                from ..models.loader import load_model as loader
            self._gen = TransformersGenerator(loader(self.spec), self.spec)
        return self._gen

    def _generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        return self._ensure().generate(messages, cfg)

    def close(self) -> None:
        self._gen = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
