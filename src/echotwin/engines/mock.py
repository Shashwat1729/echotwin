"""Mock engine for tests and offline smoke runs. Never offered in normal use."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ..exceptions import ModelNotAvailableError
from ..models.generation import GeneratorConfig, MockGenerator
from .base import BaseEngine, EngineCapabilities


def mock_allowed(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("ECHOTWIN_ENABLE_MOCK") == "1" or "PYTEST_CURRENT_TEST" in env


class MockEngine(BaseEngine):
    kind = "mock"
    capabilities = EngineCapabilities(uses_prompt_context=False, supports_seed=True, needs_network=False)

    def __init__(self, exemplars: list[str] | None = None, seed: int = 0) -> None:
        if not mock_allowed():
            raise ModelNotAvailableError(
                "Mock engine is disabled. Set ECHOTWIN_ENABLE_MOCK=1 to use it."
            )
        self.name = "mock"
        self._gen = MockGenerator(exemplars, seed=seed)

    def _generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        return self._gen.generate(messages, cfg)
