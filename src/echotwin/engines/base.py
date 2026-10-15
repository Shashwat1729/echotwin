"""Common contract for text engines.

An engine turns chat messages into one reply. ``generate_full`` never raises:
failures come back as ``Generation.error`` so callers (chat, evaluation) can show
them instead of crashing. ``generate`` keeps the older ``TextGenerator`` protocol
(returns ``str``, raises on failure) so engines also work with ``run_baseline``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..exceptions import ModelNotAvailableError
from ..models.generation import GeneratorConfig


@dataclass(frozen=True)
class Generation:
    text: str
    engine: str
    latency_s: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    uses_prompt_context: bool
    supports_seed: bool
    needs_network: bool


class BaseEngine:
    name: str = "engine"
    kind: str = "base"
    capabilities = EngineCapabilities(False, False, False)

    def _generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        raise NotImplementedError

    def generate_full(self, messages: list[dict], cfg: GeneratorConfig) -> Generation:
        start = time.perf_counter()
        try:
            text = self._generate(messages, cfg)
            return Generation(text, self.name, time.perf_counter() - start)
        except Exception as exc:
            return Generation(
                "", self.name, time.perf_counter() - start, error=str(exc) or type(exc).__name__
            )

    def generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        result = self.generate_full(messages, cfg)
        if result.error:
            raise ModelNotAvailableError(f"Engine {self.name} failed: {result.error}")
        return result.text

    def close(self) -> None:
        """Release heavy resources (weights, sockets). Safe to call twice."""
