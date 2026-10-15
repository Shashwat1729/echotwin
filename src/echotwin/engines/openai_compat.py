"""Client for OpenAI-compatible /chat/completions servers (Ollama, llama.cpp, vLLM, LM Studio).

Loopback only by default: chat history can contain private messages, so sending
it to a remote host needs ``ECHOTWIN_ALLOW_REMOTE_ENGINE=1``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from ..exceptions import ModelNotAvailableError
from ..models.generation import GeneratorConfig
from .base import BaseEngine, EngineCapabilities

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class OpenAICompatEngine(BaseEngine):
    kind = "openai"
    capabilities = EngineCapabilities(uses_prompt_context=True, supports_seed=True, needs_network=True)

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60
    ) -> None:
        host = urlparse(base_url).hostname or ""
        self._local = host in _LOOPBACK
        if not self._local and os.environ.get("ECHOTWIN_ALLOW_REMOTE_ENGINE") != "1":
            raise ModelNotAvailableError(
                f"Refusing remote engine host {host!r}: chat history would leave this machine. "
                "Set ECHOTWIN_ALLOW_REMOTE_ENGINE=1 to allow it."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"openai:{model}"

    def _generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        payload: dict = {"model": self.model, "messages": messages, "max_tokens": cfg.max_new_tokens}
        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            json.dumps(payload).encode(),
            headers,
            method="POST",
        )
        # Loopback calls must not go through a system proxy.
        open_: Callable[..., Any]
        if self._local:
            open_ =urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        else:
            open_ = urllib.request.urlopen
        with open_(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"]).strip()
