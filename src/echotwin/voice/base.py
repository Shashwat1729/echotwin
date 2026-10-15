"""Voice backend protocol.

The protocol is capability-optional: backends declare what they support via
``capabilities()``, and feature calls validate capability flags first and raise
``UnsupportedCapabilityError`` rather than silently emulating behaviour the
backend does not have.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VoiceBackend(Protocol):
    name: str

    def capabilities(self) -> dict: ...

    def load(self, config: dict) -> None: ...

    def inspect_reference(self, audio_path: Path, transcript: str | None = None) -> dict: ...

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        transcript: str | None = None,
        output_path: Path | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Path: ...

    def health_check(self) -> dict: ...


class AbstractVoiceBackend(ABC):
    """Base class enforcing capability checks and a uniform error surface."""

    name = "abstract"

    def __init__(self) -> None:
        self._loaded = False
        self._config: dict = {}

    @abstractmethod
    def capabilities(self) -> dict: ...

    def load(self, config: dict) -> None:
        self._config = dict(config or {})
        self._loaded = True

    def _require_capability(self, key: str, value: Any = True) -> Any:
        caps = self.capabilities()
        actual = caps.get(key)
        if actual != value:
            from ..exceptions import UnsupportedCapabilityError

            feature = key.replace("supports_", "").replace("_", " ")
            raise UnsupportedCapabilityError(
                f"Backend '{self.name}' does not support '{feature}' "
                f"(capability {key}={actual!r}, expected {value!r}). "
                "Check `echotwin voice backends` and the capability matrix in "
                "docs/voice-cloning.md."
            )
        return actual

    @abstractmethod
    def inspect_reference(self, audio_path: Path, transcript: str | None = None) -> dict: ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        transcript: str | None = None,
        output_path: Path | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Path: ...

    def health_check(self) -> dict:
        caps = self.capabilities()
        return {
            "backend": self.name,
            "loaded": self._loaded,
            "available": self.is_available(),
            "capabilities": caps,
        }

    def is_available(self) -> bool:
        """True when the backend's dependencies AND model weights are present.

        Verified without downloading: imports and cached model files only.
        """
        return False
