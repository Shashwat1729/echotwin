"""Backend registry: lazy discovery so heavy deps stay optional."""

from __future__ import annotations

import importlib

from ..exceptions import BackendNotAvailableError
from ..voice.base import AbstractVoiceBackend

_BACKEND_LOADERS = {
    "voxcpm": "echotwin.voice.voxcpm_backend:VoxCPMBackend",
    "chatterbox-multilingual": "echotwin.voice.alternative_backend:ChatterboxMultilingualBackend",
    "chatterbox": "echotwin.voice.alternative_backend:ChatterboxMultilingualBackend",
}

_cache: dict[str, AbstractVoiceBackend] = {}


def registered_backends() -> list[str]:
    return sorted(_BACKEND_LOADERS)


def get_backend(name: str, *, config: dict | None = None) -> AbstractVoiceBackend:
    key = name.lower()
    if key not in _BACKEND_LOADERS:
        raise BackendNotAvailableError(
            f"Unknown backend '{name}'. Known backends: {sorted(set(_BACKEND_LOADERS))}. "
            "Run 'echotwin voice backends' for capability details."
        )
    module_path, class_name = _BACKEND_LOADERS[key].split(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:  # pragma: no cover - defensive
        raise BackendNotAvailableError(f"Cannot import backend module {module_path}: {e}") from e
    backend_cls = getattr(module, class_name)
    if key in _cache:
        # Reuse the loaded instance: a fresh one re-read the multi-GB model
        # checkpoint on EVERY synthesis.
        instance = _cache[key]
        # Only (re)load when not yet loaded - reloading would re-read the
        # multi-GB checkpoint from disk for no reason.
        if config and not instance.health_check().get("loaded"):
            instance.load(config)
        return instance
    instance = backend_cls()
    if config:
        instance.load(config)
    _cache[key] = instance
    return instance


def backend_status(name: str) -> dict:
    """Status without heavy imports: capabilities come from the module which is
    imported lazily but defines capabilities as pure data."""
    try:
        backend = get_backend(name)
        return {
            "name": name,
            "module_importable": True,
            "capabilities": backend.capabilities(),
            "health": backend.health_check(),
        }
    except BackendNotAvailableError as e:
        return {"name": name, "module_importable": False, "error": str(e)}
    except Exception as e:
        return {"name": name, "module_importable": True, "error": str(e)}


def all_backend_statuses() -> list[dict]:
    return [backend_status(name) for name in sorted(set(_BACKEND_LOADERS))]


def find_available_backend(preferred: str | None = None) -> AbstractVoiceBackend | None:
    order = [preferred] if preferred else []
    order += [b for b in registered_backends() if b not in order]
    for name in order:
        try:
            backend = get_backend(name)
            if backend.is_available():
                return backend
        except BackendNotAvailableError:
            continue
    return None
