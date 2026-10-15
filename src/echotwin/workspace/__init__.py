"""Workspace store: the single source of truth for pointers, jobs and metadata."""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path

from ..settings import get_settings
from .migrations import POINTERS
from .store import LEGACY_KEYS, Store

__all__ = ["LEGACY_KEYS", "POINTERS", "Store", "get_store", "reset_store"]

_stores: dict[Path, Store] = {}
_lock = threading.Lock()


def get_store() -> Store:
    """Process-wide Store for the current ``ECHOTWIN_HOME`` (created on first use)."""
    settings = get_settings()
    settings.ensure_dirs()
    path = settings.db_path
    with _lock:
        store = _stores.get(path)
        if store is None:
            store = Store(path)
            _import_legacy_state(store, settings.home / "state.json")
            _stores[path] = store
        return store


def reset_store() -> None:
    """Close and forget every cached Store (tests, home switches)."""
    with _lock:
        for s in _stores.values():
            s.close()
        _stores.clear()


def _import_legacy_state(store: Store, legacy: Path) -> None:
    """One-time import of the old ``state.json``; never raises on bad data."""
    if not legacy.exists():
        return
    try:
        data = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    for key, value in data.items():
        if key in LEGACY_KEYS:
            store.set_pointer(LEGACY_KEYS[key], None if value is None else str(value))
        else:
            store.kv_set(key, value)
    with contextlib.suppress(OSError):
        legacy.replace(legacy.with_name("state.json.migrated"))
