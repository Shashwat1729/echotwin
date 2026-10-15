"""Legacy key/value view over the workspace store.

Old call sites use ``state.get("last_dataset_id")``; those keys map to the typed
workspace pointers. Unknown keys fall through to the store's kv table. New code
should use ``echotwin.workspace.get_store()`` and its typed pointers directly.
"""

from __future__ import annotations

from typing import Any

from ..workspace import LEGACY_KEYS, get_store


def load() -> dict:
    store = get_store()
    out: dict[str, Any] = dict(store.kv_all())
    pointers = store.pointers()
    for legacy, pointer in LEGACY_KEYS.items():
        if pointers.get(pointer) is not None:
            out[legacy] = pointers[pointer]
    return out


def save(state: dict) -> None:
    for key, value in state.items():
        set(key, value)


def get(key: str, default=None):
    store = get_store()
    if key in LEGACY_KEYS:
        value = store.pointer(LEGACY_KEYS[key])
        return default if value is None else value
    return store.kv_get(key, default)


def set(key: str, value) -> None:  # noqa: A001 - legacy public name
    store = get_store()
    if key in LEGACY_KEYS:
        store.set_pointer(LEGACY_KEYS[key], None if value is None else str(value))
    else:
        store.kv_set(key, value)
