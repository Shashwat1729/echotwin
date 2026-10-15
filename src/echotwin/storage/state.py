"""Tiny JSON state file for CLI session continuity (last import, target author...)."""

from __future__ import annotations

import json
from pathlib import Path

from ..settings import get_settings

_KEY = "state"


def _path() -> Path:
    p = get_settings().home / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save(state: dict) -> None:
    _path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def get(key: str, default=None):
    return load().get(key, default)


def set(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)
