"""Build and read the interpretable style profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..exceptions import DatasetError
from ..profiling.style_profile import (
    StyleProfile,
    build_style_profile,
    load_profile,
    profile_prompt_fragment,
    save_profile,
)
from ..settings import get_settings
from ..workspace import get_store
from . import imports


@dataclass(frozen=True)
class ProfileResult:
    path: Path
    fragment: str
    profile: StyleProfile


def build() -> ProfileResult:
    store = get_store()
    author = store.pointer("current_author")
    if not author:
        raise DatasetError("Choose the author to clone before building a profile.")
    messages, _ = imports.load_messages()
    profile = build_style_profile(messages, author)
    path = get_settings().artifacts_dir / "profiles" / f"profile-{author[:8]}.json"
    save_profile(profile, path)
    store.set_pointer("current_profile", str(path))
    return ProfileResult(path, profile_prompt_fragment(profile), profile)


def current_path() -> Path | None:
    value = get_store().pointer("current_profile")
    return Path(value) if value and Path(value).exists() else None


def current_fragment() -> str | None:
    path = current_path()
    return profile_prompt_fragment(load_profile(path)) if path else None
