"""Small HTML building blocks shared by the UI pages."""

from __future__ import annotations

from html import escape


def empty_state(title: str, body: str) -> str:
    return (
        '<div class="et-empty">'
        f'<div class="et-empty-title">{escape(title)}</div>'
        f'<div class="et-empty-body">{escape(body)}</div></div>'
    )


def status_badge(state: str, text: str) -> str:
    state = state if state in {"ok", "warn", "error", "info"} else "info"
    return f'<span class="et-badge et-{state}">{escape(text)}</span>'
