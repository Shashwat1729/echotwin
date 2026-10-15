"""Shared page context and error helpers for the studio pages."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

import gradio as gr

from ..exceptions import EchoTwinError
from . import render

log = logging.getLogger("echotwin.ui")

TAB_IDS = ("home", "import", "dataset", "profile", "evaluate", "chat", "voice", "settings")


@dataclass
class PageContext:
    rail: gr.HTML
    tabs: gr.Tabs
    timer: gr.Timer
    tab: gr.Tab | None = None  # the page's own tab, for refresh-on-select


def rail_value() -> str:
    stages = render.stage_status()
    active = next((s.key for s in stages if s.state == "current"), "")
    return render.rail_html(stages, active)


def warn(exc: Exception) -> str:
    """Toast the problem and return an inline callout so it stays visible."""
    if isinstance(exc, EchoTwinError):
        message, kind = str(exc), "warn"
    else:
        log.exception("unexpected UI error")  # logs the traceback, never message content
        message, kind = f"Unexpected error ({type(exc).__name__}). See the server log.", "error"
    with contextlib.suppress(Exception):  # outside a Gradio request (tests) there is no toast target
        gr.Warning(message)
    return render.callout(kind, message)


def go(tab_id: str) -> gr.Tabs:
    return gr.Tabs(selected=tab_id)
