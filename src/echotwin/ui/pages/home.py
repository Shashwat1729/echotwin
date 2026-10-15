"""Home: where you are, what to do next, which engines work, sample data."""

from __future__ import annotations

import gradio as gr
import pandas as pd
from rich.console import Console

from ...services import models
from .. import render
from ..context import PageContext, rail_value, warn

_NEXT = {
    "import": "Import a WhatsApp export (.zip or .txt) on the Import tab, or load the sample data.",
    "author": "Choose which participant to clone on the Import tab.",
    "dataset": "Build the training dataset on the Dataset tab.",
    "profile": "Build the style profile on the Profile tab.",
    "evaluate": "Run a baseline on the Evaluate tab, or start chatting.",
    "chat": "Chat with your clone on the Chat tab.",
}


def next_step_html() -> str:
    stages = [s for s in render.stage_status() if s.key != "voice"]
    pending = next((s for s in stages if s.state == "current" and s.key != "chat"), None)
    if pending is None:
        return render.callout("ok", _NEXT["chat"])
    return render.callout("info", _NEXT[pending.key])


def engines_df() -> pd.DataFrame:
    rows = [
        {
            "Engine": info.id,
            "Available": "yes" if info.available else "no",
            "Note": info.reason_unavailable or info.label,
        }
        for info in models.list_engines()
    ]
    return pd.DataFrame(rows, columns=["Engine", "Available", "Note"])


def load_sample() -> tuple[str, str, pd.DataFrame, str]:
    from ...demo import run_demo

    try:
        run_demo(Console(quiet=True))
        status = render.callout(
            "ok",
            "Synthetic sample loaded: 6 chats, author Aarav, dataset, profile and a baseline. "
            "This is generated data, not a real person.",
        )
    except Exception as exc:
        status = warn(exc)
    return status, next_step_html(), engines_df(), rail_value()


def refresh() -> tuple[str, pd.DataFrame]:
    return next_step_html(), engines_df()


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "Learn how one person writes from a WhatsApp export, then chat with a clone that "
        "writes like them. Everything stays on this machine."
    )
    next_step = gr.HTML(next_step_html())
    with gr.Row():
        sample_btn = gr.Button("Load sample data", variant="primary")
        gr.Markdown("A synthetic multi-chat export, so you can try every step without private data.")
    status = gr.HTML()
    gr.Markdown("### Engines")
    engines = gr.Dataframe(engines_df(), interactive=False, wrap=True)
    sample_btn.click(load_sample, outputs=[status, next_step, engines, ctx.rail])
    if ctx.tab is not None:
        ctx.tab.select(refresh, outputs=[next_step, engines])
