"""Import a WhatsApp export and choose the author to clone."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from ...services import imports
from .. import render
from ..context import PageContext, go, rail_value, warn


def _author_choices() -> list[str]:
    try:
        messages, _ = imports.load_messages()
    except Exception:
        return []
    return [p["sender_name"] for p in imports.participants_of(messages)]


def do_import(file: str | None):
    if not file:
        return render.callout("warn", "Choose an export file first."), None, gr.Dropdown(), rail_value()
    try:
        result = imports.import_export(file)
    except Exception as exc:
        return warn(exc), None, gr.Dropdown(), rail_value()
    names = [p["sender_name"] for p in result.participants]
    status = render.callout(
        "ok",
        f"Imported {result.n_messages} messages from {len(result.conversations)} chat(s). "
        "Now choose whose style to learn.",
    )
    return (
        status,
        render.participants_df(result.participants),
        gr.Dropdown(choices=names, value=None),
        rail_value(),
    )


def do_select_author(name: str | None):
    if not name:
        return render.callout("warn", "Choose a participant first."), rail_value()
    try:
        imports.select_author(name)
    except Exception as exc:
        return warn(exc), rail_value()
    return render.callout("ok", f"Author set to {name}. Next: build the dataset."), rail_value()


def current_participants() -> pd.DataFrame:
    try:
        messages, _ = imports.load_messages()
    except Exception:
        return render.participants_df([])
    return render.participants_df(imports.participants_of(messages))


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "Export a chat from WhatsApp (*Export chat, without media*) and drop the `.zip` or "
        "`.txt` here. Several chats in one zip are fine."
    )
    upload = gr.File(label="WhatsApp export", file_types=[".zip", ".txt"], type="filepath")
    import_btn = gr.Button("Import", variant="primary")
    status = gr.HTML()
    table = gr.Dataframe(current_participants(), interactive=False, wrap=True, label="Participants")
    with gr.Row():
        author = gr.Dropdown(_author_choices(), label="Author to clone", scale=3)
        author_btn = gr.Button("Use this author", scale=1)
    author_status = gr.HTML()
    next_btn = gr.Button("Next: Dataset")

    def refresh():
        return current_participants(), gr.Dropdown(choices=_author_choices())

    if ctx.tab is not None:
        ctx.tab.select(refresh, outputs=[table, author])

    import_btn.click(do_import, upload, [status, table, author, ctx.rail])
    author_btn.click(do_select_author, author, [author_status, ctx.rail])
    next_btn.click(lambda: go("dataset"), outputs=ctx.tabs)
