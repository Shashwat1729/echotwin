"""Chat with the clone. History lives in a per-browser ``ChatSession``."""

from __future__ import annotations

import gradio as gr

from ...exceptions import EchoTwinError
from ...services import chat as chat_service
from .. import components, render
from ..context import PageContext, go, warn
from .evaluate import engine_choices, methods_for, refresh_engines


def _why(reply: chat_service.ChatReply) -> str:
    lines = [f"**Engine** `{reply.engine}` · **method** `{reply.method}` · {reply.latency_s:.2f}s"]
    if reply.exemplars:
        lines.append("**Similar real replies used as examples**")
        lines.extend(f"- {ex.get('response', '')[:160]}" for ex in reply.exemplars)
    return "\n".join(lines)


def send(text: str, history: list, session: chat_service.ChatSession, engine: str, method: str):
    text = (text or "").strip()
    if not text:
        return "", history, session, gr.skip(), ""
    try:
        reply = chat_service.reply(session, text, engine, method)
    except EchoTwinError as exc:
        return text, history, session, gr.skip(), warn(exc)
    if reply.error:
        return text, history, session, gr.skip(), render.callout("error", reply.error)
    return "", list(session.history), session, _why(reply), ""


def clear():
    return [], chat_service.ChatSession(), "", ""


def build(ctx: PageContext) -> None:
    choices = engine_choices()
    with gr.Row():
        engine = gr.Dropdown(choices, value=choices[0] if choices else None, label="Engine")
        method = gr.Dropdown(
            methods_for(choices[0]) if choices else ["plain"], value="plain", label="Method"
        )
    status = gr.HTML(
        ""
        if choices
        else components.empty_state(
            "No engine ready", "Prepare a dataset to enable the n-gram engine."
        )
    )
    chatbot = gr.Chatbot(height=420, label="Clone")
    session = gr.State(chat_service.ChatSession())
    box = gr.Textbox(placeholder="Say something...", show_label=False, lines=1)
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    with gr.Accordion("Why this reply", open=False):
        why = gr.Markdown()
    back = gr.Button("Back to Dataset")

    def on_engine(engine_id: str):
        options = methods_for(engine_id) if engine_id else ["plain"]
        return gr.Dropdown(choices=options, value=options[0])

    engine.change(on_engine, engine, method)
    if ctx.tab is not None:
        ctx.tab.select(refresh_engines, engine, engine)
    inputs = [box, chatbot, session, engine, method]
    outputs = [box, chatbot, session, why, status]
    box.submit(send, inputs, outputs)
    send_btn.click(send, inputs, outputs)
    clear_btn.click(clear, outputs=[chatbot, session, why, status])
    back.click(lambda: go("dataset"), outputs=ctx.tabs)
