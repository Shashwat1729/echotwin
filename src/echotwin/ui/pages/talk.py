"""Talk: converse with a persona by text or voice, hearing replies in their voice."""

from __future__ import annotations

import gradio as gr

from ...exceptions import EchoTwinError
from ...services import chat as chat_service
from ...services import persona
from .. import render
from ..context import PageContext, warn
from .evaluate import engine_choices, methods_for


def make_persona(name: str, text: str) -> str:
    try:
        author = persona.create_from_text(name, text)
    except Exception as exc:
        return warn(exc)
    return render.callout("ok", f"Persona '{author}' is ready. Start talking below.")


def persona_note() -> str:
    st = persona.status()
    who = st.author or "nobody yet"
    kind = "ok" if st.ready_text else "warn"
    return render.callout(
        kind,
        f"Persona: {who}. Text chat {'ready' if st.ready_text else 'needs data'}. {st.voice_note}",
    )


def turn(audio, text, history, session, engine, method, speak):
    text = (text or "").strip()
    if audio and not text:
        try:
            text = persona.listen(audio)
        except Exception as exc:
            return history, session, "", None, warn(exc), None
    if not text:
        return history, session, "", None, "", None
    try:
        reply, wav, note = persona.converse(session, text, engine, method, voice=bool(speak))
    except EchoTwinError as exc:
        return history, session, "", None, warn(exc), None
    except Exception as exc:
        return (
            history,
            session,
            "",
            None,
            render.callout("error", f"{type(exc).__name__}: {exc}"),
            None,
        )
    if reply.error:
        return history, session, "", None, render.callout("error", reply.error), None
    msg = render.callout("info", note) if (speak and note) else ""
    return list(session.history), session, "", None, msg, wav


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "### Talk to a person\n"
        "Paste what they wrote or said (or import their chats), optionally add their voice "
        "in the **Voice** tab, then talk. Replies copy their style; they are **AI-generated "
        "imitations**, not the real person."
    )
    with gr.Accordion("1. Create a persona from pasted text", open=False):
        name = gr.Textbox(label="Name of the person")
        text = gr.Textbox(
            label="Their text / transcripts",
            lines=8,
            placeholder="Paste messages, speeches or transcripts. 'Name: line' keeps other speakers.",
        )
        make = gr.Button("Create persona")
        made = gr.HTML()
        make.click(make_persona, [name, text], made)
    gr.HTML(persona_note())
    choices = engine_choices()
    with gr.Row():
        engine = gr.Dropdown(choices, value=choices[0] if choices else None, label="Engine")
        method = gr.Dropdown(
            methods_for(choices[0]) if choices else ["plain"],
            value="style_rag" if choices else "plain",
            label="Method",
        )
        speak = gr.Checkbox(value=True, label="Speak replies in their voice")
    chatbot = gr.Chatbot(height=380, label="Conversation")
    session = gr.State(chat_service.ChatSession())
    with gr.Row():
        mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Say something")
        box = gr.Textbox(label="...or type", lines=2)
    send = gr.Button("Send", variant="primary")
    note = gr.HTML()
    out = gr.Audio(label="Their reply (voice)", type="filepath", interactive=False, autoplay=True)
    ins = [mic, box, chatbot, session, engine, method, speak]
    outs = [chatbot, session, box, mic, note, out]
    send.click(turn, ins, outs)
    box.submit(turn, ins, outs)
