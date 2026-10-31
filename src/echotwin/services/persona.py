"""Talk to a persona: paste text or upload chats, optionally add a voice, then converse.

A persona is just the current author's dataset plus (optionally) the current voice
profile. Text replies always work; spoken replies need a voice backend, a selected
reference and the one-time voice acknowledgement.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import EchoTwinError
from ..settings import get_settings
from ..voice.transcription import transcribe
from . import chat, datasets, imports
from . import voice as voice_service

_SPEAKER = re.compile(r"^\s*([^:\n]{1,40}):\s+(.+)$")


def chat_text_from_paste(persona: str, text: str) -> str:
    """Turn pasted text into a WhatsApp-style chat so the normal pipeline can read it.

    Lines like ``Name: message`` keep their speaker; plain lines belong to ``persona``.
    Another speaker is inserted between paragraphs so every message has context.
    """
    persona = persona.strip() or "Persona"
    t0 = time.mktime((2020, 1, 1, 9, 0, 0, 0, 0, -1))
    out: list[str] = []
    n = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER.match(line)
        who, body = (m.group(1).strip(), m.group(2).strip()) if m else (persona, line)
        stamp = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime(t0 + n * 60))
        out.append(f"[{stamp}] {who}: {body}")
        n += 1
    return "\n".join(out) + "\n"


@dataclass(frozen=True)
class PersonaStatus:
    ready_text: bool
    author: str | None
    voice_ready: bool
    voice_note: str


def create_from_text(persona: str, text: str) -> str:
    """Import pasted text, select ``persona`` as the author and prepare the dataset."""
    if not text.strip():
        raise EchoTwinError("Paste some text written or said by the person first.")
    folder = get_settings().home / "pasted"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', persona) or 'persona'}_chat.txt"
    path.write_text(chat_text_from_paste(persona, text), encoding="utf-8")
    imports.import_export(path)
    imports.select_author(persona.strip() or "Persona")
    datasets.prepare()
    return persona.strip() or "Persona"


def status() -> PersonaStatus:
    from ..workspace import get_store

    store = get_store()
    author = store.pointer("current_author")
    ready = bool(store.pointer("current_dataset"))
    prof = voice_service.current_profile()
    if prof is None or not prof.recordings:
        return PersonaStatus(
            ready, author, False, "No voice yet: upload a recording in the Voice tab."
        )
    if not (prof.reference_selection or {}).get("selected"):
        return PersonaStatus(
            ready, author, False, "Voice uploaded; choose references in the Voice tab."
        )
    if not voice_service.consent_status().granted:
        return PersonaStatus(
            ready, author, False, "Tick the voice acknowledgement in the Voice tab."
        )
    return PersonaStatus(ready, author, True, "Voice ready.")


def listen(audio_path: str | Path, language: str | None = None) -> str:
    """Speech to text for the user's spoken turn."""
    return transcribe(audio_path, language)


def speak(text: str) -> tuple[str | None, str]:
    """Synthesize ``text`` in the persona's voice. Returns (wav path or None, note)."""
    st = status()
    if not st.voice_ready:
        return None, st.voice_note
    try:
        result = voice_service.synthesize(text)
    except EchoTwinError as exc:
        return None, str(exc)
    return str(result.output_path), f"Spoken with {result.backend}."


def converse(session: chat.ChatSession, text: str, engine: str, method: str, *, voice: bool):
    """One turn: text in, reply text out, plus audio when requested and available."""
    reply = chat.reply(session, text, engine, method)
    audio, note = (None, "")
    if voice and reply.text and not reply.error:
        audio, note = speak(reply.text)
    return reply, audio, note


def default_wav_dir() -> Path:
    return get_settings().home / "exports"
