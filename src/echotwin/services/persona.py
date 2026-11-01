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


def list_voices() -> list[tuple[str, str]]:
    """Saved voices as ``(label, profile_id)`` pairs for a picker."""
    from ..voice.metadata import list_profiles

    profiles, _corrupt = list_profiles()
    return [(f"{p.name} ({len(p.recordings)} clips)", p.profile_id) for p in profiles]


def use_voice(profile_id: str | None) -> None:
    """Make a saved voice the one replies are spoken with."""
    from ..voice.metadata import load_profile

    if not profile_id:
        return
    load_profile(profile_id)  # raises if it does not exist
    from ..workspace import get_store

    get_store().set_pointer(voice_service.POINTER, profile_id)


def add_voice(
    persona_name: str,
    audio_paths: list[str],
    *,
    acknowledge: bool,
    learn_text: bool = True,
) -> str:
    """Create a voice from uploaded audio (no script needed) and make it current.

    Each upload is validated, cut at pauses when long, and transcribed
    automatically. When ``learn_text`` is set and no chat text exists yet, the
    transcripts also become the persona's writing corpus, so a voice alone is
    enough to start talking. ``acknowledge`` records the one-time statement that
    the voice is the user's own or they are entitled to use it.
    """
    from ..voice.metadata import VoiceProfile, save_profile
    from ..workspace import get_store

    name = persona_name.strip() or "Persona"
    paths = [p for p in audio_paths or [] if p]
    if not paths:
        raise EchoTwinError("Upload at least one voice recording.")
    if not acknowledge:
        raise EchoTwinError(
            "Tick the box confirming you may use this voice, and will label AI speech."
        )
    prof = VoiceProfile(name=name)
    save_profile(prof)
    get_store().set_pointer(voice_service.POINTER, prof.profile_id)
    for p in paths:
        voice_service.save_recording(p, "auto")
    voice_service.transcribe_missing()
    voice_service.select_refs(prefer_transcript=True)
    if not voice_service.consent_status().granted:
        voice_service.grant_consent(name)
    prof = voice_service.current_profile()
    transcripts = [r.transcript for r in (prof.recordings if prof else []) if r.transcript]
    note = f"Voice '{name}' ready from {len(paths)} upload(s)."
    if learn_text and transcripts and not get_store().pointer("current_dataset"):
        create_from_text(name, "\n".join(transcripts))
        note += " Their speech was transcribed and used to learn how they talk."
    return note


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
