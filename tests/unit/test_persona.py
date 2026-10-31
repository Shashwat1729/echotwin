"""Persona creation from pasted text and the voice acknowledgement wording."""

from __future__ import annotations

from echotwin.ingestion.whatsapp import WhatsAppParser
from echotwin.services.persona import chat_text_from_paste
from echotwin.voice.consent import VOICE_CONSENT_STATEMENT


def test_paste_becomes_parseable_chat():
    text = "Hello there\nFriend: how are you?\n\nI am well, thanks"
    chat = chat_text_from_paste("Grandpa", text)
    msgs = WhatsAppParser().parse_text(chat, "c0").messages
    assert [m.sender_name for m in msgs] == ["Grandpa", "Friend", "Grandpa"]
    assert msgs[1].text == "how are you?"


def test_consent_statement_is_one_plain_sentence_pair():
    assert "entitled" in VOICE_CONSENT_STATEMENT
    assert "AI-generated" in VOICE_CONSENT_STATEMENT
