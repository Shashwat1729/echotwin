from __future__ import annotations

import pytest

from echotwin.settings import reset_settings


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOTWIN_HOME", str(tmp_path / "echotwin-home"))
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def android_export() -> str:
    return """[15/01/2025, 09:12:44] Priya: Good morning everyone ☀️
[15/01/2025, 09:13:02] Aarav: morning! ummeed hai aaj ka din achha jayega
[15/01/2025, 09:14:10] Priya: guys canteen ka menu badal gaya
[15/01/2025, 09:14:12] Priya: paneer tikkaavailable from monday apparently
[15/01/2025, 09:20:00] Rohit: <Media omitted>
[15/01/2025, 09:21:05] Sana: ooh batao kitne ka hai
[15/01/2025, 09:22:10] Aarav: abhi pata nahi
[15/01/2025, 09:25:00] System: Messages and calls are end-to-end encrypted.
[16/01/2025, 10:05:12 am] Aarav: good morning!! kal ka plan kya hai?
[16/01/2025, 10:06:45 am] Priya: 10 bje library milte h
[16/01/2025, 19:40:11] Sana: आज की कक्षा बहुत अच्छी थी 😊
[16/01/2025, 19:41:00] Aarav: वाकई? मैं क्लास मिस कर गया था 😅
[16/01/2025, 19:42:00] Rohit: This message was deleted
[16/01/2025, 20:00:00] Aarav: duplicate check
[16/01/2025, 20:00:10] Aarav: duplicate check
"""


@pytest.fixture
def ios_export() -> str:
    return "[16/01/2025 19:40:11] Sana: नमस्ते मित्रों\n[16/01/2025 19:41:00] Aarav: नमस्ते! कैसे हो?\n"


@pytest.fixture
def multiline_export() -> str:
    return """[17/01/2025, 08:00:00] Aarav: first line
second line of same message
third line
[17/01/2025, 08:05:00] Priya: ok
"""


@pytest.fixture
def truncated_export() -> str:
    return "[18/01/2025, 12:00:00] Aarav: hello there\n[18/01/2025, 12:0\nrandom trailing garbage\n"


@pytest.fixture
def parsed(android_export):
    from echotwin.ingestion import WhatsAppParser

    return WhatsAppParser().parse_text(android_export, "conversation-1")


@pytest.fixture
def target_id(parsed) -> str:
    from echotwin.ingestion import pick_participant

    return pick_participant(parsed.messages, "Aarav")
