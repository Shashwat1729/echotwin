"""Integration: full CLI pipeline on synthetic data (import -> dataset -> profile -> baseline)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from echotwin.cli import app

runner = CliRunner()
EXPORT = Path(__file__).resolve().parents[2] / "examples" / "synthetic_whatsapp.txt"


def write_two_convo(tmp_path) -> Path:
    """A directory with THREE chat files = three conversations (like a WhatsApp export bundle)."""
    d = tmp_path / "chats"
    d.mkdir()
    for name, body in [
        ("WhatsApp Chat with Priya", CHAT1),
        ("WhatsApp Chat with Sana", CHAT2),
        ("WhatsApp Chat with Rohit", CHAT3),
    ]:
        (d / name).mkdir()
        (d / name / "_chat.txt").write_text(body, encoding="utf-8")
    return d


CHAT1 = """[01/01/2025, 08:00:00] Priya: hello everyone
[01/01/2025, 08:01:00] Aarav: haan bro kaise ho
[01/01/2025, 08:02:00] Priya: badhiya, tu suna
[01/01/2025, 08:03:00] Aarav: haan suna, kal milte h 6 bje library
[01/01/2025, 08:04:00] Priya: theek hai done
"""

CHAT2 = """[02/01/2025, 09:00:00] Sana: good morning
[02/01/2025, 09:01:00] Aarav: morning! aaj thoda busy hoon
[02/01/2025, 09:02:00] Sana: koi baat nahi, baad me baat karte hain
[02/01/2025, 09:03:00] Aarav: haan zaroor, sham ko call karunga
[02/01/2025, 09:04:00] Sana: ok 👍
"""

CHAT3 = """[03/01/2025, 10:00:00] Rohit: bhai match dekh liya
[03/01/2025, 10:01:00] Aarav: nhi yaar, kal dekhta hoon. bohot kaam tha
[03/01/2025, 10:02:00] Rohit: kya yaar, tu miss kar gaya
[03/01/2025, 10:03:00] Aarav: arre pakka kal dekhunga, tension mat lo
[03/01/2025, 10:04:00] Rohit: theek hai
"""


def test_full_data_pipeline(tmp_path):
    export = write_two_convo(tmp_path)
    r = runner.invoke(app, ["import", "whatsapp", str(export), "--conversation-id", "int"])
    assert r.exit_code == 0, r.output
    assert "Imported" in r.output
    assert "3 conversation" in r.output

    r = runner.invoke(app, ["participants"])
    assert r.exit_code == 0
    assert "Aarav" in r.output

    r = runner.invoke(app, ["select-author", "Aarav"])
    assert r.exit_code == 0

    r = runner.invoke(app, ["dataset", "prepare"])
    assert r.exit_code == 0, r.output
    assert "Dataset built" in r.output

    r = runner.invoke(app, ["dataset", "validate"])
    assert r.exit_code == 0
    assert "PASS" in r.output

    r = runner.invoke(app, ["dataset", "stats"])
    assert r.exit_code == 0

    r = runner.invoke(app, ["profile"])
    assert r.exit_code == 0
    assert "Typical length" in r.output

    r = runner.invoke(app, ["evaluate", "baseline", "--method", "plain"])
    assert r.exit_code == 0, r.output


def test_participants_never_auto_selects_most_active():
    """The group export's most active member must NOT be silently chosen."""
    r = runner.invoke(app, ["import", "whatsapp", str(EXPORT), "--conversation-id", "int2"])
    assert r.exit_code == 0
    from echotwin.storage import state

    assert state.get("target_sender_id") is None
    r = runner.invoke(app, ["participants"])
    # Aarav is the most active; list shows him first but selection stays empty
    assert "Aarav" in r.output


def test_voice_import_and_analyze(tmp_path):
    import wave

    import numpy as np

    sr = 16000
    t = np.arange(sr * 4) / sr
    env = np.clip(np.sin(2 * np.pi * 0.5 * t), 0, 1)
    tone = 0.3 * np.sin(2 * np.pi * 180 * t) * env
    pcm = (tone * 32767).astype("<i2")
    rec = tmp_path / "hi-test-01.wav"
    with wave.open(str(rec), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    r = runner.invoke(
        app, ["voice", "import", str(tmp_path), "--language", "hi", "--backend", "voxcpm"]
    )
    assert r.exit_code == 0, r.output
    assert "Voice profile saved" in r.output

    r = runner.invoke(app, ["voice", "analyze"])
    assert r.exit_code == 0

    r = runner.invoke(app, ["voice", "coverage"])
    assert r.exit_code == 0

    r = runner.invoke(app, ["voice", "recommend"])
    assert r.exit_code == 0
    assert "Recommended" in r.output


def test_privacy_export(tmp_path):
    out = tmp_path / "exp"
    r = runner.invoke(app, ["privacy", "export", "--out", str(out)])
    assert r.exit_code == 0
    assert (out / "echotwin-state.json").exists()
