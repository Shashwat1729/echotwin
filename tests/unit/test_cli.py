"""Unit tests: CLI surface via Typer's CliRunner (offline, no model downloads)."""

from __future__ import annotations

from typer.testing import CliRunner

from echotwin.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "echotwin" in result.output


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("doctor", "demo", "import", "participants", "dataset", "voice", "chat", "serve"):
        assert cmd in result.output


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    # healthy in CI venv; exit code 0 expected (all core deps installed)
    assert result.exit_code == 0
    assert "core deps" in result.output


def test_voice_scripts_hi():
    result = runner.invoke(app, ["voice", "scripts", "--language", "hi"])
    assert result.exit_code == 0
    assert "hi-clean-01" in result.output
    assert "नमस्ते" in result.output


def test_voice_scripts_en():
    result = runner.invoke(app, ["voice", "scripts", "--language", "en", "--category", "numbers"])
    assert result.exit_code == 0
    assert "One, two, three" in result.output


def test_voice_backends_capabilities():
    result = runner.invoke(app, ["voice", "backends"])
    assert result.exit_code == 0
    assert "voxcpm" in result.output
    assert "chatterbox" in result.output


def test_voice_plan_deterministic_output():
    r1 = runner.invoke(app, ["voice", "plan", "--languages", "hi,en"])
    r2 = runner.invoke(app, ["voice", "plan", "--languages", "hi,en"])
    assert r1.exit_code == 0
    assert r1.output == r2.output
    assert "MANDATORY" in r1.output


def test_voice_plan_unsupported_language_skipped():
    # Tamil is NOT in VoxCPM's 30 languages -> must be skipped, never planned
    result = runner.invoke(app, ["voice", "plan", "--languages", "ta,en"])
    assert result.exit_code == 0
    assert "skipped" in result.output.lower()
    assert "unsupported_language" in result.output


def test_privacy_delete_requires_yes():
    result = runner.invoke(app, ["privacy", "delete", "--scope", "all-state"])
    assert result.exit_code == 1
