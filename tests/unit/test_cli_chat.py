"""`echotwin chat` runs on the real dataset through the chat service, never on mock text."""

from __future__ import annotations

from rich.console import Console
from typer.testing import CliRunner

from echotwin.cli import app
from echotwin.demo import run_demo
from echotwin.workspace import reset_store

runner = CliRunner()


def test_chat_without_dataset_reports_error_not_fake_text():
    reset_store()
    result = runner.invoke(app, ["chat"], input="hello\n/exit\n")
    assert "twin>" not in result.output
    assert result.exit_code != 0 or "dataset" in result.output.lower()


def test_chat_replies_from_demo_dataset():
    reset_store()
    run_demo(Console(quiet=True))
    result = runner.invoke(
        app, ["chat", "--engine", "ngram", "--method", "plain"], input="kal milte hain?\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "twin>" in result.output
