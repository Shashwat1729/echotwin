"""Shared Typer apps, consoles and helpers for the CLI command modules."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from ..ingestion.schema import Message
from ..privacy.redact import RedactionConfig
from ..storage import db, state

app = typer.Typer(
    name="echotwin",
    help="EchoTwin — local-first writing-style learning and authorized voice cloning.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

voice_app = typer.Typer(help="Voice cloning: backends, recording, analysis, synthesis.")
dataset_app = typer.Typer(help="Dataset preparation and validation.")
train_app = typer.Typer(help="Training commands (SFT/LoRA/DPO).")
import_app = typer.Typer(help="Data import commands.")
evaluate_app = typer.Typer(help="Evaluation commands.")
prefs_app = typer.Typer(help="Preference annotation.")
artifacts_app = typer.Typer(help="Artifact management.")
privacy_app = typer.Typer(help="Privacy export and deletion.")

app.add_typer(voice_app, name="voice")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(import_app, name="import")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(prefs_app, name="preferences")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(privacy_app, name="privacy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fail(msg: str, code: int = 1) -> NoReturn:
    err_console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


def _load_messages_for_state() -> tuple[list[Message], str, str | None]:
    import_id = state.get("last_import_id")
    if not import_id:
        _fail("No import yet. Run: echotwin import whatsapp ./export.zip")
    imp = db.latest_import()
    if imp is None or imp["import_id"] != import_id:
        _fail("Import record missing; re-run the import command.")
    from ..settings import get_settings

    messages_path = get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"
    if not messages_path.exists():
        _fail(f"Stored messages for import {import_id} are missing; re-run the import command.")
    messages = _read_messages(messages_path)
    target = imp["target_sender_id"] or state.get("target_sender_id")
    return messages, import_id, target


def _require_target() -> tuple[list[Message], str, str]:
    messages, import_id, target = _load_messages_for_state()
    if not target:
        _fail('No target author selected yet. Run: echotwin select-author "Your Name"')
    return messages, import_id, target


def _redaction() -> RedactionConfig:
    from ..settings import get_settings

    s = get_settings()
    return RedactionConfig(enabled=s.redact_pii)


def _dump_messages(messages: list[Message], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for m in messages:
            fh.write(m.model_dump_json() + "\n")


def _read_messages(path: Path) -> list[Message]:
    return [
        Message.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
