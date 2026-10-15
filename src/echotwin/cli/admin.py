"""Doctor, demo, chat, UI, API, artifacts and privacy commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from .. import __version__
from ..exceptions import EchoTwinError
from ..storage import db
from ._common import (
    _fail,
    app,
    artifacts_app,
    console,
    err_console,
    privacy_app,
)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Environment diagnostics: deps, storage, backends, optional hardware."""
    from ..check_env import doctor_report

    report = doctor_report()
    for line in report["lines"]:
        console.print(line)
    if not report["healthy"]:
        raise typer.Exit(1)


@app.command()
def demo() -> None:
    """Run the synthetic demo end-to-end (no private data needed)."""
    from ..demo import run_demo

    run_demo(console)


@app.command("version")
def version() -> None:
    console.print(f"echotwin {__version__}")


# ---------------------------------------------------------------------------
# chat / serve / artifacts / privacy
# ---------------------------------------------------------------------------
@app.command()
def chat(
    engine: str = typer.Option(
        "ngram", help="engine id: ngram, hf:<model>, openai:<model> (see Settings in the UI)"
    ),
    method: str = typer.Option("style_fewshot", help="prompt method"),
    model: str = typer.Option("", help="shortcut for --engine hf:<model>"),
) -> None:
    """Interactive chat with your clone, using the prepared dataset and profile."""
    from ..services import chat as chat_service

    engine_id = f"hf:{model}" if model else engine
    session = chat_service.ChatSession()
    console.print(f"[dim]chat with {engine_id} ({method}); type /exit to quit[/dim]")
    while True:
        try:
            line = console.input("[bold blue]you>[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() in {"/exit", "/quit"}:
            break
        if not line.strip():
            continue
        out = chat_service.reply(session, line, engine_id, method)
        if out.error:
            err_console.print(f"[red]error:[/red] {out.error}")
            continue
        console.print(f"[bold green]twin>[/bold green] {out.text}")


@app.command()
def ui(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    share: bool = typer.Option(False, help="Allow a public Gradio share link (off by default)"),
) -> None:
    """Launch the local Gradio studio (localhost by default; never shared unless asked)."""
    from ..settings import get_settings

    s = get_settings()
    if share:
        err_console.print(
            "[yellow]Warning: --share exposes your studio (and potentially private "
            "recordings) to the public internet.[/yellow]"
        )
    from ..ui.app import build_app

    s.ensure_dirs()
    build_app().launch(
        server_name=host or s.ui_host,
        server_port=port or s.ui_port,
        inbrowser=False,
        share=share,
    )


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Run the local FastAPI service (localhost by default)."""
    import uvicorn

    from ..settings import get_settings

    s = get_settings()
    uvicorn.run(
        "echotwin.api.app:app",
        host=host or s.api_host,
        port=port or s.api_port,
        log_level="info",
    )


@artifacts_app.command("list")
def artifacts_list(kind: str = typer.Option(None)) -> None:
    table = Table(title="Artifacts")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Path")
    table.add_column("Created")
    for a in db.list_artifacts(kind):
        table.add_row(a["artifact_id"], a["kind"], a["path"], a["created_at"])
    console.print(table)


@artifacts_app.command("delete")
def artifacts_delete(artifact_id: str) -> None:
    try:
        result = db.delete_artifact(artifact_id)
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    console.print(json.dumps(result, indent=1))


@privacy_app.command("export")
def privacy_export(out: Path = typer.Option(Path("echotwin-export"))) -> None:
    """Export all local EchoTwin data (profiles, artifacts, consents) to a folder."""
    from ..privacy_exporter import export_all

    export_all(out, console)


@privacy_app.command("delete")
def privacy_delete(
    scope: str = typer.Option(
        ..., help="artifacts:voice | artifacts:datasets | profile:<voice_id> | all-state"
    ),
    yes: bool = typer.Option(False, "--yes", help="confirm deletion"),
) -> None:
    """Delete data (explicit, irreversible). Consent stays recorded."""
    from ..privacy_exporter import delete_scope

    if not yes:
        _fail("Refusing to delete without --yes")
    delete_scope(scope, console)


def main() -> None:
    try:
        app()
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
