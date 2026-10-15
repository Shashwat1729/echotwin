"""Import, participants, author, dataset and profile commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from .. import __version__
from ..dataset.builder import DatasetConfig, build_examples
from ..dataset.builder import save_jsonl as save_examples_jsonl
from ..dataset.splits import SplitConfig, make_manifest, split_dataset
from ..exceptions import EchoTwinError, ExportValidationError
from ..ingestion import (
    WhatsAppParser,
    load_export,
    participant_stats,
    pick_participant,
)
from ..ingestion.schema import Message
from ..profiling.style_profile import build_style_profile, profile_prompt_fragment, save_profile
from ..storage import db, state
from ._common import (
    _dump_messages,
    _fail,
    _load_messages_for_state,
    _require_target,
    app,
    console,
    dataset_app,
    import_app,
)


# ---------------------------------------------------------------------------
# import / participants
# ---------------------------------------------------------------------------
@import_app.command("whatsapp")
def import_whatsapp(
    path: Path = typer.Argument(
        ..., help="WhatsApp export: _chat.txt, .zip (may contain many chats), or a directory"
    ),
    conversation_prefix: str = typer.Option(
        "chat", "--conversation-id", help="Fallback conversation label when the filename gives none"
    ),
) -> None:
    """Import a WhatsApp TXT or ZIP export (every chat in the file is imported)."""
    try:
        bundle = load_export(path)
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    parser = WhatsAppParser()
    all_messages: list[Message] = []
    total_report: dict[str, Any] = {
        "total_lines": 0,
        "parsed_messages": 0,
        "continuation_lines": 0,
        "orphan_lines": 0,
        "system_messages": 0,
        "attachment_messages": 0,
        "deleted_messages": 0,
        "call_events": 0,
        "unparsed_timestamps": 0,
        "malformed_lines": 0,
        "duplicates": 0,
        "detected_formats": [],
        "conversations": [],
    }
    from ..ingestion.export_loader import conversation_id_from_name

    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"{conversation_prefix}-{i}")
        try:
            result = parser.parse_text(chat.chat_text, convo_id)
        except ExportValidationError:
            continue  # a non-chat txt inside a zip is skipped, not fatal
        all_messages.extend(result.messages)
        rep = result.report
        total_report["total_lines"] += rep.total_lines
        total_report["parsed_messages"] += rep.parsed_messages
        total_report["continuation_lines"] += rep.continuation_lines
        total_report["orphan_lines"] += rep.orphan_lines
        total_report["system_messages"] += rep.system_messages
        total_report["attachment_messages"] += rep.attachment_messages
        total_report["deleted_messages"] += rep.deleted_messages
        total_report["call_events"] += rep.call_events
        total_report["unparsed_timestamps"] += rep.unparsed_timestamps
        total_report["malformed_lines"] += rep.malformed_lines
        total_report["duplicates"] += rep.duplicates
        for f in rep.detected_formats:
            if f not in total_report["detected_formats"]:
                total_report["detected_formats"].append(f)
        total_report["conversations"].append(convo_id)

    if not all_messages:
        _fail("No messages parsed from any chat in the export.", ExportValidationError.exit_code)

    import_id = db.record_import(
        str(path), conversation_prefix, len(all_messages), None, total_report
    )
    state.set("last_import_id", import_id)
    from ..settings import get_settings

    out_dir = get_settings().artifacts_dir / "imports" / import_id
    _dump_messages(all_messages, out_dir / "messages.jsonl")
    console.print(
        f"[green]Imported[/green] {len(all_messages)} messages from "
        f"{len(bundle.chats)} conversation(s): {', '.join(total_report['conversations'])}"
    )
    console.print(f"formats detected: {total_report['detected_formats']}")
    console.print(
        f"system={total_report['system_messages']} attachments={total_report['attachment_messages']} "
        f"deleted={total_report['deleted_messages']} malformed={total_report['malformed_lines']}"
    )
    console.print("Next: [bold]echotwin participants[/bold]")


@app.command()
def participants() -> None:
    """List participants with statistics (target author is never auto-selected)."""
    messages, _import_id, _target = _load_messages_for_state()
    table = Table(title="Participants")
    table.add_column("Name")
    table.add_column("Messages")
    table.add_column("Range")
    table.add_column("Mean length")
    table.add_column("Languages")
    for st in participant_stats(messages):
        langs = ", ".join(f"{k}:{v}" for k, v in sorted(st.language_distribution.items()))
        table.add_row(
            st.sender_name,
            str(st.message_count),
            f"{st.first_timestamp} .. {st.last_timestamp}",
            f"{st.mean_length:.0f}",
            langs,
        )
    console.print(table)


@app.command("select-author")
def select_author(name: str = typer.Argument(..., help="Exact participant name or id")) -> None:
    """Select the target author (the person whose style you will learn)."""
    messages, _import_id, _target = _load_messages_for_state()
    sid = pick_participant(messages, name)
    if sid is None:
        _fail(f"Participant '{name}' not found. Run 'echotwin participants'.")
    latest = db.latest_import()
    if latest is None:
        _fail("Import record missing; re-run the import command.")
    db.update_import_target(latest["import_id"], sid)
    state.set("target_sender_id", sid)
    console.print(f"[green]Target author set[/green] to {name!r} (id={sid})")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@dataset_app.command("prepare")
def dataset_prepare(
    context_messages: int = typer.Option(4),
    max_context_chars: int = typer.Option(1200),
    min_target_chars: int = typer.Option(2),
    max_target_chars: int = typer.Option(800),
    no_dedup: bool = typer.Option(False, "--no-dedup"),
) -> None:
    """Build the contextual training dataset and leakage-resistant splits."""
    from ..settings import get_settings

    messages, import_id, target = _require_target()
    cfg = DatasetConfig(
        context_messages=context_messages,
        max_context_chars=max_context_chars,
        min_target_chars=min_target_chars,
        max_target_chars=max_target_chars,
        exact_dedup=not no_dedup,
    )
    built = build_examples(messages, target, cfg)
    split = split_dataset(built, SplitConfig())
    ds_id = f"dataset-{import_id}"
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    save_examples_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_examples_jsonl(split.train, ds_dir / "train.jsonl")
    save_examples_jsonl(split.val, ds_dir / "val.jsonl")
    save_examples_jsonl(split.test, ds_dir / "test.jsonl")
    manifest = make_manifest(
        dataset_id=ds_id,
        dataset_path=ds_dir / "examples.jsonl",
        split=split,
        built=built,
        target_sender_id=target,
        split_cfg=SplitConfig(),
        echotwin_version=__version__,
    )
    (ds_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    state.set("last_dataset_id", ds_id)
    console.print(
        f"[green]Dataset built[/green]: {manifest.counts['n_examples']} examples "
        f"from {manifest.counts['conversations']} conversation(s)"
    )
    console.print(
        f"splits: train={len(split.train)} val={len(split.val)} test={len(split.test)} "
        f"(leakage check passed={split.report.leakage_check_passed})"
    )
    for w in split.report.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


@dataset_app.command("stats")
def dataset_stats() -> None:
    """Show dataset statistics."""
    from ..settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet. Run 'echotwin dataset prepare'.")
    manifest_path = get_settings().artifacts_dir / "datasets" / ds_id / "manifest.json"
    if not manifest_path.exists():
        _fail("Dataset manifest missing; rebuild the dataset.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.print(json.dumps(manifest, indent=2, ensure_ascii=False))


@dataset_app.command("validate")
def dataset_validate() -> None:
    """Validate the current dataset (JSONL parse + leakage + duplicates)."""
    from ..dataset.builder import load_jsonl
    from ..settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    problems: list[str] = []
    try:
        train = load_jsonl(ds_dir / "train.jsonl")
        val = load_jsonl(ds_dir / "val.jsonl")
        test = load_jsonl(ds_dir / "test.jsonl")
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    train_c = {ex.metadata.conversation_id for ex in train}
    val_c = {ex.metadata.conversation_id for ex in val}
    test_c = {ex.metadata.conversation_id for ex in test}
    if train_c & test_c or train_c & val_c or val_c & test_c:
        problems.append("Leakage: conversation ids overlap across splits")
    keys = [ex.key() for ex in train]
    if len(keys) != len(set(keys)):
        problems.append("Duplicate examples in train split")
    for ex in train:
        if not ex.messages or ex.messages[-1].role != "assistant":
            problems.append(
                f"Example ends with non-assistant turn: {ex.metadata.source_message_id}"
            )
            break
    if problems:
        for p in problems:
            console.print(f"[red]FAIL[/red] {p}")
        raise typer.Exit(1)
    console.print("[green]PASS[/green] dataset validation (no leakage, no dupes, roles valid)")


# ---------------------------------------------------------------------------
# profile / evaluation / training stubs wiring
# ---------------------------------------------------------------------------
@app.command("profile")
def profile_build() -> None:
    """Build the interpretable style profile."""
    from ..settings import get_settings

    messages, _import_id, target = _require_target()
    profile = build_style_profile(messages, target)
    out = get_settings().artifacts_dir / "profiles" / f"profile-{target[:8]}.json"
    save_profile(profile, out)
    state.set("last_profile_id", str(out))
    console.print(f"[green]Style profile saved[/green] to {out}")
    console.print(profile_prompt_fragment(profile))
