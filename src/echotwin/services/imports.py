"""Import a WhatsApp export and choose the author to clone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ArtifactNotFoundError, ExportValidationError
from ..ingestion import WhatsAppParser, load_export, participant_stats, pick_participant
from ..ingestion.export_loader import conversation_id_from_name
from ..ingestion.schema import Message
from ..settings import get_settings
from ..storage import db
from ..workspace import get_store

_REPORT_KEYS = (
    "total_lines",
    "parsed_messages",
    "system_messages",
    "attachment_messages",
    "deleted_messages",
    "malformed_lines",
)


@dataclass(frozen=True)
class ImportResult:
    import_id: str
    n_messages: int
    conversations: list[str]
    participants: list[dict]


def participants_of(messages: list[Message]) -> list[dict]:
    return [
        {
            "sender_id": s.sender_id,
            "sender_name": s.sender_name,
            "message_count": s.message_count,
            "first_timestamp": s.first_timestamp.isoformat() if s.first_timestamp else None,
            "last_timestamp": s.last_timestamp.isoformat() if s.last_timestamp else None,
            "mean_length": s.mean_length,
        }
        for s in participant_stats(messages)
    ]


def import_export(path: str | Path, conversation_prefix: str = "conversation") -> ImportResult:
    """Parse a .txt, .zip or directory export and make it the current import.

    A new import invalidates the author, dataset and profile chosen for the old one.
    """
    path = Path(path)
    bundle = load_export(path)
    parser = WhatsAppParser()
    messages: list[Message] = []
    conversations: list[str] = []
    totals = dict.fromkeys(_REPORT_KEYS, 0)
    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"{conversation_prefix}-{i}")
        try:
            result = parser.parse_text(chat.chat_text, convo_id)
        except ExportValidationError:
            continue  # a non-chat .txt inside a zip is skipped, not fatal
        messages.extend(result.messages)
        conversations.append(convo_id)
        for key in _REPORT_KEYS:
            totals[key] += getattr(result.report, key)
    if not messages:
        raise ExportValidationError("No messages parsed from any chat in the export.")

    import_id = db.record_import(str(path), conversation_prefix, len(messages), None, totals)
    save_messages(import_id, messages)

    store = get_store()
    with store.write():
        store.set_pointer("current_import", import_id)
        for stale in ("current_author", "current_dataset", "current_profile"):
            store.set_pointer(stale, None)
    return ImportResult(import_id, len(messages), conversations, participants_of(messages))


def save_messages(import_id: str, messages: list[Message]) -> None:
    out_dir = get_settings().artifacts_dir / "imports" / import_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "messages.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for m in messages:
            fh.write(m.model_dump_json() + "\n")


def load_messages(import_id: str | None = None) -> tuple[list[Message], str]:
    import_id = import_id or get_store().pointer("current_import")
    if not import_id:
        raise ArtifactNotFoundError("No import yet. Import a WhatsApp export first.")
    path = get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"
    if not path.exists():
        raise ArtifactNotFoundError(f"Stored messages for import {import_id} are missing.")
    messages = [
        Message.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return messages, import_id


def select_author(name: str, import_id: str | None = None) -> str:
    """Resolve a participant name or id; sets the current author. Never auto-picks."""
    messages, import_id = load_messages(import_id)
    sender_id = pick_participant(messages, name)
    if sender_id is None:
        raise ArtifactNotFoundError(f"Participant {name!r} not found in this import.")
    store = get_store()
    with store.write():
        db.update_import_target(import_id, sender_id)
        store.set_pointer("current_import", import_id)
        store.set_pointer("current_author", sender_id)
    return sender_id
