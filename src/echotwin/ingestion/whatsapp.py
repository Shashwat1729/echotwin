"""Robust WhatsApp export parser.

Handles the real-world messiness of WhatsApp exports:

* Android format ``[15/01/2025, 14:23:45] Sender: text``
* iOS format       ``[15/01/2025 14:23:45] Sender: text``
* Some exports drop the brackets: ``15/01/2025, 14:23 - Sender: text``
* 24h and 12h (am/pm, with or without dots/space) timestamps
* ``.`` / ``/`` / ``-`` date separators, DD/MM vs MM/DD ambiguity (configurable)
* Multiline messages (continuation lines have no header)
* Unicode / emoji / non-Latin sender names (invisible bidi marks are stripped from headers)
* System messages (group events, encryption notices, deleted messages, call notices)
* Attachment placeholders (``<Media omitted>``, ``image omitted``, ``<attached: ...>``)
* Malformed / truncated files are reported, never fatal

The parser preserves original text exactly: genuine typos, capitalization, emoji and
grammar are *data* for style modeling and are never normalized away.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

from ..exceptions import ExportValidationError
from ..ingestion.schema import (
    AttachmentInfo,
    Language,
    Message,
    MessageType,
    QualityFlag,
)

# ---------------------------------------------------------------------------
# Invisible / directional marks that appear in real exports
# ---------------------------------------------------------------------------
_BIDI_MARKS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------
# A header line starts (optionally after bidi marks) with a timestamp, optional
# brackets, optional separator dash, then "Sender: text" or a system line.
_HEADER_RE = re.compile(
    r"""
    ^\s*\[?\s*
    (?P<date>\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})
    ,?\s+
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?\s?[Mm]\.?)?)
    \s*\]?
    \s*[-\u2013]?\s*
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)

_SENDER_SPLIT_RE = re.compile(r"^(?P<sender>[^:\n]{1,120}?):\s?(?P<text>.*)$")

# A line that *starts* like a header but fails full parsing is a truncated or
# corrupted header, not a multiline continuation — report it instead of
# silently appending it to the previous message.
_HEADER_ISH_RE = re.compile(r"^\s*\[?\s*\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4},?\s+\d{1,2}:\d{1,2}")

# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
_DATE_FORMATS_DAYFIRST = ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%d.%m.%y"]
_DATE_FORMATS_MONTHFIRST = ["%m/%d/%Y", "%m.%d.%Y", "%m/%d/%y"]
_DATE_FORMATS_ISO = ["%Y-%m-%d", "%Y/%m/%d"]


def _parse_am_pm(tok: str) -> tuple[str, str | None]:
    """Split a time token into 'H:M(:S)' and an optional meridiem marker."""
    m = re.search(r"([AaPp])\.?\s*([Mm])\.?", tok)
    if not m:
        return tok.strip(), None
    meridiem = m.group(1).lower()
    base = tok[: m.start()].strip()
    return base, meridiem


def parse_timestamp(date_tok: str, time_tok: str, *, dayfirst: bool = True) -> datetime | None:
    """Parse WhatsApp date+time tokens. Returns None when ambiguous/unparseable."""
    base_time, meridiem = _parse_am_pm(time_tok)
    if dayfirst:
        date_fmts = _DATE_FORMATS_DAYFIRST + _DATE_FORMATS_MONTHFIRST + _DATE_FORMATS_ISO
    else:
        date_fmts = _DATE_FORMATS_MONTHFIRST + _DATE_FORMATS_DAYFIRST + _DATE_FORMATS_ISO

    if meridiem:
        # Explicit 12-hour clock: parse with %I, then normalize (12am -> 0, 12pm -> 12).
        for dfmt in date_fmts:
            for tfmt in ("%I:%M:%S", "%I:%M"):
                try:
                    dt = datetime.strptime(f"{date_tok} {base_time}", f"{dfmt} {tfmt}")
                except ValueError:
                    continue
                hour = dt.hour % 12
                if meridiem == "p":
                    hour += 12
                return dt.replace(hour=hour)
        return None

    for fmt in [f"{d} {t}" for d in date_fmts for t in ("%H:%M:%S", "%H:%M")]:
        try:
            return datetime.strptime(f"{date_tok} {base_time}", fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# System messages / attachments / edited / deleted
# ---------------------------------------------------------------------------
# WhatsApp prints system notices with an optional actor prefix:
#   "Rohit created group "X"" / "+91 98765 43210 turned on disappearing messages"
# So a notice word may follow an actor. `_ACTOR` matches a phone-number handle or
# a short capitalized name.
_ACTOR = r"(?:\+?[\d][\d\s().-]{5,}|[\w][\w .'-]{0,40}?)\s+"
_SYSTEM_PATTERNS = [
    r"messages?( and calls)? (are|is) end-to-end encrypted",
    rf"{_ACTOR}created (this )?group",
    # "Rohit added Priya and Sana", "X added you", "Y added A to the group".
    # The negative lookahead stops user text like "I added you to my contact
    # list" from being read as a notice even when no sender was attributed.
    rf"{_ACTOR}added(?!\s+to\s+(?:my|the|your|his|her)\b).*",
    rf"{_ACTOR}removed\s+\S+",
    rf"{_ACTOR}left\b",
    rf"{_ACTOR}changed (the|their|this group'?s?) .*",
    rf"{_ACTOR}turned (on|off) .*",
    rf"{_ACTOR}joined using (this group'?s? )?invite link",
    rf"{_ACTOR}security code (with .+ )?changed",
    r"this message was deleted",
    r"you deleted this message",
    r"missed (voice|video) call",
    r"started a call",
    rf"{_ACTOR}pinned a message",
    rf"{_ACTOR}changed (your|the) phone number",
    r"this chat is with a (business|official) account",
]
# Anchored: a notice must START the line (optionally after an actor). The previous
# unanchored search swallowed real user text like "I removed the old files from
# my phone", silently deleting it from the training set.
_SYSTEM_RE = re.compile("|".join(f"(?:{p})" for p in _SYSTEM_PATTERNS), re.IGNORECASE)
# A prefix that is only digits/phone punctuation is an account handle, never a
# human participant name.
_PHONE_ACTOR_RE = re.compile(r"^\+?[\d\s().-]{6,}$")
# Some export tools (and older Android builds) print a literal "System" sender
# on notice lines. No human participant is named exactly "System", so this is a
# safe actor rather than a reason to reclassify a person's message.
_SYSTEM_ACTOR_NAMES = {"system", "whatsapp", "chat"}

_DELETED_RE = re.compile(r"^(this message was deleted|you deleted this message)", re.IGNORECASE)
_CALL_RE = re.compile(r"^(missed (voice|video) call|started a call)", re.IGNORECASE)

_ATTACHMENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^<Media omitted>$", re.I), "other"),
    (re.compile(r"^image omitted$", re.I), "image"),
    (re.compile(r"^audio omitted$", re.I), "audio"),
    (re.compile(r"^video omitted$", re.I), "video"),
    (re.compile(r"^sticker omitted$", re.I), "sticker"),
    (re.compile(r"^GIF omitted$", re.I), "gif"),
    (re.compile(r"^document omitted$", re.I), "document"),
    (re.compile(r"^contact card omitted$", re.I), "contact"),
    (re.compile(r"^<attached:\s*(?P<fn>[^>]+)>$", re.I), "other"),
    (re.compile(r"^\[attachment\]\s*(?P<fn>.*)$", re.I), "other"),
]
_EDITED_RE = re.compile(r"\s*<\s*This message was edited\s*>\s*$", re.I)


# ---------------------------------------------------------------------------
# Config + result models
# ---------------------------------------------------------------------------
class ParseConfig(BaseModel):
    dayfirst: bool = True
    mark_attachment_only: bool = True
    detect_duplicates: bool = True
    duplicate_window_seconds: int = 30


class ParseErrorRecord(BaseModel):
    line_number: int
    reason: str
    excerpt: str = Field(description="Short, possibly redacted excerpt of the raw line")


class ConversationReport(BaseModel):
    total_lines: int = 0
    parsed_messages: int = 0
    continuation_lines: int = 0
    orphan_lines: int = 0
    system_messages: int = 0
    attachment_messages: int = 0
    deleted_messages: int = 0
    call_events: int = 0
    unparsed_timestamps: int = 0
    malformed_lines: int = 0
    duplicates: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    detected_formats: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    model_config = {"arbitrary_types_allowed": False}
    conversation_id: str
    messages: list[Message]
    errors: list[ParseErrorRecord]
    report: ConversationReport


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _strip_bidi(text: str) -> str:
    return _BIDI_MARKS.sub("", text)


def _stable_id(conversation_id: str, line_number: str | int, sender: str | None, text: str) -> str:
    raw = f"{conversation_id}|{line_number}|{sender}|{text[:80]}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _is_system_notice(body: str, sender_name: str | None) -> bool:
    """True only for genuine WhatsApp system notices.

    Rules, in order:

    1. The anchored system pattern must match at the START of the text, because
       a real participant may write "I removed the old files from my phone".
    2. If the line was attributed to a human sender (``Alice: ...``), it is user
       text and can never be a system notice.
    3. If WhatsApp printed a phone-number account handle as the actor
       (``+91 98765 43210 turned on disappearing messages``), it IS a notice.
    """
    if not _SYSTEM_RE.match(body):
        return False
    if sender_name is None:
        return True
    if sender_name.strip().lower() in _SYSTEM_ACTOR_NAMES:
        return True
    return bool(_PHONE_ACTOR_RE.match(sender_name))


def _detect_attachment(text: str) -> AttachmentInfo | None:
    for rx, kind in _ATTACHMENT_RULES:
        m = rx.match(text.strip())
        if m:
            groups = m.groupdict()
            fname = groups.get("fn") or None
            return AttachmentInfo(kind=kind, filename=fname, placeholder_text=text.strip())
    return None


def _dedupe_key(msg_text: str) -> str:
    return re.sub(r"\s+", " ", msg_text.strip().lower())


@dataclass
class _Acc:
    last: Message | None = None
    text_parts: list[str] = field(default_factory=list)
    seen_recent: dict[str, list[tuple[datetime, str]]] = field(default_factory=dict)


class WhatsAppParser:
    """Stateful line-based parser for a single conversation file."""

    def __init__(self, config: ParseConfig | None = None) -> None:
        self.config = config or ParseConfig()

    # -- public API ---------------------------------------------------------
    def parse_text(self, text: str, conversation_id: str = "conversation-1") -> ParseResult:
        lines = text.splitlines()
        messages: list[Message] = []
        errors: list[ParseErrorRecord] = []
        report = ConversationReport(total_lines=len(lines))
        acc = _Acc()

        for i, raw_line in enumerate(lines, start=1):
            line = _strip_bidi(raw_line)
            if not line.strip():
                continue
            header = _HEADER_RE.match(line)
            if header:
                self._flush(acc, messages, report)
                msg = self._build_message(
                    line=line,
                    line_number=i,
                    conversation_id=conversation_id,
                    date_tok=header.group("date"),
                    time_tok=header.group("time"),
                    rest=header.group("rest"),
                    report=report,
                )
                if msg is None:
                    errors.append(
                        ParseErrorRecord(
                            line_number=i,
                            reason="malformed_header",
                            excerpt=raw_line[:80],
                        )
                    )
                    report.malformed_lines += 1
                    acc = _Acc()
                    continue
                acc = _Acc(last=msg)
                messages.append(msg)
                report.parsed_messages += 1
            else:
                # Continuation line: append to the previous message, or orphan.
                if acc.last is not None and not _HEADER_ISH_RE.match(line):
                    acc.text_parts.append(line)
                    report.continuation_lines += 1
                    if QualityFlag.MULTILINE not in acc.last.quality_flags:
                        acc.last.quality_flags.append(QualityFlag.MULTILINE)
                else:
                    errors.append(
                        ParseErrorRecord(
                            line_number=i,
                            reason="truncated_or_orphan_header",
                            excerpt=raw_line[:80],
                        )
                    )
                    if acc.last is None:
                        report.orphan_lines += 1
                    report.malformed_lines += 1
                    # A broken header interrupts the message in progress. Flush it
                    # and clear the buffer, otherwise the next continuation line
                    # would be appended to the *previous* sender's message.
                    self._flush(acc, messages, report)
                    acc = _Acc()

        self._flush(acc, messages, report)
        self._postprocess(messages, report)
        timestamps = [m.timestamp for m in messages if m.timestamp is not None]
        if timestamps:
            report.first_timestamp = min(timestamps)
            report.last_timestamp = max(timestamps)
        if not messages:
            raise ExportValidationError(
                "No messages could be parsed from the export. "
                f"Conversation '{conversation_id}' had {len(lines)} lines and "
                f"{report.orphan_lines} orphan lines. Is this a WhatsApp chat export?"
            )
        return ParseResult(
            conversation_id=conversation_id,
            messages=messages,
            errors=errors,
            report=report,
        )

    # -- internals ----------------------------------------------------------
    def _flush(self, acc: _Acc, messages: list[Message], report: ConversationReport) -> None:
        """Merge buffered continuation lines into the buffered message."""
        if acc.last is None or not acc.text_parts:
            acc.text_parts = []
            return
        extra = "\n".join(acc.text_parts)
        if acc.last.text:
            acc.last.text = f"{acc.last.text}\n{extra}"
        else:
            acc.last.text = extra
        acc.text_parts = []

    def _build_message(
        self,
        *,
        line: str,
        line_number: int,
        conversation_id: str,
        date_tok: str,
        time_tok: str,
        rest: str,
        report: ConversationReport,
    ) -> Message | None:
        ts = parse_timestamp(date_tok, time_tok, dayfirst=self.config.dayfirst)
        rest = rest.strip()
        if not rest:
            # Header with empty body — malformed for WhatsApp exports
            return None
        flags: list[QualityFlag] = []

        # Sender split. A missing colon means the body is a system/event notice or
        # a corrupted line; system detection below resolves which.
        m = _SENDER_SPLIT_RE.match(rest)
        sender_name: str | None = None
        body = rest
        if m:
            sender_name = m.group("sender").strip()
            body = m.group("text").strip()

        edited = bool(_EDITED_RE.search(body))
        if edited:
            body = _EDITED_RE.sub("", body).strip()
            flags.append(QualityFlag.EDITED_MESSAGE)

        deleted = _DELETED_RE.match(body)
        call = _CALL_RE.match(body)

        # Classification precedence: deleted > call > attachment > system > text.
        msg_type = MessageType.TEXT
        attachment: AttachmentInfo | None = None
        is_system = False

        if deleted:
            msg_type = MessageType.DELETED
            flags.append(QualityFlag.DELETED_MESSAGE)
            report.deleted_messages += 1
        elif call:
            msg_type = MessageType.CALL
            report.call_events += 1
        else:
            attachment = _detect_attachment(body) if self.config.mark_attachment_only else None
            if attachment is not None:
                msg_type = MessageType.ATTACHMENT
                flags.append(QualityFlag.ATTACHMENT_PLACEHOLDER)
                report.attachment_messages += 1
            elif _is_system_notice(body, sender_name):
                # System/event notice ("Messages … end-to-end encrypted",
                # "created group", "added …"). A message that carries a real
                # human sender is NEVER reclassified as a system event: real
                # users write "I removed the old files" all the time.
                is_system = True
                msg_type = MessageType.SYSTEM
                flags.append(QualityFlag.SYSTEM_MESSAGE)
                report.system_messages += 1
                sender_name = None

        if ts is None:
            flags.append(QualityFlag.UNPARSED_TIMESTAMP)
            report.unparsed_timestamps += 1

        if not body.strip():
            flags.append(QualityFlag.EMPTY_TEXT)

        # Format detection: each property is independent. The previous elif made every
        # bracketed line after the first also report "unbracketed".
        _bracketed = line.lstrip().startswith("[")
        if _bracketed and "bracketed" not in report.detected_formats:
            report.detected_formats.append("bracketed")
        elif not _bracketed and "unbracketed" not in report.detected_formats:
            report.detected_formats.append("unbracketed")
        if re.search(r"[AaPp]\.?\s*[Mm]\.?", time_tok):
            if "12h" not in report.detected_formats:
                report.detected_formats.append("12h")
        elif "24h" not in report.detected_formats:
            report.detected_formats.append("24h")

        sender_id = _stable_id("sender", "name", sender_name, "") if sender_name else None
        msg = Message(
            message_id=_stable_id(conversation_id, str(line_number), sender_name, body),
            conversation_id=conversation_id,
            timestamp=ts,
            sender_id=sender_id,
            sender_name=sender_name,
            text=body,
            message_type=msg_type,
            attachment=attachment,
            is_system_message=is_system,
            language=Language.UNKNOWN,
            quality_flags=flags,
            line_number=line_number,
        )
        return msg

    def _postprocess(self, messages: list[Message], report: ConversationReport) -> None:
        if not self.config.detect_duplicates:
            return
        recent: dict[str, list] = {}
        window = self.config.duplicate_window_seconds
        for msg in messages:
            key = f"{msg.sender_id}:{_dedupe_key(msg.text)}"
            ts = msg.timestamp
            bucket = recent.setdefault(key, [])
            if ts is not None:
                dup = any(
                    old is not None and abs((ts - old).total_seconds()) <= window for old in bucket
                )
            else:
                dup = len(bucket) > 0
            if dup and msg.text.strip():
                msg.quality_flags.append(QualityFlag.DUPLICATE)
                report.duplicates += 1
            bucket.append(ts)
