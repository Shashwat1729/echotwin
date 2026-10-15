"""Canonical message schema shared by ingestion, dataset, profiling and storage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ..language import Language

__all__ = [
    "AttachmentInfo",
    "Language",
    "Message",
    "MessageType",
    "ParticipantStats",
    "QualityFlag",
]


class QualityFlag(StrEnum):
    """Non-fatal issues found while parsing a message."""

    UNPARSED_TIMESTAMP = "unparsed_timestamp"
    MULTILINE = "multiline"
    ATTACHMENT_PLACEHOLDER = "attachment_placeholder"
    SYSTEM_MESSAGE = "system_message"
    DELETED_MESSAGE = "deleted_message"
    EDITED_MESSAGE = "edited_message"
    EMPTY_TEXT = "empty_text"
    MALFORMED_LINE = "malformed_line"
    DUPLICATE = "duplicate"
    NO_TIMESTAMP_LINE_ORDER = "no_timestamp_line_order"


class MessageType(StrEnum):
    TEXT = "text"
    ATTACHMENT = "attachment"
    SYSTEM = "system"
    DELETED = "deleted"
    CALL = "call"


class AttachmentInfo(BaseModel):
    kind: str = Field(description="image|audio|video|document|sticker|gif|contact|other")
    filename: str | None = None
    placeholder_text: str = Field(description="The literal placeholder text kept from the export")


class Message(BaseModel):
    """One canonical message. Timestamps are naive local time as exported."""

    message_id: str
    conversation_id: str
    timestamp: datetime | None = None
    sender_id: str | None = Field(
        default=None, description="Pseudonymous stable id; None for system lines"
    )
    sender_name: str | None = None
    text: str = ""
    message_type: MessageType = MessageType.TEXT
    attachment: AttachmentInfo | None = None
    is_system_message: bool = False
    is_target_author: bool = False
    language: Language = Language.UNKNOWN
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    line_number: int = Field(description="1-based line index of the header in the source file")

    @property
    def is_parseable_content(self) -> bool:
        """True when this message carries human-authored text usable for training."""
        if self.is_system_message or self.message_type is not MessageType.TEXT:
            return False
        return bool(self.text.strip())

    def pseudonymized_preview(self, max_len: int = 40) -> str:
        t = self.text.replace("\n", " ")[:max_len]
        return t + ("…" if len(self.text) > max_len else "")


class ParticipantStats(BaseModel):
    sender_id: str
    sender_name: str
    message_count: int = 0
    system_messages: int = 0
    attachment_messages: int = 0
    empty_messages: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    median_length: float = 0.0
    mean_length: float = 0.0
    language_distribution: dict[str, int] = Field(default_factory=dict)
    emoji_count: int = 0
    samples: list[str] = Field(
        default_factory=list, description="Recent short samples (redactable)"
    )
