"""Conversation-aware training-example construction.

Key correctness rules implemented here:

1. **No fabricated pairs.** Examples come only from real messages. Consecutive
   messages are NOT assumed to form reply pairs; we build *context windows* and
   label the example type (``natural_reply`` vs ``natural_statement``).
2. **Conversation segmentation.** Large time gaps (> config minutes) split a
   conversation into segments so a message 6 hours later is not treated as a
   reply to an earlier one.
3. **Token-aware truncation.** Context is trimmed from the oldest message first,
   within a character budget that approximates tokens (documented approximation).
4. **Style preservation.** Target text is preserved verbatim: typos, emoji,
   capitalization. No normalization of the target.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from ..exceptions import DatasetError, InsufficientDataError
from ..ingestion.schema import Message, QualityFlag
from ..language import detect_language

CHARS_PER_TOKEN_APPROX = 4  # documented approximation used only for budgeting


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ExampleMetadata(BaseModel):
    conversation_id: str
    language: str
    example_type: str  # natural_reply | natural_statement | self_continuation
    source_message_id: str | None = None
    source_line_number: int | None = None
    timestamp: str | None = None
    target_chars: int = 0
    context_chars: int = 0
    quality_flags: list[str] = Field(default_factory=list)


class TrainingExample(BaseModel):
    messages: list[ChatTurn]
    metadata: ExampleMetadata

    def key(self) -> str:
        """Stable identity used for exact dedup."""
        payload = json.dumps(
            [m.model_dump() for m in self.messages], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class DatasetConfig(BaseModel):
    context_messages: int = Field(default=4, ge=0, le=20)
    max_context_chars: int = Field(default=1200, ge=100)
    max_target_chars: int = Field(default=800, ge=10)
    min_target_chars: int = Field(default=2, ge=1)
    include_self_continuations: bool = True
    drop_low_information: bool = True
    low_information_exact: list[str] = Field(
        default_factory=lambda: [
            "ok",
            "k",
            "kk",
            "hmm",
            "hmm.",
            "yes",
            "no",
            "lol",
            "haan",
            "haan.",
        ]
    )
    low_information_max_emoji_only: bool = True
    segment_gap_minutes: int = 180
    exact_dedup: bool = True
    near_dedup_threshold: float = Field(
        default=1.0, ge=0.5, le=1.0, description="1.0 disables near-dup"
    )
    max_examples: int | None = None
    drop_duplicate_messages: bool = True
    keep_edited: bool = True


class BuiltDataset(BaseModel):
    examples: list[TrainingExample]
    config: DatasetConfig
    stats: dict


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


_EMOJI_ONLY_RE = re.compile(
    r"^[\s\u2600-\u27BF\uFE0F\u2190-\u21FF\u2B00-\u2BFF"
    r"\U0001F000-\U0001FAFF\u200D]+$"
)


def is_low_information(text: str, cfg: DatasetConfig) -> str | None:
    """Return the reason a message is low-information, or None if it passes."""
    t = text.strip()
    if not t:
        return "empty"
    if cfg.drop_low_information:
        if t.lower() in {v.lower() for v in cfg.low_information_exact}:
            return "low_info_exact"
        if cfg.low_information_max_emoji_only and _EMOJI_ONLY_RE.match(t):
            return "emoji_only"
    return None


def segment_messages(messages: list[Message], gap_minutes: int) -> list[list[Message]]:
    """Split a chronological message list at gaps > gap_minutes."""
    ordered = sorted(
        messages, key=lambda m: (m.timestamp is None, m.timestamp or m.line_number, m.line_number)
    )
    segments: list[list[Message]] = []
    current: list[Message] = []
    last_ts = None
    for m in ordered:
        if (
            last_ts is not None
            and m.timestamp is not None
            and m.timestamp - last_ts > timedelta(minutes=gap_minutes)
        ):
            segments.append(current)
            current = []
        current.append(m)
        if m.timestamp is not None:
            last_ts = m.timestamp
    if current:
        segments.append(current)
    return segments


def build_examples(
    messages: list[Message], target_sender_id: str, cfg: DatasetConfig | None = None
) -> BuiltDataset:
    cfg = cfg or DatasetConfig()
    if not any(m.sender_id == target_sender_id and m.is_parseable_content for m in messages):
        raise InsufficientDataError(
            "No usable messages from the target author. Did you select the right participant?"
        )

    by_convo: dict[str, list[Message]] = {}
    for m in messages:
        by_convo.setdefault(m.conversation_id, []).append(m)

    examples: list[TrainingExample] = []
    skipped = {
        "system": 0,
        "attachment": 0,
        "deleted": 0,
        "empty": 0,
        "duplicate": 0,
        "low_information": 0,
        "too_short": 0,
        "too_long": 0,
        "self_continuation": 0,
    }

    for convo_id, msgs in by_convo.items():
        for segment in segment_messages(msgs, cfg.segment_gap_minutes):
            content = [m for m in segment if m.is_parseable_content]
            for i, msg in enumerate(content):
                if msg.sender_id != target_sender_id:
                    continue
                if QualityFlag.DUPLICATE in msg.quality_flags and cfg.drop_duplicate_messages:
                    skipped["duplicate"] += 1
                    continue
                if QualityFlag.EDITED_MESSAGE in msg.quality_flags and not cfg.keep_edited:
                    continue

                reason = is_low_information(msg.text, cfg)
                if reason:
                    skipped["low_information"] += 1
                    continue
                if len(msg.text.strip()) < cfg.min_target_chars:
                    skipped["too_short"] += 1
                    continue
                if len(msg.text) > cfg.max_target_chars:
                    skipped["too_long"] += 1
                    continue

                prev = content[i - 1] if i > 0 else None
                prev_is_target = prev is not None and prev.sender_id == target_sender_id
                if prev_is_target and not cfg.include_self_continuations:
                    skipped["self_continuation"] += 1
                    continue

                # Context window: preceding messages, truncated oldest-first.
                window: list[Message] = []
                budget = cfg.max_context_chars
                for ctx_msg in reversed(content[max(0, i - cfg.context_messages) : i]):
                    cost = len(ctx_msg.text) + 1
                    if cost > budget:
                        break
                    window.insert(0, ctx_msg)
                    budget -= cost

                turns: list[ChatTurn] = [
                    ChatTurn(
                        role="assistant" if cm.sender_id == target_sender_id else "user",
                        content=cm.text,
                    )
                    for cm in window
                ]
                turns.append(ChatTurn(role="assistant", content=msg.text))

                context_text = "\n".join(t.content for t in turns[:-1])
                example_type = (
                    "natural_reply"
                    if prev is not None and not prev_is_target
                    else ("self_continuation" if prev_is_target else "natural_statement")
                )
                lang = detect_language(f"{msg.text} {context_text}")
                meta = ExampleMetadata(
                    conversation_id=convo_id,
                    language=lang.value,
                    example_type=example_type,
                    source_message_id=msg.message_id,
                    source_line_number=msg.line_number,
                    timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                    target_chars=len(msg.text),
                    context_chars=len(context_text),
                    quality_flags=[f.value for f in msg.quality_flags],
                )
                examples.append(TrainingExample(messages=turns, metadata=meta))

    # --- dedup ---
    if cfg.exact_dedup:
        seen: set[str] = set()
        unique: list[TrainingExample] = []
        for ex in examples:
            k = ex.key()
            if k in seen:
                skipped["duplicate_example"] = skipped.get("duplicate_example", 0) + 1
                continue
            seen.add(k)
            unique.append(ex)
        examples = unique

    if cfg.near_dedup_threshold < 1.0 and len(examples) <= 5000:
        unique = []
        for ex in examples:
            dup = False
            for kept in unique:
                if (
                    _similar(ex.messages[-1].content, kept.messages[-1].content)
                    >= cfg.near_dedup_threshold
                ):
                    dup = True
                    break
            if dup:
                skipped["near_duplicate_example"] = skipped.get("near_duplicate_example", 0) + 1
                continue
            unique.append(ex)
        examples = unique

    if cfg.max_examples is not None:
        examples = examples[: cfg.max_examples]

    if not examples:
        raise DatasetError(
            "All candidate messages were filtered out. Relax DatasetConfig "
            "(drop_low_information / min_target_chars / max_target_chars). "
            f"Skip reasons: {skipped}"
        )

    stats = compute_dataset_stats(examples)
    stats["skipped"] = skipped
    return BuiltDataset(examples=examples, config=cfg, stats=stats)


def compute_dataset_stats(examples: list[TrainingExample]) -> dict:
    if not examples:
        return {"n_examples": 0}
    target_lens = [ex.metadata.target_chars for ex in examples]
    ctx_lens = [ex.metadata.context_chars for ex in examples]
    langs: dict[str, int] = {}
    types: dict[str, int] = {}
    convos: set[str] = set()
    for ex in examples:
        langs[ex.metadata.language] = langs.get(ex.metadata.language, 0) + 1
        types[ex.metadata.example_type] = types.get(ex.metadata.example_type, 0) + 1
        convos.add(ex.metadata.conversation_id)
    n = len(examples)
    return {
        "n_examples": n,
        "conversations": len(convos),
        "languages": langs,
        "example_types": types,
        "target_chars": {
            "mean": sum(target_lens) / n,
            "min": min(target_lens),
            "max": max(target_lens),
        },
        "context_chars": {
            "mean": sum(ctx_lens) / n,
            "min": min(ctx_lens),
            "max": max(ctx_lens),
        },
        "approx_target_tokens_mean": sum(target_lens) / n / CHARS_PER_TOKEN_APPROX,
    }


def save_jsonl(examples: list[TrainingExample], path) -> int:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
    return len(examples)


def load_jsonl(path) -> list[TrainingExample]:
    from pathlib import Path

    p = Path(path)
    out: list[TrainingExample] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(TrainingExample.model_validate_json(line))
            except Exception as e:
                raise DatasetError(f"Invalid JSONL at {p}:{i}: {e}") from e
    return out
