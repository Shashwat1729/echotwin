"""Participant discovery and reporting.

The target author is NEVER auto-selected as the most active participant: the
person holding a group export is frequently not the most talkative member.
Participants are reported with statistics, and the user selects the target
explicitly (CLI argument, UI dropdown, or API call).
"""

from __future__ import annotations

from collections import Counter

from ..ingestion.schema import Message, ParticipantStats
from ..language import detect_language


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def participant_stats(messages: list[Message]) -> list[ParticipantStats]:
    stats: dict[str, ParticipantStats] = {}
    for msg in messages:
        if msg.sender_id is None or msg.sender_name is None:
            continue
        st = stats.setdefault(
            msg.sender_id,
            ParticipantStats(sender_id=msg.sender_id, sender_name=msg.sender_name),
        )
        st.message_count += 1
        if msg.is_system_message:
            st.system_messages += 1
        if msg.message_type.value == "attachment":
            st.attachment_messages += 1
        if not msg.text.strip():
            st.empty_messages += 1
        if msg.timestamp is not None:
            if st.first_timestamp is None or msg.timestamp < st.first_timestamp:
                st.first_timestamp = msg.timestamp
            if st.last_timestamp is None or msg.timestamp > st.last_timestamp:
                st.last_timestamp = msg.timestamp
        if msg.text.strip():
            lang = detect_language(msg.text)
            st.language_distribution[lang.value] = st.language_distribution.get(lang.value, 0) + 1
            st.emoji_count += sum(
                1 for ch in msg.text if ord(ch) > 0x1F000 or ch in {"☺", "♥", "☹", "✅", "❌"}
            )
            if len(st.samples) < 5 and len(msg.text) <= 80:
                st.samples.append(msg.text.replace("\n", " "))

    # length stats computed in a second pass
    lengths: dict[str, list[float]] = {sid: [] for sid in stats}
    for msg in messages:
        if msg.sender_id in lengths and msg.text.strip():
            lengths[msg.sender_id].append(len(msg.text))
    for sid, st in stats.items():
        vals = lengths[sid]
        st.mean_length = sum(vals) / len(vals) if vals else 0.0
        st.median_length = _median(vals)
    return sorted(stats.values(), key=lambda s: s.message_count, reverse=True)


def participant_report(messages: list[Message]) -> list[str]:
    """Human-readable lines for CLI/API/UI display (no raw message bodies except samples)."""
    lines = []
    for st in participant_stats(messages):
        langs = ", ".join(f"{k}:{v}" for k, v in sorted(st.language_distribution.items()))
        lines.append(
            f"{st.sender_name!r} (id={st.sender_id}): {st.message_count} messages "
            f"({st.system_messages} system, {st.attachment_messages} attachments, "
            f"{st.empty_messages} empty); {st.first_timestamp} .. {st.last_timestamp}; "
            f"mean_len={st.mean_length:.0f} median_len={st.median_length:.0f}; "
            f"langs[{langs}]"
        )
    return lines


def pick_participant(
    messages: list[Message], name_or_id: str, *, exact: bool = False
) -> str | None:
    """Resolve a user-supplied name (case-insensitive) or id to a sender_id."""
    stats = participant_stats(messages)
    needle = name_or_id.strip().lower()
    for st in stats:
        if st.sender_id == name_or_id:
            return st.sender_id
    for st in stats:
        if st.sender_name.lower() == needle:
            return st.sender_id
    if exact:
        return None
    matches = [st for st in stats if needle in st.sender_name.lower()]
    if len(matches) == 1:
        return matches[0].sender_id
    return None


def language_histogram(messages: list[Message]) -> Counter[str]:
    return Counter(detect_language(m.text).value for m in messages if m.text.strip())
