"""Interpretable communication-style profiling.

Everything here is measurable. No psychological/personality inference: the
profile describes *what the text looks like*, not who the person is.
The profile is versioned JSON, editable by the user, and stable across runs.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise

from pydantic import BaseModel, Field

from ..ingestion.schema import Message
from ..language import detect_language, script_notes

SCHEMA_VERSION = "1.0"

_EMOJI_RE = re.compile("[\u2600-\u27bf\u2b00-\u2bff\ufe0f\U0001f000-\U0001faff\u200d]")
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+")
_SENTENCE_RE = re.compile(r"[.!?…]+")

_ABBREV_RE = re.compile(r"\b([A-Za-z]{1,6})\.", re.UNICODE)
_COMMON_ABBREVIATIONS = {
    "u",
    "ur",
    "pls",
    "plz",
    "tmrw",
    "tmw",
    "okk",
    "ok",
    "k",
    "kk",
    "cnf",
    "info",
    "msg",
    "msgs",
    "grp",
    "fr",
    "frnd",
    "frnds",
    "tm",
    "bcz",
    "coz",
    "bcoz",
    "tho",
    "thx",
    "tnx",
    "tym",
    "tme",
    "abt",
    "btw",
    "asap",
    "idk",
    "ily",
    "lol",
    "lmao",
    "omg",
    "wtf",
    "ngl",
    "tbh",
    "ikr",
    "smh",
    "rn",
    "tl",
    "dr",
}
_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "namaste",
    "namaskar",
    "good morning",
    "good evening",
    "good night",
    "kaise ho",
    "kaisi ho",
    "salaam",
    "hola",
}
_ACKS = {
    "ok",
    "okay",
    "k",
    "kk",
    "hmm",
    "haan",
    "ha",
    "yes",
    "yep",
    "nope",
    "no",
    "theek",
    "thik",
    "accha",
    "acha",
    "sure",
    "done",
    "thanks",
    "thank you",
    "ty",
}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def _capitalization_case(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    letters = [c for c in t if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return "all_caps"
    if t[:1].isupper() and any(c.islower() for c in letters):
        return "capitalized"
    if letters and all(c.islower() for c in letters):
        return "lowercase"
    return "mixed"


class StyleProfile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    target_sender_id: str
    n_messages: int = 0
    date_range: dict = Field(default_factory=dict)
    length: dict = Field(default_factory=dict, description="chars/words/fragment stats")
    sentence_structure: dict = Field(default_factory=dict)
    capitalization: dict = Field(default_factory=dict)
    punctuation: dict = Field(default_factory=dict)
    emoji: dict = Field(default_factory=dict)
    vocabulary: dict = Field(default_factory=dict, description="top tokens/phrases (counts only)")
    abbreviations: dict = Field(default_factory=dict)
    greetings_acks: dict = Field(default_factory=dict)
    language_use: dict = Field(default_factory=dict)
    code_switching: dict = Field(default_factory=dict)
    activity: dict = Field(default_factory=dict)
    time_trends: dict = Field(default_factory=dict)
    limitations: dict = Field(default_factory=dict)
    editable_notes: str = Field(default="", description="User-editable free text")
    edited_by_user: bool = False


def build_style_profile(messages: list[Message], target_sender_id: str) -> StyleProfile:
    own = [m for m in messages if m.sender_id == target_sender_id and m.is_parseable_content]
    if not own:
        from ..exceptions import InsufficientDataError

        raise InsufficientDataError("No usable target messages for style profiling.")
    own.sort(key=lambda m: (m.timestamp is None, m.timestamp or datetime.min, m.line_number))

    texts = [m.text for m in own]
    n = len(texts)
    char_lens = [float(len(t)) for t in texts]
    word_counts = [len(_WORD_RE.findall(t)) for t in texts]

    # punctuation
    punct_counter: Counter[str] = Counter()
    for t in texts:
        for ch in t:
            if ch in ".,!?;:'\"-…—()[]{}":
                punct_counter[ch] += 1
    uses_question = sum(1 for t in texts if "?" in t)
    ends_with_punct = sum(1 for t in texts if t.strip() and t.strip()[-1] in ".!?…")

    # capitalization classes
    case_counter: Counter[str] = Counter(_capitalization_case(t) for t in texts)

    # emoji
    emoji_counter: Counter[str] = Counter()
    for t in texts:
        emoji_counter.update(_EMOJI_RE.findall(t))
    emoji_msgs = sum(1 for t in texts if _EMOJI_RE.search(t))
    emoji_at_end = sum(1 for t in texts if t.strip() and _EMOJI_RE.match(t.strip()[-2:]))

    # vocabulary / phrases (counts only, safe for reports)
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(w.lower() for w in _WORD_RE.findall(t))
    uni = Counter(all_tokens)
    bi = Counter(pairwise(all_tokens))
    tri = Counter(zip(all_tokens, all_tokens[1:], all_tokens[2:], strict=False))

    # abbreviations
    abbrev_hits: Counter[str] = Counter()
    for t in texts:
        for w in _WORD_RE.findall(t):
            lw = w.lower()
            if lw in _COMMON_ABBREVIATIONS:
                abbrev_hits[lw] += 1

    # greetings / acknowledgement openers
    greeting_msgs = 0
    ack_msgs = 0
    for t in texts:
        first_words = " ".join(w.lower() for w in _WORD_RE.findall(t)[:3])
        if any(g in first_words for g in _GREETINGS):
            greeting_msgs += 1
        stripped = t.strip().lower().strip(".!?")
        if len(_WORD_RE.findall(t)) <= 2 and (
            stripped in _ACKS or any(stripped.startswith(a) for a in _ACKS)
        ):
            ack_msgs += 1

    # language use + code-switching
    lang_counter: Counter[str] = Counter()
    devanagari_msgs = 0
    latin_msgs = 0
    mixed_msgs = 0
    for t in texts:
        lang = detect_language(t)
        lang_counter[lang.value] += 1
        has_dev = any("\u0900" <= c <= "\u097f" for c in t)
        has_lat = any("a" <= c.lower() <= "z" for c in t)
        devanagari_msgs += has_dev
        latin_msgs += has_lat
        mixed_msgs += has_dev and has_lat

    # activity / time trends
    hours: Counter[int] = Counter(m.timestamp.hour for m in own if m.timestamp is not None)
    first = own[0].timestamp if own[0].timestamp else None
    last = own[-1].timestamp if own[-1].timestamp else None
    monthly: dict[str, float] = {}
    for m in own:
        if m.timestamp is None:
            continue
        key = m.timestamp.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0.0) + len(m.text)
    months = sorted(monthly)
    trend_note = ""
    if len(months) >= 4:
        mid = len(months) // 2
        early = _mean([monthly[m] for m in months[:mid]]) if mid else 0.0
        late = _mean([monthly[m] for m in months[mid:]])
        trend_note = (
            f"average chars/month changed from {early:.0f} to {late:.0f} across the observed range"
        )
    else:
        trend_note = "insufficient time span (<4 months) for a trend statement"

    long_msgs = sum(1 for t in texts if len(t) > 160)

    return StyleProfile(
        target_sender_id=target_sender_id,
        n_messages=n,
        date_range={
            "first": first.isoformat() if first else None,
            "last": last.isoformat() if last else None,
        },
        length={
            "chars_mean": round(_mean(char_lens), 2),
            "chars_median": _percentile(char_lens, 0.5),
            "chars_p90": _percentile(char_lens, 0.9),
            "words_mean": round(_mean([float(w) for w in word_counts]), 2),
            "long_messages_over_160_chars": long_msgs,
            "long_share_pct": _pct(long_msgs, n),
        },
        sentence_structure={
            "sentences_per_message_mean": round(
                _mean([float(max(1, len(_SENTENCE_RE.findall(t)))) for t in texts]), 3
            ),
            "single_sentence_share_pct": _pct(
                sum(1 for t in texts if len(_SENTENCE_RE.findall(t)) <= 1), n
            ),
            "avg_words_between_punct": round(
                _mean(
                    [
                        float(len(_WORD_RE.findall(part)))
                        for t in texts
                        for part in _SENTENCE_RE.split(t.strip())
                        if part.strip()
                    ]
                ),
                2,
            ),
        },
        capitalization={k: _pct(v, n) for k, v in case_counter.items()},
        punctuation={
            "per_message": {k: round(v / n, 3) for k, v in punct_counter.most_common()},
            "question_share_pct": _pct(uses_question, n),
            "terminal_punct_share_pct": _pct(ends_with_punct, n),
        },
        emoji={
            "messages_with_emoji_pct": _pct(emoji_msgs, n),
            "per_message": round(sum(emoji_counter.values()) / n, 3),
            "top": dict(emoji_counter.most_common(10)),
            "trailing_position_share_pct_of_emoji_msgs": _pct(emoji_at_end, emoji_msgs),
        },
        vocabulary={
            "top_tokens": dict(uni.most_common(25)),
            "top_bigrams": {" ".join(k): v for k, v in bi.most_common(12)},
            "top_trigrams": {" ".join(k): v for k, v in tri.most_common(8)},
            "distinct_tokens": len(uni),
            "token_repetition_ratio": round(1.0 - len(uni) / len(all_tokens), 4)
            if all_tokens
            else 0.0,
        },
        abbreviations={
            "counts": dict(abbrev_hits.most_common(20)),
            "abbreviated_messages_share_pct": _pct(
                sum(
                    1
                    for t in texts
                    if any(w.lower() in _COMMON_ABBREVIATIONS for w in _WORD_RE.findall(t))
                ),
                n,
            ),
        },
        greetings_acks={
            "greeting_opener_share_pct": _pct(greeting_msgs, n),
            "short_ack_share_pct": _pct(ack_msgs, n),
        },
        language_use={
            "distributions": {k: _pct(v, n) for k, v in lang_counter.items()},
            "devanagari_messages": devanagari_msgs,
            "latin_messages": latin_msgs,
            "mixed_script_messages": mixed_msgs,
            "detector_note": script_notes(),
        },
        code_switching={
            "mixed_message_share_pct": _pct(mixed_msgs, n),
            "note": "Mixed-script share is a proxy for Hinglish code-mixing, "
            "not a measure of fluency.",
        },
        activity={
            "by_hour_pct": {
                str(h): _pct(v, len([m for m in own if m.timestamp]))
                for h, v in sorted(hours.items())
            },
        },
        time_trends={
            "monthly_chars": monthly,
            "trend_note": trend_note,
        },
        limitations={
            **script_notes(),
            "vocab_privacy": "Only counts are stored, never raw messages.",
            "no_psychology": "This profile contains no psychological or personality inference.",
        },
    )


def profile_prompt_fragment(profile: StyleProfile) -> str:
    """Render a compact style-fragment usable inside prompts (counts + examples only)."""
    length = profile.length
    caps = profile.capitalization
    punct = profile.punctuation
    emoji = profile.emoji
    lines = [
        "Writing-style profile (measured from the user's own messages):",
        f"- Typical length: mean {length.get('chars_mean', 0):.0f} chars / "
        f"{length.get('words_mean', 0):.0f} words per message.",
        f"- Capitalization: mostly {max(caps, key=lambda k: caps[k]) if caps else 'mixed'}.",
        f"- Terminal punctuation in {punct.get('terminal_punct_share_pct', 0)}% of messages; "
        f"questions in {punct.get('question_share_pct', 0)}%.",
        f"- Emoji: {emoji.get('messages_with_emoji_pct', 0)}% of messages contain emoji "
        f"({emoji.get('per_message', 0)} per message).",
        f"- Languages: {profile.language_use.get('distributions', {})}",
        f"- Greetings open {profile.greetings_acks.get('greeting_opener_share_pct', 0)}% of messages; "
        f"short acknowledgements {profile.greetings_acks.get('short_ack_share_pct', 0)}%.",
        "Reply in the user's natural register: preserve spelling, emoji habits, and code-mixing.",
    ]
    return "\n".join(lines)


def save_profile(profile: StyleProfile, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def load_profile(path) -> StyleProfile:
    from pathlib import Path

    return StyleProfile.model_validate_json(Path(path).read_text(encoding="utf-8"))
