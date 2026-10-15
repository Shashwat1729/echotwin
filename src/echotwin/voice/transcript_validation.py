"""Transcript verification.

Compares the intended script with what is believed to have been said:

* ``user``  — the user typed the transcript themselves (never auto-modified audio)
* ``asr``   — a local/external ASR provider *explicitly configured by the user*
  (never called automatically)

ASR is NOT ground truth, especially for Hindi, Romanized Hindi, Hinglish, names
and code-switching — mismatches are classified as *likely* categories, never
as certain errors, and the user can always correct the transcript.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from pydantic import BaseModel, Field


class TranscriptCheck(BaseModel):
    source: str = Field(description="user | asr | none")
    intended: str
    transcript: str
    normalized_similarity: float
    missing_words: list[str] = Field(default_factory=list)
    extra_words: list[str] = Field(default_factory=list)
    substituted_words: list[tuple[str, str]] = Field(default_factory=list)
    repeated_phrases: list[str] = Field(default_factory=list)
    likely_incomplete: bool = False
    flags: list[str] = Field(default_factory=list, description="e.g. asr_unreliable_for_hinglish")
    verdict: str = Field(description="match | minor_differences | mismatch | unverified")
    note: str = ""


def normalize_words(text: str) -> list[str]:
    import re

    words = re.findall(r"[\w\u0900-\u097F']+", text.lower())
    return [w.strip("'") for w in words if w.strip("'")]


def _find_repeats(words: list[str], n: int = 2, min_count: int = 2) -> list[str]:
    seen: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i : i + n])
        seen[phrase] = seen.get(phrase, 0) + 1
    return [p for p, c in seen.items() if c >= min_count]


def verify_transcript(
    intended: str,
    transcript: str,
    *,
    source: str = "user",
    language: str = "en",
) -> TranscriptCheck:
    """Align intended vs transcript and classify likely differences."""
    if not transcript or not transcript.strip():
        return TranscriptCheck(
            source=source,
            intended=intended,
            transcript=transcript or "",
            normalized_similarity=0.0,
            verdict="unverified",
            note="No transcript available; provide one manually or configure an ASR provider.",
        )

    a = normalize_words(intended)
    b = normalize_words(transcript)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    missing: list[str] = []
    extra: list[str] = []
    substituted: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            missing.extend(a[i1:i2])
        elif tag == "insert":
            extra.extend(b[j1:j2])
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                substituted.append((a[i1 + k], b[j1 + k]))
            if (i2 - i1) > (j2 - j1):
                missing.extend(a[j1 - j1 + i1 + (j2 - j1) : i2])
            elif (j2 - j1) > (i2 - i1):
                extra.extend(b[i2 - i2 + j1 + (i2 - i1) : j2])

    similarity = matcher.ratio()
    repeats = _find_repeats(b)
    likely_incomplete = len(missing) > 0 and len(missing) / max(1, len(a)) > 0.25

    flags: list[str] = []
    if source == "asr":
        if language in {"hinglish", "hi-Latn", "hi-en"}:
            flags.append("asr_unreliable_for_hinglish")
        if language == "hi":
            flags.append("asr_unreliable_for_hindi")
        flags.append("asr_not_ground_truth")

    if similarity >= 0.97:
        verdict = "match"
        note = "Transcript matches the intended script."
    elif similarity >= 0.85:
        verdict = "minor_differences"
        note = "Small differences — check names, numbers and code-switched words manually."
    else:
        verdict = "mismatch"
        note = (
            "Transcript differs substantially. This may be a pronunciation mismatch, "
            "wrong recording, or background speech — verify before using as reference."
        )

    return TranscriptCheck(
        source=source,
        intended=intended,
        transcript=transcript,
        normalized_similarity=round(similarity, 4),
        missing_words=missing[:20],
        extra_words=extra[:20],
        substituted_words=substituted[:20],
        repeated_phrases=repeats[:5],
        likely_incomplete=likely_incomplete,
        flags=flags,
        verdict=verdict,
        note=note,
    )
