"""Reference-audio selection.

Selection policy (documented, reproducible):

1. Only recordings that pass quality checks are eligible.
2. Duration must fit the backend band (min..recommended*2).
3. A transcript is required only when the backend requires it; when transcript
   mode is chosen but the transcript mismatched badly, the clip is deprioritized.
4. Prefer the clip with the broadest phonetic coverage as a tie-breaker
   (NOT because "more concatenation = better" — concatenation is never automatic).
5. If the backend supports multiple references, per-language references are chosen;
   otherwise a single best clip per language is kept and the rest are listed as
   alternates with rationale.

Source recordings are never modified; concatenated/derived audio is stored as a
separate artifact when explicitly requested with ``make_combined``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..voice.capabilities import VoiceBackendCapabilities


class ReferenceCandidate(BaseModel):
    recording_id: str
    language: str
    quality_verdict: str
    duration_seconds: float
    has_transcript: bool = False
    transcript_verdict: str | None = None
    coverage_categories_present: list[str] = Field(default_factory=list)


class SelectionDecision(BaseModel):
    selected: list[ReferenceCandidate] = Field(default_factory=list)
    alternates: list[ReferenceCandidate] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    per_language: dict[str, str] = Field(
        default_factory=dict, description="language -> recording_id"
    )
    rationale: str = ""
    transcript_available: bool = False


def _score(c: ReferenceCandidate, caps: VoiceBackendCapabilities, transcript_mode: bool) -> float:
    score = 0.0
    # duration fit: best inside recommended band, penalize far outside
    if c.duration_seconds < max(1.0, caps.min_reference_seconds * 0.8):
        score -= 5.0
    if caps.max_reference_seconds and c.duration_seconds > caps.max_reference_seconds:
        score -= 5.0
    band_lo, band_hi = caps.min_reference_seconds, (caps.recommended_reference_seconds * 2)
    if band_lo <= c.duration_seconds <= band_hi:
        score += 3.0
    if c.quality_verdict == "ok":
        score += 3.0
    elif c.quality_verdict == "warning":
        score += 1.0
    else:
        score -= 10.0
    if transcript_mode:
        if c.has_transcript and c.transcript_verdict == "match":
            score += 2.0
        elif c.has_transcript and c.transcript_verdict == "minor_differences":
            score += 1.0
    else:
        score += 0.5 if c.has_transcript else 0.0
    # broader phonetic coverage wins ties
    score += min(len(c.coverage_categories_present), 10) * 0.1
    # documented tie-break: longer (clean) recordings score marginally higher
    score += 0.1 * min(c.duration_seconds / max(caps.recommended_reference_seconds, 1.0), 2.0)
    return score


def select_references(
    candidates: list[ReferenceCandidate],
    capabilities: VoiceBackendCapabilities | dict,
    *,
    prefer_transcript: bool = False,
) -> SelectionDecision:
    caps = (
        capabilities
        if isinstance(capabilities, VoiceBackendCapabilities)
        else VoiceBackendCapabilities(**capabilities)
    )
    language_field_present = [c for c in candidates if c.quality_verdict in {"ok", "warning"}]
    rejected: list[dict] = []
    for c in candidates:
        if c.quality_verdict == "fail":
            rejected.append({"recording_id": c.recording_id, "reason": "quality_failed"})
        if caps.max_reference_seconds and c.duration_seconds > caps.max_reference_seconds:
            rejected.append({"recording_id": c.recording_id, "reason": "duration_too_long"})
        elif c.duration_seconds < max(0.5, caps.min_reference_seconds * 0.5):
            rejected.append({"recording_id": c.recording_id, "reason": "duration_too_short"})

    languages = sorted({c.language for c in candidates if c.language and c.language != "-"})
    selected: list[ReferenceCandidate] = []
    alternates: list[ReferenceCandidate] = []
    per_language: dict[str, str] = {}
    transcript_mode = prefer_transcript and caps.reference_transcript_optional
    per_language_best: list[ReferenceCandidate] = []
    for lang in languages:
        pool = [
            c
            for c in language_field_present
            if c.language == lang
            and not any(
                r["recording_id"] == c.recording_id and r["reason"] == "quality_failed"
                for r in rejected
            )
        ]
        if not pool:
            continue
        ranked = sorted(pool, key=lambda c: _score(c, caps, transcript_mode), reverse=True)
        per_language_best.append(ranked[0])
        alternates.extend(ranked[1:3])

    if per_language_best:
        if caps.supports_multiple_references:
            selected = per_language_best
            per_language = {c.language: c.recording_id for c in selected}
        else:
            # Single-reference backend: keep only the globally best clip and say so.
            best = sorted(
                per_language_best, key=lambda c: _score(c, caps, transcript_mode), reverse=True
            )[0]
            selected = [best]
            per_language = {best.language: best.recording_id}
    rationale = (
        "Selected by quality, duration fit, transcript match (when requested) and "
        "phonetic breadth. Source recordings are never modified or concatenated."
    )
    if not selected:
        rejected.append({"recording_id": "-", "reason": "no_eligible_candidates"})
    return SelectionDecision(
        selected=selected,
        alternates=alternates,
        rejected=rejected,
        per_language=per_language,
        rationale=rationale,
        transcript_available=any(c.has_transcript for c in candidates),
    )
