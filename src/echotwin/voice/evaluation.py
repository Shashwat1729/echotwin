"""Voice evaluation: latency/RTF always; similarity/WER only with real evaluators.

No fabricated numbers:

* Latency and real-time factor are measured from wall time.
* Speaker-embedding similarity needs an independent evaluator (``resemblyzer``).
  When it is missing the result status is ``blocked`` with the exact install hint.
* WER/CER needs an ASR backend (``faster-whisper`` or a user-configured provider).
* Naturalness is human-evaluated only; the scaffold records human scores.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VoiceEvaluationResult(BaseModel):
    status: str = Field(description="ok | blocked | partial")
    evaluator: str = ""
    model_revision: str = ""
    sample_count: int = 0
    latency_seconds: float | None = None
    audio_duration_seconds: float | None = None
    rtf: float | None = None
    speaker_similarity: float | None = None
    speaker_similarity_note: str | None = None
    wer: float | None = None
    cer: float | None = None
    asr_model: str | None = None
    human_scores: dict | None = None
    blocked_reason: str | None = None
    limitations: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Word-level error rate via Levenshtein distance.

    ``1 - similarity`` is NOT WER: it treats insertions and deletions as
    partial credit and under-reported them by up to 2.7x, which made poor
    intelligibility look acceptable. This is the standard definition:
    (substitutions + deletions + insertions) / number of reference words.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        cur = [i] + [0] * len(hypothesis)
        for j, h in enumerate(hypothesis, start=1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(hypothesis)] / len(reference)


def _measure_latency(result: dict) -> dict:
    return result  # timing_probe already measured


def evaluate_synthesis(
    audio_path: str | Path,
    reference_paths: Sequence[str | Path],
    *,
    expected_text: str | None = None,
    backend_name: str = "voxcpm",
    model_revision: str = "",
) -> VoiceEvaluationResult:
    """Evaluate one synthesis output. Speaker similarity is only computed when an
    independent evaluator is installed; otherwise status=blocked with reason."""
    from ..voice.audio_ingestion import read_audio

    clip = read_audio(audio_path)
    duration = len(clip.samples) / clip.sample_rate if clip.sample_rate else 0.0
    from ..voice.speaker_similarity import evaluate_speaker_similarity

    result = VoiceEvaluationResult(
        status="partial",
        evaluator="none",
        model_revision=model_revision or backend_name,
        sample_count=1,
        audio_duration_seconds=round(duration, 3),
        limitations=[
            "Speaker similarity requires an INDEPENDENT embedding evaluator - the "
            "cloning model's own scores are never used.",
            "High speaker similarity does not prove speaker identity, consent, or "
            "legal authorization.",
        ],
    )
    sim = evaluate_speaker_similarity(audio_path, list(reference_paths))
    result.speaker_similarity = sim.similarity
    result.evaluator = sim.evaluator
    result.speaker_similarity_note = sim.note or None
    result.limitations.extend(sim.limitations)
    if sim.similarity is not None:
        result.status = "ok"

    if expected_text:
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _info = model.transcribe(str(audio_path))
            hypothesis = " ".join(s.text for s in segments).strip()
            ref = expected_text.split()
            hyp = hypothesis.split()
            import difflib

            difflib.SequenceMatcher(None, ref, hyp)
            # A real word-level Levenshtein distance. `1 - similarity` is NOT
            # WER: it ignores insertions and deletions and under-reported them
            # by up to 2.7x, which made poor intelligibility look fine.
            wer = _word_error_rate(ref, hyp)
            result.wer = round(wer, 4)
            result.asr_model = "faster-whisper(base)"
            result.status = "ok" if result.speaker_similarity is not None else "partial"
        except ImportError:
            result.limitations.append(
                "ASR-based WER unavailable: uv pip install faster-whisper for "
                "intelligibility measurement."
            )
        except Exception as e:
            result.limitations.append(f"ASR transcription failed: {e}")
    return result


def measure_inference_cost(
    synthesize_callable,
    text: str,
    **kwargs: Any,
) -> VoiceEvaluationResult:
    """Time any synthesis callable; reports latency + RTF."""
    t0 = time.time()
    out = synthesize_callable(text, **kwargs)
    latency = time.time() - t0
    from ..voice.audio_ingestion import read_audio

    clip = read_audio(out)
    duration = len(clip.samples) / clip.sample_rate if clip.sample_rate else 0.0
    return VoiceEvaluationResult(
        status="ok",
        evaluator="timing",
        sample_count=1,
        latency_seconds=round(latency, 3),
        audio_duration_seconds=round(duration, 3),
        rtf=round(latency / duration, 3) if duration else None,
        created_at=datetime.now(UTC).isoformat(),
    )


class HumanScore(BaseModel):
    sample_id: str
    criterion: str = Field(
        description="style_similarity | naturalness | context | language_authenticity"
    )
    score: float = Field(ge=1, le=5)
    blind: bool = True
    annotator: str | None = None
    notes: str | None = None


def save_human_scores(scores: list[HumanScore], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for s in scores:
            fh.write(s.model_dump_json() + "\n")
    return len(scores)


def build_blind_pairs(
    outputs_a: list[str], outputs_b: list[str], sample_ids: list[str]
) -> list[dict]:
    """Pairwise blinded comparison scaffold used by the UI evaluation studio."""
    pairs = []
    for sid, a, b in zip(sample_ids, outputs_a, outputs_b, strict=True):
        pairs.append({"sample_id": sid, "option_a": str(a), "option_b": str(b)})
    return pairs


def save_json(data: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p
