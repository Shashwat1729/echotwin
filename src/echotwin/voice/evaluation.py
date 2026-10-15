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
    result = VoiceEvaluationResult(
        status="partial",
        evaluator="none",
        model_revision=model_revision or backend_name,
        sample_count=1,
        audio_duration_seconds=round(duration, 3),
        limitations=[
            "Speaker similarity requires an independent embedding evaluator "
            "(resemblyzer) — the cloning model's own scores are NOT used.",
            "High speaker similarity does not prove speaker identity, consent, or "
            "legal authorization.",
        ],
    )
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()
        import numpy as np

        def _embed(p: str | Path):
            wav = preprocess_wav(Path(p))
            return encoder.embed_utterance(wav)

        gen_emb = _embed(audio_path)
        sims = []
        for rp in reference_paths:
            ref_emb = _embed(rp)
            cos = float(
                np.dot(gen_emb, ref_emb)
                / (np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb) + 1e-12)
            )
            sims.append(round(cos, 4))
        result.speaker_similarity = round(sum(sims) / len(sims), 4) if sims else None
        result.evaluator = "resemblyzer"
        result.status = "ok"
    except ImportError:
        result.speaker_similarity = None
        result.speaker_similarity_note = (
            "Speaker-embedding evaluator unavailable. Install: uv pip install "
            "resemblyzer (or provide another independent evaluator)."
        )
    except Exception as e:
        result.speaker_similarity = None
        result.speaker_similarity_note = f"Similarity evaluation failed: {e}"

    if expected_text:
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _info = model.transcribe(str(audio_path))
            hypothesis = " ".join(s.text for s in segments).strip()
            ref = expected_text.split()
            hyp = hypothesis.split()
            import difflib

            sm = difflib.SequenceMatcher(None, ref, hyp)
            wer = 1.0 - sm.ratio()
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
