"""Speaker-similarity evaluation for cloned audio.

Why this module exists
----------------------
Measuring how close a clone is to the original speaker needs an **independent**
speaker-embedding model — never the cloning model's own score. The field
standard (Seed-TTS-eval, ClonEval, and the Open Voice Cloning Leaderboard) is:

* ``microsoft/wavlm-base-plus-sv`` (WavLM-TDNN x-vector),
* audio resampled to **16 kHz** before embedding,
* **cosine similarity** between reference and generated embeddings,
* averaged over samples.

Typical clone scores land around **0.7-0.85** cosine; below ~0.7 the clone
usually does not carry the speaker convincingly. We report that threshold
alongside the number so a score can be read in context instead of in isolation.

Documented caveats (we do not hide these):

* Embedding-based similarity is a **proxy**. It measures timbre similarity, not
  identity, consent, or legal authorisation.
* Embedding systems are **accent-biased**: accented speakers tend to score
  lower than standard-accent speakers, and human listeners rate accented clones
  as less similar than automatic metrics suggest (arXiv:2604.01562, 2026).
  Treat a lower score for an Indian-accented speaker as partly an artefact.
* Intelligibility measured by ASR inherits the same accent bias in WER.

``resemblyzer`` is supported as a lighter alternative when the WavLM stack is
unavailable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Cosine similarity at or above which two clips are treated as the same
#: speaker by the default evaluator (ECAPA-TDNN verification threshold).
#:
#: We measured the alternatives on real speech (see
#: ``artifacts/benchmarks/speaker_similarity_*.json``):
#:
#: * ECAPA-TDNN: same speaker 0.61, different speakers 0.07-0.15 -> separates.
#: * wavlm-base-plus-sv: same speaker 0.97, different speakers 0.78 -> does NOT
#:   separate (different-speaker pairs reached 0.945).
#:
#: The widely quoted 0.7 threshold comes from clone-vs-reference scoring
#: (Seed-TTS-eval / ClonEval), a different regime, and must not be applied to
#: natural-speech speaker verification.
SPEAKER_SIMILARITY_THRESHOLD = 0.25

#: Threshold used when the (less discriminative) x-vector evaluator is selected.
X_VECTOR_THRESHOLD = 0.7

#: Sample rate the WavLM-SV checkpoint expects.
EVAL_SAMPLE_RATE = 16000

WAVLM_SV_MODEL_ID = "microsoft/wavlm-base-plus-sv"


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """Anything that turns a wav file into a speaker embedding vector."""

    name: str

    def embed(self, audio_path: str | Path) -> np.ndarray: ...


@dataclass
class ECAPASpeakerEmbedder:
    """Field-standard *verification* model (``speechbrain/spkrec-ecapa-voxceleb``).

    Unlike the x-vector model this one is trained specifically for speaker
    verification and separates speakers reliably on read speech (measured: same
    speaker ~0.61 vs different speakers 0.07-0.15).

    Requires the ``speechbrain`` extra and a one-time model download.
    """

    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    device: str = "cpu"
    window_seconds: float = 3.0
    name: str = "ecapa-tdnn"

    def embed(self, audio_path: str | Path) -> np.ndarray:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        from ..voice.audio_ingestion import read_audio
        from ..voice.preprocessing import resample

        clip = read_audio(audio_path, mono=True)
        wav = clip.samples.astype(np.float32, copy=False)
        if clip.sample_rate != EVAL_SAMPLE_RATE:
            wav = resample(wav, clip.sample_rate, EVAL_SAMPLE_RATE)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        # Verification models expect a short, speech-active window.
        tensor = tensor[:, : int(EVAL_SAMPLE_RATE * self.window_seconds)]

        cache = Path.home() / ".cache" / "echotwin" / "ecapa"
        model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=cache,
            local_strategy=LocalStrategy.COPY,  # Windows: symlinks need a privilege
            run_opts={"device": self.device},
        )
        with torch.no_grad():
            emb = model.encode_batch(tensor).squeeze().cpu().numpy()
        return np.asarray(emb, dtype=np.float32)


@dataclass
class ResemblyzerEmbedder:
    """Lighter alternative embedder (needs the ``resemblyzer`` package)."""

    name: str = "resemblyzer"

    def embed(self, audio_path: str | Path) -> np.ndarray:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()
        wav = preprocess_wav(Path(audio_path))
        return np.asarray(encoder.embed_utterance(wav), dtype=np.float32)


@dataclass
class WavLMSpeakerEmbedder:
    """Field-standard embedder (``microsoft/wavlm-base-plus-sv``).

    Requires ``transformers`` + ``torch`` and downloads ~1.3 GB on first use.
    Audio is resampled to 16 kHz per the evaluation protocol.
    """

    model_id: str = WAVLM_SV_MODEL_ID
    device: str = "cpu"
    name: str = "wavlm-base-plus-sv"

    def embed(self, audio_path: str | Path) -> np.ndarray:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        from ..voice.audio_ingestion import read_audio
        from ..voice.preprocessing import resample

        clip = read_audio(audio_path, mono=True)
        wav = clip.samples.astype(np.float32, copy=False)
        if clip.sample_rate != EVAL_SAMPLE_RATE:
            wav = resample(wav, clip.sample_rate, EVAL_SAMPLE_RATE)

        extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WavLMForXVector.from_pretrained(self.model_id)
        model.eval()
        inputs: Any = extractor(
            [wav], sampling_rate=EVAL_SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch_no_grad():
            embeddings = model(**inputs).embeddings
        return np.asarray(embeddings[0].detach().cpu().numpy(), dtype=np.float32)


def torch_no_grad():
    """Return ``torch.no_grad`` (imported lazily so torch stays optional)."""
    import torch

    return torch.no_grad()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embedding vectors (0 when either is degenerate)."""
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def available_embedders() -> list[str]:
    """Names of evaluators that can actually run right now (no downloads)."""
    names: list[str] = []
    try:
        import resemblyzer  # noqa: F401

        names.append("resemblyzer")
    except Exception:
        pass
    try:
        import torch
        import transformers  # noqa: F401

        names.append("wavlm-base-plus-sv")
    except Exception:
        pass
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401

        names.append("ecapa-tdnn")
    except Exception:
        pass
    return names


def get_embedder(preferred: str | None = None) -> SpeakerEmbedder:
    """Return an embedder, preferring the field standard WavLM-SV.

    Raises ``BackendNotAvailableError`` with an actionable install hint when
    neither evaluator is usable.
    """
    from ..exceptions import BackendNotAvailableError

    order = [preferred] if preferred else []
    # ECAPA first: it is the only evaluator we measured to separate speakers.
    order += ["ecapa-tdnn", "wavlm-base-plus-sv", "resemblyzer"]
    for name in order:
        if not name:
            continue
        try:
            if name == "ecapa-tdnn":
                return ECAPASpeakerEmbedder()
            if name == "wavlm-base-plus-sv":
                return WavLMSpeakerEmbedder()
            if name == "resemblyzer":
                return ResemblyzerEmbedder()
        except Exception:
            continue
    raise BackendNotAvailableError(
        "No speaker-similarity evaluator is installed. Install one of:\n"
        "  uv pip install speechbrain torch      # spkrec-ecapa-voxceleb (recommended; "
        "the only one we measured to separate speakers)\n"
        "  uv pip install transformers torch     # wavlm-base-plus-sv (x-vector)\n"
        "  uv pip install resemblyzer             # lighter alternative\n"
        "See docs/voice-cloning.md for the evaluation protocol and its limits."
    )


@dataclass
class SpeakerSimilarityResult:
    similarity: float | None
    evaluator: str
    threshold: float = SPEAKER_SIMILARITY_THRESHOLD
    per_reference: list[float] = None  # type: ignore[assignment]
    same_speaker_baseline: float | None = None
    calibrated_score: float | None = None
    baseline_pairs: int = 0
    note: str = ""
    limitations: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_reference is None:
            self.per_reference = []
        if self.limitations is None:
            self.limitations = []


def evaluate_speaker_similarity(
    generated_path: str | Path,
    reference_paths: list[str | Path],
    *,
    embedder: SpeakerEmbedder | None = None,
) -> SpeakerSimilarityResult:
    """Cosine similarity between a clone and its reference(s), calibrated.

    Uses the field protocol (16 kHz, cosine, mean over references). Never uses
    the cloning model's own confidence as a similarity score.

    **Why calibration.** A raw cosine is not portable between reference sets.
    On real read speech (FLEURS hi_in, measured 2026-10-18) a *different*
    speaker pair reached 0.945 while a *same*-speaker pair reached 0.696 -- the
    ranges overlap, so a fixed threshold cannot tell speakers apart. We
    therefore also compute the **same-speaker baseline** (how similar the
    user's own reference clips are to each other) and report the clone's score
    relative to it. A calibrated score near 1.0 means "the clone is about as
    close to this speaker as the speaker is to themselves".
    """
    limitations = [
        "Embedding similarity is a proxy for timbre match, not proof of identity, "
        "consent or authorisation.",
        "Speaker-embedding systems are accent-biased: accented speakers tend to "
        "score lower than standard-accent speakers, so a lower score for an "
        "Indian-accented speaker may be partly an artefact.",
        "Absolute cosine scores are not portable between reference sets: on real read "
        "speech we measured different-speaker pairs as high as 0.945 and same-speaker "
        "pairs as low as 0.696. Read the calibrated score, not the raw cosine.",
    ]
    try:
        emb = embedder or get_embedder()
    except Exception as e:
        return SpeakerSimilarityResult(
            similarity=None,
            evaluator="none",
            note=str(e),
            limitations=limitations,
        )
    try:
        gen_vec = emb.embed(generated_path)
        ref_vecs: list[np.ndarray] = []
        sims: list[float] = []
        for ref in reference_paths:
            try:
                rvec = emb.embed(ref)
                ref_vecs.append(rvec)
                sims.append(round(cosine_similarity(gen_vec, rvec), 4))
            except Exception:
                continue
    except Exception as e:
        return SpeakerSimilarityResult(
            similarity=None,
            evaluator=getattr(emb, "name", "unknown"),
            note=f"Speaker-similarity evaluation failed: {e}",
            limitations=limitations,
        )
    if not sims:
        return SpeakerSimilarityResult(
            similarity=None,
            evaluator=getattr(emb, "name", "unknown"),
            note="No usable reference audio to compare against.",
            limitations=limitations,
        )

    # Same-speaker baseline from the user's own reference clips.
    baseline_vals: list[float] = []
    for i in range(len(ref_vecs)):
        for j in range(i + 1, len(ref_vecs)):
            baseline_vals.append(cosine_similarity(ref_vecs[i], ref_vecs[j]))
    baseline = round(statistics.mean(baseline_vals), 4) if baseline_vals else None
    similarity = round(sum(sims) / len(sims), 4)
    calibrated = round(similarity / baseline, 4) if baseline and baseline > 1e-6 else None
    if calibrated is None:
        note = (
            f"Cosine similarity vs {len(sims)} reference(s). Add a second reference "
            "clip to get a calibrated score (1.0 = as close as the speaker is to "
            "themselves)."
        )
    else:
        note = (
            f"Cosine {similarity} vs {len(sims)} reference(s); same-speaker baseline "
            f"{baseline} over {len(baseline_vals)} reference pair(s); calibrated score "
            f"{calibrated} (1.0 = as close to the speaker as the speaker is to "
            "themselves)."
        )
    return SpeakerSimilarityResult(
        similarity=similarity,
        evaluator=getattr(emb, "name", "unknown"),
        per_reference=sims,
        same_speaker_baseline=baseline,
        calibrated_score=calibrated,
        baseline_pairs=len(baseline_vals),
        note=note,
        limitations=limitations,
    )
