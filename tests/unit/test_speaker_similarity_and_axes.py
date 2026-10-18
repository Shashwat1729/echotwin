"""Unit tests: speaker-similarity evaluator + the new evaluation axes.

The embedder is stubbed, so these tests never download a model. The stub
encodes the *measured* behaviour of the real evaluators: same-speaker clips
have near-identical embeddings, different speakers have near-orthogonal ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echotwin.evaluation.metrics import prompt_discrimination, unsupported_claims
from echotwin.voice.speaker_similarity import (
    SPEAKER_SIMILARITY_THRESHOLD,
    X_VECTOR_THRESHOLD,
    ECAPASpeakerEmbedder,
    SpeakerEmbedder,
    available_embedders,
    cosine_similarity,
    evaluate_speaker_similarity,
    get_embedder,
)


class StubEmbedder:
    """Returns a fixed vector per 'speaker', so cosine behaves predictably."""

    name = "stub"

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, audio_path) -> np.ndarray:
        key = str(audio_path)
        if not Path(key).exists():
            # Real embedders raise on a missing file; the evaluator must skip it.
            raise FileNotFoundError(key)
        for tag, vec in self.vectors.items():
            if tag in key:
                return vec
        return np.zeros(4, dtype=np.float32)


def _stub() -> SpeakerEmbedder:
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return StubEmbedder({"same": a, "other": b})


class TestCosine:
    def test_identical_is_one(self):
        v = np.array([0.3, 0.4], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_is_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_degenerate_input_is_zero(self):
        z = np.zeros(4, dtype=np.float32)
        assert cosine_similarity(z, z) == 0.0
        assert cosine_similarity(z, np.ones(4, dtype=np.float32)) == 0.0

    def test_size_mismatch_is_zero(self):
        a = np.ones(3, dtype=np.float32)
        b = np.ones(4, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


class TestEvaluateSpeakerSimilarity:
    def test_same_speaker_scores_high(self, tmp_path):
        refs = []
        for i in range(2):
            p = tmp_path / f"same-{i}.wav"
            p.write_bytes(b"x")
            refs.append(p)
        held = tmp_path / "same-held.wav"
        held.write_bytes(b"x")
        res = evaluate_speaker_similarity(held, refs, embedder=_stub())
        assert res.similarity == pytest.approx(1.0, abs=1e-5)
        assert res.similarity >= SPEAKER_SIMILARITY_THRESHOLD
        assert res.calibrated_score == pytest.approx(1.0, abs=0.05)
        assert res.same_speaker_baseline == pytest.approx(1.0, abs=1e-5)
        assert res.baseline_pairs == 1

    def test_different_speaker_scores_low(self, tmp_path):
        ref = tmp_path / "same-0.wav"
        ref.write_bytes(b"x")
        other = tmp_path / "other.wav"
        other.write_bytes(b"x")
        res = evaluate_speaker_similarity(other, [ref], embedder=_stub())
        assert res.similarity == pytest.approx(0.0, abs=1e-6)
        assert res.similarity < SPEAKER_SIMILARITY_THRESHOLD

    def test_single_reference_has_no_calibration(self, tmp_path):
        ref = tmp_path / "same-0.wav"
        ref.write_bytes(b"x")
        held = tmp_path / "same-held.wav"
        held.write_bytes(b"x")
        res = evaluate_speaker_similarity(held, [ref], embedder=_stub())
        assert res.similarity is not None
        assert res.calibrated_score is None
        assert res.same_speaker_baseline is None
        assert "second reference" in res.note

    def test_missing_reference_is_skipped_not_fatal(self, tmp_path):
        ref = tmp_path / "same-0.wav"
        ref.write_bytes(b"x")
        held = tmp_path / "same-held.wav"
        held.write_bytes(b"x")
        res = evaluate_speaker_similarity(held, [ref, tmp_path / "nope.wav"], embedder=_stub())
        assert res.similarity is not None
        assert len(res.per_reference) == 1

    def test_no_references_is_none(self, tmp_path):
        held = tmp_path / "same-held.wav"
        held.write_bytes(b"x")
        res = evaluate_speaker_similarity(held, [], embedder=_stub())
        assert res.similarity is None
        assert "No usable reference" in res.note

    def test_limitations_state_accent_and_portability(self, tmp_path):
        ref = tmp_path / "same-0.wav"
        ref.write_bytes(b"x")
        held = tmp_path / "same-held.wav"
        held.write_bytes(b"x")
        res = evaluate_speaker_similarity(held, [ref], embedder=_stub())
        blob = " ".join(res.limitations).lower()
        assert "accent" in blob
        assert "not portable" in blob
        assert "not proof of identity" in blob


class TestEvaluatorSelection:
    def test_thresholds_documented(self):
        assert SPEAKER_SIMILARITY_THRESHOLD == 0.25
        assert X_VECTOR_THRESHOLD == 0.7

    def test_ecapa_is_preferred(self):
        try:
            emb = get_embedder()
        except Exception:
            pytest.skip("no speaker embedder installed")
        assert emb.name == "ecapa-tdnn"
        assert isinstance(emb, ECAPASpeakerEmbedder)

    def test_available_embedders_lists_installed(self):
        names = available_embedders()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


class TestPromptDiscrimination:
    def test_all_identical_is_flagged(self):
        r = prompt_discrimination(["same"] * 4)
        assert r["distinct_ratio"] == 0.25
        assert r["most_common_count"] == 4
        assert r["most_common_share"] == 1.0

    def test_all_distinct_is_good(self):
        r = prompt_discrimination(["a", "b", "c", "d"])
        assert r["distinct_ratio"] == 1.0
        assert r["most_common_count"] == 1

    def test_empty_is_safe(self):
        r = prompt_discrimination([])
        assert r["n"] == 0
        assert r["distinct_ratio"] is None

    def test_note_states_limits(self):
        r = prompt_discrimination(["a"])
        assert "not" in r["note"].lower()


class TestUnsupportedClaims:
    def test_number_not_in_context_is_flagged(self):
        r = unsupported_claims(["call me on 9876543210"], ["no numbers here"])
        assert r["n_unsupported"] == 1
        assert r["examples"][0]["token"] == "9876543210"

    def test_number_present_in_context_is_not_flagged(self):
        r = unsupported_claims(["call me on 9876543210"], ["my number is 9876543210"])
        assert r["n_unsupported"] == 0

    def test_known_term_is_not_flagged(self):
        r = unsupported_claims(["my number is 9876543210"], [], known_terms=["9876543210"])
        assert r["n_unsupported"] == 0

    def test_handle_and_url_are_checked(self):
        r = unsupported_claims(["ping @someone or https://x.co/a"], ["nothing"])
        assert r["n_unsupported"] == 2

    def test_plain_words_are_not_checked(self):
        r = unsupported_claims(["haan yaar kal milte h"], ["x"])
        assert r["n_entities_checked"] == 0
        assert r["n_unsupported"] == 0

    def test_empty_is_safe(self):
        r = unsupported_claims([], [])
        assert r["n_entities_checked"] == 0
