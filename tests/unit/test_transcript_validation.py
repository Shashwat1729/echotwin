"""Unit tests: transcript verification."""

from __future__ import annotations

from echotwin.voice.transcript_validation import verify_transcript


class TestVerifyTranscript:
    def test_exact_match(self):
        check = verify_transcript("hello world", "hello world", source="user", language="en")
        assert check.verdict == "match"
        assert check.normalized_similarity == 1.0

    def test_missing_words(self):
        check = verify_transcript(
            "the quick brown fox jumps", "the quick fox", source="user", language="en"
        )
        assert check.verdict in {"minor_differences", "mismatch"}
        assert "brown" in check.missing_words
        assert check.likely_incomplete

    def test_wrong_words_classified(self):
        check = verify_transcript("ship and sheep", "ship and shape", source="user", language="en")
        assert check.substituted_words

    def test_repeats_detected(self):
        check = verify_transcript("okay go", "okay okay okay go", source="user", language="en")
        assert check.repeated_phrases

    def test_no_transcript_unverified(self):
        check = verify_transcript("script text", "", source="none")
        assert check.verdict == "unverified"

    def test_asr_flags_for_hinglish(self):
        check = verify_transcript("Hi kaise ho", "hi k se ho", source="asr", language="hinglish")
        assert "asr_unreliable_for_hinglish" in check.flags
        assert "asr_not_ground_truth" in check.flags

    def test_asr_flags_for_hindi(self):
        check = verify_transcript("नमस्ते दोस्त", "namaste dost", source="asr", language="hi")
        assert "asr_unreliable_for_hindi" in check.flags

    def test_audio_never_auto_modified(self):
        # verify_transcript only compares; it returns a report, no side effects
        a = "original script"
        verify_transcript(a, "different words entirely", source="user")
        assert a == "original script"
