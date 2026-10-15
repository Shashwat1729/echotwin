"""Unit tests: language heuristics, style profile, privacy redaction."""

from __future__ import annotations

import json

from echotwin.ingestion.schema import Message
from echotwin.language import detect_language, detect_language_with_confidence
from echotwin.privacy.redact import RedactionConfig, redact_messages, redact_text
from echotwin.profiling.style_profile import build_style_profile, profile_prompt_fragment


class TestLanguage:
    def test_devanagari_detected(self):
        from echotwin.ingestion.schema import Language

        assert detect_language("आज मौसम अच्छा है") is Language.HI_DEVANAGARI

    def test_english(self):
        from echotwin.ingestion.schema import Language

        assert detect_language("Let's meet tomorrow at five") is Language.EN

    def test_romanized_hindi(self):
        from echotwin.ingestion.schema import Language

        assert detect_language("haan yaar main kal aa raha hoon") is Language.HI_LATN

    def test_mixed_script(self):
        from echotwin.ingestion.schema import Language

        text = "यह idea काफी अच्छा hai na"
        assert detect_language(text) is Language.HI_EN

    def test_empty_is_unknown(self):
        from echotwin.ingestion.schema import Language

        assert detect_language("") is Language.UNKNOWN

    def test_confidence_reported(self):
        _lang, conf, method = detect_language_with_confidence("haan yaar main kal aa raha hoon")
        assert 0.0 < conf <= 0.95
        assert method

    def test_never_invents_phonetics(self):
        lang, _c, _m = detect_language_with_confidence("some plain english text here")
        from echotwin.ingestion.schema import Language

        assert lang is Language.EN


class TestStyleProfile:
    def test_metrics_are_counts_not_psychology(self, parsed, target_id):
        profile = build_style_profile(parsed.messages, target_id)
        assert profile.n_messages > 0
        # forbidden psychological claims must appear only inside the disclaimer
        disclaimer = json.dumps(profile.limitations).lower()
        body = profile.model_copy(update={"limitations": {}}).model_dump_json().lower()
        for forbidden in ("extrovert", "introvert", "personality", "neurotic"):
            assert forbidden not in body, f"{forbidden} leaked into profile"
        assert "no psychological" in disclaimer  # disclaimer stays
        assert "personality" in disclaimer  # only in the disclaimer text

    def test_length_stats(self, parsed, target_id):
        profile = build_style_profile(parsed.messages, target_id)
        assert profile.length["chars_mean"] > 0
        assert profile.length["chars_median"] > 0

    def test_emoji_stats(self, parsed, target_id):
        profile = build_style_profile(parsed.messages, target_id)
        assert 0 <= profile.emoji["messages_with_emoji_pct"] <= 100

    def test_code_switching_metric(self, parsed, target_id):
        profile = build_style_profile(parsed.messages, target_id)
        assert "mixed_message_share_pct" in profile.code_switching

    def test_prompt_fragment(self, parsed, target_id):
        profile = build_style_profile(parsed.messages, target_id)
        frag = profile_prompt_fragment(profile)
        assert "mean" in frag and "emoji" in frag

    def test_save_load_roundtrip(self, parsed, target_id, tmp_path):
        from echotwin.profiling.style_profile import load_profile, save_profile

        profile = build_style_profile(parsed.messages, target_id)
        p = tmp_path / "profile.json"
        save_profile(profile, p)
        loaded = load_profile(p)
        assert loaded.n_messages == profile.n_messages
        assert loaded.limitations  # documented limits persist


class TestRedaction:
    def test_email_phone_url(self):
        text = "mail me at test.user@example.com or +91 98765 43210, see https://x.co/abc"
        out, report = redact_text(text, RedactionConfig(enabled=True))
        assert "test.user@example.com" not in out
        assert "+91 98765 43210" not in out
        assert "https://x.co/abc" not in out
        assert set(report.counts) == {"email", "phone", "url"}
        # counts only, values never echoed
        assert "example.com" not in report.model_dump_json()

    def test_disabled_by_default(self):
        text = "call me at 9876543210"
        out, report = redact_text(text, RedactionConfig())
        assert out == text and not report.redacted

    def test_selective_categories(self):
        text = "test@example.com 9876543210"
        out, _report = redact_text(text, RedactionConfig(enabled=True, categories=["email"]))
        assert "test@example.com" not in out
        assert "9876543210" in out

    def test_redact_messages(self, parsed, target_id):
        cfg = RedactionConfig(enabled=True, categories=["phone"])
        texts = ["call 9876543210 now", "no number here"]
        msgs = [
            Message.model_construct(message_id=str(i), conversation_id="c", text=t, sender_id=None)
            for i, t in enumerate(texts)
        ]
        out = redact_messages(msgs, cfg)
        assert "9876543210" not in out[0].text
        assert out[1].text == texts[1]
