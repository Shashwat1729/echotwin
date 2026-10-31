"""Unit tests: script library integrity + phonetic coverage."""

from __future__ import annotations

from echotwin.voice.phoneme_coverage import (
    CoverageReport,
    analyze_texts,
    missing_categories,
    recommend_scripts_for_gaps,
)
from echotwin.voice.script_library import (
    SCRIPT_LIBRARY,
    ScriptLanguage,
    all_scripts,
    get_script,
    render,
)


class TestScriptLibrary:
    def test_unique_ids(self):
        assert len({s.id for s in SCRIPT_LIBRARY}) == len(SCRIPT_LIBRARY)

    def test_all_languages_present(self):
        langs = {s.language for s in SCRIPT_LIBRARY}
        assert ScriptLanguage.HI in langs
        assert ScriptLanguage.EN in langs
        assert ScriptLanguage.HINGLISH in langs

    def test_no_placeholder_sentences(self):
        # every script is a real sentence (>= 8 chars, real words)
        for s in SCRIPT_LIBRARY:
            assert len(s.text) > 8, s.id
            assert not s.text.startswith("TODO"), s.id

    def test_clean_reference_scripts_have_name_field(self):
        for s in SCRIPT_LIBRARY:
            if s.category.value == "clean_reference":
                assert "[NAME]" in s.text, s.id

    def test_dates_not_fabricated(self):
        entry = get_script("hi-date-time")
        rendered = render(entry, {})
        assert "__________" in rendered  # unset placeholders render as blanks
        assert "2025" not in rendered  # no fabricated date
        rendered2 = render(entry, {"DATE": "चौदह अक्टूबर", "TIME": "सात"})
        assert "चौदह अक्टूबर" in rendered2
        assert "सात" in rendered2
        assert "__________" not in rendered2

    def test_devanagari_scripts_are_devanagari(self):
        for s in scripts_hi():
            assert s.script == "devanagari"
            assert any("\u0900" <= c <= "\u097f" for c in s.text)

    def test_hinglish_scripts_are_mixed_latin(self):
        from echotwin.language import _ROMANIZED_HI_MARKERS

        en_words = {
            "actually",
            "but",
            "so",
            "that",
            "honestly",
            "there",
            "is",
            "hello",
            "hi",
            "the",
            "a",
            "an",
            "and",
            "for",
            "let",
            "will",
            "can",
            "need",
            "no",
            "not",
            "time",
            "next",
            "step",
            "plan",
            "option",
            "first",
            "message",
            "call",
            "wait",
            "sense",
            "makes",
            "compare",
            "rush",
            "check",
            "minute",
            "reply",
            "meeting",
            "meet",
            "today",
            "now",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "free",
            "urgently",
            "just",
            "long",
            "drive",
            "promise",
            "friday",
            "morning",
            "evening",
            "done",
            "details",
            "decide",
            "decided",
            "balcony",
            "enjoy",
            "walk",
            "overall",
            "thank",
            "you",
            "help",
        }
        for s in all_scripts():
            if s.language is ScriptLanguage.HINGLISH and s.category.value != "master_take":
                words = {w.strip(".,?!").lower() for w in s.text.split()}
                hi = {w for w in words if w in _ROMANIZED_HI_MARKERS}
                en = words & en_words
                if s.category.value == "numbers":
                    # numbers drills are pure by design (phonetic drill, not conversation)
                    assert len(hi) >= 2, f"{s.id}: hi={hi}"
                else:
                    assert len(hi) >= 2 and len(en) >= 1, f"{s.id}: hi={hi} en={en}"
                assert all(ord(c) < 0x900 for c in s.text), s.id

    def test_master_take_covers_all_languages_with_mood_cues(self):
        from echotwin.voice.script_library import MASTER_TAKE_ID, get_script

        text = get_script(MASTER_TAKE_ID).text
        assert any("ऀ" <= c <= "ॿ" for c in text), "needs Hindi (Devanagari)"
        assert any(w in text.lower() for w in ("hai", "hoon", "yaar")), "needs Hinglish"
        assert any(w in text.lower() for w in (" the ", " you ", " and ")), "needs English"
        cues = [ln for ln in text.splitlines() if ln.startswith("[") and ln.endswith("]")]
        assert len(cues) >= 5, "needs mood cues"

    def test_phonetic_categories_covered(self):
        cats = {s.category.value for s in SCRIPT_LIBRARY}
        for needed in (
            "phonetic_hi_dental_retroflex",
            "phonetic_hi_aspiration",
            "phonetic_en_vowels",
            "phonetic_en_clusters",
            "numbers",
            "question",
            "short_voice_note",
            "long_paragraph",
        ):
            assert needed in cats, needed

    def test_estimate_time(self):
        assert sum(s.est_seconds for s in SCRIPT_LIBRARY) > 60

    def test_unknown_script_id_raises(self):
        import pytest

        with pytest.raises(KeyError):
            get_script("nope-999")


def scripts_hi():
    return [s for s in all_scripts() if s.language is ScriptLanguage.HI]


class TestPhonemeCoverage:
    def test_devanagari_classes_detected(self):
        report = analyze_texts(["टमाटर तीन तब डर थाली"], languages=["hi"])
        cats = report.categories
        assert cats["hi_dental"].covered
        assert cats["hi_retroflex"].covered
        assert cats["hi_aspirated"].covered
        assert cats["hi_dental"].confidence == "grapheme_exact_text_only"

    def test_english_classes_detected(self):
        report = analyze_texts(["street splash scream threshold"], languages=["en"])
        cats = report.categories
        assert cats["en_clusters"].covered
        assert cats["en_vowels"].covered

    def test_missing_reported_not_invented(self):
        report = analyze_texts(["okay"], languages=["en"])
        assert "hi_dental" in missing_categories(report)
        # missing hindi categories marked unknown-not-requested
        assert report.categories["hi_dental"].confidence == "unknown"

    def test_confidence_flag_for_english_is_approximate(self):
        report = analyze_texts(["ship sheep seat"], languages=["en"])
        assert report.categories["en_vowels"].confidence == "approximate_grapheme"

    def test_limitations_stated(self):
        report = analyze_texts(["hello world"], languages=["en"])
        assert report.limitations
        assert any("NOT" in lim or "not" in lim for lim in report.limitations)

    def test_recommend_scripts_for_gaps(self):
        report = analyze_texts(["okay"], languages=["en"])
        recs = recommend_scripts_for_gaps(report, "en", limit=3)
        assert all(r.language.value == "en" for r in recs)
        assert recs

    def test_prosody_detection(self):
        report = analyze_texts(["Are you coming? Yes I am."], languages=["en"])
        assert report.categories["prosody_question"].covered
        assert report.categories["prosody_statement"].covered

    def test_romanized_hindi_approximate_coverage(self):
        report = analyze_texts(
            ["Yaar, mujhe lagta hai ki khana kha liya, ghar pe sab theek hai"],
            languages=["hinglish"],
        )
        cats = report.categories
        assert cats["hi_aspirated"].covered
        assert cats["hi_aspirated"].confidence == "approximate_romanized"
        assert cats["hi_nasals"].covered
        # dental vs retroflex is never claimed from Latin script
        assert not cats["hi_dental"].covered
        assert not cats["hi_retroflex"].covered

    def test_romanized_needs_markers(self):
        report = analyze_texts(
            ["Please confirm the meeting schedule for tomorrow"], languages=["hinglish"]
        )
        assert not report.categories["hi_aspirated"].covered

    def test_roundtrip_model(self):
        report = analyze_texts(["test sentence"], languages=["en"])
        restored = CoverageReport(**report.model_dump())
        assert restored.categories.keys() == report.categories.keys()
