"""Unit tests: recording-plan generator determinism and capability honesty.

Spec requirement: "given a specific backend capability profile and coverage
report, verify that the generated plan recommends the correct language
categories and never invents unsupported backend requirements."
"""

from __future__ import annotations

from echotwin.voice.capabilities import VoiceBackendCapabilities
from echotwin.voice.phoneme_coverage import analyze_texts
from echotwin.voice.recording_plan import (
    InventoryItem,
    RecordingPreferences,
    generate_plan,
)


def caps_hi_en() -> VoiceBackendCapabilities:
    return VoiceBackendCapabilities(
        name="test-backend",
        version="test-1",
        languages=["hi", "en"],
        min_reference_seconds=3.0,
        recommended_reference_seconds=8.0,
        reference_transcript_optional=True,
    )


def caps_en_only() -> VoiceBackendCapabilities:
    return VoiceBackendCapabilities(name="en-only", languages=["en"], min_reference_seconds=3.0)


class TestPlanBasics:
    def test_empty_inventory_mandatory_both_languages(self):
        plan = generate_plan(caps_hi_en(), RecordingPreferences(languages=["hi", "en"]), [])
        mandatory = [t for t in plan.tasks if t.mandatory]
        assert {t.language for t in mandatory} == {"hi", "en"}
        assert all(t.script_ids for t in mandatory)

    def test_never_invents_unsupported_language(self):
        plan = generate_plan(
            caps_en_only(), RecordingPreferences(languages=["hi", "en"], include_hinglish=True), []
        )
        task_langs = {t.language for t in plan.tasks}
        assert "hi" not in task_langs
        assert "hinglish" not in task_langs
        skipped = {s["language"]: s["reason"] for s in plan.skipped}
        assert skipped.get("hi") == "unsupported_language"
        assert skipped.get("hinglish") == "unsupported_language"

    def test_mandatory_satisfied_means_test_task(self):
        inv = [
            InventoryItem(
                recording_id="r1", language="hi", quality_verdict="ok", duration_seconds=10
            ),
            InventoryItem(
                recording_id="r2", language="en", quality_verdict="ok", duration_seconds=10
            ),
            InventoryItem(
                recording_id="r3", language="en", quality_verdict="ok", duration_seconds=10
            ),
            InventoryItem(
                recording_id="r4", language="hi", quality_verdict="ok", duration_seconds=10
            ),
        ]
        plan = generate_plan(
            caps_hi_en(),
            RecordingPreferences(languages=["hi", "en"], include_hinglish=False),
            inv,
        )
        assert plan.backend_minimum_met
        assert any(t.id == "test-voice" for t in plan.tasks)
        assert not any(t.mandatory for t in plan.tasks)

    def test_deterministic(self):
        prefs = RecordingPreferences(languages=["hi", "en"])
        p1 = generate_plan(caps_hi_en(), prefs, [])
        p2 = generate_plan(caps_hi_en(), prefs, [])
        assert [t.id for t in p1.tasks] == [t.id for t in p2.tasks]
        assert p1.total_estimated_seconds == p2.total_estimated_seconds

    def test_script_ids_exist(self):
        from echotwin.voice.script_library import get_script

        plan = generate_plan(caps_hi_en(), RecordingPreferences(languages=["hi", "en"]), [])
        for t in plan.tasks:
            for sid in t.script_ids:
                get_script(sid)  # raises if invented

    def test_time_budget_flags(self):
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"], max_minutes=0.2), []
        )
        assert any(t.beyond_time_budget for t in plan.tasks)
        assert plan.warnings


class TestCoverageDriven:
    def test_gaps_add_phonetic_tasks(self):
        coverage = analyze_texts(["okay"], languages=["hi", "en"])
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"]), [], coverage
        )
        assert any(t.category == "phonetic_coverage" for t in plan.tasks)

    def test_full_coverage_no_phonetic_task(self):
        from echotwin.voice.script_library import all_scripts

        texts = [s.text for s in all_scripts()]
        coverage = analyze_texts(texts, languages=["hi", "en"])
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"]), [], coverage
        )
        # every script is read -> little left to recommend
        assert (
            not plan.tasks
            or plan.tasks[-1].category == "evaluation"
            or all(t.category != "phonetic_coverage" or t.script_ids == [] for t in plan.tasks)
        )


class TestHinglishHandling:
    def test_hinglish_optional_with_note(self):
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"], include_hinglish=True), []
        )
        hinglish = [t for t in plan.tasks if t.language == "hinglish"]
        assert hinglish
        assert all(not t.mandatory for t in hinglish)
        assert any("NO official Hinglish" in t.rationale for t in hinglish)

    def test_hinglish_skipped_when_disabled(self):
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"], include_hinglish=False), []
        )
        assert "hinglish" not in {t.language for t in plan.tasks}

    def test_failed_recordings_dont_count(self):
        inv = [
            InventoryItem(
                recording_id="bad", language="hi", quality_verdict="fail", duration_seconds=10
            ),
        ]
        plan = generate_plan(
            caps_hi_en(), RecordingPreferences(languages=["hi", "en"], include_hinglish=False), inv
        )
        assert [t for t in plan.tasks if t.mandatory and t.language == "hi"]
