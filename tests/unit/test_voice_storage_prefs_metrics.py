"""Unit tests: reference selection, voice profiles, storage sandbox, preferences, metrics."""

from __future__ import annotations

import pytest

from echotwin.exceptions import ArtifactNotFoundError, PathTraversalError, UploadValidationError
from echotwin.preference.dpo import PreferencePair, load_pairs, save_pairs, validate_pairs
from echotwin.settings import get_settings
from echotwin.storage import db, state
from echotwin.storage.paths import ensure_within, home_summary, unique_path, validate_upload
from echotwin.voice.capabilities import VoiceBackendCapabilities
from echotwin.voice.metadata import (
    VoiceProfile,
    delete_profile,
    export_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from echotwin.voice.reference_selection import ReferenceCandidate, select_references


class TestReferenceSelection:
    def test_single_reference_per_language(self):
        caps = VoiceBackendCapabilities(
            name="t", languages=["hi", "en"], supports_multiple_references=False
        )
        cands = [
            ReferenceCandidate(
                recording_id="a", language="hi", quality_verdict="ok", duration_seconds=10
            ),
            ReferenceCandidate(
                recording_id="b", language="hi", quality_verdict="ok", duration_seconds=12
            ),
            ReferenceCandidate(
                recording_id="c", language="en", quality_verdict="ok", duration_seconds=9
            ),
        ]
        d = select_references(cands, caps)
        # single-reference backend -> exactly ONE clip is used (the best: b)
        assert len(d.selected) == 1
        assert d.selected[0].recording_id == "b"
        # ...but per_language still maps the best clip for EVERY language, so
        # the language-aware reference pick in synthesis.py can choose the
        # English clip when English is requested. Previously the other
        # language's clip appeared nowhere at all.
        assert d.per_language.get("hi") == "b"
        assert d.per_language.get("en") == "c"
        assert d.alternates  # the rest remain alternates

    def test_multiple_references_when_supported(self):
        caps = VoiceBackendCapabilities(
            name="t", languages=["hi", "en"], supports_multiple_references=True
        )
        cands = [
            ReferenceCandidate(
                recording_id="a", language="hi", quality_verdict="ok", duration_seconds=10
            ),
            ReferenceCandidate(
                recording_id="b", language="hi", quality_verdict="ok", duration_seconds=12
            ),
            ReferenceCandidate(
                recording_id="c", language="en", quality_verdict="ok", duration_seconds=9
            ),
            ReferenceCandidate(
                recording_id="d", language="en", quality_verdict="ok", duration_seconds=11
            ),
        ]
        d = select_references(cands, caps)
        assert d.per_language.get("hi") == "b"  # longer breaks the tie
        assert d.per_language.get("en") == "d"
        assert len(d.selected) == 2

    def test_quality_failure_rejected(self):
        caps = VoiceBackendCapabilities(name="t", languages=["hi"])
        cands = [
            ReferenceCandidate(
                recording_id="bad", language="hi", quality_verdict="fail", duration_seconds=10
            )
        ]
        d = select_references(cands, caps)
        assert not d.selected
        assert any(r["reason"] == "quality_failed" for r in d.rejected)

    def test_too_short_rejected(self):
        caps = VoiceBackendCapabilities(name="t", languages=["hi"], min_reference_seconds=5.0)
        cands = [
            ReferenceCandidate(
                recording_id="x", language="hi", quality_verdict="ok", duration_seconds=0.5
            )
        ]
        d = select_references(cands, caps)
        assert any(r["reason"] == "duration_too_short" for r in d.rejected)

    def test_transcript_preferred_when_requested(self):
        caps = VoiceBackendCapabilities(
            name="t", languages=["hi"], reference_transcript_optional=True
        )
        cands = [
            ReferenceCandidate(
                recording_id="plain", language="hi", quality_verdict="ok", duration_seconds=10
            ),
            ReferenceCandidate(
                recording_id="witht",
                language="hi",
                quality_verdict="ok",
                duration_seconds=10,
                has_transcript=True,
                transcript_verdict="match",
            ),
        ]
        d = select_references(cands, caps, prefer_transcript=True)
        assert d.selected[0].recording_id == "witht"
        assert d.transcript_available


class TestVoiceProfiles:
    def test_save_load_list_delete(self):
        prof = VoiceProfile(name="test", backend="voxcpm")
        save_profile(prof)
        profiles, corrupt = list_profiles()
        assert not corrupt
        assert any(p.profile_id == prof.profile_id for p in profiles)
        loaded = load_profile(prof.profile_id)
        assert loaded.backend == "voxcpm"
        export_path = get_settings().home / "export.json"
        export_profile(loaded, export_path)
        assert export_path.exists()
        result = delete_profile(prof.profile_id)
        assert "retrain_note" in result
        with pytest.raises(ArtifactNotFoundError):
            load_profile(prof.profile_id)

    def test_managed_ids_not_fs_paths(self):
        prof = VoiceProfile()
        assert prof.profile_id.startswith("voice-")


class TestStorage:
    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(PathTraversalError):
            ensure_within(tmp_path, "../outside.txt")
        with pytest.raises(PathTraversalError):
            ensure_within(tmp_path, "C:\\windows\\system32\\drivers\\etc\\hosts")

    def test_relative_path_allowed(self, tmp_path):
        (tmp_path / "sub").mkdir()
        p = ensure_within(tmp_path, "sub/file.txt")
        assert str(p).startswith(str(tmp_path))

    def test_upload_validation(self):
        validate_upload("a.txt", 100)
        with pytest.raises(UploadValidationError):
            validate_upload("evil.exe", 100)
        settings = get_settings()
        with pytest.raises(UploadValidationError):
            validate_upload("a.wav", settings.max_upload_bytes + 1)

    def test_unique_path(self, tmp_path):
        p1 = unique_path(tmp_path, "x", ".txt")
        p1.write_text("a")
        p2 = unique_path(tmp_path, "x", ".txt")
        assert p1 != p2

    def test_db_artifacts_crud(self, tmp_path):
        p = tmp_path / "art.json"
        p.write_text("{}")
        aid = db.record_artifact("test", p, {"k": 1})
        arts = db.list_artifacts("test")
        assert any(a["artifact_id"] == aid for a in arts)
        res = db.delete_artifact(aid)
        assert res["file_deleted"]

    def test_consent_flow(self):
        assert not db.get_consent("unit_test")["granted"]
        db.record_consent("unit_test", True, {"who": "tester"})
        assert db.get_consent("unit_test")["granted"]
        db.record_consent("unit_test", False, {})
        assert not db.get_consent("unit_test")["granted"]

    def test_state_roundtrip(self):
        state.set("unit", 42)
        assert state.get("unit") == 42

    def test_home_summary(self):
        info = home_summary()
        assert "home" in info and info["writable"] in {True, False}


class TestPreferences:
    def test_valid_pair(self):
        pairs = [
            PreferencePair(
                context=[{"role": "user", "content": "hi"}],
                chosen="a",
                rejected="b",
                provenance="human",
            )
        ]
        v = validate_pairs(pairs)
        assert v.valid == 1 and v.human_share == 1.0

    def test_empty_rejected(self):
        pairs = [PreferencePair(context=[], chosen="", rejected="b", provenance="human")]
        v = validate_pairs(pairs)
        assert v.valid == 0
        assert v.errors

    def test_same_chosen_rejected_rejected(self):
        pairs = [
            PreferencePair(
                context=[{"role": "user", "content": "x"}],
                chosen="same",
                rejected="same",
                provenance="human",
            )
        ]
        v = validate_pairs(pairs)
        assert any("chosen equals rejected" in e for e in v.errors)

    def test_provenance_required(self):
        pairs = [
            PreferencePair(
                context=[{"role": "user", "content": "x"}], chosen="a", rejected="b", provenance=""
            )
        ]
        v = validate_pairs(pairs)
        assert any("missing provenance" in e for e in v.errors)

    def test_roundtrip(self, tmp_path):
        pairs = [
            PreferencePair(
                context=[{"role": "user", "content": "hi"}],
                chosen="a",
                rejected="b",
                provenance="synthetic-demo",
            )
        ]
        p = tmp_path / "pairs.jsonl"
        save_pairs(pairs, p)
        loaded = load_pairs(p)
        assert loaded[0].provenance == "synthetic-demo"


class TestMetricsBasics:
    def test_identical_texts(self):
        from echotwin.evaluation.metrics import char_ngram_cosine, compare_style

        s = compare_style(["hello there friend"], ["hello there friend"])
        assert s.char_ngram_cosine_3 == 1.0
        assert s.length_mean_ratio == 1.0
        assert char_ngram_cosine(["abc"], ["abc"]) == 1.0

    def test_different_texts_distance(self):
        from echotwin.evaluation.metrics import compare_style

        s = compare_style(
            ["haan yaar kal milte h"],
            ["please confirm the meeting schedule"],
        )
        assert s.char_ngram_cosine_3 < 0.5
        assert s.limitations

    def test_phrase_reproduction_not_generalization_claim(self):
        from echotwin.evaluation.metrics import top_phrase_reproduction

        r = top_phrase_reproduction(
            ["haan yaar kal milte h", "haan yaar kal milte h"], ["haan yaar kal"]
        )
        assert "reproduced" in r
        assert "stylometric" in r["method"] or "reproduction" in r["method"]

    def test_emoji_rates(self):
        from echotwin.evaluation.metrics import emoji_rate

        assert emoji_rate(["ok 👍"]) == 1.0
        assert emoji_rate(["ok"]) == 0.0
