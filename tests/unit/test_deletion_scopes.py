"""Deletion scopes must remove what the UI promises, and nothing more.

These mirror the promises printed by the privacy CLI (see
privacy_exporter): voice deletion takes profile metadata + generated audio +
recordings; dataset deletion keeps nothing conversational; unknown scopes fail
loudly instead of silently deleting everything.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from echotwin.exceptions import ConfigurationError
from echotwin.privacy_exporter import delete_scope, export_all
from echotwin.settings import get_settings
from echotwin.storage import db
from echotwin.workspace import get_store, reset_store

QUIET = Console(quiet=True)


@pytest.fixture(autouse=True)
def isolated():
    """The config fixture (tests/conftest.py) already isolates ECHOTWIN_HOME."""
    yield
    reset_store()


def _seed_voice_artifacts():
    settings = get_settings()
    profile_dir = settings.artifacts_dir / "voice_profiles" / "p1"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text("{}", encoding="utf-8")
    recordings = settings.recordings_dir / "p1"
    recordings.mkdir(parents=True, exist_ok=True)
    (recordings / "take.wav").write_bytes(b"RIFF")
    artifact_id = db.record_artifact("voice_profile", profile_dir / "profile.json")
    artifact = db.list_artifacts("voice_profile")[0]
    return settings, artifact_id, artifact


class TestVoiceDeletion:
    def test_voice_scope_removes_profiles_and_recordings(self):
        settings, artifact_id, _artifact = _seed_voice_artifacts()
        store = get_store()
        store.set_pointer("current_voice_profile", "p1")

        delete_scope("artifacts:voice", QUIET)

        assert artifact_id  # recorded in the registry before deletion
        assert not (settings.artifacts_dir / "voice_profiles").exists()
        assert not (settings.artifacts_dir / "audio").exists()
        assert not (settings.recordings_dir / "p1").exists()
        assert store.pointer("current_voice_profile") is None
        assert db.list_artifacts("voice_profile") == []

    def test_voice_scope_honors_adapters_note_but_keeps_other_kinds(self):
        settings, _a, _b = _seed_voice_artifacts()
        datasets = settings.artifacts_dir / "datasets" / "d1"
        datasets.mkdir(parents=True, exist_ok=True)
        (datasets / "examples.jsonl").write_text("[]", encoding="utf-8")

        delete_scope("artifacts:voice", QUIET)

        # pure chat datasets are only removed by artifacts:datasets
        assert datasets.exists()


class TestDatasetDeletion:
    def test_datasets_scope_clears_raw_imports_and_pointers(self):
        settings = get_settings()
        (settings.artifacts_dir / "datasets" / "d1").mkdir(parents=True, exist_ok=True)
        imports = settings.artifacts_dir / "imports"
        imports.mkdir(parents=True, exist_ok=True)
        (imports / "w1.json").write_text("[]", encoding="utf-8")
        store = get_store()
        store.set_pointer("current_dataset", "d1")
        store.set_pointer("current_import", "w1")

        delete_scope("artifacts:datasets", QUIET)

        assert not (settings.artifacts_dir / "datasets").exists()
        assert not (settings.artifacts_dir / "imports").exists()
        assert store.pointer("current_dataset") is None
        assert store.pointer("current_import") is None


class TestFullReset:
    def test_all_state_removes_db_and_uploads(self):
        settings = get_settings()
        _s, _i, _a = _seed_voice_artifacts()
        upload = (settings.exports_dir / "upload.wav")
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(b"data")

        delete_scope("all-state", QUIET)

        assert not upload.exists()
        assert db_path_gone(settings)

    def test_all_state_keeps_no_personal_payload(self):
        settings = get_settings()
        _seed_voice_artifacts()
        upload = (settings.exports_dir / "upload.wav")
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(b"data")

        delete_scope("all-state", QUIET)

        survivors = [
            p for p in settings.exports_dir.glob("**/*") if p.is_file()
        ]
        assert survivors == [], f"personal files survived all-state reset: {survivors}"
        assert recordings_gone(settings)


class TestUnknownScope:
    def test_unknown_scope_raises_instead_of_deleting(self):
        settings = get_settings()
        _s, _i, _a = _seed_voice_artifacts()
        with pytest.raises(ConfigurationError) as e:
            delete_scope("everything", QUIET)
        assert "artifacts:voice" in str(e.value)
        # nothing was touched
        assert (settings.artifacts_dir / "voice_profiles" / "p1").exists()


class TestProfileScopeKeepsRecordings:
    def test_profile_delete_keeps_recordings_when_asked(self):
        settings, _i, _a = _seed_voice_artifacts()
        from echotwin.voice.metadata import VoiceProfile, save_profile

        save_profile(VoiceProfile(profile_id="p1", name="Test voice"))
        delete_scope("profile:p1", QUIET)
        assert (settings.recordings_dir / "p1" / "take.wav").exists()


class TestExportHonesty:
    def test_export_note_mentions_datasets_not_raw_whatsapp(self):
        out = get_settings().home / "full-export"
        export_all(out, QUIET)
        state = json.loads((out / "echotwin-state.json").read_text(encoding="utf-8"))
        note = state["contents_note"]
        assert "DOES contain your own words" in note
        assert "not copied here" in note


def db_path_gone(settings) -> bool:
    return not (settings.home / "echotwin.sqlite3").exists()


def recordings_gone(settings) -> bool:
    recordings = settings.recordings_dir
    return not recordings.exists() or not any(recordings.iterdir())
