"""Voice profiles: local metadata with managed identifiers (never raw FS paths in UIs).

A profile bundles backend identity, consent, preferences, recording inventory,
reference selection, quality/coverage reports, preprocessing config and
evaluation results. Deletion removes the profile and derived artifacts; the
`retrain-note` documents that deleting artifacts does not guarantee removal of
learned information from any adapted model weights.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..settings import get_settings


class RecordingEntry(BaseModel):
    recording_id: str
    path: str = Field(description="Absolute local path of the source recording")
    language: str
    script_id: str | None = None
    category: str = "uncategorized"
    has_transcript: bool = False
    transcript: str | None = None
    quality_verdict: str = "ok"
    duration_seconds: float = 0.0
    sample_rate: int = 0
    quality_report: dict | None = None
    transcript_check: dict | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class VoiceProfile(BaseModel):
    profile_id: str = Field(default_factory=lambda: f"voice-{uuid.uuid4().hex[:12]}")
    schema_version: str = "1.0"
    name: str = "My voice profile"
    backend: str = "voxcpm"
    backend_version: str = ""
    consent: dict = Field(default_factory=dict)
    languages: list[str] = Field(default_factory=lambda: ["hi", "en"])
    script_preference: str = "auto"
    speaker_profile: str = "bilingual"
    recordings: list[RecordingEntry] = Field(default_factory=list)
    reference_selection: dict | None = None
    combined_reference: str | None = Field(
        default=None, description="Derived artifact path if user explicitly combined clips"
    )
    preprocessing: dict = Field(default_factory=lambda: {"target_sr": 16000, "mono": True})
    coverage_report: dict | None = None
    evaluations: list[dict] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat()


def _profile_dir() -> Path:
    d = get_settings().artifacts_dir / "voice_profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(profile: VoiceProfile) -> Path:
    path = _profile_dir() / f"{profile.profile_id}.json"
    profile.touch()
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_profile(profile_id: str) -> VoiceProfile:
    path = _profile_dir() / f"{profile_id}.json"
    if not path.exists():
        from ..exceptions import ArtifactNotFoundError

        raise ArtifactNotFoundError(f"No voice profile '{profile_id}'")
    return VoiceProfile.model_validate_json(path.read_text(encoding="utf-8"))


def list_profiles() -> list[VoiceProfile]:
    out = []
    for p in sorted(_profile_dir().glob("voice-*.json")):
        try:
            out.append(VoiceProfile.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def delete_profile(
    profile_id: str, *, delete_recordings: bool = False, delete_audio_artifacts: bool = True
) -> dict:
    """Delete the profile. Optionally also delete source recordings and derived audio.

    Documented limitation: deleting files does NOT guarantee that information is
    removed from any fine-tuned model weights (voice LoRA/text adapters). Retrain
    from sanitized data to remove it.
    """
    profile = load_profile(profile_id)
    removed: list[str] = []
    if delete_audio_artifacts and profile.combined_reference:
        p = Path(profile.combined_reference)
        if p.exists():
            p.unlink()
            removed.append(str(p))
    if delete_recordings:
        for rec in profile.recordings:
            p = Path(rec.path)
            if p.exists():
                try:
                    p.unlink()
                    removed.append(str(p))
                except OSError:
                    pass
    (_profile_dir() / f"{profile_id}.json").unlink(missing_ok=True)
    return {
        "profile_id": profile_id,
        "files_removed": removed,
        "retrain_note": "Deleting artifacts does not guarantee removal of learned "
        "information from adapted model weights; retrain from sanitized data.",
    }


def export_profile(
    profile: VoiceProfile, out_path: str | Path, *, include_audio: bool = False
) -> Path:
    """Export profile metadata (and optionally copies of reference audio)."""
    import shutil

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.model_dump()
    if include_audio and profile.reference_selection:
        ids = [r.get("recording_id") for r in profile.reference_selection.get("selected", [])]
        audio_dir = out.parent / f"{out.stem}_audio"
        audio_dir.mkdir(exist_ok=True)
        copied = []
        for rec in profile.recordings:
            if rec.recording_id in ids and Path(rec.path).exists():
                target = audio_dir / Path(rec.path).name
                shutil.copy2(rec.path, target)
                copied.append(str(target))
        payload["exported_audio"] = copied
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def add_recording(profile: VoiceProfile, entry: RecordingEntry) -> RecordingEntry:
    profile.recordings.append(entry)
    return entry
