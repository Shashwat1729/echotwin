"""Voice use-cases: recording intake, analysis, reference selection, consent, synthesis.

The Store pointer ``current_voice_profile`` names the profile every call works on.
Synthesis stays behind the consent gate enforced in ``voice.synthesis``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ArtifactNotFoundError, UploadValidationError
from ..settings import get_settings
from ..storage.paths import unique_path
from ..voice.audio_ingestion import read_audio
from ..voice.audio_quality import QualityThresholds, analyze_audio_file, summary_line
from ..voice.capabilities import VoiceBackendCapabilities
from ..voice.consent import (
    SCOPE_VOICE,
    VOICE_CONSENT_STATEMENT,
    ConsentRecord,
    check_consent,
)
from ..voice.consent import (
    grant_consent as _grant,
)
from ..voice.consent import (
    revoke_consent as _revoke,
)
from ..voice.metadata import RecordingEntry, VoiceProfile, load_profile, save_profile
from ..voice.phoneme_coverage import (
    CoverageReport,
    analyze_texts,
    coverage_texts_for_recordings,
)
from ..voice.recording_plan import (
    InventoryItem,
    RecordingPlan,
    RecordingPreferences,
    generate_plan,
)
from ..voice.reference_selection import ReferenceCandidate, select_references
from ..voice.registry import get_backend
from ..voice.script_library import all_scripts
from ..voice.synthesis import SynthesisResult
from ..voice.synthesis import synthesize as _synthesize
from ..voice.transcript_validation import TranscriptCheck, verify_transcript
from ..workspace import get_store

POINTER = "current_voice_profile"


@dataclass(frozen=True)
class RecordingResult:
    profile_id: str
    recording_id: str
    verdict: str
    summary: str


def current_profile() -> VoiceProfile | None:
    pid = get_store().pointer(POINTER)
    return load_profile(pid) if pid else None


def _require_profile() -> VoiceProfile:
    prof = current_profile()
    if prof is None:
        raise ArtifactNotFoundError("No voice profile yet. Save a recording first.")
    return prof


def inventory_from_profile(prof: VoiceProfile | None) -> list[InventoryItem]:
    """Build the recording inventory the planner needs from a voice profile."""
    if prof is None:
        return []
    return [
        InventoryItem(
            recording_id=r.recording_id,
            language=r.language,
            quality_verdict=r.quality_verdict,
            duration_seconds=r.duration_seconds,
            category=r.category,
            has_transcript=r.has_transcript,
        )
        for r in prof.recordings
    ]


def plan(
    backend: str, languages: list[str], max_minutes: float, single_take: bool = False
) -> RecordingPlan:
    """Generate a recording plan that reflects what the user already recorded.

    Passing an empty inventory here would tell a user with three good
    references to start over, so the plan is always built from the current
    profile when one exists.
    """
    caps = VoiceBackendCapabilities(**get_backend(backend).capabilities())
    prefs = RecordingPreferences(
        languages=languages, max_minutes=float(max_minutes), single_take=single_take
    )
    return generate_plan(caps, prefs, inventory_from_profile(current_profile()))


def _save_long_take(target: Path, clip, language: str, script_id: str) -> RecordingResult:
    """Cut a long continuous take at pauses into reference-sized clips.

    When the script that was read is known, each clip gets its matching text
    (script-aligned, not ASR-verified, and labelled as such).
    """
    import numpy as np

    from ..voice.preprocessing import write_wav
    from ..voice.segmentation import assign_transcripts, normalize_peak, split_on_pauses

    x = clip.samples if clip.samples.ndim == 1 else clip.samples.mean(axis=1)
    x = normalize_peak(np.asarray(x, dtype=np.float32))
    segments = split_on_pauses(x, clip.sample_rate)
    if not segments:
        target.unlink(missing_ok=True)
        raise UploadValidationError("No speech found in the recording.")
    script_text = ""
    if script_id:
        entry = next((s for s in all_scripts() if s.id == script_id), None)
        script_text = entry.text if entry else ""
    texts = assign_transcripts(segments, script_text) if script_text else [""] * len(segments)

    prof = current_profile() or VoiceProfile()
    worst = "ok"
    for n, (seg, text) in enumerate(zip(segments, texts, strict=True), 1):
        seg_path = target.with_name(f"{target.stem}_clip{n:02d}.wav")
        write_wav(seg_path, x[seg.start : seg.end], clip.sample_rate)
        report = analyze_audio_file(seg_path)
        if report.verdict == "fail":
            worst = "fail"
        elif report.verdict == "warning" and worst == "ok":
            worst = "warning"
        note = (
            {
                "verdict": "unverified",
                "note": "Aligned from the known script by speech-time proportion; not ASR-verified.",
            }
            if text
            else None
        )
        prof.recordings.append(
            RecordingEntry(
                recording_id=seg_path.stem,
                path=str(seg_path),
                language=language,
                script_id=script_id or None,
                quality_verdict=report.verdict,
                duration_seconds=report.duration_seconds,
                sample_rate=report.sample_rate,
                quality_report=report.model_dump(),
                category="master_take_clip",
                has_transcript=bool(text),
                transcript=text or None,
                transcript_check=note,
            )
        )
    save_profile(prof)
    get_store().set_pointer(POINTER, prof.profile_id)
    return RecordingResult(
        prof.profile_id,
        target.stem,
        worst,
        f"Long take split into {len(segments)} clips at natural pauses (script-aligned transcripts).",
    )


def _auto_transcript(path: Path, language: str) -> tuple[str | None, dict | None]:
    """Best-effort ASR for uploads without a script; silently skipped when unavailable."""
    from ..voice.transcription import transcribe

    try:
        text = transcribe(path, None if language in ("", "hinglish", "mixed") else language[:2])
    except Exception:
        return None, None
    if not text:
        return None, None
    return text, {
        "verdict": "asr",
        "note": "Transcribed automatically (faster-whisper); not hand-checked.",
    }


def save_recording(path: str | Path, language: str, script_id: str = "") -> RecordingResult:
    """Copy a recording into the workspace, validate and score it, add it to the profile."""
    src = Path(path)
    rec_dir = get_settings().recordings_dir
    rec_dir.mkdir(parents=True, exist_ok=True)
    target = unique_path(rec_dir, src.stem[:40] or "recording", src.suffix.lower() or ".wav")
    shutil.copy2(src, target)
    try:
        clip = read_audio(target)  # validates readability before accepting the file
        if clip.samples.shape[0] / clip.sample_rate > QualityThresholds().max_seconds:
            return _save_long_take(target, clip, language, script_id)
        report = analyze_audio_file(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    transcript, check = _auto_transcript(target, language) if not script_id else (None, None)
    prof = current_profile() or VoiceProfile()
    prof.recordings.append(
        RecordingEntry(
            recording_id=target.stem,
            has_transcript=bool(transcript),
            transcript=transcript,
            transcript_check=check,
            path=str(target),
            language=language,
            script_id=script_id or None,
            quality_verdict=report.verdict,
            duration_seconds=report.duration_seconds,
            sample_rate=report.sample_rate,
            quality_report=report.model_dump(),
            category="studio",
        )
    )
    save_profile(prof)
    get_store().set_pointer(POINTER, prof.profile_id)
    return RecordingResult(prof.profile_id, target.stem, report.verdict, summary_line(report))


def analyze() -> list[dict]:
    """Re-score every recording on disk and refresh phoneme coverage."""
    prof = _require_profile()
    rows = []
    for rec in prof.recordings:
        if rec.path and Path(rec.path).exists():
            rep = analyze_audio_file(rec.path)
            rec.quality_verdict = rep.verdict
            rec.quality_report = rep.model_dump()
            rows.append(
                {
                    "recording_id": rec.recording_id,
                    "verdict": rep.verdict,
                    "summary": summary_line(rep),
                }
            )
    prof.coverage_report = analyze_texts(
        coverage_texts_for_recordings(prof.recordings)
    ).model_dump()
    save_profile(prof)
    return rows


def set_transcript(recording_id: str, text: str) -> TranscriptCheck:
    """Store a transcript and verify it against the script that was read.

    Without a known script there is nothing to compare against, so the check is
    reported as unverified rather than as a spurious mismatch.
    """
    prof = _require_profile()
    rec = next((r for r in prof.recordings if r.recording_id == recording_id), None)
    if rec is None:
        raise ArtifactNotFoundError(f"Recording {recording_id!r} not found in the profile.")
    intended = ""
    if rec.script_id:
        entry = next((s for s in all_scripts() if s.id == rec.script_id), None)
        if entry is not None:
            intended = entry.text
    check = verify_transcript(intended, text, source="user", language=rec.language)
    rec.transcript = text
    rec.has_transcript = bool(text)
    rec.transcript_check = check.model_dump()
    save_profile(prof)
    return check


def coverage() -> CoverageReport | None:
    prof = _require_profile()
    return CoverageReport(**prof.coverage_report) if prof.coverage_report else None


def select_refs(prefer_transcript: bool = False):
    """Rank recordings and store the chosen references on the profile."""
    prof = _require_profile()
    caps = VoiceBackendCapabilities(**get_backend(prof.backend).capabilities())
    decision = select_references(
        [
            ReferenceCandidate(
                recording_id=r.recording_id,
                language=r.language,
                quality_verdict=r.quality_verdict,
                duration_seconds=r.duration_seconds,
                has_transcript=r.has_transcript,
                transcript_verdict=(r.transcript_check or {}).get("verdict"),
            )
            for r in prof.recordings
        ],
        caps,
        prefer_transcript=prefer_transcript,
    )
    prof.reference_selection = decision.model_dump()
    save_profile(prof)
    return decision


def grant_consent(name: str = "") -> ConsentRecord:
    return _grant("voice_cloning", {"statement": VOICE_CONSENT_STATEMENT, "by": name or "user"})


def revoke_consent() -> None:
    _revoke("voice_cloning")


def consent_status() -> ConsentRecord:
    return check_consent(SCOPE_VOICE)


def synthesize(text: str, seed: int | None = None) -> SynthesisResult:
    prof = _require_profile()
    return _synthesize(profile_id=prof.profile_id, text=text, seed=seed)
