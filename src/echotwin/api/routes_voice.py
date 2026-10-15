"""Voice routes: backends, plans, profiles, recordings, coverage, references, synthesis."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..exceptions import EchoTwinError
from ..storage import state
from . import service
from .schemas import SynthesizeRequest, VoicePlanRequest

router = APIRouter(tags=["voice"])


@router.get("/voice/backends")
def voice_backends() -> dict:
    from ..voice.registry import all_backend_statuses

    return {"backends": all_backend_statuses()}


@router.post("/voice/plans")
def voice_plan(req: VoicePlanRequest) -> dict:
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.recording_plan import InventoryItem, RecordingPreferences, generate_plan
    from ..voice.registry import get_backend

    try:
        backend = get_backend(req.backend)
        caps = VoiceBackendCapabilities(**backend.capabilities())
    except EchoTwinError as e:
        raise HTTPException(400, str(e)) from e
    inventory: list[InventoryItem] = []
    coverage = None
    if req.profile_id:
        from ..voice.metadata import load_profile
        from ..voice.phoneme_coverage import CoverageReport

        prof = load_profile(req.profile_id)
        inventory = [
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
        if prof.coverage_report:
            coverage = CoverageReport(**prof.coverage_report)
    plan = generate_plan(
        caps,
        RecordingPreferences(
            languages=req.languages,
            include_hinglish=req.include_hinglish,
            max_minutes=req.max_minutes,
        ),
        inventory,
        coverage,
    )
    return plan.model_dump()


@router.get("/voice/scripts")
def voice_scripts(language: str = "hi", category: str | None = None) -> dict:
    from ..voice.script_library import all_scripts

    entries = [e for e in all_scripts() if e.language.value == language]
    if category:
        entries = [e for e in entries if e.category.value == category]
    return {"scripts": [e.model_dump() for e in entries]}


@router.get("/voice/profiles")
def voice_profiles() -> dict:
    from ..voice.metadata import list_profiles

    return {"profiles": [p.model_dump() for p in list_profiles()]}


@router.post("/voice/profiles")
def create_voice_profile(payload: dict) -> dict:
    from ..voice.metadata import VoiceProfile, save_profile

    profile = VoiceProfile(**{k: v for k, v in payload.items() if k in VoiceProfile.model_fields})
    save_profile(profile)
    state.set("last_voice_profile_id", profile.profile_id)
    return {"profile_id": profile.profile_id}


@router.delete("/voice/profiles/{profile_id}")
def delete_voice_profile(profile_id: str) -> dict:
    from ..voice.metadata import delete_profile

    return delete_profile(profile_id)


@router.post("/voice/profiles/{profile_id}/recordings/upload")
async def upload_recording(profile_id: str, file: UploadFile = File(...)) -> dict:
    from ..voice.audio_ingestion import read_audio
    from ..voice.audio_quality import analyze_audio_file
    from ..voice.metadata import RecordingEntry, load_profile, save_profile
    from ..voice.phoneme_coverage import analyze_texts

    data = await file.read()
    path = service.save_upload(file.filename or "recording.wav", data)
    prof = load_profile(profile_id)
    try:
        clip = read_audio(path)
        report = analyze_audio_file(path)
    except EchoTwinError as e:
        raise HTTPException(400, str(e)) from e
    rec = RecordingEntry(
        recording_id=path.stem,
        path=str(path),
        language="unknown",
        quality_verdict=report.verdict,
        duration_seconds=clip.sample_rate and round(len(clip.samples) / clip.sample_rate, 3),
        sample_rate=clip.sample_rate,
        quality_report=report.model_dump(),
        category="uploaded",
    )
    prof.recordings.append(rec)
    prof.coverage_report = analyze_texts([r.transcript or "" for r in prof.recordings]).model_dump()
    save_profile(prof)
    return {"recording": rec.model_dump(), "quality": report.model_dump()}


@router.post("/voice/profiles/{profile_id}/recordings/{recording_id}/transcript")
def set_transcript(profile_id: str, recording_id: str, payload: dict) -> dict:
    from ..voice.metadata import load_profile, save_profile
    from ..voice.transcript_validation import verify_transcript

    prof = load_profile(profile_id)
    rec = next((r for r in prof.recordings if r.recording_id == recording_id), None)
    if rec is None:
        raise HTTPException(404, "recording not found")
    rec.transcript = payload.get("transcript", "")
    rec.has_transcript = bool(rec.transcript)
    if rec.has_transcript and rec.script_id:
        from ..voice.script_library import get_script

        script_text = get_script(rec.script_id).text
        check = verify_transcript(
            script_text, rec.transcript, source="user", language=payload.get("language", "en")
        )
        rec.transcript_check = check.model_dump()
    save_profile(prof)
    return {"recording": rec.model_dump()}


@router.post("/voice/coverage/analyze")
def analyze_coverage(profile_id: str) -> dict:
    from ..voice.metadata import load_profile, save_profile
    from ..voice.phoneme_coverage import analyze_texts

    prof = load_profile(profile_id)
    report = analyze_texts([r.transcript or "" for r in prof.recordings])
    prof.coverage_report = report.model_dump()
    save_profile(prof)
    return report.model_dump()


@router.post("/voice/references/select")
def select_references(profile_id: str, payload: dict | None = None) -> dict:
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.metadata import load_profile, save_profile
    from ..voice.reference_selection import ReferenceCandidate
    from ..voice.reference_selection import select_references as _select
    from ..voice.registry import get_backend

    prof = load_profile(profile_id)
    backend = get_backend(prof.backend)
    caps = VoiceBackendCapabilities(**backend.capabilities())
    candidates = [
        ReferenceCandidate(
            recording_id=r.recording_id,
            language=r.language,
            quality_verdict=r.quality_verdict,
            duration_seconds=r.duration_seconds,
            has_transcript=r.has_transcript,
            transcript_verdict=(r.transcript_check or {}).get("verdict")
            if r.transcript_check
            else None,
            coverage_categories_present=[],
        )
        for r in prof.recordings
    ]
    decision = _select(
        candidates, caps, prefer_transcript=bool((payload or {}).get("prefer_transcript"))
    )
    prof.reference_selection = decision.model_dump()
    save_profile(prof)
    return decision.model_dump()


@router.post("/voice/synthesize")
def voice_synthesize(req: SynthesizeRequest) -> dict:
    from ..voice.synthesis import synthesize

    def _work() -> dict:
        result = synthesize(
            profile_id=req.profile_id,
            text=req.text,
            backend_name=req.backend,
            seed=req.seed,
            settings=req.settings,
            language=req.language,
        )
        return result.model_dump()

    job = service.JOBS.submit("voice_synthesis", _work)
    return job.model_dump()
