"""Voice commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ..exceptions import EchoTwinError
from ..storage import state
from ._common import (
    _fail,
    console,
    voice_app,
)


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------
@voice_app.command("backends")
def voice_backends() -> None:
    """List voice backends with their real capabilities."""
    from ..voice.registry import all_backend_statuses

    rows = all_backend_statuses()
    table = Table(title="Voice backends")
    table.add_column("Backend")
    table.add_column("Available")
    table.add_column("Languages (n)")
    table.add_column("Transcript")
    table.add_column("License")
    for r in rows:
        caps = r.get("capabilities", {})
        table.add_row(
            r["name"],
            str(caps.get("available", r.get("module_importable", False))),
            str(len(caps.get("languages", []))) if caps else "-",
            "required"
            if caps.get("reference_transcript_required")
            else ("optional" if caps.get("reference_transcript_optional") else "none"),
            caps.get("license", "-") if caps else (r.get("error", "-") or "-")[:40],
        )
    console.print(table)


@voice_app.command("scripts")
def voice_scripts(
    language: str = typer.Option("hi", help="hi | en | hinglish"),
    category: str = typer.Option(None),
) -> None:
    """Show recording scripts for a language."""
    from ..voice.script_library import all_scripts

    entries = [e for e in all_scripts() if e.language.value == language]
    if category:
        entries = [e for e in entries if e.category.value == category]
    if not entries:
        _fail(f"No scripts for language '{language}'.")
    for e in entries:
        console.print(f"[bold]{e.id}[/bold] [{e.category.value}] est {e.est_seconds}s")
        console.print(f"  {e.text}")
        if e.translation:
            console.print(f"  [dim]{e.translation}[/dim]")


@voice_app.command("plan")
def voice_plan(
    backend: str = typer.Option("voxcpm"),
    languages: str = typer.Option("hi,en"),
    max_minutes: float = typer.Option(20.0),
    profile: str = typer.Option(None, help="voice profile id (uses its inventory)"),
) -> None:
    """Generate an adaptive recording plan for the selected backend."""
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.recording_plan import InventoryItem, RecordingPreferences, generate_plan
    from ..voice.registry import get_backend

    try:
        backend_obj = get_backend(backend)
        caps = VoiceBackendCapabilities(**backend_obj.capabilities())
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    inventory: list[InventoryItem] = []
    if profile:
        from ..voice.metadata import load_profile

        prof = load_profile(profile)
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
    prefs = RecordingPreferences(
        languages=[x for x in languages.split(",") if x],
        max_minutes=max_minutes,
        speaker_profile={
            "hi": "native_hindi",
            "en": "indian_english",
            "hinglish": "natural_hinglish",
        }.get(languages.split(",")[0], "bilingual"),
    )
    plan = generate_plan(caps, prefs, inventory)
    console.print(
        f"[bold]Recording plan for {plan.backend}[/bold] (total ~{plan.total_estimated_seconds}s, "
        f"within budget ~{plan.within_time_budget_seconds}s)"
    )
    for t in plan.tasks:
        flag = (
            "MANDATORY"
            if t.mandatory
            else ("beyond-budget" if t.beyond_time_budget else "optional")
        )
        console.print(f"  P{t.priority} [{flag}] {t.title} (~{t.est_seconds}s)")
        console.print(f"     {t.rationale}")
        if t.script_ids:
            console.print(f"     scripts: {', '.join(t.script_ids)}")
    for s in plan.skipped:
        console.print(f"[yellow]skipped[/yellow] {s['language']}: {s['reason']} — {s['detail']}")


@voice_app.command("record")
def voice_record(
    backend: str = typer.Option("voxcpm"), languages: str = typer.Option("hi,en")
) -> None:
    """Show the next scripts to record (browser recording lives in the Web UI)."""
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.recording_plan import RecordingPreferences, generate_plan
    from ..voice.registry import get_backend

    try:
        backend_obj = get_backend(backend)
        caps = VoiceBackendCapabilities(**backend_obj.capabilities())
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    plan = generate_plan(
        caps, RecordingPreferences(languages=[x for x in languages.split(",") if x]), []
    )
    tasks = [t for t in plan.tasks]
    console.print(
        "Read each sentence aloud naturally, then import with "
        "'echotwin voice import <folder>' or record in the Web UI Voice Studio."
    )
    for t in tasks[1:] if tasks and tasks[0].id == "test-voice" else tasks:
        console.print(f"\n[bold]{t.title}[/bold]")
        for sid in t.script_ids:
            from ..voice.script_library import get_script

            entry = get_script(sid)
            console.print(f"  ({entry.language.value}) {entry.text}")


@voice_app.command("import")
def voice_import(
    folder: Path = typer.Argument(..., help="folder of recordings (wav/flac/...)"),
    language: str = typer.Option(None, help="assign language if filenames don't encode it: hi|en"),
    profile: str = typer.Option(None, help="existing profile id to add to"),
    backend: str = typer.Option("voxcpm"),
) -> None:
    """Import recordings into a voice profile and run quality+coverage analysis."""

    from ..voice.audio_ingestion import read_audio
    from ..voice.metadata import RecordingEntry, VoiceProfile, load_profile
    from ..voice.metadata import save_profile as save_voice_profile
    from ..voice.phoneme_coverage import analyze_texts, coverage_texts_for_recordings

    if not folder.exists():
        _fail(f"Folder does not exist: {folder}")
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    )
    if not files:
        _fail("No audio files found (wav/flac/ogg/mp3/m4a).")
    prof = (
        load_profile(profile)
        if profile
        else VoiceProfile(backend=backend, languages=[language] if language else ["hi", "en"])
    )
    for p in files:
        try:
            clip = read_audio(p)
        except EchoTwinError as e:
            console.print(f"[yellow]skip[/yellow] {p.name}: {e}")
            continue
        from ..voice.audio_quality import analyze_wave_array

        report = analyze_wave_array(clip.samples, clip.sample_rate, path=str(p))
        prof.recordings.append(
            RecordingEntry(
                recording_id=p.stem,
                path=str(p),
                language=language or _guess_lang_from_name(p.stem),
                quality_verdict=report.verdict,
                duration_seconds=report.duration_seconds,
                sample_rate=report.sample_rate,
                quality_report=report.model_dump(),
                category="imported",
            )
        )
    prof.coverage_report = analyze_texts(
        coverage_texts_for_recordings(prof.recordings)
    ).model_dump()
    out = save_voice_profile(prof)
    state.set("last_voice_profile_id", prof.profile_id)
    console.print(
        f"[green]Voice profile saved[/green] {prof.profile_id} ({len(prof.recordings)} recordings) -> {out}"
    )


def _guess_lang_from_name(name: str) -> str:
    n = name.lower()
    if "hi" in n or "hindi" in n:
        return "hi"
    if "en" in n or "eng" in n:
        return "en"
    if "mix" in n or "hinglish" in n:
        return "hinglish"
    return "unknown"


@voice_app.command("analyze")
def voice_analyze(profile: str = typer.Option(None)) -> None:
    """Run quality + coverage analysis for a voice profile."""
    from ..voice.audio_quality import analyze_audio_file, summary_line
    from ..voice.metadata import load_profile
    from ..voice.metadata import save_profile as save_voice_profile
    from ..voice.phoneme_coverage import analyze_texts, coverage_texts_for_recordings

    pid = profile or state.get("last_voice_profile_id")
    if not pid:
        _fail("No voice profile. Import recordings first.")
    prof = load_profile(pid)
    for rec in prof.recordings:
        if rec.path and Path(rec.path).exists():
            rep = analyze_audio_file(rec.path)
            rec.quality_verdict = rep.verdict
            rec.quality_report = rep.model_dump()
            console.print(f"{rec.recording_id}: {rep.verdict} ({summary_line(rep)})")
    prof.coverage_report = analyze_texts(
        coverage_texts_for_recordings(prof.recordings)
    ).model_dump()
    save_voice_profile(prof)
    console.print("[green]Analysis complete[/green]")


@voice_app.command("coverage")
def voice_coverage(profile: str = typer.Option(None)) -> None:
    """Show the phonetic/language coverage report and gaps."""
    from ..voice.metadata import load_profile
    from ..voice.phoneme_coverage import CoverageReport

    pid = profile or state.get("last_voice_profile_id")
    if not pid:
        _fail("No voice profile.")
    prof = load_profile(pid)
    if not prof.coverage_report:
        _fail("No coverage report yet; run 'echotwin voice analyze'.")
    report = CoverageReport(**prof.coverage_report)
    table = Table(title="Coverage (text-derived)")
    table.add_column("Category")
    table.add_column("Count")
    table.add_column("Covered")
    table.add_column("Confidence")
    for k, v in report.categories.items():
        table.add_row(k, str(v.count), str(v.covered), v.confidence)
    console.print(table)
    console.print("[dim]" + "; ".join(report.limitations) + "[/dim]")


@voice_app.command("recommend")
def voice_recommend(profile: str = typer.Option(None), backend: str = typer.Option(None)) -> None:
    """Recommend the next recording based on gaps."""
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.metadata import load_profile
    from ..voice.phoneme_coverage import CoverageReport
    from ..voice.recording_plan import InventoryItem, RecordingPreferences, generate_plan
    from ..voice.registry import get_backend

    pid = profile or state.get("last_voice_profile_id")
    if not pid:
        _fail("No voice profile.")
    prof = load_profile(pid)
    backend_name = backend or prof.backend
    backend_obj = get_backend(backend_name)
    caps = VoiceBackendCapabilities(**backend_obj.capabilities())
    coverage = CoverageReport(**prof.coverage_report) if prof.coverage_report else None
    plan = generate_plan(
        caps,
        RecordingPreferences(languages=prof.languages),
        [
            InventoryItem(
                recording_id=r.recording_id,
                language=r.language,
                quality_verdict=r.quality_verdict,
                duration_seconds=r.duration_seconds,
                category=r.category,
                has_transcript=r.has_transcript,
            )
            for r in prof.recordings
        ],
        coverage,
    )
    top = [t for t in plan.tasks if not t.beyond_time_budget][:3]
    console.print("Recommended next recordings:")
    for t in top:
        console.print(f"  P{t.priority} {t.title} (~{t.est_seconds}s): scripts {t.script_ids}")
    if plan.backend_minimum_met:
        console.print(
            "[green]Backend minimum met — prefer testing the voice over recording more.[/green]"
        )


@voice_app.command("test")
def voice_test(
    profile: str = typer.Option(None),
    text: str = typer.Option("नमस्ते, यह EchoTwin की एक परख है।", help="test sentence"),
    backend: str = typer.Option(None),
) -> None:
    """Synthesize a test sentence with the selected backend and reference."""
    from ..voice.consent import check_consent
    from ..voice.metadata import load_profile
    from ..voice.synthesis import synthesize

    if not check_consent("voice_cloning").granted:
        _fail("Voice cloning consent required: confirm authorization in the UI/API first.")
    pid = profile or state.get("last_voice_profile_id")
    if not pid:
        _fail("No voice profile.")
    prof = load_profile(pid)
    try:
        result = synthesize(profile_id=pid, text=text, backend_name=backend or prof.backend)
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    console.print(f"[green]Test audio written[/green]: {result.output_path}")
    console.print(
        "Evaluate it with: echotwin voice evaluate --audio <path> (see Web UI for blinded comparison)."
    )


@voice_app.command("evaluate")
def voice_evaluate(
    audio: Path = typer.Argument(...),
    reference: list[Path] = typer.Argument(...),
    expected_text: str = typer.Option(None),
) -> None:
    """Evaluate generated audio (similarity if resemblyzer installed; WER if ASR installed)."""
    from ..voice.evaluation import evaluate_synthesis

    result = evaluate_synthesis(audio, reference, expected_text=expected_text)
    console.print(json.dumps(result.model_dump(), indent=1, default=str))


@voice_app.command("synthesize")
def voice_synthesize(
    text: str = typer.Option(..., "--text"),
    profile: str = typer.Option(None),
    backend: str = typer.Option(None),
    seed: int = typer.Option(None),
) -> None:
    """Synthesize speech from user-approved text using the authorized voice profile."""
    from ..voice.metadata import load_profile
    from ..voice.synthesis import synthesize

    pid = profile or state.get("last_voice_profile_id")
    if not pid:
        _fail("No voice profile.")
    prof = load_profile(pid)
    try:
        result = synthesize(
            profile_id=pid, text=text, backend_name=backend or prof.backend, seed=seed
        )
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    console.print(json.dumps(result.model_dump(), indent=2, default=str))
