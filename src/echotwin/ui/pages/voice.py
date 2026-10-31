"""Voice: plan recordings, record or upload, pick references, grant consent, synthesize."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from ...exceptions import EchoTwinError
from ...services import jobs
from ...services import voice as voice_service
from ...voice import script_library
from ...voice.registry import registered_backends
from .. import components, render
from ..context import PageContext, rail_value, warn

_REC_COLUMNS = ["ID", "Language", "Quality", "Seconds", "Transcript"]
_VERDICT_STATE = {"ok": "ok", "warning": "warn", "fail": "error"}
LANGUAGES = [lang.value for lang in script_library.ScriptLanguage]


def consent_badge() -> str:
    granted = voice_service.consent_status().granted
    return components.status_badge(
        "ok" if granted else "warn",
        "Voice consent granted" if granted else "Voice consent not granted",
    )


def recordings_df() -> pd.DataFrame:
    profile = voice_service.current_profile()
    rows = [
        {
            "ID": r.recording_id,
            "Language": r.language,
            "Quality": r.quality_verdict,
            "Seconds": round(r.duration_seconds, 1),
            "Transcript": "yes" if r.has_transcript else "no",
        }
        for r in (profile.recordings if profile else [])
    ]
    return pd.DataFrame(rows, columns=_REC_COLUMNS)


def plan_view(
    backend: str, languages: list[str], minutes: float, single_take: bool = True
) -> pd.DataFrame:
    try:
        plan = voice_service.plan(backend, languages or LANGUAGES, minutes, single_take)
    except EchoTwinError as exc:
        return pd.DataFrame(
            [
                {
                    "Task": f"Cannot build a plan: {exc}",
                    "Language": "-",
                    "Category": "-",
                    "Seconds": 0,
                    "Required": "-",
                }
            ]
        )
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "Task": f"Cannot build a plan: {type(exc).__name__}: {exc}",
                    "Language": "-",
                    "Category": "-",
                    "Seconds": 0,
                    "Required": "-",
                }
            ]
        )
    rows = [
        {
            "Task": t.title,
            "Language": t.language,
            "Category": t.category,
            "Seconds": round(t.est_seconds),
            "Required": "yes" if t.mandatory else "no",
        }
        for t in plan.tasks
    ]
    return pd.DataFrame(rows, columns=["Task", "Language", "Category", "Seconds", "Required"])


def script_choices(language: str, show_all: bool = False) -> list[str]:
    """Only the single master take unless the user asks for the full practice library."""
    master = script_library.MASTER_TAKE_ID
    if not show_all:
        return [master]
    return [master] + [s.id for s in script_library.scripts_by_language(language) if s.id != master]


def show_script(script_id: str | None) -> str:
    if not script_id:
        return ""
    return script_library.render(script_library.get_script(script_id))


def save_rec(audio: str | None, language: str, script_id: str):
    if not audio:
        return (
            components.status_badge("warn", "No audio"),
            recordings_df(),
            render.callout("warn", "Record or upload audio first."),
            rail_value(),
        )
    try:
        result = voice_service.save_recording(audio, language, script_id or "")
    except Exception as exc:
        return (
            components.status_badge("error", "Not saved"),
            recordings_df(),
            warn(exc),
            rail_value(),
        )
    badge = components.status_badge(_VERDICT_STATE.get(result.verdict, "info"), result.summary)
    return badge, recordings_df(), "", rail_value()


def do_analyze():
    try:
        reports = voice_service.analyze()
    except Exception as exc:
        return recordings_df(), warn(exc)
    bad = [r for r in reports if r["verdict"] == "fail"]
    kind = "warn" if bad else "ok"
    return recordings_df(), render.callout(
        kind, f"Analyzed {len(reports)} recording(s); {len(bad)} failed."
    )


def set_transcript(recording_id: str, text: str) -> str:
    try:
        check = voice_service.set_transcript(recording_id.strip(), text)
    except Exception as exc:
        return warn(exc)
    kind = "ok" if check.verdict == "match" else "warn"
    return render.callout(kind, f"Transcript check: {check.verdict}. {check.note}".strip())


def do_select_refs(prefer_transcript: bool) -> str:
    try:
        decision = voice_service.select_refs(bool(prefer_transcript))
    except Exception as exc:
        return warn(exc)
    if not decision.selected:
        why = "; ".join(f"{r.get('recording_id')}: {r.get('reason')}" for r in decision.rejected)
        return render.callout("warn", f"No usable reference. {why}".strip())
    ids = ", ".join(c.recording_id for c in decision.selected)
    return render.callout("ok", f"Selected: {ids}. {decision.rationale}".strip())


def grant(confirm: bool):
    if not confirm:
        return consent_badge(), render.callout("warn", "Tick the box to confirm consent.")
    voice_service.grant_consent()
    return consent_badge(), ""


def revoke() -> str:
    voice_service.revoke_consent()
    return consent_badge()


def synth_start(text: str, seed: str) -> str:
    text = (text or "").strip()
    if not text:
        return render.callout("warn", "Type the text to speak.")
    if not voice_service.consent_status().granted:
        return render.callout("warn", "Voice cloning consent has not been granted. Grant it above.")
    profile = voice_service.current_profile()
    if profile is None or not (profile.reference_selection or {}).get("selected"):
        return render.callout("warn", "Select reference recordings first.")
    try:
        seed_value = int(seed) if str(seed).strip() else None
    except ValueError:
        return render.callout("warn", "Seed must be a whole number.")

    def work(ctx: jobs.JobContext) -> dict:
        ctx.progress(0.1, "Synthesizing")
        result = voice_service.synthesize(text, seed_value)
        return {"output_path": result.output_path, "backend": result.backend}

    jobs.submit("synthesis", work, heavy=True)
    return render.callout("info", "Synthesizing. The audio appears below when ready.")


def synth_poll():
    job = next((j for j in jobs.list_jobs() if j["kind"] == "synthesis"), None)
    if job is None:
        return gr.skip(), gr.skip()
    if job["status"] not in jobs.TERMINAL:
        return render.callout("info", f"Synthesizing... {job.get('message') or ''}"), gr.skip()
    if job["status"] != "completed":
        return render.callout("error", job.get("error") or f"Job {job['status']}."), gr.skip()
    return render.callout("ok", "Done."), job["result"]["output_path"]


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "Upload or record any clear voice (a script is optional; speech is auto-transcribed). "
        "Only recreate a voice you are entitled to, and label AI speech as AI."
    )
    backends = registered_backends()
    with gr.Accordion("1. Recording plan", open=False):
        with gr.Row():
            backend = gr.Dropdown(backends, value=backends[0], label="Backend")
            langs = gr.CheckboxGroup(LANGUAGES, value=LANGUAGES, label="Languages")
            minutes = gr.Slider(2, 30, value=10, step=1, label="Minutes available")
            single_take = gr.Checkbox(
                value=True, label="Single long take (one ~100 s read, recommended)"
            )
        plan_btn = gr.Button("Show plan")
        plan_table = gr.Dataframe(interactive=False, wrap=True)
        plan_btn.click(plan_view, [backend, langs, minutes, single_take], plan_table)

    gr.Markdown(
        "### 2. Record\n"
        "Read the whole script below in **one continuous take** (Hindi, English and Hinglish are all in it). "
        "The `[brackets]` tell you the **mood** for the next lines — do the mood, "
        "**don't say the bracketed words aloud**."
    )
    language = gr.State("hinglish")
    script = gr.State(script_library.MASTER_TAKE_ID)
    gr.Textbox(
        value=show_script(script_library.MASTER_TAKE_ID),
        label="Your script",
        interactive=False,
        lines=24,
    )
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recording")
    save_btn = gr.Button("Save recording", variant="primary")
    badge = gr.HTML()
    rec_status = gr.HTML()
    table = gr.Dataframe(recordings_df(), interactive=False, label="Recordings")
    save_btn.click(save_rec, [audio, language, script], [badge, table, rec_status, ctx.rail])

    gr.Markdown("### 3. Check and choose references")
    with gr.Row():
        analyze_btn = gr.Button("Analyze recordings")
        prefer = gr.Checkbox(label="Prefer recordings with transcripts")
        refs_btn = gr.Button("Select references")
    with gr.Row():
        rec_id = gr.Textbox(label="Recording ID", scale=1)
        transcript = gr.Textbox(label="Transcript of that recording", scale=3)
        transcript_btn = gr.Button("Check transcript", scale=1)
    check_status = gr.HTML()
    analyze_btn.click(do_analyze, outputs=[table, check_status])
    refs_btn.click(do_select_refs, prefer, check_status)
    transcript_btn.click(set_transcript, [rec_id, transcript], check_status)

    gr.Markdown("### 4. Consent and synthesis")
    consent = gr.HTML(consent_badge())
    with gr.Row():
        confirm = gr.Checkbox(
            label="This is my voice, or I am entitled to recreate it. I won't use it to deceive anyone.",
            scale=5,
        )
        grant_btn = gr.Button("Grant consent")
        revoke_btn = gr.Button("Revoke")
    consent_status = gr.HTML()
    grant_btn.click(grant, confirm, [consent, consent_status])
    revoke_btn.click(revoke, outputs=consent)
    with gr.Row():
        text = gr.Textbox(label="Text to speak", scale=4)
        seed = gr.Textbox(label="Seed (optional)", scale=1)
    synth_btn = gr.Button("Synthesize", variant="primary")
    synth_status = gr.HTML()
    output = gr.Audio(label="Result", type="filepath", interactive=False)
    synth_btn.click(synth_start, [text, seed], synth_status)
    ctx.timer.tick(synth_poll, outputs=[synth_status, output])
