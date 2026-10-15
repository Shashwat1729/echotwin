"""EchoTwin Voice Capture + Personalization Studio (Gradio).

Every control calls the real library: imports parse forensically, plans adapt
to backend capabilities, recordings are quality-checked, synthesis requires
explicit voice-cloning consent, and all outputs carry provenance metadata.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..settings import get_settings
from ..storage import db, state

APP_TITLE = "EchoTwin Studio"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _messages_path(import_id: str) -> Path:
    return get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"


def _read_state_messages() -> tuple[list, str | None]:
    from ..ingestion.schema import Message

    import_id = state.get("last_import_id")
    if not import_id:
        return [], None
    p = _messages_path(import_id)
    if not p.exists():
        return [], None
    messages = [
        Message.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return messages, import_id


def dashboard_markdown() -> str:
    from ..voice.registry import all_backend_statuses

    s = get_settings()
    imp = db.latest_import()
    lines = [
        f"## {APP_TITLE}",
        f"- Home: `{s.home}` (db: `{s.db_path.name}`)",
        f"- Import: {imp['import_id'] if imp else 'none'} ({imp['parsed'] if imp else 0} messages)",
        f"- Target author: `{state.get('target_sender_id') or 'not selected'}`",
        f"- Dataset: `{state.get('last_dataset_id') or 'none'}`",
        f"- Voice profile: `{state.get('last_voice_profile_id') or 'none'}`",
        f"- Voice consent: **{'granted' if db.get_consent('voice_cloning')['granted'] else 'NOT granted'}**",
    ]
    for b in all_backend_statuses():
        caps = b.get("capabilities", {})
        lines.append(
            f"- Backend `{b['name']}`: available={caps.get('available', False)} "
            f"langs={len(caps.get('languages', []))}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data import tab
# ---------------------------------------------------------------------------
def import_file(file) -> tuple[str, list[list], str]:
    from ..ingestion import WhatsAppParser, participant_stats
    from ..ingestion.export_loader import conversation_id_from_name, load_export

    if file is None:
        return "Upload a WhatsApp .txt or .zip export first.", [], ""
    from ..storage.paths import validate_upload

    src = Path(file.name if hasattr(file, "name") else str(file))
    data = src.read_bytes()
    validate_upload(src.name, len(data))
    from ..storage.paths import unique_path

    target = unique_path(get_settings().exports_dir, src.stem[:40] or "upload", src.suffix.lower())
    target.write_bytes(data)

    bundle = load_export(target)
    parser = WhatsAppParser()
    all_messages = []
    conversations: list[str] = []
    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"chat-{i}")
        try:
            result = parser.parse_text(chat.chat_text, convo_id)
        except Exception:
            continue
        all_messages.extend(result.messages)
        conversations.append(convo_id)
    if not all_messages:
        return "No messages parsed from the export. Is this a WhatsApp chat export?", [], ""
    import_id = db.record_import(
        str(target), "chat", len(all_messages), None, {"conversations": conversations}
    )
    state.set("last_import_id", import_id)
    out_dir = get_settings().artifacts_dir / "imports" / import_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as fh:
        for m in all_messages:
            fh.write(m.model_dump_json() + "\n")
    rows = [
        [s.sender_name, s.message_count, str(s.first_timestamp), str(s.last_timestamp)]
        for s in participant_stats(all_messages)
    ]
    names = [[r[0]] for r in rows]
    return (
        f"Imported **{len(all_messages)}** messages from {len(conversations)} conversation(s). "
        "Pick the target author below — it is never auto-selected.",
        rows,
        json.dumps(names),
    )


def select_author(name: str) -> str:
    from ..ingestion import pick_participant

    messages, _ = _read_state_messages()
    if not messages:
        return "Import data first."
    sid = pick_participant(messages, (name or "").strip())
    if sid is None:
        return f"Participant {name!r} not found."
    imp = db.latest_import()
    if imp is not None:
        db.update_import_target(imp["import_id"], sid)
    state.set("target_sender_id", sid)
    return f"Target author set to {name!r}."


# ---------------------------------------------------------------------------
# dataset tab
# ---------------------------------------------------------------------------
def prepare_dataset(context_messages: int, max_context_chars: int, split_mode: str) -> str:
    from ..dataset.builder import DatasetConfig, build_examples, save_jsonl
    from ..dataset.splits import SplitConfig, make_manifest, split_dataset

    messages, import_id = _read_state_messages()
    target = state.get("target_sender_id")
    if not messages or not target:
        return "Import data and select the target author first."
    built = build_examples(
        messages,
        target,
        DatasetConfig(
            context_messages=int(context_messages), max_context_chars=int(max_context_chars)
        ),
    )
    split_cfg = SplitConfig(mode=split_mode)
    split = split_dataset(built, split_cfg)
    ds_id = f"dataset-{import_id}"
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    save_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_jsonl(split.train, ds_dir / "train.jsonl")
    save_jsonl(split.val, ds_dir / "val.jsonl")
    save_jsonl(split.test, ds_dir / "test.jsonl")
    manifest = make_manifest(
        dataset_id=ds_id,
        dataset_path=ds_dir / "examples.jsonl",
        split=split,
        built=built,
        target_sender_id=target,
        split_cfg=split_cfg,
        echotwin_version="0.1.0",
    )
    (ds_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    state.set("last_dataset_id", ds_id)
    warn = "\n".join(f"- {w}" for w in split.report.warnings)
    summary = (
        f"Dataset **{ds_id}**: {len(built.examples)} examples "
        f"(train={len(split.train)} val={len(split.val)} test={len(split.test)}, "
        f"leakage_check={split.report.leakage_check_passed})."
    )
    if warn:
        summary += f"\nWarnings:\n{warn}"
    return summary


def dataset_stats() -> str:
    ds_id = state.get("last_dataset_id")
    if not ds_id:
        return "No dataset yet."
    manifest = get_settings().artifacts_dir / "datasets" / ds_id / "manifest.json"
    if not manifest.exists():
        return "Manifest missing; rebuild the dataset."
    return manifest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# profile + evaluation tabs
# ---------------------------------------------------------------------------
def build_profile() -> str:
    from ..profiling.style_profile import build_style_profile, profile_prompt_fragment, save_profile

    messages, _ = _read_state_messages()
    target = state.get("target_sender_id")
    if not messages or not target:
        return "Import data and select the target author first."
    profile = build_style_profile(messages, target)
    out = get_settings().artifacts_dir / "profiles" / f"profile-{target[:8]}.json"
    save_profile(profile, out)
    state.set("last_profile_id", str(out))
    return f"Saved to `{out}`.\n\n{profile_prompt_fragment(profile)}"


def run_baseline(method: str, rag_k: int) -> str:
    from ..dataset.builder import load_jsonl
    from ..dataset.retrieval import RetrievalConfig, build_index_from_examples
    from ..evaluation.baseline import RETRIEVAL_METHODS, heldout_prompts, run_baseline
    from ..models.generation import MockGenerator
    from ..profiling.style_profile import load_profile, profile_prompt_fragment

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        return "Prepare a dataset first."
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    test, train, all_examples = (
        load_jsonl(ds_dir / "test.jsonl"),
        load_jsonl(ds_dir / "train.jsonl"),
        load_jsonl(ds_dir / "examples.jsonl"),
    )
    prompts = heldout_prompts(test, max_prompts=50)
    if not prompts:
        return "No held-out prompts: the split is too small (import multiple chats)."
    fragment = None
    prof_path = state.get("last_profile_id")
    if prof_path and Path(prof_path).exists():
        fragment = profile_prompt_fragment(load_profile(prof_path))
    if method == "ngram":
        from ..models.ngram import NGramGenerator, NGramModel

        if not train:
            return "Train split is empty; import multiple chats first."
        gen = NGramGenerator(
            NGramModel(order=3).train_texts([ex.messages[-1].content for ex in train]), seed=7
        )
        revision = "ngram-o3"
    else:
        gen = MockGenerator([ex.messages[-1].content for ex in train[:5]], seed=7)
        revision = "mock-1"
    _records, path, report = run_baseline(
        method=method,
        generator=gen,
        prompts=prompts,
        profile_fragment=fragment,
        few_shot=[
            {
                "context": " | ".join(t.content for t in ex.messages[:-1]),
                "response": ex.messages[-1].content,
            }
            for ex in train[:5]
        ],
        output_dir=get_settings().artifacts_dir / "evaluations" / ds_id,
        model_revision=revision,
        retrieval_index=build_index_from_examples(all_examples)
        if method in RETRIEVAL_METHODS
        else None,
        retrieval_cfg=RetrievalConfig(k=int(rag_k)),
        train_texts=[ex.messages[-1].content for ex in train],
    )
    style = report.style
    contam = report.contamination or {}
    return (
        f"Method **{method}** → `{path.name}`\n"
        f"- char-3gram cosine: {style.char_ngram_cosine_3 if style else 'n/a'}\n"
        f"- punctuation distance: {style.punctuation_distance if style else 'n/a'}\n"
        f"- emoji ref/gen: {style.emoji_rate_ref if style else 'n/a'}/"
        f"{style.emoji_rate_gen if style else 'n/a'}\n"
        f"- contamination: exact={contam.get('exact_match_rate')} "
        f"near={contam.get('near_match_rate')} verdict={contam.get('verdict')}\n"
        f"- {report.sample_size_warning or 'sample size OK'}"
    )


def chat_reply(user_text: str, method: str, history: list | None) -> tuple[str, list]:
    from ..dataset.builder import load_jsonl
    from ..dataset.retrieval import RetrievalConfig, build_index_from_examples, retrieve_exemplars
    from ..evaluation.baseline import RETRIEVAL_METHODS, build_messages
    from ..models.generation import GeneratorConfig, MockGenerator
    from ..profiling.style_profile import load_profile, profile_prompt_fragment

    history = history or []
    ds_id = state.get("last_dataset_id")
    fragment, few_shot, exemplars = None, [], []
    if ds_id:
        ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
        if (ds_dir / "train.jsonl").exists():
            train = load_jsonl(ds_dir / "train.jsonl")
            few_shot = [
                {
                    "context": " | ".join(t.content for t in ex.messages[:-1]),
                    "response": ex.messages[-1].content,
                }
                for ex in train[:5]
            ]
            if method in RETRIEVAL_METHODS and (ds_dir / "examples.jsonl").exists():
                index = build_index_from_examples(load_jsonl(ds_dir / "examples.jsonl"))
                exemplars = retrieve_exemplars(
                    index,
                    [{"role": "user", "content": user_text}],
                    cfg=RetrievalConfig(k=3),
                )
    prof_path = state.get("last_profile_id")
    if prof_path and Path(prof_path).exists():
        fragment = profile_prompt_fragment(load_profile(prof_path))
    prompt = {"messages": [{"role": "user", "content": user_text}]}
    messages = build_messages(method, prompt, fragment, few_shot, exemplars)
    from ..models.generation import TextGenerator

    gen: TextGenerator
    if method == "ngram" and ds_id:
        from ..models.ngram import NGramGenerator, NGramModel

        ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
        train_path = ds_dir / "train.jsonl"
        if train_path.exists():
            train_all = load_jsonl(train_path)
            gen = NGramGenerator(
                NGramModel(order=3).train_texts([ex.messages[-1].content for ex in train_all]),
                seed=1,
            )
        else:
            gen = MockGenerator(["haan, theek hai"], seed=1)
    else:
        gen = MockGenerator(
            [e["response"] for e in (exemplars or few_shot)] or ["haan, theek hai"], seed=1
        )
    reply = gen.generate(messages, GeneratorConfig())
    history = [
        *history,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    return reply, history


# ---------------------------------------------------------------------------
# voice studio
# ---------------------------------------------------------------------------
def voice_backends_md() -> str:
    from ..voice.registry import all_backend_statuses

    lines = ["## Voice backends"]
    for b in all_backend_statuses():
        caps = b.get("capabilities", {})
        lines.append(
            f"- **{b['name']}**: available={caps.get('available', False)}, "
            f"langs={len(caps.get('languages', []))}, "
            f"transcript={'required' if caps.get('reference_transcript_required') else 'optional'}, "
            f"license={caps.get('license', '-')}"
        )
    return "\n".join(lines)


def voice_plan(backend: str, languages: str, max_minutes: float) -> str:
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.recording_plan import RecordingPreferences, generate_plan
    from ..voice.registry import get_backend

    caps = VoiceBackendCapabilities(**get_backend(backend).capabilities())
    plan = generate_plan(
        caps,
        RecordingPreferences(
            languages=[x for x in languages.split(",") if x], max_minutes=float(max_minutes)
        ),
        [],
    )
    lines = [
        f"Plan for **{plan.backend}** (~{plan.total_estimated_seconds}s, "
        f"in-budget ~{plan.within_time_budget_seconds}s)"
    ]
    for t in plan.tasks:
        flag = (
            "MANDATORY"
            if t.mandatory
            else ("beyond-budget" if t.beyond_time_budget else "optional")
        )
        lines.append(
            f"- P{t.priority} [{flag}] {t.title} (~{t.est_seconds}s): {', '.join(t.script_ids)}"
        )
    for s in plan.skipped:
        lines.append(f"- skipped {s['language']}: {s['reason']}")
    return "\n".join(lines)


def voice_scripts(language: str) -> str:
    from ..voice.script_library import all_scripts, render

    entries = [e for e in all_scripts() if e.language.value == language]
    return "\n\n".join(f"**{e.id}** [{e.category.value}]\n{render(e, {})}" for e in entries)


def _get_profile():
    from ..voice.metadata import load_profile

    pid = state.get("last_voice_profile_id")
    if not pid:
        raise ValueError("No voice profile yet — import a recording first.")
    return load_profile(pid)


def voice_save_recording(audio_path: str | None, language: str, script_id: str) -> str:
    from ..voice.audio_ingestion import read_audio
    from ..voice.audio_quality import analyze_audio_file, summary_line
    from ..voice.metadata import RecordingEntry, VoiceProfile, save_profile

    if not audio_path:
        return "Record or upload audio first."
    src = Path(audio_path)
    rec_dir = get_settings().recordings_dir
    rec_dir.mkdir(parents=True, exist_ok=True)
    from ..storage.paths import unique_path

    target = unique_path(rec_dir, src.stem[:40] or "recording", src.suffix.lower() or ".wav")
    shutil.copy2(src, target)
    read_audio(target)  # validates readability before accepting the file
    report = analyze_audio_file(target)
    try:
        prof = _get_profile()
    except ValueError:
        prof = VoiceProfile()
    rec = RecordingEntry(
        recording_id=target.stem,
        path=str(target),
        language=language,
        script_id=script_id or None,
        quality_verdict=report.verdict,
        duration_seconds=report.duration_seconds,
        sample_rate=report.sample_rate,
        quality_report=report.model_dump(),
        category="studio",
    )
    prof.recordings.append(rec)
    save_profile(prof)
    state.set("last_voice_profile_id", prof.profile_id)
    return f"Saved `{target.name}` → profile `{prof.profile_id}`: **{report.verdict}** — {summary_line(report)}"


def voice_analyze() -> str:
    from ..voice.audio_quality import analyze_audio_file, summary_line
    from ..voice.metadata import save_profile
    from ..voice.phoneme_coverage import analyze_texts

    try:
        prof = _get_profile()
    except ValueError as e:
        return str(e)
    lines = []
    for rec in prof.recordings:
        if rec.path and Path(rec.path).exists():
            rep = analyze_audio_file(rec.path)
            rec.quality_verdict = rep.verdict
            rec.quality_report = rep.model_dump()
            lines.append(f"- {rec.recording_id}: **{rep.verdict}** — {summary_line(rep)}")
    prof.coverage_report = analyze_texts([r.transcript or "" for r in prof.recordings]).model_dump()
    save_profile(prof)
    return "\n".join(lines) or "No recordings with audio on disk."


def voice_set_transcript(recording_id: str, transcript: str) -> str:
    from ..voice.metadata import save_profile
    from ..voice.transcript_validation import verify_transcript

    try:
        prof = _get_profile()
    except ValueError as e:
        return str(e)
    rec = next((r for r in prof.recordings if r.recording_id == recording_id), None)
    if rec is None:
        return f"Recording {recording_id!r} not found in the profile."
    rec.transcript = transcript
    rec.has_transcript = bool(transcript)
    check = verify_transcript("", transcript, source="user")
    rec.transcript_check = check.model_dump()
    save_profile(prof)
    return f"Transcript saved ({check.verdict}, similarity={check.normalized_similarity})."


def voice_coverage() -> str:
    from ..voice.phoneme_coverage import CoverageReport

    try:
        prof = _get_profile()
    except ValueError as e:
        return str(e)
    if not prof.coverage_report:
        return "Run analysis first."
    report = CoverageReport(**prof.coverage_report)
    lines = ["| category | count | covered | confidence |", "|---|---|---|---|"]
    for k, v in report.categories.items():
        lines.append(f"| {k} | {v.count} | {v.covered} | {v.confidence} |")
    lines.append("")
    lines.extend(report.limitations)
    return "\n".join(lines)


def voice_select_references(prefer_transcript: bool) -> str:
    from ..voice.capabilities import VoiceBackendCapabilities
    from ..voice.metadata import save_profile
    from ..voice.reference_selection import ReferenceCandidate, select_references
    from ..voice.registry import get_backend

    try:
        prof = _get_profile()
    except ValueError as e:
        return str(e)
    caps = VoiceBackendCapabilities(**get_backend(prof.backend).capabilities())
    decision = select_references(
        [
            ReferenceCandidate(
                recording_id=r.recording_id,
                language=r.language,
                quality_verdict=r.quality_verdict,
                duration_seconds=r.duration_seconds,
                has_transcript=r.has_transcript,
                transcript_verdict=(r.transcript_check or {}).get("verdict")
                if r.transcript_check
                else None,
            )
            for r in prof.recordings
        ],
        caps,
        prefer_transcript=prefer_transcript,
    )
    prof.reference_selection = decision.model_dump()
    save_profile(prof)
    lines = [f"Selected: {[c.recording_id for c in decision.selected]}", decision.rationale]
    lines += [f"- rejected {r['recording_id']}: {r['reason']}" for r in decision.rejected]
    return "\n".join(lines)


def voice_consent_grant(name: str) -> str:
    from ..voice.consent import VOICE_CONSENT_STATEMENT, grant_consent

    grant_consent("voice_cloning", {"statement": VOICE_CONSENT_STATEMENT, "by": name or "user"})
    return "Voice-cloning consent recorded. It can be revoked anytime (Settings tab)."


def voice_synthesize(text: str, seed: str) -> tuple[str | None, str]:
    from ..voice.synthesis import synthesize

    try:
        prof = _get_profile()
    except ValueError as e:
        return None, str(e)
    try:
        result = synthesize(
            profile_id=prof.profile_id,
            text=text,
            seed=int(seed) if seed and seed.strip() else None,
        )
    except Exception as e:
        return None, f"Synthesis failed: {e}"
    return result.output_path, f"Written to `{result.output_path}` (backend={result.backend})."


# ---------------------------------------------------------------------------
# app assembly
# ---------------------------------------------------------------------------
def revoke_voice_consent() -> str:
    from ..voice.consent import revoke_consent

    revoke_consent("voice_cloning")
    return "Consent revoked. Synthesis is blocked until consent is granted again."


def build_app():
    import gradio as gr

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\nLocal-first writing-style learning + authorized voice cloning."
        )
        dash = gr.Markdown(dashboard_markdown)
        refresh = gr.Button("Refresh dashboard")
        refresh.click(dashboard_markdown, outputs=dash)

        with gr.Tab("Data import"):
            up = gr.File(label="WhatsApp .txt / .zip", file_types=[".txt", ".zip"])
            import_btn = gr.Button("Import")
            import_md = gr.Markdown()
            part_table = gr.Dataframe(
                headers=["Name", "Messages", "First", "Last"], label="Participants"
            )
            _names = gr.State("")
            author = gr.Textbox(label="Target author name")
            author_btn = gr.Button("Select author")
            author_md = gr.Markdown()
            import_btn.click(import_file, inputs=up, outputs=[import_md, part_table, _names])
            author_btn.click(select_author, inputs=author, outputs=author_md)

        with gr.Tab("Dataset"):
            ctx_n = gr.Slider(0, 10, value=4, step=1, label="Context messages")
            ctx_chars = gr.Slider(200, 4000, value=1200, step=100, label="Max context chars")
            split_mode = gr.Dropdown(
                ["conversation", "chronological"], value="conversation", label="Split mode"
            )
            prep_btn = gr.Button("Prepare dataset + splits")
            prep_md = gr.Markdown()
            stats_btn = gr.Button("Show manifest")
            stats_md = gr.Markdown()
            prep_btn.click(prepare_dataset, inputs=[ctx_n, ctx_chars, split_mode], outputs=prep_md)
            stats_btn.click(dataset_stats, outputs=stats_md)

        with gr.Tab("Style profile"):
            prof_btn = gr.Button("Build style profile")
            prof_md = gr.Markdown()
            prof_btn.click(build_profile, outputs=prof_md)

        with gr.Tab("Evaluation"):
            method = gr.Dropdown(
                ["zero_shot", "style_profile", "few_shot", "style_fewshot", "style_rag", "ngram", "mock"],
                value="style_rag",
                label="Method",
            )
            rag_k = gr.Slider(0, 8, value=3, step=1, label="Retrieved exemplars (style_rag)")
            eval_btn = gr.Button("Run baseline on held-out split")
            eval_md = gr.Markdown()
            eval_btn.click(run_baseline, inputs=[method, rag_k], outputs=eval_md)

        with gr.Tab("Chat playground"):
            chat_state = gr.State([])
            user_box = gr.Textbox(label="You")
            chat_method = gr.Dropdown(
                ["zero_shot", "style_profile", "few_shot", "style_fewshot", "style_rag", "ngram"],
                value="style_rag",
                label="Method",
            )
            send_btn = gr.Button("Send")
            reply_md = gr.Markdown()
            send_btn.click(
                chat_reply,
                inputs=[user_box, chat_method, chat_state],
                outputs=[reply_md, chat_state],
            )

        with gr.Tab("Voice studio"):
            gr.Markdown(voice_backends_md())
            with gr.Row():
                plan_backend = gr.Dropdown(["voxcpm", "chatterbox-multilingual"], value="voxcpm")
                plan_langs = gr.Textbox(value="hi,en", label="Languages")
                plan_minutes = gr.Number(value=20, label="Max minutes")
            plan_btn = gr.Button("Generate recording plan")
            plan_md = gr.Markdown()
            plan_btn.click(
                voice_plan, inputs=[plan_backend, plan_langs, plan_minutes], outputs=plan_md
            )

            with gr.Row():
                script_lang = gr.Dropdown(["hi", "en", "hinglish"], value="hi", label="Scripts")
                scripts_btn = gr.Button("Show scripts")
            scripts_md = gr.Markdown()
            scripts_btn.click(voice_scripts, inputs=script_lang, outputs=scripts_md)

            rec_audio = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Record or upload"
            )
            with gr.Row():
                rec_lang = gr.Dropdown(
                    ["hi", "en", "hinglish", "unknown"], value="hi", label="Language"
                )
                rec_script = gr.Textbox(label="Script id (optional)")
            save_btn = gr.Button("Save recording to profile")
            save_md = gr.Markdown()
            save_btn.click(
                voice_save_recording, inputs=[rec_audio, rec_lang, rec_script], outputs=save_md
            )

            analyze_btn = gr.Button("Analyze recordings")
            analyze_md = gr.Markdown()
            analyze_btn.click(voice_analyze, outputs=analyze_md)

            with gr.Row():
                tr_id = gr.Textbox(label="Recording id")
                tr_text = gr.Textbox(label="Verified transcript")
            tr_btn = gr.Button("Save transcript")
            tr_md = gr.Markdown()
            tr_btn.click(voice_set_transcript, inputs=[tr_id, tr_text], outputs=tr_md)

            cov_btn = gr.Button("Coverage report")
            cov_md = gr.Markdown()
            cov_btn.click(voice_coverage, outputs=cov_md)

            tr_pref = gr.Checkbox(label="Prefer transcript-verified references", value=False)
            sel_btn = gr.Button("Select reference audio")
            sel_md = gr.Markdown()
            sel_btn.click(voice_select_references, inputs=tr_pref, outputs=sel_md)

            consent_name = gr.Textbox(label="Name / alias for the consent record")
            consent_btn = gr.Button("Grant voice-cloning consent")
            consent_md = gr.Markdown()
            consent_btn.click(voice_consent_grant, inputs=consent_name, outputs=consent_md)

            synth_text = gr.Textbox(label="Approved text to speak", lines=3)
            synth_seed = gr.Textbox(label="Seed (optional)")
            synth_btn = gr.Button("Synthesize")
            synth_audio = gr.Audio(label="Output", type="filepath")
            synth_md = gr.Markdown()
            synth_btn.click(
                voice_synthesize, inputs=[synth_text, synth_seed], outputs=[synth_audio, synth_md]
            )

        with gr.Tab("Settings"):
            gr.Markdown(
                f"- Home: `{get_settings().home}`\n"
                f"- Max upload: {get_settings().max_upload_mb} MB\n"
                f"- API bind: {get_settings().api_host}:{get_settings().api_port} "
                f"(localhost by default)\n"
                "- Retention: keep until deleted (configure ECHOTWIN_HOME to isolate)."
            )
            revoke_btn = gr.Button("Revoke voice-cloning consent")
            revoke_md = gr.Markdown()
            revoke_btn.click(revoke_voice_consent, outputs=revoke_md)
    return demo


def main() -> None:
    s = get_settings()
    s.ensure_dirs()
    build_app().launch(server_name=s.ui_host, server_port=s.ui_port, inbrowser=False)


if __name__ == "__main__":
    main()
