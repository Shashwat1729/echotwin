"""Deterministic synthetic demo: end-to-end offline pipeline run.

Uses the checked-in multi-conversation synthetic export; no private data, no model
downloads (n-gram engine), no network.
"""

from __future__ import annotations

from pathlib import Path

from .dataset.builder import DatasetConfig, build_examples, load_jsonl, save_jsonl
from .dataset.splits import SplitConfig, split_dataset
from .engines import NGramEngine
from .evaluation.baseline import heldout_prompts, run_baseline
from .ingestion import WhatsAppParser, load_export, participant_stats, pick_participant
from .ingestion.export_loader import conversation_id_from_name
from .profiling.style_profile import build_style_profile, profile_prompt_fragment, save_profile
from .settings import get_settings
from .storage import db
from .workspace import get_store

DEMO_AUTHOR = "Aarav"


def run_demo(console) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    store = get_store()
    repo_root = Path(__file__).resolve().parents[2]
    export = repo_root / "examples" / "synthetic_multi.zip"

    console.print("[bold]== EchoTwin synthetic demo ==[/bold]")
    bundle = load_export(export)
    parser = WhatsAppParser()
    messages = []
    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"demo-{i}")
        messages.extend(parser.parse_text(chat.chat_text, convo_id).messages)
    console.print(f"1) parsed {len(messages)} messages from {len(bundle.chats)} conversations")

    stats = participant_stats(messages)
    console.print(f"2) participants: {', '.join(s.sender_name for s in stats)}")
    target_id = pick_participant(messages, DEMO_AUTHOR)
    if target_id is None:
        raise RuntimeError(f"Synthetic demo is broken: participant {DEMO_AUTHOR!r} not found.")
    import_id = db.record_import(str(export), "demo", len(messages), target_id, {})
    store.set_pointer("current_import", import_id)
    store.set_pointer("current_author", target_id)
    console.print(f"3) target author: {DEMO_AUTHOR} ({target_id})")

    built = build_examples(messages, target_id, DatasetConfig())
    split = split_dataset(built, SplitConfig())
    console.print(
        f"4) dataset: {len(built.examples)} examples; splits "
        f"train/val/test = {len(split.train)}/{len(split.val)}/{len(split.test)}"
    )
    if split.report.warnings:
        console.print(f"   [yellow]{split.report.warnings[0]}[/yellow]")

    ds_dir = settings.artifacts_dir / "datasets" / "demo"
    save_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_jsonl(split.train, ds_dir / "train.jsonl")
    save_jsonl(split.val, ds_dir / "val.jsonl")
    save_jsonl(split.test, ds_dir / "test.jsonl")
    store.set_pointer("current_dataset", "demo")

    profile = build_style_profile(messages, target_id)
    prof_path = settings.artifacts_dir / "profiles" / "demo-profile.json"
    save_profile(profile, prof_path)
    store.set_pointer("current_profile", str(prof_path))
    console.print(
        f"5) style profile: {profile.n_messages} messages, "
        f"langs={profile.language_use.get('distributions')}"
    )

    prompts = heldout_prompts(load_jsonl(ds_dir / "test.jsonl"))
    if not prompts:
        console.print("6) baseline skipped: the held-out split is empty.")
    else:
        # "plain": the n-gram engine ignores system text and exemplars, so a
        # retrieval method would be mislabelled here.
        records, path, report = run_baseline(
            method="plain",
            generator=NGramEngine.for_dataset("demo"),
            prompts=prompts,
            profile_fragment=profile_prompt_fragment(profile),
            output_dir=settings.artifacts_dir / "evaluations" / "demo",
            model_revision="ngram-demo",
            train_texts=[ex.messages[-1].content for ex in split.train],
        )
        console.print(f"6) plain baseline (n-gram): {len(records)} predictions -> {path}")
        if report.style:
            console.print(
                f"   style char3cos={report.style.char_ngram_cosine_3}, "
                f"emoji ref/gen={report.style.emoji_rate_ref}/{report.style.emoji_rate_gen}"
            )
        if report.contamination:
            console.print(
                f"   contamination: exact={report.contamination['exact_match_rate']} "
                f"near={report.contamination['near_match_rate']} "
                f"verdict={report.contamination['verdict']}"
            )
    console.print(
        "[green]Demo complete.[/green] Next: 'echotwin voice backends' and 'echotwin doctor'."
    )
