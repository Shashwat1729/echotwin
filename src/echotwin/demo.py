"""Deterministic synthetic demo: end-to-end offline pipeline run.

Uses the checked-in synthetic export; no private data, no model downloads
(mock generator), no network.
"""

from __future__ import annotations

from pathlib import Path

from .dataset.builder import DatasetConfig, build_examples, load_jsonl, save_jsonl
from .dataset.retrieval import RetrievalConfig, build_index_from_examples
from .dataset.splits import SplitConfig, split_dataset
from .evaluation.baseline import heldout_prompts, run_baseline
from .ingestion import WhatsAppParser, load_export, participant_stats, pick_participant
from .models.generation import MockGenerator
from .profiling.style_profile import build_style_profile, profile_prompt_fragment, save_profile
from .settings import get_settings
from .storage import db, state


def run_demo(console) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    repo_root = Path(__file__).resolve().parents[2]
    export = repo_root / "examples" / "synthetic_whatsapp.txt"

    console.print("[bold]== EchoTwin synthetic demo ==[/bold]")
    bundle = load_export(export)
    parser = WhatsAppParser()
    result = parser.parse_text(bundle.chats[0].chat_text, "demo")
    console.print(
        f"1) parsed {len(result.messages)} messages (formats: {result.report.detected_formats})"
    )

    stats = participant_stats(result.messages)
    console.print(f"2) participants: {', '.join(s.sender_name for s in stats)}")
    target_id = pick_participant(result.messages, "Aarav")
    if target_id is None:
        raise RuntimeError("Synthetic demo is broken: participant 'Aarav' not found.")
    import_id = db.record_import(
        str(export), "demo", len(result.messages), target_id, result.report.model_dump()
    )
    state.set("last_import_id", import_id)
    state.set("target_sender_id", target_id)
    console.print(f"3) target author: Aarav ({target_id})")

    built = build_examples(result.messages, target_id, DatasetConfig())
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
    state.set("last_dataset_id", "demo")

    profile = build_style_profile(result.messages, target_id)
    prof_path = settings.artifacts_dir / "profiles" / "demo-profile.json"
    save_profile(profile, prof_path)
    state.set("last_profile_id", str(prof_path))
    console.print(
        f"5) style profile: {profile.n_messages} messages, "
        f"langs={profile.language_use.get('distributions')}"
    )

    prompts = heldout_prompts(load_jsonl(ds_dir / "test.jsonl"))
    gen = MockGenerator(exemplar_texts=[ex.messages[-1].content for ex in split.train[:5]], seed=7)
    index = build_index_from_examples(built.examples)
    if prompts:
        records, path, report = run_baseline(
            method="style_rag",
            generator=gen,
            prompts=prompts,
            profile_fragment=profile_prompt_fragment(profile),
            output_dir=settings.artifacts_dir / "evaluations" / "demo",
            model_revision="mock-1",
            retrieval_index=index,
            retrieval_cfg=RetrievalConfig(k=3),
            train_texts=[ex.messages[-1].content for ex in split.train],
        )
        console.print(f"6) style_rag baseline (mock): {len(records)} predictions -> {path}")
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
    else:
        console.print(
            "6) baseline skipped: the synthetic export is a single conversation, so the "
            "held-out split is empty (import multiple chats for a real held-out set)."
        )
    console.print(
        "[green]Demo complete.[/green] Next: 'echotwin voice backends' and 'echotwin doctor'."
    )
