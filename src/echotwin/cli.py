"""EchoTwin command line interface.

Heavy dependencies (torch/transformers/soundfile/voxcpm) are imported lazily
inside the commands that need them, so ``echotwin --help`` and all data-plane
commands work on a minimal install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .dataset.builder import DatasetConfig, build_examples
from .dataset.builder import load_jsonl as load_examples_jsonl
from .dataset.builder import save_jsonl as save_examples_jsonl
from .dataset.splits import SplitConfig, make_manifest, split_dataset
from .evaluation.baseline import (
    RETRIEVAL_METHODS,
    heldout_prompts,
    load_predictions,
    run_baseline,
)
from .exceptions import EchoTwinError, ExportValidationError
from .ingestion import (
    WhatsAppParser,
    load_export,
    participant_stats,
    pick_participant,
)
from .ingestion.schema import Message
from .models.generation import GeneratorConfig, MockGenerator, TextGenerator
from .preference.dpo import PreferencePair, load_pairs, validate_pairs
from .privacy.redact import RedactionConfig
from .profiling.style_profile import build_style_profile, profile_prompt_fragment, save_profile
from .storage import db, state

app = typer.Typer(
    name="echotwin",
    help="EchoTwin — local-first writing-style learning and authorized voice cloning.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

voice_app = typer.Typer(help="Voice cloning: backends, recording, analysis, synthesis.")
dataset_app = typer.Typer(help="Dataset preparation and validation.")
train_app = typer.Typer(help="Training commands (SFT/LoRA/DPO).")
import_app = typer.Typer(help="Data import commands.")
evaluate_app = typer.Typer(help="Evaluation commands.")
prefs_app = typer.Typer(help="Preference annotation.")
artifacts_app = typer.Typer(help="Artifact management.")
privacy_app = typer.Typer(help="Privacy export and deletion.")

app.add_typer(voice_app, name="voice")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(import_app, name="import")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(prefs_app, name="preferences")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(privacy_app, name="privacy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fail(msg: str, code: int = 1) -> NoReturn:
    err_console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


def _load_messages_for_state() -> tuple[list[Message], str, str | None]:
    import_id = state.get("last_import_id")
    if not import_id:
        _fail("No import yet. Run: echotwin import whatsapp ./export.zip")
    imp = db.latest_import()
    if imp is None or imp["import_id"] != import_id:
        _fail("Import record missing; re-run the import command.")
    from .settings import get_settings

    messages_path = get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"
    if not messages_path.exists():
        _fail(f"Stored messages for import {import_id} are missing; re-run the import command.")
    messages = _read_messages(messages_path)
    target = imp["target_sender_id"] or state.get("target_sender_id")
    return messages, import_id, target


def _require_target() -> tuple[list[Message], str, str]:
    messages, import_id, target = _load_messages_for_state()
    if not target:
        _fail('No target author selected yet. Run: echotwin select-author "Your Name"')
    return messages, import_id, target


def _redaction() -> RedactionConfig:
    from .settings import get_settings

    s = get_settings()
    return RedactionConfig(enabled=s.redact_pii)


def _dump_messages(messages: list[Message], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for m in messages:
            fh.write(m.model_dump_json() + "\n")


def _read_messages(path: Path) -> list[Message]:
    return [
        Message.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Environment diagnostics: deps, storage, backends, optional hardware."""
    from .check_env import doctor_report

    report = doctor_report()
    for line in report["lines"]:
        console.print(line)
    if not report["healthy"]:
        raise typer.Exit(1)


@app.command()
def demo() -> None:
    """Run the synthetic demo end-to-end (no private data needed)."""
    from .demo import run_demo

    run_demo(console)


@app.command("version")
def version() -> None:
    console.print(f"echotwin {__version__}")


# ---------------------------------------------------------------------------
# import / participants
# ---------------------------------------------------------------------------
@import_app.command("whatsapp")
def import_whatsapp(
    path: Path = typer.Argument(
        ..., help="WhatsApp export: _chat.txt, .zip (may contain many chats), or a directory"
    ),
    conversation_prefix: str = typer.Option(
        "chat", "--conversation-id", help="Fallback conversation label when the filename gives none"
    ),
) -> None:
    """Import a WhatsApp TXT or ZIP export (every chat in the file is imported)."""
    try:
        bundle = load_export(path)
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    parser = WhatsAppParser()
    all_messages: list[Message] = []
    total_report: dict[str, Any] = {
        "total_lines": 0,
        "parsed_messages": 0,
        "continuation_lines": 0,
        "orphan_lines": 0,
        "system_messages": 0,
        "attachment_messages": 0,
        "deleted_messages": 0,
        "call_events": 0,
        "unparsed_timestamps": 0,
        "malformed_lines": 0,
        "duplicates": 0,
        "detected_formats": [],
        "conversations": [],
    }
    from .ingestion.export_loader import conversation_id_from_name

    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"{conversation_prefix}-{i}")
        try:
            result = parser.parse_text(chat.chat_text, convo_id)
        except ExportValidationError:
            continue  # a non-chat txt inside a zip is skipped, not fatal
        all_messages.extend(result.messages)
        rep = result.report
        total_report["total_lines"] += rep.total_lines
        total_report["parsed_messages"] += rep.parsed_messages
        total_report["continuation_lines"] += rep.continuation_lines
        total_report["orphan_lines"] += rep.orphan_lines
        total_report["system_messages"] += rep.system_messages
        total_report["attachment_messages"] += rep.attachment_messages
        total_report["deleted_messages"] += rep.deleted_messages
        total_report["call_events"] += rep.call_events
        total_report["unparsed_timestamps"] += rep.unparsed_timestamps
        total_report["malformed_lines"] += rep.malformed_lines
        total_report["duplicates"] += rep.duplicates
        for f in rep.detected_formats:
            if f not in total_report["detected_formats"]:
                total_report["detected_formats"].append(f)
        total_report["conversations"].append(convo_id)

    if not all_messages:
        _fail("No messages parsed from any chat in the export.", ExportValidationError.exit_code)

    import_id = db.record_import(
        str(path), conversation_prefix, len(all_messages), None, total_report
    )
    state.set("last_import_id", import_id)
    from .settings import get_settings

    out_dir = get_settings().artifacts_dir / "imports" / import_id
    _dump_messages(all_messages, out_dir / "messages.jsonl")
    console.print(
        f"[green]Imported[/green] {len(all_messages)} messages from "
        f"{len(bundle.chats)} conversation(s): {', '.join(total_report['conversations'])}"
    )
    console.print(f"formats detected: {total_report['detected_formats']}")
    console.print(
        f"system={total_report['system_messages']} attachments={total_report['attachment_messages']} "
        f"deleted={total_report['deleted_messages']} malformed={total_report['malformed_lines']}"
    )
    console.print("Next: [bold]echotwin participants[/bold]")


@app.command()
def participants() -> None:
    """List participants with statistics (target author is never auto-selected)."""
    messages, _import_id, _target = _load_messages_for_state()
    table = Table(title="Participants")
    table.add_column("Name")
    table.add_column("Messages")
    table.add_column("Range")
    table.add_column("Mean length")
    table.add_column("Languages")
    for st in participant_stats(messages):
        langs = ", ".join(f"{k}:{v}" for k, v in sorted(st.language_distribution.items()))
        table.add_row(
            st.sender_name,
            str(st.message_count),
            f"{st.first_timestamp} .. {st.last_timestamp}",
            f"{st.mean_length:.0f}",
            langs,
        )
    console.print(table)


@app.command("select-author")
def select_author(name: str = typer.Argument(..., help="Exact participant name or id")) -> None:
    """Select the target author (the person whose style you will learn)."""
    messages, _import_id, _target = _load_messages_for_state()
    sid = pick_participant(messages, name)
    if sid is None:
        _fail(f"Participant '{name}' not found. Run 'echotwin participants'.")
    latest = db.latest_import()
    if latest is None:
        _fail("Import record missing; re-run the import command.")
    db.update_import_target(latest["import_id"], sid)
    state.set("target_sender_id", sid)
    console.print(f"[green]Target author set[/green] to {name!r} (id={sid})")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@dataset_app.command("prepare")
def dataset_prepare(
    context_messages: int = typer.Option(4),
    max_context_chars: int = typer.Option(1200),
    min_target_chars: int = typer.Option(2),
    max_target_chars: int = typer.Option(800),
    no_dedup: bool = typer.Option(False, "--no-dedup"),
) -> None:
    """Build the contextual training dataset and leakage-resistant splits."""
    from .settings import get_settings

    messages, import_id, target = _require_target()
    cfg = DatasetConfig(
        context_messages=context_messages,
        max_context_chars=max_context_chars,
        min_target_chars=min_target_chars,
        max_target_chars=max_target_chars,
        exact_dedup=not no_dedup,
    )
    built = build_examples(messages, target, cfg)
    split = split_dataset(built, SplitConfig())
    ds_id = f"dataset-{import_id}"
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    save_examples_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_examples_jsonl(split.train, ds_dir / "train.jsonl")
    save_examples_jsonl(split.val, ds_dir / "val.jsonl")
    save_examples_jsonl(split.test, ds_dir / "test.jsonl")
    manifest = make_manifest(
        dataset_id=ds_id,
        dataset_path=ds_dir / "examples.jsonl",
        split=split,
        built=built,
        target_sender_id=target,
        split_cfg=SplitConfig(),
        echotwin_version=__version__,
    )
    (ds_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    state.set("last_dataset_id", ds_id)
    console.print(
        f"[green]Dataset built[/green]: {manifest.counts['n_examples']} examples "
        f"from {manifest.counts['conversations']} conversation(s)"
    )
    console.print(
        f"splits: train={len(split.train)} val={len(split.val)} test={len(split.test)} "
        f"(leakage check passed={split.report.leakage_check_passed})"
    )
    for w in split.report.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


@dataset_app.command("stats")
def dataset_stats() -> None:
    """Show dataset statistics."""
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet. Run 'echotwin dataset prepare'.")
    manifest_path = get_settings().artifacts_dir / "datasets" / ds_id / "manifest.json"
    if not manifest_path.exists():
        _fail("Dataset manifest missing; rebuild the dataset.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.print(json.dumps(manifest, indent=2, ensure_ascii=False))


@dataset_app.command("validate")
def dataset_validate() -> None:
    """Validate the current dataset (JSONL parse + leakage + duplicates)."""
    from .dataset.builder import load_jsonl
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    problems: list[str] = []
    try:
        train = load_jsonl(ds_dir / "train.jsonl")
        val = load_jsonl(ds_dir / "val.jsonl")
        test = load_jsonl(ds_dir / "test.jsonl")
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    train_c = {ex.metadata.conversation_id for ex in train}
    val_c = {ex.metadata.conversation_id for ex in val}
    test_c = {ex.metadata.conversation_id for ex in test}
    if train_c & test_c or train_c & val_c or val_c & test_c:
        problems.append("Leakage: conversation ids overlap across splits")
    keys = [ex.key() for ex in train]
    if len(keys) != len(set(keys)):
        problems.append("Duplicate examples in train split")
    for ex in train:
        if not ex.messages or ex.messages[-1].role != "assistant":
            problems.append(
                f"Example ends with non-assistant turn: {ex.metadata.source_message_id}"
            )
            break
    if problems:
        for p in problems:
            console.print(f"[red]FAIL[/red] {p}")
        raise typer.Exit(1)
    console.print("[green]PASS[/green] dataset validation (no leakage, no dupes, roles valid)")


# ---------------------------------------------------------------------------
# profile / evaluation / training stubs wiring
# ---------------------------------------------------------------------------
@app.command("profile")
def profile_build() -> None:
    """Build the interpretable style profile."""
    from .settings import get_settings

    messages, _import_id, target = _require_target()
    profile = build_style_profile(messages, target)
    out = get_settings().artifacts_dir / "profiles" / f"profile-{target[:8]}.json"
    save_profile(profile, out)
    state.set("last_profile_id", str(out))
    console.print(f"[green]Style profile saved[/green] to {out}")
    console.print(profile_prompt_fragment(profile))


@evaluate_app.command("baseline")
def evaluate_baseline(
    method: str = typer.Option(
        "style_rag",
        help="zero_shot | style_profile | few_shot | style_fewshot | style_rag | ngram | mock",
    ),
    model: str = typer.Option(
        "", help="registry name or org/model id (empty = deterministic mock)"
    ),
    max_prompts: int = typer.Option(50),
    few_shot: int = typer.Option(5),
    rag_k: int = typer.Option(3, help="retrieved exemplars for style_rag methods"),
) -> None:
    """Run a baseline on the frozen held-out split."""
    from .dataset.builder import load_jsonl
    from .dataset.retrieval import RetrievalConfig, build_index_from_examples
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    test = load_jsonl(ds_dir / "test.jsonl")
    train = load_jsonl(ds_dir / "train.jsonl")
    all_examples = load_jsonl(ds_dir / "examples.jsonl")
    prompts = heldout_prompts(test, max_prompts=max_prompts)
    if not prompts:
        _fail("No held-out prompts available; the split is too small.")
    fragment = None
    prof_path = state.get("last_profile_id")
    if prof_path and Path(prof_path).exists():
        from .profiling.style_profile import load_profile

        fragment = profile_prompt_fragment(load_profile(prof_path))
    if model:
        from .models.generation import TransformersGenerator
        from .models.registry import get_model_spec

        generator: TextGenerator = TransformersGenerator.from_spec(model)
        revision = get_model_spec(model).hf_model_id
    elif method == "ngram":
        from .models.ngram import NGramGenerator, NGramModel

        if not train:
            _fail("Train split is empty; import multiple chats first.")
        ngram_model = NGramModel(order=3).train_texts([ex.messages[-1].content for ex in train])
        generator = NGramGenerator(ngram_model, seed=7)
        revision = "ngram-o3"
    else:
        exemplars = [
            {
                "context": " | ".join(t.content for t in ex.messages[:-1]),
                "response": ex.messages[-1].content,
            }
            for ex in train[: max(few_shot * 4, few_shot)]
        ]
        generator = MockGenerator(exemplar_texts=[e["response"] for e in exemplars], seed=7)
        revision = "mock-1"
    index = build_index_from_examples(all_examples)
    out_dir = get_settings().artifacts_dir / "evaluations" / ds_id
    records, path, report = run_baseline(
        method=method,
        generator=generator,
        prompts=prompts,
        profile_fragment=fragment,
        few_shot=[
            {
                "context": " | ".join(t.content for t in ex.messages[:-1]),
                "response": ex.messages[-1].content,
            }
            for ex in train[:few_shot]
        ],
        output_dir=out_dir,
        model_revision=revision,
        retrieval_index=index if method in RETRIEVAL_METHODS else None,
        retrieval_cfg=RetrievalConfig(k=rag_k),
        train_texts=[ex.messages[-1].content for ex in train],
    )
    console.print(
        f"[green]{method}[/green] baseline complete: {len(records)} predictions -> {path}"
    )
    if report.style:
        s = report.style
        console.print(
            f"style: char3cos={s.char_ngram_cosine_3} punct_dist={s.punctuation_distance} "
            f"emoji ref/gen={s.emoji_rate_ref}/{s.emoji_rate_gen} "
            f"phrase_repro={s.phrase_reproduction.get('reproduced')}"
        )
    if report.contamination:
        c = report.contamination
        console.print(
            f"contamination: exact={c['exact_match_rate']} near={c['near_match_rate']} "
            f"verdict={c['verdict']}"
        )
    if report.sample_size_warning:
        console.print(f"[yellow]{report.sample_size_warning}[/yellow]")


@evaluate_app.command("pairs")
def evaluate_pairs(
    a: str = typer.Option(..., "--a", help="method name of run A (predictions file stem)"),
    b: str = typer.Option(..., "--b", help="method name of run B"),
    judge: bool = typer.Option(False, help="use the local model as judge (labeled model-based)"),
    limit: int = typer.Option(20),
) -> None:
    """Blinded pairwise comparison of two stored prediction runs."""
    from .evaluation.pairwise import build_pairs, judge_with_model, save_judgments, summarize
    from .models.generation import GeneratorConfig, MockGenerator
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    test = load_examples_jsonl(ds_dir / "test.jsonl")
    prompts = heldout_prompts(test, max_prompts=limit)
    ev_dir = get_settings().artifacts_dir / "evaluations" / ds_id
    candidates_a = sorted(ev_dir.glob(f"predictions_{a}_*.jsonl"))
    candidates_b = sorted(ev_dir.glob(f"predictions_{b}_*.jsonl"))
    if not candidates_a or not candidates_b:
        _fail(
            f"Missing prediction runs for '{a}' or '{b}' in {ev_dir}. Run 'echotwin evaluate baseline' first."
        )
    recs_a = {r["index"]: r for r in load_predictions(candidates_a[-1])}
    recs_b = {r["index"]: r for r in load_predictions(candidates_b[-1])}
    outputs_by_method = {
        a: [recs_a[i].get("generated", "") for i in sorted(recs_a)],
        b: [recs_b[i].get("generated", "") for i in sorted(recs_b)],
    }
    pairs = build_pairs(prompts, outputs_by_method, a, b)
    judgments = []
    gen = MockGenerator(seed=3)
    if judge:
        cfg = GeneratorConfig(max_new_tokens=8)
        judgments = [judge_with_model(gen, p, cfg) for p in pairs]
    out = get_settings().artifacts_dir / "evaluations" / ds_id / "pairwise"
    n = save_judgments(judgments, out / "judgments.jsonl")
    result = summarize(judgments, model_judged=n)
    console.print(json.dumps(result.model_dump(), indent=1))
    console.print(
        "[dim]Use the Web UI Evaluation Studio for human-blinded scoring; "
        "model-judged wins measure the judge, not human taste.[/dim]"
    )


@evaluate_app.command("compare")
def evaluate_compare() -> None:
    """Compare stored prediction runs on identical prompts."""
    from .evaluation.baseline import compare_runs
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ev_dir = get_settings().artifacts_dir / "evaluations" / ds_id
    paths = sorted(ev_dir.glob("predictions_*.jsonl"))
    if len(paths) < 2:
        _fail(
            f"Need at least 2 prediction runs in {ev_dir}; run 'echotwin evaluate baseline' twice."
        )
    table = compare_runs(paths)
    console.print(
        json.dumps(
            {
                "runs": list(table.get("outputs", {})),
                "aligned": table.get("aligned_prompt_indices"),
            },
            indent=2,
        )[:4000]
    )


@train_app.command("ngram")
def train_ngram(
    order: int = typer.Option(3, help="n-gram order (1-5)"),
    smoothing: float = typer.Option(0.1, help="add-k smoothing"),
) -> None:
    """Train a CPU-only word n-gram style model on the current train split."""
    from .dataset.builder import load_jsonl
    from .models.ngram import NGramModel
    from .settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    train_ex = load_jsonl(ds_dir / "train.jsonl")
    if not train_ex:
        _fail("Train split is empty; import multiple chats first.")
    model = NGramModel(order=order, smoothing_k=smoothing).train_texts(
        [ex.messages[-1].content for ex in train_ex]
    )
    out = get_settings().artifacts_dir / "models" / f"ngram-{ds_id}-o{order}.json"
    model.save(out)
    db.record_artifact("ngram_model", out, {"dataset": ds_id, "order": order, "targets": model.n_targets})
    console.print(
        f"[green]n-gram trained[/green]: order={order} targets={model.n_targets} "
        f"vocab={len(model.vocab)} -> {out}"
    )


@train_app.command("lora")
def train_lora(
    config: Path = typer.Option(..., "--config", help="YAML training config"),
) -> None:
    """Train with SFT/LoRA/QLoRA (needs the [train] extra and model download)."""
    import yaml

    from .dataset.builder import load_jsonl
    from .settings import get_settings
    from .training.sft import TrainingConfig, run_sft

    cfg_dict = yaml.safe_load(config.read_text(encoding="utf-8"))
    train_cfg = TrainingConfig(**cfg_dict)
    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    train_ex = load_jsonl(ds_dir / "train.jsonl")
    val_ex = load_jsonl(ds_dir / "val.jsonl")
    console.print(
        f"Starting {train_cfg.method} training on {len(train_ex)} examples "
        f"(val={len(val_ex)}); this needs GPU/model downloads."
    )
    record = run_sft(train_ex, val_ex, train_cfg)
    console.print(f"[green]Training finished[/green]: {record.status} -> {record.output_dir}")


@train_app.command("dpo")
def train_dpo(
    config: Path = typer.Option(..., "--config"),
    pairs: Path = typer.Option(..., "--pairs", help="preference pairs JSONL"),
    allow_demo: bool = typer.Option(False, help="allow 'synthetic-demo' pairs (dry run only)"),
) -> None:
    """Preference optimization (DPO) with TRL when available."""
    import yaml

    from .preference.dpo import train_dpo as _train_dpo

    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    pair_list = load_pairs(pairs)
    v = validate_pairs(pair_list)
    if not v.valid:
        _fail(f"Invalid preference pairs: {v.errors[:3]}")
    if v.human_share < 1.0 and not allow_demo:
        _fail("Pairs are not human-labeled. Re-run with --allow-demo for a dry run.")
    console.print(f"Running DPO on {len(pair_list)} pairs (provenances: {v.provenances}).")
    result = _train_dpo(
        model_id=cfg.get("model_id", "Qwen/Qwen2.5-3B-Instruct"),
        pairs=pair_list,
        output_dir=cfg.get("output_dir", "artifacts/experiments/dpo"),
        beta=cfg.get("beta", 0.1),
        epochs=cfg.get("epochs", 1),
        learning_rate=cfg.get("learning_rate", 5e-5),
        allow_demo=allow_demo,
    )
    console.print(json.dumps(result, indent=2, default=str))


@prefs_app.command("annotate")
def prefs_annotate() -> None:
    """Create/review preference pairs interactively (TTY) — placeholder guidance."""
    console.print(
        "Preference annotation is available in the Web UI (Evaluation Studio) and API. "
        "This CLI command prints a draft scaffold; edit 'examples/synthetic_preference_pairs.jsonl' "
        "or use the UI. Synthetic-demo pairs are blocked from DPO unless --allow-demo."
    )
    console.print(json.dumps(PreferencePair.model_json_schema()["properties"], indent=1)[:1500])


@prefs_app.command("validate")
def prefs_validate(pairs: Path = typer.Argument(...)) -> None:
    v = validate_pairs(load_pairs(pairs))
    console.print(json.dumps(v.model_dump(), indent=1))


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------
@voice_app.command("backends")
def voice_backends() -> None:
    """List voice backends with their real capabilities."""
    from .voice.registry import all_backend_statuses

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
    from .voice.script_library import all_scripts

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
    from .voice.capabilities import VoiceBackendCapabilities
    from .voice.recording_plan import InventoryItem, RecordingPreferences, generate_plan
    from .voice.registry import get_backend

    try:
        backend_obj = get_backend(backend)
        caps = VoiceBackendCapabilities(**backend_obj.capabilities())
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    inventory: list[InventoryItem] = []
    if profile:
        from .voice.metadata import load_profile

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
    from .voice.capabilities import VoiceBackendCapabilities
    from .voice.recording_plan import RecordingPreferences, generate_plan
    from .voice.registry import get_backend

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
            from .voice.script_library import get_script

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

    from .voice.audio_ingestion import read_audio
    from .voice.metadata import RecordingEntry, VoiceProfile, load_profile
    from .voice.metadata import save_profile as save_voice_profile
    from .voice.phoneme_coverage import analyze_texts

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
        from .voice.audio_quality import analyze_wave_array

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
    prof.coverage_report = analyze_texts([r.transcript or "" for r in prof.recordings]).model_dump()
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
    from .voice.audio_quality import analyze_audio_file, summary_line
    from .voice.metadata import load_profile
    from .voice.metadata import save_profile as save_voice_profile
    from .voice.phoneme_coverage import analyze_texts

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
    prof.coverage_report = analyze_texts([r.transcript or "" for r in prof.recordings]).model_dump()
    save_voice_profile(prof)
    console.print("[green]Analysis complete[/green]")


@voice_app.command("coverage")
def voice_coverage(profile: str = typer.Option(None)) -> None:
    """Show the phonetic/language coverage report and gaps."""
    from .voice.metadata import load_profile
    from .voice.phoneme_coverage import CoverageReport

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
    from .voice.capabilities import VoiceBackendCapabilities
    from .voice.metadata import load_profile
    from .voice.phoneme_coverage import CoverageReport
    from .voice.recording_plan import InventoryItem, RecordingPreferences, generate_plan
    from .voice.registry import get_backend

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
    from .voice.consent import check_consent
    from .voice.metadata import load_profile
    from .voice.synthesis import synthesize

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
    from .voice.evaluation import evaluate_synthesis

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
    from .voice.metadata import load_profile
    from .voice.synthesis import synthesize

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


# ---------------------------------------------------------------------------
# chat / serve / artifacts / privacy
# ---------------------------------------------------------------------------
@app.command()
def chat(
    model: str = typer.Option("", help="registry name or org/model (empty = offline mock)"),
    adapter: Path = typer.Option(None, help="LoRA adapter dir to merge for inference"),
    profile: Path = typer.Option(None, help="style profile JSON for style_fewshot mode"),
    style_fewshot: bool = typer.Option(False),
) -> None:
    """Interactive chat playground (context -> text; synthesis stays separate)."""
    from .models.generation import MockGenerator, TransformersGenerator
    from .profiling.style_profile import load_profile, profile_prompt_fragment

    fragment = None
    if profile and Path(profile).exists():
        fragment = profile_prompt_fragment(load_profile(profile))
    generator: TextGenerator
    if model:
        from .models.registry import get_model_spec

        generator = TransformersGenerator.from_spec(model)
        revision = get_model_spec(model).hf_model_id
    else:
        generator = MockGenerator(seed=1)
        revision = "mock-1"
    console.print(f"[dim]chat with {revision}; type /exit to quit[/dim]")
    history: list[dict] = []
    while True:
        try:
            line = console.input("[bold blue]you>[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() in {"/exit", "/quit"}:
            break
        if not line.strip():
            continue
        history.append({"role": "user", "content": line})
        system_content = "Reply in the user's natural chat style."
        if fragment and style_fewshot:
            system_content += f"\n{fragment}"
        msgs = [{"role": "system", "content": system_content}, *history[-8:]]
        reply = generator.generate(msgs, GeneratorConfig())
        history.append({"role": "assistant", "content": reply})
        console.print(f"[bold green]twin>[/bold green] {reply}")


@app.command()
def ui(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    share: bool = typer.Option(False, help="Allow a public Gradio share link (off by default)"),
) -> None:
    """Launch the local Gradio studio (localhost by default; never shared unless asked)."""
    from .settings import get_settings

    s = get_settings()
    if share:
        err_console.print(
            "[yellow]Warning: --share exposes your studio (and potentially private "
            "recordings) to the public internet.[/yellow]"
        )
    from .ui.app import build_app

    s.ensure_dirs()
    build_app().launch(
        server_name=host or s.ui_host,
        server_port=port or s.ui_port,
        inbrowser=False,
        share=share,
    )


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Run the local FastAPI service (localhost by default)."""
    import uvicorn

    from .settings import get_settings

    s = get_settings()
    uvicorn.run(
        "echotwin.api.app:app",
        host=host or s.api_host,
        port=port or s.api_port,
        log_level="info",
    )


@artifacts_app.command("list")
def artifacts_list(kind: str = typer.Option(None)) -> None:
    table = Table(title="Artifacts")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Path")
    table.add_column("Created")
    for a in db.list_artifacts(kind):
        table.add_row(a["artifact_id"], a["kind"], a["path"], a["created_at"])
    console.print(table)


@artifacts_app.command("delete")
def artifacts_delete(artifact_id: str) -> None:
    try:
        result = db.delete_artifact(artifact_id)
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)
    console.print(json.dumps(result, indent=1))


@privacy_app.command("export")
def privacy_export(out: Path = typer.Option(Path("echotwin-export"))) -> None:
    """Export all local EchoTwin data (profiles, artifacts, consents) to a folder."""
    from .privacy_exporter import export_all

    export_all(out, console)


@privacy_app.command("delete")
def privacy_delete(
    scope: str = typer.Option(
        ..., help="artifacts:voice | artifacts:datasets | profile:<voice_id> | all-state"
    ),
    yes: bool = typer.Option(False, "--yes", help="confirm deletion"),
) -> None:
    """Delete data (explicit, irreversible). Consent stays recorded."""
    from .privacy_exporter import delete_scope

    if not yes:
        _fail("Refusing to delete without --yes")
    delete_scope(scope, console)


def main() -> None:
    try:
        app()
    except EchoTwinError as e:
        _fail(str(e), e.exit_code)


if __name__ == "__main__":
    main()
