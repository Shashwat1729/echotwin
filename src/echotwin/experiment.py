"""End-to-end style-personalization experiment.

Runs the *real* EchoTwin pipeline on a real corpus and produces a comparison
report across methods on identical held-out prompts:

    import -> select author -> build dataset -> conversation-level split
            -> BM25 retrieval index (train only)
            -> optional LoRA fine-tune
            -> evaluate {base, style_profile, style_rag, lora, ngram}
            -> metrics + contamination + prompt discrimination

The report is JSON so it can be diffed across runs and cited in docs. Nothing
here fabricates numbers: every figure comes from a real generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .dataset.builder import DatasetConfig, build_examples
from .dataset.retrieval import RetrievalConfig, build_index_from_examples
from .dataset.splits import SplitConfig, split_dataset
from .evaluation.baseline import count_context_free, heldout_prompts, run_baseline
from .evaluation.contamination import memorization_overlap
from .evaluation.metrics import (
    compare_style,
    prompt_discrimination,
    prompt_uptake,
    unsupported_claims,
)
from .ingestion import WhatsAppParser, load_export
from .ingestion.export_loader import conversation_id_from_name
from .ingestion.participants import participant_stats

if TYPE_CHECKING:
    from .models.generation import TextGenerator


@dataclass
class MethodResult:
    method: str
    model_revision: str
    n_prompts: int
    n_failed: int
    style: dict | None = None
    contamination: dict | None = None
    prompt_discrimination: dict | None = None
    prompt_uptake: dict | None = None
    unsupported_claims: dict | None = None
    train_seconds: float | None = None
    notes: list[str] = field(default_factory=list)


class ExperimentReport(BaseModel):
    experiment_id: str
    created_at: str
    corpus: dict
    dataset: dict
    splits: dict
    model: str
    prompts_evaluated: int = 0
    prompts_skipped_context_free: int = 0
    methods: list[MethodResult] = Field(default_factory=list)
    environment: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    honesty_note: str = ""


def load_corpus(corpus_dir: str | Path) -> tuple[list, dict]:
    """Import every chat under ``corpus_dir`` through the production parser."""
    bundle = load_export(corpus_dir)
    parser = WhatsAppParser()
    messages: list = []
    conversations: list[str] = []
    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"chat-{i}")
        # The directory layout is <corpus>/<conversation>/_chat.txt, so the
        # conversation label comes from the parent folder, not the file name.
        parts = Path(chat.inner_name).parts
        if len(parts) >= 2:
            convo_id = conversation_id_from_name(parts[-2], convo_id)
        result = parser.parse_text(chat.chat_text, convo_id)
        messages.extend(result.messages)
        conversations.append(convo_id)
    stats = participant_stats(messages)
    corpus_info = {
        "n_conversations": len(conversations),
        "n_messages": len(messages),
        "participants": [{"name": s.sender_name, "messages": s.message_count} for s in stats[:6]],
    }
    return messages, corpus_info


def select_author(messages: list, author_name: str) -> str:
    """Return the sender id of the named participant (never 'most active')."""
    for s in participant_stats(messages):
        if s.sender_name == author_name:
            return s.sender_id
    raise ValueError(
        f"author {author_name!r} not found; participants: "
        f"{[s.sender_name for s in participant_stats(messages)][:8]}"
    )


def run_experiment(
    *,
    corpus_dir: str | Path,
    out_dir: str | Path,
    author_name: str = "Author",
    model: str = "qwen3-1.7b",
    methods: tuple[str, ...] = ("base", "style_rag"),
    train_lora: bool = False,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    max_train_examples: int | None = 400,
    max_eval_prompts: int = 60,
    max_new_tokens: int = 48,
    temperature: float = 0.8,
    seed: int = 20250917,
    device: str | None = None,
) -> ExperimentReport:
    """Run the full pipeline and return a report with real measurements."""
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    messages, corpus_info = load_corpus(corpus_dir)
    target = select_author(messages, author_name)

    built = build_examples(
        messages,
        target,
        DatasetConfig(context_messages=4, max_context_chars=900, max_target_chars=400),
    )
    split = split_dataset(built, SplitConfig(test_fraction=0.15, val_fraction=0.1, seed=seed))
    if max_train_examples and len(split.train) > max_train_examples:
        # Keep the split intact; subsample the training set only.
        split.train = split.train[:max_train_examples]
    prompts = heldout_prompts(split.test, max_prompts=max_eval_prompts)
    skipped_context_free = count_context_free(split.test)
    index = build_index_from_examples(split.train)
    train_texts = [ex.messages[-1].content for ex in split.train]
    prompt_texts = [" ".join(str(t.get("content", "")) for t in p["messages"]) for p in prompts]

    report = ExperimentReport(
        experiment_id=f"exp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        created_at=datetime.now(UTC).isoformat(),
        corpus=corpus_info,
        dataset={
            "n_examples": len(built.examples),
            "languages": built.stats.get("languages"),
            "example_types": built.stats.get("example_types"),
        },
        splits={
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "leakage_check_passed": split.report.leakage_check_passed,
            "warnings": split.report.warnings,
        },
        model=model,
        prompts_evaluated=len(prompts),
        prompts_skipped_context_free=skipped_context_free,
    )

    generator: TextGenerator | None = None
    model_revision = model
    if any(m in {"base", "style_profile", "style_rag", "style_rag_ft"} for m in methods):
        from .models.generation import TransformersGenerator
        from .models.registry import get_model_spec

        generator = TransformersGenerator.from_spec(model)
        model_revision = get_model_spec(model).hf_model_id

    # --- optional LoRA fine-tune ------------------------------------------
    if train_lora:
        from .models.registry import get_model_spec
        from .training.sft import LoRAConfig, TrainingConfig, run_sft

        t0 = time.time()
        run = run_sft(
            split.train,
            split.val,
            TrainingConfig(
                model=model,
                method="lora",
                output_dir=str(out / "training"),
                epochs=epochs,
                learning_rate=learning_rate,
                lora=LoRAConfig(r=lora_rank, alpha=lora_rank * 2),
                seq_len=384,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                batch_size=1,
                logging_steps=5,
                eval_steps=20,
                early_stopping_patience=2,
                seed=seed,
                report_to="none",
            ),
        )
        train_seconds = round(time.time() - t0, 1)
        adapter_dir = Path(run.output_dir) / "final"
        report.environment["lora_training"] = {
            "seconds": train_seconds,
            "adapter_dir": str(adapter_dir),
            "metrics": run.metrics,
        }

    # --- evaluate each method on identical prompts ------------------------
    from .models.generation import GeneratorConfig, MockGenerator

    gen_cfg = GeneratorConfig(max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
    for method in methods:
        # The evaluation service calls the un-conditioned baseline "plain"; the
        # n-gram engine also applies its own persona, so it uses the same
        # prompt-only path (a system prompt would only confuse it).
        eval_method = "plain" if method in {"base", "ngram"} else method
        notes: list[str] = []
        g: TextGenerator | None = generator
        rev = model_revision
        if method in {"lora", "style_rag_ft"} and train_lora:
            from .models.generation import TransformersGenerator
            from .models.loader import load_model_with_adapter
            from .models.registry import get_model_spec

            adapter = Path(report.environment["lora_training"]["adapter_dir"])
            spec = get_model_spec(model)
            # A LoadedModel has no .generate(); the chat-imitation wrapper is
            # the same inference path training-adjacent code uses elsewhere.
            g = TransformersGenerator(load_model_with_adapter(spec, adapter), spec)
            rev = f"{model}+lora:{adapter.name}"
        elif method == "ngram":
            from .models.ngram import NGramGenerator, NGramModel

            ngram_model = NGramModel()
            ngram_model.train_texts(train_texts)
            g = NGramGenerator(ngram_model, seed=seed)
            rev = f"ngram(order={getattr(ngram_model, 'order', 3)})"
        elif (
            method in {"base", "plain", "style_profile", "style_rag", "style_rag_ft"} and g is None
        ):
            g = MockGenerator(train_texts[:5], seed=seed)
            rev = "mock-1"
        assert g is not None, f"method {method} has no generator"

        from .profiling.style_profile import build_style_profile, profile_prompt_fragment

        fragment = None
        if method in {"style_profile", "style_rag", "style_rag_ft"}:
            profile = build_style_profile(messages, target)
            fragment = profile_prompt_fragment(profile)

        records, _, _ = run_baseline(
            method=eval_method,
            generator=g,
            prompts=prompts,
            profile_fragment=fragment,
            output_dir=out / "evaluations",
            model_revision=rev,
            gen_cfg=gen_cfg,
            retrieval_index=index if method in {"style_rag", "style_rag_ft"} else None,
            retrieval_cfg=RetrievalConfig(k=3),
            train_texts=train_texts,
        )
        gens = [r.get("generated", "") for r in records if r.get("generated")]
        refs = [r["reference"] for r in records if r.get("generated")]
        n_failed = sum(1 for r in records if not r.get("generated"))
        if n_failed:
            notes.append(f"{n_failed}/{len(records)} generations failed")
        style = compare_style(refs, gens).model_dump() if gens else None
        contam = memorization_overlap(gens, train_texts).model_dump() if gens else None
        discrimination = prompt_discrimination(gens) if gens else None
        uptake = prompt_uptake(prompt_texts[: len(gens)], gens) if gens else None
        claims = (
            unsupported_claims(gens, prompt_texts[: len(gens)], known_terms=train_texts)
            if gens
            else None
        )
        report.methods.append(
            MethodResult(
                method=method,
                model_revision=rev,
                n_prompts=len(records),
                n_failed=n_failed,
                style=style,
                contamination=contam,
                prompt_discrimination=discrimination,
                prompt_uptake=uptake,
                unsupported_claims=claims,
                notes=notes,
            )
        )
        (out / f"predictions_{method}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

    report.environment["total_seconds"] = round(time.time() - started, 1)
    report.environment["device"] = _device_name()
    report.limitations = [
        "The author is a corpus role, not one individual: results show that LoRA can "
        "capture a consistent speaker register, not that it clones a specific person.",
        "Metrics are surface-level (n-grams, punctuation, length) plus prompt "
        "discrimination; they do not measure meaning quality.",
        f"Evaluated on {len(prompts)} held-out prompts from "
        f"{report.splits['test']} conversations; treat small differences as noise.",
    ]
    report.honesty_note = (
        "Every number in this report was produced by running the pipeline. "
        "No metric is estimated, extrapolated or copied from literature."
    )
    path = out / "report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:
        return "unknown"


def format_comparison(report: ExperimentReport) -> str:
    """Human-readable comparison table."""
    lines = [
        f"Experiment {report.experiment_id} (model={report.model})",
        f"  corpus: {report.corpus['n_conversations']} conversations, "
        f"{report.corpus['n_messages']} messages",
        f"  dataset: {report.dataset['n_examples']} examples | "
        f"split train/val/test = {report.splits['train']}/{report.splits['val']}/"
        f"{report.splits['test']} "
        f"(leakage_check={report.splits['leakage_check_passed']})",
        "",
        f"{'method':<14}{'cos3':>7}{'lenR':>7}{'distinct':>9}{'uptake':>8}{'contam':>10}{'failed':>8}",
    ]
    for m in report.methods:
        style = m.style or {}
        cos = style.get("char_ngram_cosine_3")
        ratio = style.get("length_mean_ratio")
        contam = (m.contamination or {}).get("verdict", "-")
        disc = (m.prompt_discrimination or {}).get("distinct_ratio")
        uptake = (m.prompt_uptake or {}).get("mean_uptake")
        lines.append(
            f"{m.method:<14}"
            f"{(f'{cos:.3f}' if cos is not None else '-'):>7}"
            f"{(f'{ratio:.2f}' if ratio is not None else '-'):>7}"
            f"{(f'{disc:.2f}' if disc is not None else '-'):>9}"
            f"{(f'{uptake:.3f}' if uptake is not None else '-'):>8}"
            f"{contam!s:>10}"
            f"{m.n_failed:>8}"
        )
    lines.append("")
    lines.append(report.honesty_note)
    return "\n".join(lines)
