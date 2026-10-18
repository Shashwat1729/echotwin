"""Evaluation commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..dataset.builder import load_jsonl as load_examples_jsonl
from ..evaluation.baseline import (
    RETRIEVAL_METHODS,
    heldout_prompts,
    load_predictions,
    run_baseline,
)
from ..models.generation import GeneratorConfig, MockGenerator, TextGenerator
from ..profiling.style_profile import profile_prompt_fragment
from ..storage import state
from ._common import (
    _fail,
    console,
    evaluate_app,
)


@evaluate_app.command("baseline")
def evaluate_baseline(
    method: str = typer.Option(
        "style_rag",
        help="plain | zero_shot | style_profile | few_shot | style_fewshot | style_rag",
    ),
    model: str = typer.Option(
        "", help="registry name or org/model id (empty = deterministic offline generator)"
    ),
    ngram: bool = typer.Option(
        False, "--ngram", help="use the n-gram generator (pair with --method plain)"
    ),
    max_prompts: int = typer.Option(50),
    few_shot: int = typer.Option(5),
    rag_k: int = typer.Option(3, help="retrieved exemplars for style_rag methods"),
) -> None:
    """Run a baseline on the frozen held-out split."""
    from ..dataset.builder import load_jsonl
    from ..dataset.retrieval import RetrievalConfig, build_index_from_examples
    from ..settings import get_settings

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
        from ..profiling.style_profile import load_profile

        fragment = profile_prompt_fragment(load_profile(prof_path))
    if model:
        from ..models.generation import TransformersGenerator
        from ..models.registry import get_model_spec

        generator: TextGenerator = TransformersGenerator.from_spec(model)
        revision = get_model_spec(model).hf_model_id
    elif ngram:
        from ..models.ngram import NGramGenerator, NGramModel

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
    from ..evaluation.pairwise import build_pairs, judge_with_model, save_judgments, summarize
    from ..models.generation import MockGenerator
    from ..settings import get_settings

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
    from ..evaluation.baseline import compare_runs
    from ..settings import get_settings

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
