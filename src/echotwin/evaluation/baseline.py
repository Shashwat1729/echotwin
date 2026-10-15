"""Baseline methods and evaluation runs.

Baselines (all use the SAME held-out prompts for fair comparison):

* ``zero_shot``      — instruction prompt only
* ``style_profile``  — measured style-profile fragment
* ``few_shot``       — k real reference turns (never synthetic)
* ``style_fewshot``  — profile + few-shot
* ``style_rag``      — style profile + *retrieved* similar past replies (BM25 over
  the user's own messages, cross-conversation only)
* ``sft`` / ``lora`` — predictions from a trained adapter set (produced by the
  training module and re-scored here for comparability)
* ``style_rag_ft``   — retrieval-augmented generation on top of a trained model
  (the 2026 production default: fine-tune the behaviour, retrieve the content)
* ``dpo``            — preference-optimized variant predictions
* ``mock``           — deterministic offline stand-in (clearly labeled)

Reference texts for style/contamination comparisons come from the held-out split
and the TRAINING split respectively — never mixed up.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..dataset.builder import TrainingExample
from ..dataset.retrieval import RetrievalConfig, retrieve_exemplars
from ..evaluation.contamination import memorization_overlap
from ..evaluation.metrics import EvaluationReport, StyleComparison, compare_style
from ..models.generation import GeneratorConfig, save_predictions

SYSTEM_BASE = (
    "You are emulating the writing style of the person described below. "
    "Reply the way they write in chat: preserve their spelling habits, emoji use, "
    "capitalization, message length, and language mixing (Hindi/English/Hinglish). "
    "Never invent facts about them that are not in the conversation."
)

METHODS = [
    "zero_shot",
    "style_profile",
    "few_shot",
    "style_fewshot",
    "style_rag",
    "sft",
    "lora",
    "style_rag_ft",
    "dpo",
    "mock",
]

RETRIEVAL_METHODS = {"style_rag", "style_rag_ft"}


def heldout_prompts(test_examples: list[TrainingExample], max_prompts: int = 100) -> list[dict]:
    """Identical evaluation prompts from the held-out split."""
    prompts = []
    for ex in test_examples[:max_prompts]:
        context = ex.messages[:-1]
        prompts.append(
            {
                "messages": context,
                "reference": ex.messages[-1].content,
                "language": ex.metadata.language,
                "conversation_id": ex.metadata.conversation_id,
            }
        )
    return prompts


def build_messages(
    method: str,
    prompt: dict,
    profile_fragment: str | None,
    few_shot: list[dict],
    retrieval_exemplars: list[dict] | None = None,
) -> list[dict]:
    system = SYSTEM_BASE
    if (
        method in {"style_profile", "style_fewshot", "style_rag", "style_rag_ft"}
        and profile_fragment
    ):
        system += "\n\n" + profile_fragment
    messages: list[dict] = [{"role": "system", "content": system}]
    if method in {"few_shot", "style_fewshot"}:
        for ex in few_shot:
            messages.append({"role": "user", "content": ex["context"]})
            messages.append({"role": "assistant", "content": ex["response"]})
    if method in RETRIEVAL_METHODS:
        for ex in retrieval_exemplars or []:
            messages.append({"role": "user", "content": ex["context"]})
            messages.append({"role": "assistant", "content": ex["response"]})
    messages.extend(prompt["messages"])
    return messages


def run_baseline(
    *,
    method: str,
    generator,
    prompts: list[dict],
    profile_fragment: str | None = None,
    few_shot: list[dict] | None = None,
    output_dir: str | Path,
    model_revision: str,
    gen_cfg: GeneratorConfig | None = None,
    retrieval_index=None,
    retrieval_cfg: RetrievalConfig | None = None,
    train_texts: list[str] | None = None,
) -> tuple[list[dict], Path, EvaluationReport]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; expected one of {METHODS}")
    cfg = gen_cfg or GeneratorConfig()
    few_shot = few_shot or []
    if method in RETRIEVAL_METHODS and retrieval_index is None:
        raise ValueError(
            f"method '{method}' requires a retrieval_index (build one with dataset.retrieval)"
        )
    records: list[dict] = []
    for i, p in enumerate(prompts):
        exemplars: list[dict] = []
        if method in RETRIEVAL_METHODS:
            exemplars = retrieve_exemplars(
                retrieval_index,
                p["messages"],
                conversation_id=p.get("conversation_id"),
                cfg=retrieval_cfg,
            )
        messages = build_messages(method, p, profile_fragment, few_shot, exemplars)
        try:
            generated = generator.generate(messages, cfg)
        except Exception as e:
            records.append(
                {
                    "index": i,
                    "error": str(e),
                    "method": method,
                    "model_revision": model_revision,
                    "reference": p["reference"],
                    "language": p.get("language"),
                }
            )
            continue
        record = {
            "index": i,
            "method": method,
            "model_revision": model_revision,
            "generated": generated,
            "reference": p["reference"],
            "language": p.get("language"),
        }
        if method in RETRIEVAL_METHODS:
            record["retrieved_exemplars"] = [
                {"score": e["score"], "conversation_id": e["conversation_id"]} for e in exemplars
            ]
        records.append(record)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{method}_{model_revision.replace('/', '__')}.jsonl"
    save_predictions(records, pred_path)

    refs = [r["reference"] for r in records if r.get("generated")]
    gens = [r["generated"] for r in records if r.get("generated")]
    style: StyleComparison | None = compare_style(refs, gens) if gens else None
    contamination = (
        memorization_overlap(gens, train_texts).model_dump() if (train_texts and gens) else None
    )
    report = EvaluationReport(
        report_id=f"eval-{method}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        created_at=datetime.now(UTC).isoformat(),
        method=method,
        prompt_count=len(prompts),
        predictions_path=str(pred_path),
        style=style,
        contamination=contamination,
        per_example=[
            {k: r.get(k) for k in ("index", "generated", "reference", "language")}
            for r in records[:200]
        ],
        sample_size_warning=(
            "Prompt count below 30; metrics are indicative only." if len(prompts) < 30 else None
        ),
        limitations=[
            "Metrics are surface-level (style), not semantic;",
            "identical prompts across methods but a single sample per prompt (no sampling variance);",
            "language detection approximate for Hinglish;",
            "high train-overlap in contamination checks indicates memorization, not generalization.",
        ],
    )
    return records, pred_path, report


def load_predictions(path: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad predictions JSONL at {path}:{i}: {e}") from e
    return out


def compare_runs(paths: list[str | Path]) -> dict:
    """Compare several prediction runs computed on the same held-out prompts."""
    runs = {str(p): load_predictions(p) for p in paths}
    if not runs:
        return {}
    common = sorted(set.intersection(*[set(r["index"] for r in recs) for recs in runs.values()]))
    table = {
        name: {r["index"]: r.get("generated", "") for r in recs} for name, recs in runs.items()
    }
    return {"aligned_prompt_indices": common, "outputs": table}


def save_report(report: EvaluationReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p
