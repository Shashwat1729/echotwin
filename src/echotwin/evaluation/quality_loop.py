"""Quality-gate harness: measure real style-transfer quality and fail loudly.

This is the measurement loop the project is built around. It trains the CPU
n-gram engine on a training split, generates replies for a *held-out* split,
and applies hard, mechanical gates:

* no empty generations
* no degenerate repetition
* no replies shorter than a plausible minimum
* no verbatim replay of training targets (memorization)

It also reports the style metrics (char n-gram cosine, punctuation /
capitalization distance, length fidelity) so that a change can be judged by a
number instead of by vibes.

Two modes:

* **dataset** — run against the user's own prepared dataset (train/test split).
* **synthetic** — run against a bundled, deterministic Hinglish corpus, so the
  loop is runnable on a clean checkout with no private data.

The gates are deliberately *mechanical*: they must be reproducible, and a
failure must be actionable rather than a judgement call.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..dataset.builder import TrainingExample
from ..evaluation.contamination import memorization_overlap
from ..evaluation.metrics import compare_style
from ..models.generation import GeneratorConfig
from ..models.ngram import NGramConfig, NGramGenerator, NGramModel

# ---------------------------------------------------------------------------
# Bundled synthetic corpus (deterministic, fictional, Hinglish/Latin script)
# ---------------------------------------------------------------------------
SYNTHETIC_CORPUS: list[tuple[str, list[str]]] = [
    (
        "c1",
        [
            "bhai kya kar raha hai abhi",
            "main bas ghar pe hu yaar, kuch nahi kar raha",
            "acha cool, milte h kahin?",
            "haan chalo, 6 baje library milte h",
            "theek hai, main pahunch jata hu",
            "ek minute, abhi nikalta hu",
            "ok done, milte h wahan",
        ],
    ),
    (
        "c2",
        [
            "good morning everyone ☀️",
            "morning! kaise ho sab log",
            "main theek hu, kal ka din kaam ka tha",
            "haan mera bhi bohot kaam tha, thoda tired hu",
            "chai peete h? main chai bana raha hu",
            "haan pakka, 5 minute me aata hu",
        ],
    ),
    (
        "c3",
        [
            "project ka deadline kya hai yaar",
            "friday ko hai, bohot nahi time bacha hai",
            "arre yaar, abhi tak itna kaam hai",
            "hum aaj raat tak try karte h complete karne ka",
            "theek hai, main baith ke shuru karta hu",
            "ok, mujhe bata dena kya karna hai exactly",
        ],
    ),
    (
        "c4",
        [
            "maa ne kya khana banaya hai pata hai?",
            "pata nahi, abhi puchta hu unse",
            "haan bata dena, mujhe bhi bhool gaya",
            "acha sab aa jao, khana tayyar hai",
            "aa raha hu 10 minute me",
        ],
    ),
    (
        "c5",
        [
            "match dekh liya kya tune?",
            "haan dekha, bahut mast tha yaar",
            "maine bhi dekha, last me acha goal tha",
            "haan sach me, kal bhi khelte h?",
            "haan bilkul, same time pe",
        ],
    ),
    (
        "c6",
        [
            "kal film ka ticket book kar liya?",
            "haan book kar liye, 6 baje ki show hai",
            "nice, main bhi aa raha hu na with you",
            "theek hai, milte h theatre ke bahar",
            "ok done, jaldi mat aana",
        ],
    ),
]

SYNTHETIC_TRAIN_CONVOS = {"c1", "c2", "c3", "c4"}
SYNTHETIC_TEST_CONVOS = {"c5", "c6"}


class QualityGates(BaseModel):
    """Mechanical pass/fail thresholds for a personalization run."""

    max_empty: int = 0
    max_degenerate_repeats: int = 0
    max_too_short: int = 0
    min_words: int = 2
    max_length_ratio_vs_reference: float = 1.6
    max_verbatim_copy_rate: float = 0.0
    require_clean_contamination: bool = True


class QualityReport(BaseModel):
    """Outcome of one quality measurement run."""

    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    mode: str = "dataset"
    method: str = "ngram"
    n_train_targets: int = 0
    n_prompts: int = 0
    n_generations: int = 0

    empty_generations: int = 0
    degenerate_repeats: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicate_rate: float = 0.0
    length_mean_generated: float = 0.0
    length_mean_reference: float = 0.0
    length_mean_ratio: float | None = None

    contamination_verdict: str = "clean"
    verbatim_copy_rate: float = 0.0
    near_copy_rate: float = 0.0

    char_ngram_cosine_3: float | None = None
    punctuation_distance: float | None = None
    capitalization_distance: float | None = None
    repetition_rate: float | None = None
    language_distribution_distance: float | None = None

    samples: list[dict] = Field(default_factory=list)
    gate_failures: list[str] = Field(default_factory=list)
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


@dataclass
class _Prompt:
    context: list[dict]
    reference: str


def _synthetic_split() -> tuple[list[str], list[_Prompt]]:
    train: list[str] = []
    prompts: list[_Prompt] = []
    for cid, msgs in SYNTHETIC_CORPUS:
        if cid in SYNTHETIC_TRAIN_CONVOS:
            train.extend(msgs)
        if cid in SYNTHETIC_TEST_CONVOS:
            for i in range(1, len(msgs)):
                prompts.append(
                    _Prompt(
                        context=[{"role": "user", "content": msgs[i - 1]}],
                        reference=msgs[i],
                    )
                )
    return train, prompts


def _dataset_split(
    train_examples: list[TrainingExample], test_examples: list[TrainingExample]
) -> tuple[list[str], list[_Prompt]]:
    train = [ex.messages[-1].content for ex in train_examples]
    prompts = [
        _Prompt(
            context=[m.model_dump() for m in ex.messages[:-1]],
            reference=ex.messages[-1].content,
        )
        for ex in test_examples
    ]
    return train, prompts


def measure_quality(
    *,
    train_examples: list[TrainingExample] | None = None,
    test_examples: list[TrainingExample] | None = None,
    synthetic: bool = False,
    gates: QualityGates | None = None,
    order: int = 3,
    temperature: float = 0.8,
    top_k: int = 12,
    max_new_tokens: int = 64,
    seed: int = 7,
    n_samples: int = 0,
) -> QualityReport:
    """Train, generate on held-out prompts, and score against hard gates.

    ``n_samples`` > 0 generates several replies per prompt (to measure
    *diversity* of a stochastic decoder rather than a single lucky sample).
    """
    gates = gates or QualityGates()
    if synthetic:
        train_texts, prompts = _synthetic_split()
        mode = "synthetic"
    else:
        if not train_examples or not test_examples:
            raise ValueError(
                "measure_quality needs both train and test examples, or synthetic=True"
            )
        train_texts, prompts = _dataset_split(train_examples, test_examples)
        mode = "dataset"

    if not train_texts:
        raise ValueError("No training targets to learn from.")
    if not prompts:
        raise ValueError("No held-out prompts to evaluate on.")

    model = NGramModel(order=order)
    model.train_texts(train_texts)
    gen = NGramGenerator(model, config=NGramConfig(temperature=temperature, top_k=top_k, seed=seed))
    cfg = GeneratorConfig(max_new_tokens=max_new_tokens)

    generations: list[str] = []
    references: list[str] = []
    for prompt in prompts:
        reps = max(1, n_samples)
        for _ in range(reps):
            generations.append(gen.generate(prompt.context, cfg))
            references.append(prompt.reference)

    gen_lens = [len(g.split()) for g in generations]
    ref_lens = [len(r.split()) for r in references]
    max_ref = max(ref_lens) if ref_lens else 1

    empty = sum(1 for g in generations if not g.strip())
    repeats = sum(1 for g in generations if len(set(g.split())) < 0.55 * max(1, len(g.split())))
    too_short = sum(1 for n in gen_lens if n < gates.min_words)
    too_long = sum(1 for n in gen_lens if n > max_ref * gates.max_length_ratio_vs_reference)
    dup_rate = round(1 - len(set(generations)) / len(generations), 4) if generations else 0.0

    contamination = memorization_overlap(generations, train_texts)
    style = compare_style(references, generations)

    failures: list[str] = []
    if empty > gates.max_empty:
        failures.append(f"{empty} empty generation(s) (allowed {gates.max_empty})")
    if repeats > gates.max_degenerate_repeats:
        failures.append(
            f"{repeats} degenerate repetition(s) (allowed {gates.max_degenerate_repeats})"
        )
    if too_short > gates.max_too_short:
        failures.append(f"{too_short} reply(s) under {gates.min_words} words")
    if contamination.exact_match_rate > gates.max_verbatim_copy_rate:
        failures.append(
            f"verbatim copy rate {contamination.exact_match_rate} "
            f"(allowed {gates.max_verbatim_copy_rate})"
        )
    if gates.require_clean_contamination and contamination.verdict != "clean":
        failures.append(f"contamination verdict is '{contamination.verdict}', expected 'clean'")

    length_mean_gen = statistics.mean(gen_lens) if gen_lens else 0.0
    length_mean_ref = statistics.mean(ref_lens) if ref_lens else 0.0

    notes = [
        "Gates are mechanical; style scores are surface-level and not proof of meaning.",
        "A copy rate of 0 with a small corpus can mean the model is too generic — "
        "check char_ngram_cosine_3 and length_mean_ratio together.",
    ]
    if len(train_texts) < 50:
        notes.append(
            f"Only {len(train_texts)} training targets: absolute style scores are not "
            "comparable to a real corpus, and a near-100% copy rate is expected "
            "rather than a defect."
        )

    return QualityReport(
        mode=mode,
        n_train_targets=len(train_texts),
        n_prompts=len(prompts),
        n_generations=len(generations),
        empty_generations=empty,
        degenerate_repeats=repeats,
        too_short=too_short,
        too_long=too_long,
        duplicate_rate=dup_rate,
        length_mean_generated=round(length_mean_gen, 2),
        length_mean_reference=round(length_mean_ref, 2),
        length_mean_ratio=(
            round(length_mean_gen / length_mean_ref, 3) if length_mean_ref else None
        ),
        contamination_verdict=contamination.verdict,
        verbatim_copy_rate=contamination.exact_match_rate,
        near_copy_rate=contamination.near_match_rate,
        char_ngram_cosine_3=style.char_ngram_cosine_3,
        punctuation_distance=style.punctuation_distance,
        capitalization_distance=style.capitalization_distance,
        repetition_rate=style.repetition_rate_gen,
        language_distribution_distance=style.language_distribution_distance,
        samples=[
            {
                "context": (p.context[0]["content"] if p.context else "")[:120],
                "reference": p.reference[:160],
                "generated": g[:160],
            }
            for p, g in list(zip(prompts, generations, strict=False))[:5]
        ],
        gate_failures=failures,
        passed=not failures,
        notes=notes,
    )


def format_report(report: QualityReport) -> str:
    """Human-readable rendering used by the CLI."""
    lines = [
        f"quality report ({report.mode} / {report.method})",
        f"  train targets : {report.n_train_targets}",
        f"  held-out      : {report.n_prompts} prompts -> {report.n_generations} generations",
        "",
        "gates",
        f"  empty generations    : {report.empty_generations}",
        f"  degenerate repeats   : {report.degenerate_repeats}",
        f"  too short            : {report.too_short}",
        f"  too long             : {report.too_long}",
        f"  duplicate rate       : {report.duplicate_rate}",
        f"  contamination        : {report.contamination_verdict} "
        f"(verbatim {report.verbatim_copy_rate}, near {report.near_copy_rate})",
        "",
        "style",
        f"  char 3-gram cosine   : {report.char_ngram_cosine_3}",
        f"  punctuation distance : {report.punctuation_distance}",
        f"  capitalization dist  : {report.capitalization_distance}",
        f"  length mean ratio    : {report.length_mean_ratio} "
        f"(gen {report.length_mean_generated} vs ref {report.length_mean_reference})",
        f"  repetition rate      : {report.repetition_rate}",
        f"  language distance    : {report.language_distribution_distance}",
    ]
    if report.samples:
        lines.append("")
        lines.append("samples")
        for s in report.samples:
            lines.append(f"  ctx : {s['context']!r}")
            lines.append(f"  ref : {s['reference']!r}")
            lines.append(f"  gen : {s['generated']!r}")
    if report.notes:
        lines.append("")
        lines.append("notes")
        lines.extend(f"  - {n}" for n in report.notes)
    if report.gate_failures:
        lines.append("")
        lines.append("gate failures")
        lines.extend(f"  - {f}" for f in report.gate_failures)
    lines.append("")
    lines.append("RESULT: " + ("PASS" if report.passed else "FAIL"))
    return "\n".join(lines)


def save_report(report: QualityReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p
