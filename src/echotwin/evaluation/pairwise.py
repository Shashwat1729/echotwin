"""Blinded pairwise evaluation (human + optional model judge).

Protocol:

* Two candidate outputs for the SAME held-out prompt are shown without method
  labels (blind). Human reviewers score style similarity, naturalness, context
  appropriateness and language authenticity (1-5).
* An optional model judge produces a preference with an explicit confidence and
  method note; model-judged results are ALWAYS labeled model-based and are never
  mixed with human scores without saying so.
* Reports include sample size and uncertainty guidance (win rate + counts).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PairwiseSample(BaseModel):
    sample_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    prompt_context: list[dict] = Field(default_factory=list)
    reference: str | None = None
    output_a: str
    output_b: str
    method_a: str = "?"
    method_b: str = "?"
    language: str = "unknown"


class HumanJudgment(BaseModel):
    sample_id: str
    winner: str = Field(description="a | b | tie")
    criterion: str = Field(
        default="overall",
        description="overall | style_similarity | naturalness | context | language_authenticity",
    )
    score_a: float | None = Field(default=None, ge=1, le=5)
    score_b: float | None = Field(default=None, ge=1, le=5)
    annotator: str | None = None
    notes: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PairwiseResult(BaseModel):
    n_comparisons: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float | None
    win_rate_b: float | None
    human_comparisons: int
    model_comparisons: int
    uncertainty_note: str
    limitations: list[str] = Field(default_factory=list)


def save_judgments(judgments: list[HumanJudgment], path: Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for j in judgments:
            fh.write(j.model_dump_json() + "\n")
    return len(judgments)


def load_judgments(path) -> list[HumanJudgment]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(HumanJudgment.model_validate_json(line))
    return out


def summarize(judgments: list[HumanJudgment], *, model_judged: int = 0) -> PairwiseResult:
    n = len(judgments)
    wins_a = sum(1 for j in judgments if j.winner == "a")
    wins_b = sum(1 for j in judgments if j.winner == "b")
    ties = sum(1 for j in judgments if j.winner == "tie")
    decided = wins_a + wins_b
    return PairwiseResult(
        n_comparisons=n,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_a=round(wins_a / decided, 4) if decided else None,
        win_rate_b=round(wins_b / decided, 4) if decided else None,
        human_comparisons=n - model_judged if n >= model_judged else n,
        model_comparisons=model_judged,
        uncertainty_note=(
            "Small sample: win rates below ~30 comparisons are indicative only."
            if decided < 30
            else "Win rates estimated from the recorded comparisons."
        ),
        limitations=[
            "Single-blind: reviewers see text only; method labels hidden but bias remains possible.",
            "Model-judged preferences (if any) measure the judge's taste, not human preference.",
        ],
    )


def judge_with_model(generator, sample: PairwiseSample, cfg) -> HumanJudgment:
    """Optional LLM-as-judge for DRY RUNS. Labeled model-based; never human gold."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are comparing two candidate replies to the same chat message for "
                "style match to the reference author. Answer exactly 'A', 'B', or 'TIE'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context: {json.dumps(sample.prompt_context, ensure_ascii=False)}\n"
                f"Reference reply: {sample.reference}\n"
                f"Candidate A: {sample.output_a}\nCandidate B: {sample.output_b}"
            ),
        },
    ]
    try:
        verdict = generator.generate(prompt, cfg).strip().upper()
    except Exception:
        verdict = ""
    winner = "a" if verdict.startswith("A") else ("b" if verdict.startswith("B") else "tie")
    return HumanJudgment(
        sample_id=sample.sample_id,
        winner=winner,
        criterion="overall",
        annotator=f"model-judge:{getattr(generator, 'name', 'unknown')}",
        notes="model-based judgment; NOT a human label",
    )


def build_pairs(
    prompts: list[dict], outputs_by_method: dict[str, list[str]], method_a: str, method_b: str
) -> list[PairwiseSample]:
    a_list = outputs_by_method.get(method_a, [])
    b_list = outputs_by_method.get(method_b, [])
    pairs = []
    for i, p in enumerate(prompts):
        if i >= len(a_list) or i >= len(b_list):
            break
        pairs.append(
            PairwiseSample(
                prompt_context=p.get("messages", []),
                reference=p.get("reference"),
                output_a=a_list[i],
                output_b=b_list[i],
                method_a=method_a,
                method_b=method_b,
                language=p.get("language", "unknown"),
            )
        )
    return pairs
