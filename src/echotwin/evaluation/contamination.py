"""Memorization / contamination checks.

Small personal datasets (hundreds of messages) make SFT prone to verbatim
replay. Evaluation MUST measure this instead of assuming generalization:
high train-overlap + high style similarity indicates memorization, not learning.

Method: exact normalized match + token n-gram containment (cheap, deterministic).
Reported per-run; thresholds documented, verdict explicit.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_NORM_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[\w\u0900-\u097F']+")
_NGRAM = 6


def normalize(text: str) -> str:
    return _NORM_RE.sub(" ", (text or "").strip().lower())


def ngrams(tokens: list[str], n: int = _NGRAM) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class ContaminationReport(BaseModel):
    n_generations: int
    n_train_reference: int
    exact_match_rate: float
    near_match_rate: float = Field(
        description="share with >=50% token-6gram overlap with any train target"
    )
    flagged_examples: list[dict] = Field(
        default_factory=list, description="index + diagnosis (no full text)"
    )
    verdict: str = Field(description="clean | some_overlap | high_overlap")
    method_note: str = "exact normalized match + token 6-gram containment"


def memorization_overlap(
    generated_texts: list[str],
    train_texts: list[str],
    *,
    flag_threshold: float = 0.5,
) -> ContaminationReport:
    train_norm = {normalize(t) for t in train_texts if t and t.strip()}
    train_ngrams = [ngrams(_WORD_RE.findall(normalize(t))) for t in train_texts if t and t.strip()]
    train_ngrams = [g for g in train_ngrams if g]

    exact = 0
    near = 0
    flagged: list[dict] = []
    for i, g in enumerate(generated_texts):
        if not g or not g.strip():
            continue
        gn = normalize(g)
        is_exact = gn in train_norm
        gtoks = _WORD_RE.findall(gn)
        g_ng = ngrams(gtoks)
        best = 0.0
        if g_ng:
            for tn in train_ngrams:
                if not tn:
                    continue
                inter = len(g_ng & tn)
                if inter:
                    score = inter / len(g_ng)
                    if score > best:
                        best = score
                        if best >= 1.0:
                            break
        near_hit = best >= flag_threshold
        exact += int(is_exact)
        near += int(near_hit and not is_exact)
        if (is_exact or near_hit) and len(flagged) < 50:
            flagged.append(
                {
                    "index": i,
                    "exact": is_exact,
                    "best_overlap": round(best, 3),
                    "diagnosis": "verbatim copy of training target"
                    if is_exact
                    else "high token overlap with a training target",
                }
            )
    n = len(generated_texts) or 1
    exact_rate = round(exact / n, 4)
    near_rate = round(near / n, 4)
    total = exact_rate + near_rate
    if total >= 0.5:
        verdict = "high_overlap"
    elif total > 0.05:
        verdict = "some_overlap"
    else:
        verdict = "clean"
    return ContaminationReport(
        n_generations=len(generated_texts),
        n_train_reference=len(train_texts),
        exact_match_rate=exact_rate,
        near_match_rate=near_rate,
        flagged_examples=flagged,
        verdict=verdict,
    )
