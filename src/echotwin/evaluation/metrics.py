"""Measurable style/response metrics for personalization evaluation.

All metrics are computed from real predictions. Nothing here "guesses" quality:
model-based scores are reported alongside their method and sample size, and
phrase overlap is explicitly labeled stylometric reproduction — never proof of
meaning or generalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

_EMOJI_RE = re.compile("[\u2600-\u27bf\u2b00-\u2bff\ufe0f\U0001f000-\U0001faff\u200d]")
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+")
_PUNCT_CHARS = ".,!?;:'\"-…—()[]{}"


def char_ngrams(text: str, n: int = 3) -> Counter:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return Counter(t[i : i + n] for i in range(0, max(0, len(t) - n + 1)))


def word_ngrams(text: str, n: int = 2) -> Counter:
    words = _WORD_RE.findall(text.lower())
    return Counter(tuple(words[i : i + n]) for i in range(0, max(0, len(words) - n + 1)))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def char_ngram_cosine(ref_texts: list[str], gen_texts: list[str], n: int = 3) -> float:
    ref: Counter[str] = Counter()
    gen: Counter[str] = Counter()
    for t in ref_texts:
        ref.update(char_ngrams(t, n))
    for t in gen_texts:
        gen.update(char_ngrams(t, n))
    return round(_cosine(ref, gen), 4)


def punctuation_profile(text: str) -> dict[str, float]:
    """Relative punctuation frequencies (per non-space char)."""
    parts: list[str] = []
    for ch in text:
        if not ch.isspace():
            parts.append(ch)
    total = len(parts) or 1
    counts = Counter(ch for ch in parts if ch in _PUNCT_CHARS)
    return {k: v / total for k, v in sorted(counts.items())}


def profile_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return round(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2.0, 4)


def capitalization_profile(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {}
    counts: Counter[str] = Counter()
    for t in texts:
        t = t.strip()
        letters = [c for c in t if c.isalpha()]
        if not letters:
            counts["empty"] += 1
        elif all(c.isupper() for c in letters):
            counts["all_caps"] += 1
        elif t[:1].isupper() and any(c.islower() for c in letters):
            counts["capitalized"] += 1
        elif all(c.islower() for c in letters):
            counts["lowercase"] += 1
        else:
            counts["mixed"] += 1
    n = len(texts)
    return {k: v / n for k, v in counts.most_common()}


def emoji_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return round(sum(len(_EMOJI_RE.findall(t)) for t in texts) / len(texts), 4)


def question_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return round(sum(1 for t in texts if "?" in t) / len(texts), 4)


def length_stats(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    lens = sorted(float(len(t)) for t in texts)
    n = len(lens)
    return {
        "mean": round(sum(lens) / n, 2),
        "median": round(lens[n // 2], 2),
        "p90": round(lens[min(n - 1, int(n * 0.9))], 2),
        "max": round(lens[-1], 2),
    }


def abbreviation_overlap(ref_texts: list[str], gen_texts: list[str]) -> dict:
    """Abbreviation usage overlap between reference and generated texts."""
    common = {
        "u",
        "ur",
        "pls",
        "plz",
        "tmrw",
        "okk",
        "msg",
        "grp",
        "fr",
        "tho",
        "thx",
        "tym",
        "abt",
        "btw",
        "asap",
        "idk",
        "lol",
        "tbh",
        "ngl",
        "rn",
    }

    def abbrevs(texts: list[str]) -> set[str]:
        found: set[str] = set()
        for t in texts:
            for w in _WORD_RE.findall(t):
                if w.lower() in common:
                    found.add(w.lower())
        return found

    r, g = abbrevs(ref_texts), abbrevs(gen_texts)
    if not r:
        return {"ref_abbreviations": 0, "gen_abbreviations": len(g), "overlap": None}
    inter = r & g
    return {
        "ref_abbreviations": len(r),
        "gen_abbreviations": len(g),
        "overlap": round(len(inter) / len(r), 4),
        "gen_only": sorted(g - r)[:10],
    }


def top_phrase_reproduction(
    ref_texts: list[str], gen_texts: list[str], n: int = 2, top_k: int = 25
) -> dict:
    """Share of the reference corpus' top-n-grams that the generations reproduce.

    This measures STYLE REPRODUCTION. It is explicitly not evidence that the model
    learned meaning or generalized to unseen situations.
    """
    ref_ng: Counter[tuple[str, ...]] = Counter()
    for t in ref_texts:
        ref_ng.update(word_ngrams(t, n))
    top = [k for k, _ in ref_ng.most_common(top_k) if ref_ng[k] > 1]
    if not top:
        return {"top_k": 0, "reproduced": 0.0, "note": "no repeated reference n-grams"}
    gen_joined = " ".join(t.lower() for t in gen_texts)
    hits = sum(1 for k in top if " ".join(k) in gen_joined)
    return {
        "top_k": len(top),
        "reproduced": round(hits / len(top), 4),
        "method": f"{n}-gram phrase overlap (stylometric reproduction only)",
    }


def repetition_rate(texts: list[str], n: int = 3) -> float:
    """Fraction of n-grams that repeat within the generation set."""
    total = 0
    repeated = 0
    for t in texts:
        ng = word_ngrams(t, n)
        total += sum(ng.values())
        repeated += sum(v - 1 for v in ng.values() if v > 1)
    return round(repeated / total, 4) if total else 0.0


def language_mix(texts: list[str]) -> dict[str, float]:
    from ..language import detect_language

    if not texts:
        return {}
    counts = Counter(detect_language(t).value for t in texts)
    n = len(texts)
    return {k: round(v / n, 4) for k, v in counts.most_common()}


def language_distribution_distance(ref_texts: list[str], gen_texts: list[str]) -> float:
    a, b = language_mix(ref_texts), language_mix(gen_texts)
    keys = set(a) | set(b)
    return round(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2.0, 4)


def duplicate_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    norm = [re.sub(r"\s+", " ", t.strip().lower()) for t in texts]
    counts = Counter(norm)
    dups = sum(v - 1 for v in counts.values() if v > 1)
    return round(dups / len(texts), 4)


class StyleComparison(BaseModel):
    n_reference: int
    n_generated: int
    char_ngram_cosine_3: float | None = None
    punctuation_distance: float | None = None
    capitalization_distance: float | None = None
    emoji_rate_ref: float | None = None
    emoji_rate_gen: float | None = None
    length_ref: dict[str, float] = Field(default_factory=dict)
    length_gen: dict[str, float] = Field(default_factory=dict)
    length_mean_ratio: float | None = None
    question_rate_ref: float | None = None
    question_rate_gen: float | None = None
    abbreviation_overlap: dict = Field(default_factory=dict)
    phrase_reproduction: dict = Field(default_factory=dict)
    language_distribution_distance: float | None = None
    repetition_rate_gen: float | None = None
    duplicate_rate_gen: float | None = None
    limitations: list[str] = Field(default_factory=list)


def compare_style(reference_texts: list[str], generated_texts: list[str]) -> StyleComparison:
    ref_punct = punctuation_profile(" ".join(reference_texts))
    gen_punct = punctuation_profile(" ".join(generated_texts))
    ref_caps = capitalization_profile(reference_texts)
    gen_caps = capitalization_profile(generated_texts)
    ref_len = length_stats(reference_texts)
    gen_len = length_stats(generated_texts)
    ratio = round(gen_len["mean"] / ref_len["mean"], 3) if ref_len["mean"] else None
    limitations = [
        "n-gram cosine and phrase overlap measure surface similarity, not meaning;",
        "small corpora make punctuation/capitalization distances unstable;",
        "language detection is approximate for code-mixed Hinglish.",
    ]
    return StyleComparison(
        n_reference=len(reference_texts),
        n_generated=len(generated_texts),
        char_ngram_cosine_3=char_ngram_cosine(reference_texts, generated_texts, 3),
        punctuation_distance=profile_distance(ref_punct, gen_punct),
        capitalization_distance=profile_distance(ref_caps, gen_caps),
        emoji_rate_ref=emoji_rate(reference_texts),
        emoji_rate_gen=emoji_rate(generated_texts),
        length_ref=ref_len,
        length_gen=gen_len,
        length_mean_ratio=ratio,
        question_rate_ref=question_rate(reference_texts),
        question_rate_gen=question_rate(generated_texts),
        abbreviation_overlap=abbreviation_overlap(reference_texts, generated_texts),
        phrase_reproduction=top_phrase_reproduction(reference_texts, generated_texts),
        language_distribution_distance=language_distribution_distance(
            reference_texts, generated_texts
        ),
        repetition_rate_gen=repetition_rate(generated_texts),
        duplicate_rate_gen=duplicate_rate(generated_texts),
        limitations=limitations,
    )


@dataclass
class PredictionRecord:
    prompt: str
    generated: str
    reference: str | None = None
    model_revision: str | None = None
    method: str | None = None
    extra: dict | None = None


class EvaluationReport(BaseModel):
    report_id: str
    created_at: str
    method: str
    prompt_count: int
    predictions_path: str | None = None
    style: StyleComparison | None = None
    contamination: dict | None = None
    per_example: list[dict] = Field(default_factory=list)
    sample_size_warning: str | None = None
    limitations: list[str] = Field(default_factory=list)
    metrics_version: str = "1.0"
