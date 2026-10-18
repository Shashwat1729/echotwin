"""Memorization / contamination checks.

Small personal datasets (hundreds of messages) make SFT prone to verbatim
replay. Evaluation MUST measure this instead of assuming generalization:
high train-overlap + high style similarity indicates memorization, not learning.

What counts as contamination (and what does not):

* **Verbatim copy** — the generation reproduces a training target, as a whole
  phrase, on *word boundaries*. Raw substring matching is not enough: ``"ok"``
  occurs inside ``"okay"``, so ``"okay do it now"`` is not a copy of ``"ok"``.
* **Near-verbatim** — the generation's contiguous n-grams are mostly present in
  a single training target, at an n large enough to be meaningful. A two-word
  reply cannot be judged on 6-grams, so containment is only asserted at n >= 3;
  below that the check abstains instead of degrading into unigram noise.

What does NOT count: sharing vocabulary. Reusing the author's function words
("haan", "yaar", "acha") is the *goal* of style transfer, so lexical overlap is
reported for insight and deliberately excluded from the verdict.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..tokenization import tokenize_words

_NORM_RE = re.compile(r"\s+")
_NGRAM = 6
#: Containment is only asserted at this n or above (below it is unigram noise).
_MIN_VERBATIM_TARGET_WORDS = 3  # shorter targets need to cover most of the generation
_MIN_CONTAINMENT_N = 3
#: Verdict boundaries over the combined exact + near rate.
_HIGH_OVERLAP_RATE = 0.5
_ANY_OVERLAP_RATE = 0.0


def normalize(text: str) -> str:
    return _NORM_RE.sub(" ", (text or "").strip().lower())


def ngrams(tokens: list[str], n: int = _NGRAM) -> set[tuple[str, ...]]:
    """Fixed-size n-gram set.

    Short texts are emitted as their own single n-tuple. That is only sound when
    BOTH sides use the same ``n``: mixing a 5-gram with 6-grams made the sets
    structurally incapable of intersecting, so a verbatim copy of a short span
    was reported as clean. Use :func:`_shared_ngrams` when the two texts can
    differ in length.
    """
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _shared_ngrams(a: list[str], b: list[str]) -> tuple[set, set, int]:
    """N-gram sets for two texts at a common ``n`` that fits the shorter one.

    ``n`` is capped at half the shorter text so short messages still produce
    several windows, but it is never allowed below
    :data:`_MIN_CONTAINMENT_N`: a containment test over unigrams is not
    evidence of copying.
    """
    shorter = min(len(a), len(b))
    n = max(1, min(shorter, _NGRAM, max(1, shorter // 2)))
    if n < _MIN_CONTAINMENT_N:
        n = _MIN_CONTAINMENT_N
        if shorter < n:
            n = shorter
    return ngrams(a, n), ngrams(b, n), n


def _contains_phrase(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
    """True when ``needle`` occurs as a contiguous *token* run inside ``haystack``.

    Word-boundary aware, so "ok" does not match inside "okay".
    """
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    n = len(needle_tokens)
    for i in range(len(haystack_tokens) - n + 1):
        if haystack_tokens[i : i + n] == needle_tokens:
            return True
    return False


class ContaminationReport(BaseModel):
    n_generations: int
    n_train_reference: int
    exact_match_rate: float = Field(
        description="verbatim copies: the generation reproduces a training target phrase"
    )
    near_match_rate: float = Field(
        description="near-verbatim: most CONTIGUOUS n-grams of the generation appear in one training target"
    )
    lexical_overlap: float = Field(
        default=0.0,
        description=(
            "mean share of the generation's distinct words found in the most "
            "similar training target. Expected to be HIGH for style transfer "
            "(the point is to reuse their vocabulary); it is reported for "
            "insight and is NOT evidence of memorization."
        ),
    )
    flagged_examples: list[dict] = Field(
        default_factory=list, description="index + diagnosis (no full text)"
    )
    verdict: str = Field(description="clean | some_overlap | high_overlap")
    method_note: str = (
        "word-boundary verbatim phrase match + contiguous n-gram containment "
        "(n >= 3). Lexical overlap is reported separately and deliberately "
        "excluded from the verdict: imitating a person's vocabulary is the "
        "goal, not a failure."
    )


def memorization_overlap(
    generated_texts: list[str],
    train_texts: list[str],
    *,
    near_threshold: float = 0.7,
    flag_threshold: float = 0.5,
) -> ContaminationReport:
    """Measure memorization of training targets.

    Only *contiguous* copying counts. Measuring bag-of-words overlap flagged
    every Hinglish reply as contamination simply because style transfer
    reuses function words ("haan", "yaar", "acha"), which made the verdict
    useless and hid real verbatim copies among the noise.
    """
    train_token_lists = [tokenize_words(t) for t in train_texts if t and t.strip()]
    train_token_lists = [t for t in train_token_lists if t]

    exact = 0
    near = 0
    lex_total = 0.0
    lex_n = 0
    flagged: list[dict] = []
    for i, g in enumerate(generated_texts):
        if not g or not g.strip():
            continue
        gtoks = tokenize_words(g)
        if not gtoks:
            continue

        # --- verbatim copy, on word boundaries, in both directions ---
        # A SHORT training target ("ok", "haan") also occurs inside perfectly
        # original replies, so a short target may only convict a generation when
        # it accounts for most of it. Longer targets convict on a single
        # contiguous match, which is what a real copy looks like.
        is_exact = False
        for tn in train_token_lists:
            if not (_contains_phrase(tn, gtoks) or _contains_phrase(gtoks, tn)):
                continue
            if len(tn) >= _MIN_VERBATIM_TARGET_WORDS or len(tn) >= 0.6 * len(gtoks):
                is_exact = True
                break

        best_ngram = 0.0
        best_lex = 0.0
        g_tokset = set(gtoks)
        for tn in train_token_lists:
            g_ng, t_ng, n = _shared_ngrams(gtoks, tn)
            if n >= _MIN_CONTAINMENT_N and g_ng and t_ng:
                ngram_score = len(g_ng & t_ng) / len(g_ng)
                if ngram_score > best_ngram:
                    best_ngram = ngram_score
            # Informational only: shared vocabulary is the goal of style
            # transfer, so this never drives the verdict.
            t_tokset = set(tn)
            lex_score = len(g_tokset & t_tokset) / len(g_tokset)
            if lex_score > best_lex:
                best_lex = lex_score
            if best_ngram >= 1.0:
                break
        lex_total += best_lex
        lex_n += 1
        near_hit = best_ngram >= near_threshold
        exact += int(is_exact)
        near += int(near_hit and not is_exact)
        if (is_exact or near_hit) and len(flagged) < 50:
            flagged.append(
                {
                    "index": i,
                    "exact": is_exact,
                    "ngram_overlap": round(best_ngram, 3),
                    "lexical_overlap": round(best_lex, 3),
                    "diagnosis": "verbatim copy of training target"
                    if is_exact
                    else "near-verbatim: most n-grams copied from one training target",
                }
            )
    n = len(generated_texts) or 1
    exact_rate = round(exact / n, 4)
    near_rate = round(near / n, 4)
    total = exact_rate + near_rate
    if total >= _HIGH_OVERLAP_RATE:
        verdict = "high_overlap"
    elif total > _ANY_OVERLAP_RATE:
        verdict = "some_overlap"
    else:
        verdict = "clean"
    return ContaminationReport(
        n_generations=len(generated_texts),
        n_train_reference=len(train_token_lists),
        exact_match_rate=exact_rate,
        near_match_rate=near_rate,
        lexical_overlap=round(lex_total / lex_n, 4) if lex_n else 0.0,
        flagged_examples=flagged,
        verdict=verdict,
    )
