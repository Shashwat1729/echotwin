"""Pure-Python word n-gram language model for CPU-only style imitation.

Why this exists: neural fine-tuning needs a GPU, downloads, and torch. An
add-k smoothed word n-gram model with backoff trains in milliseconds on CPU
with zero heavy dependencies, and for short informal chat messages it is a
genuinely useful personalization method — historically *the* method for style
imitation, and still a strong baseline that every neural approach must beat.

Honest limits (documented, not hidden):
- No long-range coherence; quality degrades past ~15 words.
- Rare words fall back to unigram soup; the model can repeat training phrases
  verbatim — the contamination check in evaluation measures exactly this.
- Case is preserved (style signal) at the cost of sparsity.

Deterministic given a seed. Save/load as plain JSON.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F']+|[^\w\s]", re.UNICODE)

SCHEMA_VERSION = "1.0"


def tokenize(text: str) -> list[str]:
    """Split into words + single punctuation/emoji marks. Case preserved."""
    return _TOKEN_RE.findall(text or "")


class NGramConfig(BaseModel):
    order: int = Field(default=3, ge=1, le=5)
    smoothing_k: float = Field(default=0.1, ge=0.0)
    temperature: float = Field(default=0.9, ge=0.05)
    top_k: int = Field(default=40, ge=1)
    seed: int = 20250917


class NGramModel(BaseModel):
    schema_version: str = SCHEMA_VERSION
    order: int = Field(default=3, ge=1, le=5)
    smoothing_k: float = 0.1
    # context tuple -> {next word: count}, per order level 1..order
    tables: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    vocab: dict[str, int] = Field(default_factory=dict)
    lengths: list[int] = Field(default_factory=list, description="training target word-lengths")
    n_targets: int = 0

    def train_texts(self, texts: list[str]) -> NGramModel:
        vocab: Counter[str] = Counter()
        tables: dict[int, dict[tuple[str, ...], Counter[str]]] = {n: {} for n in range(1, self.order + 1)}
        lengths: list[int] = []
        n = 0
        for text in texts:
            toks = tokenize(text)
            if not toks:
                continue
            n += 1
            lengths.append(len(toks))
            vocab.update(toks)
            padded = ["<s>"] * (self.order - 1) + toks + ["</s>"]
            for level in range(1, self.order + 1):
                table = tables[level]
                for i in range(len(padded) - level):
                    ctx = tuple(padded[i : i + level - 1]) if level > 1 else ()
                    nxt = padded[i + level - 1]
                    table.setdefault(ctx, Counter())[nxt] += 1
        self.tables = {
            str(level): {json.dumps(list(ctx)): dict(c) for ctx, c in table.items()}
            for level, table in tables.items()
        }
        self.vocab = dict(vocab)
        self.lengths = lengths
        self.n_targets = n
        return self

    def _level_table(self, level: int) -> dict[str, dict[str, int]]:
        return self.tables.get(str(level), {})

    def distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        """Add-k smoothed next-word distribution with backoff to lower orders."""
        V = len(self.vocab) or 1
        for level in range(min(self.order, len(context) + 1), 0, -1):
            ctx = context[-(level - 1) :] if level > 1 else ()
            counts = self._level_table(level).get(json.dumps(list(ctx)))
            if counts:
                total = sum(counts.values())
                return {
                    w: (counts.get(w, 0) + self.smoothing_k) / (total + self.smoothing_k * V)
                    for w in list(counts) + [w for w in self.vocab if w not in counts]
                }
        # empty model: uniform over nothing
        return {}

    def generate(
        self,
        prompt: str = "",
        *,
        seed: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        max_words: int | None = None,
        min_words: int = 3,
        no_repeat_ngram: int = 2,
    ) -> str:
        if not self.vocab:
            return ""
        rng = random.Random(seed if seed is not None else self._default_seed())
        temp = temperature if temperature is not None else 0.9
        k = top_k if top_k is not None else 40
        if max_words is None:
            max_words = rng.choice(self.lengths) if self.lengths else 20
            max_words = max(1, min(max_words, 60))
        prompt_toks = tokenize(prompt)
        context = tuple(prompt_toks[-(self.order - 1) :]) if self.order > 1 else ()
        # prime: use prompt suffix only if the model knows it, else frequent start
        if context and json.dumps(list(context)) not in self._level_table(self.order):
            context = ()
        out: list[str] = []
        for _ in range(max_words):
            dist = self.distribution(context)
            if not dist:
                break
            top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]
            words, probs = zip(*top, strict=True)
            order = _sample_order(rng, words, probs, temp)
            nxt = None
            for candidate in order:
                if candidate in {"</s>", "<s>"}:
                    if len(out) >= min_words:
                        nxt = candidate
                        break
                    continue  # never stop before min_words
                if no_repeat_ngram > 1 and _creates_repeat(out, candidate, no_repeat_ngram):
                    continue  # prefer a non-repeating continuation
                nxt = candidate
                break
            if nxt is None:
                nxt = order[0]  # all candidates repeat: accept the best anyway
            if nxt in {"</s>", "<s>"}:
                break
            out.append(nxt)
            context = (*context, nxt)[-(self.order - 1) :] if self.order > 1 else ()
        return _detokenize(out)

    def _default_seed(self) -> int:
        return 20250917

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> NGramModel:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _sample_order(
    rng: random.Random, words: tuple, probs: tuple, temp: float
) -> list[str]:
    """Candidate words in sampled order (temperature-scaled, without replacement)."""
    if temp <= 0:
        return list(words)
    scaled = [math.exp(math.log(p + 1e-12) / temp) for p in probs]
    remaining = list(zip(words, scaled, strict=True))
    order: list[str] = []
    while remaining:
        s = sum(sp for _, sp in remaining)
        r = rng.random() * s
        for i, (w, sp) in enumerate(remaining):
            r -= sp
            if r <= 0:
                order.append(w)
                remaining.pop(i)
                break
    return order


def _creates_repeat(generated: list[str], candidate: str, n: int) -> bool:
    """True if appending candidate repeats an n-gram already generated."""
    if len(generated) < n - 1:
        return False
    new_gram = tuple([*generated[-(n - 1) :], candidate]) if n > 1 else (candidate,)
    return any(tuple(generated[i : i + n]) == new_gram for i in range(len(generated) - n + 1))


def _detokenize(tokens: list[str]) -> str:
    """Rejoin tokens: punctuation/emoji attach to the previous word."""
    parts: list[str] = []
    for t in tokens:
        if re.fullmatch(r"[^\w\s]", t, re.UNICODE) and parts:
            parts[-1] += t
        else:
            parts.append(t)
    return " ".join(parts).strip()


class NGramGenerator:
    """Drop-in generator (same interface as Mock/Transformers generators)."""

    def __init__(self, model: NGramModel, seed: int = 7) -> None:
        self.model = model
        self.seed = seed
        self.name = f"ngram-order{model.order}"

    def generate(self, messages: list[dict], cfg) -> str:
        prompt = messages[-1].get("content", "") if messages else ""
        max_words = max(8, (cfg.max_new_tokens or 128) // 4)
        return self.model.generate(
            prompt,
            seed=cfg.seed if cfg.seed is not None else self.seed,
            temperature=cfg.temperature if cfg.temperature else 0.9,
            max_words=max_words,
        )
