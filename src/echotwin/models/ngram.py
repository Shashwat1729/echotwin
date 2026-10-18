"""Pure-Python word n-gram language model for CPU-only style imitation.

Why this exists: neural fine-tuning needs a GPU, downloads, and torch. An
add-k smoothed word n-gram model with backoff trains in milliseconds on CPU
with zero heavy dependencies, and for short informal chat messages it is a
genuinely useful personalization method â€” historically *the* method for style
imitation, and still a strong baseline that every neural approach must beat.

Honest limits (documented, not hidden):
- No long-range coherence; quality degrades past ~15 words.
- Rare words fall back to unigram soup; the model can repeat training phrases
  verbatim â€” the contamination check in evaluation measures exactly this.
- Case is preserved (style signal) at the cost of sparsity.

Deterministic given a seed. Save/load as plain JSON.
"""

from __future__ import annotations

import hashlib
import itertools
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
    smoothing_k: float = Field(default=0.1, ge=0.0)
    # Base seed. Combined with the prompt when no explicit seed is given, so
    # unrelated prompts do not all receive the identical reply.
    seed: int = 20250917
    # context tuple -> {next word: count}, per order level 1..order
    tables: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    vocab: dict[str, int] = Field(default_factory=dict)
    lengths: list[int] = Field(default_factory=list, description="training target word-lengths")
    n_targets: int = 0
    # Normalized training targets, used to block verbatim replay at decode time.
    training_texts: list[str] = Field(default_factory=list)
    # Document frequency of each token across training targets, used to bias the
    # opening of a generation toward vocabulary relevant to the prompt.
    doc_freq: dict[str, int] = Field(default_factory=dict)
    # Tokens the author actually wrote twice in a row. Precomputed so the
    # decode-time self-loop check is an O(1) set lookup.
    doubled_tokens: set[str] = Field(default_factory=set)

    def train_texts(self, texts: list[str]) -> NGramModel:
        vocab: Counter[str] = Counter()
        tables: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: {} for n in range(1, self.order + 1)
        }
        lengths: list[int] = []
        kept_texts: list[str] = []
        doc_freq: dict[str, int] = {}
        doubled: set[str] = set()
        n = 0
        for text in texts:
            toks = tokenize(text)
            if not toks:
                continue
            n += 1
            lengths.append(len(toks))
            vocab.update(toks)
            kept_texts.append(text)
            for w in set(toks):
                doc_freq[w] = doc_freq.get(w, 0) + 1
            for a, b in itertools.pairwise(toks):
                if a == b:
                    doubled.add(a)
            padded = ["<s>"] * (self.order - 1) + toks + ["</s>"]
            for level in range(1, self.order + 1):
                table = tables[level]
                # `+ 1` is required: with `len(padded) - level` the final window
                # â€” the one whose target IS `</s>` â€” was never emitted, so the
                # model could not learn to end a message and always ran to the
                # generation cap.
                for i in range(len(padded) - level + 1):
                    ctx = tuple(padded[i : i + level - 1]) if level > 1 else ()
                    nxt = padded[i + level - 1]
                    table.setdefault(ctx, Counter())[nxt] += 1
        # `</s>` is a real next-token, so it belongs in the vocabulary. Without it
        # the smoothing denominator under-counted the support (summed to ~1.017).
        vocab.update({"<s>", "</s>"})
        self.tables = {
            str(level): {json.dumps(list(ctx)): dict(c) for ctx, c in table.items()}
            for level, table in tables.items()
        }
        self.vocab = dict(vocab)
        self.lengths = lengths
        self.n_targets = n
        self.training_texts = kept_texts
        self.doc_freq = doc_freq
        self.doubled_tokens = doubled
        return self

    def _level_table(self, level: int) -> dict[str, dict[str, int]]:
        return self.tables.get(str(level), {})

    def distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        """Add-k smoothed next-word distribution with backoff to lower orders.

        The result is a true probability distribution (sums to 1.0) over the
        union of the observed next-words for this context and the full vocab.
        """
        if not self.vocab:
            return {}
        for level in range(min(self.order, len(context) + 1), 0, -1):
            ctx = context[-(level - 1) :] if level > 1 else ()
            counts = self._level_table(level).get(json.dumps(list(ctx)))
            if counts:
                total = sum(counts.values())
                # The support is the union of observed next-words and the vocab.
                # Using len(vocab) alone under-counts whenever a context
                # contains a token outside it (the sums drifted above 1.0).
                support = list(counts) + [w for w in self.vocab if w not in counts]
                denom = total + self.smoothing_k * len(support)
                if denom <= 0:
                    return {}
                return {w: (counts.get(w, 0) + self.smoothing_k) / denom for w in support}
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
        context_bias: float = 0.6,
        copy_penalty: float = 2.5,
        max_copy_run: int = 3,
    ) -> str:
        if not self.vocab:
            return ""
        # Deterministic per (model seed, prompt) when no explicit seed is given.
        # A single constant seed made every out-of-domain prompt answer with the
        # SAME sentence, which looked like a broken model. The prompt digest uses
        # blake2b (not the builtin hash(), which is salted per process) so the
        # behaviour is reproducible across runs.
        if seed is None:
            seed = self.seed
            if prompt:
                digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
                seed = (self.seed * 1_000_003 + int.from_bytes(digest, "big")) % (2**63)
        rng = random.Random(seed)
        temp = temperature if temperature is not None else 0.9
        k = top_k if top_k is not None else 40
        # Sample a target length from the author's real message lengths;
        # `max_words` (if given) is only an upper bound.
        target = rng.choice(self.lengths) if self.lengths else 20
        hard_cap = max(1, min(max_words if max_words is not None else 60, 60))
        target = max(1, min(target, hard_cap))
        # Length control. Two separate notions:
        #   * `stop_floor`  — a HARD minimum: the reply may never be shorter.
        #   * `soft_target` — the author's typical message length, sampled from
        #     their own distribution. Once the reply reaches it, ending becomes
        #     *likely* (see the EOS boost in _sample).
        # Previously the target was used as the hard floor, so the reply could
        # not end before it, and EOS probability stayed flat afterwards — which
        # made replies pile up at the generation ceiling (mean 10.6 words for an
        # author whose messages average 6.0).
        stop_floor = max(1, min(min_words, hard_cap))
        soft_target = max(stop_floor, min(target, hard_cap))
        prompt_toks = tokenize(prompt)
        context = tuple(prompt_toks[-(self.order - 1) :]) if self.order > 1 else ()
        # prime: use prompt suffix only if the model knows it, else frequent start
        if context and json.dumps(list(context)) not in self._level_table(self.order):
            context = ()

        # --- prompt conditioning -------------------------------------------
        # An unseen context used to reset to the unigram prior, so every prompt
        # produced the SAME reply. Bias the first token toward vocabulary that
        # the author actually used alongside the prompt's content words, which
        # makes the reply respond to what was said.
        bias: dict[str, float] = {}
        if context_bias > 0 and prompt_toks:
            seen = self.doc_freq
            if seen:
                stop = {t for t in prompt_toks if seen.get(t, 0) <= 1}
                for t in prompt_toks:
                    if t in stop or len(t) < 2:
                        continue
                    freq = seen.get(t)
                    if freq:
                        bias[t] = min(1.0, freq / max(1, self.n_targets)) * context_bias
                # Tokens that follow the prompt's words somewhere in the corpus
                # are the strongest evidence of a relevant continuation.
                for i, w in enumerate(prompt_toks[:-1]):
                    nxt = prompt_toks[i + 1]
                    if w in self.vocab and nxt in self.vocab and nxt not in {"</s>", "<s>"}:
                        bias[nxt] = bias.get(nxt, 0.0) + context_bias

        out: list[str] = []
        # Never generate a message longer than the longest one the person ever
        # wrote (with slack); a chat reply that is 3x their longest real message
        # is an obvious artifact.
        length_ceiling = int(max(self.lengths) * 1.5) + 1 if self.lengths else hard_cap
        ceiling = max(stop_floor, min(hard_cap, length_ceiling))
        out = self._sample(
            rng,
            temp=temp,
            k=k,
            ceiling=ceiling,
            stop_floor=stop_floor,
            soft_target=soft_target,
            no_repeat_ngram=no_repeat_ngram,
            start_context=context,
            bias=bias,
        )
        text = _detokenize(out)
        # --- copy penalty --------------------------------------------------
        # A small corpus makes verbatim replay common, and a reply that is a
        # substring of a training message is memorization, not style. Retry with
        # a fresh seed a couple of times before giving up, so the model still
        # answers rather than returning an empty string.
        if copy_penalty > 0 and self.training_texts:
            # Escalating temperature on each retry: a hotter sample escapes a
            # copy that a cool one keeps re-deriving.
            for attempt in range(3):
                if not _reproduces_training(text, self.training_texts, max_copy_run):
                    break
                retry_rng = random.Random(rng.random())
                retry = _detokenize(
                    self._sample(
                        retry_rng,
                        temp=min(1.4, temp + 0.2 * (attempt + 1)),
                        k=k,
                        ceiling=ceiling,
                        stop_floor=stop_floor,
                        soft_target=soft_target,
                        no_repeat_ngram=no_repeat_ngram,
                        start_context=context,
                        # Keep the prompt conditioning: the retry exists to
                        # escape a verbatim copy, which is most likely on an
                        # in-distribution prompt — exactly where dropping the
                        # bias hurts most. The bias decays inside _sample.
                        bias=bias,
                    )
                )
                # A retry that produced nothing must not replace real output.
                if not retry.strip():
                    break
                text = retry
        return text

    def _sample(
        self,
        rng: random.Random,
        *,
        temp: float,
        k: int,
        ceiling: int,
        stop_floor: int,
        soft_target: int,
        no_repeat_ngram: int,
        start_context: tuple[str, ...],
        bias: dict[str, float] | None,
        eos_boost: float = 6.0,
    ) -> list[str]:
        """Sample one message.

        Single source of truth for the decode loop: the copy-penalty retry used
        to run a second, subtly different copy of this loop, which is how the
        retry path ended up able to return an empty string.
        """
        out: list[str] = []
        context = start_context
        local_bias = dict(bias) if bias else None
        for _ in range(ceiling):
            dist = self.distribution(context)
            if not dist:
                break
            # `<s>` only pads contexts; it is never something the person wrote.
            # Filter BEFORE the top-k slice: slicing first could drop every real
            # word (at an empty context `<s>` is the argmax), leaving nothing.
            ranked = [(w, p) for w, p in sorted(dist.items(), key=lambda kv: -kv[1]) if w != "<s>"]
            if not ranked:
                break
            if local_bias and out:
                # The bias describes how to OPEN a reply. Re-applying it at every
                # step re-raised the same tokens forever and produced loops like
                # "yaar yaar, yaar abhi ... yaar", so it decays after the opening.
                local_bias = {w: v * 0.25 for w, v in local_bias.items() if v > 0.05}
            if local_bias:
                # Renormalize so the sample stays a distribution.
                scale = sum(p for _w, p in ranked)
                if scale > 0:
                    ranked = [(w, p / scale + local_bias.get(w, 0.0)) for w, p in ranked]
                    ranked.sort(key=lambda kv: -kv[1])
            # Length shaping: once the reply has reached the author's typical
            # message length, ending should become the most likely outcome.
            # Without this the EOS probability stayed flat (~0.1) at every
            # position, so replies that did not stop early ran all the way to
            # the generation ceiling and were ~75% longer than the author's.
            if len(out) >= soft_target and eos_boost > 0:
                # The boost grows the further the reply runs past the author's
                # typical length, so a reply that overshoots converges instead
                # of piling up at the generation ceiling.
                overshoot = len(out) - soft_target
                boost = eos_boost * (1.0 + overshoot)
                ranked = [(w, p * boost if w == "</s>" else p) for w, p in ranked]
                ranked.sort(key=lambda kv: -kv[1])
            top = ranked[:k]
            if not top:
                break
            words, probs = zip(*top, strict=True)
            order = _sample_order(rng, words, probs, temp)
            nxt = None
            for candidate in order:
                if candidate in {"</s>", "<s>"}:
                    if len(out) >= stop_floor:
                        nxt = candidate
                        break
                    continue  # too early to end
                # A reply never starts with punctuation or a bare emoji.
                if not out and _is_punct_or_symbol(candidate):
                    continue
                if no_repeat_ngram > 1 and _creates_repeat(out, candidate, no_repeat_ngram):
                    continue
                # Never emit the same token twice in a row unless the author wrote
                # such a doubling; it is the classic decode loop.
                if out and candidate == out[-1] and out[-1] not in self.doubled_tokens:
                    continue
                nxt = candidate
                break
            if nxt is None:
                # Every sampled candidate was rejected. Take the best real word
                # from the full distribution rather than returning nothing.
                real = [
                    c
                    for c in order
                    if c not in {"</s>", "<s>"} and not (not out and _is_punct_or_symbol(c))
                ]
                if real:
                    nxt = real[0]
                else:
                    best_real = [
                        (w, p)
                        for w, p in sorted(dist.items(), key=lambda kv: -kv[1])
                        if w not in {"</s>", "<s>"} and not (not out and _is_punct_or_symbol(w))
                    ]
                    if best_real:
                        nxt = best_real[0][0]
                    else:
                        # This author only ever writes punctuation/emoji. Their
                        # reply IS punctuation, so allow it rather than
                        # returning an empty message.
                        any_token = [
                            (w, p)
                            for w, p in sorted(dist.items(), key=lambda kv: -kv[1])
                            if w not in {"</s>", "<s>"}
                        ]
                        nxt = any_token[0][0] if any_token else None
            if nxt is None or nxt in {"</s>", "<s>"}:
                break
            out.append(nxt)
            context = (*context, nxt)[-(self.order - 1) :] if self.order > 1 else ()
        return out

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


def _sample_order(rng: random.Random, words: tuple, probs: tuple, temp: float) -> list[str]:
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


def _is_punct_or_symbol(token: str) -> bool:
    """True for punctuation/emoji marks, which cannot open a sentence."""
    return not re.fullmatch(r"[\w\u0900-\u097F']+", token or "", re.UNICODE)


def _reproduces_training(text: str, training_texts: list[str], min_words: int) -> bool:
    """True if `text` is a slice of a training message.

    ``min_words`` is a floor so a 1-token reply is not flagged on a token that
    happens to be common; 3+ word copies are the ones worth retrying.
    """
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(norm.split()) < min(3, min_words):
        return False
    for t in training_texts:
        tn = re.sub(r"\s+", " ", t.strip().lower())
        if norm and (norm in tn or tn in norm):
            return True
    return False


def _doubles_in_training(token: str, training_texts: list[str]) -> bool:
    """True if the author ever wrote this token twice in a row.

    Kept for clarity, but generation uses the precomputed
    ``NGramModel.doubled_tokens`` set instead: this scans the whole corpus and
    is called per candidate, which made generation ~6x slower.
    """
    pat = re.compile(rf"(?:^|\s){re.escape(token)}(?:\s+){re.escape(token)}(?:\s|$)")
    return any(pat.search(t) for t in training_texts)


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

    def __init__(
        self,
        model: NGramModel,
        seed: int = 7,
        config: NGramConfig | None = None,
    ) -> None:
        self.model = model
        self.seed = seed
        # Previously the config was accepted but never read: temperature and
        # top_k were hardcoded, so tuning them silently did nothing.
        self.config = config or NGramConfig(order=model.order, seed=seed)
        self.name = f"ngram-order{model.order}"

    def generate(self, messages: list[dict], cfg) -> str:
        prompt = messages[-1].get("content", "") if messages else ""
        # `max_new_tokens` is a SAFETY CEILING, not a target length. Passing
        # `max_new_tokens // 4` as the target made every reply run to that cap
        # (~16 words) instead of the length the author actually writes
        # (mean 7.8 on a real corpus), so replies came out roughly twice too
        # long. Let the model sample its own target from the training length
        # distribution and only clamp it to the caller's ceiling.
        hard_ceiling = (cfg.max_new_tokens or 128) // 4
        return self.model.generate(
            prompt,
            # None means "derive a seed from the prompt" inside the model, so
            # the same prompt is reproducible while different prompts get
            # different replies. An explicit cfg.seed pins everything.
            seed=cfg.seed,
            # `or` would discard an explicit 0.0; only None means "unset".
            temperature=(
                cfg.temperature if cfg.temperature is not None else self.config.temperature
            ),
            # The caller's top_k wins: top_k=1 is the standard determinism knob
            # and it was silently ignored, so greedy and unconstrained decoding
            # were indistinguishable.
            top_k=cfg.top_k if cfg.top_k is not None else self.config.top_k,
            max_words=hard_ceiling,
        )
