"""One tokenizer for the whole project.

Before this module, four different tokenizers coexisted and disagreed about
what a "word" is:

* ``dataset/retrieval.py`` used ``\\w`` — which matches ``str.isalnum()``, and
  therefore **dropped every Devanagari matra** (category ``Mn``/``Mc``), shredding
  Hindi words into bare consonants.
* ``models/ngram.py`` matched ``[\\w\\u0900-\\u097F']+`` — Devanagari only.
* ``evaluation/metrics.py`` matched ``[A-Za-z\\u0900-\\u097F']+`` — no digits, no
  Cyrillic, no Arabic, no emoji.
* ``evaluation/contamination.py`` matched ``[\\w\\u0900-\\u097F']+``.

The result was that style metrics reported ``0.0`` for non-Latin personas and
retrieval was effectively random for Hindi — while looking like it had worked.

This module defines a single rule, applied identically everywhere:

    A word is a maximal run of characters whose Unicode category starts with
    ``L`` (letter), ``M`` (mark) or ``N`` (number), optionally joined by an
    apostrophe.

That covers Latin, Devanagari, Bengali/Tamil/Telugu/Kannada/Malayalam,
Cyrillic, Arabic, Han, emoji, and digits, and it needs no per-script ranges
and no optional dependency.
"""

from __future__ import annotations

import unicodedata

#: Characters that may join two word parts without breaking the word.
JOINERS = frozenset({"'", "\u2019", "\u02bc", "_"})

#: Unicode category prefixes that count as part of a word.
_WORD_CATEGORIES = ("L", "M", "N")


def is_word_char(ch: str) -> bool:
    """True when ``ch`` is a letter, mark or number (and therefore word-ish)."""
    if not ch:
        return False
    return unicodedata.category(ch).startswith(_WORD_CATEGORIES)


def tokenize_words(text: str) -> list[str]:
    """Split ``text`` into lowercase words.

    >>> tokenize_words("Haan yaar, 302 rooms — नहीं!")
    ['haan', 'yaar', '302', 'rooms', 'नहीं']
    """
    if not text:
        return []
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if is_word_char(ch):
            current.append(ch.lower())
        elif ch in JOINERS and current:
            # A joiner only continues the word if a word character follows.
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current).strip("".join(JOINERS)))
                current = []
    if current:
        tokens.append("".join(current).strip("".join(JOINERS)))
    return [t for t in tokens if t]


def token_set(text: str) -> set[str]:
    """Distinct tokens of ``text`` (for containment / overlap checks)."""
    return set(tokenize_words(text))


def ngram_windows(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """All contiguous ``n``-token windows (a short text yields itself)."""
    if n <= 0:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
