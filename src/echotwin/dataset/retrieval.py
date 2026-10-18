"""Retrieval-augmented personalization: BM25 over the author's own messages.

Rationale (2026 consensus): fine-tuning locks in style/behaviour, retrieval
supplies content. For chat imitation the strongest cheap pipeline is
``style_rag`` — retrieve the user's *own* similar past replies as few-shot
exemplars — optionally stacked on a LoRA-adapted model (``style_rag_ft``).

Leakage control (critical): retrieval strictly excludes the conversation being
evaluated, so held-out prompts can only retrieve exemplars from OTHER
conversations. Cross-conversation legitimate reuse is fine; within-conversation
retrieval would leak the exact answer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..ingestion.schema import Message
from ..tokenization import tokenize_words as tokenize


@dataclass
class StyleExemplar:
    doc_id: str
    context: str
    response: str
    conversation_id: str
    language: str = "unknown"
    tokens: set[str] = field(default_factory=set)


class BM25Index:
    """Self-contained BM25 (no external dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[StyleExemplar] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.len: list[int] = []
        self.avgdl: float = 0.0

    def add(self, ex: StyleExemplar) -> None:
        toks = tokenize(f"{ex.context} {ex.response}")
        tf = Counter(toks)
        self.docs.append(ex)
        self.tf.append(tf)
        self.len.append(len(toks))
        for t in tf:
            self.df[t] += 1

    def finalize(self) -> None:
        self.avgdl = sum(self.len) / len(self.len) if self.len else 0.0

    def __len__(self) -> int:
        return len(self.docs)

    def search(
        self, query: str, k: int = 5, exclude_conversation_ids: set[str] | None = None
    ) -> list[tuple[StyleExemplar, float]]:
        exclude = exclude_conversation_ids or set()
        q_toks = tokenize(query)
        n = len(self.docs)
        if n == 0 or not q_toks:
            return []
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self.tf):
            if self.docs[i].conversation_id in exclude:
                continue
            score = 0.0
            dl = self.len[i] or 1
            for t in set(q_toks):
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * (f * (self.k1 + 1))
                    / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
                )
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], s) for i, s in scores[:k]]


def build_index_from_messages(messages: list[Message], target_sender_id: str) -> BM25Index:
    """Index every target-authored message with its immediate context."""
    idx = BM25Index()
    msgs = sorted(
        messages,
        key=lambda m: (
            m.conversation_id,
            m.timestamp is None,
            m.timestamp or m.line_number,
            m.line_number,
        ),
    )
    last_by_convo: dict[str, Message] = {}
    for m in msgs:
        if m.sender_id != target_sender_id or not m.is_parseable_content:
            last_by_convo[m.conversation_id] = m
            continue
        prev = last_by_convo.get(m.conversation_id)
        context = ""
        if prev is not None and prev.sender_id != target_sender_id:
            context = prev.text
        idx.add(
            StyleExemplar(
                doc_id=m.message_id,
                context=context,
                response=m.text,
                conversation_id=m.conversation_id,
            )
        )
        last_by_convo[m.conversation_id] = m
    idx.finalize()
    return idx


def build_index_from_examples(examples) -> BM25Index:
    from ..dataset.builder import TrainingExample

    idx = BM25Index()
    for ex in examples:
        if not isinstance(ex, TrainingExample):
            continue
        context = " ".join(t.content for t in ex.messages[:-1])
        idx.add(
            StyleExemplar(
                doc_id=ex.metadata.source_message_id or "",
                context=context,
                response=ex.messages[-1].content,
                conversation_id=ex.metadata.conversation_id,
                language=ex.metadata.language,
            )
        )
    idx.finalize()
    return idx


class RetrievalConfig(BaseModel):
    k: int = Field(default=3, ge=0, le=10)
    min_score: float = 0.0
    exclude_current_conversation: bool = True
    max_context_chars: int = 240


def retrieve_exemplars(
    index: BM25Index,
    prompt_messages: list[dict],
    *,
    conversation_id: str | None = None,
    cfg: RetrievalConfig | None = None,
) -> list[dict]:
    """Return few-shot exemplars [{'context','response'}] for a prompt.

    Never fabricates: returns [] when nothing scores above threshold.

    Leakage control: when ``exclude_current_conversation`` is on (the default)
    and no ``conversation_id`` is supplied there is no way to know which
    conversation is under evaluation, so retrieval returns nothing rather than
    risking the evaluated answer being served back as its own exemplar. Silently
    degrading to "no exclusion" inflated held-out scores in practice.
    """
    cfg = cfg or RetrievalConfig()
    if cfg.k == 0 or not prompt_messages:
        return []
    exclude: set[str] = set()
    if cfg.exclude_current_conversation:
        if not conversation_id:
            return []
        exclude = {conversation_id}
    query = " ".join(m.get("content", "") for m in prompt_messages)
    hits = index.search(query, k=cfg.k, exclude_conversation_ids=exclude)
    out: list[dict] = []
    for ex, score in hits:
        if score < cfg.min_score:
            continue
        out.append(
            {
                "context": ex.context[: cfg.max_context_chars],
                "response": ex.response[: cfg.max_context_chars],
                "score": round(float(score), 4),
                "conversation_id": ex.conversation_id,
                "language": ex.language,
            }
        )
    return out
