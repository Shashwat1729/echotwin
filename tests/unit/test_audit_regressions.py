"""Regression tests for defects found in the correctness audit.

Each test here failed against the pre-fix code. They are written to be
behavior-level, not implementation-level, so a rewrite that keeps the
behavior correct still passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from echotwin.dataset.builder import BuiltDataset, DatasetConfig, TrainingExample
from echotwin.dataset.retrieval import (
    BM25Index,
    RetrievalConfig,
    StyleExemplar,
    retrieve_exemplars,
    tokenize,
)
from echotwin.dataset.splits import SplitConfig, split_dataset
from echotwin.evaluation.contamination import memorization_overlap
from echotwin.ingestion import WhatsAppParser
from echotwin.models.ngram import NGramConfig, NGramGenerator, NGramModel
from echotwin.training.sft import format_for_sft
from echotwin.voice.audio_quality import analyze_wave_array
from echotwin.voice.capabilities import VoiceBackendCapabilities
from echotwin.voice.reference_selection import ReferenceCandidate, select_references

# ---------------------------------------------------------------------------
# Ingestion: real user text must never be swallowed as a system event
# ---------------------------------------------------------------------------
USER_TEXT_THAT_LOOKS_LIKE_NOTICES = [
    "I removed the old files from my phone",
    "I added you to my contact list",
    "I changed the subject to project x",
    "I turned off the notifications for this chat",
    "I created a group for the project",
    "just a normal message",
]


class TestSystemMessageClassification:
    @pytest.mark.parametrize("text", USER_TEXT_THAT_LOOKS_LIKE_NOTICES)
    def test_attributed_user_text_is_never_system(self, text):
        res = WhatsAppParser().parse_text(f"15/01/2025, 10:00:00 - Alice: {text}\n", "c")
        assert len(res.messages) == 1
        msg = res.messages[0]
        assert not msg.is_system_message
        assert msg.message_type.value == "text"
        assert msg.sender_name == "Alice"
        assert msg.is_parseable_content, "user text was dropped from the training set"

    def test_user_text_survives_dataset_building(self):
        body = "\n".join(
            f"15/01/2025, 10:0{i}:00 - Alice: {t}"
            for i, t in enumerate(USER_TEXT_THAT_LOOKS_LIKE_NOTICES)
        )
        res = WhatsAppParser().parse_text(body + "\n", "c")
        trainable = [m for m in res.messages if m.is_parseable_content]
        assert len(trainable) == len(USER_TEXT_THAT_LOOKS_LIKE_NOTICES)

    @pytest.mark.parametrize(
        "line",
        [
            "Messages and calls are end-to-end encrypted.",
            'Rohit created group "Hostel Crew"',
            "Rohit added Priya and Sana",
            "Sana left",
            "+91 98765 43210 turned on disappearing messages",
            "Priya changed the group description to study group",
        ],
    )
    def test_genuine_system_notices_still_detected(self, line):
        res = WhatsAppParser().parse_text(f"[10/01/2025, 20:00:00] {line}\n", "c")
        assert len(res.messages) == 1
        assert res.messages[0].is_system_message, f"missed system notice: {line}"
        assert res.messages[0].sender_id is None

    def test_literal_system_sender_is_a_notice(self):
        res = WhatsAppParser().parse_text(
            "[15/01/2025, 09:25:00] System: Messages and calls are end-to-end encrypted.\n", "c"
        )
        assert res.messages[0].is_system_message

    def test_detected_formats_has_no_false_unbracketed(self):
        res = WhatsAppParser().parse_text(
            "[15/01/2025, 10:00:00] A: one\n[15/01/2025, 10:01:00] A: two\n", "c"
        )
        assert "bracketed" in res.report.detected_formats
        assert "unbracketed" not in res.report.detected_formats

    def test_garbage_after_truncated_header_is_not_absorbed(self):
        res = WhatsAppParser().parse_text(
            "15/01/2025, 10:00:00 - Alice: alice body here\n"
            "15/01/2025, 10:0\n"
            "orphaned continuation line\n"
            "15/01/2025, 10:02:00 - Bob: bob body\n",
            "c",
        )
        alice = next(m for m in res.messages if m.sender_name == "Alice")
        assert "orphaned" not in alice.text
        assert any(e.reason == "truncated_or_orphan_header" for e in res.errors)


# ---------------------------------------------------------------------------
# n-gram: end-of-sentence learning + probability normalization
# ---------------------------------------------------------------------------
class TestNgramTermination:
    @pytest.fixture
    def model(self) -> NGramModel:
        m = NGramModel(order=3)
        m.train_texts(
            [
                "haan bro kaise ho",
                "main thoda busy hoon",
                "kal milte h library",
                "hoon aaj kal ka plan kya hai",
                "bhai sab theek hai na yaar",
            ]
        )
        return m

    def test_eos_is_in_vocab_and_reachable(self, model: NGramModel):
        assert "</s>" in model.vocab
        reachable = any("</s>" in counts for counts in model.tables["1"].values()) or any(
            "</s>" in counts for counts in model.tables["3"].values()
        )
        assert reachable, "model cannot learn to end a message"

    @pytest.mark.parametrize("context", [(), ("library",), ("yaar",), ("kaise", "ho")])
    def test_distribution_sums_to_one(self, model: NGramModel, context):
        dist = model.distribution(context)
        assert dist
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9)

    def test_generation_terminates_without_hitting_cap(self, model: NGramModel):
        lengths = [len(model.generate("", seed=s, max_words=30).split()) for s in range(40)]
        assert not any(x >= 30 for x in lengths), (
            "every generation ran to the cap: the model never learned </s>"
        )

    def test_config_is_actually_honored(self, model: NGramModel):
        from echotwin.models.generation import GeneratorConfig

        greedy = NGramGenerator(model, config=NGramConfig(temperature=0.05, top_k=1))
        a = greedy.generate([{"role": "user", "content": "x"}], GeneratorConfig(max_new_tokens=40))
        b = greedy.generate([{"role": "user", "content": "x"}], GeneratorConfig(max_new_tokens=40))
        assert a == b, "low temperature + top_k=1 must be deterministic"
        assert a

    def test_seed_determinism(self, model: NGramModel):
        assert model.generate("", seed=11) == model.generate("", seed=11)


# ---------------------------------------------------------------------------
# Contamination: verbatim copies of any length must be caught
# ---------------------------------------------------------------------------
class TestContaminationDetection:
    def test_short_generation_verbatim_inside_long_target(self):
        r = memorization_overlap(
            ["going to the market tomorrow"],
            ["i am going to the market tomorrow morning with my brother and sister"],
        )
        assert r.verdict == "high_overlap"
        assert r.exact_match_rate == 1.0

    def test_short_target_inside_long_generation(self):
        r = memorization_overlap(
            ["i am going to the market tomorrow morning with my brother and sister today"],
            ["going to the market tomorrow"],
        )
        assert r.verdict == "high_overlap"

    def test_devanagari_verbatim_copy(self):
        r = memorization_overlap(["आज मौसम बहुत अच्छा है"], ["आज मौसम बहुत अच्छा है लगता है"])
        assert r.verdict == "high_overlap"

    def test_genuinely_independent_text_is_clean(self):
        r = memorization_overlap(
            ["haan yaar kal milte h bhai"],
            ["the meeting starts at ten thirty in the morning", "paneer tikka kaisa tha"],
        )
        assert r.verdict == "clean"
        assert r.near_match_rate == 0.0

    def test_no_train_data_is_clean(self):
        assert memorization_overlap(["anything"], []).verdict == "clean"


# ---------------------------------------------------------------------------
# Retrieval: leakage control and script coverage
# ---------------------------------------------------------------------------
class TestRetrievalLeakageAndCoverage:
    def _index(self) -> BM25Index:
        idx = BM25Index()
        idx.add(StyleExemplar("d1", "what time tomorrow", "meet at 5", "CONVO_UNDER_TEST"))
        idx.add(StyleExemplar("d2", "what time tomorrow", "meet at 6", "other-convo"))
        idx.finalize()
        return idx

    def test_no_conversation_id_never_leaks(self):
        out = retrieve_exemplars(
            self._index(),
            [{"role": "user", "content": "what time tomorrow"}],
            cfg=RetrievalConfig(k=3),
        )
        assert out == [], (
            "without a conversation_id there is no way to exclude the evaluated "
            "conversation, so retrieval must return nothing"
        )

    def test_conversation_is_excluded_when_known(self):
        out = retrieve_exemplars(
            self._index(),
            [{"role": "user", "content": "what time tomorrow"}],
            conversation_id="CONVO_UNDER_TEST",
            cfg=RetrievalConfig(k=3),
        )
        assert out
        assert all(e["conversation_id"] != "CONVO_UNDER_TEST" for e in out)

    def test_exclusion_can_be_disabled_explicitly(self):
        out = retrieve_exemplars(
            self._index(),
            [{"role": "user", "content": "what time tomorrow"}],
            cfg=RetrievalConfig(k=3, exclude_current_conversation=False),
        )
        assert len(out) == 2

    @pytest.mark.parametrize(
        ("text", "expect_present"),
        [
            ("Room 302 at 5pm", "302"),
            ("café naïve", "café"),
            ("Привет мир", "привет"),
            ("good morning 😀", "good"),
        ],
    )
    def test_tokenizer_keeps_digits_and_scripts(self, text, expect_present):
        assert expect_present in tokenize(text)

    def test_devanagari_still_tokenized(self):
        assert tokenize("आज मौसम अच्छा है")


# ---------------------------------------------------------------------------
# Splits: fraction math and chronological honesty
# ---------------------------------------------------------------------------
def _examples(n: int, dated: bool = True) -> list[TrainingExample]:
    return [
        TrainingExample.model_validate(
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i} reply text"},
                ],
                "metadata": {
                    "conversation_id": f"c{i}",
                    "language": "en",
                    "example_type": "natural_reply",
                    "timestamp": f"2025-01-{i + 1:02d}" if dated else None,
                },
            }
        )
        for i in range(n)
    ]


def _built(exs):
    return BuiltDataset(examples=exs, config=DatasetConfig(), stats={})


class TestSplitMath:
    def test_explicit_zero_fraction_holds_out_nothing(self):
        split = split_dataset(
            _built(_examples(10)), SplitConfig(test_fraction=0.0, val_fraction=0.0)
        )
        assert len(split.test) == 0
        assert len(split.val) == 0
        assert len(split.train) == 10

    def test_small_corpus_still_gets_a_holdout(self):
        split = split_dataset(_built(_examples(4)), SplitConfig())
        assert len(split.test) >= 1
        assert len(split.train) >= 1
        assert split.report.leakage_check_passed

    def test_chronological_without_dates_warns(self):
        split = split_dataset(
            _built(_examples(6, dated=False)),
            SplitConfig(mode="chronological", test_fraction=0.25),
        )
        assert any("no example has a timestamp" in w for w in split.report.warnings)

    def test_chronological_holds_out_newest(self):
        split = split_dataset(
            _built(_examples(8, dated=True)),
            SplitConfig(mode="chronological", test_fraction=0.25, val_fraction=0.25),
        )
        assert split.test
        assert min(ex.metadata.timestamp for ex in split.test) > min(
            ex.metadata.timestamp for ex in split.train
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10])
    def test_train_never_empties_and_no_leak(self, n):
        split = split_dataset(_built(_examples(n)), SplitConfig())
        assert len(split.train) >= 1, "split left the model with no training data"
        assert split.report.leakage_check_passed


# ---------------------------------------------------------------------------
# SFT masking
# ---------------------------------------------------------------------------
class _WordTokenizer:
    """Deterministic word-level tokenizer with a chat template."""

    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def _ids(self, text: str, **kw):
        ids = [1000 + (abs(hash(w)) % 9000) for w in text.split()]
        if kw.get("truncation") and isinstance(kw.get("max_length"), int):
            ids = ids[: kw["max_length"]]
        return ids

    def __call__(self, text, **kw):
        ids = self._ids(text, **kw)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TestSFTMasking:
    def test_echoed_reply_does_not_unmask_the_user_turn(self):
        tok = _WordTokenizer()
        h = lambda w: 1000 + (abs(hash(w)) % 9000)  # noqa: E731
        messages = [
            {"role": "user", "content": "sure"},
            {"role": "assistant", "content": "sure"},
            {"role": "user", "content": "and now"},
            {"role": "assistant", "content": "ok"},
        ]
        enc = format_for_sft(tok, messages, 512, train_on_last_only=False)
        unmasked = {
            i for i, lab in zip(enc["input_ids"], enc["labels"], strict=True) if lab != -100
        }
        # "sure" appears twice; only the ASSISTANT occurrence may be unmasked.
        assert unmasked == {h("sure"), h("ok")}

    def test_last_only_masks_the_prompt(self):
        tok = _WordTokenizer()
        enc = format_for_sft(
            tok,
            [
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "haan yaar"},
            ],
            512,
        )
        assert enc["trainable_tokens"] > 0
        assert any(lab != -100 for lab in enc["labels"])

    def test_truncation_that_cuts_the_target_raises_instead_of_guessing(self):
        """A target truncated away must be an explicit error, not a silent guess.

        Training on a boundary we cannot locate would mask the wrong tokens, so
        ``format_for_sft`` refuses; the dataset layer drops such examples.
        """
        from echotwin.exceptions import TrainingError

        tok = _WordTokenizer()
        with pytest.raises(TrainingError):
            format_for_sft(
                tok,
                [
                    {"role": "user", "content": "x " * 400},
                    {"role": "assistant", "content": "y " * 400},
                ],
                64,
            )


# ---------------------------------------------------------------------------
# Voice reference selection
# ---------------------------------------------------------------------------
class TestReferenceEligibility:
    def _caps(self) -> VoiceBackendCapabilities:
        return VoiceBackendCapabilities(
            name="t", languages=["en", "hi"], min_reference_seconds=3.0, max_reference_seconds=60.0
        )

    def test_over_length_clip_is_not_selected(self):
        d = select_references(
            [
                ReferenceCandidate(
                    recording_id="toolong",
                    language="en",
                    quality_verdict="ok",
                    duration_seconds=300.0,
                )
            ],
            self._caps(),
        )
        assert d.selected == []
        assert any(r["reason"] == "duration_too_long" for r in d.rejected)

    def test_candidate_is_rejected_only_once(self):
        d = select_references(
            [
                ReferenceCandidate(
                    recording_id="bad",
                    language="en",
                    quality_verdict="fail",
                    duration_seconds=0.1,
                )
            ],
            self._caps(),
        )
        real = [r for r in d.rejected if r["recording_id"] == "bad"]
        assert len(real) == 1, f"double-counted rejection: {real}"

    def test_good_candidates_still_selected(self):
        d = select_references(
            [
                ReferenceCandidate(
                    recording_id="ok_en", language="en", quality_verdict="ok", duration_seconds=10.0
                ),
                ReferenceCandidate(
                    recording_id="ok_hi", language="hi", quality_verdict="ok", duration_seconds=8.0
                ),
            ],
            self._caps(),
        )
        assert d.selected


class TestQualitySeverityContract:
    def test_clipping_uses_error_severity(self):
        t = np.arange(16000 * 3) / 16000
        clip = np.clip(1.4 * np.sin(2 * np.pi * 180 * t), -1, 1).astype(np.float32)
        report = analyze_wave_array(clip, 16000)
        severities = {i.severity for i in report.issues}
        assert "failure" not in severities, "severity must match the declared contract"
        assert "error" in severities
        assert report.verdict == "fail"
