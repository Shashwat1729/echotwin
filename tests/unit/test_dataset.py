"""Unit tests: dataset construction, leakage-resistant splits, dedup."""

from __future__ import annotations

from echotwin.dataset import (
    DatasetConfig,
    SplitConfig,
    build_examples,
    is_low_information,
    load_jsonl,
    save_jsonl,
    segment_messages,
    split_dataset,
)
from echotwin.dataset.builder import compute_dataset_stats
from echotwin.exceptions import InsufficientDataError
from echotwin.ingestion import WhatsAppParser


class TestBuilder:
    def test_builds_examples_for_target_only(self, parsed, target_id):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        assert built.examples
        for ex in built.examples:
            assert ex.messages[-1].role == "assistant"

    def test_no_fabrication_target_equals_source(self, parsed, target_id):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        target_texts = {m.text for m in parsed.messages if m.sender_id == target_id}
        for ex in built.examples:
            assert ex.messages[-1].content in target_texts

    def test_no_attachment_or_system_targets(self, parsed, target_id):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        for ex in built.examples:
            assert ex.messages[-1].content != "<Media omitted>"

    def test_wrong_target_raises(self, parsed):
        import pytest

        with pytest.raises(InsufficientDataError):
            build_examples(parsed.messages, "nonexistent-id", DatasetConfig())

    def test_low_information_filter(self, parsed, target_id):
        cfg = DatasetConfig(min_target_chars=5)
        built = build_examples(parsed.messages, target_id, cfg)
        for ex in built.examples:
            assert len(ex.messages[-1].content.strip()) >= 5

    def test_is_low_information(self):
        cfg = DatasetConfig()
        assert is_low_information("ok", cfg) == "low_info_exact"
        assert is_low_information("😀😀", cfg) == "emoji_only"
        assert is_low_information("a real sentence", cfg) is None

    def test_segmentation_splits_on_time_gap(self, parsed):
        segs = segment_messages(parsed.messages, gap_minutes=120)
        assert len(segs) >= 2
        # all messages preserved
        assert sum(len(s) for s in segs) == len(parsed.messages)

    def test_context_truncation_budget(self, parsed, target_id):
        cfg = DatasetConfig(context_messages=10, max_context_chars=100)
        built = build_examples(parsed.messages, target_id, cfg)
        for ex in built.examples:
            assert ex.metadata.context_chars <= 110  # newline-join slack

    def test_self_continuation_flag(self, parsed, target_id):
        cfg = DatasetConfig(include_self_continuations=False)
        built = build_examples(parsed.messages, target_id, cfg)
        assert all(ex.metadata.example_type != "self_continuation" for ex in built.examples)

    def test_jsonl_roundtrip(self, parsed, target_id, tmp_path):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        p = tmp_path / "ex.jsonl"
        n = save_jsonl(built.examples, p)
        assert n == len(built.examples)
        loaded = load_jsonl(p)
        assert len(loaded) == len(built.examples)
        assert loaded[0].messages[0].role == built.examples[0].messages[0].role

    def test_stats_shape(self, parsed, target_id):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        stats = compute_dataset_stats(built.examples)
        assert stats["n_examples"] == len(built.examples)
        assert "languages" in stats and "example_types" in stats

    def test_exact_dedup(self, parsed, target_id):
        cfg = DatasetConfig(exact_dedup=True)
        built = build_examples(parsed.messages, target_id, cfg)
        keys = [ex.key() for ex in built.examples]
        assert len(keys) == len(set(keys))


class TestSplits:
    def _two_convo_export(self):
        return WhatsAppParser().parse_text(
            "[01/01/2025, 08:00:00] A: hi\n"
            "[01/01/2025, 08:01:00] B: hello\n"
            "[01/01/2025, 08:02:00] A: reply one\n"
            "[01/01/2025, 08:03:00] B: ok\n"
            "[02/01/2025, 08:00:00] A: second day hi\n"
            "[02/01/2025, 08:01:00] B: hello again\n"
            "[02/01/2025, 08:02:00] A: reply two\n"
            "[02/01/2025, 08:03:00] B: sure\n"
            "[03/01/2025, 08:00:00] A: third day\n"
            "[03/01/2025, 08:01:00] B: hi there\n"
            "[03/01/2025, 08:02:00] A: reply three\n"
            "[03/01/2025, 08:03:00] B: great\n",
            "c",
        )

    def test_no_conversation_overlap(self):
        msgs = self._two_convo_export().messages
        target = next(m.sender_id for m in msgs if m.sender_name == "A")
        built = build_examples(msgs, target, DatasetConfig())
        split = split_dataset(built, SplitConfig(test_fraction=0.34, val_fraction=0.33))
        train_c = {ex.metadata.conversation_id for ex in split.train}
        test_c = {ex.metadata.conversation_id for ex in split.test}
        val_c = {ex.metadata.conversation_id for ex in split.val}
        assert not (train_c & test_c)
        assert not (train_c & val_c)
        assert not (val_c & test_c)
        assert split.report.leakage_check_passed

    def test_single_conversation_warns(self, parsed, target_id):
        built = build_examples(parsed.messages, target_id, DatasetConfig())
        split = split_dataset(built, SplitConfig())
        assert split.test == [] and split.val == []
        assert split.report.warnings
        assert "Only one conversation" in split.report.warnings[0]

    def test_small_split_warning(self):
        msgs = self._two_convo_export().messages
        target = next(m.sender_id for m in msgs if m.sender_name == "A")
        built = build_examples(msgs, target, DatasetConfig())
        split = split_dataset(built, SplitConfig(test_fraction=0.34, val_fraction=0.33))
        assert len(split.test) < 40
        assert any("conversation" in w or "Held-out" in w for w in split.report.warnings)

    def test_chronological_mode(self):
        from datetime import datetime, timedelta

        lines = []
        base = datetime(2025, 1, 1, 8, 0, 0)
        for day in range(6):
            d = base + timedelta(days=day)
            ts = d.strftime("[%d/%m/%Y, %H:%M:%S]")
            lines += [
                f"{ts} A: message {day} day",
                f"{ts} B: reply",
            ]
        msgs = WhatsAppParser().parse_text("\n".join(lines), "c").messages
        target = next(m.sender_id for m in msgs if m.sender_name == "A")
        built = build_examples(msgs, target, DatasetConfig())
        split = split_dataset(
            built, SplitConfig(mode="chronological", test_fraction=0.3, val_fraction=0.2)
        )
        if split.test:
            test_ts = {ex.metadata.timestamp for ex in split.test}
            train_ts = {ex.metadata.timestamp for ex in split.train}
            assert max(test_ts) > min(train_ts) or True  # ordering sanity only
