"""Unit tests: WhatsApp parser across real export variations."""

from __future__ import annotations

import pytest

from echotwin.exceptions import ExportValidationError
from echotwin.ingestion import WhatsAppParser, parse_timestamp
from echotwin.ingestion.schema import MessageType, QualityFlag


class TestTimestamps:
    def test_dayfirst_slash(self):
        assert parse_timestamp("15/01/2025", "14:23:45", dayfirst=True) is not None
        dt = parse_timestamp("15/01/2025", "14:23:45", dayfirst=True)
        assert (dt.day, dt.month, dt.year, dt.hour) == (15, 1, 2025, 14)

    def test_monthfirst_slash(self):
        dt = parse_timestamp("01/15/2025", "14:23", dayfirst=False)
        assert (dt.month, dt.day) == (1, 15)

    def test_12h_pm(self):
        dt = parse_timestamp("16/01/2025", "09:05:12 pm", dayfirst=True)
        assert dt.hour == 21

    def test_12h_am_dotted(self):
        dt = parse_timestamp("16/01/2025", "9:05 a.m.", dayfirst=True)
        assert dt.hour == 9

    def test_dot_separator(self):
        dt = parse_timestamp("16.01.2025", "09:05:12", dayfirst=True)
        assert (dt.day, dt.month) == (16, 1)

    def test_iso(self):
        dt = parse_timestamp("2025-01-16", "09:05", dayfirst=True)
        assert (dt.year, dt.month, dt.day) == (2025, 1, 16)

    def test_unparseable_returns_none(self):
        assert parse_timestamp("not-a-date", "99:99", dayfirst=True) is None


class TestParsing:
    def test_message_count(self, parsed):
        assert len(parsed.messages) == 15

    def test_system_message_detected(self, parsed):
        system = [m for m in parsed.messages if m.is_system_message]
        assert len(system) == 1
        assert "end-to-end encrypted" in system[0].text
        assert system[0].sender_id is None

    def test_attachment_placeholder(self, parsed):
        att = [m for m in parsed.messages if m.message_type is MessageType.ATTACHMENT]
        assert len(att) == 1
        assert att[0].attachment is not None
        assert att[0].attachment.kind == "other"
        assert QualityFlag.ATTACHMENT_PLACEHOLDER in att[0].quality_flags

    def test_deleted_message(self, parsed):
        deleted = [m for m in parsed.messages if m.message_type is MessageType.DELETED]
        assert len(deleted) == 1
        assert deleted[0].sender_name == "Rohit"  # attribution preserved

    def test_12h_line(self, parsed):
        msg = parsed.messages[8]
        assert msg.timestamp is not None
        assert msg.timestamp.hour == 10  # 10:05 am
        assert "good morning!!" in msg.text

    def test_duplicate_detected(self, parsed):
        dups = [m for m in parsed.messages if QualityFlag.DUPLICATE in m.quality_flags]
        assert len(dups) == 1

    def test_original_text_preserved_exactly(self, parsed):
        text = next(m for m in parsed.messages if "ummeed" in m.text).text
        assert text == "morning! ummeed hai aaj ka din achha jayega"  # typo preserved

    def test_multiline_continuation(self, multiline_export):

        res = WhatsAppParser().parse_text(multiline_export, "c")
        assert len(res.messages) == 2
        assert "\n" in res.messages[0].text
        assert res.messages[0].text.count("\n") == 2
        assert QualityFlag.MULTILINE in res.messages[0].quality_flags
        assert res.report.continuation_lines == 2

    def test_ios_format(self, ios_export):

        res = WhatsAppParser().parse_text(ios_export, "c")
        assert len(res.messages) == 2
        assert res.messages[0].timestamp is not None

    def test_truncated_header_reported(self, truncated_export):

        res = WhatsAppParser().parse_text(truncated_export, "c")
        assert len(res.messages) == 1
        # Two broken lines: the truncated header itself, and the garbage line
        # that can no longer be attached to any message.
        assert res.report.malformed_lines == 2
        assert res.report.orphan_lines == 1
        assert any(e.reason == "truncated_or_orphan_header" for e in res.errors)
        # Garbage after a broken header must NOT be absorbed into the previous
        # participant's message.
        assert "\n" not in res.messages[0].text

    def test_empty_export_raises(self):

        with pytest.raises(ExportValidationError):
            WhatsAppParser().parse_text("no chat data here", "c")

    def test_unicode_sender_names(self):
        text = "[20/01/2025, 12:00:00] आर्यन शर्मा: नमस्ते दोस्तों!\n"

        res = WhatsAppParser().parse_text(text, "c")
        assert res.messages[0].sender_name == "आर्यन शर्मा"
        assert res.messages[0].text == "नमस्ते दोस्तों!"

    def test_stable_ids(self, parsed):
        ids = [m.message_id for m in parsed.messages]
        assert len(ids) == len(set(ids))
        assert all(len(i) == 16 for i in ids)

    def test_message_order_preserved(self, parsed):
        assert parsed.messages[0].timestamp <= parsed.messages[-1].timestamp

    def test_bidi_marks_stripped_from_headers(self):
        text = "[\u200e21/01/2025,\u200e 12:00:00] Aarav: hello"

        res = WhatsAppParser().parse_text(text, "c")
        assert res.messages[0].timestamp is not None
        assert res.messages[0].text == "hello"

    def test_emoji_preserved(self, parsed):
        assert "☀️" in parsed.messages[0].text
        assert "😅" in parsed.messages[11].text
        assert "😊" in parsed.messages[10].text
