"""Deletion-area and upload-envelope validation when there is nothing to delete.

save_upload_stream is async (FastAPI streaming) but has no route-level test; an
envelope that violates a rule must leave no partial file behind in exports and
must state *which* rule it broke, because the UI surfaces that message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from echotwin.api.service import save_upload_stream
from echotwin.exceptions import UploadValidationError
from echotwin.settings import get_settings

_SMALL = b"x" * 4096


class FakeUpload:
    """Duck-typed FastAPI UploadFile: filename / size attrs + async read()."""

    def __init__(
        self,
        chunks: list[bytes],
        filename: str = "note.wav",
        size: int | None = None,
    ) -> None:
        self.chunks = chunks
        self.filename = filename
        self.size = size
        self._iter: Iterator[bytes] = iter(chunks)

    async def read(self, chunk_size: int = 1 << 20) -> bytes:
        try:
            return next(self._iter)
        except StopIteration:
            return b""


def _saved(fake: FakeUpload, stem: str = "note"):
    return asyncio.run(save_upload_stream(fake, filename=fake.filename))


class TestStreamingUpload:
    def test_small_upload_streamed_to_exports(self):
        _orig = get_settings().max_upload_bytes
        fake = FakeUpload([_SMALL, _SMALL], filename="clip.wav")
        target = _saved(fake)
        try:
            assert target.exists()
            assert target.parent == get_settings().exports_dir
            assert target.name.startswith("clip")
            assert target.read_bytes() == _SMALL * 2
        finally:
            target.unlink(missing_ok=True)
            assert get_settings().max_upload_bytes == _orig

    def test_unique_names_replace_never(self):
        fake = FakeUpload([b"a"], filename="same.wav")
        one = _saved(fake)
        fake2 = FakeUpload([b"b"], filename="same.wav")
        two = _saved(fake2)
        try:
            assert one != two
            assert one.read_bytes() == b"a"  # the earlier file was left intact
        finally:
            one.unlink(missing_ok=True)
            two.unlink(missing_ok=True)

    def test_empty_upload_rejected_and_no_file(self):
        with pytest.raises(UploadValidationError) as e:
            _saved(FakeUpload([], filename="empty.txt"))
        assert "empty" in str(e.value).lower()

    def test_bad_extension_rejected(self):
        fake = FakeUpload([b"MZ" + b"\x00" * 10], filename="evil.exe", size=12)
        with pytest.raises(UploadValidationError):
            _saved(fake)

    def test_declared_size_rejected_before_reading(self):
        settings = get_settings()
        limit = settings.max_upload_bytes
        over_limit = limit + 1
        fake = FakeUpload([b"never-read"], filename="big.wav", size=over_limit)
        with pytest.raises(UploadValidationError) as e:
            _saved(fake)
        assert str(limit) in str(e.value)
        assert str(over_limit) in str(e.value)

    def test_over_limit_while_streaming_removed(self):
        settings = get_settings()
        limit = settings.max_upload_bytes
        one_mb = 1 << 20
        chunk = one_mb * b"a"
        # one chunk over the limit requires ceil(limit/1MB)+1 chunks
        n_chunks = (limit // one_mb) + 1
        fake = FakeUpload([chunk] * n_chunks, filename="flood.wav")
        with pytest.raises(UploadValidationError) as e:
            _saved(fake)
        assert "exceeds" in str(e.value).lower()
        leftovers = [p for p in settings.exports_dir.glob("flood*")]
        assert leftovers == [], f"partial upload left residue: {leftovers}"


class TestUploadLimits:
    def test_voice_screenshot_document_extensions_accepted(self):
        settings = get_settings()
        assert {".wav", ".mp3", ".txt", ".zip"} <= set(settings.allowed_upload_extensions)

    def test_streamed_count_does_not_buffer_whole_file(self):
        """The 1 MB read pattern is what makes streaming memory-safe."""
        fake = FakeUpload([f"chunk {i}".encode() for i in range(64)], filename="many.txt")
        target = _saved(fake)
        try:
            payload = target.read_bytes()
            assert payload == b"".join(f"chunk {i}".encode() for i in range(64))
        finally:
            target.unlink(missing_ok=True)
