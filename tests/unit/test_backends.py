"""Unit tests: backend registry and capability honesty."""

from __future__ import annotations

import pytest

from echotwin.exceptions import BackendNotAvailableError, UnsupportedCapabilityError
from echotwin.voice.base import AbstractVoiceBackend
from echotwin.voice.capabilities import VoiceBackendCapabilities
from echotwin.voice.registry import (
    all_backend_statuses,
    get_backend,
    registered_backends,
)


class TestRegistry:
    def test_registered_backends(self):
        names = registered_backends()
        assert "voxcpm" in names
        assert "chatterbox-multilingual" in names

    def test_unknown_backend_raises(self):
        with pytest.raises(BackendNotAvailableError):
            get_backend("no-such-backend")

    def test_statuses_no_download(self):
        statuses = all_backend_statuses()
        assert statuses
        for s in statuses:
            assert "name" in s


class TestVoxCPMCapabilities:
    def test_languages_include_hi_en(self):
        caps = get_backend("voxcpm").capabilities()
        assert "hi" in caps["languages"] and "en" in caps["languages"]
        # Hinglish must NOT be claimed as a supported language
        assert "hinglish" not in caps["languages"]

    def test_transcript_optional_not_required(self):
        caps = get_backend("voxcpm").capabilities()
        assert caps["reference_transcript_optional"] is True
        assert caps["reference_transcript_required"] is False

    def test_multiple_references_not_claimed(self):
        caps = get_backend("voxcpm").capabilities()
        assert caps["supports_multiple_references"] is False

    def test_license_documented(self):
        caps = get_backend("voxcpm").capabilities()
        assert "Apache-2.0" in caps["license"]

    def test_adaptation_is_separate_workflow(self):
        # EchoTwin does not implement speaker adaptation, so it must not
        # advertise the capability; the upstream route is documented instead.
        caps = get_backend("voxcpm").capabilities()
        assert caps["supports_speaker_adaptation"] is False
        assert caps["speaker_adaptation_min_minutes"] is None
        assert any(
            "SFT/LoRA" in lim or "upstream" in lim.lower()
            for lim in caps["known_limitations"]
        ), caps["known_limitations"]


class TestChatterboxCapabilities:
    def test_languages(self):
        caps = get_backend("chatterbox").capabilities()
        assert "hi" in caps["languages"]
        assert "MIT" in caps["license"]

    def test_no_speaker_adaptation(self):
        caps = get_backend("chatterbox").capabilities()
        assert caps["supports_speaker_adaptation"] is False


class TestCapabilityGating:
    def test_capability_check_raises(self):
        class Dummy(AbstractVoiceBackend):
            name = "dummy"

            def capabilities(self):
                return VoiceBackendCapabilities(name="dummy", supports_seed=False).public_dict()

            def inspect_reference(self, audio_path, transcript=None):
                return {}

            def synthesize(
                self, text, reference_audio, transcript=None, output_path=None, seed=None, **kw
            ):
                return None

        d = Dummy()
        with pytest.raises(UnsupportedCapabilityError):
            d._require_capability("supports_seed", True)

    def test_capability_check_passes(self):
        class Dummy2(AbstractVoiceBackend):
            name = "dummy2"

            def capabilities(self):
                return VoiceBackendCapabilities(
                    name="dummy2", supports_zero_shot_reference_conditioning=True
                ).public_dict()

            def inspect_reference(self, audio_path, transcript=None):
                return {}

            def synthesize(
                self, text, reference_audio, transcript=None, output_path=None, seed=None, **kw
            ):
                return None

        d = Dummy2()
        assert d._require_capability("supports_zero_shot_reference_conditioning", True) is True
