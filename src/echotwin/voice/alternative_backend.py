"""Alternative backend adapter: Chatterbox Multilingual (Resemble AI).

Verified against public sources on 2026-10-15:
* PyPI: ``chatterbox-tts``; Python API ``from chatterbox.mtl_tts import
  ChatterboxMultilingualTTS``; MIT-licensed weights.
* 23 languages including hi and en; zero-shot voice cloning from a 3-10s reference;
  reference language need NOT match target language (cross-lingual transfer);
* Built-in PerTh watermarking of generated audio (provenance helpful).
* No speaker fine-tuning workflow; reference conditioning only.

Kept as a documented alternative for users who cannot run VoxCPM (e.g. no 8GB
GPU). Like all backends here, it does nothing at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import BackendNotAvailableError
from ..voice.base import AbstractVoiceBackend
from ..voice.capabilities import VoiceBackendCapabilities

CHATTERBOX_LANGUAGES = [
    "ar",
    "zh",
    "cs",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "no",
    "pl",
    "pt",
    "ru",
    "es",
    "sv",
    "tr",
    "vi",
]


class ChatterboxMultilingualBackend(AbstractVoiceBackend):
    name = "chatterbox-multilingual"

    def __init__(self, device: str = "auto") -> None:
        super().__init__()
        self.device = device
        self._model: Any = None

    def capabilities(self) -> dict:
        cap = VoiceBackendCapabilities(
            name=self.name,
            version="Chatterbox Multilingual (resemble-ai, verified 2026-10-15)",
            installed=self._package_installed(),
            available=self.is_available(),
            languages=CHATTERBOX_LANGUAGES,
            scripts=["devanagari", "latin"],
            accents_notes="Zero-shot, cross-language reference transfer; accent bleed "
            "possible without cfg tuning (upstream notes).",
            min_reference_seconds=3.0,
            max_reference_seconds=30.0,
            recommended_reference_seconds=6.0,
            reference_formats=["wav", "flac", "mp3"],
            sample_rates=[16000, 24000, 44100, 48000],
            reference_transcript_required=False,
            reference_transcript_optional=False,
            supports_multiple_references=False,
            supports_zero_shot_reference_conditioning=True,
            supports_speaker_adaptation=False,
            supports_seed=False,
            supports_streaming=False,
            license="MIT (model weights); Resemble terms prohibit impersonation misuse",
            hardware="GPU recommended (6GB+); CPU works slowly",
            known_limitations=[
                "Hinglish is not an official language; mixing is untested here.",
                "No per-speaker fine-tuning workflow.",
                "Output is watermarked (PerTh) by design — good for provenance.",
            ],
            reference_docs_url="https://github.com/resemble-ai/chatterbox",
        )
        return cap.public_dict()

    def _package_installed(self) -> bool:
        try:
            import chatterbox  # noqa: F401

            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """True when the package is importable AND its weights are cached locally.

        Mirrors the VoxCPM probe: the *instance* never holds a model until
        ``load()`` runs, so checking ``self._model`` made this permanently
        False and disabled the documented no-8GB-GPU fallback.
        """
        if not self._package_installed():
            return False
        import os
        from pathlib import Path

        cache = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
        )
        return any(cache.glob("models--ResembleAI--chatterbox*")) if cache.exists() else False

    def load(self, config: dict) -> None:
        super().load(config)
        if not self._package_installed():
            raise BackendNotAvailableError(
                "chatterbox-tts is not installed. See docs/voice-cloning.md "
                "(pip install chatterbox-tts)."
            )
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as e:
            raise BackendNotAvailableError(
                "Could not import ChatterboxMultilingualTTS; the installed "
                "chatterbox version may differ: " + str(e)
            ) from e
        device = config.get("device", self.device)
        try:
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        except Exception as e:
            raise BackendNotAvailableError(f"Chatterbox failed to load: {e}") from e
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._model is None:
            self.load(self._config)

    def inspect_reference(self, audio_path: Path, transcript: str | None = None) -> dict:
        from ..voice.audio_ingestion import read_audio
        from ..voice.audio_quality import analyze_audio_file

        p = Path(audio_path)
        clip = read_audio(p)
        quality = analyze_audio_file(p)
        duration = len(clip.samples) / clip.sample_rate if clip.sample_rate else 0.0
        caps = self.capabilities()
        notes = []
        if duration < caps["min_reference_seconds"]:
            notes.append("Reference shorter than the 3-10s guidance.")
        if quality.verdict == "fail":
            notes.append("Quality checks failed; re-record before using as reference.")
        return {
            "backend": self.name,
            "path": str(p),
            "duration_seconds": round(duration, 3),
            "quality": quality.model_dump(),
            "notes": notes,
            "recommended_mode": "isolated",
        }

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        transcript: str | None = None,
        output_path: Path | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> Path:
        self._ensure_loaded()
        if seed is not None:
            # Chatterbox Multilingual has no deterministic seeding; the
            # capability gate raises before the backend is called.
            self._require_capability("supports_seed", True)
        language_id = kwargs.get("language_id", "en")
        ref = Path(reference_audio)
        out = Path(output_path) if output_path else Path("echotwin_chatterbox_output.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        gen_kwargs = {"text": text, "audio_prompt_path": str(ref), "language_id": language_id}
        if kwargs.get("exaggeration") is not None:
            gen_kwargs["exaggeration"] = kwargs["exaggeration"]
        if kwargs.get("cfg_weight") is not None:
            gen_kwargs["cfg_weight"] = kwargs["cfg_weight"]
        try:
            wav = self._model.generate(**gen_kwargs)
            import soundfile as sf

            sf.write(str(out), wav, self._model.sr)
        except Exception as e:
            raise BackendNotAvailableError(
                f"Chatterbox synthesis failed: {e}. Verify the API signature of your "
                "installed chatterbox-tts version."
            ) from e
        return out
