"""VoxCPM backend adapter.

Verified against the official repository on 2026-10-15:
* PyPI: ``pip install voxcpm`` (Python >=3.10,<3.13; PyTorch >=2.5; CUDA>=12 for GPU,
  CPU/MPS supported with ``device="cpu"`` / ``"mps"``)
* Model: ``openbmb/VoxCPM2`` (2B params, Apache-2.0, 48kHz output, 16kHz reference input)
* 30 languages including hi and en. Hinglish is NOT an official language — we make
  no code-switching claim; users may test it empirically (evaluation suite reports
  what works).
* Modes: isolated reference (``reference_wav_path``) and ultimate/continuation
  cloning (``prompt_wav_path`` + ``prompt_text``); optional denoiser for reference
  enhancement.
* Seed, cfg_value, inference_timesteps supported; streaming via ``generate_streaming``.
* Speaker adaptation (SFT/LoRA) is a separate workflow via the upstream repo's
  training scripts and is NOT run by this adapter.

The adapter performs NO downloads at import. ``is_available()`` verifies the
``voxcpm`` package and cached HF weights without downloading.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..exceptions import BackendNotAvailableError, UnsupportedCapabilityError
from ..voice.base import AbstractVoiceBackend
from ..voice.capabilities import VoiceBackendCapabilities

VOXCPM_30_LANGUAGES = [
    "ar",
    "my",
    "zh",
    "da",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "el",
    "he",
    "hi",
    "id",
    "it",
    "ja",
    "km",
    "ko",
    "lo",
    "ms",
    "no",
    "pl",
    "pt",
    "ru",
    "es",
    "sw",
    "sv",
    "tl",
    "th",
    "tr",
    "vi",
]

VOXCPM_MODEL_ID = "openbmb/VoxCPM2"


class VoxCPMBackend(AbstractVoiceBackend):
    name = "voxcpm"

    def __init__(self, model_id: str = VOXCPM_MODEL_ID, device: str = "auto") -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._sample_rate = 48000

    def capabilities(self) -> dict:
        cap = VoiceBackendCapabilities(
            name=self.name,
            version="VoxCPM2 (openbmb/VoxCPM2, verified 2026-10-15)",
            installed=self._package_installed(),
            available=self.is_available(),
            languages=VOXCPM_30_LANGUAGES,
            scripts=["devanagari", "latin"],
            accents_notes="Benchmarks are strongest for en/zh; Hindi intelligibility is "
            "weaker (MiniMax MLS WER ~19.7 upstream). Accent preservation with an "
            "Indian-English speaker reference has not been benchmarked by EchoTwin.",
            min_reference_seconds=3.0,
            max_reference_seconds=120.0,
            recommended_reference_seconds=8.0,
            reference_formats=["wav", "flac"],
            sample_rates=[16000, 48000],
            reference_transcript_required=False,
            reference_transcript_optional=True,
            supports_multiple_references=False,
            supports_zero_shot_reference_conditioning=True,
            supports_speaker_adaptation=True,  # via upstream SFT/LoRA scripts
            speaker_adaptation_min_minutes=5.0,
            supports_seed=True,
            supports_streaming=True,
            license="Apache-2.0 (model + code)",
            hardware="NVIDIA GPU ~8GB VRAM recommended; CPU/MPS supported (slow)",
            known_limitations=[
                "Hinglish is not an official language; code-switching quality is untested here.",
                "Reference audio input is 16kHz; output is 48kHz.",
                "Controllable/style-guarded generation can vary between runs (upstream note).",
                "Hindi WER in the upstream multilingual benchmark is high; verify Hindi "
                "output with the evaluation suite before relying on it.",
            ],
            reference_docs_url="https://github.com/OpenBMB/VoxCPM",
        )
        return cap.public_dict()

    # -- availability ------------------------------------------------------
    def _package_installed(self) -> bool:
        try:
            import voxcpm  # noqa: F401

            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """True when the package is importable AND weights are cached locally.

        Weight detection uses HF Hub cache layout without downloading anything.
        ModelScope/local-dir users should pass a local path as ``model_id``.
        """
        if not self._package_installed():
            return False
        import os

        if Path(self.model_id).exists():  # local snapshot dir
            return True
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        safe = self.model_id.replace("/", "--")
        return (cache / f"models--{safe}").exists()

    # -- loading -----------------------------------------------------------
    def load(self, config: dict) -> None:
        super().load(config)
        if not self._package_installed():
            raise BackendNotAvailableError(
                "The 'voxcpm' package is not installed. Install it with "
                "'pip install voxcpm' (see docs/voice-cloning.md for requirements: "
                "Python>=3.10,<3.13, torch>=2.5; GPU recommended)."
            )
        from voxcpm import VoxCPM

        device = config.get("device", self.device)
        load_denoiser = bool(config.get("load_denoiser", False))
        try:
            self._model = VoxCPM.from_pretrained(
                self.model_id, load_denoiser=load_denoiser, device=device
            )
        except Exception as e:
            raise BackendNotAvailableError(
                f"VoxCPM model could not be loaded from '{self.model_id}': {e}. "
                "The first load downloads ~4GB of weights — see scripts/download_models.py."
            ) from e
        self._sample_rate = int(getattr(self._model.tts_model, "sample_rate", 48000))
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._model is None:
            self.load(self._config)

    # -- inspection --------------------------------------------------------
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
            notes.append(
                f"Reference shorter than recommended minimum ({duration:.1f}s < "
                f"{caps['min_reference_seconds']}s)."
            )
        if clip.sample_rate not in (16000, 48000) and caps["sample_rates"]:
            notes.append(
                f"Sample rate {clip.sample_rate} will be resampled to "
                f"{caps['sample_rates'][0]} for best results."
            )
        if quality.verdict == "fail":
            notes.append("Quality checks failed; re-record before using as reference.")
        return {
            "backend": self.name,
            "path": str(p),
            "duration_seconds": round(duration, 3),
            "sample_rate": clip.sample_rate,
            "channels": clip.channels,
            "transcript": transcript,
            "quality": quality.model_dump(),
            "notes": notes,
            "recommended_mode": ("ultimate" if transcript else "isolated"),
        }

    # -- synthesis ----------------------------------------------------------
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
        if not text or not text.strip():
            raise UnsupportedCapabilityError("Empty text; nothing to synthesize.")
        ref = Path(reference_audio)
        if not ref.exists():
            raise BackendNotAvailableError(f"Reference audio not found: {ref}")
        use_transcript_mode = bool(transcript and transcript.strip())
        gen_kwargs: dict = {"text": text, "reference_wav_path": str(ref)}
        if use_transcript_mode:
            gen_kwargs["prompt_wav_path"] = str(ref)
            gen_kwargs["prompt_text"] = transcript
        if seed is not None:
            gen_kwargs["seed"] = seed
        if kwargs.get("cfg_value") is not None:
            gen_kwargs["cfg_value"] = kwargs["cfg_value"]
        if kwargs.get("inference_timesteps") is not None:
            gen_kwargs["inference_timesteps"] = kwargs["inference_timesteps"]

        out = Path(output_path) if output_path else Path("echotwin_voxcpm_output.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav = self._model.generate(**gen_kwargs)
            import soundfile as sf

            sf.write(str(out), wav, self._sample_rate)
        except Exception as e:
            raise BackendNotAvailableError(
                f"VoxCPM synthesis failed: {e}. If this is an OOM, close other GPU "
                "workloads or use a machine with >=8GB VRAM; CPU mode works but is slow."
            ) from e
        return out

    def synthesize_streaming(self, text: str, reference_audio: Path, **kwargs):
        self._ensure_loaded()
        gen_kwargs: dict = {"text": text, "reference_wav_path": str(reference_audio)}
        if kwargs.get("seed") is not None:
            gen_kwargs["seed"] = kwargs["seed"]
        return self._model.generate_streaming(**gen_kwargs)

    def timing_probe(self, text: str, reference_audio: Path) -> dict:
        """Measure synthesis latency + real-time factor for the benchmark suite."""
        t0 = time.time()
        out = self.synthesize(text, reference_audio)
        latency = time.time() - t0
        from ..voice.audio_ingestion import read_audio

        clip = read_audio(out)
        duration = len(clip.samples) / clip.sample_rate if clip.sample_rate else 0.0
        return {
            "latency_seconds": round(latency, 3),
            "audio_duration_seconds": round(duration, 3),
            "rtf": round(latency / duration, 3) if duration else None,
            "output_path": str(out),
            "backend": self.name,
            "model_id": self.model_id,
        }
