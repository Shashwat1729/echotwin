"""Voice backend capability schema.

Capabilities describe what the *installed backend version* actually supports.
They are declared per backend version and verified against upstream docs at
implementation time; nothing here is assumed generic TTS capability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceBackendCapabilities(BaseModel):
    name: str
    version: str = ""
    installed: bool = False
    available: bool = Field(default=False, description="deps + weights present locally")
    languages: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    accents_notes: str = ""
    min_reference_seconds: float = 3.0
    max_reference_seconds: float | None = 60.0
    recommended_reference_seconds: float = 10.0
    reference_formats: list[str] = Field(default_factory=lambda: ["wav", "flac"])
    sample_rates: list[int] = Field(default_factory=lambda: [16000, 22050, 24000])
    reference_transcript_required: bool = False
    reference_transcript_optional: bool = False
    supports_multiple_references: bool = False
    supports_zero_shot_reference_conditioning: bool = True
    supports_speaker_adaptation: bool = Field(
        default=False, description="fine-tuning/speaker adaptation workflow"
    )
    speaker_adaptation_min_minutes: float | None = None
    supports_seed: bool = True
    supports_streaming: bool = False
    license: str = ""
    hardware: str = ""
    known_limitations: list[str] = Field(default_factory=list)
    reference_docs_url: str = ""

    def supports_language(self, lang: str) -> bool:
        if not self.languages:
            return False
        if lang in self.languages:
            return True
        base = lang.split("-")[0]
        return (
            any(lang_code.split("-")[0] == base for lang_code in self.languages)
            or "multilingual" in self.languages
        )

    def public_dict(self) -> dict:
        return self.model_dump()


CAPABILITY_FIELDS = list(VoiceBackendCapabilities.model_fields)
