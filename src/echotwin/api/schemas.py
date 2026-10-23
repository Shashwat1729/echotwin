"""Request/response schemas for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    path: str = Field(description="Server-side path OR uploaded file saved via /imports/upload")


class ImportResponse(BaseModel):
    import_id: str
    messages: int
    formats: list[str] = Field(default_factory=list)
    participants: list[dict] = Field(default_factory=list)


class ParticipantResponse(BaseModel):
    sender_id: str
    sender_name: str
    message_count: int
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    mean_length: float


class TargetAuthorRequest(BaseModel):
    name_or_id: str


class DatasetPrepareRequest(BaseModel):
    """Bounds are enforced here so bad input returns 422, never a 500."""

    context_messages: int = Field(default=4, ge=0, le=20)
    max_context_chars: int = Field(default=1200, ge=100, le=20000)
    min_target_chars: int = Field(default=2, ge=1, le=1000)
    max_target_chars: int = Field(default=800, ge=10, le=20000)
    split_mode: Literal["conversation", "chronological"] = "conversation"
    test_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    val_fraction: float = Field(default=0.1, ge=0.0, le=0.5)


class GenerateRequest(BaseModel):
    context: list[dict] = Field(description="chat turns [{'role','content'}]")
    engine: str = Field(default="", description="engine id; empty uses model or the n-gram engine")
    model: str = Field(default="", description="model spec name; same as engine 'hf:<model>'")
    method: str = "style_fewshot"
    max_new_tokens: int = 128
    seed: int | None = None
    temperature: float = 0.7


class EvalRequest(BaseModel):
    method: str = "plain"
    engine: str = "ngram"
    model: str = ""
    rag_k: int = Field(default=3, ge=0, le=10)
    max_prompts: int = Field(default=50, ge=1, le=500)


class TrainRequest(BaseModel):
    """Training job parameters.

    Unknown fields are forwarded to ``training.sft.TrainingConfig``; the config
    itself is validated *before* the job starts so a bad request is a 422, not
    a job that fails halfway through.
    """

    model_config = {"extra": "allow"}  # remaining fields feed training.sft.TrainingConfig

    method: Literal["full", "lora", "qlora"] = "lora"
    epochs: float = Field(default=3.0, gt=0, le=50)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=256)
    seq_len: int = Field(default=1024, ge=32, le=32768)
    output_dir: str | None = Field(
        default=None, description="Defaults to ECHOTWIN_HOME/artifacts/experiments"
    )


class GenerateResponse(BaseModel):
    generated: str
    model_revision: str
    method: str


class VoicePlanRequest(BaseModel):
    backend: str = "voxcpm"
    languages: list[str] = Field(default_factory=lambda: ["hi", "en"])
    include_hinglish: bool = True
    max_minutes: float = 20.0
    profile_id: str | None = None


class ConsentRequest(BaseModel):
    scope: str
    granted: bool
    details: dict = Field(default_factory=dict)


class SynthesizeRequest(BaseModel):
    profile_id: str
    text: str
    backend: str | None = None
    seed: int | None = None
    language: str = "auto"
    settings: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    hint: str | None = None
