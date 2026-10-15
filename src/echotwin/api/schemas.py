"""Request/response schemas for the API."""

from __future__ import annotations

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
    context_messages: int = 4
    max_context_chars: int = 1200
    min_target_chars: int = 2
    max_target_chars: int = 800
    split_mode: str = "conversation"
    test_fraction: float = 0.1
    val_fraction: float = 0.1


class GenerateRequest(BaseModel):
    context: list[dict] = Field(description="chat turns [{'role','content'}]")
    engine: str = Field(default="", description="engine id; empty uses model or the n-gram engine")
    model: str = Field(default="", description="model spec name; same as engine 'hf:<model>'")
    style_profile_path: str | None = None
    method: str = "style_fewshot"
    max_new_tokens: int = 128
    seed: int | None = None
    temperature: float = 0.7


class EvalRequest(BaseModel):
    method: str = "plain"
    engine: str = "ngram"
    model: str = ""
    rag_k: int = 3
    max_prompts: int = 50


class TrainRequest(BaseModel):
    model_config = {"extra": "allow"}  # remaining fields feed training.sft.TrainingConfig


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
