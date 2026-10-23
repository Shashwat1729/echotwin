"""Configurable model registry.

Entries carry licensing and suitability metadata so the CLI/UI can warn before a
user fine-tunes a model they cannot legally deploy. Compatibility is validated at
load time, not assumed here: the registry records *intended* facts and the loader
verifies chat-template/tokenizer support when it actually runs.

Model facts verified against Hugging Face model cards on 2026-10-15:

* Qwen3-4B-Instruct-2507 is a NON-THINKING-only instruct model (Apache-2.0).
* Qwen3-1.7B / Qwen3-4B are hybrid thinking models: chat imitation MUST set
  ``enable_thinking=False`` via chat template kwargs (otherwise replies start with
  reasoning blocks — the classic chat-ft failure).
* Gemma 4 E4B / Gemma 3n E4B are current Google options (Gemma Terms of Use).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    hf_model_id: str
    family: str = ""
    license: str
    params_b: float | None = None
    languages_note: str = ""
    chat_template_expected: bool = True
    multilingual_notes: str = ""
    vram_fp16_gb_approx: float | None = None
    vram_4bit_gb_approx: float | None = None
    notes: str = ""
    # thinking-mode handling (chat-imitation critical)
    thinking_mode: str = Field(
        default="none",
        description="none = non-thinking only | switchable = hybrid, must disable | always = thinks by default",
    )
    recommended_sampling: dict = Field(
        default_factory=lambda: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    )
    verified_on: str = "2026-10-15"
    status: str = Field(default="current", description="current | legacy")


REGISTRY: dict[str, ModelSpec] = {
    # --- primary recommendation: small, multilingual, non-thinking, Apache-2.0 ---
    "qwen3-4b-it-2507": ModelSpec(
        name="qwen3-4b-it-2507",
        hf_model_id="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen3",
        license="Apache-2.0",
        params_b=4.0,
        languages_note="Strong multilingual incl. Hindi (Devanagari tokenization); "
        "non-thinking only — ideal for chat-style imitation.",
        multilingual_notes="DEFAULT choice for Hindi+English personalization.",
        thinking_mode="none",
        vram_fp16_gb_approx=9.5,
        vram_4bit_gb_approx=4.0,
    ),
    "qwen3.5-4b": ModelSpec(
        name="qwen3.5-4b",
        hf_model_id="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        license="Apache-2.0",
        params_b=4.0,
        languages_note="Newest Qwen multimodal line (Feb 2026); thinks by default.",
        thinking_mode="always",
        notes="Disable thinking via chat_template_kwargs={'enable_thinking': False}.",
        vram_fp16_gb_approx=10.0,
        vram_4bit_gb_approx=4.0,
    ),
    "qwen3-1.7b": ModelSpec(
        name="qwen3-1.7b",
        hf_model_id="Qwen/Qwen3-1.7B",
        family="qwen3",
        license="Apache-2.0",
        params_b=1.7,
        languages_note="Hybrid thinking model; use the -2507 instruct or /no_think to get replies.",
        thinking_mode="switchable",
        multilingual_notes="CPU-friendly alternative.",
        vram_fp16_gb_approx=4.5,
        vram_4bit_gb_approx=2.0,
    ),
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        hf_model_id="Qwen/Qwen3-0.6B",
        family="qwen3",
        license="Apache-2.0",
        params_b=0.6,
        languages_note="Smallest Qwen3; strict smoke tests and very slow CPUs only — "
        "too small to hold base conversation quality for measured comparisons.",
        thinking_mode="switchable",
        multilingual_notes="Ultra-CPU tier; runs on any laptop, ~1.5 GB fp16.",
        vram_fp16_gb_approx=1.5,
        vram_4bit_gb_approx=0.8,
    ),
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        hf_model_id="Qwen/Qwen3-4B",
        family="qwen3",
        license="Apache-2.0",
        params_b=4.0,
        languages_note="Hybrid thinking model (256K ctx).",
        thinking_mode="switchable",
        vram_fp16_gb_approx=9.5,
        vram_4bit_gb_approx=4.0,
    ),
    # --- Google options ---
    "gemma3-1b-it": ModelSpec(
        name="gemma3-1b-it",
        hf_model_id="google/gemma-3-1b-it",
        family="gemma",
        license="Gemma Terms of Use (see docs/model-licenses.md)",
        params_b=1.0,
        languages_note="Primarily English; reasonable Hinglish tolerance, weak Devanagari generation.",
        multilingual_notes="Not recommended as primary model for Devanagari Hindi output.",
        vram_fp16_gb_approx=3.0,
        vram_4bit_gb_approx=1.5,
        status="legacy",
    ),
    "gemma3-4b-it": ModelSpec(
        name="gemma3-4b-it",
        hf_model_id="google/gemma-3-4b-it",
        family="gemma",
        license="Gemma Terms of Use (see docs/model-licenses.md)",
        params_b=4.0,
        languages_note="Bigger English skill, similar Hindi limitations to 1b.",
        vram_fp16_gb_approx=10.0,
        vram_4bit_gb_approx=4.0,
        status="legacy",
    ),
    "gemma4-e4b-it": ModelSpec(
        name="gemma4-e4b-it",
        hf_model_id="google/gemma-4-E4B-it",
        family="gemma4",
        license="Gemma Terms of Use (see docs/model-licenses.md)",
        params_b=4.0,
        languages_note="Gemma 4 (2026): 140+ languages, multimodal input, long context.",
        multilingual_notes="Strong multilingual option; requires accepting the Gemma license on HF.",
        vram_fp16_gb_approx=10.0,
        vram_4bit_gb_approx=4.0,
    ),
    "gemma3n-e4b-it": ModelSpec(
        name="gemma3n-e4b-it",
        hf_model_id="google/gemma-3n-E4B-it",
        family="gemma3n",
        license="Gemma Terms of Use (see docs/model-licenses.md)",
        params_b=4.0,
        languages_note="Effective-4B model running in ~4GB VRAM; good for laptops.",
        vram_fp16_gb_approx=4.0,
        vram_4bit_gb_approx=3.0,
    ),
    # --- other ecosystems (kept for choice) ---
    "qwen2.5-3b-it": ModelSpec(
        name="qwen2.5-3b-it",
        hf_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        license="Qwen License (see docs/model-licenses.md)",
        params_b=3.1,
        languages_note="Tokenizes Devanagari efficiently.",
        vram_fp16_gb_approx=7.5,
        vram_4bit_gb_approx=3.0,
        status="legacy",
    ),
    "llama3.2-3b-it": ModelSpec(
        name="llama3.2-3b-it",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        license="Llama 3.2 Community License (requires Meta acceptance)",
        params_b=3.2,
        languages_note="Multilingual but Devanagari weaker than Qwen at this scale.",
        multilingual_notes="Requires accepting Meta's license on Hugging Face first.",
        vram_fp16_gb_approx=7.5,
        vram_4bit_gb_approx=3.0,
    ),
}

DEFAULT_MODEL = "qwen3-4b-it-2507"


def get_model_spec(name: str) -> ModelSpec:
    key = name.lower()
    if key in REGISTRY:
        return REGISTRY[key]
    # Allow passing a raw HF id; everything else must be checked by the loader.
    if "/" in name:
        return ModelSpec(
            name=name,
            hf_model_id=name,
            license="unknown — verify model card before use",
            thinking_mode="switchable",  # safest assumption for unknown ids
        )
    raise KeyError(
        f"Unknown model '{name}'. Registered: {sorted(REGISTRY)}; or pass a raw 'org/model' id."
    )


def list_models() -> list[ModelSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def recommended_for_cpu() -> list[str]:
    return ["qwen3-0.6b", "qwen3-1.7b", "gemma3n-e4b-it"]


def recommended_for_hindi() -> list[str]:
    return ["qwen3-4b-it-2507", "qwen3.5-4b", "qwen3-4b"]
