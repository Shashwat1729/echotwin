"""Generation interfaces: a real transformers-based generator and a deterministic mock.

The mock enables full-stack testing of baselines, evaluation and the UI/API
without any model download, while the real path uses the tokenizer chat template
when available.

Chat-imitation specific handling (verified 2026-10-15):

* Qwen3-hybrid models (e.g. Qwen/Qwen3-4B, Qwen3.5) default to THINKING mode.
  Emulating chat replies requires ``chat_template_kwargs={"enable_thinking": False}``
  — without it, generations start with reasoning blocks. The generator applies
  this automatically for specs whose ``thinking_mode`` is "switchable"/"always",
  and strips any stray ``...\n</think>§281§ `` blocks from output as a safety net.
* Sampling follows the vendor-recommended recipe for the model (e.g. Qwen3
  non-thinking: temp 0.7, top_p 0.8, top_k 20) unless the caller overrides it.
"""

from __future__ import annotations

import contextlib
import random
import re
from typing import Protocol

from pydantic import BaseModel, Field

from ..exceptions import ModelNotAvailableError
from ..models.registry import ModelSpec, get_model_spec

_TT_OPEN = "<" + "think" + ">"
_TT_CLOSE = "</" + "think" + ">"
_THINK_BLOCK_RE = re.compile(
    r"^\s*" + re.escape(_TT_OPEN) + r".*?" + re.escape(_TT_CLOSE) + r"\s*", re.DOTALL
)


class GeneratorConfig(BaseModel):
    # None means "use the model spec recommendation, else the global default"
    temperature: float | None = 0.7
    top_p: float | None = 0.8
    top_k: int | None = 20
    min_p: float | None = 0.0
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0
    seed: int | None = None

    def resolve(self, spec: ModelSpec | None) -> dict:
        rec = (spec.recommended_sampling if spec else {}) or {}
        temp = self.temperature if self.temperature is not None else rec.get("temperature", 0.7)
        top_p = self.top_p if self.top_p is not None else rec.get("top_p", 0.9)
        top_k = self.top_k if self.top_k is not None else rec.get("top_k", 50)
        min_p = self.min_p if self.min_p is not None else rec.get("min_p", 0.0)
        return {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }


class TextGenerator(Protocol):
    """Interface every generator (real or mock) implements."""

    name: str

    def generate(self, messages: list[dict], cfg: GeneratorConfig) -> str: ...


class TransformersGenerator:
    """Wraps a LoadedModel with chat-imitation-safe decoding."""

    def __init__(self, loaded, spec: ModelSpec | None = None) -> None:
        self.loaded = loaded
        self.spec = spec or loaded.spec
        self.name = self.spec.hf_model_id

    @classmethod
    def from_spec(cls, spec_name: str, *, quantize: str | None = None) -> TransformersGenerator:
        from ..models.loader import load_model

        spec = get_model_spec(spec_name)
        loaded = load_model(spec, quantize=quantize)
        return cls(loaded, spec)

    def _template_kwargs(self) -> dict:
        """Chat-template kwargs required for chat-style (non-thinking) output."""
        if self.spec.thinking_mode in {"switchable", "always"}:
            # Qwen3 hybrid switch: disable thinking at template level
            return {"enable_thinking": False}
        return {}

    def _render_prompt(self, messages: list[dict]) -> str:
        tok = self.loaded.tokenizer
        kwargs = self._template_kwargs()
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except TypeError:
            # tokenizer does not accept these kwargs; try without them
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                raise ModelNotAvailableError(f"apply_chat_template failed: {e}") from e

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return _THINK_BLOCK_RE.sub("", text).strip()

    def generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        import torch

        tok = self.loaded.tokenizer
        loaded = self.loaded
        if cfg.seed is not None:
            random.seed(cfg.seed)
            with contextlib.suppress(Exception):
                torch.manual_seed(cfg.seed)

        prompt = self._render_prompt(messages)
        s = cfg.resolve(self.spec)

        def _do_generate():
            inputs = tok(prompt, return_tensors="pt")
            if loaded.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            gen_kwargs = dict(
                max_new_tokens=s["max_new_tokens"],
                do_sample=s["temperature"] > 0,
                temperature=max(s["temperature"], 1e-5),
                top_p=s["top_p"],
                repetition_penalty=s["repetition_penalty"],
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            if s.get("top_k"):
                gen_kwargs["top_k"] = int(s["top_k"])
            if s.get("min_p"):
                gen_kwargs["min_p"] = float(s["min_p"])
            with torch.no_grad():
                out = loaded.model.generate(**inputs, **gen_kwargs)
            return tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        try:
            text = _do_generate()
        except RuntimeError as e:
            raise ModelNotAvailableError(f"Generation failed: {e}") from e
        return self._strip_thinking(text)


class MockGenerator:
    """Deterministic stand-in that paraphrases a few-shot exemplar.

    Used by tests and by the offline demo so the whole evaluate/compare flow runs
    without downloading weights. Its outputs are NOT a model and never presented
    as one: callers label the method 'mock'.
    """

    def __init__(self, exemplar_texts: list[str] | None = None, seed: int = 0) -> None:
        self.exemplar_texts = [t for t in (exemplar_texts or []) if t.strip()][:8]
        self.name = "mock-echo"
        self._rng = random.Random(seed)

    def generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        prompt = messages[-1]["content"] if messages else ""
        if self.exemplar_texts:
            pick = self._rng.choice(self.exemplar_texts)
            return f"{pick} (mock reply to: {prompt[:40]})"
        return f"mock reply to: {prompt[:48]}"


def save_predictions(records: list[dict], path) -> int:
    import json
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def load_predictions(path) -> list[dict]:
    import json
    import pathlib

    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class BaselineResult(BaseModel):
    method: str
    model_revision: str
    n_prompts: int
    predictions_path: str
    config: dict = Field(default_factory=dict)
