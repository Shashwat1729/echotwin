"""N-gram engine: instant, CPU-only, trained from the current dataset's train split."""

from __future__ import annotations

from ..models.generation import GeneratorConfig
from ..models.ngram import NGramModel
from ..settings import get_settings
from .base import BaseEngine, EngineCapabilities


class NGramEngine(BaseEngine):
    kind = "ngram"
    # Only the last user message seeds generation; system text, few-shot pairs and
    # retrieved exemplars are ignored, so prompt-building methods change nothing.
    capabilities = EngineCapabilities(uses_prompt_context=False, supports_seed=True, needs_network=False)

    def __init__(self, model: NGramModel, name: str | None = None) -> None:
        self.model = model
        self.name = name or f"ngram-order{model.order}"

    def _generate(self, messages: list[dict], cfg: GeneratorConfig) -> str:
        prompt = messages[-1].get("content", "") if messages else ""
        text = self.model.generate(
            prompt,
            seed=cfg.seed,
            temperature=cfg.temperature if cfg.temperature else 0.9,
            max_words=max(8, (cfg.max_new_tokens or 128) // 4),
        )
        if not text:
            raise ValueError("n-gram model is empty; prepare a dataset with train examples first")
        return text

    @classmethod
    def for_dataset(cls, dataset_id: str) -> NGramEngine:
        """Load the cached model for a dataset, retraining if train.jsonl is newer."""
        from ..dataset.builder import load_jsonl

        settings = get_settings()
        train = settings.artifacts_dir / "datasets" / dataset_id / "train.jsonl"
        cache = settings.artifacts_dir / "models" / f"ngram-{dataset_id}.json"
        if cache.exists() and train.exists() and cache.stat().st_mtime >= train.stat().st_mtime:
            return cls(NGramModel.load(cache), name=f"ngram-{dataset_id}")
        texts = [
            ex.messages[-1].content
            for ex in (load_jsonl(train) if train.exists() else [])
            if ex.messages and ex.messages[-1].role == "assistant" and ex.messages[-1].content.strip()
        ]
        if not texts:
            raise ValueError(f"dataset {dataset_id} has no train examples")
        model = NGramModel(order=3).train_texts(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        model.save(cache)
        return cls(model, name=f"ngram-{dataset_id}")
