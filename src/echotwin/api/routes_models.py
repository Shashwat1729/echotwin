"""Model routes: generation, training jobs, evaluation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..exceptions import EchoTwinError
from ..storage import state
from . import service
from .schemas import GenerateRequest, GenerateResponse

router = APIRouter(tags=["models"])


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    from ..models.generation import (
        GeneratorConfig,
        MockGenerator,
        TextGenerator,
        TransformersGenerator,
    )

    cfg = GeneratorConfig(
        max_new_tokens=req.max_new_tokens, seed=req.seed, temperature=req.temperature
    )
    generator: TextGenerator
    if req.model:
        try:
            generator = TransformersGenerator.from_spec(req.model)
        except EchoTwinError as e:
            raise HTTPException(400, str(e)) from e
        from ..models.registry import get_model_spec

        revision = get_model_spec(req.model).hf_model_id
    else:
        ds_id = state.get("last_dataset_id")
        exemplars: list[str] = []
        if ds_id:
            from ..dataset.builder import load_jsonl
            from ..settings import get_settings

            path = get_settings().artifacts_dir / "datasets" / ds_id / "train.jsonl"
            if path.exists():
                exemplars = [ex.messages[-1].content for ex in load_jsonl(path)[:8]]
        generator = MockGenerator(exemplars, seed=1)
        revision = "mock-1"
    fragment = None
    if req.style_profile_path:
        from ..profiling.style_profile import load_profile, profile_prompt_fragment

        fragment = profile_prompt_fragment(load_profile(req.style_profile_path))
    method = req.method or "direct"
    few_shot_turns: list[dict] = []
    ds_id_for_exemplars = state.get("last_dataset_id")
    if method in {"few_shot", "style_fewshot"} and ds_id_for_exemplars:
        from ..dataset.builder import load_jsonl
        from ..settings import get_settings

        train_path = get_settings().artifacts_dir / "datasets" / ds_id_for_exemplars / "train.jsonl"
        if train_path.exists():
            for ex in load_jsonl(train_path)[:3]:
                few_shot_turns.append(
                    {"role": "user", "content": " | ".join(t.content for t in ex.messages[:-1])}
                )
                few_shot_turns.append({"role": "assistant", "content": ex.messages[-1].content})
    elif method in {"style_rag", "style_rag_ft"} and ds_id_for_exemplars:
        from ..dataset.builder import load_jsonl
        from ..dataset.retrieval import (
            RetrievalConfig,
            build_index_from_examples,
            retrieve_exemplars,
        )
        from ..settings import get_settings

        examples_path = (
            get_settings().artifacts_dir / "datasets" / ds_id_for_exemplars / "examples.jsonl"
        )
        if examples_path.exists():
            index = build_index_from_examples(load_jsonl(examples_path))
            for hit in retrieve_exemplars(index, list(req.context), cfg=RetrievalConfig(k=3)):
                few_shot_turns.append({"role": "user", "content": hit["context"]})
                few_shot_turns.append({"role": "assistant", "content": hit["response"]})
    messages = list(req.context)
    if fragment:
        messages = [{"role": "system", "content": fragment}, *few_shot_turns, *messages]
    else:
        messages = [*few_shot_turns, *messages]
    out = generator.generate(messages, cfg)
    return GenerateResponse(generated=out, model_revision=revision, method=method)


@router.post("/training/jobs")
def training_job(payload: dict) -> dict:
    from ..dataset.builder import load_jsonl
    from ..settings import get_settings
    from ..training.sft import TrainingConfig, run_sft

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        raise HTTPException(400, "prepare a dataset first")

    def _work() -> dict:
        import dataclasses

        cfg = TrainingConfig(**payload)
        ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
        train_ex = load_jsonl(ds_dir / "train.jsonl")
        val_ex = load_jsonl(ds_dir / "val.jsonl")
        record = run_sft(train_ex, val_ex, cfg)
        return dataclasses.asdict(record)

    job = service.JOBS.submit("training", _work)
    return job.model_dump()


@router.post("/evaluations/baseline")
def evaluations_baseline(payload: dict) -> dict:
    from ..dataset.builder import load_jsonl
    from ..dataset.retrieval import RetrievalConfig, build_index_from_examples
    from ..evaluation.baseline import RETRIEVAL_METHODS, heldout_prompts, run_baseline
    from ..models.generation import (
        GeneratorConfig,
        MockGenerator,
        TextGenerator,
        TransformersGenerator,
    )
    from ..models.registry import get_model_spec
    from ..settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        raise HTTPException(400, "prepare a dataset first")

    def _work() -> dict:
        ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
        test = load_jsonl(ds_dir / "test.jsonl")
        train = load_jsonl(ds_dir / "train.jsonl")
        all_examples = load_jsonl(ds_dir / "examples.jsonl")
        prompts = heldout_prompts(test, max_prompts=int(payload.get("max_prompts", 50)))
        if not prompts:
            raise ValueError("No held-out prompts available; the split is too small.")
        method = payload.get("method", "mock")
        fragment = None
        prof_path = payload.get("style_profile_path") or state.get("last_profile_id")
        if prof_path:
            from pathlib import Path as _Path

            from ..profiling.style_profile import load_profile, profile_prompt_fragment

            if _Path(prof_path).exists():
                fragment = profile_prompt_fragment(load_profile(prof_path))
        few_shot_n = int(payload.get("few_shot", 5))
        rag_k = int(payload.get("rag_k", 3))
        model_name = payload.get("model", "")
        generator: TextGenerator
        if model_name:
            try:
                generator = TransformersGenerator.from_spec(model_name)
            except EchoTwinError as e:
                raise ValueError(str(e)) from e
            revision = get_model_spec(model_name).hf_model_id
        else:
            exemplars = [
                {
                    "context": " | ".join(t.content for t in ex.messages[:-1]),
                    "response": ex.messages[-1].content,
                }
                for ex in train[: max(few_shot_n * 4, few_shot_n)]
            ]
            generator = MockGenerator(exemplar_texts=[e["response"] for e in exemplars], seed=7)
            revision = "mock-1"
        index = build_index_from_examples(all_examples) if method in RETRIEVAL_METHODS else None
        gen_cfg = GeneratorConfig(
            max_new_tokens=int(payload.get("max_new_tokens", 128)),
            seed=payload.get("seed"),
            temperature=float(payload.get("temperature", 0.7)),
        )
        _records, path, report = run_baseline(
            method=method,
            generator=generator,
            prompts=prompts,
            profile_fragment=fragment,
            few_shot=[
                {
                    "context": " | ".join(t.content for t in ex.messages[:-1]),
                    "response": ex.messages[-1].content,
                }
                for ex in train[:few_shot_n]
            ],
            output_dir=get_settings().artifacts_dir / "evaluations" / ds_id,
            model_revision=revision,
            gen_cfg=gen_cfg,
            retrieval_index=index,
            retrieval_cfg=RetrievalConfig(k=rag_k),
            train_texts=[ex.messages[-1].content for ex in train],
        )
        return {"predictions": str(path), "report": report.model_dump()}

    job = service.JOBS.submit("evaluation", _work)
    return job.model_dump()
