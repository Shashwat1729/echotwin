"""Model routes: generation, training jobs, evaluation."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from ..exceptions import DatasetError, ModelNotAvailableError
from ..models.generation import GeneratorConfig
from ..services import chat, datasets, jobs
from ..services import evaluation as evaluation_service
from .schemas import EvalRequest, GenerateRequest, GenerateResponse, TrainRequest

router = APIRouter(tags=["models"])


def _engine_id(engine: str, model: str) -> str:
    return engine or (f"hf:{model}" if model else "ngram")


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.context:
        raise DatasetError("context must contain at least one turn")
    *history, last = req.context
    session = chat.ChatSession(history=list(history))
    method = req.method or "style_fewshot"
    cfg = GeneratorConfig(
        max_new_tokens=req.max_new_tokens, seed=req.seed, temperature=req.temperature
    )
    out = chat.reply(
        session, last.get("content", ""), _engine_id(req.engine, req.model), method, cfg
    )
    if out.error:
        raise ModelNotAvailableError(out.error)
    return GenerateResponse(generated=out.text, model_revision=out.engine, method=method)


@router.post("/training/jobs")
def training_job(req: TrainRequest) -> dict:
    from ..settings import get_settings
    from ..storage.paths import ensure_within
    from ..training.sft import TrainingConfig, run_sft

    payload = req.model_dump()
    ds = datasets.load()  # raises DatasetError when nothing is prepared

    # Validate the training config *before* starting: a bad request must be a
    # 4xx, not a job that dies after creating directories.
    payload.setdefault("output_dir", str(get_settings().artifacts_dir / "experiments"))
    if not payload.get("output_dir"):
        payload["output_dir"] = str(get_settings().artifacts_dir / "experiments")
    try:
        cfg = TrainingConfig(**payload)
    except Exception as e:
        raise DatasetError(f"Invalid training configuration: {e}") from e
    # The output directory is client-supplied: keep it inside the workspace.
    out_dir = ensure_within(get_settings().home, cfg.output_dir)
    cfg = cfg.model_copy(update={"output_dir": str(out_dir)})

    def _work(ctx: jobs.JobContext) -> dict:
        ctx.progress(0.05, "Training")
        return dataclasses.asdict(run_sft(ds.train, ds.val, cfg))

    return jobs.submit("training", _work, heavy=True)


@router.post("/evaluations/baseline")
def evaluations_baseline(req: EvalRequest) -> dict:
    datasets.load()  # fail fast with 4xx instead of a failed job
    engine_id = _engine_id(req.engine, req.model)

    def _work(ctx: jobs.JobContext) -> dict:
        result = evaluation_service.run(
            req.method, engine_id, req.rag_k, req.max_prompts, progress=ctx.progress
        )
        return {
            "run_id": result.run_id,
            "predictions": str(result.predictions_path),
            "report": result.report.model_dump(),
        }

    return jobs.submit("evaluation", _work, heavy=not engine_id.startswith("ngram"))
