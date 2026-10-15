"""Run a baseline method on the frozen held-out split with any engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..dataset.retrieval import RetrievalConfig, build_index_from_examples
from ..engines import registry
from ..evaluation.baseline import RETRIEVAL_METHODS, heldout_prompts, run_baseline
from ..exceptions import EvaluationError
from ..settings import get_settings
from . import datasets, profiles

Progress = Callable[[float, str], None]
FEW_SHOT_N = 5


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    predictions_path: Path
    report: object


def run(
    method: str,
    engine_id: str,
    rag_k: int = 3,
    max_prompts: int = 50,
    progress: Progress | None = None,
) -> EvalResult:
    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    ds = datasets.load()
    if not ds.test:
        raise EvaluationError(
            "The held-out test split is empty. Import at least two conversations so one can be held out."
        )
    if not ds.train:
        raise EvaluationError("The train split is empty; import more conversations first.")
    prompts = heldout_prompts(ds.test, max_prompts=max_prompts)
    if not prompts:
        raise EvaluationError("No held-out prompts available; the split is too small.")

    tick(0.05, "Loading engine")
    engine = registry.get_engine(engine_id, dataset_id=ds.dataset_id)
    index = build_index_from_examples(ds.all) if method in RETRIEVAL_METHODS else None

    tick(0.2, f"Generating {len(prompts)} replies")
    _, path, report = run_baseline(
        method=method,
        generator=engine,
        prompts=prompts,
        profile_fragment=profiles.current_fragment(),
        few_shot=datasets.few_shot_pairs(ds.train, FEW_SHOT_N),
        output_dir=get_settings().artifacts_dir / "evaluations" / ds.dataset_id,
        model_revision=engine.name,
        retrieval_index=index,
        retrieval_cfg=RetrievalConfig(k=rag_k),
        train_texts=[ex.messages[-1].content for ex in ds.train],
    )
    tick(1.0, "Done")
    return EvalResult(path.stem, path, report)
