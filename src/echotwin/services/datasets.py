"""Build and load the training dataset and its leakage-resistant splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..dataset.builder import DatasetConfig, TrainingExample, build_examples, load_jsonl, save_jsonl
from ..dataset.splits import SplitConfig, make_manifest, split_dataset
from ..exceptions import ArtifactNotFoundError, DatasetError
from ..settings import get_settings
from ..workspace import get_store
from . import imports


@dataclass(frozen=True)
class DatasetResult:
    dataset_id: str
    n_examples: int
    n_train: int
    n_val: int
    n_test: int
    leakage_check_passed: bool
    warnings: list[str]


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: str
    train: list[TrainingExample]
    val: list[TrainingExample]
    test: list[TrainingExample]
    all: list[TrainingExample]
    dir: Path


def prepare(
    context_messages: int = 4,
    max_context_chars: int = 1200,
    split_mode: str = "conversation",
    *,
    min_target_chars: int = 2,
    max_target_chars: int = 800,
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
) -> DatasetResult:
    store = get_store()
    author = store.pointer("current_author")
    if not author:
        raise DatasetError("Choose the author to clone before preparing a dataset.")
    messages, import_id = imports.load_messages()
    built = build_examples(
        messages,
        author,
        DatasetConfig(
            context_messages=context_messages,
            max_context_chars=max_context_chars,
            min_target_chars=min_target_chars,
            max_target_chars=max_target_chars,
        ),
    )
    split_cfg = SplitConfig(mode=split_mode, test_fraction=test_fraction, val_fraction=val_fraction)
    split = split_dataset(built, split_cfg)

    dataset_id = f"dataset-{import_id}"
    ds_dir = get_settings().artifacts_dir / "datasets" / dataset_id
    save_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_jsonl(split.train, ds_dir / "train.jsonl")
    save_jsonl(split.val, ds_dir / "val.jsonl")
    save_jsonl(split.test, ds_dir / "test.jsonl")
    manifest = make_manifest(
        dataset_id=dataset_id,
        dataset_path=ds_dir / "examples.jsonl",
        split=split,
        built=built,
        target_sender_id=author,
        split_cfg=split_cfg,
        echotwin_version=__version__,
    )
    (ds_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    store.set_pointer("current_dataset", dataset_id)
    return DatasetResult(
        dataset_id,
        len(built.examples),
        len(split.train),
        len(split.val),
        len(split.test),
        split.report.leakage_check_passed,
        list(split.report.warnings),
    )


def _dataset_dir(dataset_id: str | None) -> tuple[str, Path]:
    dataset_id = dataset_id or get_store().pointer("current_dataset")
    if not dataset_id:
        raise DatasetError("No dataset yet. Prepare a dataset first.")
    ds_dir = get_settings().artifacts_dir / "datasets" / dataset_id
    if not ds_dir.exists():
        raise ArtifactNotFoundError(f"Dataset {dataset_id} is missing on disk; prepare it again.")
    return dataset_id, ds_dir


def load(dataset_id: str | None = None) -> LoadedDataset:
    dataset_id, ds_dir = _dataset_dir(dataset_id)
    return LoadedDataset(
        dataset_id,
        load_jsonl(ds_dir / "train.jsonl"),
        load_jsonl(ds_dir / "val.jsonl"),
        load_jsonl(ds_dir / "test.jsonl"),
        load_jsonl(ds_dir / "examples.jsonl"),
        ds_dir,
    )


def manifest(dataset_id: str | None = None) -> dict:
    _, ds_dir = _dataset_dir(dataset_id)
    path = ds_dir / "manifest.json"
    if not path.exists():
        raise ArtifactNotFoundError("Dataset manifest is missing; prepare the dataset again.")
    return json.loads(path.read_text(encoding="utf-8"))


def few_shot_pairs(examples: list[TrainingExample], n: int) -> list[dict]:
    """First n examples as {'context','response'} pairs for few-shot prompts."""
    return [
        {
            "context": " | ".join(t.content for t in ex.messages[:-1]),
            "response": ex.messages[-1].content,
        }
        for ex in examples[:n]
    ]
