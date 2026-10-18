"""Leakage-resistant dataset splits.

Rules enforced:

* Splits happen by **conversation** (or chronologically), never by scattering
  adjacent messages randomly — random splits leak style verbatim.
* A hard assertion checks that no conversation_id appears in more than one split.
* The evaluation (test) split is frozen once written and reported as insufficient
  when it is too small to support conclusions.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ..dataset.builder import BuiltDataset, TrainingExample

MIN_TEST_EXAMPLES_WARN = 40
MIN_VAL_EXAMPLES_WARN = 20


class SplitConfig(BaseModel):
    mode: str = Field(default="conversation", description="conversation | chronological")
    test_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    val_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    seed: int = 20250917
    min_test_examples_warn: int = MIN_TEST_EXAMPLES_WARN


class SplitReport(BaseModel):
    mode: str
    n_train: int
    n_val: int
    n_test: int
    train_conversations: int
    val_conversations: int
    test_conversations: int
    leakage_check_passed: bool
    warnings: list[str] = Field(default_factory=list)
    frozen: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DatasetSplit(BaseModel):
    train: list[TrainingExample]
    val: list[TrainingExample]
    test: list[TrainingExample]
    report: SplitReport


def _conversation_keys(examples: list[TrainingExample]) -> dict[str, list[TrainingExample]]:
    by_convo: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_convo.setdefault(ex.metadata.conversation_id, []).append(ex)
    return by_convo


def split_dataset(built: BuiltDataset, cfg: SplitConfig | None = None) -> DatasetSplit:
    cfg = cfg or SplitConfig()
    examples = built.examples
    warnings: list[str] = []
    if cfg.mode not in {"conversation", "chronological"}:
        raise ValueError(f"Unknown split mode: {cfg.mode}")

    by_convo = _conversation_keys(examples)
    convo_ids = sorted(by_convo.keys())
    if len(convo_ids) < 2:
        warnings.append(
            "Only one conversation available; a held-out split would be empty. "
            "Import more conversations before trusting evaluation numbers."
        )
        return DatasetSplit(
            train=examples,
            val=[],
            test=[],
            report=SplitReport(
                mode=cfg.mode,
                n_train=len(examples),
                n_val=0,
                n_test=0,
                train_conversations=1,
                val_conversations=0,
                test_conversations=0,
                leakage_check_passed=True,
                warnings=warnings,
            ),
        )

    def _counts(n_total: int, test_fraction: float, val_fraction: float) -> tuple[int, int]:
        """Split sizes honouring an explicit zero fraction.

        ``max(1, int(n * f))`` turned ``f=0`` into 1, so a caller asking for no
        holdout silently lost 2 conversations. A zero fraction now really means
        zero; if both holdouts are requested but would consume everything, one
        conversation is carved out for training instead.
        """
        n_test = int(n_total * test_fraction)
        n_val = int(n_total * val_fraction)
        # A requested-but-rounded-to-zero fraction (e.g. 10% of 4) still gets one
        # holdout conversation while training can spare it. An EXPLICIT 0.0 is
        # honoured as zero.
        if n_total >= 3 and test_fraction > 0 and n_test == 0:
            n_test = 1
        if n_total >= 4 and val_fraction > 0 and n_val == 0:
            n_val = 1
        if n_test + n_val >= n_total:
            if n_test == 0 and n_val == 0:
                return 0, 0
            n_val = 0
            n_test = max(1, n_total // 4)
        return n_test, n_val

    if cfg.mode == "conversation":
        rng = random.Random(cfg.seed)
        shuffled = convo_ids[:]
        rng.shuffle(shuffled)
        n_test, n_val = _counts(len(shuffled), cfg.test_fraction, cfg.val_fraction)
        test_ids = set(shuffled[:n_test])
        val_ids = set(shuffled[n_test : n_test + n_val])
        train_ids = [c for c in shuffled[n_test + n_val :]]
    else:  # chronological: newest conversations are held out
        dated = {
            c: min((ex.metadata.timestamp or "" for ex in by_convo[c]), default="")
            for c in convo_ids
        }
        if all(not v for v in dated.values()):
            # Every example lacks a timestamp, so a chronological order is not
            # derivable and sorting would silently fall back to alphabetical.
            warnings.append(
                "Chronological split requested but no example has a timestamp; "
                "the ordering would be arbitrary. Import exports with intact "
                "timestamps to enable temporal leakage protection."
            )
        ordered = sorted(convo_ids, key=lambda c: (dated[c] == "", dated[c], c))
        n_test, n_val = _counts(len(ordered), cfg.test_fraction, cfg.val_fraction)
        test_ids = set(ordered[-n_test:]) if n_test else set()
        val_ids = set(ordered[-n_test - n_val : -n_test]) if n_val else set()
        train_ids = [c for c in ordered[: len(ordered) - n_test - n_val]]

    def pick(ids: set[str] | list[str]) -> list[TrainingExample]:
        return [ex for c in ids for ex in by_convo[c]]

    train, val, test = pick(train_ids), pick(val_ids), pick(test_ids)

    # Hard leakage check.
    train_convos = {ex.metadata.conversation_id for ex in train}
    val_convos = {ex.metadata.conversation_id for ex in val}
    test_convos = {ex.metadata.conversation_id for ex in test}
    passed = (
        not (train_convos & test_convos)
        and not (train_convos & val_convos)
        and not (val_convos & test_convos)
    )
    if not passed:
        raise ValueError(
            "Leakage guard triggered: conversation overlap between splits "
            f"train/test={train_convos & test_convos} train/val={train_convos & val_convos}"
        )

    if len(test) < cfg.min_test_examples_warn:
        warnings.append(
            f"Held-out split has only {len(test)} examples (< {cfg.min_test_examples_warn}). "
            "Treat evaluation metrics as indicative only; collect/import more data."
        )
    if len(val) < MIN_VAL_EXAMPLES_WARN:
        warnings.append(
            f"Validation split has only {len(val)} examples; early stopping and "
            "best-checkpoint selection will be noisy."
        )

    return DatasetSplit(
        train=train,
        val=val,
        test=test,
        report=SplitReport(
            mode=cfg.mode,
            n_train=len(train),
            n_val=len(val),
            n_test=len(test),
            train_conversations=len(train_convos),
            val_conversations=len(val_convos),
            test_conversations=len(test_convos),
            leakage_check_passed=passed,
            warnings=warnings,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset manifest (reproducible experiments)
# ---------------------------------------------------------------------------
class DatasetManifest(BaseModel):
    dataset_id: str
    schema_version: str = "1.0"
    created_at: str
    target_sender_id: str
    conversation_ids: list[str]
    counts: dict = Field(default_factory=dict, description="Dataset statistics (heterogeneous)")
    splits: dict[str, int]
    dataset_sha256: str
    split_sha256: str
    config_sha256: str
    dataset_config: dict
    split_config: dict
    warnings: list[str] = Field(default_factory=list)
    echotwin_version: str = ""


def _sha256_file(path) -> str:
    import hashlib
    from pathlib import Path

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest(
    *,
    dataset_id: str,
    dataset_path,
    split: DatasetSplit,
    built: BuiltDataset,
    target_sender_id: str,
    split_cfg: SplitConfig,
    echotwin_version: str,
) -> DatasetManifest:
    config_sha = hashlib.sha256(
        json.dumps(built.config.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    split_cfg_sha = hashlib.sha256(
        json.dumps(split_cfg.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    return DatasetManifest(
        dataset_id=dataset_id,
        created_at=datetime.now(UTC).isoformat(),
        target_sender_id=target_sender_id,
        conversation_ids=sorted({ex.metadata.conversation_id for ex in built.examples}),
        counts=built.stats,
        splits={
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        dataset_sha256=_sha256_file(dataset_path),
        split_sha256=split_cfg_sha,
        config_sha256=config_sha,
        dataset_config=built.config.model_dump(),
        split_config=split_cfg.model_dump(),
        warnings=split.report.warnings,
        echotwin_version=echotwin_version,
    )
