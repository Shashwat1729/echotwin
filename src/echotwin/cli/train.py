"""Training and preference commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..preference.dpo import PreferencePair, load_pairs, validate_pairs
from ..storage import db, state
from ._common import (
    _fail,
    console,
    prefs_app,
    train_app,
)


@train_app.command("ngram")
def train_ngram(
    order: int = typer.Option(3, help="n-gram order (1-5)"),
    smoothing: float = typer.Option(0.1, help="add-k smoothing"),
) -> None:
    """Train a CPU-only word n-gram style model on the current train split."""
    from ..dataset.builder import load_jsonl
    from ..models.ngram import NGramModel
    from ..settings import get_settings

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    train_ex = load_jsonl(ds_dir / "train.jsonl")
    if not train_ex:
        _fail("Train split is empty; import multiple chats first.")
    model = NGramModel(order=order, smoothing_k=smoothing).train_texts(
        [ex.messages[-1].content for ex in train_ex]
    )
    out = get_settings().artifacts_dir / "models" / f"ngram-{ds_id}-o{order}.json"
    model.save(out)
    db.record_artifact(
        "ngram_model", out, {"dataset": ds_id, "order": order, "targets": model.n_targets}
    )
    console.print(
        f"[green]n-gram trained[/green]: order={order} targets={model.n_targets} "
        f"vocab={len(model.vocab)} -> {out}"
    )


@train_app.command("lora")
def train_lora(
    config: Path = typer.Option(..., "--config", help="YAML training config"),
) -> None:
    """Train with SFT/LoRA/QLoRA (needs the [train] extra and model download)."""
    import yaml

    from ..dataset.builder import load_jsonl
    from ..settings import get_settings
    from ..training.sft import TrainingConfig, run_sft

    cfg_dict = yaml.safe_load(config.read_text(encoding="utf-8"))
    train_cfg = TrainingConfig(**cfg_dict)
    ds_id = state.get("last_dataset_id")
    if not ds_id:
        _fail("No dataset prepared yet.")
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    train_ex = load_jsonl(ds_dir / "train.jsonl")
    val_ex = load_jsonl(ds_dir / "val.jsonl")
    console.print(
        f"Starting {train_cfg.method} training on {len(train_ex)} examples "
        f"(val={len(val_ex)}); this needs GPU/model downloads."
    )
    record = run_sft(train_ex, val_ex, train_cfg)
    console.print(f"[green]Training finished[/green]: {record.status} -> {record.output_dir}")


@train_app.command("dpo")
def train_dpo(
    config: Path = typer.Option(..., "--config"),
    pairs: Path = typer.Option(..., "--pairs", help="preference pairs JSONL"),
    allow_demo: bool = typer.Option(False, help="allow 'synthetic-demo' pairs (dry run only)"),
) -> None:
    """Preference optimization (DPO) with TRL when available."""
    import yaml

    from ..preference.dpo import train_dpo as _train_dpo

    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    pair_list = load_pairs(pairs)
    v = validate_pairs(pair_list)
    if not v.valid:
        _fail(f"Invalid preference pairs: {v.errors[:3]}")
    if v.human_share < 1.0 and not allow_demo:
        _fail("Pairs are not human-labeled. Re-run with --allow-demo for a dry run.")
    console.print(f"Running DPO on {len(pair_list)} pairs (provenances: {v.provenances}).")
    result = _train_dpo(
        model_id=cfg.get("model_id", "Qwen/Qwen2.5-3B-Instruct"),
        pairs=pair_list,
        output_dir=cfg.get("output_dir", "artifacts/experiments/dpo"),
        beta=cfg.get("beta", 0.1),
        epochs=cfg.get("epochs", 1),
        learning_rate=cfg.get("learning_rate", 5e-5),
        allow_demo=allow_demo,
    )
    console.print(json.dumps(result, indent=2, default=str))


@prefs_app.command("annotate")
def prefs_annotate() -> None:
    """Create/review preference pairs interactively (TTY) — placeholder guidance."""
    console.print(
        "Preference annotation is available in the Web UI (Evaluation Studio) and API. "
        "This CLI command prints a draft scaffold; edit 'examples/synthetic_preference_pairs.jsonl' "
        "or use the UI. Synthetic-demo pairs are blocked from DPO unless --allow-demo."
    )
    console.print(json.dumps(PreferencePair.model_json_schema()["properties"], indent=1)[:1500])


@prefs_app.command("validate")
def prefs_validate(pairs: Path = typer.Argument(...)) -> None:
    """Validate a preference-pairs file (missing file = error, not success)."""

    if not pairs.exists():
        _fail(f"Preference pairs file not found: {pairs}")
    pair_list = load_pairs(pairs)
    if not pair_list:
        _fail(f"No preference pairs found in {pairs} (file is empty?)")
    v = validate_pairs(pair_list)
    console.print(json.dumps(v.model_dump(), indent=1))
    if not v.valid:
        raise typer.Exit(1)
