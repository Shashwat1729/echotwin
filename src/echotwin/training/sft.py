"""SFT with LoRA/QLoRA. Real training via transformers Trainer + PEFT.

Design decisions (documented trade-offs):

* We use the transformers ``Trainer`` directly rather than depending on TRL's
  ``SFTTrainer``: TRL's API shifts between releases, and we need exact control
  over target masking (verified prefix + response-span fallback, non-thinking
  template consistency). Every masking path is unit-tested offline.
* Target masking: prompt tokens get label -100 so the loss is only over the
  assistant response (correct SFT behavior).
* QLoRA requires CUDA + bitsandbytes. If unavailable we raise
  ``UnsupportedCapabilityError`` — we never silently substitute another method.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..dataset.builder import TrainingExample
from ..exceptions import TrainingError
from ..models.loader import apply_lora, hardware_report, load_model
from ..models.registry import get_model_spec


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            ".*q_proj",
            ".*k_proj",
            ".*v_proj",
            ".*o_proj",
            ".*gate_proj",
            ".*up_proj",
            ".*down_proj",
        ]
    )
    bias: str = "none"


class TrainingConfig(BaseModel):
    model: str = "qwen3-4b-it-2507"
    method: str = Field(default="lora", description="full | lora | qlora")
    output_dir: str
    epochs: float = 3.0
    learning_rate: float = 1e-4
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seq_len: int = 1024
    train_on_last_only: bool = Field(
        default=True,
        description="Mask everything except the final assistant response (False trains all assistant turns)",
    )
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    logging_steps: int = 5
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 2
    seed: int = 20250917
    gradient_checkpointing: bool = False
    early_stopping_patience: int | None = Field(
        default=3, description="Personal datasets are small: early stopping is ON by default"
    )
    resume_from_checkpoint: str | None = None
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"
    run_name: str | None = None
    max_steps_override: int | None = Field(
        default=None, description="Set by smoke tests to bound runtime"
    )


@dataclass
class TrainingRunRecord:
    run_id: str
    output_dir: str
    config: dict
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    hardware: dict = field(default_factory=dict)


def _render_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    """Render the chat template in non-thinking mode when the tokenizer supports it.

    Hybrid thinking models must see the same non-thinking format at train time
    that generation uses; tokenizers without the switch fall back plainly.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int | None:
    if not needle or start >= len(haystack):
        return None
    stop = len(haystack) - len(needle)
    for i in range(max(start, 0), stop + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _last_subseq(haystack: list[int], needle: list[int]) -> int | None:
    last: int | None = None
    start = 0
    while True:
        pos = _find_subseq(haystack, needle, start)
        if pos is None:
            return last
        last = pos
        start = pos + 1


def format_for_sft(
    tokenizer,
    messages: list[dict],
    max_len: int,
    *,
    train_on_last_only: bool = True,
) -> dict:
    """Tokenize chat messages with verified target masking.

    Labels are -100 everywhere except the assistant response span(s). The prompt
    boundary is verified (not assumed by prefix length): if the prompt encoding
    is not an exact prefix of the full encoding — BOS/whitespace drift across
    templates — the response span is located by subsequence search instead.
    Spans that cannot be located are left masked (never trained on blindly);
    examples with zero trainable tokens should be skipped by the caller.
    """
    text = _render_template(tokenizer, messages, add_generation_prompt=False)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )
    full_ids = list(enc["input_ids"])
    labels = [-100] * len(full_ids)

    boundary: int | None
    if train_on_last_only:
        # A context-free example (conversation opener) has no prompt turns; most chat
        # templates reject an empty message list, so fall through to span search.
        prompt = (
            _render_template(tokenizer, messages[:-1], add_generation_prompt=True)
            if len(messages) > 1
            else ""
        )
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if prompt_ids and full_ids[: len(prompt_ids)] == prompt_ids:
            boundary = len(prompt_ids)
        else:
            response_ids = list(
                tokenizer(messages[-1].get("content", ""), add_special_tokens=False)["input_ids"]
            )
            boundary = _last_subseq(full_ids, response_ids)
            if boundary is None:
                raise TrainingError(
                    "Could not locate the target response in the tokenized prompt; "
                    "refusing to guess the mask boundary. Check the tokenizer chat template."
                )
        labels[boundary:] = full_ids[boundary:]
    else:
        pos = 0
        found_any = False
        for i, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            content_ids = list(
                tokenizer(m.get("content", ""), add_special_tokens=False)["input_ids"]
            )
            if not content_ids:
                continue
            # Start after everything rendered before this turn. Searching from
            # the previous span (or from 0) matched the WRONG occurrence when a
            # user message repeated the reply text, which trained the model to
            # emit user turns.
            prefix_ids = list(
                tokenizer(
                    _render_template(tokenizer, messages[:i], add_generation_prompt=False),
                    add_special_tokens=False,
                )["input_ids"]
            )
            search_from = max(pos, min(len(prefix_ids), len(full_ids)))
            hit = _find_subseq(full_ids, content_ids, search_from)
            if hit is None:
                continue  # leave uncertain spans masked; never train blindly
            labels[hit : hit + len(content_ids)] = full_ids[hit : hit + len(content_ids)]
            pos = hit + len(content_ids)
            found_any = True
        if not found_any:
            raise TrainingError(
                "Could not locate any assistant span in the tokenized prompt; "
                "refusing to guess the mask boundary."
            )
    enc["labels"] = labels
    enc["trainable_tokens"] = sum(1 for lab in labels if lab != -100)
    return enc


def _collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        ids = b["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append(b.get("attention_mask", [1] * len(ids)) + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


class _ListDataset:
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def _encode_split(
    examples: list[TrainingExample],
    tokenizer,
    seq_len: int,
    *,
    train_on_last_only: bool = True,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    skipped = 0
    for ex in examples:
        msgs = [m.model_dump() for m in ex.messages]
        try:
            item = format_for_sft(tokenizer, msgs, seq_len, train_on_last_only=train_on_last_only)
        except TrainingError:
            skipped += 1
            continue
        if item.get("trainable_tokens", 0) <= 0:
            skipped += 1
            continue
        out.append(item)
    return out, skipped


def _training_args_compat(**kwargs):
    """Create TrainingArguments tolerating v4->v5 kwarg renames.

    Known renames are applied one at a time, driven by the error message, so an
    unrelated unknown argument still fails loudly instead of being dropped.
    """
    from transformers import TrainingArguments

    renames = {
        "eval_strategy": "evaluation_strategy",
        "evaluation_strategy": "eval_strategy",
        # transformers 5 removed warmup_ratio; warmup_steps < 1 is read as a ratio.
        "warmup_ratio": "warmup_steps",
    }
    args = dict(kwargs)
    for _ in range(len(renames) + 1):
        try:
            return TrainingArguments(**args)
        except TypeError as e:
            bad = next((k for k in renames if f"'{k}'" in str(e) and k in args), None)
            if bad is None:
                raise TrainingError(f"Incompatible transformers TrainingArguments: {e}") from e
            args[renames[bad]] = args.pop(bad)
    raise TrainingError("Incompatible transformers TrainingArguments: rename loop did not converge")


def run_sft(
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None,
    cfg: TrainingConfig,
    *,
    run_dir: str | Path | None = None,
) -> TrainingRunRecord:
    started = datetime.now(UTC)
    run_id = cfg.run_name or f"run-{started.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    hw = hardware_report()
    record = TrainingRunRecord(
        run_id=run_id,
        output_dir=str(out_dir),
        config=cfg.model_dump(),
        started_at=started.isoformat(),
        hardware=hw,
    )

    spec = get_model_spec(cfg.model)
    quantize = "4bit" if cfg.method == "qlora" else None
    if cfg.method == "full":
        quantize = None

    try:
        loaded = load_model(spec, quantize=quantize)
        model = loaded.model
        tokenizer = loaded.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if cfg.method in {"lora", "qlora"}:
            if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
                # Checkpointed LoRA needs the embedding output to require grad.
                model.enable_input_require_grads()
            model, targets = apply_lora(model, tokenizer, cfg.lora)
            record.metrics["lora_target_modules"] = targets[:8]

        train_enc, train_skipped = _encode_split(
            train_examples, tokenizer, cfg.seq_len, train_on_last_only=cfg.train_on_last_only
        )
        val_enc, val_skipped = _encode_split(
            val_examples or [], tokenizer, cfg.seq_len, train_on_last_only=cfg.train_on_last_only
        )
        record.metrics["skipped_unmaskable_examples"] = {"train": train_skipped, "val": val_skipped}
        if not train_enc:
            raise TrainingError(
                "No trainable examples after masking (all targets were truncated away or "
                "unlocatable). Shorten context, raise seq_len, or check the chat template."
            )
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

        has_eval = bool(val_enc)
        eval_every = cfg.eval_steps or max(25, cfg.logging_steps * 5)
        save_every = (
            cfg.save_steps if cfg.save_steps is not None else max(50, cfg.logging_steps * 10)
        )
        if has_eval:
            # load_best_model_at_end needs save_steps to be a round multiple of eval_steps.
            save_every = max(eval_every, round(save_every / eval_every) * eval_every)
        args_kwargs = dict(
            output_dir=str(out_dir),
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_ratio=cfg.warmup_ratio,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler,
            weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps,
            save_steps=save_every,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            report_to=cfg.report_to,
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if cfg.gradient_checkpointing
            else None,
            remove_unused_columns=False,
            # Best-checkpoint selection REQUIRES an eval strategy. Gating it on
            # `cfg.eval_steps` (which defaults to None) produced
            # eval_strategy="no" together with load_best_model_at_end=True,
            # which transformers rejects with a ValueError.
            eval_strategy="steps" if has_eval else "no",
            eval_steps=eval_every,
            load_best_model_at_end=bool(has_eval),
            metric_for_best_model="eval_loss" if has_eval else None,
            greater_is_better=False if has_eval else None,
            label_names=["labels"],
        )
        if cfg.max_steps_override is not None:
            args_kwargs["max_steps"] = cfg.max_steps_override
        args = _training_args_compat(**{k: v for k, v in args_kwargs.items() if v is not None})

        from transformers import Trainer

        callbacks: list[Any] = []
        if has_eval and cfg.early_stopping_patience:
            try:
                from transformers import EarlyStoppingCallback

                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
                )
            except Exception:
                record.metrics["early_stopping"] = "unavailable in this transformers version"

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_ListDataset(train_enc),
            eval_dataset=_ListDataset(val_enc) if has_eval else None,
            data_collator=lambda b: _collate(b, pad_id),
            callbacks=callbacks,
        )

        t0 = time.time()
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        elapsed = time.time() - t0

        metrics = dict(trainer.state.log_history[-1] if trainer.state.log_history else {})
        eval_metrics: dict = {}
        if has_eval:
            eval_metrics = trainer.evaluate()
        record.metrics.update(
            {
                "train_seconds": round(elapsed, 2),
                "last_log": metrics,
                "eval": eval_metrics,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
            }
        )

        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "status": "completed",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "config": cfg.model_dump(),
            "hardware": hw,
            "metrics": record.metrics,
            "dataset_sizes": {"train": len(train_examples), "val": len(val_examples or [])},
            "model_revision": spec.hf_model_id,
            "method": cfg.method,
            "seed": cfg.seed,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        record.status = "completed"
        record.finished_at = manifest["finished_at"]
        record.metrics = manifest["metrics"]
        return record
    except Exception as e:
        record.status = "failed"
        record.error = str(e)
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "config": cfg.model_dump(),
            "hardware": hw,
            "error": str(e),
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        raise TrainingError(f"Training failed ({run_id}): {e}") from e


def resume_sft(
    cfg: TrainingConfig,
    checkpoint: str,
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None = None,
) -> TrainingRunRecord:
    """Resume training from a Trainer checkpoint inside the run directory.

    The same dataset objects must be passed (the checkpoint stores optimizer and
    RNG state, not the data). Use ``cfg.run_name`` identical to the original run.
    """
    if not checkpoint:
        raise TrainingError("resume_sft requires a checkpoint path")
    if not Path(checkpoint).exists():
        raise TrainingError(f"Checkpoint does not exist: {checkpoint}")
    cfg2 = cfg.model_copy(update={"resume_from_checkpoint": checkpoint})
    return run_sft(train_examples, val_examples, cfg2)


def config_hash(cfg: TrainingConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]
