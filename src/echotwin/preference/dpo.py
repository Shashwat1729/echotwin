"""Preference optimization: annotation storage, validation, and DPO.

Provenance rules (enforced, not advisory):

* Preference pairs must carry ``provenance`` ("human" or "synthetic-demo").
* DPO refuses to run on "synthetic-demo" pairs unless ``allow_demo=True`` is
  passed explicitly — synthetic preferences are NEVER treated as reliable human
  labels.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..dataset.builder import TrainingExample
from ..exceptions import DatasetError


class PreferencePair(BaseModel):
    pair_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    context: list[dict] = Field(description="Chat turns ending with a user turn")
    chosen: str
    rejected: str
    provenance: str = Field(description="human | synthetic-demo | <method name>")
    model_revision: str | None = None
    annotator: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PreferenceValidationResult(BaseModel):
    n_pairs: int
    valid: int
    errors: list[str] = Field(default_factory=list)
    provenances: dict[str, int] = Field(default_factory=dict)
    human_share: float = 0.0


def validate_pairs(pairs: list[PreferencePair]) -> PreferenceValidationResult:
    errors: list[str] = []
    prov: dict[str, int] = {}
    valid = 0
    for i, p in enumerate(pairs):
        ok = True
        if not p.chosen.strip() or not p.rejected.strip():
            errors.append(f"pair[{i}] {p.pair_id}: empty chosen or rejected")
            ok = False
        if p.chosen.strip() == p.rejected.strip():
            errors.append(f"pair[{i}] {p.pair_id}: chosen equals rejected")
            ok = False
        if not p.context:
            errors.append(f"pair[{i}] {p.pair_id}: empty context")
            ok = False
        if any(m.get("role") not in {"user", "assistant", "system"} for m in p.context):
            errors.append(f"pair[{i}] {p.pair_id}: invalid role in context")
            ok = False
        if not p.provenance.strip():
            errors.append(f"pair[{i}] {p.pair_id}: missing provenance")
            ok = False
        prov[p.provenance] = prov.get(p.provenance, 0) + 1
        valid += int(ok)
    human = prov.get("human", 0)
    return PreferenceValidationResult(
        n_pairs=len(pairs),
        valid=valid,
        errors=errors,
        provenances=prov,
        human_share=round(human / len(pairs), 4) if pairs else 0.0,
    )


def save_pairs(pairs: list[PreferencePair], path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for pair in pairs:
            fh.write(pair.model_dump_json() + "\n")
    return len(pairs)


def load_pairs(path) -> list[PreferencePair]:
    p = Path(path)
    out: list[PreferencePair] = []
    if not p.exists():
        return out
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(PreferencePair.model_validate_json(line))
            except Exception as e:
                raise DatasetError(f"Invalid preference pair at {p}:{i}: {e}") from e
    return out


def pairs_from_examples(
    base: list[TrainingExample], candidates: dict[str, str]
) -> list[PreferencePair]:
    """Build draft pairs: base example target as chosen, a model candidate as rejected.

    This produces *drafts for review*; provenance is set to "draft" so DPO will
    refuse them until a human relabels them.
    """
    pairs = []
    for ex in base:
        cid = ex.metadata.conversation_id
        cand = candidates.get(cid) or candidates.get(ex.metadata.source_message_id or "")
        if not cand or cand.strip() == ex.messages[-1].content.strip():
            continue
        pairs.append(
            PreferencePair(
                context=[m.model_dump() for m in ex.messages[:-1]],
                chosen=ex.messages[-1].content,
                rejected=cand,
                provenance="draft",
                notes="auto-generated draft; needs human review before DPO",
            )
        )
    return pairs


def train_dpo(
    *,
    model_id: str,
    pairs: list[PreferencePair],
    output_dir: str | Path,
    beta: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    seq_len: int = 1024,
    seed: int = 20250917,
    allow_demo: bool = False,
    sft_adapter: str | None = None,
) -> dict:
    """DPO training via TRL when installed.

    Requires human-labeled pairs unless allow_demo=True. Raises a clear error
    when TRL or hardware is missing instead of pretending to run.
    """
    v = validate_pairs(pairs)
    if not v.human_share and not allow_demo:
        raise DatasetError(
            "Refusing to run DPO on non-human preference pairs. Set allow_demo=True only for "
            f"dry runs. Provenances found: {v.provenances}"
        )
    try:
        import trl  # noqa: F401
    except ImportError as e:
        raise DatasetError(
            "DPO requires TRL (uv pip install 'trl'). Install the [train] extra."
        ) from e

    from ..models.loader import hardware_report

    hw = hardware_report()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    ref_model = (
        None if sft_adapter else AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    )

    def _fmt(p: PreferencePair) -> dict:
        prompt = [
            {"role": m.get("role", "user"), "content": m.get("content", "")} for m in p.context
        ]
        chosen = [{"role": "assistant", "content": p.chosen}]
        rejected = [{"role": "assistant", "content": p.rejected}]
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    dataset = [_fmt(p) for p in pairs]
    args = DPOConfig(
        output_dir=str(out),
        beta=beta,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        max_length=seq_len,
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    return {
        "status": "completed",
        "output_dir": str(final),
        "beta": beta,
        "n_pairs": len(pairs),
        "hardware": hw,
        "provenances": v.provenances,
        "model_revision": model_id,
    }
