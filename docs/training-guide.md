# Training Guide

## Method ladder (cheapest first)

1. `zero_shot` / `style_profile` / `few_shot` / `style_fewshot` — prompting.
2. `style_rag` — profile + BM25-retrieved exemplars from your own messages.
   Often the best result per unit effort. Start here.
3. `sft` / `lora` — LoRA fine-tuning when held-out metrics justify it.
4. `style_rag_ft` — retrieval on top of the trained adapter (best overall).
5. `dpo` — preference optimization on **human** labels only.

## Configuration

Copy and edit `configs/training.yaml`:

```bash
echotwin train lora --config configs/training.yaml
```

Defaults are research-backed for small personal datasets: LoRA r=16 on all
projection modules, lr=1e-4, cosine schedule, warmup 5%, gradient accumulation
for an effective batch of ~8, early stopping (patience 3) ON.

## Correctness details

- Chat-template formatting via `apply_chat_template`; Qwen3 hybrids train
  with thinking disabled, matching inference.
- Completion-only loss: prompt tokens masked with a **verified** boundary
  (token-prefix check → response-span search → loud `TrainingError`).
  Examples with zero trainable tokens are skipped and counted.
- Best-checkpoint selection on validation loss; resumption via
  `resume_from_checkpoint`; every run writes `manifest.json` (config,
  hardware, dataset hashes, metrics, failures).

## QLoRA

`method: qlora` requires a CUDA GPU + bitsandbytes. Without them it raises
`UnsupportedCapabilityError` — it never silently trains a different method.

## Anti-overfitting for tiny datasets

- Prefer `style_rag` first; train for few epochs with early stopping.
- Watch the **contamination report** in every evaluation: high train-overlap
  with high style scores means memorization, not generalization.
- Retrain from sanitized data if you need information removed — deleting an
  adapter does not guarantee removal from weights.

## When not to train

Single-digit examples, a single conversation, or no held-out split: collect
more chats first. The tooling will tell you exactly this.
