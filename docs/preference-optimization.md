# Preference Optimization

## Workflow

1. Generate candidate replies for held-out contexts (any two methods).
2. A reviewer picks the better reply per context (blinded in the UI).
3. Pairs are stored with context, chosen, rejected, model revision,
   annotator, confidence, notes, and **provenance**.
4. DPO trains from human pairs and is compared against the same SFT
   checkpoint on held-out data.

## Provenance rule (enforced)

- `human` — real reviewer labels. DPO runs.
- `synthetic-demo` — hand-authored demo labels. DPO **refuses** unless
  `--allow-demo` is passed explicitly (dry runs only).
- `draft` — auto-generated candidates awaiting review. DPO refuses.

Synthetic preferences are never treated as reliable human labels.

## DPO configuration

`echotwin train dpo --config <yaml> --pairs pairs.jsonl [--allow-demo]`
requires the `[train]` extra (TRL). Monitor repetition, relevance, diversity,
and degradation on unseen contexts. Reward-model training and GRPO are
research extensions (see `docs/research-roadmap.md`) — implemented only with
a reliable objective, suitable data, sufficient compute, and measurable
benefit.
