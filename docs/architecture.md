# EchoTwin Architecture

EchoTwin is a **local-first, hybrid personalization system**: fine-tuning locks
in *style and behavior*, retrieval supplies *content*. This is the 2026
production consensus (Prompt → RAG → Fine-tune), adapted to single-person
chat imitation.

## The text pipeline

```
WhatsApp export (.txt / .zip / directory)
        │
        ▼
┌───────────────┐   explicit, never auto-selected
│ participant    │──► target author
│ statistics     │
└───────┬───────┘
        ▼
┌───────────────┐   context windows · time-gap segmentation ·
│ dataset       │──► token-budget truncation · low-info filter
│ builder       │   dedup · JSONL + manifest
└───────┬───────┘
        │
        ├──► leakage-resistant splits (by conversation, frozen held-out)
        │
        ├──► style profile (measured, editable, no psychology)
        │
        ├──► BM25 index over the author's own messages
        │         (conversation-excluded retrieval for style_rag)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ method ladder (same held-out prompts for all)     │
│ zero_shot → style_profile → few_shot →            │
│ style_fewshot → style_rag → sft/lora →            │
│ style_rag_ft → dpo                                │
└───────┬──────────────────────────────────────────┘
        ▼
evaluation: style metrics + contamination (memorization)
            + blinded pairwise (human gold, labeled model-judge)
```

### Why this order

1. **Baselines before fine-tuning.** `style_rag` (profile + retrieved
   exemplars) is often good enough and costs nothing to train. Train only
   when held-out metrics justify it.
2. **Retrieval + fine-tuning combine** (`style_rag_ft`): the adapter carries
   the register, retrieved exemplars carry the content.
3. **Contamination is measured, not assumed.** Small personal datasets
   memorize; every evaluation reports exact + near-duplicate overlap against
   training targets alongside style similarity.

### Training specifics

- Completion-only loss: prompt tokens are label-masked (`-100`) with a
  **verified** boundary (prefix check → response-span search → loud failure).
- Qwen3 hybrid models train and generate with thinking disabled
  (`enable_thinking=False`); stray thinking blocks are stripped at decode.
- LoRA r=16, lr=1e-4, early stopping ON by default; QLoRA requires CUDA +
  bitsandbytes and fails loudly otherwise.

## The voice pipeline

```
backend capabilities (verified, never invented)
        │
        ▼
recording plan (mandatory minimum → optional coverage → test-first)
        │
        ▼
55-sentence hi/en/Hinglish script library (editable, no fake dates)
        │
        ▼
record (browser) / upload → quality analysis → transcript verification
        │
        ▼
coverage report → adaptive recommendations → reference selection
(single clip unless the backend supports several; sources never modified)
        │
        ▼
consent-gated synthesis → provenance metadata → evaluation
(latency/RTF always; similarity/WER only with real evaluators)
```

Speaker adaptation (LoRA/SFT) is a **separate, backend-official workflow** —
never confused with the text LoRA pipeline.

## Runtime topology

- **CLI** (`echotwin ...`): full pipeline, lazy heavy imports.
- **API** (FastAPI, localhost default): validated schemas, sandboxed uploads,
  background jobs for training/synthesis.
- **UI** (Gradio, localhost default): every control calls the same library
  functions as the CLI; long jobs poll status instead of blocking.
- **Storage**: SQLite metadata + JSONL artifacts under `ECHOTWIN_HOME`.
  Raw content never enters logs; PII redaction is configurable.
