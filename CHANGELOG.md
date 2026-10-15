# Changelog

All notable changes to this project are documented here. Dates are UTC.

## [0.1.0] - 2026-10-15

First public development release (private repository).

### Added

- WhatsApp TXT/ZIP/directory ingestion: Android + iOS variants, 12h/24h
  timestamps, multiline messages, Unicode/emoji, attachments, deleted/edited
  markers, system events, truncated-header reporting, per-chat conversation ids.
- Explicit target-author selection (never auto-selected).
- Conversation-aware dataset builder: context windows, time-gap segmentation,
  token-budget truncation, low-information filtering, exact + near dedup,
  JSONL export, reproducible manifests.
- Leakage-resistant splits by conversation (random + chronological), frozen
  held-out sets, hard overlap assertions, insufficient-data warnings.
- Pure-Python BM25 retrieval over the author's own messages with
  conversation-level exclusion; `style_rag` / `style_rag_ft` methods.
- Interpretable style profiles (length, punctuation, capitalization, emoji,
  vocabulary, abbreviations, language use, code-switching, activity trends).
- Model registry with 2026-verified entries (Qwen3 family default,
  Apache-2.0), thinking-mode handling (`enable_thinking=False` + output
  stripping), vendor-recommended sampling recipes.
- Real SFT/LoRA/QLoRA pipeline (transformers Trainer + PEFT) with verified
  target masking, checkpointing, resumption, early stopping, run manifests;
  QLoRA fails loudly without CUDA + bitsandbytes.
- Preference pairs with provenance gating; TRL-based DPO refuses
  non-human labels unless explicitly overridden for dry runs.
- Evaluation: style metrics, contamination/memorization checks, blinded
  pairwise protocol with human-score storage and labeled model-judge option.
- Bilingual voice studio: 55-sentence Hindi/English/Hinglish script library
  (no fabricated dates), recording-plan generator, audio-quality analysis,
  transcript verification, grapheme-level coverage reports, adaptive
  recommendations, reference selection, synthesis with provenance metadata.
- VoxCPM2 adapter (verified capabilities) + Chatterbox Multilingual
  alternative; consent-gated synthesis; local voice-profile management.
- FastAPI service (localhost default, sandboxed uploads, async jobs),
  full Typer CLI, Gradio studio UI with working controls.
- 190+ pytest tests, ruff + mypy clean, synthetic demo data + Colab notebook.

### Known limitations

- Real model training needs a GPU and multi-GB downloads; voice synthesis
  needs a backend install + weights. The demo and all unit tests run on CPU.
- Hinglish is not an official language of any bundled backend; mixing quality
  must be measured per user with the evaluation suite.
- Speaker-similarity/ASR evaluators are optional extras; without them the
  voice evaluation reports `blocked` with install hints instead of numbers.
