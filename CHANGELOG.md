# Changelog

All notable changes to this project are documented here. Dates are UTC.

## [Unreleased]

### Added

- Talk tab: converse with a persona built from pasted text or chats, with spoken replies; auto-transcription of uploaded voice; one-tick voice acknowledgement (no name/signing).
- Training: fixes for transformers 5 (`warmup_ratio`), eval/save step alignment, context-free examples, gradient checkpointing.
- Single-take voice recording: long takes are peak-normalised, split at pauses and given script-aligned transcripts.
- Personalised multilingual master script built from the user's own messages (`scripts/build_personal_script.py`).
- WhatsApp `msgstore.db` text exporter and a sanitised private Kaggle dataset builder; `personal` Kaggle tier with consent-gated VoxCPM cloning.

### Fixed

- Opus codec overshoot (peak slightly above 1.0) is no longer reported as clipping.

### Changed

- Architecture overhaul: CLI, API and UI are thin adapters over a shared
  `services/` layer; duplicated logic is removed from the adapters.
- Workspace state moved from a global `state.json` to a transactional SQLite
  `Store` (pointers, jobs, key-value). Legacy state migrates once.
- Gradio studio rebuilt as a guided stage rail with one page per step, a shared
  jobs panel and timer, and typed errors shown as callouts.
- `/generate` and `echotwin chat` use pluggable engines (`ngram`, `hf:<m>`,
  `openai:<m>`) and honour chat history; no dataset gives an error, not text.
- Demo data is a multi-conversation synthetic export so the held-out split is
  non-empty; the demo now stores its messages so every service works on it.
- `ngram` and `mock` are no longer evaluation methods; use the `ngram` engine
  and method `plain` (CLI: `evaluate baseline --ngram --method plain`).

### Added

- Job runner with persisted progress, cancel, and restart cleanup; endpoints
  `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`.
- Loopback-only guard for OpenAI-compatible endpoints
  (`ECHOTWIN_ALLOW_REMOTE_ENGINE=1` to override).

### Fixed

- Chat ignored history and rebuilt its index on every message.
- `/generate` reloaded the model per request.
- Mojibake in a module docstring.

### Known limits

- `LocalHFEngine` is unit-tested with a fake loader only; not verified end to
  end with real weights.

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
