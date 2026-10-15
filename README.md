# EchoTwin

Local-first, privacy-preserving toolkit that learns a **consenting person's**
writing style from their own WhatsApp history, fine-tunes and evaluates a
small language model, and generates speech through **authorized** voice
cloning — with Hindi, English, and Hinglish support.

> **Status:** v0.1.0 development release. Every capability below is backed by
> working code and tests (`pytest -q`, `ruff`, `mypy` all green). Anything
> requiring downloads or a GPU says so explicitly. No fabricated metrics,
> screenshots, or checkpoints.

## Highlights

- **Robust WhatsApp ingestion** — TXT/ZIP/directory, Android + iOS variants,
  12h/24h clocks, multiline messages, emoji, attachments, deleted/edited
  markers, truncated-file reporting. Every chat becomes its own conversation.
- **Explicit authorship** — the target author is selected by you, never
  auto-guessed from activity.
- **Conversation-aware datasets** — context windows, time-gap segmentation,
  token-budget truncation, dedup, JSONL + manifests, leakage-proof splits.
- **Hybrid personalization (2026 best practice)** — method ladder
  `zero_shot → style_profile → few_shot → style_fewshot → style_rag →
  sft/lora → style_rag_ft → dpo`, all scored on identical held-out prompts.
- **Honest evaluation** — style metrics **plus** contamination (memorization)
  checks **plus** blinded pairwise comparison. Small samples are labeled
  indicative, not conclusive.
- **Modern small models** — Qwen3 family default (`qwen3-4b-it-2507`,
  Apache-2.0, non-thinking), with thinking-mode handling and vendor sampling
  recipes; LoRA/QLoRA training with verified target masking.
- **Bilingual voice studio** — 55 real Hindi/English/Hinglish recording
  scripts, adaptive plans, quality + transcript + coverage analysis,
  reference selection, consent-gated synthesis with provenance.
- **Three interfaces, one library** — Typer CLI, FastAPI service (localhost
  default), Gradio studio (localhost default). Heavy ML stays optional.

## Quickstart

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[voice,train,ui,lang,dev]"
.\.venv\Scripts\echotwin.exe demo        # 60-second synthetic demo, fully offline
.\.venv\Scripts\echotwin.exe doctor       # environment diagnostics
python -m echotwin.ui                     # local website: http://127.0.0.1:7860
```

Your own data (stays on your machine):

```powershell
echotwin import whatsapp ./export.zip
echotwin participants
echotwin select-author "Your Name"
echotwin dataset prepare
echotwin evaluate baseline --method style_rag
```

Try it in Colab (same code, your own runtime):
`notebooks/echotwin_demo.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shashwat1729/echotwin/blob/main/notebooks/echotwin_demo.ipynb)

## Architecture

See [`docs/architecture.md`](docs/architecture.md). In short: fine-tuning
locks in style, retrieval supplies content; contamination is measured, not
assumed; voice cloning is reference-conditioned, consent-gated, and evaluated
with independent tools only.

## What needs downloads or a GPU

| Capability | Requirement |
|---|---|
| `train lora` (fp16 LoRA) | `[train]` extra, model weights, ~10GB VRAM |
| `train lora` (QLoRA) | CUDA GPU + bitsandbytes (fails loudly otherwise) |
| `voice test` / `synthesize` | Backend (`pip install voxcpm`) + weights (~4GB) |
| Everything else (incl. demo, tests, UI) | CPU only, no downloads |

## CLI / API

- Full command list: `echotwin --help` and [`docs/api-reference.md`](docs/api-reference.md).
- API: `echotwin serve` → http://127.0.0.1:8765/docs (localhost default).

## Privacy & consent

Local-first, no telemetry, no external calls unless configured. Raw message
text never enters logs. Voice synthesis requires an explicit consent record
and outputs carry provenance metadata. Deleting artifacts does not guarantee
removal from trained weights — retrain from sanitized data. Details:
[`docs/privacy-and-security.md`](docs/privacy-and-security.md). Never commit
real chats, recordings, or credentials (git-ignored by default).

## Model licensing

Code is Apache-2.0. Base/speech models carry their own terms (Apache-2.0,
MIT, Gemma Terms, Llama Community). Check
[`docs/model-licenses.md`](docs/model-licenses.md) before deploying.

## Docs

`docs/` covers architecture, quickstart, data preparation, training,
evaluation, preference optimization, voice cloning + recording, pronunciation
coverage, hardware, privacy, licenses, troubleshooting, API/CLI, FAQ, and the
research roadmap.

## Troubleshooting

Start with `echotwin doctor` and [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Roadmap

See [`docs/research-roadmap.md`](docs/research-roadmap.md) — reward modeling,
embedding retrieval, backend-official voice adaptation, on-device export.

## Citation

See [`CITATION.cff`](CITATION.cff). If EchoTwin helps your work, cite it.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
