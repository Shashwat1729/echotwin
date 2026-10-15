# Troubleshooting

## `echotwin doctor` exits non-zero

Read the lines: missing **core** deps or an unwritable home directory fail;
missing **optional** deps (torch, voxcpm, resemblyzer) only report. Blocked
DLLs (enterprise application control) are reported as `blocked`, not fatal.

## Import parses zero messages

- Confirm the file is a WhatsApp export (`_chat.txt` or a zip containing
  chat files), not a PDF or a renamed file.
- Check encoding: the loader tries UTF-8 (BOM-aware), UTF-16, then Latin-1.
- `malformed_lines`/`orphan_lines` in the report point at the exact lines.

## "Only one conversation available"

Import more chats (a zip/directory with several `_chat.txt` files). A single
conversation cannot produce a held-out split — by design.

## Baseline has no held-out prompts

Same cause as above. The mock baseline needs a non-empty test split.

## Training fails immediately

- `UnsupportedCapabilityError` for QLoRA on CPU: expected — use `lora`.
- `ModelNotAvailableError`: install the `[train]` extra and download weights
  explicitly (`scripts/download_models.py`).
- CUDA OOM: lower `seq_len`, raise `gradient_accumulation_steps`, enable
  `gradient_checkpointing`, or switch to `qlora` on GPU.

## Voice synthesis fails

- `BackendNotAvailableError`: `pip install voxcpm` + download weights; check
  `echotwin voice backends`.
- Consent error: grant voice-cloning consent in the UI/API first.
- No reference selection: run analyze → recommend → select references.

## UI/API won't start

- Ports in use: set `ECHOTWIN_API_PORT` / `ECHOTWIN_UI_PORT`.
- Never use `--share` or `0.0.0.0` on untrusted networks.
