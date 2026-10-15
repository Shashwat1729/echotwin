# Quickstart

## Install (Windows PowerShell shown; POSIX equivalent in `scripts/bootstrap.sh`)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[voice,train,ui,lang,dev]"
.\.venv\Scripts\python scripts/check_environment.py
```

## 60-second synthetic demo (no private data, no downloads)

```powershell
.\.venv\Scripts\echotwin.exe demo
```

This parses the checked-in synthetic multi-chat export, builds a dataset and
style profile, and runs a `plain` n-gram baseline with a contamination report.
The same sample loads from the studio's Home tab ("Load sample data").

## Your own data (local only)

```powershell
echotwin import whatsapp ./export.zip   # TXT, ZIP, or directory; every chat becomes a conversation
echotwin participants                   # inspect, never auto-selects
echotwin select-author "Your Name"
echotwin dataset prepare                # splits with leakage checks
echotwin dataset validate
echotwin profile                        # measurable style profile
echotwin evaluate baseline --method style_rag
```

With 2+ conversations you get a real held-out split. With one conversation the
tooling warns you instead of fabricating evaluation numbers.

## Interfaces

- CLI: `echotwin --help` (everything works offline except training/synthesis).
- API: `echotwin serve` → http://127.0.0.1:8765/docs
- UI: `python -m echotwin.ui` (or `echotwin ui`) → http://127.0.0.1:7860
  (localhost only). A left rail walks Import → Author → Dataset → Profile →
  Evaluate → Chat, with Voice and Settings alongside; long work runs as jobs
  you can watch and cancel in the Jobs panel.
- Colab: `notebooks/echotwin_demo.ipynb` runs the CPU demo end-to-end.

## Real training / voice (optional, heavy)

- Text: `echotwin train lora --config configs/training.yaml` (needs GPU +
  model download; see `docs/training-guide.md` and `hardware-requirements.md`).
- Voice: `echotwin voice backends`, then the Voice Studio tab
  (`docs/voice-cloning.md`). Synthesis needs a backend install + weights and
  an explicit voice-cloning consent record.
