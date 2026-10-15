#!/usr/bin/env bash
# EchoTwin POSIX bootstrap: creates .venv, installs extras, runs checks.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[voice,train,ui,lang,dev]"
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
echo "Done. Try: .venv/bin/echotwin demo"
