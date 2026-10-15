PY ?= .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
	PY := .venv/Scripts/python.exe
else
	PY := .venv/bin/python
endif

.PHONY: install test lint format typecheck doctor demo ui serve clean

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[voice,train,ui,lang,dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

typecheck:
	$(PY) -m mypy src/echotwin

doctor:
	$(PY) scripts/check_environment.py

demo:
	$(PY) -m echotwin.cli demo

ui:
	$(PY) -m echotwin.ui

serve:
	$(PY) -m echotwin.cli serve

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
