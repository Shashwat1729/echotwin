# Contributing to EchoTwin

Thanks for considering a contribution. EchoTwin handles sensitive data
(conversations, voices), so correctness and honesty matter more than speed.

## Ground rules

1. **Real features only.** Do not claim a capability without working code and
   tests. No fabricated metrics, screenshots, checkpoints, or demo results.
2. **Local-first.** No telemetry, no network calls without explicit user
   configuration. Keep heavy ML dependencies optional.
3. **Privacy by default.** Never log raw message text. Keep PII out of reports.
4. **Tests for behavior.** Unit tests for parsing/dataset/coverage logic;
   integration tests for CLI/API flows. Mock expensive models; label mocks.
5. **Small, reviewable diffs.** One concern per PR, with updated docs.

## Workflow

```bash
git checkout -b feat/short-description
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check src tests
.venv/Scripts/python -m ruff format --check src tests
.venv/Scripts/python -m mypy src/echotwin
```

## Commit messages

Conventional commits: `feat(scope): description`, `fix(scope): description`,
`docs(scope): description`, `test(scope): description`, `refactor(scope): ...`.

## Reporting issues

Use the issue templates. For security issues, see SECURITY.md — do not open
public issues for vulnerabilities.
