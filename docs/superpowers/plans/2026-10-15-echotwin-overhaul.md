# EchoTwin Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native execution chosen by the user: "continue implementation end to end"). Steps use checkbox syntax.

**Goal:** Replace the flaky UI/state/generation plumbing with a workspace store, pluggable text engines, a typed service layer, persisted jobs, and a rebuilt Gradio app.

**Architecture:** CLI, API and Gradio are thin adapters over `echotwin.services`. Services use a SQLite `Store` (WAL, migrations, typed workspace pointers) and a `TextEngine` registry. Legacy `storage.state` / `storage.db` stay as thin shims over the Store so untouched call sites keep working while adapters migrate.

**Tech Stack:** Python 3.12 (venv `.venv`), pydantic v2, SQLite stdlib, FastAPI, Typer, Gradio 6.30, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-15-echotwin-overhaul-design.md`

## Global Constraints

- Run Python via `.venv/Scripts/python.exe`; tests via `.venv/Scripts/python.exe -m pytest -q`.
- Base `requires-python >=3.11,<3.13`; no new hard dependencies (Gradio/torch stay optional extras).
- No message content in logs, job messages, or DB rows.
- Localhost bind defaults unchanged; `OpenAICompatEngine` remote URL requires `ECHOTWIN_ALLOW_REMOTE_ENGINE=1`.
- Mock engine only with `ECHOTWIN_ENABLE_MOCK=1` or in tests (`PYTEST_CURRENT_TEST`).
- Existing CLI commands/flags and API paths keep working.
- All 209 existing tests stay green at the end of every task (tests may be edited only where code moved).
- Commit after each task on branch `overhaul`; never push.

## Review Focus

- Concurrent pointer writes from threads must not lose updates (Task 1 test).
- Corrupt/partial `state.json` during migration must not crash startup (Task 1 test).
- Empty train split / no dataset: chat and eval return typed errors, never fake text (Task 5 tests).
- Job whose function raises, and job left `running` after a crash (Task 4 tests).
- Unavailable engine selected (missing weights, unreachable endpoint) returns `Generation(error=...)` (Task 2 tests).

## File Structure

```
src/echotwin/workspace/{__init__,store,migrations}.py   # Task 1
src/echotwin/storage/{db,state}.py                      # shims over Store (Task 1)
src/echotwin/engines/{__init__,base,ngram,mock,local_hf,openai_compat,registry}.py  # Task 2
src/echotwin/services/{__init__,jobs,imports,datasets,profiles,evaluation,chat,voice,models}.py  # Tasks 4-6
src/echotwin/api/*                                      # Task 7
src/echotwin/cli/{__init__,data,train,evaluate,voice,admin}.py  # Task 8
src/echotwin/ui/{app,theme,components,render}.py, ui/pages/*.py # Tasks 9-10
scripts/generate_demo_data.py, examples/synthetic_multi.zip     # Task 3
```

---

### Task 0: Branch and baseline commit

- [ ] **Step 1:** `git switch -c overhaul`
- [ ] **Step 2:** `.venv/Scripts/python.exe -m pytest -q` -> all pass.
- [ ] **Step 3:** `git status`; commit existing in-progress work (`models/ngram.py`, `tests/unit/test_ngram.py`, `scripts/fetch_sample_audio.py`, modified files). Do not add `sample_audio/` or any real recordings.

### Task 1: Workspace store, migrations, state/db shims

**Files:**
- Create: `src/echotwin/workspace/__init__.py`, `store.py`, `migrations.py`
- Modify: `src/echotwin/storage/state.py`, `src/echotwin/storage/db.py` (same public names, delegate to Store)
- Test: `tests/unit/test_workspace_store.py`

**Interfaces (produces):**
- `Store(path: Path)`; `get_store() -> Store` (singleton keyed by `settings.db_path`); `reset_store()` for tests.
- `Store.conn() -> sqlite3.Connection` (per-thread, WAL, `foreign_keys=ON`, `busy_timeout=5000`, `row_factory=Row`).
- `Store.pointer(name) -> str | None`; `Store.set_pointer(name, value: str | None)`; `POINTERS = ("current_import","current_author","current_dataset","current_profile","current_voice_profile")`.
- `Store.kv_get(key, default=None)`; `Store.kv_set(key, value)` (JSON-encoded).
- Jobs: `job_insert(job: dict)`, `job_update(job_id, **fields)`, `job_get(job_id) -> dict | None`, `job_list() -> list[dict]`, `fail_interrupted() -> int`.
- `LEGACY_KEYS = {"last_import_id":"current_import","target_sender_id":"current_author","last_dataset_id":"current_dataset","last_profile_id":"current_profile","last_voice_profile_id":"current_voice_profile"}`.

- [ ] **Step 1: Failing tests:** pointer round-trip; unknown key via kv; 8 threads x 50 writes of different pointers/kv keys, all final values present; `state.json` migration (valid -> pointers set, file renamed `state.json.migrated`; corrupt JSON -> no crash, renamed); `fail_interrupted` marks `running` jobs failed; reopening same DB does not re-run migrations (`schema_version` unchanged).
- [ ] **Step 2:** Run -> FAIL.
- [ ] **Step 3:** Implement schema v1 = existing tables (artifacts, consents, imports, runs) + `workspace` (single row `default` with the five pointer columns), `kv`, `job`, `schema_version`. `migrations.py` holds ordered `MIGRATIONS`; runner applies those above current version in one transaction.
- [ ] **Step 4:** Rewrite `storage/state.py` (`load/save/get/set` map legacy keys to pointers, else kv) and `storage/db.py` (same function names and return shapes on the Store connection). Fix mojibake comments. One-time `state.json` import on first `get_store()`.
- [ ] **Step 5:** Full suite -> PASS. Commit `feat(workspace): SQLite store with migrations, typed pointers, job table`.

### Task 2: Engines package

**Files:**
- Create: `src/echotwin/engines/{__init__,base,ngram,mock,local_hf,openai_compat,registry}.py`
- Modify: `src/echotwin/models/ngram.py` (re-export shim), `src/echotwin/evaluation/baseline.py` (`METHODS`)
- Test: `tests/unit/test_engines.py`

**Interfaces (produces):**
- `GenConfig` = `models.generation.GeneratorConfig` re-exported.
- `Generation(text: str, engine: str, latency_s: float, error: str | None = None)`.
- `EngineCapabilities(uses_prompt_context: bool, supports_seed: bool, needs_network: bool)`.
- `BaseEngine`: `name`, `kind`, `capabilities`; `generate_full(messages, cfg) -> Generation` (exceptions -> `error`); `generate(messages, cfg) -> str` (raises `ModelNotAvailableError` on error; satisfies existing `TextGenerator` protocol); `close()`.
- `NGramEngine(model, seed=7)`, `NGramEngine.for_dataset(dataset_id)` (cached model at `artifacts/models/ngram-<dataset_id>.json`, retrained when train.jsonl is newer).
- `MockEngine(exemplars, seed)`; guarded by `mock_allowed()` (env `ECHOTWIN_ENABLE_MOCK=1` or `PYTEST_CURRENT_TEST`).
- `LocalHFEngine(spec_name, loader=load_model)` (lazy load).
- `OpenAICompatEngine(base_url, model, api_key=None, timeout=60)`: POST `{base_url}/chat/completions` via `urllib.request`; non-loopback host rejected unless `ECHOTWIN_ALLOW_REMOTE_ENGINE=1`.
- `EngineInfo(id, kind, label, available, reason_unavailable, capabilities)`; `registry.list_engines(dataset_id=None) -> list[EngineInfo]`; `registry.get_engine(engine_id, dataset_id=None) -> BaseEngine` (one resident heavy engine behind a lock; old engine `close()`d on switch); `registry.register_factory(prefix, factory)` for tests. Ids: `ngram`, `hf:<spec>`, `openai:<name>`, `mock`. OpenAI config from kv key `engine_openai` = `{"base_url","model","api_key"}`.

- [ ] **Step 1: Failing tests:** n-gram deterministic per seed; failing engine returns `Generation.error`; `mock_allowed` guard; OpenAICompat against a stdlib `http.server` stub returning `{"choices":[{"message":{"content":"hi"}}]}`; remote URL rejected; registry lists `ngram` available and `hf:*` unavailable with a reason; `get_engine` returns the same instance twice and closes the old one on switch.
- [ ] **Step 2:** FAIL. **Step 3:** Implement; move n-gram code to `engines/ngram.py`, leave `models/ngram.py` as `from ..engines.ngram import *`.
- [ ] **Step 4:** `evaluation/baseline.py`: remove `"ngram"`/`"mock"` from `METHODS`, add `"plain"` (prompt only, no system message). Grep src and tests for `"ngram"`/`"mock"` method usage and migrate to engine selection; CLI keeps `--method ngram|mock` as aliases (`ngram` -> engine `ngram`, method `plain`).
- [ ] **Step 5:** Suite PASS. Commit `feat(engines): pluggable text engines and registry`.

### Task 3: Multi-conversation demo data

**Files:** Modify `scripts/generate_demo_data.py`, `src/echotwin/demo.py`; Create `examples/synthetic_multi.zip`, `tests/integration/test_demo_multi.py`

- [ ] **Step 1: Test:** after `run_demo`, the test split is non-empty and an evaluation predictions file exists.
- [ ] **Step 2:** FAIL. **Step 3:** Generator emits at least 6 deterministic synthetic chat `.txt` files (same author "Aarav", distinct contacts, mixed Hindi/English/Hinglish) zipped; `run_demo` loads the zip, parses every chat, sets Store pointers, runs `style_rag` with `NGramEngine`.
- [ ] **Step 4:** PASS. Commit `feat(demo): multi-conversation synthetic export`.

### Task 4: Jobs service

**Files:** Create `services/{__init__,jobs}.py`; Test `tests/unit/test_jobs.py`

**Interfaces:** `JobContext.progress(frac, msg="")`, `JobContext.cancelled() -> bool`; `jobs.submit(kind, fn: Callable[[JobContext], dict | None], *, heavy=False) -> dict`; `jobs.get(job_id)`; `jobs.list_jobs()`; `jobs.cancel(job_id) -> bool`; `jobs.wait(job_id, timeout=30) -> dict`; `jobs.startup_cleanup()`. Job dict keys: `job_id, kind, status(queued|running|completed|failed|cancelled), progress, message, created_at, started_at, finished_at, result, error`.

- [ ] **Step 1: Tests:** success stores result and progress=1; raising fn -> `failed`, `error` = class name + message truncated to 200 chars, no traceback; cancel before start -> `cancelled`; cooperative cancel mid-run; `startup_cleanup` marks stale `running` as failed `interrupted`; two heavy jobs never overlap.
- [ ] **Step 2-4:** FAIL, implement over Store job methods + `ThreadPoolExecutor(2)` + `Semaphore(1)` for heavy, PASS. Commit `feat(services): persisted job runner`.

### Task 5: Core services

**Files:** Create `services/{imports,datasets,profiles,evaluation,chat,models}.py`; Test `tests/unit/test_services.py`

**Interfaces (use `get_store()`):**
- `imports.import_export(path) -> ImportResult(import_id, n_messages, conversations, participants)`; sets `current_import`; writes `artifacts/imports/<id>/messages.jsonl`.
- `imports.load_messages(import_id=None) -> tuple[list[Message], str]`; `imports.select_author(name) -> str` (sets `current_author`, updates import row).
- `datasets.prepare(context_messages=4, max_context_chars=1200, split_mode="conversation") -> DatasetResult(dataset_id, n_examples, n_train, n_val, n_test, leakage_check_passed, warnings)`; `datasets.load(dataset_id=None) -> LoadedDataset(train, val, test, all, dir)`; `datasets.manifest(dataset_id=None) -> dict`.
- `profiles.build() -> ProfileResult(path, fragment, profile)`; `profiles.current_fragment() -> str | None`.
- `evaluation.run(method, engine_id, rag_k=3, max_prompts=50, progress=None) -> EvalResult(run_id, predictions_path, report)`; empty test split -> `EchoTwinError`.
- `chat.ChatSession(history: list[dict])`; `chat.reply(session, text, engine_id, method) -> ChatReply(text, engine, method, latency_s, exemplars, error)`; retrieval index and fragment cached by `(dataset_id, mtime)`; last 6 turns sent to engine; no dataset -> `EchoTwinError`, never mock text.

- [ ] **Step 1: Tests** on temp `ECHOTWIN_HOME` with `examples/synthetic_multi.zip`: import -> participants include Aarav -> select -> prepare (n_test > 0) -> profile -> evaluate `ngram`/`plain` and `ngram`/`style_rag` -> chat twice (session length 4; `build_index_from_examples` called once via monkeypatch counter); chat with no dataset raises; eval on empty test split raises.
- [ ] **Step 2-4:** FAIL, implement by moving logic out of `ui/app.py`, `api/routes_*.py`, `cli.py`, PASS. Commit `feat(services): import, dataset, profile, evaluation, chat`.

### Task 6: Voice service

**Files:** Create `services/voice.py`; Test `tests/unit/test_voice_service.py`

**Interfaces:** `current_profile()`, `plan(backend, languages, max_minutes)`, `save_recording(path, language, script_id) -> RecordingResult(recording_id, verdict, summary)`, `analyze()`, `set_transcript(recording_id, text)`, `coverage()`, `select_references(prefer_transcript)`, `grant_consent(name)`, `revoke_consent()`, `consent_status()`, `synthesize(text, seed)`.

- [ ] **Step 1: Tests:** generated sine wav saved -> verdict and profile pointer set; synthesize without consent -> `ConsentRequiredError`; with fake backend -> wav + provenance sidecar; revoke blocks again.
- [ ] **Step 2-4:** FAIL, move voice logic from `ui/app.py`, PASS. Commit `feat(services): voice use-cases`.

### Task 7: API re-point and /jobs

**Files:** Modify `api/routes_data.py`, `routes_models.py`, `routes_voice.py`, `app.py`, `schemas.py`; extend `tests/integration/test_api.py`

- [ ] **Step 1: Tests:** `GET /jobs`; `POST /evaluations/baseline` returns job that completes via `/jobs/{id}`; `POST /jobs/unknown/cancel` -> 404; `/generate` twice reuses engine; `/generate` with no dataset returns 4xx, not mock text.
- [ ] **Step 2-4:** Routes call services; pydantic bodies `EvalRequest`, `TrainRequest`; one `EchoTwinError` handler -> 400/404/409; `jobs.startup_cleanup()` on startup. Suite PASS. Commit `refactor(api): thin routes over services, add jobs endpoints`.

### Task 8: CLI package split

**Files:** Create `src/echotwin/cli/{__init__,data,train,evaluate,voice,admin}.py`; delete `src/echotwin/cli.py`; entry point `echotwin.cli:main` stays valid; `tests/unit/test_cli.py` unchanged.

- [ ] **Step 1:** Baseline `test_cli.py` pass. **Step 2:** Move command groups preserving names/flags/help; register sub-apps in `__init__.py`; bodies call services where one exists. **Step 3:** Suite PASS. Commit `refactor(cli): split into package, call services`.

### Task 9: UI foundation

**Files:** Create `ui/theme.py`, `ui/components.py`, `ui/render.py`; Test `tests/unit/test_ui_render.py`

**Interfaces:** `render.stage_status() -> list[StageInfo(key, label, state, blocker)]` (stages `import, author, dataset, profile, evaluate, chat, voice`; states `done|current|blocked`); `render.rail_html(stages, active)`; `render.metric_cards_html(items)`; `render.eval_tables(report) -> tuple[DataFrame, DataFrame]`; `render.participants_df(rows)`; `render.jobs_df(jobs)`; `render.callout(kind, text)` (HTML-escaped); `theme.build_theme()`, `theme.CSS`; `components.empty_state(title, body)`.

- [ ] **Step 1: Tests:** empty workspace -> import current, later stages blocked with named blocker; pointers set -> progression; `callout` escapes `<script>`; `eval_tables` handles `style=None`.
- [ ] **Step 2-4:** FAIL, implement, PASS. Commit `feat(ui): theme, components, render helpers`.

### Task 10: UI pages and app assembly

**Files:** Create `ui/pages/{home,import_,dataset,profile,evaluate,chat,voice,settings}.py`; rewrite `ui/app.py`; keep `ui/__main__.py`.

- Each page: `build(ctx: PageContext) -> None`; handlers return primitives/DataFrames/HTML and convert `EchoTwinError` to `gr.Warning` + inline callout.
- Layout: header; left rail + jobs summary; right tabs per stage + Home + Settings; `gr.Timer(1.0)` refreshes jobs and rail.
- Chat: `gr.Chatbot` messages format (verify in Gradio 6.30), engine dropdown (unavailable engines labelled with reason), method list limited by engine capabilities, "Why this reply" accordion.
- Voice: Plan, Record (script prompter + `gr.Audio` + quality badge), Analyze & coverage, References, Consent, Synthesize (job).
- Home: next-step card, engine status, "Load sample data". Settings: OpenAI-compat endpoint form (kv `engine_openai`), models table, consent revoke, delete data with confirmation.

- [ ] **Step 1: Smoke tests** `tests/unit/test_ui_smoke.py`: `build_app()` constructs; handler functions driven end to end on temp home (load sample -> select author -> prepare -> chat via n-gram).
- [ ] **Step 2-4:** FAIL, implement, PASS. **Step 5:** run app in background, exercise with `gradio_client`, screenshot with browser tools, fix visible defects. **Step 6:** commit `feat(ui): rebuilt Gradio studio`.

### Task 11: Hardening, docs, gates

- [ ] **Step 1:** Single `__version__` source used by UI and dataset manifests.
- [ ] **Step 2:** Update `docs/architecture.md`, `quickstart.md`, `api-reference.md`, `README.md`, `CHANGELOG.md`; state that `LocalHFEngine` is unit-tested with a fake loader and not verified end to end on this machine.
- [ ] **Step 3:** `pytest -q`, `ruff check src tests`, `mypy src` all green.
- [ ] **Step 4:** Commit `docs: architecture, quickstart, changelog for overhaul`.

---

## Self-review

- Spec coverage: store/migrations (T1), engines/registry/mock gate/methods cleanup (T2), demo data (T3), jobs (T4), services (T5-6), API + jobs routes (T7), CLI split (T8), UI theme/rail/pages/timer/chat/voice/settings (T9-10), cleanups (T1 mojibake, T11 version), testing/gates (T11). The legacy `state` shim exists only for call sites not yet migrated and is removed from adapters in T7, T8 and T10.
- Open spec item resolved: n-gram offers `plain` and `style_rag`, where `style_rag` primes generation with the top retrieved exemplar's leading tokens. If T5 tests show this is unreliable, fall back to `plain` only and update the UI method list.
- Type consistency: `Generation`, `EngineInfo`, `ChatReply`, `EvalResult` and the job dict keys are defined once above and reused verbatim.
