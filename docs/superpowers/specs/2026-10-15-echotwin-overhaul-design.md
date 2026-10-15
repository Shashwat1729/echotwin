# EchoTwin Overhaul — Design Spec

Date: 2026-10-15
Status: draft for review
Path: architectural (restructures how UI, API, CLI and storage fit together)

## 1. Purpose and success criteria

EchoTwin is a local-first toolkit that learns one consenting person's chat
writing style from their WhatsApp export, evaluates personalization methods on
held-out data, lets the user chat with the resulting "twin", and clones their
voice from authorized recordings.

**Audience (decided):** the owner, as a personal daily-use tool on their own
data. Not a public showcase.

**Text-engine direction (decided):** pluggable, local-first engine. Mock output
is never presented as a twin.

Success criteria:

1. A user can go import → author → dataset → profile → evaluate → chat → voice
   in the Gradio UI without touching the CLI, and every stage says what is
   missing when it is blocked.
2. The Chat page produces replies from a real engine (n-gram on CPU by
   default; local HF model or OpenAI-compatible endpoint when configured) and
   always shows which engine, method and exemplars produced each reply.
3. Pipeline logic exists once (services). UI, API and CLI contain no copies.
4. Long operations (training, evaluation, synthesis, model load) run as
   persisted jobs with progress and cancel; the UI never blocks on them.
5. State survives crashes and concurrent writes. No `state.json` last-write-wins.
6. The existing 209 tests keep passing (ported where they target moved code);
   new services and UI smoke tests are added.

Non-goals: public hosting, multi-user auth, new training algorithms, new voice
backends, a non-Gradio frontend.

## 2. Problems being fixed (from code review)

| # | Problem | Where |
|---|---|---|
| P1 | Chat uses `MockGenerator` ("(mock reply to: …)") or n-gram; no real LLM path in UI | `ui/app.py:chat_reply` |
| P2 | Chat ignores history; rebuilds BM25 index and retrains n-gram per message | `ui/app.py:chat_reply` |
| P3 | Dataset prep, baseline run, generator selection copy-pasted in UI, API, CLI | `ui/app.py`, `api/routes_models.py`, `cli.py` |
| P4 | `ngram` listed as a *method* though it is an *engine*; methods and engines conflated | `evaluation/baseline.py` |
| P5 | Global `state.json` (`last_*` keys), non-atomic read-modify-write, no project concept | `storage/state.py` |
| P6 | `JobManager` in memory only, mutates shared `Job` across threads, `progress` never updated, no cancel | `api/service.py` |
| P7 | `/generate` reloads the model on every request | `api/routes_models.py` |
| P8 | SQLite opens a connection and runs `executescript` on every call; no WAL | `storage/db.py` |
| P9 | UI is one 731-line file mixing widgets and logic; flat tabs; raw Markdown output; unused `gr.State`; long calls block | `ui/app.py` |
| P10 | Demo export is one conversation, so held-out split is empty and baseline is skipped | `demo.py`, `examples/` |
| P11 | Mojibake in comments (`â€”`) | `storage/db.py` |
| P12 | Heavy imports inside UI handlers; errors surface as raw strings | `ui/app.py` |

## 3. Target architecture

```
          ┌────────┐   ┌────────┐   ┌────────────┐
          │  CLI   │   │  API   │   │  Gradio UI │     thin adapters
          └───┬────┘   └───┬────┘   └─────┬──────┘     (parse input, render output)
              └────────────┼──────────────┘
                           ▼
              ┌─────────────────────────┐
              │ services/               │   typed use-cases, no formatting
              │ imports · datasets ·    │   input/output are pydantic models
              │ profiles · evaluation · │
              │ chat · voice · jobs     │
              └───┬──────────────┬──────┘
                  ▼              ▼
        ┌───────────────┐  ┌───────────────────┐
        │ workspace/    │  │ engines/          │   existing domain modules
        │ SQLite store  │  │ text engines      │   (ingestion, dataset,
        │ + migrations  │  │ + registry        │   profiling, evaluation,
        └───────────────┘  └───────────────────┘   voice, training) stay and
                                                    are called by services
```

Rule: adapters import `services`, never domain modules directly. Services
import domain modules. Nothing under `services/` imports `gradio`, `typer` or
`fastapi`.

### 3.1 Workspace store (`echotwin/workspace/`)

Replaces `storage/state.py` and extends `storage/db.py`.

- One SQLite file under `ECHOTWIN_HOME`, `journal_mode=WAL`, `foreign_keys=ON`,
  one connection per thread via a small `Store` class (no per-call
  `executescript`). Schema applied once by a versioned migration runner
  (`schema_version` table, ordered migration functions).
- Entities: `workspace`, `import`, `dataset`, `style_profile`, `run`
  (evaluation or training), `voice_profile`, `job`, `consent`, `artifact`.
- A `workspace` has pointers `current_import`, `current_author`,
  `current_dataset`, `current_profile`, `current_voice_profile`, updated
  transactionally. A default workspace `default` is created on first use, so
  CLI behavior is unchanged for single-project users.
- One-time import of existing `state.json` and `consents` rows on first open
  (read, copy, rename `state.json` to `state.json.migrated`).
- `Store` exposes typed methods (`set_current_dataset(id)`, `get_workspace()`),
  not arbitrary key get/set.
- Message content stays out of the DB (existing privacy rule). Messages and
  datasets remain JSONL artifacts; the DB stores ids, paths and metadata.

### 3.2 Engines (`echotwin/engines/`)

Separates *who generates* from *how the prompt is built*.

```python
class TextEngine(Protocol):
    name: str                      # "ngram-o3", "hf:Qwen3-4B-Instruct-2507", ...
    kind: Literal["ngram", "local_hf", "openai_compat", "mock"]
    capabilities: EngineCapabilities   # uses_prompt_context, supports_seed, ...
    def generate(self, messages: list[dict], cfg: GenConfig) -> Generation: ...
    def close(self) -> None: ...

class Generation(BaseModel):
    text: str
    engine: str
    latency_s: float
    error: str | None = None
```

- `NGramEngine`: wraps existing `models/ngram.py` (moved here). Trained model
  cached on disk per dataset (`NGramModel.save/load` already exist), loaded once.
- `LocalHFEngine`: wraps existing `TransformersGenerator` / `loader.py`, plus an
  optional LoRA adapter path from a training run.
- `OpenAICompatEngine`: HTTP client to a user-configured `base_url` and model
  name (Ollama, llama.cpp server, vLLM). Disabled unless configured. Local URL
  only unless `ECHOTWIN_ALLOW_REMOTE_ENGINE=1`, to preserve the "no external
  calls unless configured" privacy guarantee. HTTP library to be chosen at plan
  time from what `pyproject.toml` already provides (`httpx` if present, else
  stdlib `urllib`).
- `MockEngine`: unchanged behavior, only constructible when
  `ECHOTWIN_ENABLE_MOCK=1` or inside tests. The UI engine picker never lists it
  by default.
- `EngineRegistry`: `list_engines()` returns each engine with
  `available: bool` and `reason_unavailable: str | None` (missing package,
  missing weights, endpoint unreachable). `get(engine_id)` keeps one resident
  heavy engine (LRU size 1) behind a lock; switching engines calls `close()` on
  the old one. This fixes P7.
- Default engine when nothing is configured: `NGramEngine`.

`ngram` and `mock` are removed from `evaluation.baseline.METHODS`; methods stay
`zero_shot … dpo`. An engine that ignores the prompt context (n-gram,
`uses_prompt_context=False`) makes `zero_shot`, `style_profile` and `few_shot`
produce identical output, so the UI offers only methods meaningful for the
selected engine. For n-gram that is `plain` (and optionally exemplar-biased
`style_rag`; exact semantics are a plan-level decision, fallback is `plain`
only).

### 3.3 Services (`echotwin/services/`)

One module per use-case. Each function takes a `Store` (or workspace id) and a
typed request, returns a typed result, and raises existing `EchoTwinError`
subclasses on expected failure. No printing, no Markdown.

| Service | Main operations |
|---|---|
| `imports` | `import_export(path) -> ImportResult`, `list_participants(import_id)`, `select_author(import_id, name)` |
| `datasets` | `prepare(cfg) -> DatasetResult` (examples, splits, manifest, leakage report), `get_manifest(id)` |
| `profiles` | `build(...) -> StyleProfile`, `fragment(profile)` |
| `evaluation` | `run(method, engine_id, cfg) -> EvalResult`, `compare(run_ids)`; heavy path runs as a job |
| `chat` | `ChatSession` holding history; `reply(session, text, engine_id, method) -> ChatReply` with `engine`, `method`, `exemplars`, `latency_s` |
| `voice` | `plan`, `save_recording`, `analyze`, `set_transcript`, `coverage`, `select_references`, `grant_consent`, `revoke_consent`, `synthesize` (job) |
| `jobs` | `submit(kind, fn) -> Job`, `get`, `list`, `cancel`; progress callback passed to `fn` |

Caches owned by services, keyed by dataset id and file mtime: retrieval index,
trained n-gram model, loaded style profile. Fixes P2.

### 3.4 Jobs (`services/jobs.py`)

- Persisted in the `job` table (id, kind, status, progress 0..1, message,
  created/started/finished, result JSON, error). Survives UI reloads. On
  startup, jobs left `running` are marked `failed: interrupted`.
- `ThreadPoolExecutor(max_workers=2)`; at most one heavy (GPU or model-load)
  job at a time via a semaphore.
- `fn(ctx)` receives a `JobContext` with `ctx.progress(frac, msg)` and
  `ctx.cancelled()` (cooperative cancel). Fixes P6.
- All state mutation goes through the `Store` (lock-protected), not shared
  mutable objects.

### 3.5 API (`echotwin/api/`)

- Routes become thin: validate schema, call service, map `EchoTwinError` to HTTP
  status. Same paths as today (`/generate`, `/training/jobs`,
  `/evaluations/baseline`, voice and data routes) so existing clients and
  `tests/integration/test_api.py` keep working.
- Added: `/jobs`, `/jobs/{id}`, `/jobs/{id}/cancel`.
- `/generate` uses `EngineRegistry` (fixes P7). `payload: dict` endpoints get
  pydantic request models.

### 3.6 CLI (`echotwin/cli.py`)

- Split the 1180-line file into a `cli/` package: `__init__.py` (app wiring),
  `data.py`, `train.py`, `evaluate.py`, `voice.py`, `admin.py`. Commands, flags
  and output stay the same; bodies call services. `demo` and `doctor` behavior
  is unchanged.

### 3.7 Gradio UI (`echotwin/ui/`)

Gradio 6.30 is installed. Verify API details (theme/css passed to `launch()`
vs `Blocks()`, `gr.Timer`, `gr.Chatbot` message format) against the installed
version at implementation time.

```
ui/
  __init__.py
  __main__.py
  app.py            # build_app(): assemble theme, rail, pages, jobs drawer
  theme.py          # one custom theme + CSS (light/dark)
  components.py     # stage_rail, metric_cards, status_badge, job_list, empty_state
  render.py         # typed result -> DataFrame / plot / HTML (pure functions)
  pages/
    home.py         # workspace overview, next-step action, engine status, load sample data
    import_.py      # upload, parse report, participants table, author select
    dataset.py      # options, build, split + leakage report, manifest viewer
    profile.py      # style profile cards (length, emoji, language mix, punctuation)
    evaluate.py     # engine + method pickers, run job, metrics table, per-example diff
    chat.py         # gr.Chatbot, engine/method pickers, "why this reply" accordion
    voice.py        # sub-steps: plan → record → analyze → select → consent → synthesize
    settings.py     # engines (endpoint/model), models, privacy, consent revoke, delete data
```

Layout and behavior:

- **Stage rail** (left column, always visible): Import → Author → Dataset →
  Profile → Evaluate → Chat → Voice. Each stage shows done / current / blocked.
  A blocked stage opens an `empty_state` naming the missing prerequisite with a
  button that jumps to it. Rail state is derived from `Store` workspace
  pointers and refreshed after every service call.
- **Jobs drawer** (top bar): count of running jobs; expands to progress and
  cancel. Driven by a `gr.Timer` (about 1 s) polling `services.jobs.list`,
  active only while a job is running.
- **Results**: metric cards for headline numbers, `gr.Dataframe` for tables,
  plots for distributions and comparisons. Markdown only for prose. Small-sample
  and contamination warnings render as a visible callout.
- **Chat**: real `gr.Chatbot` with a persistent `ChatSession` in `gr.State`.
  Engine picker lists available engines; unavailable ones are shown disabled
  with `reason_unavailable`. Each reply has an accordion: engine, method,
  latency, retrieved exemplars (conversation id, score). With no data or no
  author the page shows `empty_state`, never mock output.
- **Voice**: a script prompter shows the next script from the recording plan
  beside the `gr.Audio` recorder. Saving runs quality analysis and shows a
  pass/warn/fail badge per clip. Reference selection shows chosen and rejected
  clips with reasons. Synthesize is disabled until consent is granted and a
  reference is selected, and runs as a job.
- **Error handling**: handlers convert `EchoTwinError` to `gr.Warning` /
  `gr.Error` plus an inline message. Unexpected exceptions are logged (no
  message content) and shown as a generic error with a log id.
- **Theme**: one restrained palette, system light/dark, readable typography,
  consistent spacing, defined only in `theme.py`.

### 3.8 Demo data (P10)

- Extend `scripts/generate_demo_data.py` to emit a synthetic WhatsApp export of
  at least 6 conversations (zip of several chats) so the conversation-level
  held-out split is non-empty. Check in under `examples/`. `echotwin demo` and
  the Home "Load sample data" button use it. Synthetic only, labeled as such.

### 3.9 Cleanups

- Fix mojibake in `storage/db.py` and any other non-UTF-8-clean text
  (search for `â€`).
- Remove duplicated generator and prompt-building code once services own it.
- Read `__version__` from one place (UI hardcodes `"0.1.0"`).
- Keep the in-progress n-gram work (`models/ngram.py`, `tests/unit/test_ngram.py`
  are currently untracked); it moves to `engines/ngram.py` in Phase 2.

## 4. Data flow (happy path)

1. Import: upload zip → `imports.import_export` parses, stores `messages.jsonl`
   artifact, records `import`, sets `current_import`.
2. Author: `list_participants` → user picks → `select_author` sets
   `current_author`.
3. Dataset: `datasets.prepare` builds examples, splits by conversation, writes
   artifacts and manifest, sets `current_dataset`.
4. Profile: `profiles.build` writes profile JSON, sets `current_profile`.
5. Evaluate: `evaluation.run` (job) on identical held-out prompts with the
   chosen engine and method; stores a `run`; UI shows metrics, contamination,
   per-example diff.
6. Chat: `chat.reply` with the chosen engine and method; history in session.
7. Voice: independent track needing only consent and recordings. Text and voice
   link only through approved text the user supplies to speak.

## 5. Error handling and safety

- Consent gate (`require_voice_consent`) and provenance sidecar stay and are
  covered by service tests.
- Upload validation (`validate_upload`) and path sandboxing (`ensure_within`)
  stay; services call them, adapters never bypass them.
- No message content in logs (existing rule), including new job `message` and
  `error` fields: job errors store exception type and a short sanitized message.
- `OpenAICompatEngine`: local base URL by default; remote requires an explicit
  env flag, because message content leaves the machine only then. The Settings
  page states this plainly.
- Engine failures return `Generation(error=…)`; chat shows the error and never
  substitutes fake text.

## 6. Testing

- **Services**: unit tests per service on a temp `ECHOTWIN_HOME` using existing
  synthetic data; port logic tests from `tests/unit/*` where code moved.
- **Store**: migration test (fresh DB and `state.json` upgrade); concurrent
  write test (threads setting different pointers, none lost).
- **Engines**: n-gram determinism; registry availability reasons; OpenAI-compat
  against a local stub HTTP server; mock gated by env flag.
- **Jobs**: progress, cancel, interrupted-on-restart.
- **API**: existing integration tests plus `/jobs` routes.
- **CLI**: existing `test_cli.py` passes unchanged.
- **UI**: build the app and assert no exceptions at construct time; drive key
  flows with `gradio_client` (load sample → author → dataset → chat reply from
  the n-gram engine). No browser dependency in CI. Manual visual check with
  browser tools at the end of the UI phase.
- Quality gates: `pytest -q`, `ruff`, `mypy` green before each phase ends.

## 7. Phasing (each phase leaves the repo green and runnable)

1. **Foundation:** `workspace/` store + migration; `Store` replaces `state.py`
   and `db.py` call sites; mojibake fix.
2. **Engines:** `engines/` package, registry, `METHODS` cleanup, mock gating,
   n-gram moved in.
3. **Services + jobs:** extract use-cases; API and CLI re-pointed at services;
   `cli/` package split; `/jobs` routes.
4. **Demo data:** multi-conversation synthetic export; `demo` command updated.
5. **UI rebuild:** theme, components, rail, pages, jobs drawer, chat, voice
   flow; old `ui/app.py` logic deleted.
6. **Hardening:** UI smoke tests; docs update (`architecture.md`,
   `quickstart.md`, `api-reference.md`, `README.md`); changelog.

## 8. Risks and open points

- Gradio 6.x API differs from older docs: check the installed version at
  implementation time and keep UI-only code isolated.
- n-gram `style_rag` semantics are undecided (3.2); fallback is `plain` only.
- The dev machine has no GPU and no cached model weights: `LocalHFEngine` is
  covered by unit tests with a fake loader, not an end-to-end run. Docs must
  say so and must not claim measured quality for it.
- Moving DB and state touches most modules. Phase 1 must land with all existing
  tests green before anything builds on it.
- Scope is large; phases are independently shippable, so work can stop after
  any phase.
- The working tree has 12 modified files and 3 untracked files from earlier
  work. Phase 1 starts by reviewing and committing or stashing that work so the
  overhaul diff stays reviewable.
