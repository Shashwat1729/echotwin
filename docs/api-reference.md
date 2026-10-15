# API & CLI Reference

## CLI (`echotwin --help` is authoritative)

```
echotwin doctor | demo | version
echotwin import whatsapp ./export.zip [--conversation-id chat]
echotwin participants
echotwin select-author "Name"
echotwin dataset prepare | validate | stats
echotwin profile
echotwin evaluate baseline [--method style_rag] [--model ""] [--rag-k 3]
echotwin evaluate compare | pairs --a M1 --b M2 [--judge]
echotwin train lora --config configs/training.yaml
echotwin train dpo --config C --pairs P.jsonl [--allow-demo]
echotwin preferences annotate | validate P.jsonl
echotwin voice backends | plan | scripts | record | import | analyze
echotwin voice coverage | recommend | test | evaluate | synthesize
echotwin chat [--model ""] [--style-fewshot]
echotwin serve [--host 127.0.0.1] [--port 8765]
echotwin ui
echotwin artifacts list | delete ID
echotwin privacy export [--out DIR] | delete --scope S --yes
```

Every command validates arguments, exits non-zero with actionable errors, and
works without model downloads except `train`, `chat --model`, and
`voice test|synthesize` (backend-dependent).

## HTTP API (FastAPI, OpenAPI at `/docs` when serving)

Base: `http://127.0.0.1:8765/api/v1`. Binds localhost by default.

| Method & path | Purpose |
|---|---|
| GET `/health`, `/capabilities` | Health, backends, recent jobs/artifacts |
| POST `/imports/upload` | Upload + parse export (multipart) |
| GET `/imports/{id}/participants` | Participant statistics |
| POST `/imports/{id}/target-author` | Select target author |
| POST `/dataset/prepare`, GET `/dataset/stats` | Build dataset + manifest |
| GET/POST/PUT `/profile`, `/profile/build` | View, build (job), edit notes |
| POST `/generate` | Text generation (`method`: direct/few-shot/style_rag; `model`: "" = mock) |
| POST `/training/jobs`, GET `/jobs/{id}` | Async training |
| POST `/evaluations/baseline` | Async baseline (method/model/retrieval/contamination) |
| GET `/voice/backends`, POST `/voice/plans`, GET `/voice/scripts` | Voice capabilities + plans |
| GET/POST/DELETE `/voice/profiles...` | Profile CRUD, uploads, transcripts, coverage, references |
| POST `/voice/synthesize` | Async consent-gated synthesis |
| GET/DELETE `/artifacts...` | Artifact listing/deletion |
| POST `/consents` | Grant/revoke consent records |

Long tasks return `{job_id, status}`; poll `/jobs/{job_id}`. Uploads are
size/extension-validated and sandboxed under `ECHOTWIN_HOME`.
