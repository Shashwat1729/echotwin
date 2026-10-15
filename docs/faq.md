# FAQ

## Do I need a GPU?

No — for ingestion, datasets, profiles, baselines, planning, the API, and
the UI. Yes — for practical LoRA training (QLoRA needs CUDA) and voice
synthesis (backend weights, ~8GB VRAM recommended).

## Will it copy my writing perfectly?

No, and you should distrust any tool that promises that. EchoTwin measures
similarity (style metrics), memorization (contamination), and uncertainty
(sample sizes), and supports blind human comparison. Iterate: more diverse
chats + `style_rag` first, train second.

## Which model should I use?

Default `qwen3-4b-it-2507` (Apache-2.0, non-thinking, strong Hindi). Laptop
with little VRAM: `gemma3n-e4b-it` or `qwen3-1.7b`. See `configs/models.yaml`
and `docs/model-licenses.md`.

## Few-shot, RAG, or fine-tuning?

In order: style profile → few-shot → `style_rag` (retrieved exemplars) →
LoRA → LoRA + retrieval → DPO on human labels. Stop when held-out metrics
stop improving; watch contamination for memorization.

## Can it clone Hinglish (code-switching)?

No backend officially supports Hinglish as a language. The studio records
optional code-switching samples and you measure quality with the evaluation
suite. Timbre transfer usually works; language-authenticity varies.

## Is my data uploaded anywhere?

No. Everything stays under `ECHOTWIN_HOME` unless you explicitly configure
otherwise. The Colab notebook runs the same local code inside your own
runtime.

## Can I delete everything?

Yes: `echotwin privacy delete --scope all-state --yes`. Deleting adapters
does not guarantee forgetting — retrain from sanitized data for that.

## Can I use it on someone else's chats or voice?

Only with their explicit consent. Non-consensual impersonation tooling is out
of scope and against the Code of Conduct.
