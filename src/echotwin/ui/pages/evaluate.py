"""Run a baseline on the held-out split and compare generated text to the real replies."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from ...evaluation.metrics import EvaluationReport
from ...exceptions import EchoTwinError
from ...services import chat as chat_service
from ...services import datasets, jobs, models
from ...services import evaluation as evaluation_service
from .. import components, render
from ..context import PageContext, warn

_EMPTY_METRICS = pd.DataFrame(columns=["Metric", "Reference", "Generated"])
_EMPTY_EXAMPLES = pd.DataFrame(columns=["#", "Language", "Reference", "Generated"])


def engine_choices(include_unavailable: bool = False) -> list[str]:
    return [
        e.id
        for e in models.list_engines()
        if e.kind != "mock" and (e.available or include_unavailable)
    ]


def refresh_engines(current: str | None) -> gr.Dropdown:
    """Re-list engines when a tab opens; the dataset or endpoint may have changed."""
    choices = engine_choices()
    value = current if current in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def methods_for(engine_id: str) -> list[str]:
    """Prompt-building methods only matter for engines that read the prompt."""
    for info in models.list_engines():
        if info.id == engine_id or info.kind == engine_id.split(":", 1)[0]:
            if not info.capabilities.uses_prompt_context:
                return ["plain"]
            break
    return list(chat_service.CHAT_METHODS)


def _unavailable_reason(engine_id: str) -> str | None:
    for info in models.list_engines():
        if info.id == engine_id:
            return None if info.available else (info.reason_unavailable or "Engine unavailable.")
    return f"Unknown engine {engine_id!r}."


def start_eval(engine_id: str, method: str, max_prompts: float) -> str:
    reason = _unavailable_reason(engine_id)
    if reason:
        return render.callout("warn", f"{engine_id} is not available: {reason}")
    try:
        datasets.load()  # fail now with a readable message instead of a failed job
        limit = int(max_prompts)

        def work(ctx: jobs.JobContext) -> dict:
            result = evaluation_service.run(method, engine_id, 3, limit, progress=ctx.progress)
            return {
                "run_id": result.run_id,
                "report": result.report.model_dump(),
            }

        jobs.submit("evaluation", work, heavy=not engine_id.startswith("ngram"))
    except EchoTwinError as exc:
        return warn(exc)
    return render.callout("info", f"Evaluating {engine_id} with {method}. Results appear below.")


def _latest_job() -> dict | None:
    return next((j for j in jobs.list_jobs() if j["kind"] == "evaluation"), None)


def eval_poll(last_marker: str):
    job = _latest_job()
    if job is None:
        return "", gr.skip(), gr.skip(), gr.skip(), last_marker
    marker = f"{job['job_id']}:{job['status']}"
    if job["status"] not in jobs.TERMINAL:
        pct = round(float(job.get("progress") or 0) * 100)
        text = f"Evaluating... {pct}% {job.get('message') or ''}".strip()
        return render.callout("info", text), gr.skip(), gr.skip(), gr.skip(), marker
    if marker == last_marker:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), marker
    if job["status"] != "completed":
        reason = job.get("error") or f"Job {job['status']}."
        return render.callout("error", reason), gr.skip(), gr.skip(), gr.skip(), marker
    report = EvaluationReport.model_validate(job["result"]["report"])
    metrics, examples = render.eval_tables(report)
    style = report.style
    contamination = report.contamination or {}
    cards = render.metric_cards_html(
        [
            {"label": "Prompts scored", "value": len(report.per_example)},
            {
                "label": "Style match (char 3-gram)",
                "value": "-" if style is None else f"{style.char_ngram_cosine_3:.2f}",
                "hint": "1.0 = identical character habits",
            },
            {
                "label": "Length ratio",
                "value": "-" if style is None else f"{style.length_mean_ratio:.2f}",
                "hint": "generated / real",
            },
            {"label": "Train overlap", "value": contamination.get("verdict", "-")},
        ]
    )
    return render.callout("ok", "Evaluation finished."), cards, metrics, examples, marker


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "Generate replies for held-out prompts and compare them with what the author really "
        "wrote. Metrics measure style, not whether a reply is good."
    )
    choices = engine_choices()
    with gr.Row():
        engine = gr.Dropdown(choices, value=choices[0] if choices else None, label="Engine")
        method = gr.Dropdown(
            methods_for(choices[0]) if choices else ["plain"], value="plain", label="Method"
        )
        max_prompts = gr.Slider(5, 200, value=50, step=5, label="Max prompts")
    run_btn = gr.Button("Run evaluation", variant="primary")
    status = gr.HTML()
    cards = gr.HTML(
        components.empty_state("No results yet", "Run an evaluation to compare styles.")
    )
    metrics = gr.Dataframe(_EMPTY_METRICS, interactive=False, label="Style metrics")
    examples = gr.Dataframe(_EMPTY_EXAMPLES, interactive=False, wrap=True, label="Examples")
    marker = gr.State("")

    def on_engine(engine_id: str):
        options = methods_for(engine_id) if engine_id else ["plain"]
        return gr.Dropdown(choices=options, value=options[0])

    engine.change(on_engine, engine, method)
    if ctx.tab is not None:
        ctx.tab.select(refresh_engines, engine, engine)
    run_btn.click(start_eval, [engine, method, max_prompts], status)
    ctx.timer.tick(eval_poll, marker, [status, cards, metrics, examples, marker])
