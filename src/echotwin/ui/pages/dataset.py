"""Build the leakage-checked training dataset."""

from __future__ import annotations

import gradio as gr

from ...services import datasets
from .. import components, render
from ..context import PageContext, go, rail_value, warn


def _cards(result: datasets.DatasetResult) -> str:
    return render.metric_cards_html(
        [
            {"label": "Examples", "value": result.n_examples},
            {"label": "Train", "value": result.n_train},
            {"label": "Validation", "value": result.n_val},
            {"label": "Held-out test", "value": result.n_test, "hint": "never trained on"},
            {
                "label": "Leakage check",
                "value": "passed" if result.leakage_check_passed else "FAILED",
            },
        ]
    )


def do_prepare(context_messages: float, test_fraction: float):
    try:
        result = datasets.prepare(
            context_messages=int(context_messages), test_fraction=float(test_fraction)
        )
        manifest = datasets.manifest()
    except Exception as exc:
        return components.empty_state("No dataset", "Fix the problem below."), warn(exc), {}, rail_value()
    notes = list(result.warnings)
    if result.n_test == 0:
        notes.append("The held-out test split is empty, so evaluation cannot run. Import more chats.")
    warnings_html = "".join(render.callout("warn", n) for n in notes)
    return _cards(result), warnings_html, manifest, rail_value()


def build(ctx: PageContext) -> None:
    gr.Markdown(
        "Turn the chat into (context, reply) examples for the chosen author, then split by "
        "conversation so the held-out set is never seen during training."
    )
    with gr.Row():
        context = gr.Slider(1, 12, value=4, step=1, label="Context messages")
        test_fraction = gr.Slider(0.05, 0.4, value=0.1, step=0.05, label="Held-out fraction")
    build_btn = gr.Button("Build dataset", variant="primary")
    cards = gr.HTML(components.empty_state("No dataset yet", "Build one to see its size and splits."))
    warnings = gr.HTML()
    with gr.Accordion("Manifest", open=False):
        manifest = gr.JSON()
    next_btn = gr.Button("Next: Profile")
    build_btn.click(do_prepare, [context, test_fraction], [cards, warnings, manifest, ctx.rail])
    next_btn.click(lambda: go("profile"), outputs=ctx.tabs)
