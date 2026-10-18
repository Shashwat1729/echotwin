"""Style profile: how the author writes, as numbers and a prompt fragment."""

from __future__ import annotations

import gradio as gr

from ...services import profiles
from .. import components, render
from ..context import PageContext, go, rail_value, warn


def _cards(profile: dict) -> str:
    length = profile.get("length", {})
    emoji = profile.get("emoji", {})
    punct = profile.get("punctuation", {})
    langs = profile.get("language_use", {}).get("distributions", {})
    top_lang = max(langs, key=langs.get) if langs else "-"
    return render.metric_cards_html(
        [
            {"label": "Messages", "value": profile.get("n_messages", 0)},
            {"label": "Mean length", "value": f"{length.get('chars_mean', 0):.0f} chars"},
            {"label": "Emoji messages", "value": f"{emoji.get('messages_with_emoji_pct', 0):.0f}%"},
            {"label": "Questions", "value": f"{punct.get('question_share_pct', 0):.0f}%"},
            {"label": "Main language", "value": top_lang, "hint": f"{langs.get(top_lang, 0):.0f}%"},
        ]
    )


def do_build():
    try:
        result = profiles.build()
    except Exception as exc:
        return (
            components.empty_state("No profile", "Fix the problem below."),
            "",
            warn(exc),
            rail_value(),
        )
    profile = result.profile
    data = profile if isinstance(profile, dict) else profile.model_dump(mode="json")
    return _cards(data), result.fragment, data, rail_value()


def build(ctx: PageContext) -> None:
    gr.Markdown("Measure how the author writes. The fragment below is what prompts receive.")
    build_btn = gr.Button("Build profile", variant="primary")
    cards = gr.HTML(
        components.empty_state("No profile yet", "Build one to see the author's habits.")
    )
    fragment = gr.Textbox(label="Prompt fragment", lines=6, interactive=False)
    with gr.Accordion("Full profile", open=False):
        data = gr.JSON()
    next_btn = gr.Button("Next: Evaluate")
    build_btn.click(do_build, outputs=[cards, fragment, data, ctx.rail])
    next_btn.click(lambda: go("evaluate"), outputs=ctx.tabs)
