"""Settings: engines, voice backends, local endpoint, and data deletion."""

from __future__ import annotations

import gradio as gr
import pandas as pd
from rich.console import Console

from ...privacy_exporter import delete_scope
from ...services import models
from ...services import voice as voice_service
from ...voice.registry import all_backend_statuses
from .. import render
from ..context import PageContext, rail_value, warn
from .home import engines_df

SCOPES = {
    "Datasets and profile": "artifacts:datasets",
    "Voice recordings and profiles": "artifacts:voice",
    "Everything": "all-state",
}


def backends_df() -> pd.DataFrame:
    rows = []
    for status in all_backend_statuses():
        health = status.get("health") or {}
        rows.append(
            {
                "Backend": status["name"],
                "Importable": "yes" if status.get("module_importable") else "no",
                "Note": status.get("error") or health.get("message") or "",
            }
        )
    return pd.DataFrame(rows, columns=["Backend", "Importable", "Note"])


def save_openai(base_url: str, model: str, api_key: str) -> str:
    try:
        models.configure_openai(base_url.strip(), model.strip(), api_key.strip() or None)
    except Exception as exc:
        return warn(exc)
    return render.callout("ok", f"Saved endpoint for {model.strip()}.")


def clear_openai() -> str:
    models.clear_openai()
    return render.callout("ok", "Endpoint removed.")


def revoke_consent() -> str:
    voice_service.revoke_consent()
    return render.callout("ok", "Voice consent revoked.")


def delete_data(scope_label: str, typed: str):
    if typed.strip() != "DELETE":
        return render.callout("warn", "Type DELETE to confirm."), rail_value()
    scope = SCOPES.get(scope_label)
    if scope is None:
        return render.callout("warn", "Choose what to delete."), rail_value()
    try:
        delete_scope(scope, Console(quiet=True))
    except Exception as exc:
        return warn(exc), rail_value()
    return render.callout("ok", f"Deleted: {scope_label}."), rail_value()


def build(ctx: PageContext) -> None:
    gr.Markdown("### Engines")
    engines = gr.Dataframe(engines_df(), interactive=False, wrap=True)
    gr.Markdown("### Voice backends")
    backends = gr.Dataframe(backends_df(), interactive=False, wrap=True)
    refresh = gr.Button("Refresh")
    refresh.click(lambda: (engines_df(), backends_df()), outputs=[engines, backends])

    gr.Markdown(
        "### Local model server\nAn OpenAI-compatible server on this machine (Ollama, "
        "llama.cpp, vLLM). Remote hosts are refused unless `ECHOTWIN_ALLOW_REMOTE_ENGINE=1`."
    )
    with gr.Row():
        url = gr.Textbox(label="Base URL", placeholder="http://127.0.0.1:11434/v1")
        model = gr.Textbox(label="Model")
        key = gr.Textbox(label="API key (optional)", type="password")
    with gr.Row():
        save_btn = gr.Button("Save endpoint", variant="primary")
        clear_btn = gr.Button("Remove endpoint")
    endpoint_status = gr.HTML()
    save_btn.click(save_openai, [url, model, key], endpoint_status)
    clear_btn.click(clear_openai, outputs=endpoint_status)

    gr.Markdown("### Privacy")
    revoke_btn = gr.Button("Revoke voice consent")
    revoke_status = gr.HTML()
    revoke_btn.click(revoke_consent, outputs=revoke_status)
    with gr.Row():
        scope = gr.Dropdown(list(SCOPES), label="Delete")
        typed = gr.Textbox(label='Type "DELETE" to confirm')
        delete_btn = gr.Button("Delete", variant="stop")
    delete_status = gr.HTML()
    delete_btn.click(delete_data, [scope, typed], [delete_status, ctx.rail])
