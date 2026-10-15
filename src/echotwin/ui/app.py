"""EchoTwin Studio: a guided Gradio app over the service layer.

Pages live in ``ui/pages`` and call ``echotwin.services`` only; this module just
assembles the layout (stage rail, tabs, jobs panel) and launches it.
"""

from __future__ import annotations

from dataclasses import replace

import gradio as gr

from .. import __version__
from ..services import jobs
from ..settings import get_settings
from . import render
from .context import PageContext, rail_value
from .pages import chat, dataset, evaluate, home, import_, profile, settings, voice
from .theme import CSS, build_theme

APP_TITLE = "EchoTwin Studio"

_PAGES = (
    ("Home", "home", home),
    ("Import", "import", import_),
    ("Dataset", "dataset", dataset),
    ("Profile", "profile", profile),
    ("Evaluate", "evaluate", evaluate),
    ("Chat", "chat", chat),
    ("Voice", "voice", voice),
    ("Settings", "settings", settings),
)


def _jobs_view():
    return render.jobs_df(jobs.list_jobs()[:10]), rail_value()


def cancel_job(job_id: str) -> str:
    job_id = (job_id or "").strip()
    if not job_id:
        return render.callout("warn", "Enter a job id.")
    try:
        jobs.cancel(job_id)
    except Exception as exc:
        return render.callout("warn", str(exc))
    return render.callout("ok", f"Cancel requested for {job_id}.")


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            f'<div class="et-header"><h1>{APP_TITLE}</h1>'
            f"<span>v{__version__} · everything runs on this machine</span></div>"
        )
        timer = gr.Timer(1.0)
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                rail = gr.HTML(rail_value())
            with gr.Column(scale=5), gr.Tabs(selected="home") as tabs:
                ctx = PageContext(rail=rail, tabs=tabs, timer=timer)
                for label, tab_id, page in _PAGES:
                    with gr.Tab(label, id=tab_id) as tab:
                        page.build(replace(ctx, tab=tab))
        with gr.Accordion("Jobs", open=False):
            jobs_table = gr.Dataframe(render.jobs_df([]), interactive=False, wrap=True)
            with gr.Row():
                job_id = gr.Textbox(label="Job id", scale=3)
                cancel_btn = gr.Button("Cancel job", scale=1)
            cancel_status = gr.HTML()
            cancel_btn.click(cancel_job, job_id, cancel_status)
        timer.tick(_jobs_view, outputs=[jobs_table, rail])
    return demo


def launch(host: str, port: int, share: bool = False) -> None:
    jobs.startup_cleanup()
    build_app().launch(
        server_name=host,
        server_port=port,
        inbrowser=False,
        share=share,
        theme=build_theme(),
        css=CSS,
    )


def main() -> None:
    s = get_settings()
    s.ensure_dirs()
    launch(s.ui_host, s.ui_port)


if __name__ == "__main__":
    main()
