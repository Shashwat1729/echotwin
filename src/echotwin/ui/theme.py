"""The one theme and stylesheet for the studio (system light and dark)."""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )


CSS = """
:root {
  --et-ok: #15803d; --et-warn: #b45309; --et-error: #b91c1c; --et-info: #4338ca;
  --et-surface: #f8fafc; --et-border: #e2e8f0; --et-muted: #64748b;
}
.dark {
  --et-ok: #4ade80; --et-warn: #fbbf24; --et-error: #f87171; --et-info: #a5b4fc;
  --et-surface: #1e293b; --et-border: #334155; --et-muted: #94a3b8;
}
.et-header { display: flex; align-items: baseline; gap: 12px; padding: 4px 0 8px; }
.et-header h1 { margin: 0; font-size: 1.4rem; }
.et-header span { color: var(--et-muted); font-size: 0.9rem; }
.et-rail { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.et-stage { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 8px;
  border: 1px solid transparent; font-weight: 500; }
.et-stage.et-active { background: var(--et-surface); border-color: var(--et-border); }
.et-dot { width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: 0.75rem; border: 2px solid var(--et-border); }
.et-stage[data-state="done"] .et-dot { background: var(--et-ok); border-color: var(--et-ok); color: #fff; }
.et-stage[data-state="current"] .et-dot { border-color: var(--et-info); color: var(--et-info); }
.et-stage[data-state="blocked"] { opacity: 0.5; }
.et-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.et-card { background: var(--et-surface); border: 1px solid var(--et-border); border-radius: 10px;
  padding: 12px 14px; }
.et-label { color: var(--et-muted); font-size: 0.8rem; }
.et-value { font-size: 1.5rem; font-weight: 650; }
.et-hint { color: var(--et-muted); font-size: 0.75rem; margin-top: 2px; }
.et-callout { border-left: 4px solid var(--et-info); background: var(--et-surface);
  padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
.et-callout.et-warn { border-color: var(--et-warn); }
.et-callout.et-error { border-color: var(--et-error); }
.et-callout.et-ok { border-color: var(--et-ok); }
.et-empty { text-align: center; padding: 32px 16px; border: 1px dashed var(--et-border);
  border-radius: 12px; color: var(--et-muted); }
.et-empty-title { font-size: 1.05rem; font-weight: 600; color: inherit; margin-bottom: 4px; }
.et-badge { padding: 2px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  border: 1px solid currentColor; }
.et-badge.et-ok { color: var(--et-ok); } .et-badge.et-warn { color: var(--et-warn); }
.et-badge.et-error { color: var(--et-error); } .et-badge.et-info { color: var(--et-info); }
"""
