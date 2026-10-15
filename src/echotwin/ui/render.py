"""Pure functions that turn workspace state and typed results into UI values.

Nothing here imports Gradio. Every string that reaches HTML is escaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import pandas as pd

from ..evaluation.metrics import EvaluationReport
from ..settings import get_settings
from ..workspace import get_store


@dataclass(frozen=True)
class StageInfo:
    key: str
    label: str
    state: str  # done | current | blocked
    blocker: str


# key, label, pointer that marks it done (None = checked separately), prerequisite key, blocker text
_STAGES = (
    ("import", "Import", "current_import", None, ""),
    ("author", "Author", "current_author", "import", "Import a WhatsApp export first."),
    ("dataset", "Dataset", "current_dataset", "author", "Choose the author to clone first."),
    ("profile", "Profile", "current_profile", "author", "Choose the author to clone first."),
    ("evaluate", "Evaluate", None, "dataset", "Prepare a dataset first."),
    ("chat", "Chat", None, "dataset", "Prepare a dataset first."),
    ("voice", "Voice", "current_voice_profile", None, ""),
)


def _has_evaluation(dataset_id: str | None) -> bool:
    if not dataset_id:
        return False
    folder = get_settings().artifacts_dir / "evaluations" / dataset_id
    return folder.exists() and any(folder.glob("*.jsonl"))


def stage_status() -> list[StageInfo]:
    pointers = get_store().pointers()
    done: dict[str, bool] = {}
    for key, _label, pointer, _pre, _blocker in _STAGES:
        if key == "evaluate":
            done[key] = _has_evaluation(pointers.get("current_dataset"))
        elif key == "chat":
            done[key] = False  # chatting never "completes"
        else:
            done[key] = bool(pointer and pointers.get(pointer))
    stages = []
    for key, label, _pointer, prerequisite, blocker in _STAGES:
        if done[key]:
            state, text = "done", ""
        elif prerequisite and not done[prerequisite]:
            state, text = "blocked", blocker
        else:
            state, text = "current", ""
        stages.append(StageInfo(key, label, state, text))
    return stages


def rail_html(stages: list[StageInfo], active: str) -> str:
    items = []
    for number, stage in enumerate(stages, start=1):
        classes = "et-stage" + (" et-active" if stage.key == active else "")
        title = f' title="{escape(stage.blocker)}"' if stage.blocker else ""
        items.append(
            f'<li class="{classes}" data-state="{escape(stage.state)}"{title}>'
            f'<span class="et-dot">{number}</span><span>{escape(stage.label)}</span></li>'
        )
    return f'<ol class="et-rail">{"".join(items)}</ol>'


def metric_cards_html(items: list[dict]) -> str:
    cards = []
    for item in items:
        hint = (
            f'<div class="et-hint">{escape(str(item.get("hint", "")))}</div>'
            if item.get("hint")
            else ""
        )
        cards.append(
            '<div class="et-card">'
            f'<div class="et-label">{escape(str(item["label"]))}</div>'
            f'<div class="et-value">{escape(str(item["value"]))}</div>{hint}</div>'
        )
    return f'<div class="et-cards">{"".join(cards)}</div>'


def callout(kind: str, text: str) -> str:
    kind = kind if kind in {"info", "warn", "error", "ok"} else "info"
    return f'<div class="et-callout et-{kind}">{escape(text)}</div>'


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def eval_tables(report: EvaluationReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["Metric", "Reference", "Generated"]
    rows: list[list[str]] = []
    style = report.style
    if style is not None:
        rows = [
            ["Emoji rate", _fmt(style.emoji_rate_ref), _fmt(style.emoji_rate_gen)],
            ["Question rate", _fmt(style.question_rate_ref), _fmt(style.question_rate_gen)],
            [
                "Mean length (chars)",
                _fmt(style.length_ref.get("mean")),
                _fmt(style.length_gen.get("mean")),
            ],
            ["Length ratio (gen/ref)", "-", _fmt(style.length_mean_ratio)],
            ["Char 3-gram cosine", "-", _fmt(style.char_ngram_cosine_3)],
            ["Punctuation distance", "-", _fmt(style.punctuation_distance)],
            ["Capitalization distance", "-", _fmt(style.capitalization_distance)],
            ["Repetition rate", "-", _fmt(style.repetition_rate_gen)],
        ]
    metrics = pd.DataFrame(rows, columns=columns)
    examples = pd.DataFrame(
        [
            {
                "#": row.get("index"),
                "Language": row.get("language"),
                "Reference": row.get("reference"),
                "Generated": row.get("generated"),
            }
            for row in report.per_example
        ],
        columns=["#", "Language", "Reference", "Generated"],
    )
    return metrics, examples


def participants_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": r.get("sender_name"),
                "Messages": r.get("message_count"),
                "Avg length": round(float(r.get("mean_length") or 0), 1),
                "ID": r.get("sender_id"),
            }
            for r in rows
        ],
        columns=["Name", "Messages", "Avg length", "ID"],
    )


def jobs_df(jobs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Job": j.get("job_id"),
                "Kind": j.get("kind"),
                "Status": j.get("status"),
                "Progress": f"{round(float(j.get('progress') or 0) * 100)}%",
                "Message": j.get("message") or j.get("error") or "",
            }
            for j in jobs
        ],
        columns=["Job", "Kind", "Status", "Progress", "Message"],
    )
