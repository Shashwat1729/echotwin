"""Pure UI helpers: stage rail state, escaped HTML, result tables."""

from __future__ import annotations

from echotwin.evaluation.metrics import EvaluationReport, StyleComparison
from echotwin.ui import components, render, theme
from echotwin.workspace import get_store, reset_store


def _state(stages, key):
    return next(s for s in stages if s.key == key)


def setup_function(_):
    reset_store()


def test_empty_workspace_starts_at_import_and_names_blockers():
    stages = render.stage_status()
    assert [s.key for s in stages] == [
        "import",
        "author",
        "dataset",
        "profile",
        "evaluate",
        "chat",
        "voice",
    ]
    assert _state(stages, "import").state == "current"
    for key in ("author", "dataset", "profile", "evaluate", "chat"):
        stage = _state(stages, key)
        assert stage.state == "blocked"
        assert stage.blocker
    assert "import" in _state(stages, "author").blocker.lower()
    assert _state(stages, "voice").state == "current"  # independent of the text pipeline


def test_pointers_advance_the_rail():
    store = get_store()
    store.set_pointer("current_import", "imp-1")
    store.set_pointer("current_author", "sender-1")
    stages = render.stage_status()
    assert _state(stages, "import").state == "done"
    assert _state(stages, "author").state == "done"
    assert _state(stages, "dataset").state == "current"
    assert _state(stages, "profile").state == "current"
    assert _state(stages, "chat").state == "blocked"
    store.set_pointer("current_dataset", "dataset-imp-1")
    stages = render.stage_status()
    assert _state(stages, "dataset").state == "done"
    assert _state(stages, "chat").state == "current"


def test_callout_escapes_markup():
    html = render.callout("warn", "<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_empty_state_escapes_markup():
    html = components.empty_state("<b>Title</b>", "<img src=x onerror=1>")
    assert "<b>" not in html and "<img" not in html


def test_rail_and_cards_escape_and_mark_active():
    stages = render.stage_status()
    html = render.rail_html(stages, "import")
    assert 'data-state="current"' in html
    assert "et-active" in html
    cards = render.metric_cards_html([{"label": "<i>x</i>", "value": "1", "hint": ""}])
    assert "<i>" not in cards


def test_eval_tables_handle_missing_style():
    report = EvaluationReport(
        report_id="r", created_at="t", method="plain", prompt_count=0, style=None
    )
    metrics, examples = render.eval_tables(report)
    assert list(metrics.columns) == ["Metric", "Reference", "Generated"]
    assert metrics.empty
    assert examples.empty


def test_eval_tables_fill_rows():
    report = EvaluationReport(
        report_id="r",
        created_at="t",
        method="plain",
        prompt_count=1,
        style=StyleComparison(
            n_reference=3,
            n_generated=3,
            emoji_rate_ref=0.2,
            emoji_rate_gen=0.1,
            length_mean_ratio=0.9,
        ),
        per_example=[{"index": 0, "generated": "hi", "reference": "hello", "language": "en"}],
    )
    metrics, examples = render.eval_tables(report)
    assert not metrics.empty
    assert examples.iloc[0]["Generated"] == "hi"


def test_tables_for_participants_and_jobs():
    df = render.participants_df(
        [{"sender_id": "a", "sender_name": "Aarav", "message_count": 5, "mean_length": 12.5}]
    )
    assert list(df["Name"]) == ["Aarav"]
    jobs = render.jobs_df(
        [
            {
                "job_id": "j1",
                "kind": "evaluation",
                "status": "running",
                "progress": 0.5,
                "message": "x",
            }
        ]
    )
    assert jobs.iloc[0]["Progress"] == "50%"
    assert render.jobs_df([]).empty


def test_theme_builds_with_css():
    assert theme.build_theme() is not None
    assert "et-rail" in theme.CSS
