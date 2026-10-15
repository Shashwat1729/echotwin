"""The offline demo must produce a real held-out split and a predictions file."""

from __future__ import annotations

import zipfile
from pathlib import Path

from rich.console import Console

from echotwin.dataset.builder import load_jsonl
from echotwin.demo import run_demo
from echotwin.settings import get_settings
from echotwin.workspace import get_store

ZIP = Path(__file__).resolve().parents[2] / "examples" / "synthetic_multi.zip"


def test_zip_has_six_distinct_chats():
    with zipfile.ZipFile(ZIP) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt")]
    assert len(names) >= 6 and len(set(names)) == len(names)


def test_run_demo_has_heldout_split_and_predictions():
    run_demo(Console(quiet=True))
    ds = get_settings().artifacts_dir / "datasets" / "demo"
    assert len(load_jsonl(ds / "test.jsonl")) > 0
    assert len(load_jsonl(ds / "train.jsonl")) > 0
    preds = list((get_settings().artifacts_dir / "evaluations" / "demo").glob("*.jsonl"))
    assert preds and preds[0].stat().st_size > 0
    ptr = get_store().pointers()
    assert ptr["current_dataset"] == "demo" and ptr["current_import"] and ptr["current_author"]


def test_demo_state_is_usable_by_services():
    from echotwin.services import imports, profiles

    run_demo(Console(quiet=True))
    messages, _ = imports.load_messages()
    assert messages
    assert profiles.build().profile.n_messages > 0
