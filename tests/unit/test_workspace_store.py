"""Workspace store: pointers, kv, jobs, migrations, legacy state.json import."""

from __future__ import annotations

import json
import threading

import pytest

from echotwin.settings import reset_settings
from echotwin.workspace import POINTERS, Store, get_store, reset_store


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "db.sqlite3")


def test_pointer_roundtrip(store):
    assert store.pointer("current_dataset") is None
    store.set_pointer("current_dataset", "ds-1")
    assert store.pointer("current_dataset") == "ds-1"
    store.set_pointer("current_dataset", None)
    assert store.pointer("current_dataset") is None


def test_unknown_pointer_rejected(store):
    with pytest.raises(ValueError):
        store.set_pointer("current_bogus", "x")


def test_kv_roundtrip(store):
    assert store.kv_get("missing", 5) == 5
    store.kv_set("unit", {"a": [1, 2]})
    assert store.kv_get("unit") == {"a": [1, 2]}


def test_concurrent_writes_do_not_lose_updates(store):
    def worker(i: int) -> None:
        for n in range(50):
            store.set_pointer(POINTERS[i % len(POINTERS)], f"v{i}-{n}")
            store.kv_set(f"k{i}", n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(store.kv_get(f"k{i}") == 49 for i in range(8))
    for name in POINTERS:
        assert store.pointer(name) is not None


def test_migrations_not_rerun(tmp_path):
    p = tmp_path / "db.sqlite3"
    s1 = Store(p)
    v1 = s1.schema_version()
    s1.set_pointer("current_import", "i1")
    s2 = Store(p)
    assert s2.schema_version() == v1
    assert s2.pointer("current_import") == "i1"


def test_job_lifecycle_and_interrupted(store):
    store.job_insert({"job_id": "j1", "kind": "x", "status": "running", "created_at": "t"})
    store.job_insert({"job_id": "j2", "kind": "x", "status": "completed", "created_at": "t"})
    assert store.fail_interrupted() == 1
    j1 = store.job_get("j1")
    assert j1["status"] == "failed" and j1["error"] == "interrupted"
    assert store.job_get("j2")["status"] == "completed"
    store.job_update("j2", progress=0.5, result={"a": 1})
    assert store.job_get("j2")["result"] == {"a": 1}
    assert {j["job_id"] for j in store.job_list()} == {"j1", "j2"}
    assert store.job_get("nope") is None


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOTWIN_HOME", str(tmp_path))
    reset_settings()
    reset_store()
    yield tmp_path
    reset_store()
    reset_settings()


def test_legacy_state_json_migrated(home):
    (home / "state.json").write_text(
        json.dumps({"last_dataset_id": "ds-9", "target_sender_id": "s1", "custom": 3}),
        encoding="utf-8",
    )
    store = get_store()
    assert store.pointer("current_dataset") == "ds-9"
    assert store.pointer("current_author") == "s1"
    assert store.kv_get("custom") == 3
    assert not (home / "state.json").exists()
    assert (home / "state.json.migrated").exists()


def test_corrupt_state_json_does_not_crash(home):
    (home / "state.json").write_text("{not json", encoding="utf-8")
    store = get_store()
    assert store.pointer("current_dataset") is None
    assert (home / "state.json.migrated").exists()


def test_legacy_state_shim_maps_keys(home):
    from echotwin.storage import state

    state.set("last_import_id", "imp1")
    state.set("unit", 42)
    assert get_store().pointer("current_import") == "imp1"
    assert state.get("last_import_id") == "imp1"
    assert state.get("unit") == 42
    assert state.load()["last_import_id"] == "imp1"
