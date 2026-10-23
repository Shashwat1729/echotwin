"""SQLite workspace store: typed pointers, kv, jobs, shared by every adapter.

Design notes
------------
* WAL + ``busy_timeout`` so the CLI, API, UI and job threads can share one file.
* One connection per thread, in autocommit mode; writes use explicit
  ``BEGIN IMMEDIATE`` transactions so read-modify-write cycles cannot interleave.
* Message content is never stored here - only ids, paths and metadata.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .migrations import MIGRATIONS, POINTERS

LEGACY_KEYS = {
    "last_import_id": "current_import",
    "target_sender_id": "current_author",
    "last_dataset_id": "current_dataset",
    "last_profile_id": "current_profile",
    "last_voice_profile_id": "current_voice_profile",
}

_JOB_COLUMNS = {
    "kind",
    "status",
    "progress",
    "message",
    "created_at",
    "started_at",
    "finished_at",
    "result",
    "error",
}


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._migrate()

    # -- connections ---------------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
            self._local.depth = 0
            with self._lock:
                self._all.append(c)
        return c

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Write transaction; nested use joins the outer transaction."""
        c = self.conn()
        if self._local.depth > 0:
            self._local.depth += 1
            try:
                yield c
            finally:
                self._local.depth -= 1
            return
        c.execute("BEGIN IMMEDIATE")
        self._local.depth = 1
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")
        finally:
            self._local.depth = 0

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                with contextlib.suppress(sqlite3.Error):
                    c.close()
            self._all.clear()
        self._local = threading.local()

    # -- migrations ----------------------------------------------------------
    def _migrate(self) -> None:
        c = self.conn()
        c.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")
        with self.write():
            row = c.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] if row and row["v"] is not None else 0
            for i, fn in enumerate(MIGRATIONS[current:], start=current + 1):
                fn(c)
                c.execute("INSERT INTO schema_version(version) VALUES (?)", (i,))

    def schema_version(self) -> int:
        row = self.conn().execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return int(row["v"] or 0)

    # -- pointers ------------------------------------------------------------
    def pointer(self, name: str) -> str | None:
        _check_pointer(name)
        row = (
            self.conn()
            .execute(f"SELECT {name} AS v FROM workspace WHERE name='default'")
            .fetchone()
        )
        return row["v"] if row else None

    def set_pointer(self, name: str, value: str | None) -> None:
        _check_pointer(name)
        with self.write() as c:
            c.execute(
                f"UPDATE workspace SET {name}=? WHERE name='default'",
                (value,),
            )

    def pointers(self) -> dict[str, str | None]:
        row = self.conn().execute("SELECT * FROM workspace WHERE name='default'").fetchone()
        return {p: row[p] for p in POINTERS}

    # -- kv ------------------------------------------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.conn().execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        with self.write() as c:
            c.execute(
                "INSERT INTO kv(key, value_json) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                (key, json.dumps(value, default=str)),
            )

    def kv_all(self) -> dict[str, Any]:
        rows = self.conn().execute("SELECT key, value_json FROM kv").fetchall()
        return {r["key"]: json.loads(r["value_json"]) for r in rows}

    # -- jobs ----------------------------------------------------------------
    def job_insert(self, job: dict) -> None:
        with self.write() as c:
            c.execute(
                "INSERT INTO job(job_id, kind, status, progress, message, created_at, "
                "started_at, finished_at, result_json, error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    job["job_id"],
                    job["kind"],
                    job.get("status", "queued"),
                    float(job.get("progress", 0.0)),
                    job.get("message", ""),
                    job["created_at"],
                    job.get("started_at"),
                    job.get("finished_at"),
                    json.dumps(job["result"], default=str)
                    if job.get("result") is not None
                    else None,
                    job.get("error"),
                ),
            )

    def job_update(self, job_id: str, **fields: Any) -> None:
        bad = set(fields) - _JOB_COLUMNS
        if bad:
            raise ValueError(f"unknown job fields: {sorted(bad)}")
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            if k == "result":
                cols.append("result_json=?")
                vals.append(json.dumps(v, default=str) if v is not None else None)
            else:
                cols.append(f"{k}=?")
                vals.append(v)
        with self.write() as c:
            c.execute(f"UPDATE job SET {', '.join(cols)} WHERE job_id=?", (*vals, job_id))

    def job_get(self, job_id: str) -> dict | None:
        row = self.conn().execute("SELECT * FROM job WHERE job_id=?", (job_id,)).fetchone()
        return _job_dict(row) if row else None

    def job_list(self) -> list[dict]:
        rows = self.conn().execute("SELECT * FROM job ORDER BY created_at DESC").fetchall()
        return [_job_dict(r) for r in rows]

    def fail_interrupted(self) -> int:
        """Mark jobs left over by a crashed process as failed.

        Both 'running' and 'queued' rows are terminalised: a job that was
        queued when the process died can never start, and leaving it queued
        makes it un-cancellable (and visible forever) in the UI.
        """
        with self.write() as c:
            cur = c.execute(
                "UPDATE job SET status='failed', error='interrupted' "
                "WHERE status IN ('running', 'queued')"
            )
            return cur.rowcount


def _check_pointer(name: str) -> None:
    if name not in POINTERS:
        raise ValueError(f"unknown workspace pointer {name!r}; expected one of {POINTERS}")


def _job_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    raw = d.pop("result_json", None)
    d["result"] = json.loads(raw) if raw else None
    return d
