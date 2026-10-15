"""Ordered, idempotent schema migrations for the workspace store.

Each migration runs once, inside the transaction opened by the runner. Append
new functions to ``MIGRATIONS``; never edit or reorder existing ones.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

POINTERS = (
    "current_import",
    "current_author",
    "current_dataset",
    "current_profile",
    "current_voice_profile",
)


def _v1_initial(conn: sqlite3.Connection) -> None:
    # executescript() issues an implicit COMMIT, which would break the runner's
    # transaction, so statements are executed one by one.
    statements = [
        """CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS consents (
            consent_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            granted INTEGER NOT NULL,
            granted_at TEXT,
            details_json TEXT DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS imports (
            import_id TEXT PRIMARY KEY,
            source_path TEXT,
            conversation_id TEXT,
            created_at TEXT NOT NULL,
            parsed INTEGER,
            target_sender_id TEXT,
            stats_json TEXT DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT,
            created_at TEXT NOT NULL,
            manifest_path TEXT,
            metadata_json TEXT DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS workspace (
            name TEXT PRIMARY KEY,
            current_import TEXT,
            current_author TEXT,
            current_dataset TEXT,
            current_profile TEXT,
            current_voice_profile TEXT
        )""",
        "INSERT OR IGNORE INTO workspace(name) VALUES ('default')",
        """CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS job (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            result_json TEXT,
            error TEXT
        )""",
    ]
    for stmt in statements:
        conn.execute(stmt)


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_v1_initial]
