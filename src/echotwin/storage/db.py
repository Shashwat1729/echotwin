"""Local SQLite-backed state: artifacts, consents, imports, runs.

SQLite via stdlib only. Raw message content is never written here â€” only
metadata, pseudonymous ids and paths, so the database itself is low-sensitivity
(though it still lives under the user's private ECHOTWIN_HOME).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..exceptions import ArtifactNotFoundError
from ..settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS consents (
    consent_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    granted INTEGER NOT NULL,
    granted_at TEXT,
    details_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS imports (
    import_id TEXT PRIMARY KEY,
    source_path TEXT,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    parsed INTEGER,
    target_sender_id TEXT,
    stats_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT,
    created_at TEXT NOT NULL,
    manifest_path TEXT,
    metadata_json TEXT DEFAULT '{}'
);
"""


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def get_connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def record_artifact(kind: str, path: Path, metadata: dict | None = None) -> str:
    artifact_id = uuid.uuid4().hex[:16]
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO artifacts(artifact_id, kind, path, created_at, metadata_json) VALUES(?,?,?,?,?)",
            (artifact_id, kind, str(path), _now(), json.dumps(metadata, default=str)),
        )
    return artifact_id


def list_artifacts(kind: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if kind:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE kind=? ORDER BY created_at DESC", (kind,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        d["exists"] = Path(d["path"]).exists()
        out.append(d)
    return out


def delete_artifact(artifact_id: str, *, delete_files: bool = True) -> dict:
    """Permanently delete an artifact record and (optionally) its file.

    Deleting files does NOT guarantee information removal from trained model
    weights â€” retraining from sanitized data may be required (documented).
    """
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"No artifact {artifact_id}")
        path = Path(row["path"])
        removed_file = False
        if delete_files and path.exists():
            path.unlink()
            removed_file = True
        conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,))
    return {"artifact_id": artifact_id, "path": str(path), "file_deleted": removed_file}


def delete_artifacts_of_kind(kind: str, *, delete_files: bool = True) -> int:
    n = 0
    for art in list_artifacts(kind):
        try:
            delete_artifact(art["artifact_id"], delete_files=delete_files)
            n += 1
        except ArtifactNotFoundError:
            continue
    return n


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------
def record_consent(scope: str, granted: bool, details: dict | None = None) -> str:
    consent_id = f"consent-{scope}"
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO consents(consent_id, scope, granted, granted_at, details_json) VALUES(?,?,?,?,?)",
            (
                consent_id,
                scope,
                int(granted),
                _now() if granted else None,
                json.dumps(details, default=str),
            ),
        )
    return consent_id


def get_consent(scope: str) -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM consents WHERE scope=?", (scope,)).fetchone()
    if row is None:
        return {"scope": scope, "granted": False, "granted_at": None, "exists": False}
    d = dict(row)
    d["granted"] = bool(d["granted"])
    d["details"] = json.loads(d.pop("details_json") or "{}")
    d["exists"] = True
    return d


def require_consent(scope: str) -> None:
    from ..exceptions import ConsentRequiredError

    c = get_consent(scope)
    if not c["granted"]:
        raise ConsentRequiredError(
            f"Consent '{scope}' has not been granted. See docs/privacy-and-security.md."
        )


# ---------------------------------------------------------------------------
# Imports & runs
# ---------------------------------------------------------------------------
def record_import(
    source_path: str, conversation_id: str, parsed: int, target_sender_id: str | None, stats: dict
) -> str:
    import_id = uuid.uuid4().hex[:12]
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO imports(import_id, source_path, conversation_id, created_at, parsed, target_sender_id, stats_json) VALUES(?,?,?,?,?,?,?)",
            (
                import_id,
                source_path,
                conversation_id,
                _now(),
                parsed,
                target_sender_id,
                json.dumps(stats, default=str),
            ),
        )
    return import_id


def update_import_target(import_id: str, target_sender_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE imports SET target_sender_id=? WHERE import_id=?",
            (target_sender_id, import_id),
        )
        return cur.rowcount > 0


def latest_import() -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM imports ORDER BY created_at DESC LIMIT 1").fetchone()
    if row is None:
        return None
    d = dict(row)
    d["stats"] = json.loads(d.pop("stats_json") or "{}")
    return d


def record_run(run_id: str, kind: str, status: str, manifest_path: str | None, metadata: dict):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, kind, status, created_at, manifest_path, metadata_json) VALUES(?,?,?,?,?,?)",
            (run_id, kind, status, _now(), manifest_path, json.dumps(metadata, default=str)),
        )


def list_runs(kind: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if kind:
            rows = conn.execute(
                "SELECT * FROM runs WHERE kind=? ORDER BY created_at DESC", (kind,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        out.append(d)
    return out
