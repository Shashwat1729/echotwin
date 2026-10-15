"""Privacy export and deletion helpers used by 'echotwin privacy' commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .settings import get_settings
from .storage import db
from .workspace import get_store, reset_store


def export_all(out_dir: Path, console) -> None:
    settings = get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # state + consents + artifacts list (never raw message bodies)
    state = (
        json.loads((settings.home / "state.json").read_text(encoding="utf-8"))
        if (settings.home / "state.json").exists()
        else {}
    )
    consents = {}
    for scope in ("whatsapp_import", "voice_cloning"):
        consents[scope] = db.get_consent(scope)
    payload = {
        "state": state,
        "consents": consents,
        "artifacts": db.list_artifacts(),
        "exports_note": "Raw messages/recordings are NOT included. Export them separately "
        "from your WhatsApp app/recording device if needed.",
    }
    (out_dir / "echotwin-state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # voice profiles (metadata + optionally copies)
    profiles_src = settings.artifacts_dir / "voice_profiles"
    if profiles_src.exists():
        shutil.copytree(profiles_src, out_dir / "voice_profiles", dirs_exist_ok=True)

    # datasets (user content: their own words)
    datasets_src = settings.artifacts_dir / "datasets"
    if datasets_src.exists():
        shutil.copytree(datasets_src, out_dir / "datasets", dirs_exist_ok=True)

    console.print(f"[green]Export written[/green] to {out_dir}")


def delete_scope(scope: str, console) -> None:
    settings = get_settings()
    if scope.startswith("profile:"):
        profile_id = scope.split(":", 1)[1]
        from .voice.metadata import delete_profile

        result = delete_profile(profile_id, delete_recordings=False)
        console.print(json.dumps(result, indent=1))
        return
    if scope == "artifacts:voice":
        n = db.delete_artifacts_of_kind("voice_profile", delete_files=True)
        prof_dir = settings.artifacts_dir / "voice_profiles"
        if prof_dir.exists():
            shutil.rmtree(prof_dir, ignore_errors=True)
        console.print(f"deleted {n} voice artifacts + profiles")
        return
    if scope == "artifacts:datasets":
        ds_dir = settings.artifacts_dir / "datasets"
        if ds_dir.exists():
            shutil.rmtree(ds_dir, ignore_errors=True)
        (settings.home / "state.json").unlink(missing_ok=True)
        store = get_store()
        for pointer in ("current_dataset", "current_profile"):
            store.set_pointer(pointer, None)
        console.print("deleted datasets and workspace pointers")
        return
    if scope == "all-state":
        shutil.rmtree(settings.artifacts_dir, ignore_errors=True)
        reset_store()  # close connections so the WAL files can be removed
        for suffix in ("", "-wal", "-shm"):
            (settings.home / f"echotwin.sqlite3{suffix}").unlink(missing_ok=True)
        (settings.home / "state.json").unlink(missing_ok=True)
        console.print(
            "deleted artifacts, database and CLI state. Recordings under "
            f"{settings.recordings_dir} are untouched — delete them manually if needed."
        )
        console.print(
            "[yellow]Note: deleting artifacts does NOT remove learned "
            "information from trained adapters; retrain from sanitized data.[/yellow]"
        )
        return
    raise ValueError(
        f"Unknown scope '{scope}'. Use profile:<id>, artifacts:voice, artifacts:datasets, all-state."
    )
