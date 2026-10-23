"""Privacy export and deletion helpers used by 'echotwin privacy' commands.

Two rules govern this module:

1. **A delete scope must delete what it claims.** The UI labels
   ``artifacts:voice`` as "Voice recordings and profiles", so it removes the
   recordings too. ``all-state`` removes the raw uploads as well. Anything a
   scope deliberately leaves behind is *printed*, so the user is never misled
   about what still exists on disk.
2. **The export must be honest.** It contains the user's own words (the
   datasets), so the note says exactly that instead of claiming raw messages
   are absent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .exceptions import ConfigurationError
from .settings import get_settings
from .storage import db
from .workspace import get_store, reset_store


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def export_all(out_dir: Path, console) -> None:
    settings = get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = get_store()
    consents = {}
    for scope in ("whatsapp_import", "voice_cloning"):
        consents[scope] = db.get_consent(scope)
    payload = {
        "pointers": store.pointers(),
        "workspace": store.kv_all(),
        "consents": consents,
        "artifacts": db.list_artifacts(),
        "contents_note": (
            "This export DOES contain your own words: the datasets below are the "
            "training examples built from your messages. Raw WhatsApp archives and "
            "voice recordings are not copied here - export them from your phone or "
            "recorder if you need them."
        ),
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
    console.print("[dim]Contains your training datasets (your own words).[/dim]")


def delete_scope(scope: str, console) -> None:
    settings = get_settings()
    store = get_store()

    if scope.startswith("profile:"):
        profile_id = scope.split(":", 1)[1]
        from .voice.metadata import delete_profile

        result = delete_profile(profile_id, delete_recordings=False)
        console.print(json.dumps(result, indent=1))
        console.print(
            "[dim]Source recordings were kept (delete_recordings=False). "
            "Use 'artifacts:voice' to remove them.[/dim]"
        )
        return

    if scope == "artifacts:voice":
        n = db.delete_artifacts_of_kind("voice_profile", delete_files=True)
        _remove_tree(settings.artifacts_dir / "voice_profiles")
        _remove_tree(settings.artifacts_dir / "audio")
        # Recordings live outside artifacts/ - the UI promises they go too.
        recordings = settings.recordings_dir
        removed_recordings = _remove_tree(recordings)
        store.set_pointer("current_voice_profile", None)
        console.print(f"deleted {n} voice artifacts, voice profiles and generated audio")
        if removed_recordings:
            console.print(f"deleted recordings under {recordings}")
        else:
            console.print(f"[dim]no recordings found under {recordings}[/dim]")
        console.print(
            "[yellow]Note: deleting artifacts does NOT remove learned information "
            "from trained adapters; retrain from sanitized data.[/yellow]"
        )
        return

    if scope == "artifacts:datasets":
        _remove_tree(settings.artifacts_dir / "datasets")
        # Parsed messages live under imports/ - they are the raw conversations.
        _remove_tree(settings.artifacts_dir / "imports")
        (settings.home / "state.json").unlink(missing_ok=True)
        for pointer in ("current_dataset", "current_profile", "current_author", "current_import"):
            store.set_pointer(pointer, None)
        console.print("deleted datasets, parsed message imports and workspace pointers")
        return

    if scope == "all-state":
        _remove_tree(settings.artifacts_dir)
        reset_store()  # close connections so the WAL files can be removed
        for suffix in ("", "-wal", "-shm"):
            (settings.home / f"echotwin.sqlite3{suffix}").unlink(missing_ok=True)
        (settings.home / "state.json").unlink(missing_ok=True)
        # The raw uploads are the user's private conversations - they must go too.
        exports_removed = _remove_tree(settings.exports_dir)
        recordings_removed = _remove_tree(settings.recordings_dir)
        console.print("deleted artifacts, database, CLI state and raw uploads")
        if exports_removed:
            console.print(f"deleted raw exports under {settings.exports_dir}")
        if recordings_removed:
            console.print(f"deleted recordings under {settings.recordings_dir}")
        console.print(
            "[yellow]Note: deleting artifacts does NOT remove learned information "
            "from trained adapters; retrain from sanitized data.[/yellow]"
        )
        return

    raise ConfigurationError(
        f"Unknown scope '{scope}'. Use profile:<id>, artifacts:voice, "
        "artifacts:datasets, all-state."
    )
