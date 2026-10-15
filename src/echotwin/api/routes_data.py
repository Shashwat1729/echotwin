"""Data-plane routes: imports, participants, dataset, profile, artifacts, consent."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..exceptions import EchoTwinError
from ..ingestion import WhatsAppParser, load_export, participant_stats, pick_participant
from ..storage import db, state
from . import service
from .schemas import (
    ConsentRequest,
    DatasetPrepareRequest,
    ImportResponse,
    TargetAuthorRequest,
)

router = APIRouter(tags=["data"])


def _parse_import(path: Path, conversation_prefix: str) -> ImportResponse:
    from ..ingestion.export_loader import conversation_id_from_name
    from ..ingestion.schema import Message

    bundle = load_export(path)
    parser = WhatsAppParser()
    all_messages: list[Message] = []
    conversations: list[str] = []
    totals = {
        "total_lines": 0,
        "parsed_messages": 0,
        "system_messages": 0,
        "attachment_messages": 0,
        "deleted_messages": 0,
        "malformed_lines": 0,
    }
    for i, chat in enumerate(bundle.chats):
        convo_id = conversation_id_from_name(chat.inner_name, f"{conversation_prefix}-{i}")
        try:
            result = parser.parse_text(chat.chat_text, convo_id)
        except EchoTwinError:
            continue
        all_messages.extend(result.messages)
        conversations.append(convo_id)
        rep = result.report
        totals["total_lines"] += rep.total_lines
        totals["parsed_messages"] += rep.parsed_messages
        totals["system_messages"] += rep.system_messages
        totals["attachment_messages"] += rep.attachment_messages
        totals["deleted_messages"] += rep.deleted_messages
        totals["malformed_lines"] += rep.malformed_lines
    if not all_messages:
        raise HTTPException(400, "No messages parsed from the export")
    import_id = db.record_import(str(path), conversation_prefix, len(all_messages), None, totals)
    state.set("last_import_id", import_id)
    out_dir = get_settings_artifacts() / "imports" / import_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "messages.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for m in all_messages:
            fh.write(m.model_dump_json() + "\n")
    participants = [
        {
            "sender_id": s.sender_id,
            "sender_name": s.sender_name,
            "message_count": s.message_count,
            "first_timestamp": s.first_timestamp.isoformat() if s.first_timestamp else None,
            "last_timestamp": s.last_timestamp.isoformat() if s.last_timestamp else None,
            "mean_length": s.mean_length,
        }
        for s in participant_stats(all_messages)
    ]
    return ImportResponse(
        import_id=import_id,
        messages=len(all_messages),
        formats=conversations,
        participants=participants,
    )


def get_settings_artifacts() -> Path:
    from ..settings import get_settings

    return get_settings().artifacts_dir


@router.post("/imports/upload", response_model=ImportResponse)
async def import_upload(file: UploadFile = File(...)) -> ImportResponse:
    try:
        data = await file.read()
        target = service.save_upload(file.filename or "upload.zip", data)
        return _parse_import(target, "conversation-1")
    except EchoTwinError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/imports/{import_id}/participants")
def participants(import_id: str) -> dict:
    from ..ingestion.schema import Message

    imp = db.latest_import()
    if not imp or imp["import_id"] != import_id:
        raise HTTPException(404, "import not found")
    messages_path = get_settings_artifacts() / "imports" / import_id / "messages.jsonl"
    if not messages_path.exists():
        raise HTTPException(404, "stored messages missing for this import")
    messages = [
        Message.model_validate_json(line)
        for line in messages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = [
        {
            "sender_id": s.sender_id,
            "sender_name": s.sender_name,
            "message_count": s.message_count,
            "first_timestamp": s.first_timestamp.isoformat() if s.first_timestamp else None,
            "last_timestamp": s.last_timestamp.isoformat() if s.last_timestamp else None,
            "mean_length": s.mean_length,
        }
        for s in participant_stats(messages)
    ]
    return {"import_id": import_id, "participants": stats}


@router.post("/imports/{import_id}/target-author")
def select_target(import_id: str, req: TargetAuthorRequest) -> dict:
    from ..ingestion.schema import Message

    imp = db.latest_import()
    if not imp or imp["import_id"] != import_id:
        raise HTTPException(404, "import not found")
    messages_path = get_settings_artifacts() / "imports" / import_id / "messages.jsonl"
    if not messages_path.exists():
        raise HTTPException(404, "stored messages missing for this import")
    messages = [
        Message.model_validate_json(line)
        for line in messages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sid = pick_participant(messages, req.name_or_id)
    if sid is None:
        raise HTTPException(404, f"participant '{req.name_or_id}' not found")
    db.update_import_target(import_id, sid)
    state.set("target_sender_id", sid)
    state.set("last_import_id", import_id)
    return {"target_sender_id": sid, "name": req.name_or_id}


@router.post("/dataset/prepare")
def dataset_prepare(req: DatasetPrepareRequest) -> dict:
    from ..dataset.builder import DatasetConfig, build_examples, save_jsonl
    from ..dataset.splits import SplitConfig, make_manifest, split_dataset
    from ..ingestion.schema import Message
    from ..settings import get_settings

    import_id = state.get("last_import_id")
    target = state.get("target_sender_id")
    if not import_id or not target:
        raise HTTPException(400, "import and target author must be set first")
    messages_path = get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"
    if not messages_path.exists():
        raise HTTPException(404, "stored messages missing for this import")
    messages = [
        Message.model_validate_json(line)
        for line in messages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    built = build_examples(
        messages,
        target,
        DatasetConfig(
            context_messages=req.context_messages,
            max_context_chars=req.max_context_chars,
            min_target_chars=req.min_target_chars,
            max_target_chars=req.max_target_chars,
        ),
    )
    split_cfg = SplitConfig(
        mode=req.split_mode, test_fraction=req.test_fraction, val_fraction=req.val_fraction
    )
    split = split_dataset(
        built,
        split_cfg,
    )
    ds_id = f"dataset-{import_id}"
    ds_dir = get_settings().artifacts_dir / "datasets" / ds_id
    save_jsonl(built.examples, ds_dir / "examples.jsonl")
    save_jsonl(split.train, ds_dir / "train.jsonl")
    save_jsonl(split.val, ds_dir / "val.jsonl")
    save_jsonl(split.test, ds_dir / "test.jsonl")
    manifest = make_manifest(
        dataset_id=ds_id,
        dataset_path=ds_dir / "examples.jsonl",
        split=split,
        built=built,
        target_sender_id=target,
        split_cfg=split_cfg,
        echotwin_version="0.1.0",
    )
    (ds_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    state.set("last_dataset_id", ds_id)
    return {
        "dataset_id": ds_id,
        "examples": len(built.examples),
        "splits": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "warnings": split.report.warnings,
    }


@router.get("/dataset/stats")
def dataset_stats() -> dict:
    import json

    ds_id = state.get("last_dataset_id")
    if not ds_id:
        raise HTTPException(400, "no dataset")
    ds_dir = get_settings_artifacts() / "datasets" / ds_id
    manifest = ds_dir / "manifest.json"
    if not manifest.exists():
        raise HTTPException(404, "manifest missing")
    return json.loads(manifest.read_text(encoding="utf-8"))


@router.get("/profile")
def get_profile() -> dict:
    p = state.get("last_profile_id")
    if not p or not Path(p).exists():
        raise HTTPException(404, "no profile; build it first")
    return json.loads(Path(p).read_text(encoding="utf-8"))


@router.post("/profile/build")
def build_profile() -> dict:
    from ..profiling.style_profile import save_profile
    from ..settings import get_settings
    from .service import JOBS

    import_id = state.get("last_import_id")
    target = state.get("target_sender_id")

    def _work() -> dict:
        from ..ingestion.schema import Message
        from ..profiling.style_profile import build_style_profile

        messages_path = get_settings().artifacts_dir / "imports" / import_id / "messages.jsonl"
        if not messages_path.exists():
            raise HTTPException(404, "stored messages missing for this import")
        messages = [
            Message.model_validate_json(line)
            for line in messages_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        profile = build_style_profile(messages, target)
        out = get_settings().artifacts_dir / "profiles" / f"profile-{target[:8]}.json"
        save_profile(profile, out)
        state.set("last_profile_id", str(out))
        return {"profile_path": str(out), "messages": profile.n_messages}

    if not import_id or not target:
        raise HTTPException(400, "set target author first")
    job = JOBS.submit("profile", _work)
    return job.model_dump()


@router.put("/profile")
def edit_profile(payload: dict) -> dict:
    """Edit the style profile JSON (e.g. editable_notes)."""

    p = state.get("last_profile_id")
    if not p or not Path(p).exists():
        raise HTTPException(404, "no profile")
    from ..profiling.style_profile import StyleProfile

    profile = StyleProfile.model_validate_json(Path(p).read_text(encoding="utf-8"))
    allowed = {"editable_notes", "notes", "name"}
    data = {k: v for k, v in payload.items() if k in allowed}
    if not data:
        raise HTTPException(400, f"only {sorted(allowed)} are editable in v0.1")
    profile = profile.model_copy(update={**data, "edited_by_user": True})
    Path(p).write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return {"edited": True, "fields": sorted(data)}


@router.get("/artifacts")
def list_artifacts(kind: str | None = None) -> dict:
    return {"artifacts": db.list_artifacts(kind)}


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str) -> dict:
    try:
        return db.delete_artifact(artifact_id)
    except EchoTwinError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/consents")
def consent(req: ConsentRequest) -> dict:
    db.record_consent(req.scope, req.granted, req.details)
    return db.get_consent(req.scope)
