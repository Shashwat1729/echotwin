"""Data-plane routes: imports, participants, dataset, profile, artifacts, consent."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, UploadFile

from ..exceptions import ArtifactNotFoundError, DatasetError, EchoTwinError
from ..profiling.style_profile import StyleProfile
from ..services import datasets, imports, jobs, profiles
from ..storage import db
from ..workspace import get_store
from . import service
from .schemas import (
    ConsentRequest,
    DatasetPrepareRequest,
    ImportResponse,
    TargetAuthorRequest,
)

router = APIRouter(tags=["data"])


@router.post("/imports/upload", response_model=ImportResponse)
async def import_upload(file: UploadFile = File(...)) -> ImportResponse:
    data = await file.read()
    target = service.save_upload(file.filename or "upload.zip", data)
    result = imports.import_export(target, "conversation-1")
    return ImportResponse(
        import_id=result.import_id,
        messages=result.n_messages,
        formats=result.conversations,
        participants=result.participants,
    )


@router.get("/imports/{import_id}/participants")
def participants(import_id: str) -> dict:
    messages, _ = imports.load_messages(import_id)
    return {"import_id": import_id, "participants": imports.participants_of(messages)}


@router.post("/imports/{import_id}/target-author")
def select_target(import_id: str, req: TargetAuthorRequest) -> dict:
    sid = imports.select_author(req.name_or_id, import_id)
    return {"target_sender_id": sid, "name": req.name_or_id}


@router.post("/dataset/prepare")
def dataset_prepare(req: DatasetPrepareRequest) -> dict:
    result = datasets.prepare(
        req.context_messages,
        req.max_context_chars,
        req.split_mode,
        min_target_chars=req.min_target_chars,
        max_target_chars=req.max_target_chars,
        test_fraction=req.test_fraction,
        val_fraction=req.val_fraction,
    )
    return {
        "dataset_id": result.dataset_id,
        "examples": result.n_examples,
        "splits": {"train": result.n_train, "val": result.n_val, "test": result.n_test},
        "warnings": result.warnings,
    }


@router.get("/dataset/stats")
def dataset_stats() -> dict:
    return datasets.manifest()


def _profile_path():
    path = profiles.current_path()
    if path is None:
        raise ArtifactNotFoundError("No profile yet. Build it first.")
    return path


@router.get("/profile")
def get_profile() -> dict:
    return json.loads(_profile_path().read_text(encoding="utf-8"))


@router.post("/profile/build")
def build_profile() -> dict:
    if not get_store().pointer("current_author"):
        raise DatasetError("Choose the author to clone before building a profile.")

    def _work(_ctx: jobs.JobContext) -> dict:
        result = profiles.build()
        return {"profile_path": str(result.path), "messages": result.profile.n_messages}

    return jobs.submit("profile", _work)


@router.put("/profile")
def edit_profile(payload: dict) -> dict:
    """Edit the style profile JSON (e.g. editable_notes)."""
    path = _profile_path()
    profile = StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))
    allowed = {"editable_notes", "notes", "name"}
    data = {k: v for k, v in payload.items() if k in allowed}
    if not data:
        raise EchoTwinError(f"Only {sorted(allowed)} are editable.")
    profile = profile.model_copy(update={**data, "edited_by_user": True})
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return {"edited": True, "fields": sorted(data)}


@router.get("/artifacts")
def list_artifacts(kind: str | None = None) -> dict:
    return {"artifacts": db.list_artifacts(kind)}


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str) -> dict:
    try:
        return db.delete_artifact(artifact_id)
    except EchoTwinError as e:
        raise ArtifactNotFoundError(str(e)) from e


@router.post("/consents")
def consent(req: ConsentRequest) -> dict:
    db.record_consent(req.scope, req.granted, req.details)
    return db.get_consent(req.scope)
