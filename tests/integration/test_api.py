"""Integration: API endpoints with the FastAPI TestClient."""

from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from echotwin.api.app import app

client = TestClient(app)

SYNTH_CHAT = (
    '[10/01/2025, 20:00:02] Rohit created group "Test"\n'
    "[10/01/2025, 20:01:00] Priya: hello everyone ☀️\n"
    "[10/01/2025, 20:02:00] Aarav: haan bro kaise ho\n"
    "[10/01/2025, 20:03:00] Aarav: main thoda busy hoon aaj\n"
    "[10/01/2025, 20:04:00] Priya: theek hai\n"
    "[10/01/2025, 20:05:00] Aarav: kal milte h, 6 bje library\n"
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["host_binding"] == "127.0.0.1"


def test_capabilities():
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert any(b["name"] == "voxcpm" for b in body["voice_backends"])


def test_import_upload_txt():
    files = {"file": ("chat.txt", io.BytesIO(SYNTH_CHAT.encode("utf-8")), "text/plain")}
    r = client.post("/api/v1/imports/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["messages"] >= 6
    names = {p["sender_name"] for p in body["participants"]}
    assert {"Aarav", "Priya"} <= names


def test_import_upload_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("WhatsApp Chat with Test/_chat.txt", SYNTH_CHAT)
    files = {"file": ("export.zip", buf.getvalue(), "application/zip")}
    r = client.post("/api/v1/imports/upload", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["messages"] >= 6


def test_import_rejects_bad_extension():
    files = {"file": ("evil.exe", b"MZ", "application/octet-stream")}
    r = client.post("/api/v1/imports/upload", files=files)
    assert r.status_code in (400, 422)


def test_participants_and_target_flow():
    files = {"file": ("chat.txt", io.BytesIO(SYNTH_CHAT.encode("utf-8")), "text/plain")}
    imp = client.post("/api/v1/imports/upload", files=files).json()
    import_id = imp["import_id"]
    r = client.get(f"/api/v1/imports/{import_id}/participants")
    assert r.status_code == 200
    assert r.json()["participants"]
    r = client.post(
        f"/api/v1/imports/{import_id}/target-author",
        json={"name_or_id": "Aarav"},
    )
    assert r.status_code == 200
    assert r.json()["target_sender_id"]


def test_dataset_and_profile_and_generate():
    files = {"file": ("chat.txt", io.BytesIO(SYNTH_CHAT.encode("utf-8")), "text/plain")}
    imp = client.post("/api/v1/imports/upload", files=files).json()
    import_id = imp["import_id"]
    client.post(f"/api/v1/imports/{import_id}/target-author", json={"name_or_id": "Aarav"})
    r = client.post("/api/v1/dataset/prepare", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["examples"] >= 1
    assert set(body["splits"]) == {"train", "val", "test"}
    r = client.get("/api/v1/dataset/stats")
    assert r.status_code == 200
    r = client.post("/api/v1/profile/build")
    assert r.status_code == 200
    assert r.json()["status"] in {"queued", "running", "completed"}
    r = client.post(
        "/api/v1/generate",
        json={"context": [{"role": "user", "content": "hello?"}], "model": ""},
    )
    assert r.status_code == 200
    assert r.json()["generated"]
    assert r.json()["model_revision"] == "mock-1"


def test_voice_plan_route():
    r = client.post("/api/v1/voice/plans", json={"backend": "voxcpm", "languages": ["hi", "en"]})
    assert r.status_code == 200
    assert r.json()["tasks"]


def test_voice_scripts_route():
    r = client.get("/api/v1/voice/scripts", params={"language": "hinglish"})
    assert r.status_code == 200
    assert r.json()["scripts"]


def test_voice_profile_crud():
    r = client.post("/api/v1/voice/profiles", json={"name": "api-test", "backend": "voxcpm"})
    assert r.status_code == 200
    pid = r.json()["profile_id"]
    r = client.get("/api/v1/voice/profiles")
    assert any(p["profile_id"] == pid for p in r.json()["profiles"])
    r = client.delete(f"/api/v1/voice/profiles/{pid}")
    assert r.status_code == 200


def test_consent_route():
    r = client.post("/api/v1/consents", json={"scope": "unit_voice", "granted": True})
    assert r.status_code == 200
    assert r.json()["granted"] is True


def test_artifacts_route():
    r = client.get("/api/v1/artifacts")
    assert r.status_code == 200


def test_synthesize_without_consent_blocked():
    r = client.post("/api/v1/voice/profiles", json={"name": "nocon", "backend": "voxcpm"})
    pid = r.json()["profile_id"]
    r = client.post("/api/v1/voice/synthesize", json={"profile_id": pid, "text": "test"})
    # job is created (async) but will fail on consent; check job status
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    import time

    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "failed"
    assert (
        "consent" in (job["error"] or "").lower() or "authorization" in (job["error"] or "").lower()
    )
