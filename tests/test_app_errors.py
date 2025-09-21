import asyncio
import os
from typing import List

import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("client")
def test_upload_invalid_format(client: TestClient):
    files = {
        "file": ("readme.txt", b"hello", "text/plain"),
    }
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
    assert "Invalid file format" in r.json()["detail"]


@pytest.mark.usefixtures("client")
def test_transcribe_upload_not_found(client: TestClient):
    r = client.post("/api/transcribe", json={"upload_id": "does-not-exist"})
    assert r.status_code == 404


@pytest.mark.usefixtures("client")
def test_get_job_not_found(client: TestClient):
    r = client.get("/api/jobs/unknown")
    assert r.status_code == 404


@pytest.mark.usefixtures("client")
def test_download_not_completed(client: TestClient):
    # Inject a job that is not completed
    from src.app import main as app_main

    job_id = "job-not-completed"
    app_main.jobs_store[job_id] = {
        "status": "processing",
    }

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 400


@pytest.mark.usefixtures("client")
def test_download_missing_midi(client: TestClient):
    # Inject a completed job but no MIDI in store
    from src.app import main as app_main

    job_id = "job-done-no-midi"
    app_main.jobs_store[job_id] = {
        "status": "completed",
    }

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 404


@pytest.mark.usefixtures("client")
def test_list_jobs_pagination(client: TestClient, monkeypatch):
    from src.app import main as app_main

    async def noop_process(job_id: str, file_path: str):  # noqa: ARG001 - signature must match
        # Do nothing (avoid heavy dependencies)
        return None

    # Ensure background task is a no-op so tests remain fast
    monkeypatch.setattr(app_main, "process_audio_task", noop_process, raising=True)

    created_job_ids: List[str] = []

    # Create three jobs via upload + transcribe
    for i in range(3):
        files = {"file": (f"t{i}.wav", b"\x00\x00\x00\x00", "audio/wav")}
        r_up = client.post("/api/upload", files=files)
        assert r_up.status_code == 200
        upload_id = r_up.json()["upload_id"]

        r_tr = client.post("/api/transcribe", json={"upload_id": upload_id})
        assert r_tr.status_code == 200
        created_job_ids.append(r_tr.json()["job_id"])

    # Paginate: limit=2, offset=1 -> exactly two jobs returned
    r_list = client.get("/api/jobs", params={"limit": 2, "offset": 1})
    assert r_list.status_code == 200
    payload = r_list.json()
    assert payload["total"] == 3
    assert len(payload["jobs"]) == 2
    assert payload["limit"] == 2
    assert payload["offset"] == 1


@pytest.mark.usefixtures("client")
def test_process_audio_task_error_flow(tmp_path, client: TestClient):  # noqa: ARG001
    """Ensure process_audio_task sets job to failed and cleans up file on exception."""
    from datetime import datetime

    from src.app import main as app_main

    # Prepare a temp file
    wav_path = tmp_path / "bad.wav"
    wav_path.write_bytes(b"\x00\x00\x00\x00")

    job_id = "boom-job"
    app_main.jobs_store[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "result_url": None,
        "error": None,
    }

    class BoomTranscriber:
        async def transcribe(self, file_path: str, job_id: str, jobs_store):  # noqa: D401, ARG002
            raise RuntimeError("boom")

    # Inject failing transcriber to avoid importing heavy module
    app_main.app.state.transcriber = BoomTranscriber()

    # Run the task and verify failure path
    asyncio.run(app_main.process_audio_task(job_id, str(wav_path)))

    assert app_main.jobs_store[job_id]["status"] == "failed"
    assert "boom" in app_main.jobs_store[job_id]["error"]
    assert not os.path.exists(str(wav_path))
