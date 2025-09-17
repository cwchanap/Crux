import asyncio
import os
import struct

from fastapi.testclient import TestClient


def _minimal_midi() -> bytes:
    # Build a minimal valid MIDI file with one empty track
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 96)
    # Track with only End-of-Track meta event
    track_data = b"\x00\xff\x2f\x00"
    track = b"MTrk" + struct.pack(">I", len(track_data)) + track_data
    return header + track


def test_root_ok(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Drum Transcription" in resp.text


def test_upload_transcribe_download_flow(client: TestClient, monkeypatch):
    # Import app module to access stores and functions
    from src.app import main as app_main

    async def fake_process_audio_task(job_id: str, file_path: str):
        # Simulate some progress updates
        app_main.jobs_store[job_id]["status"] = "processing"
        app_main.jobs_store[job_id]["progress"] = 50
        # Produce a minimal MIDI result
        app_main.midi_store[job_id] = _minimal_midi()
        # Mark job completed
        app_main.jobs_store[job_id]["status"] = "completed"
        app_main.jobs_store[job_id]["progress"] = 100
        app_main.jobs_store[job_id]["result_url"] = f"/api/jobs/{job_id}/download"
        # Cleanup uploaded file if exists
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    # Monkeypatch the background task to avoid heavy dependencies
    monkeypatch.setattr(app_main, "process_audio_task", fake_process_audio_task, raising=True)

    # 1) Upload a small dummy wav file
    files = {
        "file": ("test.wav", b"\x00\x00\x00\x00", "audio/wav"),
    }
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    upload_id = data["upload_id"]

    # 2) Start transcription (creates job and schedules background task)
    r2 = client.post("/api/transcribe", json={"upload_id": upload_id})
    assert r2.status_code == 200, r2.text
    job = r2.json()
    job_id = job["job_id"]

    # 3) Manually run our patched background task (deterministic for tests)
    file_path = app_main.jobs_store[job_id]["metadata"]["file_path"]
    asyncio.run(fake_process_audio_task(job_id, file_path))

    # 4) Verify job status is completed
    r3 = client.get(f"/api/jobs/{job_id}")
    assert r3.status_code == 200
    status = r3.json()
    assert status["status"] == "completed"
    assert status["result_url"].endswith(f"/api/jobs/{job_id}/download")

    # 5) Download result and verify it's MIDI
    r4 = client.get(f"/api/jobs/{job_id}/download")
    assert r4.status_code == 200
    assert r4.headers["content-type"].startswith("audio/midi")
    content = r4.content
    assert content[:4] == b"MThd"
