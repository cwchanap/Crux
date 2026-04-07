import asyncio
import builtins
import os
import struct
from pathlib import Path

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
        "file": ("test.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
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


def test_cors_rejects_unknown_origin(client: TestClient):
    resp = client.get("/api/jobs", headers={"Origin": "https://evil.com"})
    # Unknown origins should not receive the CORS allow-origin header
    assert "access-control-allow-origin" not in resp.headers


def test_cors_allows_known_origin(client: TestClient):
    from src.app import main as app_main

    origin = (
        os.getenv("CORS_ALLOWED_ORIGINS", ",".join(app_main.ALLOWED_ORIGINS)).split(",")[0].strip()
    )
    if not origin:
        origin = "http://localhost:4330"

    resp = client.get("/api/jobs", headers={"Origin": origin})
    assert resp.headers.get("access-control-allow-origin") == origin


def test_upload_rejects_oversized_file(client: TestClient):
    big_content = b"\x00" * (51 * 1024 * 1024)  # 51 MB
    files = {"file": ("big.wav", big_content, "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 413


def test_upload_rejects_oversized_file_after_closing_temp_file(client: TestClient, monkeypatch):
    from src.app import main as app_main

    monkeypatch.setattr(app_main, "MAX_UPLOAD_BYTES", 8, raising=True)

    real_open = builtins.open
    real_unlink = Path.unlink
    temp_file_closed = {"value": False}

    class TrackingWriter:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                return self._wrapped.__exit__(exc_type, exc, tb)
            finally:
                temp_file_closed["value"] = True

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def tracking_open(file, mode="r", *args, **kwargs):
        wrapped = real_open(file, mode, *args, **kwargs)
        if "temp_uploads" in str(file) and mode == "wb":
            return TrackingWriter(wrapped)
        return wrapped

    def guarded_unlink(self, *args, **kwargs):
        assert temp_file_closed["value"], "temp file must be closed before unlink"
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    files = {"file": ("big.wav", b"012345678", "audio/wav")}
    resp = client.post("/api/upload", files=files)

    assert resp.status_code == 413


def test_upload_rejects_wrong_extension(client: TestClient):
    files = {"file": ("malware.exe", b"\x00\x01", "application/octet-stream")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400


def test_upload_rejects_disguised_file(client: TestClient):
    # .wav extension but PE magic bytes
    pe_magic = b"MZ\x90\x00" + b"\x00" * 8
    files = {"file": ("fake.wav", pe_magic, "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400


def test_upload_rejects_disguised_m4a(client: TestClient):
    # File with .m4a extension but non-audio content (no ftyp box at offset 4)
    bad_content = b"\x00\x00\x00\xfe" + b"\x00" * 8
    files = {"file": ("fake.m4a", bad_content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400


def test_upload_accepts_mpeg25_mp3(client: TestClient):
    mp3_header = b"\xff\xe3\x18\xc4" + b"\x00" * 8
    files = {"file": ("voice.mp3", mp3_header, "audio/mpeg")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 200, resp.text


def test_upload_rejects_missing_filename(client: TestClient):
    # UploadFile.filename can be None/empty; the server must reject with a 4xx.
    # An empty filename causes the multipart parser to strip the filename field,
    # so FastAPI may return 422 (form validation) or our handler returns 400.
    files = {"file": ("", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code in (400, 422)


def test_upload_rejects_m4a_with_generic_isom_brand(client: TestClient):
    # Build a minimal ftyp box with major_brand=b"isom" (generic MP4, not audio-only).
    # isom was removed from M4A_AUDIO_BRANDS so this should now be rejected.
    ftyp_size = (16).to_bytes(4, "big")  # 16-byte ftyp box
    ftyp_tag = b"ftyp"
    major_brand = b"isom"
    minor_version = b"\x00" * 4
    content = ftyp_size + ftyp_tag + major_brand + minor_version
    files = {"file": ("video.m4a", content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400


def test_upload_accepts_m4a_with_audio_brand(client: TestClient):
    # Build a minimal ftyp box with major_brand=b"M4A " (audio-specific brand).
    ftyp_size = (16).to_bytes(4, "big")
    ftyp_tag = b"ftyp"
    major_brand = b"M4A "
    minor_version = b"\x00" * 4
    content = ftyp_size + ftyp_tag + major_brand + minor_version
    files = {"file": ("audio.m4a", content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    # M4A with audio brand should pass format validation (status 200)
    assert resp.status_code == 200, resp.text


def test_upload_accepts_m4a_with_audio_compat_brand_when_ftyp_extends_to_eof(
    client: TestClient,
):
    content = b"\x00\x00\x00\x00ftypisom\x00\x00\x00\x00M4A "
    files = {"file": ("audio.m4a", content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 200, resp.text


def test_upload_accepts_m4a_with_audio_compat_brand_in_extended_ftyp_box(
    client: TestClient,
):
    content = (
        (1).to_bytes(4, "big") + b"ftyp" + (28).to_bytes(8, "big") + b"isom" + b"\x00" * 4 + b"M4A "
    )
    files = {"file": ("audio.m4a", content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 200, resp.text


def test_upload_closes_upload_file(client: TestClient, monkeypatch):
    from starlette.datastructures import UploadFile

    close_called = {"value": False}
    original_close = UploadFile.close

    async def tracking_close(self):
        close_called["value"] = True
        return await original_close(self)

    monkeypatch.setattr(UploadFile, "close", tracking_close)

    files = {"file": ("test.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}
    resp = client.post("/api/upload", files=files)

    assert resp.status_code == 200, resp.text
    assert close_called["value"]
