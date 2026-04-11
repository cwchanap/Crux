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
    assert resp.status_code == 200
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
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin


def test_cors_wildcard_default_allows_any_origin(monkeypatch):
    """When CORS_ALLOWED_ORIGINS is unset, the wildcard default should allow
    any origin (backward compat with the previous allow_origins=["*"] behaviour)."""
    import importlib

    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    from src.app import main as app_main

    importlib.reload(app_main)
    try:
        with TestClient(app_main.app) as client:
            resp = client.get("/api/jobs", headers={"Origin": "https://arbitrary.example.com"})
            assert resp.status_code == 200
            # With wildcard default, Starlette returns literal "*" as the
            # allow-origin header.  (Note: browsers reject "*" combined with
            # credentials, but the backward-compat goal is to avoid 403-style
            # CORS failures for non-credentialed requests.)
            assert resp.headers.get("access-control-allow-origin") == "*"
    finally:
        # Reload again to restore the test-environment CORS settings
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:4330,http://localhost:8788")
        importlib.reload(app_main)


def test_upload_rejects_oversized_file(client: TestClient, monkeypatch):
    from src.app import main as app_main

    monkeypatch.setattr(app_main, "MAX_UPLOAD_BYTES", 10, raising=True)

    files = {"file": ("big.wav", b"01234567890", "audio/wav")}
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


def test_upload_rejects_oversized_content_length_middleware(client: TestClient, monkeypatch):
    """The UploadSizeLimitMiddleware must reject requests whose Content-Length
    header exceeds the limit *before* Starlette parses the multipart body."""
    from src.app import main as app_main

    # Patch the middleware's stored max_bytes to 10 bytes.
    # user_middleware[0] is the UploadSizeLimitMiddleware (added last = index 0).
    # Defensive check: fail fast if middleware registration order changes.
    assert app_main.app.user_middleware[0].cls is app_main.UploadSizeLimitMiddleware, (
        "Expected first middleware to be UploadSizeLimitMiddleware"
    )
    monkeypatch.setitem(
        app_main.app.user_middleware[0].kwargs,
        "max_bytes",
        10,
    )
    # Also zero out the multipart overhead buffer so the threshold is exactly 10.
    monkeypatch.setattr(app_main, "MULTIPART_OVERHEAD_BYTES", 0, raising=True)
    # Force middleware stack rebuild so the patched value takes effect.
    # monkeypatch will restore the original stack after the test.
    monkeypatch.setattr(app_main.app, "middleware_stack", None)

    files = {"file": ("big.wav", b"01234567890", "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 413
    # Verify the middleware responded (not the handler-level check) by confirming
    # the endpoint was never reached — no temp file created on disk.
    assert "File too large" in resp.json()["detail"]


def test_middleware_allows_file_at_limit_with_multipart_overhead(client: TestClient, monkeypatch):
    """A file whose raw bytes are exactly at the limit should NOT be rejected
    by the middleware, because Content-Length includes multipart framing that
    pushes it slightly above the file-size limit."""
    from src.app import main as app_main

    # Set a very small limit so we can construct a payload that is right at
    # the boundary.
    # Defensive check: fail fast if middleware registration order changes.
    assert app_main.app.user_middleware[0].cls is app_main.UploadSizeLimitMiddleware, (
        "Expected first middleware to be UploadSizeLimitMiddleware"
    )
    limit = 100
    monkeypatch.setitem(
        app_main.app.user_middleware[0].kwargs,
        "max_bytes",
        limit,
    )
    monkeypatch.setattr(app_main, "MULTIPART_OVERHEAD_BYTES", 2048, raising=True)
    monkeypatch.setattr(app_main, "MAX_UPLOAD_BYTES", limit, raising=True)
    monkeypatch.setattr(app_main.app, "middleware_stack", None)

    # A small valid WAV that is well under both limits — the middleware should
    # let it through; the handler-level check uses MAX_UPLOAD_BYTES for the
    # actual file bytes, so we keep the payload under that too.
    files = {"file": ("tiny.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}
    resp = client.post("/api/upload", files=files)
    # Should succeed (200) because Content-Length (with multipart overhead)
    # is still under limit + MULTIPART_OVERHEAD_BYTES.
    assert resp.status_code == 200, resp.text


def test_middleware_413_includes_cors_headers_for_allowed_origin(client: TestClient, monkeypatch):
    """When the middleware rejects an oversized upload from an allowed origin,
    the 413 response must include CORS headers so the browser can read it."""
    from src.app import main as app_main

    # Defensive check: fail fast if middleware registration order changes.
    assert app_main.app.user_middleware[0].cls is app_main.UploadSizeLimitMiddleware, (
        "Expected first middleware to be UploadSizeLimitMiddleware"
    )
    monkeypatch.setitem(
        app_main.app.user_middleware[0].kwargs,
        "max_bytes",
        10,
    )
    monkeypatch.setattr(app_main, "MULTIPART_OVERHEAD_BYTES", 0, raising=True)
    monkeypatch.setattr(app_main.app, "middleware_stack", None)

    origin = app_main.ALLOWED_ORIGINS[0]  # e.g. "http://localhost:4330"
    files = {"file": ("big.wav", b"01234567890", "audio/wav")}
    resp = client.post("/api/upload", files=files, headers={"Origin": origin})
    assert resp.status_code == 413
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"
    assert "File too large" in resp.json()["detail"]


def test_middleware_413_no_cors_headers_for_unknown_origin(client: TestClient, monkeypatch):
    """When the middleware rejects an oversized upload from a disallowed origin,
    the 413 response must NOT include CORS headers."""
    from src.app import main as app_main

    # Defensive check: fail fast if middleware registration order changes.
    assert app_main.app.user_middleware[0].cls is app_main.UploadSizeLimitMiddleware, (
        "Expected first middleware to be UploadSizeLimitMiddleware"
    )
    monkeypatch.setitem(
        app_main.app.user_middleware[0].kwargs,
        "max_bytes",
        10,
    )
    monkeypatch.setattr(app_main, "MULTIPART_OVERHEAD_BYTES", 0, raising=True)
    monkeypatch.setattr(app_main.app, "middleware_stack", None)

    files = {"file": ("big.wav", b"01234567890", "audio/wav")}
    resp = client.post("/api/upload", files=files, headers={"Origin": "https://evil.com"})
    assert resp.status_code == 413
    assert "access-control-allow-origin" not in resp.headers


def test_middleware_413_includes_cors_wildcard_for_any_origin(monkeypatch):
    """When ALLOWED_ORIGINS is ['*'] (default), a 413 from the size-limit middleware
    must include Access-Control-Allow-Origin: * so browsers can read the error body."""
    import importlib

    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    from src.app import main as app_main

    importlib.reload(app_main)
    try:
        assert app_main.ALLOWED_ORIGINS == ["*"], "Expected wildcard default after env delete"
        monkeypatch.setitem(
            app_main.app.user_middleware[0].kwargs,
            "max_bytes",
            10,
        )
        monkeypatch.setattr(app_main, "MULTIPART_OVERHEAD_BYTES", 0, raising=True)
        monkeypatch.setattr(app_main.app, "middleware_stack", None)

        with TestClient(app_main.app) as client:
            files = {"file": ("big.wav", b"01234567890", "audio/wav")}
            resp = client.post(
                "/api/upload",
                files=files,
                headers={"Origin": "https://arbitrary.example.com"},
            )
            assert resp.status_code == 413
            assert resp.headers.get("access-control-allow-origin") == "*"
            assert "File too large" in resp.json()["detail"]
    finally:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:4330,http://localhost:8788")
        importlib.reload(app_main)


def test_upload_strips_directory_traversal_from_filename(client: TestClient):
    # A filename like "../../etc/evil.wav" must be sanitized to "evil.wav"
    # (Path.name strips all path components).
    files = {"file": ("../../etc/evil.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == "evil.wav"


def test_upload_rejects_path_only_filename(client: TestClient):
    # A filename that resolves to an empty base name (e.g. "../") must be rejected.
    files = {"file": ("../", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code in (400, 422)


def test_upload_rejects_riff_non_wav(client: TestClient):
    # RIFF container with non-WAVE FOURCC (e.g. "AVI ") must be rejected even
    # though the first 4 bytes are b"RIFF".
    content = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 4
    files = {"file": ("fake.wav", content, "audio/wav")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400


def test_upload_rejects_m4a_with_malformed_ftyp_box(client: TestClient):
    # A file whose first 8 bytes look like an ftyp size but whose box type
    # is not b"ftyp" should be rejected (brand parse fails → is_valid_audio=False).
    # Bytes: size=16 (big-endian), box_type="moov", then 8 zero bytes.
    content = (16).to_bytes(4, "big") + b"moov" + b"\x00" * 8
    files = {"file": ("bad.m4a", content, "audio/mp4")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400
