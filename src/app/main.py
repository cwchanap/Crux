"""
FastAPI server for drum transcription using TensorFlow 2.x
Optimized for Cloudflare Workers deployment
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# NOTE: Avoid importing the heavy transcriber (and TensorFlow) at module import time.
DrumTranscriber = None  # will be lazily imported when needed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
# 2 MB buffer above MAX_UPLOAD_BYTES so that clients whose Content-Length
# slightly overshoots the limit (e.g. due to multipart framing or metadata)
# are not spuriously rejected.  The buffer is intentionally large relative
# to actual multipart overhead (~1-2 KB) to avoid false negatives at the
# boundary.
MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024  # 2 MB
CHUNK_SIZE = 64 * 1024  # 64 KB read chunks

# Allowlist of MP4 major/compatible brands that represent audio-only containers.
# Generic container brands (b"isom", b"mp42") are intentionally excluded because
# they are shared by video MP4 files and would weaken the audio-only guard.
M4A_AUDIO_BRANDS: frozenset[bytes] = frozenset([b"M4A ", b"M4B ", b"M4P ", b"aac ", b"f4a "])
MAX_FTYP_COMPAT_SCAN_BYTES = 4096

# Magic bytes for allowed audio formats
AUDIO_MAGIC: list[tuple[bytes, str]] = [
    (b"RIFF", "wav"),  # WAV
    (b"ID3", "mp3"),  # MP3 with ID3 tag
    (b"\xff\xfb", "mp3"),  # MPEG-1 Layer 3, no CRC
    (b"\xff\xfa", "mp3"),  # MPEG-1 Layer 3, with CRC
    (b"\xff\xf3", "mp3"),  # MPEG-2 Layer 3, no CRC
    (b"\xff\xf2", "mp3"),  # MPEG-2 Layer 3, with CRC
    (b"\xff\xe3", "mp3"),  # MPEG-2.5 Layer 3, no CRC
    (b"\xff\xe2", "mp3"),  # MPEG-2.5 Layer 3, with CRC
    (b"fLaC", "flac"),  # FLAC
]


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class UploadSizeLimitMiddleware:
    """Reject uploads whose Content-Length exceeds the limit *before* the
    multipart body is parsed by Starlette.  Without this guard the server
    fully receives and spools a huge file only to reject it after the fact.

    The middleware uses *max_bytes + MULTIPART_OVERHEAD_BYTES* as its ceiling
    so that files right at the documented limit are not falsely rejected due
    to multipart boundaries and headers inflating the Content-Length header.

    Because this middleware sits outside CORSMiddleware, it injects the
    appropriate CORS headers into its 413 response so that cross-origin
    browser clients receive a readable error instead of a generic CORS
    failure.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") in ("POST", "PUT", "PATCH"):
            origin: Optional[bytes] = None
            content_length: Optional[int] = None
            invalid_content_length = False
            for name, value in scope.get("headers", []):
                if name == b"origin":
                    origin = value
                elif name == b"content-length":
                    try:
                        content_length = int(value)
                    except (ValueError, TypeError):
                        invalid_content_length = True
            if invalid_content_length:
                # A malformed Content-Length is always invalid — reject
                # immediately rather than silently passing the request
                # through and relying solely on the handler-level check.
                resp_headers: list[tuple[bytes, bytes]] = [
                    (b"content-type", b"application/json"),
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": 400,
                        "headers": resp_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"detail":"Invalid Content-Length header"}',
                    }
                )
                return
            if (
                content_length is not None
                and content_length > self.max_bytes + MULTIPART_OVERHEAD_BYTES
            ):
                # Build response headers: always include content-type
                # and, when an Origin is present that matches an
                # allowed origin, include the CORS allow-origin header.
                resp_headers: list[tuple[bytes, bytes]] = [
                    (b"content-type", b"application/json"),
                ]
                if origin:
                    origin_str = origin.decode("latin-1")
                    if origin_str in ALLOWED_ORIGINS:
                        resp_headers.append((b"access-control-allow-origin", origin))
                        resp_headers.append((b"access-control-allow-credentials", b"true"))
                        resp_headers.append((b"vary", b"Origin"))
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": resp_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": (
                            '{"detail":"File too large. Maximum size is'
                            f" {self.max_bytes // (1024 * 1024)} MB"
                            '"}'
                        ).encode(),
                    }
                )
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Drum Transcription API",
    version="1.0.0",
    description="API for drum transcription. Swagger UI available at /api/docs",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

# CORS configuration for Cloudflare Workers
# Default to the local development allowlist when CORS_ALLOWED_ORIGINS is not
# set. Operators should set CORS_ALLOWED_ORIGINS explicitly in production to
# lock down allowed origins.
_CORS_RAW = os.getenv("CORS_ALLOWED_ORIGINS")
if _CORS_RAW is not None:
    ALLOWED_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()]
else:
    logger.warning(
        "CORS_ALLOWED_ORIGINS is unset – defaulting to the localhost allowlist. "
        "Set CORS_ALLOWED_ORIGINS explicitly in production."
    )
    ALLOWED_ORIGINS = ["http://localhost:4330", "http://localhost:8788"]

if "*" in ALLOWED_ORIGINS:
    raise ValueError(
        "CORS_ALLOWED_ORIGINS must not contain '*' when credentials are enabled. "
        "Browsers reject 'Access-Control-Allow-Origin: *' combined with "
        "'Access-Control-Allow-Credentials: true'.  List explicit origins instead."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# NOTE: add_middleware wraps the entire existing stack, so the *last* call
# becomes the outermost middleware and runs *first* on inbound requests.
# Adding UploadSizeLimitMiddleware after CORSMiddleware means the size check
# executes before CORS and before Starlette starts parsing the body.
app.add_middleware(UploadSizeLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)

# In-memory storage for demo (will be replaced with Cloudflare KV/D1)
jobs_store: Dict[str, Dict[str, Any]] = {}
midi_store: Dict[str, bytes] = {}
uploads_store: Dict[str, Dict[str, Any]] = {}


def _read_m4a_brands(file_path: Path, total_bytes: int) -> tuple[bytes, set[bytes]]:
    with open(file_path, "rb") as f:
        ftyp_size_bytes = f.read(4)
        box_type = f.read(4)
        if len(ftyp_size_bytes) != 4 or box_type != b"ftyp":
            raise ValueError("Invalid ftyp box header")

        ftyp_size = int.from_bytes(ftyp_size_bytes, "big")
        compat_offset = 16
        if ftyp_size == 0:
            ftyp_size = total_bytes
        elif ftyp_size == 1:
            large_size_bytes = f.read(8)
            if len(large_size_bytes) != 8:
                raise ValueError("Incomplete extended ftyp size")
            ftyp_size = int.from_bytes(large_size_bytes, "big")
            compat_offset = 24

        if not compat_offset <= ftyp_size <= total_bytes:
            raise ValueError("Invalid ftyp size")

        major_brand = f.read(4)
        minor_version = f.read(4)
        if len(major_brand) != 4 or len(minor_version) != 4:
            raise ValueError("Incomplete ftyp box body")

        compat_bytes = ftyp_size - compat_offset
        compat_scan_bytes = min(compat_bytes, MAX_FTYP_COMPAT_SCAN_BYTES)
        compat_brands: set[bytes] = set()

        while compat_scan_bytes >= 4:
            brand = f.read(4)
            if len(brand) != 4:
                raise ValueError("Incomplete compatible brand entry")
            compat_brands.add(brand)
            compat_scan_bytes -= 4

        if compat_bytes > MAX_FTYP_COMPAT_SCAN_BYTES:
            logger.debug(
                "Truncated compatible brand scan from %s to %s bytes",
                compat_bytes,
                MAX_FTYP_COMPAT_SCAN_BYTES,
            )

        return major_brand, compat_brands


def _detect_audio_format(header: bytes, file_path: Path, total_bytes: int) -> Optional[str]:
    detected_format = next(
        (audio_format for magic, audio_format in AUDIO_MAGIC if header.startswith(magic)),
        None,
    )

    if header[:4] == b"RIFF":
        return "wav" if header[8:12] == b"WAVE" else None

    if len(header) >= 12 and header[4:8] == b"ftyp":
        try:
            major_brand, compat_brands = _read_m4a_brands(file_path, total_bytes)
        except (OSError, ValueError) as exc:
            logger.warning("Could not parse ftyp box: %s — rejecting upload", exc, exc_info=True)
            return None

        if (major_brand in M4A_AUDIO_BRANDS) or bool(compat_brands & M4A_AUDIO_BRANDS):
            return "m4a"
        return None

    return detected_format


@app.on_event("startup")
async def startup_load_model():
    """Initialize and cache the transcriber/model at server startup.

    To keep tests and lightweight environments fast, we do not import TensorFlow or load
    the model unless explicitly requested by setting PRELOAD_MODEL=1.
    """
    preload = os.getenv("PRELOAD_MODEL") == "1"
    app.state.transcriber = None
    if preload:
        try:
            # Lazy import only if preloading is requested
            from src.app.transcriber import DrumTranscriber as _DrumTranscriber  # type: ignore

            app.state.transcriber = _DrumTranscriber()
        except Exception as exc:
            logger.error(
                "PRELOAD_MODEL=1 but model failed to load at startup: %s",
                exc,
                exc_info=True,
            )
            app.state.transcriber = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    updated_at: datetime
    progress: int = 0
    result_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    job_id: str
    message: str
    status_url: str


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    file_size: int
    message: str


class StartJobRequest(BaseModel):
    upload_id: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    html_path = Path(__file__).resolve().parents[2] / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Please build the UI first: cd ui && npm install && npm run build</h1>"
        )
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    # Validate that a filename was supplied (UploadFile.filename is Optional[str])
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    # Sanitize filename: strip path components to prevent directory traversal
    safe_name = Path(file.filename.strip()).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Validate file extension (against sanitized name)
    allowed_exts = (".mp3", ".wav", ".m4a", ".flac")
    if not safe_name.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload MP3, WAV, M4A, or FLAC",
        )

    # Stream upload to temp file to avoid memory buffering
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    upload_id = str(uuid.uuid4())
    temp_file_path = temp_dir / f"{upload_id}_temp"

    total_bytes = 0
    header = b""
    file_too_large = False
    try:
        with open(temp_file_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    file_too_large = True
                    break
                f.write(chunk)
                if len(header) < 12:
                    header += chunk[: 12 - len(header)]
    except BaseException:
        temp_file_path.unlink(missing_ok=True)
        logger.error("Upload write failed for %s — temp file cleaned up", upload_id, exc_info=True)
        raise

    if file_too_large:
        temp_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    # Validate magic bytes (need at least 4 bytes)
    if total_bytes < 4:
        temp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is too small to be valid audio")

    # Ensure we have at least 12 bytes for full header validation
    if len(header) < 12:
        # Read additional bytes from file if needed
        with open(temp_file_path, "rb") as f:
            header = f.read(12)

    detected_format = _detect_audio_format(header, temp_file_path, total_bytes)
    if detected_format is None:
        temp_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="File content does not match a supported audio format",
        )

    expected_format = safe_name.rsplit(".", maxsplit=1)[-1].lower()
    if expected_format != detected_format:
        temp_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="File extension does not match detected audio format",
        )

    # Rename temp file to final name
    file_path = temp_dir / f"{upload_id}_{safe_name}"
    temp_file_path.rename(file_path)

    upload_info = {
        "upload_id": upload_id,
        "filename": safe_name,
        "file_size": total_bytes,
        "file_path": str(file_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    uploads_store[upload_id] = upload_info

    return UploadResponse(
        upload_id=upload_id,
        filename=safe_name,
        file_size=total_bytes,
        message="File uploaded successfully",
    )


@app.post("/api/transcribe", response_model=JobResponse)
async def start_transcription(
    background_tasks: BackgroundTasks,
    request: StartJobRequest,
):
    """
    Start transcription job for a previously uploaded file
    Returns a job ID for tracking progress
    """
    # Check if upload exists
    if request.upload_id not in uploads_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    upload_info = uploads_store[request.upload_id]

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create job entry
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
        "result_url": None,
        "error": None,
        "metadata": {
            "filename": upload_info["filename"],
            "file_size": upload_info["file_size"],
            "file_path": upload_info["file_path"],
            "upload_id": request.upload_id,
        },
    }
    jobs_store[job_id] = job

    # Add background task for processing
    background_tasks.add_task(process_audio_task, job_id, upload_info["file_path"])

    return JobResponse(
        job_id=job_id,
        message="Transcription started successfully",
        status_url=f"/api/jobs/{job_id}",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a transcription job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]
    return JobStatus(**job)


@app.get("/api/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the transcribed MIDI file"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if job_id not in midi_store:
        raise HTTPException(status_code=404, detail="MIDI file not found")

    # Create temporary file for download
    temp_path = f"temp_downloads/{job_id}.mid"
    os.makedirs("temp_downloads", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(midi_store[job_id])

    return FileResponse(temp_path, media_type="audio/midi", filename=f"drums_{job_id}.mid")


@app.get("/api/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """List all jobs with pagination"""
    all_jobs = list(jobs_store.values())
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)

    return {
        "total": len(all_jobs),
        "jobs": all_jobs[offset : offset + limit],
        "limit": limit,
        "offset": offset,
    }


async def process_audio_task(job_id: str, file_path: str):
    """
    Background task to process audio file
    This will be moved to a separate worker service for Cloudflare
    """
    try:
        # Update job status
        jobs_store[job_id]["status"] = "processing"
        jobs_store[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        jobs_store[job_id]["progress"] = 10

        # Get preloaded transcriber if available; otherwise create and cache it
        transcriber = getattr(app.state, "transcriber", None)
        if transcriber is None:
            # Lazy import to avoid importing TensorFlow during app/module import
            from src.app.transcriber import DrumTranscriber as _DrumTranscriber  # type: ignore

            transcriber = _DrumTranscriber()
            app.state.transcriber = transcriber

        # Update progress
        jobs_store[job_id]["progress"] = 30

        # Process audio
        midi_data = await transcriber.transcribe(file_path, job_id, jobs_store)

        # Store MIDI result
        midi_store[job_id] = midi_data

        # Update job as completed
        jobs_store[job_id]["status"] = "completed"
        jobs_store[job_id]["progress"] = 100
        jobs_store[job_id]["result_url"] = f"/api/jobs/{job_id}/download"
        jobs_store[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Clean up temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove input file %s — manual cleanup may be required", file_path
            )

    except Exception as e:
        logger.exception("Transcription job %s failed: %s", job_id, e)
        jobs_store[job_id]["status"] = "failed"
        jobs_store[job_id]["error"] = str(e)
        jobs_store[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove input file %s — manual cleanup may be required", file_path
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
