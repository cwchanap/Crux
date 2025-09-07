"""
FastAPI server for drum transcription using TensorFlow 2.x
Optimized for Cloudflare Workers deployment
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.app.transcriber import DrumTranscriber

app = FastAPI(
    title="Drum Transcription API",
    version="1.0.0",
    description="API for drum transcription. Swagger UI available at /api/docs",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

# CORS configuration for Cloudflare Workers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (will be replaced with Cloudflare KV/D1)
jobs_store: Dict[str, Dict[str, Any]] = {}
midi_store: Dict[str, bytes] = {}
uploads_store: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_load_model():
    """Initialize and cache the transcriber/model at server startup"""
    try:
        app.state.transcriber = DrumTranscriber()
    except Exception:
        # Do not block server startup; background task will fallback to lazy init
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
    """
    Upload an audio file without starting transcription
    Returns an upload ID for later processing
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Please upload MP3, WAV, M4A, or FLAC"
        )

    # Generate upload ID
    upload_id = str(uuid.uuid4())

    # Save file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{upload_id}_{file.filename}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Store upload info
    upload_info = {
        "upload_id": upload_id,
        "filename": file.filename,
        "file_size": len(content),
        "file_path": str(file_path),
        "created_at": datetime.utcnow().isoformat(),
    }
    uploads_store[upload_id] = upload_info

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        file_size=len(content),
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
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
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
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        jobs_store[job_id]["progress"] = 10

        # Get preloaded transcriber if available; otherwise create and cache it
        transcriber = getattr(app.state, "transcriber", None)
        if transcriber is None:
            transcriber = DrumTranscriber()
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
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()

        # Clean up temp file
        os.remove(file_path)

    except Exception as e:
        # Update job as failed
        jobs_store[job_id]["status"] = "failed"
        jobs_store[job_id]["error"] = str(e)
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()

        # Clean up temp file if exists
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
