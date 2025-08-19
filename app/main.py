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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Drum Transcription API", version="1.0.0")

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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Please build the UI first: cd ui && npm install && npm run build</h1>"
        )
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


# Mount static files for the SPA (must be after API routes)
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


@app.post("/api/transcribe", response_model=JobResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload an audio file for drum transcription
    Returns a job ID for tracking progress
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Please upload MP3, WAV, M4A, or FLAC"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{job_id}_{file.filename}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

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
            "filename": file.filename,
            "file_size": len(content),
            "file_path": str(file_path),
        },
    }
    jobs_store[job_id] = job

    # Add background task for processing
    background_tasks.add_task(process_audio_task, job_id, str(file_path))

    return JobResponse(
        job_id=job_id, message="Job created successfully", status_url=f"/api/jobs/{job_id}"
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

        # Import processing modules (lazy loading for better startup time)
        from app.transcriber import DrumTranscriber

        # Initialize transcriber
        transcriber = DrumTranscriber()

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
