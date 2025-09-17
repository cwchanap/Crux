from typing import Any, Dict

import pytest

from src.app.storage import StorageAdapter


@pytest.mark.asyncio
async def test_local_storage_adapter_crud(tmp_path):
    adapter = StorageAdapter(storage_type="local", config={"base_path": str(tmp_path)})

    job_id = "job-123"
    job_data: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "progress": 0,
        "result_url": None,
        "error": None,
    }

    # Save job
    ok = await adapter.save_job(job_id, job_data)
    assert ok is True

    # Get job
    loaded = await adapter.get_job(job_id)
    assert loaded is not None
    assert loaded["job_id"] == job_id
    assert loaded["status"] == "pending"

    # Update status
    ok2 = await adapter.update_job_status(job_id, status="processing", progress=50)
    assert ok2 is True

    loaded2 = await adapter.get_job(job_id)
    assert loaded2 is not None
    assert loaded2["status"] == "processing"
    assert loaded2["progress"] == 50

    # Save MIDI and retrieve it
    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff/\x00"
    ok3 = await adapter.save_midi(job_id, midi_bytes)
    assert ok3 is True

    midi_loaded = await adapter.get_midi(job_id)
    assert midi_loaded == midi_bytes

    # List jobs
    jobs = await adapter.list_jobs(limit=10, offset=0)
    assert isinstance(jobs, list)
    assert any(j.get("job_id") == job_id for j in jobs)
