import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from src.app.storage import StorageAdapter


@pytest.mark.asyncio
async def test_local_storage_adapter_update_nonexistent(tmp_path: Path):
    adapter = StorageAdapter(storage_type="local", config={"base_path": str(tmp_path)})

    ok = await adapter.update_job_status("missing", status="processing")
    assert ok is False


@pytest.mark.asyncio
async def test_local_storage_adapter_get_missing(tmp_path: Path):
    adapter = StorageAdapter(storage_type="local", config={"base_path": str(tmp_path)})

    assert await adapter.get_job("nope") is None
    assert await adapter.get_midi("nope") is None


@pytest.mark.asyncio
async def test_local_storage_adapter_list_pagination_and_order(tmp_path: Path):
    adapter = StorageAdapter(storage_type="local", config={"base_path": str(tmp_path)})

    base_jobs_dir = Path(tmp_path) / "jobs"

    def make_job(job_id: str, created_at: float) -> Dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "updated_at": created_at,
            "progress": 0,
            "result_url": None,
            "error": None,
        }

    # Create three jobs and manipulate file mtimes to define ordering
    now = time.time()
    jobs = [
        ("a", now - 30),
        ("b", now - 20),
        ("c", now - 10),
    ]

    for job_id, ts in jobs:
        ok = await adapter.save_job(job_id, make_job(job_id, ts))
        assert ok
        job_file = base_jobs_dir / f"{job_id}.json"
        # Update mtime to control list order (newest first expected)
        os.utime(job_file, (ts, ts))

    # Expect order: c, b, a
    all_jobs = await adapter.list_jobs(limit=10, offset=0)
    assert [j["job_id"] for j in all_jobs] == ["c", "b", "a"]

    page = await adapter.list_jobs(limit=1, offset=1)
    assert len(page) == 1
    assert page[0]["job_id"] == "b"
