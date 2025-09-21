from typing import Any, Dict, Tuple

import pytest

from src.app.storage import StorageAdapter


class FakeResponse:
    def __init__(
        self, status_code: int, json_data: Any | None = None, content: bytes | None = None
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content or b""

    def json(self):  # noqa: D401
        return self._json_data


class FakeAsyncClient:
    """Minimal async client stub that supports context manager and get/put."""

    def __init__(self, mapping: Dict[Tuple[str, str], FakeResponse]):
        self._mapping = mapping

    async def __aenter__(self):  # noqa: D401
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401, ANN001
        return False

    async def get(self, url: str, headers=None, params=None):  # noqa: D401, ANN001
        # Ignore headers/params in this simple fake; key by method+url
        return self._mapping.get(("GET", url), FakeResponse(404))

    async def put(self, url: str, headers=None, json=None, content=None):  # noqa: D401, ANN001
        return self._mapping.get(("PUT", url), FakeResponse(404))


@pytest.mark.asyncio
async def test_cloudflare_kv_crud_and_list(monkeypatch):
    # Arrange adapter pointing to a fake account/namespace
    adapter = StorageAdapter(
        storage_type="cloudflare_kv",
        config={
            "account_id": "acc",
            "namespace_id": "ns",
            "api_token": "tok",
        },
    )

    base = "https://api.cloudflare.com/client/v4/accounts/acc/storage/kv/namespaces/ns"

    job_id = "job-1"
    job_obj: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "progress": 0,
        "result_url": None,
        "error": None,
    }
    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff/\x00"

    # Prepare responses mapping
    mapping: Dict[Tuple[str, str], FakeResponse] = {}

    # Save job
    mapping[("PUT", f"{base}/values/job_{job_id}")] = FakeResponse(200)
    # Get job
    mapping[("GET", f"{base}/values/job_{job_id}")] = FakeResponse(200, json_data=job_obj)
    # Keys list -> one job key
    mapping[("GET", f"{base}/keys")] = FakeResponse(
        200, json_data={"result": [{"name": f"job_{job_id}"}]}
    )
    # Save MIDI
    mapping[("PUT", f"{base}/values/midi_{job_id}")] = FakeResponse(200)
    # Get MIDI
    mapping[("GET", f"{base}/values/midi_{job_id}")] = FakeResponse(200, content=midi_bytes)

    # Patch httpx.AsyncClient to our fake; ensure each instantiation sees the same mapping
    import httpx  # local import to patch

    def client_factory(*args, **kwargs):  # noqa: D401, ANN001
        return FakeAsyncClient(mapping)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory, raising=True)

    # Act & Assert
    assert await adapter.save_job(job_id, job_obj) is True

    loaded = await adapter.get_job(job_id)
    assert loaded is not None and loaded["job_id"] == job_id

    listed = await adapter.list_jobs(limit=10, offset=0)
    assert isinstance(listed, list)
    assert listed and listed[0]["job_id"] == job_id

    assert await adapter.save_midi(job_id, midi_bytes) is True
    midi_loaded = await adapter.get_midi(job_id)
    assert midi_loaded == midi_bytes

    # Update job status (invokes GET then PUT under the hood)
    # Ensure mapping covers the PUT already; reuse same endpoint
    updated_ok = await adapter.update_job_status(job_id, status="processing", progress=25)
    assert updated_ok is True


@pytest.mark.asyncio
async def test_cloudflare_kv_get_missing(monkeypatch):
    adapter = StorageAdapter(
        storage_type="cloudflare_kv",
        config={"account_id": "acc", "namespace_id": "ns", "api_token": "tok"},
    )
    base = "https://api.cloudflare.com/client/v4/accounts/acc/storage/kv/namespaces/ns"

    mapping: Dict[Tuple[str, str], FakeResponse] = {}
    mapping[("GET", f"{base}/values/job_missing")] = FakeResponse(404)

    import httpx

    def client_factory(*args, **kwargs):  # noqa: D401, ANN001
        return FakeAsyncClient(mapping)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory, raising=True)

    assert await adapter.get_job("missing") is None
