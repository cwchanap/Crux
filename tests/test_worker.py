import importlib
import json
import sys
import types

import pytest


class FakeResponse:
    @staticmethod
    def new(body, status=200, headers=None):  # noqa: D401, ANN001
        return {"body": body, "status": status, "headers": headers or {}}


class FakeAssets:
    def fetch(self, request):  # noqa: D401, ANN001
        return "ASSET-FETCHED"


class FakeEnv:
    def __init__(self):
        self.ASSETS = FakeAssets()


class FakeHeaders(dict):
    def get(self, key, default=None):  # noqa: D401, ANN001
        return super().get(key, default)


class FakeRequest:
    def __init__(self, url: str):
        self.url = url
        self.headers = FakeHeaders({"host": "example.com"})


@pytest.fixture(autouse=True)
def patch_js_module(monkeypatch):
    # Provide a fake 'js' module so src.worker can import Response
    js_mod = types.SimpleNamespace(Response=FakeResponse)
    monkeypatch.setitem(sys.modules, "js", js_mod)
    # Reload worker to pick up the fake 'js' module, if already imported
    if "src.worker" in sys.modules:
        importlib.reload(sys.modules["src.worker"])  # type: ignore
    yield


@pytest.mark.asyncio
async def test_worker_health_endpoint():
    import src.worker as worker

    req = FakeRequest("https://example.com/api/health")
    env = FakeEnv()

    resp = await worker.on_fetch(req, env)
    assert isinstance(resp, dict)
    assert resp["status"] == 200
    data = json.loads(resp["body"])  # body is a JSON string
    assert data.get("status") == "healthy"


@pytest.mark.asyncio
async def test_worker_static_passthrough():
    import src.worker as worker

    req = FakeRequest("https://example.com/")
    env = FakeEnv()
    resp = await worker.on_fetch(req, env)
    assert resp == "ASSET-FETCHED"


@pytest.mark.asyncio
async def test_worker_unsupported_api():
    import src.worker as worker

    req = FakeRequest("https://example.com/api/other")
    env = FakeEnv()
    resp = await worker.on_fetch(req, env)
    assert isinstance(resp, dict)
    assert resp["status"] == 501
    data = json.loads(resp["body"])  # JSON string body
    assert data.get("error")
