from pathlib import Path

from fastapi.testclient import TestClient


def test_root_fallback_when_index_missing(client: TestClient, monkeypatch):
    """If `static/index.html` is missing, the endpoint should return fallback HTML."""
    # Keep original implementation so we can forward for all other paths
    original_exists = Path.exists

    def fake_exists(self: Path):  # noqa: D401
        # Simulate that the specific static index does not exist
        if str(self).endswith("/static/index.html"):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists, raising=False)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Please build the UI first" in resp.text
