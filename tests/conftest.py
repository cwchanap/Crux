import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure the heavy model is not preloaded during tests
os.environ.setdefault("PRELOAD_MODEL", "0")

# Force a deterministic CORS allowlist for every test run.
os.environ["CORS_ALLOWED_ORIGINS"] = "http://localhost:4330,http://localhost:8788"


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    # Import here to ensure PRELOAD_MODEL is applied before app import side effects
    from src.app import main as app_main

    # Reset in-memory stores for isolation between tests
    app_main.jobs_store.clear()
    app_main.midi_store.clear()
    app_main.uploads_store.clear()

    with TestClient(app_main.app) as c:
        yield c
