from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    load_checkpoint_acquisition_evidence,
    load_checkpoint_acquisition_request,
    render_checkpoint_acquisition_evidence,
)

REQUEST_PATH = (
    Path("config")
    / "benchmark"
    / "backends"
    / f"{OFFICIAL_BACKEND_ID}.checkpoint-acquisition-request.json"
)
EXPECTED_ARCHIVE_SHA256 = "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"


def _read_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def test_checkpoint_request_is_complete_without_final_backend_lock() -> None:
    request = load_checkpoint_acquisition_request(REQUEST_PATH)

    assert request.backend_id == OFFICIAL_BACKEND_ID
    assert request.archive.sha256 == EXPECTED_ARCHIVE_SHA256
    assert len(request.archive_members) == 4
    assert len(request.published_component_names) == 3
    assert {member.role for member in request.archive_members} == {
        "pointer",
        "published_component",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["archive_members"].append(  # type: ignore[index]
            payload["archive_members"][0]  # type: ignore[index]
        ),
        lambda payload: payload["archive_members"].pop(),  # type: ignore[index]
        lambda payload: payload["archive_members"][0].update({"name": "renamed"}),  # type: ignore[index]
        lambda payload: payload["archive_members"][0].update({"name": "../checkpoint"}),  # type: ignore[index]
        lambda payload: payload["archive_members"].append(  # type: ignore[index]
            {
                "name": "extra",
                "role": "published_component",
                "sha256": "a" * 64,
                "size": 1,
            }
        ),
        lambda payload: payload.update({"unknown": True}),
    ],
    ids=("duplicate", "missing", "renamed", "unsafe", "extra", "unknown"),
)
def test_checkpoint_request_rejects_member_set_contradictions(
    tmp_path: Path,
    mutate: object,
) -> None:
    payload = _read_payload(REQUEST_PATH)
    assert callable(mutate)
    mutate(payload)
    path = tmp_path / "request.json"
    _write_payload(path, payload)

    with pytest.raises(CheckpointAcquisitionError):
        load_checkpoint_acquisition_request(path)


def test_checkpoint_evidence_reauthenticates_the_complete_request(tmp_path: Path) -> None:
    request = load_checkpoint_acquisition_request(REQUEST_PATH)
    evidence, content = render_checkpoint_acquisition_evidence(
        request,
        acquisition_mode="cache_verify",
        model_artifact_set_sha256="a" * 64,
        cache_path=PurePosixPath("artifacts/benchmark/model-cache/sha256/test"),
    )
    path = tmp_path / "evidence.json"
    path.write_bytes(content)

    assert load_checkpoint_acquisition_evidence(path, request=request) == evidence
