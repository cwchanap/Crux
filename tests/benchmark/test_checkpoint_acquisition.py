from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from runtime.oaf_tf1.model import (
    OafCheckpointConfig,
    OafModelConfig,
    OafModelConfigError,
    load_model_config,
)
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    load_checkpoint_acquisition_evidence,
    load_checkpoint_acquisition_request,
    prepare_oaf_checkpoint,
    render_checkpoint_acquisition_evidence,
)

REQUEST_PATH = (
    Path("config")
    / "benchmark"
    / "backends"
    / f"{OFFICIAL_BACKEND_ID}.checkpoint-acquisition-request.json"
)
EXPECTED_ARCHIVE_SHA256 = "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"

_COMPONENT_CONTENT = {
    "model.ckpt-569400.data-00000-of-00001": b"data-component",
    "model.ckpt-569400.index": b"index-component",
    "model.ckpt-569400.meta": b"meta-component",
}


def _synthetic_archive(tmp_path: Path) -> tuple[Path, OafModelConfig]:
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, content in _COMPONENT_CONTENT.items():
            output.writestr(name, content)
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    config = OafModelConfig(
        backend_id="test-backend",
        model_id="test-model",
        architecture_id="test-architecture",
        upstream_source_commit="0" * 40,
        training_data_map_id="test-data",
        native_output_space_id="test-output",
        native_metadata_schema_id="test-metadata",
        max_input_audio_frames=None,
        checkpoint=OafCheckpointConfig(
            url="https://example.invalid/checkpoint.zip",
            archive_name="checkpoint.zip",
            archive_sha256=archive_sha256,
            components={
                name: hashlib.sha256(content).hexdigest()
                for name, content in _COMPONENT_CONTENT.items()
            },
        ),
    )
    return archive, config


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


def test_load_model_config_uses_the_neutral_model_source() -> None:
    config = load_model_config()

    assert config.backend_id == "magenta-egmd-tf1-94529798-8hit-v1"
    assert config.model_id == "magenta-egmd-ckpt-569400-v1"
    assert config.upstream_source_commit == "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
    assert config.max_input_audio_frames is None
    assert config.checkpoint.archive_sha256 == EXPECTED_ARCHIVE_SHA256
    assert config.checkpoint.components == {
        "model.ckpt-569400.data-00000-of-00001": (
            "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5"
        ),
        "model.ckpt-569400.index": (
            "475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a"
        ),
        "model.ckpt-569400.meta": (
            "e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422"
        ),
    }


def test_load_model_config_rejects_invalid_component_hash(tmp_path: Path) -> None:
    source = json.loads(Path("runtime/oaf_tf1/model.json").read_text(encoding="utf-8"))
    source["checkpoint"]["components"]["model.ckpt-569400.index"] = "bad"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(OafModelConfigError, match="model.ckpt-569400.index"):
        load_model_config(path)


def test_prepare_oaf_checkpoint_publishes_verified_local_archive(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)

    result = prepare_oaf_checkpoint(
        config,
        tmp_path / "cache",
        download=False,
        archive_path=archive,
    )

    assert result == tmp_path / "cache" / "sha256" / config.checkpoint.archive_sha256
    assert {path.name: path.read_bytes() for path in result.iterdir()} == _COMPONENT_CONTENT


def test_prepare_oaf_checkpoint_verify_only_reuses_existing_cache(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    first = prepare_oaf_checkpoint(
        config,
        tmp_path / "cache",
        download=False,
        archive_path=archive,
    )

    second = prepare_oaf_checkpoint(config, tmp_path / "cache", download=False)

    assert second == first


def test_prepare_oaf_checkpoint_verify_only_requires_cache(tmp_path: Path) -> None:
    _archive, config = _synthetic_archive(tmp_path)

    with pytest.raises(CheckpointAcquisitionError, match="checkpoint cache is missing"):
        prepare_oaf_checkpoint(config, tmp_path / "cache", download=False)


def test_prepare_oaf_checkpoint_rejects_changed_archive_hash(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    changed = OafModelConfig(
        backend_id=config.backend_id,
        model_id=config.model_id,
        architecture_id=config.architecture_id,
        upstream_source_commit=config.upstream_source_commit,
        training_data_map_id=config.training_data_map_id,
        native_output_space_id=config.native_output_space_id,
        native_metadata_schema_id=config.native_metadata_schema_id,
        max_input_audio_frames=config.max_input_audio_frames,
        checkpoint=OafCheckpointConfig(
            url=config.checkpoint.url,
            archive_name=config.checkpoint.archive_name,
            archive_sha256="0" * 64,
            components=config.checkpoint.components,
        ),
    )

    with pytest.raises(CheckpointAcquisitionError, match="archive hash"):
        prepare_oaf_checkpoint(changed, tmp_path / "cache", download=False, archive_path=archive)


def test_prepare_oaf_checkpoint_rejects_changed_component_hash(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    components = dict(config.checkpoint.components)
    components["model.ckpt-569400.index"] = "0" * 64
    changed = OafModelConfig(
        backend_id=config.backend_id,
        model_id=config.model_id,
        architecture_id=config.architecture_id,
        upstream_source_commit=config.upstream_source_commit,
        training_data_map_id=config.training_data_map_id,
        native_output_space_id=config.native_output_space_id,
        native_metadata_schema_id=config.native_metadata_schema_id,
        max_input_audio_frames=config.max_input_audio_frames,
        checkpoint=OafCheckpointConfig(
            url=config.checkpoint.url,
            archive_name=config.checkpoint.archive_name,
            archive_sha256=config.checkpoint.archive_sha256,
            components=components,
        ),
    )

    with pytest.raises(CheckpointAcquisitionError, match="component hash"):
        prepare_oaf_checkpoint(changed, tmp_path / "cache", download=False, archive_path=archive)
