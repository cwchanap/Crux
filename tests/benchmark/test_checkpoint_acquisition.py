from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from runtime.oaf_tf1.model import (
    OafCheckpointConfig,
    OafModelConfig,
    OafModelConfigError,
    load_model_config,
)
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    prepare_oaf_checkpoint,
)

EXPECTED_ARCHIVE_SHA256 = "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"
EXPECTED_COMPONENT_SHA256 = {
    "model.ckpt-569400.data-00000-of-00001": (
        "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5"
    ),
    "model.ckpt-569400.index": ("475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a"),
    "model.ckpt-569400.meta": ("e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422"),
}
_COMPONENT_CONTENT = {
    "model.ckpt-569400.data-00000-of-00001": b"data-component",
    "model.ckpt-569400.index": b"index-component",
    "model.ckpt-569400.meta": b"meta-component",
}
_CHECKPOINT_POINTER = (
    b'model_checkpoint_path: "model.ckpt-569400"\nall_model_checkpoint_paths: "model.ckpt-569400"\n'
)


def _synthetic_archive(
    tmp_path: Path,
    *,
    pointer: bytes | None = _CHECKPOINT_POINTER,
    members: dict[str, bytes] | None = None,
) -> tuple[Path, OafModelConfig]:
    members = dict(_COMPONENT_CONTENT if members is None else members)
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, content in members.items():
            output.writestr(name, content)
        if pointer is not None:
            output.writestr("checkpoint", pointer)
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


def _with_checkpoint(config: OafModelConfig, **changes: object) -> OafModelConfig:
    values = {
        "backend_id": config.backend_id,
        "model_id": config.model_id,
        "architecture_id": config.architecture_id,
        "upstream_source_commit": config.upstream_source_commit,
        "training_data_map_id": config.training_data_map_id,
        "native_output_space_id": config.native_output_space_id,
        "native_metadata_schema_id": config.native_metadata_schema_id,
        "max_input_audio_frames": config.max_input_audio_frames,
        "checkpoint": config.checkpoint,
    }
    values.update(changes)
    return OafModelConfig(**values)


def test_model_config_contains_final_checkpoint_identity() -> None:
    config = load_model_config()

    assert config.backend_id == "magenta-egmd-tf1-94529798-8hit-v1"
    assert config.model_id == "magenta-egmd-ckpt-569400-v1"
    assert config.upstream_source_commit == "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
    assert config.max_input_audio_frames is None
    assert config.checkpoint.archive_sha256 == EXPECTED_ARCHIVE_SHA256
    assert config.checkpoint.components == EXPECTED_COMPONENT_SHA256


def test_model_config_rejects_invalid_component_hash(tmp_path: Path) -> None:
    source = json.loads(Path("runtime/oaf_tf1/model.json").read_text(encoding="utf-8"))
    source["checkpoint"]["components"]["model.ckpt-569400.index"] = "bad"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(OafModelConfigError, match="model.ckpt-569400.index"):
        load_model_config(path)


def test_prepare_checkpoint_verifies_archive_pointer_members_and_components(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)

    result = prepare_oaf_checkpoint(
        config,
        tmp_path / "cache",
        download=False,
        archive_path=archive,
    )

    assert result == tmp_path / "cache" / "sha256" / config.checkpoint.archive_sha256
    assert {path.name: path.read_bytes() for path in result.iterdir()} == _COMPONENT_CONTENT


def test_prepare_checkpoint_publishes_runtime_traversable_permissions(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)

    result = prepare_oaf_checkpoint(
        config,
        tmp_path / "cache",
        download=False,
        archive_path=archive,
    )

    assert stat.S_IMODE(result.lstat().st_mode) == 0o755
    assert {stat.S_IMODE(path.lstat().st_mode) for path in result.iterdir()} == {0o644}


def test_prepare_checkpoint_normalizes_verified_cache_permissions(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    result = prepare_oaf_checkpoint(
        config,
        tmp_path / "cache",
        download=False,
        archive_path=archive,
    )
    result.chmod(0o700)
    for path in result.iterdir():
        path.chmod(0o600)

    assert prepare_oaf_checkpoint(config, tmp_path / "cache", download=False) == result
    assert stat.S_IMODE(result.lstat().st_mode) == 0o755
    assert {stat.S_IMODE(path.lstat().st_mode) for path in result.iterdir()} == {0o644}


def test_prepare_checkpoint_verify_only_reuses_and_verifies_cache(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    first = prepare_oaf_checkpoint(config, tmp_path / "cache", download=False, archive_path=archive)

    assert prepare_oaf_checkpoint(config, tmp_path / "cache", download=False) == first
    (first / "model.ckpt-569400.index").write_bytes(b"changed")
    with pytest.raises(CheckpointAcquisitionError, match="component hash"):
        prepare_oaf_checkpoint(config, tmp_path / "cache", download=False)


def test_prepare_checkpoint_verify_only_requires_cache(tmp_path: Path) -> None:
    _archive, config = _synthetic_archive(tmp_path)

    with pytest.raises(CheckpointAcquisitionError, match="checkpoint cache is missing"):
        prepare_oaf_checkpoint(config, tmp_path / "cache", download=False)


def test_prepare_checkpoint_rejects_changed_archive_hash(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    changed = _with_checkpoint(
        config,
        checkpoint=OafCheckpointConfig(
            url=config.checkpoint.url,
            archive_name=config.checkpoint.archive_name,
            archive_sha256="0" * 64,
            components=config.checkpoint.components,
        ),
    )

    with pytest.raises(CheckpointAcquisitionError, match="archive hash"):
        prepare_oaf_checkpoint(changed, tmp_path / "cache", download=False, archive_path=archive)


def test_prepare_checkpoint_rejects_changed_component_hash(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(tmp_path)
    components = dict(config.checkpoint.components)
    components["model.ckpt-569400.index"] = "0" * 64
    changed = _with_checkpoint(
        config,
        checkpoint=OafCheckpointConfig(
            url=config.checkpoint.url,
            archive_name=config.checkpoint.archive_name,
            archive_sha256=config.checkpoint.archive_sha256,
            components=components,
        ),
    )

    with pytest.raises(CheckpointAcquisitionError, match="component hash"):
        prepare_oaf_checkpoint(changed, tmp_path / "cache", download=False, archive_path=archive)


def test_prepare_checkpoint_rejects_noncanonical_pointer(tmp_path: Path) -> None:
    archive, config = _synthetic_archive(
        tmp_path,
        pointer=b'model_checkpoint_path: "model.ckpt-123"\n',
    )

    with pytest.raises(CheckpointAcquisitionError, match="checkpoint pointer"):
        prepare_oaf_checkpoint(config, tmp_path / "cache", download=False, archive_path=archive)


@pytest.mark.parametrize(
    "members",
    [
        {**_COMPONENT_CONTENT, "extra": b"unexpected"},
        {
            "model.ckpt-569400.data-00000-of-00001": _COMPONENT_CONTENT[
                "model.ckpt-569400.data-00000-of-00001"
            ],
            "model.ckpt-569400.index": _COMPONENT_CONTENT["model.ckpt-569400.index"],
        },
    ],
    ids=("extra-member", "missing-member"),
)
def test_prepare_checkpoint_rejects_non_exact_archive_members(
    tmp_path: Path, members: dict[str, bytes]
) -> None:
    archive, config = _synthetic_archive(tmp_path, members=members)

    with pytest.raises(CheckpointAcquisitionError, match="archive members|component"):
        prepare_oaf_checkpoint(config, tmp_path / "cache", download=False, archive_path=archive)


def test_checkpoint_acquisition_has_no_legacy_request_or_evidence_api() -> None:
    import src.benchmark.checkpoint_acquisition as acquisition

    assert not hasattr(acquisition, "load_checkpoint_acquisition_request")
    assert not hasattr(acquisition, "load_checkpoint_acquisition_evidence")
