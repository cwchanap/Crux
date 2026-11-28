from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from runtime.oaf_tf1.model import (
    OafCheckpointConfig,
    OafModelConfig,
    OafModelConfigError,
    load_model_config,
)
from scripts import freeze_muscriptor_model
from src.benchmark import checkpoint_acquisition
from src.benchmark.artifact_io import ArtifactPublicationError
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    prepare_oaf_checkpoint,
)
from src.benchmark.muscriptor_model import MuscriptorModelLockError
from tests.benchmark.muscriptor_run_fixtures import CONFIG, WEIGHTS, _lock

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


@pytest.fixture
def hf_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(WEIGHTS)
    (snapshot / "config.json").write_bytes(CONFIG)
    download = Mock(side_effect=lambda *, filename, **_kwargs: str(snapshot / filename))
    hub = SimpleNamespace(hf_hub_download=download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return SimpleNamespace(path=snapshot, download=download, hub=hub)


@pytest.mark.parametrize("has_config", [False, True])
def test_prepare_muscriptor_restores_only_missing_files_by_locked_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hf_snapshot, has_config: bool
) -> None:
    lock = _lock()
    checkpoint_dir = tmp_path / "checkpoint"
    if has_config:
        checkpoint_dir.mkdir()
        (checkpoint_dir / lock.checkpoint_config_filename).write_bytes(CONFIG)

    assert (
        checkpoint_acquisition.prepare_muscriptor_checkpoint(lock, checkpoint_dir, download=True)
        == checkpoint_dir
    )
    filenames = [lock.checkpoint_filename]
    if not has_config:
        filenames.append(lock.checkpoint_config_filename)
    assert hf_snapshot.download.call_args_list == [
        call(repo_id=lock.checkpoint_repo_id, revision=lock.checkpoint_revision, filename=filename)
        for filename in filenames
    ]
    assert (checkpoint_dir / lock.checkpoint_filename).read_bytes() == WEIGHTS
    assert (checkpoint_dir / lock.checkpoint_config_filename).read_bytes() == CONFIG

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    probe = Mock(side_effect=PermissionError("read-only checkpoint directory"))
    monkeypatch.setattr(checkpoint_acquisition.tempfile, "TemporaryFile", probe)
    for download in (False, True):
        assert (
            checkpoint_acquisition.prepare_muscriptor_checkpoint(
                lock, checkpoint_dir, download=download
            )
            == checkpoint_dir
        )
    probe.assert_not_called()


@pytest.mark.parametrize("bad_bytes", [b"short", b"?" * len(WEIGHTS)])
def test_prepare_muscriptor_rejects_wrong_bytes_before_publication(
    tmp_path: Path, hf_snapshot, bad_bytes: bytes
) -> None:
    lock = _lock()
    (hf_snapshot.path / lock.checkpoint_filename).write_bytes(bad_bytes)
    checkpoint_dir = tmp_path / "checkpoint"

    with pytest.raises(CheckpointAcquisitionError, match="byte length|SHA-256"):
        checkpoint_acquisition.prepare_muscriptor_checkpoint(lock, checkpoint_dir, download=True)

    assert not (checkpoint_dir / lock.checkpoint_filename).exists()


def test_prepare_muscriptor_checks_destination_write_access_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hf_snapshot
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    probe = Mock(side_effect=PermissionError("read-only checkpoint directory"))
    monkeypatch.setattr(checkpoint_acquisition.tempfile, "TemporaryFile", probe)

    with pytest.raises(CheckpointAcquisitionError, match="read-only checkpoint directory"):
        checkpoint_acquisition.prepare_muscriptor_checkpoint(_lock(), checkpoint_dir, download=True)

    probe.assert_called_once_with(dir=checkpoint_dir)
    hf_snapshot.download.assert_not_called()


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_prepare_muscriptor_checks_both_existing_files_before_download(
    tmp_path: Path, hf_snapshot, filename: str
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    conflicting_file = checkpoint_dir / filename
    conflicting_file.write_bytes(b"conflict")

    with pytest.raises(CheckpointAcquisitionError):
        checkpoint_acquisition.prepare_muscriptor_checkpoint(_lock(), checkpoint_dir, download=True)

    hf_snapshot.download.assert_not_called()
    assert conflicting_file.read_bytes() == b"conflict"


@pytest.mark.parametrize("destination", ["symlink", "dangling-symlink", "file", "symlink-file"])
def test_prepare_muscriptor_rejects_unsafe_destinations_before_download(
    tmp_path: Path, hf_snapshot, destination: str
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    other = tmp_path / "other"
    if destination == "symlink":
        other.mkdir()
        checkpoint_dir.symlink_to(other, target_is_directory=True)
    elif destination == "dangling-symlink":
        checkpoint_dir.symlink_to(other, target_is_directory=True)
    elif destination == "file":
        checkpoint_dir.touch()
    else:
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").symlink_to(other)

    with pytest.raises(CheckpointAcquisitionError):
        checkpoint_acquisition.prepare_muscriptor_checkpoint(_lock(), checkpoint_dir, download=True)

    hf_snapshot.download.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_prepare_muscriptor_maps_verify_only_failures_to_acquisition_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    lock = _lock()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    if corrupt:
        (checkpoint_dir / lock.checkpoint_filename).write_bytes(b"corrupt")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(CheckpointAcquisitionError, match="checkpoint") as exc_info:
        checkpoint_acquisition.prepare_muscriptor_checkpoint(lock, checkpoint_dir, download=False)

    assert isinstance(exc_info.value.__cause__, MuscriptorModelLockError)


def test_prepare_muscriptor_keeps_bytes_on_publication_conflict(
    tmp_path: Path, hf_snapshot
) -> None:
    lock = _lock()
    checkpoint_dir = tmp_path / "checkpoint"
    weights = checkpoint_dir / lock.checkpoint_filename

    def conflicting_download(*, filename, **_kwargs):
        checkpoint_dir.mkdir(exist_ok=True)
        weights.write_bytes(b"other writer")
        return str(hf_snapshot.path / filename)

    hf_snapshot.download.side_effect = conflicting_download
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        checkpoint_acquisition.prepare_muscriptor_checkpoint(lock, checkpoint_dir, download=True)

    assert isinstance(exc_info.value.__cause__, ArtifactPublicationError)
    assert weights.read_bytes() == b"other writer"


def test_prepare_muscriptor_never_publishes_through_a_replaced_directory(
    tmp_path: Path, hf_snapshot
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    other = tmp_path / "other"
    other.mkdir()

    def replace_directory(*, filename, **_kwargs):
        checkpoint_dir.rmdir()
        checkpoint_dir.symlink_to(other, target_is_directory=True)
        return str(hf_snapshot.path / filename)

    hf_snapshot.download.side_effect = replace_directory
    with pytest.raises(CheckpointAcquisitionError):
        checkpoint_acquisition.prepare_muscriptor_checkpoint(_lock(), checkpoint_dir, download=True)

    assert list(other.iterdir()) == []


@pytest.fixture
def freeze_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hf_snapshot) -> dict:
    lock = _lock()
    info = SimpleNamespace(
        id=lock.checkpoint_repo_id,
        sha=lock.checkpoint_revision,
        siblings=[
            SimpleNamespace(rfilename=name)
            for name in (lock.checkpoint_filename, lock.checkpoint_config_filename)
        ],
        card_data={"license": "cc-by-nc-4.0"},
    )
    hf_snapshot.hub.HfApi = Mock(return_value=SimpleNamespace(model_info=Mock(return_value=info)))
    monkeypatch.setattr(freeze_muscriptor_model, "_verify_package", lambda: "MIT")
    return {
        "variant": "medium",
        "device": "cpu",
        "dtype": "float32",
        "checkpoint_dir": tmp_path / "freeze-checkpoint",
        "output": tmp_path / "model.json",
    }


def test_freeze_uses_resolved_revision_and_publishes_a_loadable_lock(
    freeze_args, hf_snapshot
) -> None:
    lock = freeze_muscriptor_model.freeze_model(**freeze_args)

    assert lock == _lock()
    hf_snapshot.hub.HfApi.return_value.model_info.assert_called_once_with(lock.checkpoint_repo_id)
    assert hf_snapshot.download.call_args_list == [
        call(repo_id=lock.checkpoint_repo_id, revision=lock.checkpoint_revision, filename=filename)
        for filename in (lock.checkpoint_filename, lock.checkpoint_config_filename)
    ]
    assert freeze_args["output"].is_file()


@pytest.mark.parametrize("status", [401, 403, None])
def test_freeze_preserves_safe_download_details_and_cause(
    freeze_args, hf_snapshot, status: int | None
) -> None:
    if status is None:
        hub_error = TimeoutError("credential-bearing exception detail")
        detail = "TimeoutError"
    else:
        hub_error = RuntimeError("credential-bearing exception detail")
        hub_error.response = SimpleNamespace(status_code=status)
        detail = f"HTTP {status}"
    hf_snapshot.download.side_effect = hub_error

    with pytest.raises(freeze_muscriptor_model.FreezeError, match=detail) as exc_info:
        freeze_muscriptor_model.freeze_model(**freeze_args)

    assert str(hub_error) not in str(exc_info.value)
    cause = exc_info.value.__cause__
    assert isinstance(cause, CheckpointAcquisitionError)
    assert cause.__cause__ is hub_error
    assert not freeze_args["output"].exists()


def test_freeze_metadata_failure_preserves_status_without_exposing_raw_error(
    freeze_args, hf_snapshot, capsys
) -> None:
    hub_error = RuntimeError("credential-bearing metadata detail")
    hub_error.response = SimpleNamespace(status_code=403)
    hf_snapshot.hub.HfApi.return_value.model_info.side_effect = hub_error
    args = [
        option
        for name, value in freeze_args.items()
        for option in (f"--{name.replace('_', '-')}", str(value))
    ]

    assert freeze_muscriptor_model.main(args) == 1
    captured = capsys.readouterr()
    assert "model metadata unavailable" in captured.err
    assert "HTTP 403" in captured.err
    assert str(hub_error) not in captured.err
    hf_snapshot.download.assert_not_called()
    assert not freeze_args["output"].exists()
