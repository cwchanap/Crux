"""Coverage for uncovered lines in muscriptor_model.py and backends/muscriptor.py."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys
import types
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    BackendDescriptor,
    canonical_json_bytes,
)
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.backends.muscriptor import (
    MuscriptorBackend,
    MuscriptorBackendError,
    create_backend,
)
from src.benchmark.muscriptor_model import (
    MUSCRIPTOR_MODEL_SCHEMA,
    MUSCRIPTOR_RELEASE_COMMIT,
    MuscriptorModelLock,
    MuscriptorModelLockError,
    derive_muscriptor_model_id,
    load_muscriptor_model_lock,
    model_lock_payload,
    verify_muscriptor_checkpoint,
)

WEIGHTS = b"frozen safetensors bytes"
CONFIG = b'{"model_type":"muscriptor"}'
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()
CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()
REVISION = "a" * 40


# ---------------------------------------------------------------------------
# Shared model-lock helpers
# ---------------------------------------------------------------------------


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": MUSCRIPTOR_MODEL_SCHEMA,
        "package_name": "muscriptor",
        "package_version": "0.3.0",
        "upstream_source_commit": MUSCRIPTOR_RELEASE_COMMIT,
        "code_license": "MIT",
        "weight_license": "CC BY-NC 4.0",
        "checkpoint_variant": "medium",
        "checkpoint_repo_id": "MuScriptor/muscriptor-medium",
        "checkpoint_revision": REVISION,
        "checkpoint_filename": "model.safetensors",
        "checkpoint_sha256": WEIGHTS_SHA256,
        "checkpoint_byte_length": len(WEIGHTS),
        "checkpoint_config_filename": "config.json",
        "checkpoint_config_sha256": CONFIG_SHA256,
        "checkpoint_config_byte_length": len(CONFIG),
        "model_id": f"muscriptor-medium-{REVISION[:12]}-{WEIGHTS_SHA256[:12]}",
        "device": "cpu",
        "dtype": "float32",
        "input_sample_rate_hz": 16000,
        "chunk_duration_sec": Decimal("5.0"),
        "use_sampling": False,
        "temperature": Decimal("1.0"),
        "cfg_coef": Decimal("1.0"),
        "instruments": ["drums"],
        "batch_size": 1,
        "no_eos_is_ok": True,
        "beam_size": 1,
        "prelude_forcing": True,
        "native_output_space_id": "muscriptor-drums-midi128-v1",
        "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
        "training_data_map_id": "muscriptor-training-data-v0.3.0",
    }
    payload.update(overrides)
    return payload


def _write_lock(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "model.json"
    path.write_bytes(canonical_json_bytes(_payload(**overrides), trailing_newline=True))
    return path


def _valid_lock(tmp_path: Path) -> MuscriptorModelLock:
    return load_muscriptor_model_lock(_write_lock(tmp_path))


def _construct_lock(base: MuscriptorModelLock, **overrides: object) -> MuscriptorModelLock:
    kwargs = asdict(base)
    kwargs.update(overrides)
    return MuscriptorModelLock(**kwargs)


def _make_checkpoint(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(WEIGHTS)
    (checkpoint_dir / "config.json").write_bytes(CONFIG)
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Backend helpers (copied from test_muscriptor_backend.py conventions)
# ---------------------------------------------------------------------------


@dataclass
class FakeNoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class FakeNoteEndEvent:
    end_time: float
    start_event: FakeNoteStartEvent


@dataclass
class FakeProgressEvent:
    completed: int
    total: int


class FakeModel:
    def __init__(self, events: tuple[object, ...] = ()) -> None:
        self.events = events
        self.transcribe_calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object):
        self.transcribe_calls.append((audio, kwargs))
        yield from self.events


class FakeRaisingModel:
    def transcribe(self, audio: object, **kwargs: object):
        raise RuntimeError("inference crashed")


class FakeTranscriptionModel:
    load_calls: list[dict[str, object]] = []
    model: object

    @classmethod
    def load_model(cls, **kwargs: object) -> object:
        cls.load_calls.append(kwargs)
        return cls.model


def _fake_muscriptor(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[object, ...] = (),
    *,
    version: str | None = "0.3.0",
    model: object | None = None,
) -> object:
    if model is None:
        model = FakeModel(events)
    FakeTranscriptionModel.load_calls = []
    FakeTranscriptionModel.model = model
    module = types.ModuleType("muscriptor")
    if version is not None:
        module.__version__ = version
    module.TranscriptionModel = FakeTranscriptionModel
    module.NoteStartEvent = FakeNoteStartEvent
    module.NoteEndEvent = FakeNoteEndEvent
    module.ProgressEvent = FakeProgressEvent
    monkeypatch.setitem(sys.modules, "muscriptor", module)
    return model


def _audio(tmp_path: Path) -> CanonicalAudio:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake wav")
    return CanonicalAudio(
        path=path,
        source_audio_id="song/audio.wav",
        source_audio_sha256="a" * 64,
        input_view_id="full-mix-v1",
        input_audio_sha256="b" * 64,
        byte_length=8,
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=1,
    )


# ===========================================================================
# muscriptor_model.py - derive_muscriptor_model_id
# ===========================================================================


def test_derive_model_id_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="MuscriptorModelLock"):
        derive_muscriptor_model_id("not a lock")  # type: ignore[arg-type]


# ===========================================================================
# muscriptor_model.py - load_muscriptor_model_lock
# ===========================================================================


def test_load_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="Path"):
        load_muscriptor_model_lock("not a path")  # type: ignore[arg-type]


def test_load_unreadable_path_raises_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(MuscriptorModelLockError, match="unavailable"):
        load_muscriptor_model_lock(missing)


def test_load_rejects_missing_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_bytes(canonical_json_bytes(_payload(), trailing_newline=False))
    with pytest.raises(MuscriptorModelLockError, match="final newline"):
        load_muscriptor_model_lock(path)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_bytes(b"not valid json\n")
    with pytest.raises(MuscriptorModelLockError):
        load_muscriptor_model_lock(path)


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    path = _write_lock(tmp_path, schema="wrong.schema/v1")
    with pytest.raises(MuscriptorModelLockError, match="schema"):
        load_muscriptor_model_lock(path)


def test_load_wraps_field_construction_overflow(tmp_path: Path) -> None:
    """A huge int chunk_duration_sec overflows float() inside _numeric_float."""
    path = tmp_path / "model.json"
    path.write_bytes(
        canonical_json_bytes(_payload(chunk_duration_sec=10**1000), trailing_newline=True)
    )
    with pytest.raises(MuscriptorModelLockError, match="fields are invalid"):
        load_muscriptor_model_lock(path)


# ===========================================================================
# muscriptor_model.py - verify_muscriptor_checkpoint
# ===========================================================================


def test_verify_checkpoint_rejects_non_lock(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="MuscriptorModelLock"):
        verify_muscriptor_checkpoint("not a lock", tmp_path)  # type: ignore[arg-type]


def test_verify_checkpoint_rejects_non_path_dir(tmp_path: Path) -> None:
    lock = _valid_lock(tmp_path)
    with pytest.raises(TypeError, match="Path"):
        verify_muscriptor_checkpoint(lock, "not a path")  # type: ignore[arg-type]


def test_verify_checkpoint_rejects_symlink_dir(tmp_path: Path) -> None:
    lock = _valid_lock(tmp_path)
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(MuscriptorModelLockError, match="unavailable"):
        verify_muscriptor_checkpoint(lock, link)


def test_verify_checkpoint_rejects_non_dir(tmp_path: Path) -> None:
    lock = _valid_lock(tmp_path)
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir")
    with pytest.raises(MuscriptorModelLockError, match="unavailable"):
        verify_muscriptor_checkpoint(lock, file_path)


def test_verify_checkpoint_rejects_same_length_different_sha(tmp_path: Path) -> None:
    lock = _valid_lock(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"X" * len(WEIGHTS))
    (checkpoint_dir / "config.json").write_bytes(CONFIG)
    with pytest.raises(MuscriptorModelLockError, match="SHA-256 differs"):
        verify_muscriptor_checkpoint(lock, checkpoint_dir)


# ===========================================================================
# muscriptor_model.py - model_lock_payload
# ===========================================================================


def test_model_lock_payload_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="MuscriptorModelLock"):
        model_lock_payload("not a lock")  # type: ignore[arg-type]


def test_model_lock_payload_returns_canonical_payload(tmp_path: Path) -> None:
    lock = _valid_lock(tmp_path)
    payload = model_lock_payload(lock)
    assert payload["schema"] == MUSCRIPTOR_MODEL_SCHEMA
    assert payload["package_name"] == "muscriptor"
    assert payload["instruments"] == ["drums"]
    assert payload["chunk_duration_sec"] == Decimal("5.0")
    assert payload["temperature"] == Decimal("1.0")
    assert payload["cfg_coef"] == Decimal("1.0")


# ===========================================================================
# muscriptor_model.py - direct construction validation (via __post_init__)
# ===========================================================================


def test_construct_rejects_invalid_model_id_format(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="model_id is invalid"):
        _construct_lock(base, model_id="muscriptor-medium-xxxxxxxxxxxx-yyyyyyyyyyyy")


def test_construct_rejects_empty_training_data_map_id(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="training_data_map_id"):
        _construct_lock(base, training_data_map_id="")


def test_construct_rejects_non_numeric_chunk_duration(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="chunk_duration_sec"):
        _construct_lock(base, chunk_duration_sec="abc")


def test_construct_rejects_non_finite_chunk_duration(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="chunk_duration_sec"):
        _construct_lock(base, chunk_duration_sec=float("inf"))


def test_construct_rejects_non_str_checkpoint_sha256(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="checkpoint_sha256"):
        _construct_lock(base, checkpoint_sha256=None)


def test_construct_rejects_invalid_variant(tmp_path: Path) -> None:
    base = _valid_lock(tmp_path)
    with pytest.raises(MuscriptorModelLockError, match="checkpoint_variant"):
        _construct_lock(base, checkpoint_variant="huge")


# ===========================================================================
# backends/muscriptor.py - MuscriptorBackend.__init__
# ===========================================================================


def test_backend_uses_env_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    monkeypatch.setenv("CRUX_MUSCRIPTOR_CHECKPOINT_DIR", str(checkpoint_dir))
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=None)
    assert isinstance(backend.descriptor(), BackendDescriptor)


def test_backend_rejects_non_path_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    with pytest.raises(TypeError, match="Path"):
        MuscriptorBackend(
            model_lock_path=lock_path,
            checkpoint_dir="not a path",  # type: ignore[arg-type]
        )


def test_backend_accepts_lock_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_muscriptor(monkeypatch)
    lock = _valid_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock, checkpoint_dir=checkpoint_dir)
    assert isinstance(backend.descriptor(), BackendDescriptor)


def test_backend_uses_env_model_lock_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    monkeypatch.setenv("CRUX_MUSCRIPTOR_MODEL_LOCK", str(lock_path))
    backend = MuscriptorBackend(model_lock_path=None, checkpoint_dir=checkpoint_dir)
    assert isinstance(backend.descriptor(), BackendDescriptor)


def test_backend_rejects_non_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="descriptor is invalid"):
        MuscriptorBackend(
            model_lock_path=lock_path,
            checkpoint_dir=checkpoint_dir,
            descriptor="not a descriptor",  # type: ignore[arg-type]
        )


def test_backend_rejects_when_normalize_raises_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)

    def _raise_type_error(value: object) -> dict[str, str]:
        raise TypeError("bad")

    monkeypatch.setattr(
        "src.benchmark.backends.muscriptor.normalize_known_backend_descriptor",
        _raise_type_error,
    )
    with pytest.raises(ValueError, match="descriptor is invalid"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)


def test_backend_rejects_when_normalize_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)

    def _raise_value_error(value: object) -> dict[str, str]:
        raise ValueError("bad")

    monkeypatch.setattr(
        "src.benchmark.backends.muscriptor.normalize_known_backend_descriptor",
        _raise_value_error,
    )
    with pytest.raises(ValueError, match="descriptor is invalid"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)


# ===========================================================================
# backends/muscriptor.py - _verify_package_version
# ===========================================================================


def test_backend_rejects_module_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, version="0.2.0")
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="package version"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)


def test_backend_rejects_installed_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, version="0.3.0")
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.0")
    with pytest.raises(ValueError, match="package version"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)


def test_backend_rejects_when_both_versions_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, version=None)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)

    def _raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    with pytest.raises(ValueError, match="unavailable"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)


# ===========================================================================
# backends/muscriptor.py - descriptor / transcribe / close
# ===========================================================================


def test_backend_descriptor_returns_stored_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    assert isinstance(backend.descriptor(), BackendDescriptor)


def test_backend_transcribe_rejects_when_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    backend.close()
    with pytest.raises(MuscriptorBackendError, match="closed"):
        backend.transcribe(_audio(tmp_path))


def test_backend_transcribe_rejects_non_canonical_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    with pytest.raises(TypeError, match="CanonicalAudio"):
        backend.transcribe("not audio")  # type: ignore[arg-type]


def test_backend_wraps_inference_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_muscriptor(monkeypatch, model=FakeRaisingModel())
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    with pytest.raises(MuscriptorBackendError, match="transcription failed"):
        backend.transcribe(_audio(tmp_path))


def test_backend_close_sets_closed_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    assert backend._closed is False
    backend.close()
    assert backend._closed is True


# ===========================================================================
# backends/muscriptor.py - create_backend
# ===========================================================================


def test_create_backend_uses_env_model_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch)
    lock_path = _write_lock(tmp_path)
    checkpoint_dir = _make_checkpoint(tmp_path)
    monkeypatch.setenv("CRUX_MUSCRIPTOR_MODEL_LOCK", str(lock_path))
    backend = create_backend(checkpoint_dir=checkpoint_dir, model_lock_path=None)
    assert isinstance(backend.descriptor(), BackendDescriptor)
