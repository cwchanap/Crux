from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backends import CanonicalAudio
from src.benchmark.backends.muscriptor import MuscriptorBackend, MuscriptorBackendError
from src.benchmark.muscriptor_model import MUSCRIPTOR_MODEL_SCHEMA, MUSCRIPTOR_RELEASE_COMMIT

WEIGHTS = b"verified safetensors"
CONFIG = b'{"model":"fake"}'
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()
CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()
REVISION = "a" * 40


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
    def __init__(self, events: tuple[object, ...]):
        self.events = events
        self.transcribe_calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object):
        self.transcribe_calls.append((audio, kwargs))
        yield from self.events


class FakeTranscriptionModel:
    load_calls: list[dict[str, object]] = []
    model: FakeModel

    @classmethod
    def load_model(cls, **kwargs: object) -> FakeModel:
        cls.load_calls.append(kwargs)
        return cls.model


def _fake_muscriptor(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[object, ...],
    *,
    version: str = "0.3.0",
) -> FakeModel:
    model = FakeModel(events)
    FakeTranscriptionModel.load_calls = []
    FakeTranscriptionModel.model = model
    module = types.ModuleType("muscriptor")
    module.__version__ = version
    module.TranscriptionModel = FakeTranscriptionModel
    module.NoteStartEvent = FakeNoteStartEvent
    module.NoteEndEvent = FakeNoteEndEvent
    module.ProgressEvent = FakeProgressEvent
    monkeypatch.setitem(sys.modules, "muscriptor", module)
    return model


def _write_lock(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(WEIGHTS)
    (checkpoint_dir / "config.json").write_bytes(CONFIG)
    lock_path = tmp_path / "model.json"
    payload = {
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
    lock_path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
    return lock_path, checkpoint_dir


def _audio(tmp_path: Path) -> CanonicalAudio:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake wav")
    return CanonicalAudio(
        path=path,
        source_audio_id="song/audio.wav",
        source_audio_sha256="a" * 64,
        input_view_id="full-mix-v1",
        input_audio_sha256="b" * 64,
        byte_length=46,
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=1,
    )


def test_backend_loads_verified_weights_and_forwards_frozen_transcription_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = FakeNoteStartEvent(pitch=38, start_time=1.234567, index=7, instrument="drums")
    model = _fake_muscriptor(
        monkeypatch,
        (
            FakeProgressEvent(completed=0, total=1),
            start,
            FakeNoteEndEvent(end_time=1.5, start_event=start),
            FakeProgressEvent(completed=1, total=1),
        ),
    )
    lock_path, checkpoint_dir = _write_lock(tmp_path)
    audio = _audio(tmp_path)

    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    prediction = backend.transcribe(audio)

    assert FakeTranscriptionModel.load_calls == [
        {
            "weights_path": checkpoint_dir / "model.safetensors",
            "device": "cpu",
            "dtype": "float32",
        }
    ]
    assert model.transcribe_calls == [
        (
            audio.path,
            {
                "use_sampling": False,
                "temperature": 1.0,
                "cfg_coef": 1.0,
                "instruments": ["drums"],
                "batch_size": 1,
                "no_eos_is_ok": True,
                "beam_size": 1,
                "prelude_forcing": True,
            },
        )
    ]
    assert prediction.audio == audio
    assert len(prediction.events) == 1
    assert prediction.events[0].time_sec == start.start_time
    assert prediction.events[0].native_midi_note == start.pitch
    assert prediction.events[0].native_class_id == "drums:midi_38"
    assert prediction.events[0].model_output_bin is None
    assert prediction.events[0].confidence is None
    assert prediction.events[0].velocity_midi is None
    assert prediction.events[0].native_metadata == {"instrument_group": "drums"}


@pytest.mark.parametrize(
    "event",
    [
        FakeNoteStartEvent(pitch=38, start_time=1.0, index=1, instrument="piano"),
        FakeNoteStartEvent(pitch=38, start_time=float("nan"), index=1, instrument="drums"),
        FakeNoteStartEvent(pitch=38, start_time=float("inf"), index=1, instrument="drums"),
        FakeNoteStartEvent(pitch=38, start_time=-0.001, index=1, instrument="drums"),
        FakeNoteStartEvent(pitch=-1, start_time=1.0, index=1, instrument="drums"),
        FakeNoteStartEvent(pitch=128, start_time=1.0, index=1, instrument="drums"),
    ],
)
def test_backend_rejects_invalid_note_start_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: FakeNoteStartEvent
) -> None:
    _fake_muscriptor(monkeypatch, (event,))
    lock_path, checkpoint_dir = _write_lock(tmp_path)
    backend = MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)

    with pytest.raises(MuscriptorBackendError):
        backend.transcribe(_audio(tmp_path))


def test_backend_does_not_load_model_when_descriptor_identity_is_not_lock_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, ())
    lock_path, checkpoint_dir = _write_lock(tmp_path)

    from src.benchmark.backend_identity import build_descriptor

    bad_payload = {
        "architecture_id": "muscriptor-transformer-v0.3.0",
        "backend_id": "muscriptor-v0.3.0-drums-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v2",
        "model_id": "muscriptor-medium-0123456789ab-fedcba987654",
        "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
        "native_output_space_id": "muscriptor-drums-midi128-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "muscriptor-training-data-v0.3.0",
        "upstream_source_commit": MUSCRIPTOR_RELEASE_COMMIT,
    }
    descriptor = build_descriptor(
        bad_payload, frozenset(bad_payload), bad_payload["descriptor_schema"]
    )

    with pytest.raises(ValueError):
        MuscriptorBackend(
            model_lock_path=lock_path,
            checkpoint_dir=checkpoint_dir,
            descriptor=descriptor,
        )
    assert FakeTranscriptionModel.load_calls == []


def test_backend_rejects_package_version_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, (), version="0.2.0")
    lock_path, checkpoint_dir = _write_lock(tmp_path)

    with pytest.raises(ValueError, match="package version"):
        MuscriptorBackend(model_lock_path=lock_path, checkpoint_dir=checkpoint_dir)
    assert FakeTranscriptionModel.load_calls == []


def test_backend_rejects_device_or_dtype_that_differs_from_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_muscriptor(monkeypatch, ())
    lock_path, checkpoint_dir = _write_lock(tmp_path)

    with pytest.raises(ValueError, match="device"):
        MuscriptorBackend(
            model_lock_path=lock_path,
            checkpoint_dir=checkpoint_dir,
            device="mps",
        )
    assert FakeTranscriptionModel.load_calls == []

    with pytest.raises(ValueError, match="dtype"):
        MuscriptorBackend(
            model_lock_path=lock_path,
            checkpoint_dir=checkpoint_dir,
            dtype="float16",
        )
    assert FakeTranscriptionModel.load_calls == []
