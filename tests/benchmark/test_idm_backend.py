from __future__ import annotations

import struct
from pathlib import Path

import pytest

from src.benchmark.backend_identity import IDM_BACKEND_ID, sha256_hex
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.backends.idm import (
    IDM_ADAPTER_REVISION,
    IdmBackend,
    IdmBackendError,
)
from src.benchmark.idm_model import (
    IDM_REQUEST_TIMEOUT_SECONDS,
    IDM_TRAIN_CLASSES,
    IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
    load_idm_model_lock,
)
from src.benchmark.worker_process import WorkerProcessError

MODEL_LOCK_PATH = Path(__file__).parents[2] / "runtime" / "idm" / "model.json"


def _audio(input_root: Path, name: str = "song.wav") -> CanonicalAudio:
    content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    input_root.mkdir(parents=True, exist_ok=True)
    path = input_root / name
    path.write_bytes(content)
    digest = sha256_hex(content)
    return CanonicalAudio(path, "song", digest, "view", digest, len(content), 44100, 1, 2, 2)


def _ready(**overrides: object) -> dict[str, object]:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    ready: dict[str, object] = {
        "type": "ready",
        "backend_id": IDM_BACKEND_ID,
        "model_id": lock.model_id,
        "train_classes": list(IDM_TRAIN_CLASSES),
        "sample_rate_hz": 44100,
        "activation_rate_hz": 172.265625,
    }
    ready.update(overrides)
    return ready


class _FakeWorker:
    def __init__(self, ready: dict[str, object], response: dict[str, object]) -> None:
        self.ready = ready
        self.response = response
        self.requests: list[str] = []
        self.close_count = 0

    def request(self, path: str) -> dict[str, object]:
        self.requests.append(path)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def close(self) -> None:
        self.close_count += 1


def _event() -> dict[str, object]:
    return {
        "class_index": 4,
        "native_class_id": "KD",
        "frame_index": 215,
        "time_sec": 1.248526,
        "onset_score": 0.83,
        "native_velocity": 1.337421,
    }


def _backend(
    tmp_path: Path,
    worker: _FakeWorker,
    calls: list[tuple[list[str], dict[str, object]]] | None = None,
    **kwargs: object,
) -> IdmBackend:
    runtime_python = tmp_path / "runtime-python"
    model_root = tmp_path / "model-root"
    input_root = tmp_path / "input"
    model_root.mkdir()
    input_root.mkdir()

    def factory(command: list[str], **factory_kwargs: object) -> _FakeWorker:
        if calls is not None:
            calls.append((command, factory_kwargs))
        return worker

    return IdmBackend(
        runtime_python,
        MODEL_LOCK_PATH,
        model_root,
        input_root,
        process_factory=factory,
        **kwargs,
    )


def test_idm_backend_launches_isolated_python_and_forwards_frozen_timeouts(
    tmp_path: Path,
) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []
    backend = _backend(tmp_path, worker, calls)
    try:
        backend.transcribe(_audio(tmp_path / "input"))
    finally:
        backend.close()

    assert calls == [
        (
            [
                str(tmp_path / "runtime-python"),
                str(Path(__file__).parents[2] / "runtime" / "idm" / "worker.py"),
                "--model-root",
                str(tmp_path / "model-root"),
            ],
            {
                "timeout_seconds": IDM_REQUEST_TIMEOUT_SECONDS,
                "close_timeout_seconds": IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
            },
        )
    ]


def test_idm_backend_decodes_events_and_reuses_worker(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": [_event()]})
    backend = _backend(tmp_path, worker)
    audio = _audio(tmp_path / "input")
    try:
        first = backend.transcribe(audio)
        second = backend.transcribe(audio)
    finally:
        backend.close()

    assert first.events == second.events
    event = first.events[0]
    assert event.native_class_id == "KD"
    assert event.model_output_bin == 4
    assert event.native_midi_note is None
    assert event.native_metadata == {"frame_index": "215", "native_velocity": "1.337421"}
    assert event.confidence == 0.83
    assert event.velocity_midi == round((1.337421 / 2.0) * 127)
    assert worker.requests == [str(audio.path.resolve()), str(audio.path.resolve())]


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "wrong"},
        {"train_classes": ["KD"]},
        {"sample_rate_hz": 48000},
        {"activation_rate_hz": 100.0},
    ],
)
def test_idm_backend_rejects_wrong_ready_identity_before_request(
    tmp_path: Path, override: dict[str, object]
) -> None:
    worker = _FakeWorker(_ready(**override), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "worker_identity_invalid"
    assert worker.requests == []
    assert worker.close_count == 1


def test_idm_backend_maps_worker_protocol_failure_and_poison(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), WorkerProcessError("worker response timed out"))
    backend = _backend(tmp_path, worker)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "worker_protocol_failed"
    assert worker.close_count == 1
    with pytest.raises(IdmBackendError) as poisoned:
        backend.transcribe(_audio(tmp_path / "input", "second.wav"))
    assert poisoned.value.code == "worker_protocol_failed"


def test_idm_backend_maps_worker_start_failure(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()

    def factory(_command: list[str], **_kwargs: object) -> object:
        raise OSError("cannot start")

    backend = IdmBackend(
        tmp_path / "python",
        MODEL_LOCK_PATH,
        tmp_path / "model",
        root,
        process_factory=factory,
    )
    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(root))
    assert raised.value.code == "worker_start_failed"


def test_idm_backend_malformed_event_poisoning(tmp_path: Path) -> None:
    malformed = _event()
    malformed["native_class_id"] = "SD"
    worker = _FakeWorker(_ready(), {"id": "request", "events": [malformed]})
    backend = _backend(tmp_path, worker)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "native_event_invalid"
    assert worker.close_count == 1


def test_idm_backend_close_is_idempotent_and_surfaces_finalization_failure(
    tmp_path: Path,
) -> None:
    class CloseFailingWorker(_FakeWorker):
        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("close failed")

    worker = CloseFailingWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    backend.transcribe(_audio(tmp_path / "input"))

    with pytest.raises(IdmBackendError) as raised:
        backend.close()
    assert raised.value.code == "worker_close_failed"
    backend.close()
    assert worker.close_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_index", -1),
        ("class_index", 9),
        ("frame_index", -1),
        ("onset_score", float("nan")),
        ("onset_score", 1.1),
        ("native_velocity", 0.0),
        ("native_velocity", 2.1),
        ("time_sec", 2.0),
    ],
)
def test_idm_backend_rejects_invalid_raw_event_values(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = _event()
    raw[field] = value
    worker = _FakeWorker(_ready(), {"id": "request", "events": [raw]})
    backend = _backend(tmp_path, worker)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "native_event_invalid"


def test_idm_backend_rejects_input_outside_root(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    outside = _audio(tmp_path / "outside")

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(outside)
    assert raised.value.code == "input_path_invalid"


def test_idm_adapter_revision_is_frozen() -> None:
    assert IDM_ADAPTER_REVISION == "crux.idm-adapter/v1"
