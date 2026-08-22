from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import IDM_BACKEND_ID, canonical_json_bytes, sha256_hex
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.backends.idm import (
    IDM_ADAPTER_REVISION,
    IdmBackend,
    IdmBackendError,
    _attest_runtime_artifacts,
    _isolated_worker_environment,
    build_worker_command,
)
from src.benchmark.idm_model import (
    IDM_REQUEST_TIMEOUT_SECONDS,
    IDM_TRAIN_CLASSES,
    IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
    load_idm_model_lock,
)
from src.benchmark.worker_process import WorkerProcessError

REPOSITORY_ROOT = Path(__file__).parents[2]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime" / "idm"
MODEL_LOCK_PATH = RUNTIME_ROOT / "model.json"
WHEEL_NAME = "inverse_drum_machine-0.1.0-py3-none-any.whl"
MODEL_RELATIVE_PATHS = (
    "pretrained/idm-44-train-kits/checkpoints/model.yaml",
    "pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt",
)
SYNTHETIC_CONFIG = b"synthetic IDM model config\n"
SYNTHETIC_CHECKPOINT = b"synthetic IDM checkpoint\n"
WHEEL_SHA256 = hashlib.sha256((RUNTIME_ROOT / "wheels" / WHEEL_NAME).read_bytes()).hexdigest()


def _audio(
    input_root: Path,
    name: str = "song.wav",
    *,
    frame_count: int = 440744,
) -> CanonicalAudio:
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
    return CanonicalAudio(
        path, "song", digest, "view", digest, len(content), 44100, 1, 2, frame_count
    )


def _ready(lock_path: Path = MODEL_LOCK_PATH, **overrides: object) -> dict[str, object]:
    lock = load_idm_model_lock(lock_path)
    ready: dict[str, object] = {
        "type": "ready",
        "backend_id": IDM_BACKEND_ID,
        "model_id": lock.model_id,
        "model_name": lock.model_name,
        "train_classes": list(IDM_TRAIN_CLASSES),
        "python_version": lock.python_version,
        "sample_rate_hz": 44100,
        "activation_rate_hz": 172.265625,
        "device": lock.device,
        "dtype": lock.dtype,
    }
    ready.update(overrides)
    return ready


class _FakeWorker:
    def __init__(self, ready: dict[str, object], response: dict[str, object]) -> None:
        self.ready = ready
        self.response = response
        self.requests: list[str] = []
        self.request_identities: list[tuple[int | None, str | None]] = []
        self.close_count = 0

    def request(
        self,
        path: str,
        *,
        audio_byte_length: int | None = None,
        audio_sha256: str | None = None,
    ) -> dict[str, object]:
        self.requests.append(path)
        self.request_identities.append((audio_byte_length, audio_sha256))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def close(self) -> None:
        self.close_count += 1


def _event(*, frame_index: int = 215) -> dict[str, object]:
    return {
        "class_index": 4,
        "native_class_id": "KD",
        "frame_index": frame_index,
        "time_sec": frame_index / 172.265625,
        "onset_score": 0.83,
        "native_velocity": 1.337421,
    }


def _copy_attested_runtime(tmp_path: Path) -> tuple[Path, Path]:
    artifact_root = tmp_path / "runtime"
    wheel_root = artifact_root / "wheels"
    wheel_root.mkdir(parents=True)
    for name in ("uv.lock", "idm-wheel-provenance.json"):
        shutil.copyfile(RUNTIME_ROOT / name, artifact_root / name)
    shutil.copyfile(RUNTIME_ROOT / "wheels" / WHEEL_NAME, wheel_root / WHEEL_NAME)

    config_sha256 = hashlib.sha256(SYNTHETIC_CONFIG).hexdigest()
    checkpoint_sha256 = hashlib.sha256(SYNTHETIC_CHECKPOINT).hexdigest()
    lock_payload = json.loads((RUNTIME_ROOT / "model.json").read_text(encoding="utf-8"))
    lock_payload["activation_rate_hz"] = Decimal("172.265625")
    lock_payload["velocity_exponent"] = Decimal(str(lock_payload["velocity_exponent"]))
    lock_payload["velocity_max_value"] = Decimal("2.0")
    lock_payload["velocity_threshold"] = Decimal(str(lock_payload["velocity_threshold"]))
    lock_payload.update(
        {
            "model_config_sha256": config_sha256,
            "model_config_byte_length": len(SYNTHETIC_CONFIG),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_byte_length": len(SYNTHETIC_CHECKPOINT),
            "model_id": f"idm-44-train-kits-456656868538-{checkpoint_sha256[:12]}",
        }
    )
    (artifact_root / "model.json").write_bytes(
        canonical_json_bytes(lock_payload, trailing_newline=True)
    )

    model_root = tmp_path / "model-root"
    for relative, content in (
        (MODEL_RELATIVE_PATHS[0], SYNTHETIC_CONFIG),
        (MODEL_RELATIVE_PATHS[1], SYNTHETIC_CHECKPOINT),
    ):
        destination = model_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return artifact_root, model_root


def _backend(
    tmp_path: Path,
    worker: _FakeWorker,
    calls: list[tuple[list[str], dict[str, object]]] | None = None,
    *,
    artifact_root: Path | None = None,
    model_root: Path | None = None,
    **kwargs: object,
) -> IdmBackend:
    runtime_python = tmp_path / "runtime-python"
    input_root = tmp_path / "input"
    if artifact_root is None:
        artifact_root, default_model_root = _copy_attested_runtime(tmp_path)
        model_lock_path = artifact_root / "model.json"
        model_root = default_model_root if model_root is None else model_root
    else:
        model_lock_path = artifact_root / "model.json"
        model_root = artifact_root.parent / "model-root" if model_root is None else model_root
    model_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(exist_ok=True)

    expected_model_id = load_idm_model_lock(model_lock_path).model_id
    source_model_id = load_idm_model_lock(MODEL_LOCK_PATH).model_id
    if worker.ready.get("model_id") == source_model_id:
        worker.ready["model_id"] = expected_model_id

    def factory(command: list[str], **factory_kwargs: object) -> _FakeWorker:
        if calls is not None:
            calls.append((command, factory_kwargs))
        return worker

    return IdmBackend(
        runtime_python,
        model_lock_path,
        model_root,
        input_root,
        process_factory=factory,
        **kwargs,
    )


def test_idm_backend_launches_isolated_python_and_forwards_frozen_timeouts(
    tmp_path: Path,
) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []
    backend = _backend(
        tmp_path,
        worker,
        calls,
        artifact_root=artifact_root,
        model_root=model_root,
    )
    try:
        backend.transcribe(_audio(tmp_path / "input"))
    finally:
        backend.close()

    assert calls == [
        (
            [
                str(tmp_path / "runtime-python"),
                "-I",
                "-S",
                str(RUNTIME_ROOT / "worker.py"),
                "--model-root",
                str(model_root),
                "--site-packages",
                str(tmp_path.parent / "lib" / "python3.11" / "site-packages"),
                "--wheel-path",
                str(artifact_root / "wheels" / WHEEL_NAME),
                "--wheel-sha256",
                WHEEL_SHA256,
                "--model-config-path",
                str(model_root / MODEL_RELATIVE_PATHS[0]),
                "--model-config-sha256",
                hashlib.sha256(SYNTHETIC_CONFIG).hexdigest(),
                "--model-config-byte-length",
                str(len(SYNTHETIC_CONFIG)),
                "--checkpoint-path",
                str(model_root / MODEL_RELATIVE_PATHS[1]),
                "--checkpoint-sha256",
                hashlib.sha256(SYNTHETIC_CHECKPOINT).hexdigest(),
                "--checkpoint-byte-length",
                str(len(SYNTHETIC_CHECKPOINT)),
            ],
            {
                "timeout_seconds": IDM_REQUEST_TIMEOUT_SECONDS,
                "close_timeout_seconds": IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
                "env": _isolated_worker_environment(),
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
    assert worker.request_identities == [
        (audio.byte_length, audio.input_audio_sha256),
        (audio.byte_length, audio.input_audio_sha256),
    ]


@pytest.mark.parametrize("replacement", ("leaf", "ancestor", "none"))
def test_idm_worker_reads_only_verified_staged_audio_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    from runtime.idm import worker

    original = b"verified wav bytes"
    substituted = b"substituted wav bytes"
    input_root = tmp_path / "inputs"
    staged_parent = input_root / "20"
    staged_parent.mkdir(parents=True)
    staged_path = staged_parent / "verified.wav"
    staged_path.write_bytes(original)

    if replacement == "leaf":
        staged_path.unlink()
        staged_path.symlink_to(tmp_path / "substituted.wav")
        (tmp_path / "substituted.wav").write_bytes(substituted)
    elif replacement == "ancestor":
        staged_parent.rename(tmp_path / "original-parent")
        (tmp_path / "substituted-parent").mkdir()
        (tmp_path / "substituted-parent" / "verified.wav").write_bytes(substituted)
        staged_parent.symlink_to(tmp_path / "substituted-parent", target_is_directory=True)

    soundfile_calls: list[str] = []

    class FakeSoundFile:
        LibsndfileError = RuntimeError

        @staticmethod
        def info(stream):
            assert not isinstance(stream, (str, Path))
            assert stream.read() == original
            soundfile_calls.append("info")
            return type(
                "Info",
                (),
                {"format": "WAV", "subtype": "PCM_16", "samplerate": 44100, "channels": 1},
            )()

        @staticmethod
        def read(stream, *, dtype, always_2d):
            assert dtype == "float32"
            assert always_2d is True
            stream.seek(0)
            assert stream.read() == original
            soundfile_calls.append("read")

            class Samples:
                shape = (1, 1)

                def __getitem__(self, key):
                    del key
                    return object()

            return Samples(), 44100

    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)

    class FakeTensor:
        def to(self, _device):
            return self

        def unsqueeze(self, _axis):
            return self

    class FakeTorch:
        @staticmethod
        def from_numpy(_samples):
            return FakeTensor()

    if replacement == "none":
        result = worker._load_audio(
            str(staged_path),
            len(original),
            hashlib.sha256(original).hexdigest(),
            FakeTorch(),
            "cpu",
        )
        assert isinstance(result, FakeTensor)
        assert soundfile_calls == ["info", "read"]
        return

    with pytest.raises(ValueError, match="audio input or format is invalid"):
        worker._load_audio(
            str(staged_path),
            len(original),
            hashlib.sha256(original).hexdigest(),
            FakeTorch(),
            "cpu",
        )
    assert soundfile_calls == []


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({"id": "request", "audio_path": "/tmp/verified.wav"}, "request is invalid"),
        (
            {
                "id": "request",
                "audio_path": "/tmp/verified.wav",
                "audio_byte_length": True,
                "audio_sha256": "0" * 64,
            },
            "audio byte length is invalid",
        ),
        (
            {
                "id": "request",
                "audio_path": "/tmp/verified.wav",
                "audio_byte_length": 1,
                "audio_sha256": "not-a-sha",
            },
            "audio digest is invalid",
        ),
    ],
)
def test_idm_worker_validates_audio_identity_protocol(
    payload: dict[str, object], expected_message: str
) -> None:
    from runtime.idm import worker

    with pytest.raises(ValueError, match=expected_message):
        worker._valid_request(payload)


@pytest.mark.parametrize(
    ("kind", "expected_message"),
    [
        ("mismatch", "audio bytes do not match verified identity"),
        ("directory", "audio input is not regular"),
    ],
)
def test_idm_worker_rejects_staged_audio_identity_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, expected_message: str
) -> None:
    from runtime.idm import worker

    original = b"verified wav bytes"
    staged_path = tmp_path / "verified.wav"
    staged_path.write_bytes(original)
    if kind == "directory":
        staged_path.unlink()
        staged_path.mkdir()

    class FakeTorch:
        pass

    monkeypatch.setitem(sys.modules, "soundfile", object())
    with pytest.raises(ValueError, match=expected_message):
        worker._load_audio(
            str(staged_path),
            len(original) + (1 if kind == "mismatch" else 0),
            hashlib.sha256(original).hexdigest(),
            FakeTorch(),
            "cpu",
        )


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "wrong"},
        {"model_name": "wrong"},
        {"train_classes": ["KD"]},
        {"python_version": "3.11.11"},
        {"sample_rate_hz": 48000},
        {"activation_rate_hz": 100.0},
        {"device": "mps"},
        {"dtype": "float16"},
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


@pytest.mark.parametrize(("field", "value"), [("device", "mps"), ("dtype", "float16")])
def test_idm_backend_rejects_non_kiss_runtime_lock_before_worker_start(
    tmp_path: Path, field: str, value: str
) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    lock_path = artifact_root / "model.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["activation_rate_hz"] = Decimal("172.265625")
    payload["velocity_exponent"] = Decimal(str(payload["velocity_exponent"]))
    payload["velocity_max_value"] = Decimal("2.0")
    payload["velocity_threshold"] = Decimal(str(payload["velocity_threshold"]))
    payload[field] = value
    lock_path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))

    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    factory_calls: list[object] = []

    def factory(*_args: object, **_kwargs: object) -> _FakeWorker:
        factory_calls.append(1)
        return worker

    with pytest.raises(IdmBackendError) as raised:
        IdmBackend(
            tmp_path / "runtime-python",
            lock_path,
            model_root,
            tmp_path / "input",
            process_factory=factory,
        )

    assert raised.value.code == "descriptor_invalid"
    assert factory_calls == []


def test_idm_worker_ready_reports_effective_runtime_model_and_tensor_facts(
    tmp_path: Path,
) -> None:
    from runtime.idm import worker

    class Parameter:
        device = "cpu"
        dtype = "float32"

    class Encoder:
        sampling_rate = 44100
        frame_rate = 172.265625

    class Model:
        model_name = "idm-44-train-kits"
        train_classes = list(IDM_TRAIN_CLASSES)
        encoder = Encoder()

        def parameters(self):
            return iter((Parameter(),))

    output = io.StringIO()
    result = worker.serve_requests(
        io.StringIO(),
        output,
        model_root=tmp_path,
        model_loader=lambda _root: (Model(), object(), "cpu"),
    )

    assert result == 0
    ready = json.loads(output.getvalue())
    assert ready["model_name"] == "idm-44-train-kits"
    assert ready["train_classes"] == list(IDM_TRAIN_CLASSES)
    expected_python_version = ".".join(str(part) for part in sys.version_info[:3])
    assert ready["python_version"] == expected_python_version
    assert ready["sample_rate_hz"] == 44100
    assert ready["activation_rate_hz"] == 172.265625
    assert ready["device"] == "cpu"
    assert ready["dtype"] == "float32"


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
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    root = tmp_path / "input"
    root.mkdir()

    def factory(_command: list[str], **_kwargs: object) -> object:
        raise OSError("cannot start")

    backend = IdmBackend(
        tmp_path / "python",
        artifact_root / "model.json",
        model_root,
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


@pytest.mark.parametrize("artifact", ["wheel", "provenance", "runtime", "model"])
def test_idm_backend_rejects_tampered_attested_artifacts_before_worker_start(
    tmp_path: Path, artifact: str
) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    if artifact == "wheel":
        (artifact_root / "wheels" / WHEEL_NAME).write_bytes(b"tampered wheel")
    elif artifact == "provenance":
        provenance_path = artifact_root / "idm-wheel-provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["source_commit"] = "0" * 40
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif artifact == "runtime":
        (artifact_root / "uv.lock").write_bytes((artifact_root / "uv.lock").read_bytes() + b"\n")
    else:
        checkpoint = model_root / MODEL_RELATIVE_PATHS[1]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []
    backend = _backend(tmp_path, worker, calls, artifact_root=artifact_root, model_root=model_root)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "runtime_artifact_invalid"
    assert calls == []
    assert worker.requests == []


def test_idm_backend_binds_worker_import_to_attested_wheel_path(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []
    backend = _backend(tmp_path, worker, calls, artifact_root=artifact_root, model_root=model_root)
    try:
        backend.transcribe(_audio(tmp_path / "input"))
    finally:
        backend.close()

    command = calls[0][0]
    assert command[command.index("--wheel-path") + 1] == str(artifact_root / "wheels" / WHEEL_NAME)
    assert command[command.index("--model-root") + 1] == str(model_root)
    assert str(model_root / "idm") not in command


def test_idm_backend_accepts_exact_activation_frame_boundary(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    audio = _audio(tmp_path / "input", frame_count=440744)
    frame_limit = math.ceil(audio.audio_frame_count / 256) + 13
    worker = _FakeWorker(
        _ready(), {"id": "request", "events": [_event(frame_index=frame_limit - 1)]}
    )
    backend = _backend(tmp_path, worker, artifact_root=artifact_root, model_root=model_root)

    try:
        prediction = backend.transcribe(audio)
    finally:
        backend.close()

    assert prediction.events[0].native_metadata["frame_index"] == str(frame_limit - 1)


def test_idm_backend_rejects_one_past_activation_frame_boundary(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    audio = _audio(tmp_path / "input", frame_count=440744)
    frame_limit = math.ceil(audio.audio_frame_count / 256) + 13
    worker = _FakeWorker(_ready(), {"id": "request", "events": [_event(frame_index=frame_limit)]})
    backend = _backend(tmp_path, worker, artifact_root=artifact_root, model_root=model_root)

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(audio)

    assert raised.value.code == "native_event_invalid"


def test_idm_worker_stages_verified_model_bytes_before_source_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.idm import worker

    config_path = tmp_path / "model.yaml"
    checkpoint_path = tmp_path / "model.ckpt"
    config_path.write_bytes(SYNTHETIC_CONFIG)
    checkpoint_path.write_bytes(SYNTHETIC_CHECKPOINT)
    original_read = worker._read_attested_file

    def read_then_swap(path: Path, digest: str, length: int) -> bytes:
        content = original_read(path, digest, length)
        if path == config_path:
            path.write_bytes(b"swapped after host preflight")
        return content

    monkeypatch.setattr(worker, "_read_attested_file", read_then_swap)
    project_root = tmp_path / "project"
    worker._stage_model_files(
        project_root,
        config_path,
        hashlib.sha256(SYNTHETIC_CONFIG).hexdigest(),
        len(SYNTHETIC_CONFIG),
        checkpoint_path,
        hashlib.sha256(SYNTHETIC_CHECKPOINT).hexdigest(),
        len(SYNTHETIC_CHECKPOINT),
    )

    staged_config = project_root / MODEL_RELATIVE_PATHS[0]
    staged_checkpoint = project_root / MODEL_RELATIVE_PATHS[1]
    assert staged_config.read_bytes() == SYNTHETIC_CONFIG
    assert staged_checkpoint.read_bytes() == SYNTHETIC_CHECKPOINT
    for staged in (staged_config, staged_checkpoint):
        assert not staged.is_symlink()
        assert stat.S_IMODE(staged.stat().st_mode) == 0o600


def test_idm_worker_rejects_source_swap_after_host_preflight(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    lock_path = artifact_root / "model.json"
    lock = load_idm_model_lock(lock_path)
    attested = _attest_runtime_artifacts(lock, lock_path, model_root)
    model_config_path = model_root / MODEL_RELATIVE_PATHS[0]
    model_config_path.write_bytes(b"swapped after host preflight")

    from runtime.idm import worker

    with pytest.raises(ValueError, match="attested file"):
        worker._stage_model_files(
            tmp_path / "project",
            attested[2],
            attested[3],
            attested[4],
            attested[5],
            attested[6],
            attested[7],
        )


def test_idm_worker_command_blocks_pythonpath_sitecustomize_preload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    marker = tmp_path / "preloaded"
    (attacker_root / "idm").mkdir()
    (attacker_root / "idm" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('idm')\n",
        encoding="utf-8",
    )
    (attacker_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('sitecustomize')\nimport idm\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, sys\n"
        "print(json.dumps({'idm': any(name == 'idm' or name.startswith('idm.') "
        "for name in sys.modules), 'sitecustomize': 'sitecustomize' in sys.modules}))\n",
        encoding="utf-8",
    )
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        monkeypatch.setenv(key, str(attacker_root))
    command = build_worker_command(
        Path(sys.executable),
        tmp_path / "model-root",
        wheel_path=tmp_path / WHEEL_NAME,
        wheel_sha256=WHEEL_SHA256,
        site_packages=tmp_path / "site-packages",
        model_config_path=tmp_path / "model.yaml",
        model_config_sha256="a" * 64,
        model_config_byte_length=1,
        checkpoint_path=tmp_path / "model.ckpt",
        checkpoint_sha256="b" * 64,
        checkpoint_byte_length=1,
    )
    command[command.index(str(RUNTIME_ROOT / "worker.py"))] = str(probe)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert command[1:3] == ["-I", "-S"]
    assert json.loads(result.stdout) == {"idm": False, "sitecustomize": False}
    assert not marker.exists()
    isolated = _isolated_worker_environment()
    assert all(key not in isolated for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"))
