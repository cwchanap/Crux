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
    _decode_native_event,
    _default_runtime_sync,
    _isolated_worker_environment,
    _raise_worker_error,
    _validate_descriptor,
    _validate_ready,
    _validate_wheel_provenance,
    _wheel_sha256_from_runtime_lock,
    build_worker_command,
    descriptor_for_lock,
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

    kwargs.setdefault("runtime_sync", lambda *_: None)
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
        runtime_sync=lambda *_: None,
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


def test_idm_backend_invokes_runtime_sync_before_worker_start(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[Path, Path]] = []
    sync_calls: list[tuple[Path, Path, str]] = []

    def sync(runtime_root: Path, runtime_python: Path, python_version: str) -> None:
        sync_calls.append((runtime_root, runtime_python, python_version))

    backend = _backend(
        tmp_path,
        worker,
        calls,
        artifact_root=artifact_root,
        model_root=model_root,
        runtime_sync=sync,
    )
    try:
        backend.transcribe(_audio(tmp_path / "input"))
    finally:
        backend.close()

    expected_version = load_idm_model_lock(artifact_root / "model.json").python_version
    assert len(sync_calls) == 1
    assert sync_calls[0][0] == artifact_root
    assert sync_calls[0][1] == tmp_path / "runtime-python"
    # The exact frozen interpreter triple must reach the runtime sync.
    assert sync_calls[0][2] == expected_version
    # Worker was started only after the sync succeeded.
    assert len(calls) == 1


def test_idm_backend_poisons_on_runtime_sync_failure(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []

    def failing_sync(_root: Path, _python: Path, _version: str) -> None:
        raise RuntimeError("venv does not match uv.lock")

    backend = _backend(
        tmp_path,
        worker,
        calls,
        artifact_root=artifact_root,
        model_root=model_root,
        runtime_sync=failing_sync,
    )

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "runtime_artifact_invalid"
    assert calls == []
    assert worker.requests == []
    assert backend._poisoned

    # A poisoned backend must not retry the sync on a second transcribe.
    with pytest.raises(IdmBackendError) as again:
        backend.transcribe(_audio(tmp_path / "input"))
    assert again.value.code == "worker_protocol_failed"


def test_idm_backend_preserves_uv_stderr_through_runtime_sync_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uv stderr captured by _default_runtime_sync must survive _ensure_process()'s wrap."""
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")

    def failing_run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(1, args[0], stderr="boom diagnostic")

    monkeypatch.setattr(subprocess, "run", failing_run)

    backend = _backend(
        tmp_path,
        worker,
        calls,
        artifact_root=artifact_root,
        model_root=model_root,
        runtime_sync=lambda root, _python, version: _default_runtime_sync(
            root, root / ".venv" / "bin" / "python", version
        ),
    )

    with pytest.raises(IdmBackendError) as raised:
        backend.transcribe(_audio(tmp_path / "input"))

    assert raised.value.code == "runtime_artifact_invalid"
    assert "boom diagnostic" in str(raised.value)
    assert calls == []


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


# ---------------------------------------------------------------------------
# Coverage tests for backends/idm.py error and edge-case paths
# ---------------------------------------------------------------------------


# --- IdmBackend.__init__ validation ---


def test_idm_backend_init_rejects_non_path(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    with pytest.raises(TypeError, match="runtime_python must be a Path"):
        IdmBackend(
            "not-a-path",  # type: ignore[arg-type]
            artifact_root / "model.json",
            model_root,
            tmp_path / "input",
        )


def test_idm_backend_init_rejects_non_positive_timeout(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        IdmBackend(
            tmp_path / "python",
            artifact_root / "model.json",
            model_root,
            tmp_path / "input",
            timeout_seconds=0,
        )


def test_idm_backend_init_rejects_non_positive_close_timeout(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    with pytest.raises(ValueError, match="close_timeout_seconds must be positive"):
        IdmBackend(
            tmp_path / "python",
            artifact_root / "model.json",
            model_root,
            tmp_path / "input",
            close_timeout_seconds=-1,
        )


def test_idm_backend_init_rejects_invalid_model_lock(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    lock_path = artifact_root / "model.json"
    lock_path.write_bytes(b"invalid\n")
    with pytest.raises(IdmBackendError, match="IDM model lock is invalid") as raised:
        IdmBackend(tmp_path / "python", lock_path, model_root, tmp_path / "input")
    assert raised.value.code == "descriptor_invalid"


# --- IdmBackend descriptor handling ---


def test_idm_backend_descriptor_returns_frozen_descriptor(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    try:
        assert backend.descriptor().payload["backend_id"] == IDM_BACKEND_ID
    finally:
        backend.close()


def test_idm_backend_accepts_matching_explicit_descriptor(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    lock = load_idm_model_lock(artifact_root / "model.json")
    expected = descriptor_for_lock(lock)
    backend = IdmBackend(
        tmp_path / "python",
        artifact_root / "model.json",
        model_root,
        tmp_path / "input",
        process_factory=lambda *a, **kw: worker,
        descriptor=expected,
    )
    try:
        assert backend.descriptor() == expected
    finally:
        backend.close()


def test_idm_backend_rejects_mismatched_explicit_descriptor(tmp_path: Path) -> None:
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    lock = load_idm_model_lock(artifact_root / "model.json")
    wrong = descriptor_for_lock(lock)
    wrong_payload = dict(wrong.payload)
    wrong_payload["model_id"] = "wrong-model"
    from src.benchmark.backend_identity import build_descriptor

    wrong_descriptor = build_descriptor(
        wrong_payload,
        frozenset(wrong_payload),
        "crux.transcription-backend-descriptor/v2",
    )
    with pytest.raises(IdmBackendError, match="descriptor does not match") as raised:
        IdmBackend(
            tmp_path / "python",
            artifact_root / "model.json",
            model_root,
            tmp_path / "input",
            descriptor=wrong_descriptor,
        )
    assert raised.value.code == "descriptor_invalid"


# --- transcribe error paths ---


def test_transcribe_rejects_after_close(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    backend.close()
    with pytest.raises(IdmBackendError, match="backend is closed") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "backend_closed"


def test_transcribe_rejects_non_canonical_audio(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    try:
        with pytest.raises(TypeError, match="audio must be CanonicalAudio"):
            backend.transcribe("not-audio")  # type: ignore[arg-type]
    finally:
        backend.close()


def test_transcribe_reraises_idm_backend_error_without_poison(tmp_path: Path) -> None:
    class IdmErrorWorker(_FakeWorker):
        def request(self, *a, **kw):
            raise IdmBackendError("custom", code="native_event_invalid")

    worker = IdmErrorWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    try:
        with pytest.raises(IdmBackendError, match="custom"):
            backend.transcribe(_audio(tmp_path / "input"))
        assert not backend._poisoned
    finally:
        backend.close()


def test_transcribe_poisons_on_generic_exception(tmp_path: Path) -> None:
    class GenericErrorWorker(_FakeWorker):
        def request(self, *a, **kw):
            raise RuntimeError("unexpected")

    worker = GenericErrorWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    with pytest.raises(IdmBackendError, match="worker request failed") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "worker_protocol_failed"
    assert backend._poisoned


def test_transcribe_poisons_on_non_mapping_response(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), "not-a-mapping")  # type: ignore[arg-type]
    backend = _backend(tmp_path, worker)
    with pytest.raises(IdmBackendError, match="worker response is invalid") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "worker_protocol_failed"
    assert backend._poisoned


def test_transcribe_handles_worker_error_response(tmp_path: Path) -> None:
    worker = _FakeWorker(
        _ready(),
        {"id": "request", "error": {"code": "native_event_invalid", "message": "bad event"}},
    )
    backend = _backend(tmp_path, worker)
    with pytest.raises(IdmBackendError, match="bad event") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "native_event_invalid"
    assert not backend._poisoned


def test_transcribe_poisons_on_protocol_error_response(tmp_path: Path) -> None:
    worker = _FakeWorker(
        _ready(),
        {"id": "request", "error": {"code": "worker_protocol_failed", "message": "proto fail"}},
    )
    backend = _backend(tmp_path, worker)
    with pytest.raises(IdmBackendError, match="proto fail") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "worker_protocol_failed"
    assert backend._poisoned


def test_transcribe_poisons_on_non_list_events(tmp_path: Path) -> None:
    worker = _FakeWorker(_ready(), {"id": "request", "events": "not-a-list"})
    backend = _backend(tmp_path, worker)
    with pytest.raises(IdmBackendError, match="worker response is invalid") as raised:
        backend.transcribe(_audio(tmp_path / "input"))
    assert raised.value.code == "worker_protocol_failed"
    assert backend._poisoned


# --- _ensure_process edge cases ---


def test_ensure_process_poisons_on_non_mapping_ready(tmp_path: Path) -> None:
    class BadReadyWorker(_FakeWorker):
        def __init__(self) -> None:
            super().__init__("not-a-mapping", {"id": "request", "events": []})  # type: ignore[arg-type]

    worker = BadReadyWorker()
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    worker.ready = "not-a-mapping"  # type: ignore[assignment]

    runtime_python = tmp_path / "runtime-python"
    input_root = tmp_path / "input"
    input_root.mkdir(exist_ok=True)

    def factory(*a, **kw):
        return worker

    backend = IdmBackend(
        runtime_python,
        artifact_root / "model.json",
        model_root,
        input_root,
        process_factory=factory,
        runtime_sync=lambda *_: None,
    )
    with pytest.raises(IdmBackendError, match="worker ready response is invalid") as raised:
        backend.transcribe(_audio(input_root))
    assert raised.value.code == "worker_ready_invalid"
    assert backend._poisoned


# --- _poison edge cases ---


def test_poison_is_idempotent_and_swallows_close_exception(tmp_path: Path) -> None:
    class CloseFailingWorker(_FakeWorker):
        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("close failed")

    worker = CloseFailingWorker(_ready(), {"id": "request", "events": []})
    backend = _backend(tmp_path, worker)
    backend.transcribe(_audio(tmp_path / "input"))
    backend._poison()  # should not raise despite close failure
    backend._poison()  # idempotent
    assert worker.close_count == 1


# --- build_worker_command validation ---


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"runtime_python": "x"}, "runtime_python and model_root must be Paths"),
        ({"model_root": "x"}, "runtime_python and model_root must be Paths"),
        ({"wheel_path": "x"}, "wheel_path must be a Path"),
        ({"site_packages": None}, "all attested runtime and model arguments are required"),
        ({"model_config_path": None}, "all attested runtime and model arguments are required"),
        ({"checkpoint_path": None}, "all attested runtime and model arguments are required"),
        ({"site_packages": "x"}, "site_packages and model_config_path must be Paths"),
        ({"model_config_path": "x"}, "site_packages and model_config_path must be Paths"),
        ({"checkpoint_path": "x"}, "checkpoint_path must be a Path"),
        ({"model_config_sha256": 123}, "attested model file identities are invalid"),
        ({"model_config_byte_length": -1}, "attested model file identities are invalid"),
        ({"checkpoint_byte_length": "x"}, "attested model file identities are invalid"),
        ({"wheel_sha256": "not-a-hash"}, "wheel_sha256 must be lowercase SHA-256"),
    ],
)
def test_build_worker_command_rejects_invalid_arguments(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    defaults: dict[str, object] = {
        "runtime_python": tmp_path / "python",
        "model_root": tmp_path / "model",
        "wheel_path": tmp_path / "wheel.whl",
        "wheel_sha256": "a" * 64,
        "site_packages": tmp_path / "site-packages",
        "model_config_path": tmp_path / "config.yaml",
        "model_config_sha256": "b" * 64,
        "model_config_byte_length": 1,
        "checkpoint_path": tmp_path / "model.ckpt",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_byte_length": 1,
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        build_worker_command(**defaults)  # type: ignore[arg-type]


# --- descriptor_for_lock ---


def test_descriptor_for_lock_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="lock must be an IdmModelLock"):
        descriptor_for_lock("not-a-lock")  # type: ignore[arg-type]


# --- _validate_descriptor ---


def test_validate_descriptor_rejects_non_descriptor() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    expected = descriptor_for_lock(lock)
    with pytest.raises(IdmBackendError, match="backend descriptor is invalid"):
        _validate_descriptor("not-a-descriptor", expected)  # type: ignore[arg-type]


def test_validate_descriptor_rejects_mismatch() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    expected = descriptor_for_lock(lock)
    wrong_payload = dict(expected.payload)
    wrong_payload["model_id"] = "wrong"
    from src.benchmark.backend_identity import build_descriptor

    wrong = build_descriptor(
        wrong_payload,
        frozenset(wrong_payload),
        "crux.transcription-backend-descriptor/v2",
    )
    with pytest.raises(IdmBackendError, match="descriptor does not match"):
        _validate_descriptor(wrong, expected)


# --- _validate_ready ---


def _lock() -> object:
    return load_idm_model_lock(MODEL_LOCK_PATH)


@pytest.mark.parametrize(
    ("ready", "code"),
    [
        ("not-a-mapping", "worker_ready_invalid"),
        ({"type": "wrong"}, "worker_ready_invalid"),
        ({"type": "ready"}, "worker_ready_invalid"),
        (
            {
                "type": "ready",
                "backend_id": "wrong",
                "model_id": "x",
                "model_name": "x",
                "train_classes": [],
                "python_version": "x",
                "sample_rate_hz": 44100,
                "activation_rate_hz": 172.265625,
                "device": "cpu",
                "dtype": "float32",
            },
            "worker_identity_invalid",
        ),
    ],
)
def test_validate_ready_rejects_invalid_responses(ready: object, code: str) -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(IdmBackendError) as raised:
        _validate_ready(ready, lock)  # type: ignore[arg-type]
    assert raised.value.code == code


def test_validate_ready_rejects_non_int_sample_rate() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    ready = _ready(sample_rate_hz="44100")
    with pytest.raises(IdmBackendError, match="sample rate is invalid") as raised:
        _validate_ready(ready, lock)
    assert raised.value.code == "worker_identity_invalid"


def test_validate_ready_rejects_invalid_frame_rate() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    ready = _ready(activation_rate_hz="not-a-number")
    with pytest.raises(IdmBackendError, match="frame rate is invalid") as raised:
        _validate_ready(ready, lock)
    assert raised.value.code == "worker_identity_invalid"


# --- _decode_native_event ---


def test_decode_native_event_rejects_non_mapping() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(IdmBackendError, match="worker event is invalid") as raised:
        _decode_native_event("not-a-mapping", lock, 1000)  # type: ignore[arg-type]
    assert raised.value.code == "native_event_invalid"


def test_decode_native_event_rejects_wrong_key_set() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(IdmBackendError, match="worker event is invalid") as raised:
        _decode_native_event({"wrong": "keys"}, lock, 1000)
    assert raised.value.code == "native_event_invalid"


def test_decode_native_event_rejects_time_not_matching_frame() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    event = _event()
    event["time_sec"] = 999.0
    with pytest.raises(IdmBackendError, match="time does not match frame") as raised:
        _decode_native_event(event, lock, 10000)
    assert raised.value.code == "native_event_invalid"


def test_decode_native_event_handles_zero_velocity() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    event = _event()
    event["native_velocity"] = 0.0000001  # rounds to 0 after quantize
    result = _decode_native_event(event, lock, 10000)
    assert result.native_metadata["native_velocity"] == "0"


# --- _raise_worker_error ---


def test_raise_worker_error_with_code_and_message() -> None:
    with pytest.raises(IdmBackendError, match="custom error") as raised:
        _raise_worker_error({"code": "native_event_invalid", "message": "custom error"})
    assert raised.value.code == "native_event_invalid"


def test_raise_worker_error_with_non_mapping() -> None:
    with pytest.raises(IdmBackendError, match="worker response is invalid") as raised:
        _raise_worker_error("not-a-mapping")
    assert raised.value.code == "worker_protocol_failed"


def test_raise_worker_error_with_missing_fields() -> None:
    with pytest.raises(IdmBackendError, match="worker response is invalid") as raised:
        _raise_worker_error({"code": 123, "message": "bad"})
    assert raised.value.code == "worker_protocol_failed"


# --- _wheel_sha256_from_runtime_lock ---


def _make_toml(packages: list[dict]) -> bytes:
    lines: list[str] = []
    for pkg in packages:
        lines.append("[[package]]")
        lines.append(f'name = "{pkg.get("name", "inverse-drum-machine")}"')
        lines.append(f'version = "{pkg.get("version", "0.1.0")}"')
        if "source" in pkg:
            lines.append(f'source = {{ path = "{pkg["source"]}" }}')
        if "wheels" in pkg:
            wheel_lines = []
            for wheel in pkg["wheels"]:
                parts = [f'filename = "{wheel["filename"]}"']
                parts.append(f'hash = "{wheel["hash"]}"')
                wheel_lines.append("{ " + ", ".join(parts) + " }")
            lines.append("wheels = [\n    " + ",\n    ".join(wheel_lines) + ",\n]")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def test_wheel_sha256_rejects_multiple_matches() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    toml = _make_toml([{}, {}])
    with pytest.raises(ValueError, match="exactly one IDM package"):
        _wheel_sha256_from_runtime_lock(toml, lock)


def test_wheel_sha256_rejects_wrong_source_path() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    toml = _make_toml(
        [{"source": "wrong/path", "wheels": [{"filename": "x.whl", "hash": "sha256:" + "a" * 64}]}]
    )
    with pytest.raises(ValueError, match="source path is invalid"):
        _wheel_sha256_from_runtime_lock(toml, lock)


def test_wheel_sha256_rejects_invalid_hash_format() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    toml = _make_toml(
        [
            {
                "source": f"wheels/{WHEEL_NAME}",
                "wheels": [{"filename": WHEEL_NAME, "hash": "not-a-hash"}],
            }
        ]
    )
    with pytest.raises(ValueError, match="wheel digest is invalid"):
        _wheel_sha256_from_runtime_lock(toml, lock)


# --- _validate_wheel_provenance ---


def _valid_wheel_bytes() -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("idm/__init__.py", b"hello")
    return buf.getvalue()


def _valid_provenance(
    wheel_name: str = WHEEL_NAME,
    wheel_sha: str = "a" * 64,
    wheel_bytes: bytes | None = None,
) -> bytes:
    if wheel_bytes is None:
        wheel_bytes = _valid_wheel_bytes()
    return (
        json.dumps(
            {
                "schema": "crux.idm-wheel-provenance/v1",
                "source_commit": "456656868538205ef756912c7cf5b0fd936de8af",
                "package_name": "inverse-drum-machine",
                "package_version": "0.1.0",
                "wheel": {
                    "path": wheel_name,
                    "sha256": wheel_sha,
                    "byte_length": len(wheel_bytes),
                    "tag": "py3-none-any",
                },
                "packaged_idm_files": [
                    {
                        "path": "idm/__init__.py",
                        "sha256": hashlib.sha256(b"hello").hexdigest(),
                        "byte_length": 5,
                    },
                ],
            }
        ).encode("utf-8")
        + b"\n"
    )


def test_validate_wheel_provenance_rejects_bad_newline() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    content = _valid_provenance()[:-1]  # no trailing newline
    with pytest.raises(ValueError, match="newline is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_invalid_json() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(ValueError, match="provenance is invalid"):
        _validate_wheel_provenance(b"not json\n", WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_non_dict() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(ValueError, match="provenance is invalid"):
        _validate_wheel_provenance(b"[]\n", WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_wrong_identity() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["source_commit"] = "0" * 40
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="provenance identity is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_wrong_wheel_record() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["wheel"]["tag"] = "wrong-tag"
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="wheel record is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_empty_inventory() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["packaged_idm_files"] = []
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="package inventory is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_inventory_not_matching_wheel() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    wheel_bytes = _valid_wheel_bytes()
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["wheel"]["byte_length"] = len(wheel_bytes)
    provenance["packaged_idm_files"][0]["byte_length"] = 999  # wrong length
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="package inventory does not match"):
        _validate_wheel_provenance(content, WHEEL_NAME, wheel_bytes, "a" * 64, lock)


# --- Additional coverage for remaining edge cases ---


def test_ensure_process_poisons_on_missing_ready_attribute(tmp_path: Path) -> None:
    class NoReadyWorker:
        response = {"id": "request", "events": []}
        requests: list[str] = []
        close_count = 0

        def close(self) -> None:
            self.close_count += 1

    worker = NoReadyWorker()
    artifact_root, model_root = _copy_attested_runtime(tmp_path)
    input_root = tmp_path / "input"
    input_root.mkdir(exist_ok=True)

    def factory(*a, **kw):
        return worker

    backend = IdmBackend(
        tmp_path / "python",
        artifact_root / "model.json",
        model_root,
        input_root,
        process_factory=factory,
        runtime_sync=lambda *_: None,
    )
    with pytest.raises(IdmBackendError, match="worker ready response is invalid") as raised:
        backend.transcribe(_audio(input_root))
    assert raised.value.code == "worker_ready_invalid"
    assert backend._poisoned


def test_build_worker_command_uses_default_wheel_path(tmp_path: Path) -> None:
    command = build_worker_command(
        tmp_path / "python",
        tmp_path / "model",
        wheel_sha256="a" * 64,
        site_packages=tmp_path / "site-packages",
        model_config_path=tmp_path / "config.yaml",
        model_config_sha256="b" * 64,
        model_config_byte_length=1,
        checkpoint_path=tmp_path / "model.ckpt",
        checkpoint_sha256="c" * 64,
        checkpoint_byte_length=1,
    )
    assert "--wheel-path" in command
    idx = command.index("--wheel-path")
    assert command[idx + 1] == str(
        Path(__file__).resolve().parents[2] / "runtime" / "idm" / WHEEL_NAME
    )


def test_decode_native_event_rejects_non_float_time() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    event = _event()
    event["time_sec"] = 1  # int, not float
    with pytest.raises(IdmBackendError, match="worker event time is invalid") as raised:
        _decode_native_event(event, lock, 10000)
    assert raised.value.code == "native_event_invalid"


def test_decode_native_event_rejects_nonfinite_velocity() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    event = _event()
    event["native_velocity"] = float("inf")
    with pytest.raises(IdmBackendError, match="worker event velocity is invalid") as raised:
        _decode_native_event(event, lock, 10000)
    assert raised.value.code == "native_event_invalid"


def test_activation_frame_limit_rejects_invalid_frame_count(tmp_path: Path) -> None:
    from src.benchmark.backends.idm import _activation_frame_limit

    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    audio = CanonicalAudio(
        Path(),
        "song",
        "d" * 64,
        "view",
        "c" * 64,
        46,
        44100,
        1,
        2,
        "not-int",  # type: ignore[arg-type]
    )
    with pytest.raises(IdmBackendError, match="frame count is invalid") as raised:
        _activation_frame_limit(audio, lock)
    assert raised.value.code == "input_audio_invalid"


def test_runtime_site_packages_rejects_invalid_types() -> None:
    from src.benchmark.backends.idm import _runtime_site_packages

    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(TypeError, match="runtime_python and lock have invalid types"):
        _runtime_site_packages("not-a-path", lock)  # type: ignore[arg-type]


def test_wheel_sha256_rejects_wrong_wheel_filename() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    toml = _make_toml(
        [
            {
                "source": f"wheels/{WHEEL_NAME}",
                "wheels": [{"filename": "wrong.whl", "hash": "sha256:" + "a" * 64}],
            }
        ]
    )
    with pytest.raises(ValueError, match="wheel identity is invalid"):
        _wheel_sha256_from_runtime_lock(toml, lock)


def test_wheel_sha256_rejects_malformed_toml() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(ValueError, match="wheel identity is invalid"):
        _wheel_sha256_from_runtime_lock(b"not valid toml = \n", lock)


def test_wheel_sha256_rejects_missing_package_key() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    with pytest.raises(ValueError, match="wheel identity is invalid"):
        _wheel_sha256_from_runtime_lock(b'other = "value"\n', lock)


def test_validate_wheel_provenance_rejects_non_dict_wheel() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["wheel"] = "not-a-dict"
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="wheel record is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_non_dict_inventory_record() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["packaged_idm_files"] = ["not-a-dict"]
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="package inventory is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_invalid_inventory_path() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    provenance = json.loads(_valid_provenance()[:-1])
    provenance["packaged_idm_files"][0]["path"] = "not-idm/file.py"
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="package inventory is invalid"):
        _validate_wheel_provenance(content, WHEEL_NAME, _valid_wheel_bytes(), "a" * 64, lock)


def test_validate_wheel_provenance_rejects_inventory_names_mismatch() -> None:
    lock = load_idm_model_lock(MODEL_LOCK_PATH)
    wheel_bytes = _valid_wheel_bytes()
    provenance = json.loads(_valid_provenance(wheel_bytes=wheel_bytes)[:-1])
    provenance["packaged_idm_files"].append(
        {"path": "idm/extra.py", "sha256": "d" * 64, "byte_length": 1}
    )
    content = (json.dumps(provenance) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="package inventory does not match"):
        _validate_wheel_provenance(content, WHEEL_NAME, wheel_bytes, "a" * 64, lock)


# --- _default_runtime_sync ---


def test_default_runtime_sync_succeeds_when_venv_matches_and_uv_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing venv must not trigger --offline (recovery from partial install).

    uv creates ``.venv/bin/python`` before the full dependency set is installed,
    so the venv interpreter existing is not proof the environment is complete.
    The sync must stay online-capable so a partially-installed runtime can be
    reconciled by fetching the missing packages from the registry.
    """
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")

    captured_commands: list[list[str]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        assert kwargs.get("timeout") is not None, "uv sync must be bounded by a timeout"
        command = args[0]
        assert isinstance(command, list)
        captured_commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _default_runtime_sync(runtime_root, venv_python, "3.11.12")

    assert len(captured_commands) == 1
    assert captured_commands[0][0] == "/usr/local/bin/uv"
    assert "--project" in captured_commands[0]
    assert str(runtime_root) in captured_commands[0]
    assert "--frozen" in captured_commands[0]
    assert "--offline" not in captured_commands[0]
    # The exact frozen interpreter triple must pin uv's choice; the pyproject
    # only constrains ==3.11.* and an ambient UV_PYTHON would win otherwise.
    assert captured_commands[0][-2:] == ["--python", "3.11.12"]


def test_default_runtime_sync_rejects_unresolvable_runtime_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sync that fails to create the venv must not pass post-sync verification."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_python = runtime_root / ".venv" / "bin" / "python"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0),
    )

    with pytest.raises(RuntimeError, match="runtime python is unavailable"):
        _default_runtime_sync(runtime_root, runtime_python, "3.11.12")


def test_default_runtime_sync_rejects_mismatched_venv_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    expected = runtime_root / ".venv" / "bin" / "python"
    expected.parent.mkdir(parents=True)
    expected.write_text("#!/bin/sh\n", encoding="utf-8")

    other = tmp_path / "other-venv" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="runtime python must point to the locked project venv"):
        _default_runtime_sync(runtime_root, other, "3.11.12")


def test_default_runtime_sync_rejects_missing_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="uv is not available on PATH"):
        _default_runtime_sync(runtime_root, venv_python, "3.11.12")


def test_default_runtime_sync_rejects_uv_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args[0])),
    )

    with pytest.raises(RuntimeError, match="uv sync --frozen failed"):
        _default_runtime_sync(runtime_root, venv_python, "3.11.12")


def test_default_runtime_sync_rejects_uv_sync_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(RuntimeError, match="uv sync --frozen failed"):
        _default_runtime_sync(runtime_root, venv_python, "3.11.12")


def test_default_runtime_sync_strips_uv_override_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited uv overrides must not subvert the mandatory frozen sync.

    ``UV_PROJECT_ENVIRONMENT`` would redirect the sync to another env;
    ``UV_NO_SYNC`` makes uv skip environment updates entirely (no-op'ing the
    sync); ``UV_OFFLINE`` blocks the registry fetches a partial install needs.
    """
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "other-env"))
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert "UV_PROJECT_ENVIRONMENT" not in env
        assert "UV_NO_SYNC" not in env
        assert "UV_OFFLINE" not in env
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _default_runtime_sync(runtime_root, venv_python, "3.11.12")


def test_default_runtime_sync_rejects_symlink_resolved_base_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolve()-equivalent base interpreter must not pass the lexical guard.

    uv symlinks ``.venv/bin/python`` to the underlying managed interpreter; the
    guard must compare lexically so a caller cannot pass that base interpreter
    directly and have the worker derive site-packages from the wrong root.
    """
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    # Create the venv python as a symlink to a base interpreter, mirroring uv.
    base_interpreter = tmp_path / "base-python"
    base_interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.symlink_to(base_interpreter)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")

    with pytest.raises(RuntimeError, match="runtime python must point to the locked project venv"):
        _default_runtime_sync(runtime_root, base_interpreter, "3.11.12")


def test_default_runtime_sync_bootstraps_fresh_checkout_without_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh checkout (no venv) syncs without --offline and materializes the venv."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    venv_python = runtime_root / ".venv" / "bin" / "python"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        assert kwargs.get("timeout") is not None, "uv sync must be bounded by a timeout"
        command = args[0]
        assert isinstance(command, list)
        assert "--frozen" in command
        assert "--offline" not in command
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _default_runtime_sync(runtime_root, venv_python, "3.11.12")

    assert venv_python.exists()


def test_default_runtime_sync_reports_captured_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")

    def fake_run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(1, args[0], stderr="boom diagnostic")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="uv sync --frozen failed") as raised:
        _default_runtime_sync(runtime_root, venv_python, "3.11.12")
    assert "boom diagnostic" in str(raised.value)


def test_default_runtime_sync_wraps_sync_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    venv_python = runtime_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], 1)),
    )

    with pytest.raises(RuntimeError, match="uv sync --frozen timed out"):
        _default_runtime_sync(runtime_root, venv_python, "3.11.12")
