"""Host adapter for the isolated Inverse Drum Machine worker."""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.benchmark.backend_identity import (
    IDM_BACKEND_ID,
    IDM_DESCRIPTOR_IDENTITIES,
    IDM_DESCRIPTOR_KEYS,
    IDM_DESCRIPTOR_SCHEMA,
    BackendDescriptor,
    StrictJsonError,
    build_descriptor,
    normalize_known_backend_descriptor,
    quantize_six,
)
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.backends.oaf import _build_factory_kwargs
from src.benchmark.idm_model import (
    IDM_REQUEST_TIMEOUT_SECONDS,
    IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
    IdmModelLock,
    IdmModelLockError,
    load_idm_model_lock,
)
from src.benchmark.worker_process import WorkerProcess, WorkerProcessError

IDM_ADAPTER_REVISION = "crux.idm-adapter/v1"
IDM_WORKER_PATH = Path(__file__).resolve().parents[3] / "runtime" / "idm" / "worker.py"
IDM_TIME_TOLERANCE_FRAMES = 0.5


class IdmBackendError(RuntimeError):
    """The IDM worker or its host-boundary contract failed."""

    def __init__(self, message: str, *, code: str = "idm_backend_error") -> None:
        self.code = code
        super().__init__(message)


ProcessFactory = Callable[..., Any]


class IdmBackend:
    """Adapt one persistent isolated IDM worker to the benchmark protocol."""

    def __init__(
        self,
        runtime_python: Path,
        model_lock_path: Path,
        model_root: Path,
        input_root: Path,
        *,
        process_factory: ProcessFactory = WorkerProcess.start,
        timeout_seconds: float = IDM_REQUEST_TIMEOUT_SECONDS,
        close_timeout_seconds: float = IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
        descriptor: BackendDescriptor | None = None,
    ) -> None:
        for name, value in (
            ("runtime_python", runtime_python),
            ("model_lock_path", model_lock_path),
            ("model_root", model_root),
            ("input_root", input_root),
        ):
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive")

        try:
            lock = load_idm_model_lock(model_lock_path)
        except (IdmModelLockError, OSError, TypeError, ValueError) as error:
            raise IdmBackendError("IDM model lock is invalid", code="descriptor_invalid") from error

        expected_descriptor = descriptor_for_lock(lock)
        if descriptor is None:
            descriptor = expected_descriptor
        else:
            _validate_descriptor(descriptor, expected_descriptor)

        self._runtime_python = runtime_python
        self._model_lock_path = model_lock_path
        self._model_root = model_root
        self._input_root = input_root
        self._process_factory = process_factory
        self._timeout_seconds = timeout_seconds
        self._close_timeout_seconds = close_timeout_seconds
        self._lock = lock
        self._descriptor = descriptor
        self._process: Any | None = None
        self._closed = False
        self._poisoned = False

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        if self._closed:
            raise IdmBackendError("backend is closed", code="backend_closed")
        if self._poisoned:
            raise IdmBackendError("worker is poisoned", code="worker_protocol_failed")
        if not isinstance(audio, CanonicalAudio):
            raise TypeError("audio must be CanonicalAudio")

        request_path = self._input_path(audio.path)
        process = self._ensure_process()
        try:
            response = process.request(request_path)
        except WorkerProcessError as error:
            self._poison()
            raise IdmBackendError(str(error), code="worker_protocol_failed") from error
        except IdmBackendError:
            raise
        except Exception as error:
            self._poison()
            raise IdmBackendError("worker request failed", code="worker_protocol_failed") from error

        if not isinstance(response, Mapping):
            self._poison()
            raise IdmBackendError("worker response is invalid", code="worker_protocol_failed")
        if "error" in response:
            try:
                _raise_worker_error(response["error"])
            except IdmBackendError as error:
                if error.code == "worker_protocol_failed":
                    self._poison()
                raise
        raw_events = response.get("events")
        if not isinstance(raw_events, (list, tuple)):
            self._poison()
            raise IdmBackendError("worker response is invalid", code="worker_protocol_failed")
        try:
            events = tuple(_decode_native_event(value, self._lock) for value in raw_events)
        except IdmBackendError as error:
            if error.code == "native_event_invalid":
                self._poison()
            raise
        return NativePrediction(audio=audio, descriptor=self._descriptor, events=events)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, self._process = self._process, None
        if process is not None:
            try:
                process.close()
            except Exception as error:
                raise IdmBackendError("worker close failed", code="worker_close_failed") from error

    def _ensure_process(self) -> Any:
        if self._process is not None:
            return self._process
        if self._poisoned:
            raise IdmBackendError("worker is poisoned", code="worker_protocol_failed")

        command = build_worker_command(
            self._runtime_python,
            self._model_root,
        )
        try:
            factory_params = inspect.signature(self._process_factory).parameters
        except (TypeError, ValueError):
            factory_params = {}
        kwargs = _build_factory_kwargs(
            factory_params,
            timeout_seconds=self._timeout_seconds,
            close_timeout_seconds=self._close_timeout_seconds,
        )
        try:
            process = self._process_factory(command, **kwargs)
        except (OSError, RuntimeError, ValueError) as error:
            raise IdmBackendError(
                "worker could not be started", code="worker_start_failed"
            ) from error
        try:
            _validate_ready(process.ready, self._lock)
        except IdmBackendError:
            self._poisoned = True
            try:
                process.close()
            finally:
                raise
        except (AttributeError, TypeError, ValueError) as error:
            self._poisoned = True
            try:
                process.close()
            finally:
                raise IdmBackendError(
                    "worker ready response is invalid", code="worker_ready_invalid"
                ) from error
        self._process = process
        return process

    def _input_path(self, path: Path) -> str:
        try:
            root = self._input_root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise IdmBackendError(
                "canonical input must remain beneath the input root", code="input_path_invalid"
            ) from error
        return os.fspath(resolved)

    def _poison(self) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        process, self._process = self._process, None
        if process is not None:
            try:
                process.close()
            except Exception:
                pass


def build_worker_command(runtime_python: Path, model_root: Path) -> list[str]:
    """Build the exact command that keeps IDM in its isolated Python runtime."""
    if not isinstance(runtime_python, Path) or not isinstance(model_root, Path):
        raise TypeError("runtime_python and model_root must be Paths")
    return [
        os.fspath(runtime_python),
        os.fspath(IDM_WORKER_PATH),
        "--model-root",
        os.fspath(model_root),
    ]


def descriptor_for_lock(lock: IdmModelLock) -> BackendDescriptor:
    if not isinstance(lock, IdmModelLock):
        raise TypeError("lock must be an IdmModelLock")
    payload = dict(IDM_DESCRIPTOR_IDENTITIES)
    payload.update(
        {
            "model_id": lock.model_id,
            "training_data_map_id": lock.training_data_map_id,
            "upstream_source_commit": lock.repository_revision,
        }
    )
    return build_descriptor(payload, IDM_DESCRIPTOR_KEYS, IDM_DESCRIPTOR_SCHEMA)


def _validate_descriptor(value: BackendDescriptor, expected: BackendDescriptor) -> None:
    if not isinstance(value, BackendDescriptor):
        raise IdmBackendError("backend descriptor is invalid", code="descriptor_invalid")
    if value != expected:
        raise IdmBackendError(
            "backend descriptor does not match the model lock", code="descriptor_invalid"
        )
    try:
        normalized = build_descriptor(value.payload, IDM_DESCRIPTOR_KEYS, IDM_DESCRIPTOR_SCHEMA)
        if normalized != value:
            raise IdmBackendError("backend descriptor is invalid", code="descriptor_invalid")
        normalize_known_backend_descriptor(value.payload)
    except IdmBackendError:
        raise
    except (StrictJsonError, TypeError, ValueError) as error:
        raise IdmBackendError("backend descriptor is invalid", code="descriptor_invalid") from error


def _validate_ready(value: Mapping[str, object], lock: IdmModelLock) -> None:
    if not isinstance(value, Mapping) or value.get("type") != "ready":
        raise IdmBackendError("worker ready response is invalid", code="worker_ready_invalid")
    required = {
        "backend_id",
        "model_id",
        "train_classes",
        "sample_rate_hz",
        "activation_rate_hz",
    }
    if any(key not in value for key in required):
        raise IdmBackendError("worker ready response is invalid", code="worker_ready_invalid")
    if value.get("backend_id") != IDM_BACKEND_ID:
        raise IdmBackendError("worker backend id is invalid", code="worker_identity_invalid")
    if value.get("model_id") != lock.model_id:
        raise IdmBackendError("worker model id is invalid", code="worker_identity_invalid")
    classes = value.get("train_classes")
    if not isinstance(classes, (list, tuple)) or tuple(classes) != lock.train_classes:
        raise IdmBackendError("worker classes are invalid", code="worker_identity_invalid")
    sample_rate = value.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise IdmBackendError("worker sample rate is invalid", code="worker_identity_invalid")
    if sample_rate != lock.sample_rate_hz:
        raise IdmBackendError("worker sample rate is invalid", code="worker_identity_invalid")
    frame_rate = value.get("activation_rate_hz")
    if type(frame_rate) not in {int, float} or not math.isfinite(float(frame_rate)):
        raise IdmBackendError("worker frame rate is invalid", code="worker_identity_invalid")
    if float(frame_rate) != lock.activation_rate_hz:
        raise IdmBackendError("worker frame rate is invalid", code="worker_identity_invalid")


def _decode_native_event(value: object, lock: IdmModelLock) -> NativeEvent:
    if not isinstance(value, Mapping):
        raise IdmBackendError("worker event is invalid", code="native_event_invalid")
    required = {
        "class_index",
        "native_class_id",
        "frame_index",
        "time_sec",
        "onset_score",
        "native_velocity",
    }
    if set(value) != required:
        raise IdmBackendError("worker event is invalid", code="native_event_invalid")

    class_index = value["class_index"]
    if type(class_index) is not int or not 0 <= class_index < len(lock.train_classes):
        raise IdmBackendError("worker event class index is invalid", code="native_event_invalid")
    native_class_id = value["native_class_id"]
    if native_class_id != lock.train_classes[class_index]:
        raise IdmBackendError("worker event class identity is invalid", code="native_event_invalid")

    frame_index = value["frame_index"]
    if type(frame_index) is not int or frame_index < 0:
        raise IdmBackendError("worker event frame index is invalid", code="native_event_invalid")
    time_sec = value["time_sec"]
    if type(time_sec) is not float or not math.isfinite(time_sec) or time_sec < 0:
        raise IdmBackendError("worker event time is invalid", code="native_event_invalid")
    expected_time = frame_index / lock.activation_rate_hz
    if abs(time_sec - expected_time) > IDM_TIME_TOLERANCE_FRAMES / lock.activation_rate_hz:
        raise IdmBackendError("worker event time does not match frame", code="native_event_invalid")

    onset_score = value["onset_score"]
    if (
        type(onset_score) is not float
        or not math.isfinite(onset_score)
        or not 0 <= onset_score <= 1
    ):
        raise IdmBackendError("worker event onset score is invalid", code="native_event_invalid")
    native_velocity = value["native_velocity"]
    if (
        type(native_velocity) is not float
        or not math.isfinite(native_velocity)
        or not 0 < native_velocity <= 2
    ):
        raise IdmBackendError("worker event velocity is invalid", code="native_event_invalid")
    try:
        canonical_velocity = format(quantize_six(native_velocity), "f").rstrip("0").rstrip(".")
    except StrictJsonError as error:
        raise IdmBackendError(
            "worker event velocity is invalid", code="native_event_invalid"
        ) from error
    if not canonical_velocity:
        canonical_velocity = "0"
    return NativeEvent(
        time_sec=time_sec,
        native_class_id=native_class_id,
        model_output_bin=class_index,
        native_midi_note=None,
        native_metadata={
            "frame_index": str(frame_index),
            "native_velocity": canonical_velocity,
        },
        confidence=onset_score,
        velocity_midi=round((native_velocity / 2.0) * 127),
    )


def _raise_worker_error(value: object) -> None:
    if isinstance(value, Mapping):
        code = value.get("code")
        message = value.get("message")
        if isinstance(code, str) and isinstance(message, str):
            raise IdmBackendError(message, code=code)
    raise IdmBackendError("worker response is invalid", code="worker_protocol_failed")


__all__ = [
    "IDM_ADAPTER_REVISION",
    "IDM_BACKEND_ID",
    "IDM_REQUEST_TIMEOUT_SECONDS",
    "IDM_TIME_TOLERANCE_FRAMES",
    "IDM_WORKER_CLOSE_TIMEOUT_SECONDS",
    "IdmBackend",
    "IdmBackendError",
    "build_worker_command",
    "descriptor_for_lock",
]
