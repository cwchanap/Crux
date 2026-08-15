"""Thin host adapter for the reusable OaF worker."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_IDENTITIES,
    OAF_DESCRIPTOR_KEYS,
    OAF_DESCRIPTOR_SCHEMA,
    BackendDescriptor,
    build_descriptor,
    normalize_known_backend_descriptor,
)
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.worker_process import WorkerProcess, WorkerProcessError

IMAGE = "crux-oaf-tf1:local"
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
RESTORED_TENSOR_COUNT = 78


class OafBackendError(RuntimeError):
    """The OaF worker or its host-boundary contract failed."""

    def __init__(self, message: str, *, code: str = "oaf_backend_error") -> None:
        self.code = code
        super().__init__(message)


ProcessFactory = Callable[..., Any]


class OafBackend:
    """Adapt one persistent worker into the shared backend protocol."""

    def __init__(
        self,
        checkpoint_dir: Path,
        input_root: Path,
        *,
        image: str = IMAGE,
        process_factory: ProcessFactory = WorkerProcess.start,
        timeout_seconds: float = 30.0,
        close_timeout_seconds: float = 30.0,
        descriptor: BackendDescriptor | None = None,
    ) -> None:
        if not isinstance(checkpoint_dir, Path) or not isinstance(input_root, Path):
            raise TypeError("checkpoint_dir and input_root must be Paths")
        if not isinstance(image, str) or not image:
            raise ValueError("image must be a nonempty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive")
        self._checkpoint_dir = checkpoint_dir
        self._input_root = input_root
        self._image = image
        self._process_factory = process_factory
        self._timeout_seconds = timeout_seconds
        self._close_timeout_seconds = close_timeout_seconds
        self._process: Any | None = None
        self._closed = False
        self._descriptor = descriptor or _load_descriptor()
        _validate_descriptor(self._descriptor)

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        if self._closed:
            raise OafBackendError("backend is closed", code="backend_closed")
        if not isinstance(audio, CanonicalAudio):
            raise TypeError("audio must be CanonicalAudio")
        relative = self._relative_input_path(audio.path)
        process = self._ensure_process()
        try:
            response = process.request(relative)
        except OafBackendError:
            raise
        except WorkerProcessError as error:
            raise OafBackendError(str(error), code="worker_error") from error
        except RuntimeError as error:
            raise OafBackendError(str(error), code="worker_error") from error
        except Exception as error:
            raise OafBackendError("worker request failed", code="worker_error") from error

        if not isinstance(response, Mapping):
            raise OafBackendError("worker response is invalid", code="worker_response_invalid")
        if "error" in response:
            _raise_worker_error(response["error"])
        raw_events = response.get("events")
        if not isinstance(raw_events, (list, tuple)):
            raise OafBackendError("worker response is invalid", code="worker_response_invalid")
        events = tuple(_decode_native_event(value) for value in raw_events)
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
                raise OafBackendError("worker close failed", code="worker_close_failed") from error

    def _ensure_process(self) -> Any:
        if self._process is not None:
            return self._process
        command = build_docker_command(
            self._checkpoint_dir,
            self._input_root,
            image=self._image,
        )
        try:
            process = self._process_factory(
                command,
                timeout_seconds=self._timeout_seconds,
                close_timeout_seconds=self._close_timeout_seconds,
            )
        except TypeError:
            # Tiny injected fakes commonly accept only the command.  The production
            # WorkerProcess accepts the timeout keywords above.
            process = self._process_factory(command)
        except (OSError, RuntimeError, ValueError) as error:
            raise OafBackendError(
                "worker could not be started", code="worker_start_failed"
            ) from error
        try:
            _validate_ready(process.ready)
        except OafBackendError:
            try:
                process.close()
            finally:
                raise
        except (AttributeError, TypeError, ValueError) as error:
            try:
                process.close()
            finally:
                raise OafBackendError(
                    "worker ready response is invalid", code="worker_ready_invalid"
                ) from error
        self._process = process
        return process

    def _relative_input_path(self, path: Path) -> str:
        try:
            root = self._input_root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise OafBackendError(
                "canonical input must remain beneath the input root", code="input_path_invalid"
            ) from error
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise OafBackendError(
                "canonical input must remain beneath the input root", code="input_path_invalid"
            )
        return relative.as_posix()


def build_docker_command(
    checkpoint_dir: Path,
    input_root: Path,
    *,
    image: str = IMAGE,
) -> list[str]:
    """Build the deliberately small read-only worker container command."""
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network=none",
        "--read-only",
        "--tmpfs=/tmp:rw",
        f"--mount=type=bind,src={checkpoint_dir},dst=/model,readonly",
        f"--mount=type=bind,src={input_root},dst=/input,readonly",
        "--workdir=/input",
        image,
    ]


def create_backend(
    checkpoint_dir: Path | None = None,
    input_root: Path | None = None,
    *,
    image: str = IMAGE,
    process_factory: ProcessFactory = WorkerProcess.start,
    timeout_seconds: float = 30.0,
    close_timeout_seconds: float = 30.0,
    descriptor: BackendDescriptor | None = None,
) -> OafBackend:
    config = _load_model_config()
    if checkpoint_dir is None:
        cache_root = Path(
            os.environ.get("CRUX_OAF_CHECKPOINT_CACHE", "artifacts/benchmark/model-cache")
        )
        checkpoint_dir = cache_root / "sha256" / config.checkpoint.archive_sha256
    if input_root is None:
        input_root = Path(os.environ.get("CRUX_OAF_INPUT_ROOT", "artifacts/benchmark/inputs"))
    return OafBackend(
        checkpoint_dir,
        input_root,
        image=image,
        process_factory=process_factory,
        timeout_seconds=timeout_seconds,
        close_timeout_seconds=close_timeout_seconds,
        descriptor=descriptor,
    )


def _load_model_config() -> Any:
    from runtime.oaf_tf1.model import load_model_config

    path = Path(__file__).resolve().parents[3] / "runtime" / "oaf_tf1" / "model.json"
    return load_model_config(path)


def _load_descriptor() -> BackendDescriptor:
    config = _load_model_config()
    payload = dict(OAF_DESCRIPTOR_IDENTITIES)
    payload.update(
        {
            "backend_id": config.backend_id,
            "model_id": config.model_id,
            "architecture_id": config.architecture_id,
            "upstream_source_commit": config.upstream_source_commit,
            "training_data_map_id": config.training_data_map_id,
            "native_output_space_id": config.native_output_space_id,
            "native_metadata_schema_id": config.native_metadata_schema_id,
            "prediction_schema": "crux.drum-prediction-events/v2",
            "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        }
    )
    return build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)


def _validate_ready(value: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping) or value.get("type") != "ready":
        raise OafBackendError("worker ready response is invalid", code="worker_ready_invalid")
    if value.get("backend_id") != OAF_BACKEND_ID:
        raise OafBackendError("worker backend id is invalid", code="worker_identity_invalid")
    if value.get("restored_tensor_count") != RESTORED_TENSOR_COUNT:
        raise OafBackendError(
            "worker restored tensor count is invalid", code="worker_identity_invalid"
        )


def _validate_descriptor(value: BackendDescriptor) -> None:
    if not isinstance(value, BackendDescriptor):
        raise OafBackendError("backend descriptor is invalid", code="descriptor_invalid")
    payload = dict(value.payload)
    if set(payload) != set(OAF_DESCRIPTOR_KEYS):
        raise OafBackendError("backend descriptor is invalid", code="descriptor_invalid")
    try:
        normalized = build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)
        if normalized.sha256 != value.sha256:
            raise OafBackendError("backend descriptor is invalid", code="descriptor_invalid")
        if normalized.payload != value.payload:
            raise OafBackendError("backend descriptor is invalid", code="descriptor_invalid")
        normalize_known_backend_descriptor(payload)
    except (OafBackendError, TypeError, ValueError) as error:
        if isinstance(error, OafBackendError):
            raise
        raise OafBackendError("backend descriptor is invalid", code="descriptor_invalid") from error
    if payload.get("backend_id") != OAF_BACKEND_ID:
        raise OafBackendError("backend descriptor backend id is invalid", code="descriptor_invalid")


def _decode_native_event(value: object) -> NativeEvent:
    if not isinstance(value, Mapping):
        raise OafBackendError("worker event is invalid", code="native_event_invalid")
    required = {
        "time_sec",
        "native_class_id",
        "model_output_bin",
        "native_midi_note",
        "upstream_8hit_group_id",
        "confidence",
        "velocity_midi",
    }
    if set(value) != required:
        raise OafBackendError("worker event is invalid", code="native_event_invalid")
    group = value["upstream_8hit_group_id"]
    if group is not None and not isinstance(group, str):
        raise OafBackendError("worker event is invalid", code="native_event_invalid")
    if not isinstance(value["native_class_id"], str):
        raise OafBackendError("worker event is invalid", code="native_event_invalid")
    return NativeEvent(
        time_sec=_float_field(value["time_sec"], "time_sec"),
        native_class_id=value["native_class_id"],
        model_output_bin=_int_or_none(value["model_output_bin"], "model_output_bin"),
        native_midi_note=_int_or_none(value["native_midi_note"], "native_midi_note"),
        native_metadata={"upstream_8hit_group_id": group},
        confidence=_float_or_none(value["confidence"], "confidence"),
        velocity_midi=_int_or_none(value["velocity_midi"], "velocity_midi"),
    )


def _raise_worker_error(value: object) -> None:
    if isinstance(value, Mapping):
        code = value.get("code")
        message = value.get("message")
        if isinstance(code, str) and isinstance(message, str):
            raise OafBackendError(message, code=code)
    raise OafBackendError("worker response is invalid", code="worker_response_invalid")


def _float_field(value: object, field: str) -> float:
    if type(value) is not float:
        raise OafBackendError(f"worker event {field} is invalid", code="native_event_invalid")
    return value


def _float_or_none(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _float_field(value, field)


def _int_or_none(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise OafBackendError(f"worker event {field} is invalid", code="native_event_invalid")
    return value


__all__ = [
    "IMAGE",
    "OAF_BACKEND_ID",
    "OAF_ADAPTER_REVISION",
    "OafBackend",
    "OafBackendError",
    "RESTORED_TENSOR_COUNT",
    "build_docker_command",
    "create_backend",
]
