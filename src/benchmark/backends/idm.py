"""Host adapter for the isolated Inverse Drum Machine worker."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import tomllib
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.benchmark.artifact_io import read_regular_file_no_follow
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
    IDM_RUNTIME_SYNC_TIMEOUT_SECONDS,
    IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
    IdmModelLock,
    IdmModelLockError,
    load_idm_model_lock,
    verify_idm_model_files,
    verify_idm_runtime_lock,
)
from src.benchmark.worker_process import WorkerProcess, WorkerProcessError

IDM_ADAPTER_REVISION = "crux.idm-adapter/v1"
IDM_WORKER_PATH = Path(__file__).resolve().parents[3] / "runtime" / "idm" / "worker.py"
IDM_RUNTIME_LOCK_NAME = "uv.lock"
IDM_WHEEL_NAME = "inverse_drum_machine-0.1.0-py3-none-any.whl"
IDM_PROVENANCE_NAME = "idm-wheel-provenance.json"
IDM_TIME_TOLERANCE_FRAMES = 0.5
# The locked transform pads to a hop multiple, centered mel adds one frame, and
# the pinned temporal backbone adds twelve frames (three padded convolutions).
IDM_ACTIVATION_FRAME_PADDING = 13
_PYTHON_IMPORT_DISCOVERY_ENVIRONMENT_KEYS = frozenset(
    {
        "PYTHONCASEOK",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
    }
)


class IdmBackendError(RuntimeError):
    """The IDM worker or its host-boundary contract failed."""

    def __init__(self, message: str, *, code: str = "idm_backend_error") -> None:
        self.code = code
        super().__init__(message)


ProcessFactory = Callable[..., Any]
RuntimeSync = Callable[[Path, Path], None]


def _default_runtime_sync(runtime_root: Path, runtime_python: Path) -> None:
    """Materialize or reconcile the isolated runtime from the frozen uv.lock.

    Runs ``uv sync --project <runtime_root> --frozen``.  When the project venv
    already exists the sync stays ``--offline``, verifying/reconciling strictly
    from the local uv cache; on a fresh checkout (no venv) the sync may reach
    the registry to materialize ``<runtime_root>/.venv`` from the frozen lock.
    Then verifies that the venv python sits at the expected path.  Raises
    ``RuntimeError`` on any failure (missing ``uv``, non-zero exit, timeout,
    wrong venv path).

    The venv-path guard compares lexically rather than via ``resolve()``: uv
    symlinks ``.venv/bin/python`` to the underlying base/managed interpreter,
    so a resolve()-based equality would accept that base interpreter directly
    and let the worker import packages from the wrong environment.  The
    subprocess environment is also sanitized to drop ``UV_PROJECT_ENVIRONMENT``,
    which would otherwise redirect ``uv sync`` away from
    ``<runtime_root>/.venv`` even when ``--project`` targets ``runtime_root``.
    """
    expected_python = runtime_root / ".venv" / "bin" / "python"
    if runtime_python != expected_python:
        raise RuntimeError("runtime python must point to the locked project venv")
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        raise RuntimeError("uv is not available on PATH")
    sync_env = dict(os.environ)
    sync_env.pop("UV_PROJECT_ENVIRONMENT", None)
    offline = expected_python.exists()
    sync_label = "uv sync --frozen --offline" if offline else "uv sync --frozen"
    command = [uv_binary, "sync", "--project", os.fspath(runtime_root), "--frozen"]
    if offline:
        command.append("--offline")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=sync_env,
            timeout=IDM_RUNTIME_SYNC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{sync_label} timed out") from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        suffix = f": {stderr}" if stderr else ""
        raise RuntimeError(f"{sync_label} failed{suffix}") from error
    except OSError as error:
        raise RuntimeError(f"{sync_label} failed") from error
    if not runtime_python.exists() or not expected_python.exists():
        raise RuntimeError("runtime python is unavailable")


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
        runtime_sync: RuntimeSync | None = None,
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
        _validate_supported_runtime(lock)

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
        self._runtime_sync = runtime_sync if runtime_sync is not None else _default_runtime_sync
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
        frame_limit = _activation_frame_limit(audio, self._lock)
        process = self._ensure_process()
        try:
            response = process.request(
                request_path,
                audio_byte_length=audio.byte_length,
                audio_sha256=audio.input_audio_sha256,
            )
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
            events = tuple(
                _decode_native_event(value, self._lock, frame_limit) for value in raw_events
            )
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
            raise IdmBackendError(
                "worker is poisoned", code="worker_protocol_failed"
            )  # pragma: no cover - transcribe guards

        (
            wheel_path,
            wheel_sha256,
            model_config_path,
            model_config_sha256,
            model_config_byte_length,
            checkpoint_path,
            checkpoint_sha256,
            checkpoint_byte_length,
        ) = _attest_runtime_artifacts(
            self._lock,
            self._model_lock_path,
            self._model_root,
        )
        runtime_root = self._model_lock_path.parent
        try:
            self._runtime_sync(runtime_root, self._runtime_python)
        except Exception as error:
            self._poisoned = True
            raise IdmBackendError(
                f"IDM runtime environment does not match the frozen uv.lock: {error}",
                code="runtime_artifact_invalid",
            ) from error
        site_packages = _runtime_site_packages(self._runtime_python, self._lock)
        command = build_worker_command(
            self._runtime_python,
            self._model_root,
            wheel_path=wheel_path,
            wheel_sha256=wheel_sha256,
            site_packages=site_packages,
            model_config_path=model_config_path,
            model_config_sha256=model_config_sha256,
            model_config_byte_length=model_config_byte_length,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_byte_length=checkpoint_byte_length,
        )
        try:
            factory_params = inspect.signature(self._process_factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive for unsignatured callables
            factory_params = {}  # pragma: no cover - defensive for unsignatured callables
        kwargs: dict[str, Any] = _build_factory_kwargs(
            factory_params,
            timeout_seconds=self._timeout_seconds,
            close_timeout_seconds=self._close_timeout_seconds,
        )
        if _factory_accepts_keyword(factory_params, "env"):
            kwargs["env"] = _isolated_worker_environment()
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


def build_worker_command(
    runtime_python: Path,
    model_root: Path,
    *,
    wheel_path: Path | None = None,
    wheel_sha256: str | None = None,
    site_packages: Path | None = None,
    model_config_path: Path | None = None,
    model_config_sha256: str | None = None,
    model_config_byte_length: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_byte_length: int | None = None,
) -> list[str]:
    """Build the exact command that keeps IDM in its isolated Python runtime."""
    if not isinstance(runtime_python, Path) or not isinstance(model_root, Path):
        raise TypeError("runtime_python and model_root must be Paths")
    if wheel_path is None:
        wheel_path = IDM_WORKER_PATH.parent / IDM_WHEEL_NAME
    if not isinstance(wheel_path, Path):
        raise TypeError("wheel_path must be a Path")
    artifact_arguments = (
        wheel_sha256,
        site_packages,
        model_config_path,
        model_config_sha256,
        model_config_byte_length,
        checkpoint_path,
        checkpoint_sha256,
        checkpoint_byte_length,
    )
    if any(argument is None for argument in artifact_arguments):
        raise ValueError("all attested runtime and model arguments are required")
    if not isinstance(site_packages, Path) or not isinstance(model_config_path, Path):
        raise TypeError("site_packages and model_config_path must be Paths")
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a Path")
    if (
        not isinstance(model_config_sha256, str)
        or not isinstance(checkpoint_sha256, str)
        or type(model_config_byte_length) is not int
        or model_config_byte_length < 0
        or type(checkpoint_byte_length) is not int
        or checkpoint_byte_length < 0
    ):
        raise ValueError("attested model file identities are invalid")
    if (
        not isinstance(wheel_sha256, str)
        or len(wheel_sha256) != 64
        or any(character not in "0123456789abcdef" for character in wheel_sha256)
    ):
        raise ValueError("wheel_sha256 must be lowercase SHA-256")
    command = [
        os.fspath(runtime_python),
        "-I",
        "-S",
        os.fspath(IDM_WORKER_PATH),
        "--model-root",
        os.fspath(model_root),
        "--site-packages",
        os.fspath(site_packages),
        "--wheel-path",
        os.fspath(wheel_path),
    ]
    command.extend(("--wheel-sha256", wheel_sha256))
    command.extend(
        (
            "--model-config-path",
            os.fspath(model_config_path),
            "--model-config-sha256",
            model_config_sha256,
            "--model-config-byte-length",
            str(model_config_byte_length),
            "--checkpoint-path",
            os.fspath(checkpoint_path),
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--checkpoint-byte-length",
            str(checkpoint_byte_length),
        )
    )
    return command


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
        if (
            normalized != value
        ):  # pragma: no cover - equality with expected guarantees normalization
            raise IdmBackendError("backend descriptor is invalid", code="descriptor_invalid")
        normalize_known_backend_descriptor(value.payload)
    except IdmBackendError:  # pragma: no cover - equality with expected guarantees normalization
        raise
    except (StrictJsonError, TypeError, ValueError) as error:  # pragma: no cover
        raise IdmBackendError("backend descriptor is invalid", code="descriptor_invalid") from error


def _validate_supported_runtime(lock: IdmModelLock) -> None:
    if (
        lock.device != "cpu" or lock.dtype != "float32"
    ):  # pragma: no cover - lock enforces CPU float32
        raise IdmBackendError(
            "IDM KISS runtime supports only CPU float32", code="descriptor_invalid"
        )  # pragma: no cover - lock enforces CPU float32


def _validate_ready(value: Mapping[str, object], lock: IdmModelLock) -> None:
    if not isinstance(value, Mapping) or value.get("type") != "ready":
        raise IdmBackendError("worker ready response is invalid", code="worker_ready_invalid")
    required = {
        "backend_id",
        "model_id",
        "model_name",
        "train_classes",
        "python_version",
        "sample_rate_hz",
        "activation_rate_hz",
        "device",
        "dtype",
    }
    if any(key not in value for key in required):
        raise IdmBackendError("worker ready response is invalid", code="worker_ready_invalid")
    if value.get("backend_id") != IDM_BACKEND_ID:
        raise IdmBackendError("worker backend id is invalid", code="worker_identity_invalid")
    if value.get("model_id") != lock.model_id:
        raise IdmBackendError("worker model id is invalid", code="worker_identity_invalid")
    if value.get("model_name") != lock.model_name:
        raise IdmBackendError("worker model name is invalid", code="worker_identity_invalid")
    if value.get("python_version") != lock.python_version:
        raise IdmBackendError("worker Python version is invalid", code="worker_identity_invalid")
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
    if value.get("device") != lock.device or value.get("dtype") != lock.dtype:
        raise IdmBackendError("worker device or dtype is invalid", code="worker_identity_invalid")
    if (
        lock.device != "cpu" or lock.dtype != "float32"
    ):  # pragma: no cover - lock enforces CPU float32
        raise IdmBackendError(
            "IDM KISS runtime supports only CPU float32", code="worker_identity_invalid"
        )  # pragma: no cover - lock enforces CPU float32


def _decode_native_event(value: object, lock: IdmModelLock, frame_limit: int) -> NativeEvent:
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
    if type(frame_index) is not int or not 0 <= frame_index < frame_limit:
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
    except StrictJsonError as error:  # pragma: no cover - isfinite check above prevents this
        raise IdmBackendError(
            "worker event velocity is invalid", code="native_event_invalid"
        ) from error  # pragma: no cover - isfinite check above prevents this
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


def _activation_frame_limit(audio: CanonicalAudio, lock: IdmModelLock) -> int:
    frame_count = audio.audio_frame_count
    if type(frame_count) is not int or frame_count < 0:
        raise IdmBackendError("canonical audio frame count is invalid", code="input_audio_invalid")
    return (
        frame_count + lock.mel_hop_length - 1
    ) // lock.mel_hop_length + IDM_ACTIVATION_FRAME_PADDING


def _runtime_site_packages(runtime_python: Path, lock: IdmModelLock) -> Path:
    if not isinstance(runtime_python, Path) or not isinstance(lock, IdmModelLock):
        raise TypeError("runtime_python and lock have invalid types")
    version_parts = lock.python_version.split(".")
    version = ".".join(version_parts[:2])
    return runtime_python.parent.parent / "lib" / f"python{version}" / "site-packages"


def _factory_accepts_keyword(params: Mapping[str, inspect.Parameter], name: str) -> bool:
    parameter = params.get(name)
    return (
        parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
    ) or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in params.values())


def _isolated_worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in _PYTHON_IMPORT_DISCOVERY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    return environment


def _attest_runtime_artifacts(
    lock: IdmModelLock,
    model_lock_path: Path,
    model_root: Path,
) -> tuple[Path, str, Path, str, int, Path, str, int]:
    runtime_root = model_lock_path.parent
    runtime_lock_path = runtime_root / IDM_RUNTIME_LOCK_NAME
    wheel_path = runtime_root / "wheels" / IDM_WHEEL_NAME
    provenance_path = runtime_root / IDM_PROVENANCE_NAME
    try:
        model_config_path, checkpoint_path = verify_idm_model_files(lock, model_root)
        verify_idm_runtime_lock(lock, runtime_lock_path)
        runtime_lock_bytes = read_regular_file_no_follow(runtime_lock_path)
        wheel_sha256 = _wheel_sha256_from_runtime_lock(runtime_lock_bytes, lock)
        wheel_bytes = read_regular_file_no_follow(wheel_path)
        actual_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
        if actual_sha256 != wheel_sha256:
            raise ValueError("IDM runtime wheel digest does not match uv.lock")
        provenance_bytes = read_regular_file_no_follow(provenance_path)
        _validate_wheel_provenance(
            provenance_bytes,
            wheel_path.name,
            wheel_bytes,
            wheel_sha256,
            lock,
        )
        return (
            wheel_path.absolute(),
            wheel_sha256,
            model_config_path.absolute(),
            lock.model_config_sha256,
            lock.model_config_byte_length,
            checkpoint_path.absolute(),
            lock.checkpoint_sha256,
            lock.checkpoint_byte_length,
        )
    except (OSError, TypeError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
        raise IdmBackendError(
            "IDM runtime or model artifacts are not attested", code="runtime_artifact_invalid"
        ) from error


def _wheel_sha256_from_runtime_lock(runtime_lock_bytes: bytes, lock: IdmModelLock) -> str:
    try:
        payload = tomllib.loads(runtime_lock_bytes.decode("utf-8"))
        packages = payload["package"]
        matches = [
            package
            for package in packages
            if package.get("name") == lock.package_name
            and package.get("version") == lock.package_version
        ]
        if len(matches) != 1:
            raise ValueError("runtime lock does not identify exactly one IDM package")
        package = matches[0]
        source = package["source"]
        if source.get("path") != f"wheels/{IDM_WHEEL_NAME}":
            raise ValueError("runtime lock IDM source path is invalid")
        wheels = package["wheels"]
        if len(wheels) != 1 or wheels[0].get("filename") != IDM_WHEEL_NAME:
            raise ValueError("runtime lock IDM wheel identity is invalid")
        wheel_hash = wheels[0].get("hash")
    except (
        AttributeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        raise ValueError("runtime lock IDM wheel identity is invalid") from error
    if (
        not isinstance(wheel_hash, str)
        or not wheel_hash.startswith("sha256:")
        or len(wheel_hash) != len("sha256:") + 64
    ):
        raise ValueError("runtime lock IDM wheel digest is invalid")
    return wheel_hash.removeprefix("sha256:")


def _validate_wheel_provenance(
    content: bytes,
    wheel_name: str,
    wheel_bytes: bytes,
    wheel_sha256: str,
    lock: IdmModelLock,
) -> None:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("IDM wheel provenance newline is invalid")
    try:
        provenance = json.loads(content[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("IDM wheel provenance is invalid") from error
    if not isinstance(provenance, dict):
        raise ValueError("IDM wheel provenance is invalid")
    if (
        provenance.get("schema") != "crux.idm-wheel-provenance/v1"
        or provenance.get("source_commit") != lock.repository_revision
        or provenance.get("package_name") != lock.package_name
        or provenance.get("package_version") != lock.package_version
    ):
        raise ValueError("IDM wheel provenance identity is invalid")
    wheel = provenance.get("wheel")
    if not isinstance(wheel, dict):
        raise ValueError("IDM wheel provenance wheel record is invalid")
    if (
        wheel.get("path") != wheel_name
        or wheel.get("sha256") != wheel_sha256
        or wheel.get("byte_length") != len(wheel_bytes)
        or wheel.get("tag") != "py3-none-any"
    ):
        raise ValueError("IDM wheel provenance wheel record is invalid")
    packaged_files = provenance.get("packaged_idm_files")
    if not isinstance(packaged_files, list) or not packaged_files:
        raise ValueError("IDM wheel provenance package inventory is invalid")
    expected_files: dict[str, tuple[str, int]] = {}
    for record in packaged_files:
        if not isinstance(record, dict):
            raise ValueError("IDM wheel provenance package inventory is invalid")
        path = record.get("path")
        digest = record.get("sha256")
        byte_length = record.get("byte_length")
        if (
            not isinstance(path, str)
            or not path.startswith("idm/")
            or path.endswith("/")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(byte_length) is not int
            or byte_length < 0
            or path in expected_files
        ):
            raise ValueError("IDM wheel provenance package inventory is invalid")
        expected_files[path] = (digest, byte_length)
    with zipfile.ZipFile(_bytes_as_file(wheel_bytes)) as wheel_archive:
        packaged_names = {
            name
            for name in wheel_archive.namelist()
            if name.startswith("idm/") and not name.endswith("/")
        }
        if packaged_names != set(expected_files):
            raise ValueError("IDM wheel package inventory does not match wheel")
        for path, (digest, byte_length) in expected_files.items():
            source = wheel_archive.read(path)
            if len(source) != byte_length or hashlib.sha256(source).hexdigest() != digest:
                raise ValueError("IDM wheel package inventory does not match wheel")


def _bytes_as_file(content: bytes) -> Any:
    from io import BytesIO

    return BytesIO(content)


__all__ = [
    "IDM_ADAPTER_REVISION",
    "IDM_BACKEND_ID",
    "IDM_REQUEST_TIMEOUT_SECONDS",
    "IDM_RUNTIME_SYNC_TIMEOUT_SECONDS",
    "IDM_TIME_TOLERANCE_FRAMES",
    "IDM_WORKER_CLOSE_TIMEOUT_SECONDS",
    "IdmBackend",
    "IdmBackendError",
    "build_worker_command",
    "descriptor_for_lock",
]
