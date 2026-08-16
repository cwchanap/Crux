"""Direct adapter for the optional MuScriptor v0.3.0 package."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
from pathlib import Path
from typing import Any

from src.benchmark.backend_identity import (
    MUSCRIPTOR_DESCRIPTOR_IDENTITIES,
    MUSCRIPTOR_DESCRIPTOR_KEYS,
    MUSCRIPTOR_DESCRIPTOR_SCHEMA,
    BackendDescriptor,
    build_descriptor,
    expected_muscriptor_model_id,
    normalize_known_backend_descriptor,
)
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.muscriptor_model import (
    MuscriptorModelLock,
    load_muscriptor_model_lock,
    verify_muscriptor_checkpoint,
)

DEFAULT_MODEL_LOCK_PATH = Path("runtime/muscriptor/model.json")
DEFAULT_CHECKPOINT_DIR = Path("artifacts/muscriptor/checkpoint")


class MuscriptorBackendError(RuntimeError):
    """The MuScriptor adapter or its upstream model failed."""

    def __init__(self, message: str, *, code: str = "muscriptor_backend_error") -> None:
        self.code = code
        super().__init__(message)


class MuscriptorBackend:
    """Adapt one verified MuScriptor model to the benchmark backend protocol."""

    def __init__(
        self,
        model_lock_path: Path | MuscriptorModelLock | None = None,
        checkpoint_dir: Path | None = None,
        *,
        device: str | None = None,
        dtype: str | None = None,
        descriptor: BackendDescriptor | None = None,
    ) -> None:
        lock = _load_lock(model_lock_path)
        if checkpoint_dir is None:
            checkpoint_dir = Path(
                os.environ.get("CRUX_MUSCRIPTOR_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR))
            )
        if not isinstance(checkpoint_dir, Path):
            raise TypeError("checkpoint_dir must be a Path")
        selected_device = lock.device if device is None else device
        selected_dtype = lock.dtype if dtype is None else dtype
        if selected_device != lock.device:
            raise ValueError("device does not match the model lock")
        if selected_dtype != lock.dtype:
            raise ValueError("dtype does not match the model lock")

        weights_path = verify_muscriptor_checkpoint(lock, checkpoint_dir)
        expected_descriptor = _descriptor_for_lock(lock)
        if descriptor is None:
            descriptor = expected_descriptor
        else:
            _validate_descriptor(descriptor, expected_descriptor)

        muscriptor = importlib.import_module(lock.package_name)
        _verify_package_version(muscriptor, lock.package_name, lock.package_version)
        note_start_type = getattr(muscriptor, "NoteStartEvent")
        model = _load_model(
            muscriptor,
            weights_path=weights_path,
            device=selected_device,
            dtype=selected_dtype,
        )

        self._lock = lock
        self._descriptor = descriptor
        self._model = model
        self._note_start_type = note_start_type
        self._closed = False

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        if self._closed:
            raise MuscriptorBackendError("backend is closed", code="backend_closed")
        if not isinstance(audio, CanonicalAudio):
            raise TypeError("audio must be CanonicalAudio")

        events: list[NativeEvent] = []
        try:
            stream = self._model.transcribe(
                audio.path,
                use_sampling=self._lock.use_sampling,
                temperature=self._lock.temperature,
                cfg_coef=self._lock.cfg_coef,
                instruments=list(self._lock.instruments),
                batch_size=self._lock.batch_size,
                no_eos_is_ok=self._lock.no_eos_is_ok,
                beam_size=self._lock.beam_size,
                prelude_forcing=self._lock.prelude_forcing,
            )
            for event in stream:
                if isinstance(event, self._note_start_type):
                    events.append(_native_event(event))
        except MuscriptorBackendError:
            raise
        except Exception as error:
            raise MuscriptorBackendError(
                "MuScriptor transcription failed", code="inference_failed"
            ) from error
        return NativePrediction(audio=audio, descriptor=self._descriptor, events=tuple(events))

    def close(self) -> None:
        self._closed = True


def create_backend(
    checkpoint_dir: Path | None = None,
    model_lock_path: Path | None = None,
    *,
    device: str | None = None,
    dtype: str | None = None,
    descriptor: BackendDescriptor | None = None,
) -> MuscriptorBackend:
    """Construct one verified MuScriptor backend from local frozen artifacts."""
    if model_lock_path is None:
        model_lock_path = Path(
            os.environ.get("CRUX_MUSCRIPTOR_MODEL_LOCK", str(DEFAULT_MODEL_LOCK_PATH))
        )
    return MuscriptorBackend(
        model_lock_path=model_lock_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        dtype=dtype,
        descriptor=descriptor,
    )


def _load_lock(model_lock_path: Path | MuscriptorModelLock | None) -> MuscriptorModelLock:
    if isinstance(model_lock_path, MuscriptorModelLock):
        return model_lock_path
    if model_lock_path is None:
        model_lock_path = Path(
            os.environ.get("CRUX_MUSCRIPTOR_MODEL_LOCK", str(DEFAULT_MODEL_LOCK_PATH))
        )
    return load_muscriptor_model_lock(model_lock_path)


def _descriptor_for_lock(lock: MuscriptorModelLock) -> BackendDescriptor:
    payload = dict(MUSCRIPTOR_DESCRIPTOR_IDENTITIES)
    payload.update(
        {
            "model_id": expected_muscriptor_model_id(lock),
            "training_data_map_id": lock.training_data_map_id,
            "upstream_source_commit": lock.upstream_source_commit,
        }
    )
    descriptor = build_descriptor(payload, MUSCRIPTOR_DESCRIPTOR_KEYS, MUSCRIPTOR_DESCRIPTOR_SCHEMA)
    _validate_descriptor(descriptor, descriptor)
    return descriptor


def _validate_descriptor(value: BackendDescriptor, expected: BackendDescriptor) -> None:
    if not isinstance(value, BackendDescriptor):
        raise ValueError("backend descriptor is invalid")
    if value != expected:
        raise ValueError("backend descriptor does not match the model lock")
    try:
        normalize_known_backend_descriptor(value.payload)
    except (TypeError, ValueError) as error:
        raise ValueError("backend descriptor is invalid") from error


def _verify_package_version(module: Any, package_name: str, expected: str) -> None:
    module_version = getattr(module, "__version__", None)
    try:
        installed_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if module_version is not None and module_version != expected:
        raise ValueError("MuScriptor package version does not match the model lock")
    if installed_version is not None and installed_version != expected:
        raise ValueError("MuScriptor package version does not match the model lock")
    if module_version is None and installed_version is None:
        raise ValueError("MuScriptor package version is unavailable")


def _load_model(
    module: Any,
    *,
    weights_path: Path,
    device: str,
    dtype: str,
) -> Any:
    model_type = getattr(module, "TranscriptionModel", module)
    load_model = getattr(model_type, "load_model")
    return load_model(weights_path=weights_path, device=device, dtype=dtype)


def _native_event(event: Any) -> NativeEvent:
    instrument = event.instrument
    pitch = event.pitch
    onset = event.start_time
    if instrument != "drums":
        raise MuscriptorBackendError(
            "MuScriptor emitted a non-drum event", code="native_event_invalid"
        )
    if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127:
        raise MuscriptorBackendError(
            "MuScriptor emitted an invalid MIDI pitch", code="native_event_invalid"
        )
    if not isinstance(onset, float) or not math.isfinite(onset) or onset < 0:
        raise MuscriptorBackendError(
            "MuScriptor emitted an invalid onset", code="native_event_invalid"
        )
    return NativeEvent(
        time_sec=onset,
        native_class_id=f"drums:midi_{pitch}",
        model_output_bin=None,
        native_midi_note=pitch,
        native_metadata={"instrument_group": instrument},
        confidence=None,
        velocity_midi=None,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_MODEL_LOCK_PATH",
    "MuscriptorBackend",
    "MuscriptorBackendError",
    "create_backend",
]
