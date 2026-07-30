from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from src.benchmark.backend_identity import BackendDescriptor


@dataclass(frozen=True)
class BackendError:
    code: str
    message: str

    def __post_init__(self) -> None:
        if (
            not self.code
            or self.code[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in self.code
            )
        ):
            raise ValueError("backend error code must be stable lowercase snake case")
        if not self.message:
            raise ValueError("backend error message must not be empty")
        try:
            self.message.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("backend error message must be valid UTF-8") from None
        if any(ord(character) < 32 or ord(character) == 127 for character in self.message):
            raise ValueError("backend error message must be sanitized")


class _BackendFailure(RuntimeError):
    def __init__(self, error: BackendError) -> None:
        if not isinstance(error, BackendError):
            raise TypeError("backend failure requires one BackendError")
        self.error = error
        super().__init__(error.code)


class BackendItemFailure(_BackendFailure):
    pass


class BackendFatalFailure(_BackendFailure):
    pass


@dataclass(frozen=True)
class PublishedArtifact:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class MidiDerivative:
    midi: PublishedArtifact
    sidecar: PublishedArtifact


@dataclass(frozen=True)
class CanonicalAudio:
    path: Path
    source_audio_id: str
    source_audio_sha256: str
    input_view_id: str
    input_audio_sha256: str
    byte_length: int
    sample_rate: int
    channel_count: int
    sample_width_bytes: int
    audio_frame_count: int


@dataclass(frozen=True)
class NativeEvent:
    time_sec: float
    native_class_id: str
    model_output_bin: int | None
    native_midi_note: int | None
    native_metadata: Mapping[str, str | None]
    confidence: float | None
    velocity_midi: int | None


@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]
    backend_lock_sha256: str | None
    runtime_lock_sha256: str | None
    parameter_lock_sha256: str | None
    model_artifact_set_sha256: str | None
    upstream_source_commit: str | None
    training_data_map_id: str | None


@dataclass(frozen=True)
class TensorCoverageCheck:
    status: Literal["passed", "failed", "not_run", "not_applicable"]
    required_count: int
    restored_count: int
    non_inference_count: int
    required_inventory_sha256: str | None
    non_inference_inventory_sha256: str | None
    report: PublishedArtifact | None


@dataclass(frozen=True)
class SmokeCheck:
    status: Literal["passed", "failed", "not_run", "not_applicable"]
    audio_sha256: str | None
    oracle_sha256: str | None
    prediction: PublishedArtifact | None


@dataclass(frozen=True)
class BackendVerification:
    status: Literal["verified", "failed", "environment_unsupported"]
    descriptor: BackendDescriptor | None
    max_input_audio_frames: int | None
    backend_lock_sha256: str | None
    runtime_lock_sha256: str | None
    parameter_lock_sha256: str | None
    seal_evidence_sha256: str | None
    execution_attestation: PublishedArtifact | None
    tensor_coverage: TensorCoverageCheck
    smoke: SmokeCheck
    errors: tuple[BackendError, ...]


class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...

    def verify(self) -> BackendVerification: ...

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...

    def close(self) -> None: ...
