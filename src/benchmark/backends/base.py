"""Shared, backend-neutral prediction values and protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.benchmark.backend_identity import BackendDescriptor


# Frozen public schema fields are deliberately explicit.  A canonical audio value is
# independent of whichever backend produced a prediction.
# pylint: disable=too-many-instance-attributes
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


# pylint: enable=too-many-instance-attributes


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


class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...

    def close(self) -> None: ...
