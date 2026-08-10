"""Bridge mapped prediction artifacts into common-class benchmark events."""

from __future__ import annotations

from src.benchmark.models import BenchmarkEvent
from src.benchmark.prediction_artifact import (
    MappedPredictionEvent,
    PredictionArtifact,
    read_prediction_artifact,
)


def prediction_to_benchmark_events(artifact: PredictionArtifact) -> tuple[BenchmarkEvent, ...]:
    """Return only mapped events, using the common taxonomy for scoring."""
    if not isinstance(artifact, PredictionArtifact):
        raise TypeError("artifact must be PredictionArtifact")
    audio = artifact.prediction.audio
    return tuple(
        _benchmark_event(audio.source_audio_id, audio.input_view_id, event)
        for event in artifact.prediction.events
        if event.mapping_status == "mapped"
    )


def read_scorer_events(content: bytes) -> tuple[BenchmarkEvent, ...]:
    return prediction_to_benchmark_events(read_prediction_artifact(content))


def _benchmark_event(
    chart_id: str,
    input_view_id: str,
    event: MappedPredictionEvent,
) -> BenchmarkEvent:
    if event.common_class is None:
        raise ValueError("mapped event common_class must be non-null")
    native = event.native
    return BenchmarkEvent(
        chart_id=chart_id,
        time_sec=native.time_sec,
        canonical_class=event.common_class,
        source="prediction",
        metadata={
            "input_view_id": input_view_id,
            "native_class_id": native.native_class_id,
            "native_midi_note": native.native_midi_note,
            "model_output_bin": native.model_output_bin,
            "detailed_canonical_class": event.canonical_class,
            "prediction_map_version": event.prediction_map_version,
        },
    )


__all__ = ["prediction_to_benchmark_events", "read_scorer_events"]
