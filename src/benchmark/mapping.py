from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.benchmark.backends import NativePrediction
from src.benchmark.models import BenchmarkEvent
from src.benchmark.taxonomy import (
    DRUM_LANE_IDS as _DRUM_LANE_IDS,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP,
    OAF_PREDICTION_MAP,
    ClassMapping,
    DetailedDrumClass,
    PredictionMap,
)

if TYPE_CHECKING:
    from src.benchmark.prediction_artifact import MappedPrediction

DRUM_LANE_IDS = _DRUM_LANE_IDS


@dataclass(frozen=True)
class MappingDiagnostics:
    unmapped: dict[str, int]


DEFAULT_MIDI_NOTE_MAP: dict[int, DetailedDrumClass] = {
    36: "kick",
    38: "snare",
    41: "low_or_floor_tom",
    42: "closed_hihat",
    45: "low_or_floor_tom",
    46: "open_hihat",
    47: "low_or_floor_tom",
    49: "crash",
    50: "high_tom",
    51: "ride",
}


def map_dtx_events(
    events: list[BenchmarkEvent],
    lane_map: Mapping[str, ClassMapping] | None = None,
) -> tuple[list[BenchmarkEvent], MappingDiagnostics]:
    lane_map = DTX_LANE_MAP if lane_map is None else lane_map
    mapped: list[BenchmarkEvent] = []
    unmapped: dict[str, int] = {}
    for event in events:
        lane_id = str(event.metadata.get("lane_id", event.canonical_class)).upper()
        class_mapping = lane_map.get(lane_id)
        if class_mapping is None:
            unmapped[lane_id] = unmapped.get(lane_id, 0) + 1
            continue
        metadata = {
            **event.metadata,
            "lane_id": lane_id,
            "common_class": class_mapping.common_class,
        }
        mapped.append(
            replace(
                event,
                canonical_class=class_mapping.canonical_class,
                metadata=metadata,
            )
        )
    return mapped, MappingDiagnostics(unmapped=unmapped)


def map_midi_events(
    events: list[BenchmarkEvent],
    note_map: Mapping[int, DetailedDrumClass] | None = None,
) -> tuple[list[BenchmarkEvent], MappingDiagnostics]:
    note_map = DEFAULT_MIDI_NOTE_MAP if note_map is None else note_map
    mapped: list[BenchmarkEvent] = []
    unmapped: dict[str, int] = {}
    for event in events:
        midi_note = int(event.metadata["midi_note"])
        canonical_class = note_map.get(midi_note)
        if canonical_class is None:
            key = str(midi_note)
            unmapped[key] = unmapped.get(key, 0) + 1
            continue
        mapped.append(replace(event, canonical_class=canonical_class))
    return mapped, MappingDiagnostics(unmapped=unmapped)


def map_oaf_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = OAF_PREDICTION_MAP,
) -> tuple["MappedPrediction", MappingDiagnostics]:
    """Map OaF native groups to the shared detailed/common drum taxonomy."""
    from src.benchmark.prediction_artifact import MappedPrediction, MappedPredictionEvent

    if prediction.descriptor.payload.get("backend_id") != prediction_map.backend_id:
        raise ValueError("prediction backend_id does not match prediction map")
    if prediction.descriptor.payload.get("native_output_space_id") != (
        prediction_map.native_output_space_id
    ):
        raise ValueError("prediction native_output_space_id does not match prediction map")

    mapped_events: list[MappedPredictionEvent] = []
    unmapped: dict[str, int] = {}
    for native in prediction.events:
        group = native.native_metadata["upstream_8hit_group_id"]
        key = "null" if group is None else str(group)
        class_mapping = prediction_map.classes.get(key)
        is_mapped = class_mapping is not None and class_mapping.common_class is not None
        if not is_mapped:
            unmapped[key] = unmapped.get(key, 0) + 1
        mapped_events.append(
            MappedPredictionEvent(
                native=native,
                canonical_class=(class_mapping.canonical_class if is_mapped else None),
                common_class=(class_mapping.common_class if is_mapped else None),
                mapping_status="mapped" if is_mapped else "unmapped",
                prediction_map_version=prediction_map.map_id,
            )
        )
    return MappedPrediction(
        audio=prediction.audio,
        descriptor=prediction.descriptor,
        events=tuple(mapped_events),
    ), MappingDiagnostics(unmapped=unmapped)
