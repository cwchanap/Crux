from __future__ import annotations

from dataclasses import dataclass, replace

from src.benchmark.models import BenchmarkEvent


@dataclass(frozen=True)
class DtxClassMapping:
    collapsed_class: str
    native_class: str


@dataclass(frozen=True)
class MappingDiagnostics:
    unmapped: dict[str, int]


DEFAULT_DTX_LANE_MAP: dict[str, DtxClassMapping] = {
    "01": DtxClassMapping("kick", "kick"),
    "02": DtxClassMapping("snare", "snare"),
    "03": DtxClassMapping("closed_hihat", "closed_hihat"),
    "04": DtxClassMapping("open_hihat", "open_hihat"),
    "05": DtxClassMapping("crash", "crash"),
    "06": DtxClassMapping("ride", "ride"),
    "07": DtxClassMapping("low_tom", "low_tom"),
    "08": DtxClassMapping("mid_tom", "mid_tom"),
    "09": DtxClassMapping("high_tom", "high_tom"),
    "0A": DtxClassMapping("closed_hihat", "pedal_hihat"),
    "0B": DtxClassMapping("crash", "crash_2"),
    "0C": DtxClassMapping("ride", "ride_2"),
    "11": DtxClassMapping("closed_hihat", "closed_hihat"),
    "12": DtxClassMapping("snare", "snare"),
    "13": DtxClassMapping("kick", "kick"),
    "14": DtxClassMapping("high_tom", "high_tom"),
    "15": DtxClassMapping("low_tom", "low_tom"),
    "16": DtxClassMapping("crash", "crash"),
    "17": DtxClassMapping("low_tom", "floor_tom"),
    "18": DtxClassMapping("open_hihat", "open_hihat"),
    "19": DtxClassMapping("ride", "ride"),
    "1A": DtxClassMapping("crash", "left_cymbal"),
    "1B": DtxClassMapping("closed_hihat", "pedal_hihat"),
    "1C": DtxClassMapping("kick", "left_kick"),
}

DEFAULT_MIDI_NOTE_MAP: dict[int, str] = {
    36: "kick",
    38: "snare",
    41: "low_tom",
    42: "closed_hihat",
    45: "low_tom",
    46: "open_hihat",
    47: "mid_tom",
    49: "crash",
    50: "high_tom",
    51: "ride",
}


def map_dtx_events(
    events: list[BenchmarkEvent],
    lane_map: dict[str, DtxClassMapping] | None = None,
) -> tuple[list[BenchmarkEvent], MappingDiagnostics]:
    lane_map = DEFAULT_DTX_LANE_MAP if lane_map is None else lane_map
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
            "native_class": class_mapping.native_class,
        }
        mapped.append(
            replace(
                event,
                canonical_class=class_mapping.collapsed_class,
                metadata=metadata,
            )
        )
    return mapped, MappingDiagnostics(unmapped=unmapped)


def map_midi_events(
    events: list[BenchmarkEvent],
    note_map: dict[int, str] | None = None,
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
