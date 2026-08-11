"""Pure in-memory mapping and common projection for native references."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.taxonomy import (
    DTX_LANE_MAP,
    IGNORED_NON_DRUM_LANES,
    ClassMapping,
    CommonDrumClass,
    DetailedDrumClass,
    project_to_common,
)


@dataclass(frozen=True)
class MappedReferenceEvent:
    native: NativeReferenceEvent
    canonical_class: DetailedDrumClass
    common_class: CommonDrumClass


@dataclass(frozen=True)
class CommonReferenceEvent:
    canonical_audio_time: Decimal
    common_class: CommonDrumClass
    source_events: tuple[MappedReferenceEvent, ...]


@dataclass(frozen=True)
class ReferenceMappingDiagnostics:
    unmapped: Mapping[str, int]
    ignored: Mapping[str, int]
    duplicate_common_event_count: int


@dataclass(frozen=True)
class ReferenceMappingResult:
    mapped_events: tuple[MappedReferenceEvent, ...]
    common_events: tuple[CommonReferenceEvent, ...]
    diagnostics: ReferenceMappingDiagnostics


def project_common_reference_events(
    mapped_events: tuple[MappedReferenceEvent, ...],
) -> tuple[CommonReferenceEvent, ...]:
    """Collapse mapped events by exact audio time and common class."""
    grouped: dict[tuple[Decimal, CommonDrumClass], list[MappedReferenceEvent]] = {}
    for event in mapped_events:
        key = (Decimal(str(event.native.audio_time_sec)), event.common_class)
        grouped.setdefault(key, []).append(event)

    return tuple(
        CommonReferenceEvent(
            canonical_audio_time=canonical_audio_time,
            common_class=common_class,
            source_events=tuple(grouped[(canonical_audio_time, common_class)]),
        )
        for canonical_audio_time, common_class in sorted(grouped, key=lambda key: (key[0], key[1]))
    )


def map_reference_events(
    events: tuple[NativeReferenceEvent, ...],
    *,
    lane_map: Mapping[str, ClassMapping] = DTX_LANE_MAP,
    ignored_lanes: frozenset[str] = IGNORED_NON_DRUM_LANES,
) -> ReferenceMappingResult:
    """Map native lanes and project the mapped events into common classes."""
    mapped_events: list[MappedReferenceEvent] = []
    unmapped: dict[str, int] = {}
    ignored: dict[str, int] = {}

    for event in events:
        lane_id = event.lane_id
        if lane_id in ignored_lanes:
            ignored[lane_id] = ignored.get(lane_id, 0) + 1
            continue

        mapping = lane_map.get(lane_id)
        if mapping is None or mapping.canonical_class is None:
            unmapped[lane_id] = unmapped.get(lane_id, 0) + 1
            continue

        common_class = mapping.common_class or project_to_common(mapping.canonical_class)
        mapped_events.append(
            MappedReferenceEvent(
                native=event,
                canonical_class=mapping.canonical_class,
                common_class=common_class,
            )
        )

    mapped = tuple(mapped_events)
    common = project_common_reference_events(mapped)
    diagnostics = ReferenceMappingDiagnostics(
        unmapped=MappingProxyType(dict(sorted(unmapped.items()))),
        ignored=MappingProxyType(dict(sorted(ignored.items()))),
        duplicate_common_event_count=sum(len(event.source_events) - 1 for event in common),
    )
    return ReferenceMappingResult(
        mapped_events=mapped,
        common_events=common,
        diagnostics=diagnostics,
    )
