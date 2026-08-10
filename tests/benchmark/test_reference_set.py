from __future__ import annotations

from decimal import Decimal

import pytest

from src.benchmark.reference_set import (
    CommonReferenceEvent,
    MappedReferenceEvent,
    map_reference_events,
    project_common_reference_events,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.taxonomy import DTX_LANE_MAP


def native_event(lane_id: str, audio_time_sec: float) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=7,
        selected_chart_key="7/chart.dtx",
        selected_chart_content_hash="a" * 64,
        source_audio_key="7/audio.wav",
        source_audio_content_hash="b" * 64,
        source_order=0,
        measure=1,
        position=0.0,
        lane_id=lane_id,
        note_id=f"{lane_id}-note",
        chart_time_sec=audio_time_sec,
        audio_time_sec=audio_time_sec,
    )


def mapped_event(lane_id: str, audio_time_sec: float) -> MappedReferenceEvent:
    native = native_event(lane_id, audio_time_sec)
    mapping = DTX_LANE_MAP[lane_id]
    assert mapping.canonical_class is not None
    assert mapping.common_class is not None
    return MappedReferenceEvent(
        native=native,
        canonical_class=mapping.canonical_class,
        common_class=mapping.common_class,
    )


def test_map_reference_events_preserves_native_identity_and_taxonomy() -> None:
    events = tuple(
        native_event(lane_id, float(index))
        for index, lane_id in enumerate(("13", "18", "14", "15", "16", "19"))
    )

    result = map_reference_events(events)

    assert tuple(mapped.native for mapped in result.mapped_events) == events
    assert all(mapped.native is event for mapped, event in zip(result.mapped_events, events))
    assert [mapped.canonical_class for mapped in result.mapped_events] == [
        "kick",
        "open_hihat",
        "high_tom",
        "low_or_floor_tom",
        "crash",
        "ride",
    ]
    assert [mapped.common_class for mapped in result.mapped_events] == [
        "kick",
        "hihat",
        "tom",
        "tom",
        "crash",
        "ride",
    ]


def test_map_reference_events_reports_unknown_and_ignored_without_mutating_source() -> None:
    unknown = native_event("ZZ", 1.0)
    ignored = native_event("2A", 2.0)
    events = (unknown, ignored)

    result = map_reference_events(events, ignored_lanes=frozenset({"2A"}))

    assert events == (unknown, ignored)
    assert result.mapped_events == ()
    assert result.common_events == ()
    assert result.diagnostics.unmapped == {"ZZ": 1}
    assert result.diagnostics.ignored == {"2A": 1}
    assert result.diagnostics.duplicate_common_event_count == 0


def test_map_reference_events_diagnostics_are_immutable() -> None:
    result = map_reference_events(
        (native_event("ZZ", 1.0), native_event("2A", 2.0)),
        ignored_lanes=frozenset({"2A"}),
    )

    with pytest.raises(TypeError):
        result.diagnostics.unmapped["ZZ"] = 2
    with pytest.raises(TypeError):
        result.diagnostics.ignored["2A"] = 2


def test_project_common_reference_events_collapses_only_exact_time_and_class() -> None:
    mapped = (
        mapped_event("14", 1.0),
        mapped_event("15", 1.0),
        mapped_event("15", 1.001),
    )

    common = project_common_reference_events(mapped)

    assert [(event.canonical_audio_time, event.common_class) for event in common] == [
        (Decimal("1.0"), "tom"),
        (Decimal("1.001"), "tom"),
    ]
    assert len(common[0].source_events) == 2
    assert common[0].source_events == mapped[:2]
    assert common[1].source_events == (mapped[2],)


def test_project_common_reference_events_collapses_open_and_closed_hihat_at_exact_time() -> None:
    mapped = (
        mapped_event("11", 2.0),
        mapped_event("18", 2.0),
        mapped_event("18", 2.000001),
    )

    common = project_common_reference_events(mapped)

    assert [(event.canonical_audio_time, event.common_class) for event in common] == [
        (Decimal("2.0"), "hihat"),
        (Decimal("2.000001"), "hihat"),
    ]
    assert common[0].source_events == mapped[:2]
    assert common[1].source_events == (mapped[2],)


def test_map_reference_events_composes_common_projection_and_duplicate_diagnostic() -> None:
    events = (
        native_event("14", 3.0),
        native_event("15", 3.0),
        native_event("15", 3.001),
    )

    result = map_reference_events(events)

    assert isinstance(result.common_events[0], CommonReferenceEvent)
    assert [(event.canonical_audio_time, event.common_class) for event in result.common_events] == [
        (Decimal("3.0"), "tom"),
        (Decimal("3.001"), "tom"),
    ]
    assert result.diagnostics.duplicate_common_event_count == 1
