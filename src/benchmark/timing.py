from __future__ import annotations

import math
from bisect import bisect_right

from src.benchmark.dtx_parser import DtxBpmEvent, ParsedDtxChart
from src.benchmark.models import BenchmarkEvent, DtxEvent

BEATS_PER_MEASURE = 4.0


def dtx_events_to_timed_events(chart: ParsedDtxChart) -> list[BenchmarkEvent]:
    measure_starts = _measure_start_beats(chart)
    tempo_points = _tempo_points(chart, measure_starts)
    return [
        BenchmarkEvent(
            chart_id=event.chart_id,
            time_sec=_time_at_beat(
                _event_beat(event, chart.measure_lengths, measure_starts),
                tempo_points,
            ),
            canonical_class=event.lane_id,
            source="ground_truth",
            metadata={"lane_id": event.lane_id, "note_id": event.note_id},
        )
        for event in sorted(
            chart.events, key=lambda item: (item.measure, item.position, item.lane_id)
        )
    ]


def _max_measure(chart: ParsedDtxChart) -> int:
    measures = [event.measure for event in chart.events]
    measures.extend(event.measure for event in chart.bpm_events)
    measures.extend(chart.measure_lengths)
    return max(measures, default=0)


def _measure_lengths_by_measure(chart: ParsedDtxChart) -> list[float]:
    current = 1.0
    lengths: list[float] = []
    for measure in range(_max_measure(chart) + 2):
        current = chart.measure_lengths.get(measure, current)
        if current <= 0:
            raise ValueError(f"measure {measure} has non-positive length")
        lengths.append(current)
    return lengths


def _measure_start_beats(chart: ParsedDtxChart) -> list[float]:
    starts = [0.0]
    for length in _measure_lengths_by_measure(chart):
        starts.append(starts[-1] + length * BEATS_PER_MEASURE)
    return starts


def _event_beat(
    event: DtxEvent | DtxBpmEvent, measure_lengths: dict[int, float], starts: list[float]
) -> float:
    current_length = 1.0
    for measure in range(event.measure + 1):
        current_length = measure_lengths.get(measure, current_length)
    return starts[event.measure] + event.position * current_length * BEATS_PER_MEASURE


def _tempo_points(
    chart: ParsedDtxChart, measure_starts: list[float]
) -> list[tuple[float, float, float]]:
    points = [(0.0, chart.base_bpm)]
    for event in chart.bpm_events:
        points.append((_event_beat(event, chart.measure_lengths, measure_starts), event.bpm))
    points.sort(key=lambda item: item[0])

    resolved: list[tuple[float, float, float]] = []
    current_time = 0.0
    previous_beat = points[0][0]
    previous_bpm = points[0][1]
    for beat, bpm in points:
        if resolved and math.isclose(beat, resolved[-1][0], abs_tol=1e-9):
            chart.warnings.append(f"duplicate tempo at beat {beat:.6f}; using last value")
            resolved[-1] = (resolved[-1][0], resolved[-1][1], bpm)
            previous_bpm = bpm
            continue
        if beat > previous_beat:
            current_time += (beat - previous_beat) * 60.0 / previous_bpm
        resolved.append((beat, current_time, bpm))
        previous_beat = beat
        previous_bpm = bpm
    return resolved


def _time_at_beat(beat: float, tempo_points: list[tuple[float, float, float]]) -> float:
    beats = [point[0] for point in tempo_points]
    index = max(0, bisect_right(beats, beat) - 1)
    tempo_beat, tempo_time, bpm = tempo_points[index]
    return tempo_time + (beat - tempo_beat) * 60.0 / bpm
