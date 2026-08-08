from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

from src.benchmark.dtx_parser import DtxBgmEvent, DtxBpmEvent, ParsedDtxChart
from src.benchmark.models import BenchmarkEvent, DtxEvent

BEATS_PER_MEASURE = 4.0


@dataclass
class DtxTimingMap:
    """Resolved DTX chart-time timing for native, BGM and BPM events.

    Encapsulates the existing timing math (sticky measure lengths, measure
    start beats, tempo points) so that playable events, BGM control data and
    tempo events all resolve through one consistent clock.  Built once via
    :func:`build_dtx_timing_map`; query per event with :meth:`time_sec`.
    """

    _measure_lengths: list[float]
    _measure_starts: list[float]
    _tempo_points: list[tuple[float, float, float]]

    def time_sec(self, event: DtxEvent | DtxBgmEvent | DtxBpmEvent) -> float:
        beat = _event_beat(event, self._measure_lengths, self._measure_starts)
        return _time_at_beat(beat, self._tempo_points)


def build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap:
    measure_lengths = _measure_lengths_by_measure(chart)
    measure_starts = _measure_start_beats(measure_lengths)
    tempo_points = _tempo_points(chart, measure_starts, measure_lengths)
    return DtxTimingMap(measure_lengths, measure_starts, tempo_points)


def dtx_events_to_chart_time_events(chart: ParsedDtxChart) -> list[BenchmarkEvent]:
    # Legacy folder/MIDI scoring uses chart time. HPA-325 consumes HPA-323
    # audio-time reference artifacts and must not use this path.
    timing_map = build_dtx_timing_map(chart)
    return [
        BenchmarkEvent(
            chart_id=event.chart_id,
            time_sec=timing_map.time_sec(event),
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
    measures.extend(event.measure for event in chart.bgm_events)
    measures.extend(event.measure for event in chart.bpm_events)
    measures.extend(chart.measure_lengths)
    return max(measures, default=0)


def _measure_lengths_by_measure(chart: ParsedDtxChart) -> list[float]:
    """Resolve a per-measure length list honouring the sticky channel-02 rule.

    Channel 02 is sticky: once a measure length is set it carries forward to
    every later measure until another explicit length supersedes it.  Measure 0
    with no explicit length stays at the 1.0 default.
    """
    lengths: list[float] = []
    current = 1.0
    for measure in range(_max_measure(chart) + 2):
        if measure in chart.measure_lengths:
            current = chart.measure_lengths[measure]
        if current <= 0:
            raise ValueError(f"measure {measure} has non-positive length")
        lengths.append(current)
    return lengths


def _measure_start_beats(measure_lengths: list[float]) -> list[float]:
    starts = [0.0]
    for length in measure_lengths:
        starts.append(starts[-1] + length * BEATS_PER_MEASURE)
    return starts


def _event_beat(
    event: DtxEvent | DtxBgmEvent | DtxBpmEvent,
    measure_lengths: list[float],
    starts: list[float],
) -> float:
    current_length = measure_lengths[event.measure]
    return starts[event.measure] + event.position * current_length * BEATS_PER_MEASURE


def _tempo_points(
    chart: ParsedDtxChart,
    measure_starts: list[float],
    measure_lengths: list[float],
) -> list[tuple[float, float, float]]:
    points = [(0.0, chart.base_bpm)]
    for event in chart.bpm_events:
        points.append((_event_beat(event, measure_lengths, measure_starts), event.bpm))
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
