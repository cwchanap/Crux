"""Compare native events at the precision of the archived TF1 predictions."""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby
from math import ceil
from statistics import median

from runtime.oaf_tf1.model import frame_time_seconds
from src.benchmark.backend_identity import JsonValue, quantize_six
from src.benchmark.backends.base import NativeEvent
from src.benchmark.backends.oaf import _decode_native_event
from src.benchmark.models import BenchmarkEvent
from src.benchmark.oaf_smoke_oracle import SmokeOracleEvent
from src.benchmark.prediction_artifact import PredictionArtifact
from src.benchmark.scoring import ScoreResult, score_events

MICROSECONDS_PER_SECOND = 1_000_000
STRICT_TOLERANCE_SEC = Decimal("0.001")
FRAME_SHIFT_TOLERANCE_SEC = frame_time_seconds(1) + 0.000001
MAX_EXAMPLE_DELTAS = 8
CHANGE_FIELDS = (
    "model_output_bin",
    "native_midi_note",
    "upstream_8hit_group_id",
    "velocity_midi",
)


def baseline_native_events(artifact: PredictionArtifact) -> tuple[NativeEvent, ...]:
    """Extract native events from the verified archive, preserving its recorded order."""
    return tuple(event.native for event in artifact.prediction.events)


def candidate_native_events(events: tuple[SmokeOracleEvent, ...]) -> tuple[NativeEvent, ...]:
    """Decode the experiment's exact native event records for comparison."""
    return tuple(
        _decode_native_event(
            {
                "time_sec": struct.unpack(">d", bytes.fromhex(event.time_sec_binary64))[0],
                "native_class_id": event.native_class_id,
                "model_output_bin": event.model_output_bin,
                "native_midi_note": event.native_midi_note,
                "upstream_8hit_group_id": event.upstream_8hit_group_id,
                "confidence": (
                    None
                    if event.confidence_binary64 is None
                    else struct.unpack(">d", bytes.fromhex(event.confidence_binary64))[0]
                ),
                "velocity_midi": event.velocity_midi,
            }
        )
        for event in events
    )


@dataclass(frozen=True)
class _ArchivedEvent:
    index: int
    native: NativeEvent
    time_sec: Decimal
    confidence: Decimal | None

    def sort_key(self) -> tuple[Decimal, str, int, int, int, Decimal]:
        native = self.native
        return (
            self.time_sec,
            native.native_class_id,
            -1 if native.model_output_bin is None else native.model_output_bin,
            -1 if native.native_midi_note is None else native.native_midi_note,
            -1 if native.velocity_midi is None else native.velocity_midi,
            Decimal(-1) if self.confidence is None else self.confidence,
        )

    def native_fields(self) -> dict[str, JsonValue]:
        return {
            "time_sec": self.time_sec,
            "confidence": self.confidence,
            "native_class_id": self.native.native_class_id,
            "model_output_bin": self.native.model_output_bin,
            "native_midi_note": self.native.native_midi_note,
            "upstream_8hit_group_id": self.native.native_metadata.get("upstream_8hit_group_id"),
            "velocity_midi": self.native.velocity_midi,
        }

    def as_match_event(self, side: str) -> BenchmarkEvent:
        # The matcher only sorts, compares and subtracts times. Match on the integer
        # microsecond grid so a pair exactly 1 ms apart always meets the boundary.
        return BenchmarkEvent(
            chart_id="",
            time_sec=int(self.time_sec * MICROSECONDS_PER_SECOND),
            canonical_class=self.native.native_class_id,
            source=side,
            metadata={"original_index": self.index, **self.native_fields()},
        )


def _archived_view(events: tuple[NativeEvent, ...]) -> list[_ArchivedEvent]:
    rounded = [
        _ArchivedEvent(
            index,
            event,
            quantize_six(event.time_sec),
            None if event.confidence is None else quantize_six(event.confidence),
        )
        for index, event in enumerate(events)
    ]
    return sorted(rounded, key=_ArchivedEvent.sort_key)


def _original_indexes(events: list[BenchmarkEvent]) -> list[int]:
    return sorted(event.metadata["original_index"] for event in events)


def _absolute_stats(deltas: list[Decimal]) -> dict[str, JsonValue]:
    if not deltas:
        return {"max": None, "median": None, "p95": None}
    absolute = sorted(abs(delta) for delta in deltas)
    return {
        "max": absolute[-1],
        "median": median(absolute),
        "p95": absolute[ceil(Decimal("0.95") * (len(absolute) - 1))],
    }


def _pair_identity(baseline: _ArchivedEvent, candidate: _ArchivedEvent) -> dict[str, JsonValue]:
    return {
        "baseline_index": baseline.index,
        "candidate_index": candidate.index,
        "time_delta_sec": candidate.time_sec - baseline.time_sec,
    }


def _report_matches(
    result: ScoreResult,
    baseline: dict[int, _ArchivedEvent],
    candidate: dict[int, _ArchivedEvent],
    *,
    bucket: str,
    examples: list[JsonValue],
) -> dict[str, JsonValue]:
    time_deltas: list[Decimal] = []
    confidence_deltas: list[Decimal] = []
    missing_confidence = {"baseline": 0, "candidate": 0, "either": 0}
    changed_fields = dict.fromkeys(CHANGE_FIELDS, 0)
    pairs: list[JsonValue] = []
    for match in result.matches:
        left = baseline[match.ground_truth.metadata["original_index"]]
        right = candidate[match.prediction.metadata["original_index"]]
        time_delta = right.time_sec - left.time_sec
        time_deltas.append(time_delta)
        if left.confidence is None or right.confidence is None:
            missing_confidence["baseline"] += left.confidence is None
            missing_confidence["candidate"] += right.confidence is None
            missing_confidence["either"] += 1
            confidence_delta = None
        else:
            confidence_delta = right.confidence - left.confidence
            confidence_deltas.append(confidence_delta)

        left_fields, right_fields = left.native_fields(), right.native_fields()
        changes: dict[str, JsonValue] = {}
        for field in CHANGE_FIELDS:
            if left_fields[field] != right_fields[field]:
                changed_fields[field] += 1
                changes[field] = {"baseline": left_fields[field], "candidate": right_fields[field]}

        if bucket == "frame_shift_candidates":
            pairs.append(_pair_identity(left, right))
        if len(examples) < MAX_EXAMPLE_DELTAS and (
            time_delta or left.confidence != right.confidence or changes
        ):
            examples.append(
                {
                    **_pair_identity(left, right),
                    "bucket": bucket,
                    "time_abs_delta_sec": abs(time_delta),
                    "confidence_delta": confidence_delta,
                    "confidence_abs_delta": (
                        None if confidence_delta is None else abs(confidence_delta)
                    ),
                    "changed_fields": changes,
                }
            )

    report: dict[str, JsonValue] = {
        "matched_count": len(result.matches),
        "unmatched_baseline_count": len(result.unmatched_ground_truth),
        "unmatched_candidate_count": len(result.unmatched_predictions),
        "time_abs_sec": _absolute_stats(time_deltas),
        "confidence_abs": _absolute_stats(confidence_deltas),
        "missing_confidence": {**missing_confidence},
        "changed_fields": {**changed_fields},
    }
    if bucket == "frame_shift_candidates":
        report["pairs"] = pairs
    return report


def _rounding_collisions(events: list[_ArchivedEvent]) -> dict[str, JsonValue]:
    collisions: list[JsonValue] = []
    event_count = 0
    for _, group in groupby(events, key=_ArchivedEvent.sort_key):
        indexes = [event.index for event in group]
        if len(indexes) > 1:
            collisions.append([*indexes])
            event_count += len(indexes)
    return {
        "group_count": len(collisions),
        "event_count": event_count,
        "candidate_index_groups": collisions,
    }


def compare_events(
    baseline: tuple[NativeEvent, ...], candidate: tuple[NativeEvent, ...]
) -> dict[str, JsonValue]:
    """Measure strict archive agreement and possible one-frame shifts among residuals."""
    left = _archived_view(baseline)
    right = _archived_view(candidate)
    strict = score_events(
        [event.as_match_event("baseline") for event in left],
        [event.as_match_event("candidate") for event in right],
        tolerance_sec=int(STRICT_TOLERANCE_SEC * MICROSECONDS_PER_SECOND),
        offset_sec=0,
    )
    shifts = score_events(
        strict.unmatched_ground_truth,
        strict.unmatched_predictions,
        tolerance_sec=int(FRAME_SHIFT_TOLERANCE_SEC * MICROSECONDS_PER_SECOND),
        offset_sec=0,
    )

    left_by_index = {event.index: event for event in left}
    right_by_index = {event.index: event for event in right}
    examples: list[JsonValue] = []
    shift_report = _report_matches(
        shifts, left_by_index, right_by_index, bucket="frame_shift_candidates", examples=examples
    )
    strict_report = _report_matches(
        strict, left_by_index, right_by_index, bucket="strict", examples=examples
    )
    return {
        "ordered_equal": [event.native_fields() for event in left]
        == [event.native_fields() for event in right],
        "baseline_native_class_counts": dict(Counter(event.native_class_id for event in baseline)),
        "candidate_native_class_counts": dict(
            Counter(event.native_class_id for event in candidate)
        ),
        "rounding_collisions": _rounding_collisions(right),
        "strict": strict_report,
        "strict_unmatched_baseline": [*_original_indexes(strict.unmatched_ground_truth)],
        "strict_unmatched_candidate": [*_original_indexes(strict.unmatched_predictions)],
        "frame_shift_candidates": shift_report,
        "remaining_removals": [*_original_indexes(shifts.unmatched_ground_truth)],
        "remaining_additions": [*_original_indexes(shifts.unmatched_predictions)],
        "example_deltas": examples,
    }
