"""Archive-precision comparisons must distinguish agreement from frame shifts."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.oaf_tf215.selected_metrics import (
    compare_events,
    decode_candidate_native_events,
    extract_baseline_native_events,
)
from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.backends.base import NativeEvent
from src.benchmark.oaf_smoke_oracle import _event_from_native
from src.benchmark.prediction_artifact import read_prediction_artifact


def _event(time_sec: float, native_class_id: str = "midi_38") -> NativeEvent:
    return NativeEvent(
        time_sec, native_class_id, 17, 38, {"upstream_8hit_group_id": "snare"}, 0.5, 80
    )


def test_event_adapters_recover_native_fields_and_candidate_bits() -> None:
    path = Path(__file__).parent / "schema_goldens" / "crux.drum-prediction-events-v2.jsonl"
    baseline = extract_baseline_native_events(read_prediction_artifact(path.read_bytes()))
    assert baseline == (
        NativeEvent(0.5, "midi_46", 25, 46, {"upstream_8hit_group_id": "hihat"}, 0.75, 96),
    )

    native = (_event(0.09287981859410431), replace(_event(-0.0), confidence=None))
    raw = tuple(_event_from_native(event) for event in native)
    assert tuple(_event_from_native(event) for event in decode_candidate_native_events(raw)) == raw


def test_inserted_event_does_not_shift_later_pairing() -> None:
    baseline = (_event(1.0), _event(2.0))
    candidate = (_event(0.5, "midi_36"), *baseline)
    result = compare_events(baseline, candidate)

    assert result["ordered_equal"] is False
    assert result["baseline_native_class_counts"] == {"midi_38": 2}
    assert result["candidate_native_class_counts"] == {"midi_36": 1, "midi_38": 2}
    assert result["strict"]["matched_count"] == 2
    assert result["frame_shift_candidates"]["matched_count"] == 0
    assert result["remaining_additions"] == [0]
    assert result["remaining_removals"] == []


@pytest.mark.parametrize(
    ("baseline_time", "candidate_time", "signed_delta"),
    [
        (0.003, 0.003 + 512 / 44100, "0.011610"),
        (0.003 + 512 / 44100, 0.003, "-0.011610"),
        (19 * 512 / 44100, 20 * 512 / 44100, "0.011610"),
    ],
)
def test_single_frame_is_only_a_residual_match(
    baseline_time: float, candidate_time: float, signed_delta: str
) -> None:
    result = compare_events((_event(baseline_time),), (_event(candidate_time),))

    assert result["strict"]["matched_count"] == 0
    assert result["strict_unmatched_baseline"] == [0]
    assert result["strict_unmatched_candidate"] == [0]
    assert result["frame_shift_candidates"]["matched_count"] == 1
    assert result["frame_shift_candidates"]["pairs"] == [
        {"baseline_index": 0, "candidate_index": 0, "time_delta_sec": Decimal(signed_delta)}
    ]
    assert result["frame_shift_candidates"]["time_abs_sec"]["max"] == abs(Decimal(signed_delta))
    assert result["remaining_additions"] == result["remaining_removals"] == []


def test_residuals_cannot_reuse_strict_matches_or_pair_at_two_frames() -> None:
    frame = 512 / 44100
    baseline = (_event(0.0), _event(frame), _event(1.0), _event(2.0))
    candidate = (_event(0.0), _event(1.0 + 2 * frame), _event(2.0 + frame, "midi_36"))
    result = compare_events(baseline, candidate)

    assert result["strict"]["matched_count"] == 1
    assert result["frame_shift_candidates"]["matched_count"] == 0
    assert result["remaining_removals"] == [1, 2, 3]
    assert result["remaining_additions"] == [1, 2]


def test_ambiguous_residual_uses_first_time_in_stable_matcher_order() -> None:
    frame = 512 / 44100
    result = compare_events((_event(2 * frame), _event(0.0)), (_event(frame),))

    assert result["frame_shift_candidates"]["pairs"] == [
        {"baseline_index": 1, "candidate_index": 0, "time_delta_sec": Decimal("0.011610")}
    ]
    assert result["remaining_removals"] == [0]
    assert result["remaining_additions"] == []


def test_rounded_order_is_stable_and_duplicate_keys_preserve_multiplicity() -> None:
    quiet = replace(_event(1.0), confidence=0.1)
    loud = replace(_event(1.0), confidence=0.6)
    reordered = (
        replace(loud, time_sec=1.0000001),
        replace(quiet, time_sec=1.0000002),
    )
    result = compare_events((quiet, loud), reordered)
    assert result["ordered_equal"] is True
    assert result["strict"]["confidence_abs"]["max"] == Decimal(0)

    result = compare_events((quiet,), (quiet, replace(quiet, time_sec=1.0000002)))
    assert result["ordered_equal"] is False
    assert result["strict"]["matched_count"] == 1
    assert result["rounding_collisions"] == {
        "group_count": 1,
        "event_count": 2,
        "candidate_index_groups": [[0, 1]],
    }
    assert result["remaining_additions"] == [1]


def test_metric_summary_is_exact_and_example_cap_does_not_limit_counts() -> None:
    baseline = tuple(_event(float(index)) for index in range(10))
    candidate = tuple(
        replace(
            event,
            time_sec=event.time_sec + 0.001,
            confidence=None if index == 0 else 0.5 - index / 100,
            velocity_midi=70,
            model_output_bin=18,
            native_midi_note=39,
            native_metadata={"upstream_8hit_group_id": "tom"},
        )
        for index, event in enumerate(baseline)
    )
    result = compare_events(baseline, candidate)
    strict = result["strict"]

    assert strict["matched_count"] == 10  # The exact 1 ms boundary is inclusive.
    assert strict["time_abs_sec"] == {key: Decimal("0.001") for key in ("max", "median", "p95")}
    assert strict["confidence_abs"] == {
        "max": Decimal("0.09"),
        "median": Decimal("0.05"),
        "p95": Decimal("0.09"),
    }
    assert strict["missing_confidence"] == {"baseline": 0, "candidate": 1, "either": 1}
    assert strict["changed_fields"] == {
        "model_output_bin": 10,
        "native_midi_note": 10,
        "upstream_8hit_group_id": 10,
        "velocity_midi": 10,
    }
    assert len(result["example_deltas"]) == 8
    assert result["example_deltas"][1]["confidence_delta"] == Decimal("-0.01")
    assert strict_json_loads(canonical_json_bytes(result)) == result


def test_empty_comparison_has_no_delta_statistics() -> None:
    result = compare_events((), ())

    assert result["ordered_equal"] is True
    assert result["strict"]["time_abs_sec"] == {"max": None, "median": None, "p95": None}
    assert result["frame_shift_candidates"]["confidence_abs"] == {
        "max": None,
        "median": None,
        "p95": None,
    }
    assert result["example_deltas"] == []
