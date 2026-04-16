import pytest

from src.benchmark.models import BenchmarkEvent
from src.benchmark.scoring import score_events, score_events_with_alignment


def event(time_sec: float, canonical_class: str, source: str) -> BenchmarkEvent:
    return BenchmarkEvent("song", time_sec, canonical_class, source)


def test_scores_exact_match():
    result = score_events(
        [event(1.0, "kick", "ground_truth")], [event(1.0, "kick", "prediction")], 0.05
    )

    assert result.summary.true_positives == 1
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 0
    assert result.summary.f1 == 1.0


def test_class_mismatch_does_not_match():
    result = score_events(
        [event(1.0, "kick", "ground_truth")], [event(1.0, "snare", "prediction")], 0.05
    )

    assert result.summary.true_positives == 0
    assert result.summary.false_positives == 1
    assert result.summary.false_negatives == 1


def test_duplicate_predictions_leave_false_positive():
    result = score_events(
        [event(1.0, "kick", "ground_truth")],
        [event(1.0, "kick", "prediction"), event(1.01, "kick", "prediction")],
        0.05,
    )

    assert result.summary.true_positives == 1
    assert result.summary.false_positives == 1


def test_p95_uses_upper_tail_element():
    ground_truth = [event(float(index * 10), "kick", "ground_truth") for index in range(12)]
    predictions = [
        event(float(index * 10) + error, "kick", "prediction")
        for index, error in enumerate(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
        )
    ]

    result = score_events(ground_truth, predictions, 0.5)

    assert result.summary.p95_abs_error_sec == pytest.approx(0.12)


def test_equal_distance_candidates_choose_lowest_sorted_index():
    result = score_events(
        [event(1.0, "kick", "ground_truth")],
        [event(0.9, "kick", "prediction_a"), event(1.1, "kick", "prediction_b")],
        0.2,
    )

    assert result.matches[0].prediction.source == "prediction_a"
    assert [prediction.source for prediction in result.unmatched_predictions] == ["prediction_b"]


def test_alignment_applies_single_global_offset():
    ground_truth = [event(1.0, "kick", "ground_truth"), event(2.0, "snare", "ground_truth")]
    predictions = [event(1.1, "kick", "prediction"), event(2.1, "snare", "prediction")]

    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec=0.05)

    assert result.raw.summary.true_positives == 0
    assert result.aligned.summary.true_positives == 2
    assert round(result.aligned.summary.offset_sec, 3) == -0.1


def test_alignment_recovers_offsets_beyond_previous_search_window():
    ground_truth = [event(1.0, "kick", "ground_truth"), event(2.0, "snare", "ground_truth")]
    predictions = [event(1.3, "kick", "prediction"), event(2.3, "snare", "prediction")]

    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec=0.05)

    assert result.raw.summary.true_positives == 0
    assert result.aligned.summary.true_positives == 2
    assert round(result.aligned.summary.offset_sec, 3) == -0.3
