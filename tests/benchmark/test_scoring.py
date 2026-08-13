import numpy as np
import pytest

from src.benchmark.models import BenchmarkEvent
from src.benchmark.scoring import (
    _alignment_histogram,
    percentile,
    score_events,
    score_events_with_alignment,
)


def event(time_sec: float, canonical_class: str, source: str) -> BenchmarkEvent:
    return BenchmarkEvent("song", time_sec, canonical_class, source)


def test_percentile_uses_existing_upper_nearest_rank_convention() -> None:
    values = [0.0, 0.25, 0.5, 0.75, 1.0]

    assert percentile(values, 0.10) == 0.25
    assert percentile(values, 0.25) == 0.25
    assert percentile(values, 0.75) == 0.75
    assert percentile(values, 0.90) == 1.0


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


def test_matching_maximizes_true_positives_for_dense_hits():
    result = score_events(
        [event(0.1, "kick", "ground_truth_a"), event(0.2, "kick", "ground_truth_b")],
        [event(0.0, "kick", "prediction_a"), event(0.1, "kick", "prediction_b")],
        0.11,
    )

    assert result.summary.true_positives == 2
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 0


def test_simultaneous_same_class_hits_match_one_to_one() -> None:
    result = score_events(
        [event(1.0, "kick", "gt_a"), event(1.0, "kick", "gt_b")],
        [event(1.0, "kick", "pred_a"), event(1.0, "kick", "pred_b")],
        0.03,
    )
    assert result.summary.true_positives == 2
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 0


def test_empty_prediction_song_counts_all_references_as_false_negatives() -> None:
    result = score_events(
        [event(1.0, "kick", "gt"), event(2.0, "snare", "gt")],
        [],
        0.05,
    )
    assert result.summary.true_positives == 0
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 2
    assert result.summary.f1 == 0.0


def test_class_present_on_only_one_side_is_retained() -> None:
    result = score_events(
        [event(1.0, "kick", "gt")],
        [event(1.0, "snare", "pred")],
        0.05,
    )
    assert [item.canonical_class for item in result.unmatched_ground_truth] == ["kick"]
    assert [item.canonical_class for item in result.unmatched_predictions] == ["snare"]


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


def test_alignment_chooses_stable_offset_for_symmetric_tie():
    ground_truth = [event(1.0, "kick", "ground_truth"), event(2.0, "kick", "ground_truth")]
    predictions = [event(0.9, "kick", "prediction_a"), event(2.1, "kick", "prediction_b")]

    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec=0.05)

    assert result.aligned.summary.true_positives == 1
    assert round(result.aligned.summary.offset_sec, 3) == -0.1


def test_alignment_refines_offset_beyond_histogram_bin():
    ground_truth = [event(1.0, "kick", "ground_truth"), event(2.0, "snare", "ground_truth")]
    predictions = [event(1.123, "kick", "prediction"), event(2.123, "snare", "prediction")]

    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec=0.05)

    assert result.aligned.summary.true_positives == 2
    assert round(result.aligned.summary.offset_sec, 3) == -0.123


def test_alignment_histogram_rejects_negative_times():
    with pytest.raises(ValueError, match="negative times"):
        _alignment_histogram([1.0, -0.5, 2.0], num_bins=10, bin_size=0.01)


def test_alignment_histogram_accepts_non_negative_times():
    result = _alignment_histogram([0.0, 0.01, 0.02], num_bins=10, bin_size=0.01)
    assert isinstance(result, np.ndarray)
    assert result.sum() == 3


def test_alignment_histogram_accepts_empty_input():
    result = _alignment_histogram([], num_bins=10, bin_size=0.01)
    assert isinstance(result, np.ndarray)
    assert result.sum() == 0


def test_alignment_empty_ground_truth_returns_zero_offset():
    pred = [event(1.0, "kick", "prediction")]
    result = score_events_with_alignment([], pred, 0.05)

    assert result.raw.summary.offset_sec == 0.0
    assert result.aligned.summary.offset_sec == 0.0
    assert result.raw.summary.true_positives == 0
    assert result.raw.summary.false_positives == 1


def test_alignment_no_shared_classes_returns_zero_offset():
    gt = [event(1.0, "kick", "ground_truth")]
    pred = [event(1.0, "snare", "prediction")]
    result = score_events_with_alignment(gt, pred, 0.05)

    assert result.raw.summary.offset_sec == 0.0
    assert result.aligned.summary.offset_sec == 0.0
