from src.benchmark.models import BenchmarkEvent
from src.benchmark.scoring import score_events


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
