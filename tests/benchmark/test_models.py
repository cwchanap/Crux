import pytest

from src.benchmark.models import BenchmarkEvent, DtxEvent, MatchResult, ScoreSummary


def test_dtx_event_normalizes_lane_id_to_uppercase():
    event = DtxEvent(chart_id="song", measure=0, position=0.5, lane_id="1a", note_id="01")
    assert event.lane_id == "1A"
    assert event.note_id == "01"


def test_benchmark_event_is_orderable_by_time_then_class():
    events = [
        BenchmarkEvent("song", 1.0, "snare", "prediction"),
        BenchmarkEvent("song", 0.5, "kick", "ground_truth"),
    ]
    assert [event.canonical_class for event in sorted(events)] == ["kick", "snare"]


def test_benchmark_event_is_hashable_with_metadata():
    event_with_meta = BenchmarkEvent("song", 1.0, "kick", "ground_truth", {"key": "value"})
    event_without_meta = BenchmarkEvent("song", 1.0, "kick", "ground_truth")
    # Must not raise TypeError despite dict field
    assert isinstance(hash(event_with_meta), int)
    assert isinstance(hash(event_without_meta), int)
    # Same non-metadata fields should produce same hash
    assert hash(event_with_meta) == hash(event_without_meta)


def test_score_summary_handles_zero_denominators():
    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    assert summary.precision is None
    assert summary.recall is None
    assert summary.f1 is None


def test_score_summary_computes_metrics_with_nonzero_counts():
    summary = ScoreSummary(true_positives=8, false_positives=2, false_negatives=2)
    assert summary.precision == 0.8
    assert summary.recall == 0.8
    assert summary.f1 == pytest.approx(0.8)


def test_score_summary_f1_zero_when_only_false_positives():
    summary = ScoreSummary(true_positives=0, false_positives=5, false_negatives=0)
    assert summary.precision == 0.0
    assert summary.recall is None
    assert summary.f1 == 0.0
    result = MatchResult(
        ground_truth=BenchmarkEvent("song", 0.5, "kick", "ground_truth"),
        prediction=BenchmarkEvent("song", 0.45, "kick", "prediction"),
        timing_error_sec=-0.05,
    )
    assert result.absolute_error_sec == 0.05


def test_score_summary_f1_zero_when_only_false_negatives():
    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=5)
    assert summary.precision is None
    assert summary.recall == 0.0
    assert summary.f1 == 0.0


def test_score_summary_f1_zero_when_no_matches():
    summary = ScoreSummary(true_positives=0, false_positives=3, false_negatives=4)
    assert summary.precision == 0.0
    assert summary.recall == 0.0
    assert summary.f1 == 0.0
