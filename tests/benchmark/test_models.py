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


def test_score_summary_handles_zero_denominators():
    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    assert summary.precision == 0.0
    assert summary.recall == 0.0
    assert summary.f1 == 0.0
    result = MatchResult(
        ground_truth=BenchmarkEvent("song", 0.5, "kick", "ground_truth"),
        prediction=BenchmarkEvent("song", 0.45, "kick", "prediction"),
        timing_error_sec=-0.05,
    )
    assert result.absolute_error_sec == 0.05
