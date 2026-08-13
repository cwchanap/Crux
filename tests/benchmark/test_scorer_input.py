from decimal import Decimal

from src.benchmark.reference_set import CommonReferenceEvent
from src.benchmark.scorer_input import reference_to_benchmark_events


def test_reference_common_events_become_ground_truth_events() -> None:
    common = (
        CommonReferenceEvent(Decimal("0.500000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.250000"), "hihat", ()),
    )

    result = reference_to_benchmark_events("42", common)

    assert [
        (item.chart_id, item.time_sec, item.canonical_class, item.source) for item in result
    ] == [
        ("42", 0.5, "kick", "ground_truth"),
        ("42", 1.25, "hihat", "ground_truth"),
    ]
    assert all(type(item.time_sec) is float for item in result)


def test_reference_adapter_preserves_common_projection_order() -> None:
    common = (
        CommonReferenceEvent(Decimal("1.000000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.000000"), "snare", ()),
    )
    result = reference_to_benchmark_events("song", common)
    assert [(item.time_sec, item.canonical_class) for item in result] == [
        (1.0, "kick"),
        (1.0, "snare"),
    ]
