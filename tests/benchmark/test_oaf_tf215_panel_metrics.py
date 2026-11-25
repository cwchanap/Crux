"""Score the TF2.15 panel using the production common drum taxonomy."""

from decimal import Decimal
from pathlib import Path

import pytest

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backends.base import NativeEvent
from src.benchmark.oaf_smoke_oracle import _event_from_native
from src.benchmark.oaf_tf215_runtime import build_candidate_descriptor
from src.benchmark.prediction_artifact import read_prediction_artifact
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_timing import NativeReferenceEvent


def test_panel_score_rounds_before_raw_matching_and_preserves_extra_events() -> None:
    from experiments.oaf_tf215.panel_metrics import score_panel_song

    path = Path(__file__).parent / "schema_goldens" / "crux.drum-prediction-events-v2.jsonl"
    baseline = read_prediction_artifact(path.read_bytes())
    mapping = map_reference_events(
        (
            NativeReferenceEvent(
                simfile_id=10,
                selected_chart_key="10/chart.dtx",
                selected_chart_content_hash="a" * 64,
                source_audio_key="10/audio.wav",
                source_audio_content_hash="b" * 64,
                source_order=0,
                measure=1,
                position=0.0,
                lane_id="11",
                note_id="01",
                chart_time_sec=0.5,
                audio_time_sec=0.5,
            ),
        )
    )
    events = tuple(
        _event_from_native(
            NativeEvent(
                time_sec,
                "midi_42",
                21,
                42,
                {"upstream_8hit_group_id": group},
                0.5,
                80,
            )
        )
        for time_sec, group in [(0.4499996, "hihat"), (0.4499997, "hihat"), (0.45, "sticks")]
    )

    score = score_panel_song(
        baseline, events, mapping, build_candidate_descriptor(load_model_config())
    )

    assert score == {
        "reference_event_count": 1,
        "baseline": {"tp": 1, "fp": 0, "fn": 0, "f1": Decimal(1), "unmapped": 0},
        "candidate": {"tp": 1, "fp": 1, "fn": 0, "f1": Decimal("0.666667"), "unmapped": 1},
        "f1_delta": Decimal("-0.333333"),
    }
    with pytest.raises(ValueError, match="candidate backend"):
        score_panel_song(baseline, events, mapping, baseline.prediction.descriptor)


def test_panel_aggregates_macro_f1_separately_from_pooled_counts() -> None:
    from experiments.oaf_tf215.panel_metrics import aggregate_panel_scores

    totals = aggregate_panel_scores(
        [
            {
                "reference_event_count": 1,
                "baseline": {"tp": 1, "fp": 0, "fn": 0, "f1": Decimal(1), "unmapped": 0},
                "candidate": {"tp": 0, "fp": 0, "fn": 1, "f1": Decimal(0), "unmapped": 2},
                "f1_delta": Decimal(-1),
            },
            {
                "reference_event_count": 10,
                "baseline": {"tp": 5, "fp": 0, "fn": 5, "f1": Decimal("0.666667"), "unmapped": 1},
                "candidate": {"tp": 7, "fp": 3, "fn": 3, "f1": Decimal("0.7"), "unmapped": 0},
                "f1_delta": Decimal("0.033333"),
            },
        ]
    )

    assert totals == {
        "scored_song_count": 2,
        "baseline": {
            "tp": 6,
            "fp": 0,
            "fn": 5,
            "unmapped": 1,
            "macro_f1": Decimal("0.833333"),
            "micro_f1": Decimal("0.705882"),
        },
        "candidate": {
            "tp": 7,
            "fp": 3,
            "fn": 4,
            "unmapped": 2,
            "macro_f1": Decimal("0.35"),
            "micro_f1": Decimal("0.666667"),
        },
        "macro_f1_delta": Decimal("-0.483333"),
        "micro_f1_delta": Decimal("-0.039216"),
        "micro_f1_nonregression": False,
    }


def test_panel_gate_detects_a_regression_below_display_precision() -> None:
    from experiments.oaf_tf215.panel_metrics import aggregate_panel_scores

    count = 10**20
    totals = aggregate_panel_scores(
        [
            {
                "reference_event_count": count,
                "baseline": {"tp": count, "fp": 0, "fn": 0, "f1": Decimal(1), "unmapped": 0},
                "candidate": {"tp": count - 1, "fp": 0, "fn": 1, "f1": Decimal(1), "unmapped": 0},
                "f1_delta": Decimal(0),
            },
        ]
    )

    assert totals["baseline"]["micro_f1"] == totals["candidate"]["micro_f1"] == Decimal(1)
    assert totals["micro_f1_nonregression"] is False
