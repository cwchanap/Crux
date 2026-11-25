"""Score the panel's frozen and candidate predictions against the same reference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from typing import Literal, TypedDict

from experiments.oaf_tf215.selected_metrics import candidate_native_events
from src.benchmark.backend_identity import (
    OAF_TF215_BACKEND_ID,
    BackendDescriptor,
    JsonValue,
    quantize_six,
)
from src.benchmark.backends.base import NativePrediction
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.models import ScoreSummary
from src.benchmark.oaf_smoke_oracle import SmokeOracleEvent
from src.benchmark.prediction_artifact import PredictionArtifact
from src.benchmark.reference_set import ReferenceMappingResult
from src.benchmark.scorer_input import (
    _benchmark_event,
    prediction_to_benchmark_events,
    reference_to_benchmark_events,
)
from src.benchmark.scoring import score_events
from src.benchmark.taxonomy import OAF_PREDICTION_MAP

RAW_TOLERANCE_SEC = 0.05


class SongSideScore(TypedDict):
    tp: int
    fp: int
    fn: int
    f1: Decimal | None
    unmapped: int


class PanelSongScore(TypedDict):
    reference_event_count: int
    baseline: SongSideScore
    candidate: SongSideScore
    f1_delta: Decimal | None


def _f1(tp: int, fp: int, fn: int) -> Fraction | None:
    denominator = 2 * tp + fp + fn
    return Fraction(2 * tp, denominator) if denominator else None


def _display(value: Fraction | None) -> Decimal | None:
    return None if value is None else quantize_six(float(value))


def _delta(baseline: Fraction | None, candidate: Fraction | None) -> Decimal | None:
    return None if baseline is None or candidate is None else _display(candidate - baseline)


def _song_score(summary: ScoreSummary, unmapped: int) -> tuple[SongSideScore, Fraction | None]:
    tp, fp, fn = summary.true_positives, summary.false_positives, summary.false_negatives
    f1 = _f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _display(f1), "unmapped": unmapped}, f1


def score_panel_song(
    baseline: PredictionArtifact,
    events: tuple[SmokeOracleEvent, ...],
    mapping: ReferenceMappingResult,
    descriptor: BackendDescriptor,
) -> PanelSongScore:
    """Use the frozen OaF taxonomy and raw 50 ms production scoring policy."""
    if descriptor.payload.get("backend_id") != OAF_TF215_BACKEND_ID:
        raise ValueError("panel scoring requires the TF2.15 candidate backend")

    audio = baseline.prediction.audio
    native = tuple(
        replace(event, time_sec=float(quantize_six(event.time_sec)))
        for event in candidate_native_events(events)
    )
    mapped, diagnostics = map_oaf_prediction(
        NativePrediction(audio, descriptor, native),
        replace(OAF_PREDICTION_MAP, backend_id=OAF_TF215_BACKEND_ID),
    )
    # Keep multiplicity after rounding, including candidates that share an artifact key.
    candidate_predictions = [
        _benchmark_event(audio.source_audio_id, audio.input_view_id, event)
        for event in mapped.events
        if event.mapping_status == "mapped"
    ]
    baseline_predictions = list(prediction_to_benchmark_events(baseline))
    reference = list(reference_to_benchmark_events(audio.source_audio_id, mapping.common_events))
    baseline_summary = score_events(reference, baseline_predictions, RAW_TOLERANCE_SEC).summary
    candidate_summary = score_events(reference, candidate_predictions, RAW_TOLERANCE_SEC).summary
    baseline_score, baseline_f1 = _song_score(
        baseline_summary, baseline.event_count - len(baseline_predictions)
    )
    candidate_score, candidate_f1 = _song_score(
        candidate_summary, sum(diagnostics.unmapped.values())
    )
    return {
        "reference_event_count": len(reference),
        "baseline": baseline_score,
        "candidate": candidate_score,
        "f1_delta": _delta(baseline_f1, candidate_f1),
    }


def aggregate_panel_scores(rows: Iterable[PanelSongScore]) -> dict[str, JsonValue]:
    """Aggregate paired rows and apply the pooled comparison on exact fractions."""
    rows = tuple(rows)
    output: dict[str, JsonValue] = {"scored_song_count": len(rows)}
    macro_f1: dict[str, Fraction | None] = {}
    micro_f1: dict[str, Fraction | None] = {}
    sides: tuple[Literal["baseline", "candidate"], ...] = ("baseline", "candidate")
    for side in sides:
        scores = [row[side] for row in rows]
        tp = sum(score["tp"] for score in scores)
        fp = sum(score["fp"] for score in scores)
        fn = sum(score["fn"] for score in scores)
        ratios = [
            value
            for score in scores
            if (value := _f1(score["tp"], score["fp"], score["fn"])) is not None
        ]
        macro_f1[side] = sum(ratios, start=Fraction(0)) / len(ratios) if ratios else None
        micro_f1[side] = _f1(tp, fp, fn)
        output[side] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "unmapped": sum(score["unmapped"] for score in scores),
            "macro_f1": _display(macro_f1[side]),
            "micro_f1": _display(micro_f1[side]),
        }
    baseline, candidate = micro_f1["baseline"], micro_f1["candidate"]
    return {
        **output,
        "macro_f1_delta": _delta(macro_f1["baseline"], macro_f1["candidate"]),
        "micro_f1_delta": _delta(baseline, candidate),
        "micro_f1_nonregression": (
            None if baseline is None or candidate is None else candidate >= baseline
        ),
    }
