from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median

from src.benchmark.models import BenchmarkEvent, MatchResult, ScoreSummary


@dataclass(frozen=True)
class ScoreResult:
    summary: ScoreSummary
    matches: list[MatchResult]
    unmatched_ground_truth: list[BenchmarkEvent]
    unmatched_predictions: list[BenchmarkEvent]


def score_events(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
    offset_sec: float = 0.0,
) -> ScoreResult:
    adjusted_predictions = [
        BenchmarkEvent(
            event.chart_id,
            event.time_sec + offset_sec,
            event.canonical_class,
            event.source,
            event.metadata,
        )
        for event in predictions
    ]

    classes = sorted({event.canonical_class for event in ground_truth + adjusted_predictions})
    matches: list[MatchResult] = []
    unmatched_ground_truth: list[BenchmarkEvent] = []
    unmatched_predictions: list[BenchmarkEvent] = []

    for canonical_class in classes:
        gt_class = sorted(
            [event for event in ground_truth if event.canonical_class == canonical_class],
            key=lambda event: event.time_sec,
        )
        pred_class = sorted(
            [event for event in adjusted_predictions if event.canonical_class == canonical_class],
            key=lambda event: event.time_sec,
        )
        class_matches, class_unmatched_gt, class_unmatched_pred = _match_class(
            gt_class, pred_class, tolerance_sec
        )
        matches.extend(class_matches)
        unmatched_ground_truth.extend(class_unmatched_gt)
        unmatched_predictions.extend(class_unmatched_pred)

    errors = sorted(match.absolute_error_sec for match in matches)
    summary = ScoreSummary(
        true_positives=len(matches),
        false_positives=len(unmatched_predictions),
        false_negatives=len(unmatched_ground_truth),
        median_abs_error_sec=median(errors) if errors else None,
        p95_abs_error_sec=_percentile(errors, 0.95) if errors else None,
        offset_sec=offset_sec,
    )
    return ScoreResult(summary, matches, unmatched_ground_truth, unmatched_predictions)


def _match_class(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
) -> tuple[list[MatchResult], list[BenchmarkEvent], list[BenchmarkEvent]]:
    available_predictions = set(range(len(predictions)))
    matches: list[MatchResult] = []
    unmatched_ground_truth: list[BenchmarkEvent] = []

    for gt_event in ground_truth:
        best_index = None
        best_error = None
        for pred_index in sorted(available_predictions):
            error = predictions[pred_index].time_sec - gt_event.time_sec
            if abs(error) > tolerance_sec:
                continue
            if best_error is None or (abs(error), pred_index) < (abs(best_error), best_index):
                best_index = pred_index
                best_error = error
        if best_index is None or best_error is None:
            unmatched_ground_truth.append(gt_event)
            continue
        available_predictions.remove(best_index)
        matches.append(MatchResult(gt_event, predictions[best_index], best_error))

    unmatched_predictions = [predictions[index] for index in sorted(available_predictions)]
    return matches, unmatched_ground_truth, unmatched_predictions


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    index = min(len(values) - 1, ceil((len(values) - 1) * percentile))
    return values[index]


@dataclass(frozen=True)
class AlignedScoreResult:
    raw: ScoreResult
    aligned: ScoreResult


def score_events_with_alignment(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
) -> AlignedScoreResult:
    raw = score_events(ground_truth, predictions, tolerance_sec)
    offset = _choose_global_offset(ground_truth, predictions, tolerance_sec)
    aligned = score_events(ground_truth, predictions, tolerance_sec, offset_sec=offset)
    return AlignedScoreResult(raw=raw, aligned=aligned)


def _choose_global_offset(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
) -> float:
    candidates = {0.0}
    for gt_event in ground_truth:
        for prediction in predictions:
            if gt_event.canonical_class != prediction.canonical_class:
                continue
            offset = gt_event.time_sec - prediction.time_sec
            candidates.add(offset)

    best_offset = 0.0
    best_score = (-1, float("-inf"), float("-inf"))
    for offset in sorted(candidates):
        result = score_events(ground_truth, predictions, tolerance_sec, offset_sec=offset)
        median_error = result.summary.median_abs_error_sec
        offset_magnitude = round(abs(offset), 12)
        score = (
            result.summary.true_positives,
            -median_error if median_error is not None else 0.0,
            -offset_magnitude,
        )
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset
