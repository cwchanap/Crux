from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from math import ceil
from statistics import median

import numpy as np
from scipy.signal import correlate, correlation_lags

from src.benchmark.models import BenchmarkEvent, MatchResult, ScoreSummary

ALIGNMENT_CANDIDATE_LIMIT = 9
ALIGNMENT_BIN_SIZE_MAX_SEC = 0.01
ALIGNMENT_BIN_SIZE_MIN_SEC = 0.001


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
        dataclasses.replace(
            event,
            time_sec=event.time_sec + offset_sec,
            metadata=dict(event.metadata),
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
    matches: list[MatchResult] = []
    unmatched_ground_truth: list[BenchmarkEvent] = []
    unmatched_predictions: list[BenchmarkEvent] = []
    gt_index = 0
    pred_index = 0

    while gt_index < len(ground_truth) and pred_index < len(predictions):
        gt_event = ground_truth[gt_index]
        pred_event = predictions[pred_index]
        error = pred_event.time_sec - gt_event.time_sec

        if error < -tolerance_sec:
            unmatched_predictions.append(pred_event)
            pred_index += 1
            continue
        if error > tolerance_sec:
            unmatched_ground_truth.append(gt_event)
            gt_index += 1
            continue

        matches.append(MatchResult(gt_event, pred_event, error))
        gt_index += 1
        pred_index += 1

    unmatched_ground_truth.extend(ground_truth[gt_index:])
    unmatched_predictions.extend(predictions[pred_index:])
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
    if not ground_truth or not predictions:
        return 0.0

    shared_classes = sorted(
        {event.canonical_class for event in ground_truth}
        & {event.canonical_class for event in predictions}
    )
    if not shared_classes:
        return 0.0

    bin_size = _alignment_bin_size(tolerance_sec)
    max_time = max(
        max(event.time_sec for event in ground_truth),
        max(event.time_sec for event in predictions),
    )
    num_bins = max(1, int(round(max_time / bin_size)) + 1)
    total_correlation = np.zeros((num_bins * 2) - 1, dtype=np.float64)
    lags = correlation_lags(num_bins, num_bins, mode="full")

    for canonical_class in shared_classes:
        gt_times = [
            event.time_sec for event in ground_truth if event.canonical_class == canonical_class
        ]
        pred_times = [
            event.time_sec for event in predictions if event.canonical_class == canonical_class
        ]
        if not gt_times or not pred_times:
            continue
        gt_histogram = _alignment_histogram(gt_times, num_bins, bin_size)
        pred_histogram = _alignment_histogram(pred_times, num_bins, bin_size)
        total_correlation += correlate(gt_histogram, pred_histogram, mode="full", method="fft")

    candidate_offsets = _alignment_candidates(total_correlation, lags, bin_size)
    best_offset, best_result, best_score = _best_alignment_candidate(
        ground_truth,
        predictions,
        tolerance_sec,
        candidate_offsets,
    )
    if best_result is None or not best_result.matches:
        return best_offset

    refined_offset = median(best_offset - match.timing_error_sec for match in best_result.matches)
    refined_result = score_events(
        ground_truth, predictions, tolerance_sec, offset_sec=refined_offset
    )
    refined_score = _alignment_score(refined_result.summary, refined_offset)
    if refined_score > best_score:
        return refined_offset
    return best_offset


def _alignment_bin_size(tolerance_sec: float) -> float:
    if tolerance_sec <= 0:
        return ALIGNMENT_BIN_SIZE_MIN_SEC
    return min(
        ALIGNMENT_BIN_SIZE_MAX_SEC,
        max(ALIGNMENT_BIN_SIZE_MIN_SEC, tolerance_sec / 5),
    )


def _alignment_histogram(times: list[float], num_bins: int, bin_size: float) -> np.ndarray:
    times_arr = np.asarray(times)
    if times_arr.size > 0 and np.any(times_arr < 0):
        raise ValueError(
            f"_alignment_histogram received negative times (min={times_arr.min():.6f}). "
            "All event times must be non-negative."
        )
    bins = np.rint(times_arr / bin_size).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)
    return np.bincount(bins, minlength=num_bins).astype(np.float64)


def _alignment_candidates(
    total_correlation: np.ndarray,
    lags: np.ndarray,
    bin_size: float,
) -> list[float]:
    if total_correlation.size == 0:
        return [0.0]

    ranked_indices = sorted(
        range(len(lags)),
        key=lambda index: (
            -total_correlation[index],
            abs(int(lags[index])),
            int(lags[index]),
        ),
    )

    candidates: list[float] = []
    seen: set[float] = set()
    for index in ranked_indices:
        offset = round(float(lags[index] * bin_size), 12)
        if offset in seen:
            continue
        seen.add(offset)
        candidates.append(offset)
        if len(candidates) >= ALIGNMENT_CANDIDATE_LIMIT:
            break

    if 0.0 not in seen:
        candidates.append(0.0)
    return candidates


def _best_alignment_candidate(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
    candidate_offsets: list[float],
) -> tuple[float, ScoreResult | None, tuple[int, float, float]]:
    best_offset = 0.0
    best_result: ScoreResult | None = None
    best_score = (-1, float("-inf"), float("-inf"))
    for offset in candidate_offsets:
        result = score_events(ground_truth, predictions, tolerance_sec, offset_sec=offset)
        score = _alignment_score(result.summary, offset)
        if score > best_score:
            best_offset = offset
            best_result = result
            best_score = score
    return best_offset, best_result, best_score


def _alignment_score(summary: ScoreSummary, offset: float) -> tuple[int, float, float]:
    median_error = summary.median_abs_error_sec
    return (
        summary.true_positives,
        -median_error if median_error is not None else 0.0,
        -round(abs(offset), 12),
    )
