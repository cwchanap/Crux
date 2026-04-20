#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from src.app.transcriber import DrumTranscriber
from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.mapping import DEFAULT_MIDI_NOTE_MAP, map_dtx_events
from src.benchmark.models import BenchmarkEvent, ScoreSummary
from src.benchmark.scoring import score_events_with_alignment
from src.benchmark.timing import dtx_events_to_timed_events

DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/mapping-calibration")
DEFAULT_TOLERANCE_MS = 50
DEFAULT_TOP_N = (9, 12, 18, 24, 27)
DEFAULT_PER_CLASS_K = (1, 2, 3, 4)
INFERENCE_CONTEXT_FRAMES = 128
INFERENCE_CENTRAL_FRAMES = 768


@dataclass(frozen=True)
class ChartCalibrationData:
    chart_id: str
    ground_truth: list[BenchmarkEvent]
    predictions_by_bin: dict[int, list[BenchmarkEvent]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off calibration of TF2 E-GMD output bins against a prepared benchmark corpus.",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        required=True,
        help="Prepared corpus charts directory.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Prepared corpus audio directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Calibration artifact directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=DEFAULT_TOLERANCE_MS,
        help=f"Benchmark tolerance in milliseconds. Default: {DEFAULT_TOLERANCE_MS}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DrumTranscriber.MODEL_ONSET_THRESHOLD,
        help="Onset peak threshold to use for raw TF2 bin extraction.",
    )
    parser.add_argument(
        "--min-gap-frames",
        type=int,
        default=DrumTranscriber.MODEL_ONSET_MIN_GAP_FRAMES,
        help="Refractory gap between onset peaks for one TF2 output bin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    charts_dir = args.charts_dir.resolve()
    audio_dir = args.audio_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = DrumTranscriber()
    if transcriber.model is None:
        raise RuntimeError("TF2 model did not load; calibration requires a working model")

    chart_data = load_prepared_corpus(
        charts_dir=charts_dir,
        audio_dir=audio_dir,
        transcriber=transcriber,
        threshold=args.threshold,
        min_gap_frames=args.min_gap_frames,
    )
    print(f"Loaded {len(chart_data)} prepared chart(s)", flush=True)
    tolerance_sec = args.tolerance_ms / 1000.0

    pair_scores = score_bin_class_pairs(chart_data, tolerance_sec)
    print("Finished pairwise bin/class ranking", flush=True)
    class_order = corpus_class_order(chart_data)
    candidate_mappings = build_candidate_mappings(pair_scores, class_order)
    print(
        f"Built {len(candidate_mappings)} candidate mapping strategy/strategies",
        flush=True,
    )

    baseline_name = "current_production_overlap"
    baseline_mapping = current_production_mapping()
    baseline_result = evaluate_mapping(chart_data, baseline_mapping, tolerance_sec)
    print("Evaluated current production overlap baseline", flush=True)

    candidate_results: list[dict[str, Any]] = []
    for strategy_name, mapping in candidate_mappings.items():
        result = evaluate_mapping(chart_data, mapping, tolerance_sec)
        result["strategy_name"] = strategy_name
        result["mapping_size"] = len(mapping)
        candidate_results.append(result)
        print(
            f"Evaluated candidate {strategy_name} ({len(mapping)} assigned bins)",
            flush=True,
        )

    candidate_results.sort(key=lambda item: strategy_sort_key(item["aligned"]), reverse=True)
    best_candidate = candidate_results[0] if candidate_results else None

    summary = {
        "tolerance_ms": args.tolerance_ms,
        "threshold": args.threshold,
        "min_gap_frames": args.min_gap_frames,
        "charts_dir": str(charts_dir),
        "audio_dir": str(audio_dir),
        "charts": [chart.chart_id for chart in chart_data],
        "ground_truth_class_counts": corpus_ground_truth_class_counts(chart_data),
        "baseline": {
            "strategy_name": baseline_name,
            "mapping_size": len(baseline_mapping),
            **baseline_result,
        },
        "best_candidate": best_candidate,
        "all_candidates": candidate_results,
        "top_bins_by_class": top_bins_by_class(pair_scores),
    }
    best_candidate_mapping = None
    best_candidate_per_chart = {}
    if best_candidate is not None:
        best_candidate_mapping = {
            "strategy_name": best_candidate["strategy_name"],
            "mapping": serialize_mapping(
                candidate_mappings[best_candidate["strategy_name"]],
                pair_scores,
            ),
        }
        best_candidate_per_chart = best_candidate["per_chart"]

    best_mapping = {
        "baseline_strategy_name": baseline_name,
        "baseline_mapping": serialize_mapping(baseline_mapping, pair_scores),
        "best_candidate": best_candidate_mapping,
    }
    per_chart = {
        chart.chart_id: {
            "ground_truth_class_counts": class_counts(chart.ground_truth),
            "predicted_bin_counts": {
                str(bin_index): len(events)
                for bin_index, events in sorted(chart.predictions_by_bin.items())
            },
            "baseline": baseline_result["per_chart"][chart.chart_id],
            "best_candidate": best_candidate_per_chart.get(chart.chart_id),
        }
        for chart in chart_data
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "best_mapping.json", best_mapping)
    write_json(output_dir / "per_chart.json", per_chart)

    baseline_f1 = baseline_result["aligned"]["f1"]
    if best_candidate is None:
        print("No candidate mappings produced.", flush=True)
        return

    best_candidate_data: dict[str, Any] = best_candidate
    best_f1 = best_candidate_data["aligned"]["f1"]  # pylint: disable=unsubscriptable-object
    print(f"Baseline aligned F1 @{args.tolerance_ms}ms: {baseline_f1:.4f}", flush=True)
    print(
        f"Best candidate "  # pylint: disable=unsubscriptable-object
        f"{best_candidate_data['strategy_name']} aligned F1 @{args.tolerance_ms}ms: "  # pylint: disable=unsubscriptable-object
        f"{best_f1:.4f}",
        flush=True,
    )
    print(f"Artifacts written to {output_dir}", flush=True)


def load_prepared_corpus(
    charts_dir: Path,
    audio_dir: Path,
    transcriber: DrumTranscriber,
    threshold: float,
    min_gap_frames: int,
) -> list[ChartCalibrationData]:
    chart_items: list[ChartCalibrationData] = []
    for dtx_path in sorted(charts_dir.glob("*.dtx")):
        audio_path = find_audio(audio_dir, dtx_path.stem)
        ground_truth = load_ground_truth(dtx_path)
        predictions_by_bin = extract_predictions_by_bin(
            transcriber=transcriber,
            audio_path=audio_path,
            chart_id=dtx_path.stem,
            threshold=threshold,
            min_gap_frames=min_gap_frames,
        )
        chart_items.append(
            ChartCalibrationData(
                chart_id=dtx_path.stem,
                ground_truth=ground_truth,
                predictions_by_bin=predictions_by_bin,
            )
        )
        print(
            f"Prepared {dtx_path.stem}: "
            f"{len(ground_truth)} ground-truth events, "
            f"{sum(len(events) for events in predictions_by_bin.values())} raw bin onsets",
            flush=True,
        )
    if not chart_items:
        raise FileNotFoundError(f"no .dtx charts found under {charts_dir}")
    return chart_items


def find_audio(audio_dir: Path, chart_id: str) -> Path:
    for suffix in (".wav", ".mp3", ".m4a", ".flac"):
        path = audio_dir / f"{chart_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing audio for chart_id {chart_id}")


def load_ground_truth(dtx_path: Path) -> list[BenchmarkEvent]:
    chart = parse_dtx_file(dtx_path, chart_id=dtx_path.stem)
    timed_events = dtx_events_to_timed_events(chart)
    ground_truth, diagnostics = map_dtx_events(timed_events)
    if diagnostics.unmapped:
        raise ValueError(f"unmapped DTX lanes in {dtx_path}: {diagnostics.unmapped}")
    return ground_truth


def extract_predictions_by_bin(
    transcriber: DrumTranscriber,
    audio_path: Path,
    chart_id: str,
    threshold: float,
    min_gap_frames: int,
) -> dict[int, list[BenchmarkEvent]]:
    audio, _ = librosa.load(str(audio_path), sr=transcriber.sample_rate, mono=True)
    spec = transcriber._compute_spectrogram_for_model(audio, transcriber.sample_rate)
    onset_probs, velocity_values = run_model_in_chunks(transcriber, spec, chart_id)
    predictions_by_bin: dict[int, list[BenchmarkEvent]] = {}

    for bin_index in range(onset_probs.shape[1]):
        onset_indices = transcriber._find_onset_peaks(
            onset_probs[:, bin_index],
            threshold=threshold,
            min_gap_frames=min_gap_frames,
        )
        if len(onset_indices) == 0:
            continue
        events: list[BenchmarkEvent] = []
        for frame_index in onset_indices:
            velocity = int(np.clip(velocity_values[frame_index, bin_index] * 127, 1, 127))
            events.append(
                BenchmarkEvent(
                    chart_id=chart_id,
                    time_sec=(frame_index * transcriber.hop_length) / transcriber.MODEL_SAMPLE_RATE,
                    canonical_class=str(bin_index),
                    source="prediction",
                    metadata={
                        "model_bin": bin_index,
                        "velocity": velocity,
                    },
                )
            )
        predictions_by_bin[bin_index] = events

    return predictions_by_bin


def run_model_in_chunks(
    transcriber: DrumTranscriber,
    spec: np.ndarray,
    chart_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = spec.shape[0]
    total_chunks = max(1, int(np.ceil(total_frames / INFERENCE_CENTRAL_FRAMES)))
    onset_probs: np.ndarray | None = None
    velocity_values: np.ndarray | None = None

    for chunk_index, central_start in enumerate(range(0, total_frames, INFERENCE_CENTRAL_FRAMES)):
        input_start = max(0, central_start - INFERENCE_CONTEXT_FRAMES)
        input_end = min(
            total_frames,
            central_start + INFERENCE_CENTRAL_FRAMES + INFERENCE_CONTEXT_FRAMES,
        )
        chunk_input = spec[input_start:input_end][np.newaxis, :, :, np.newaxis]
        outputs = transcriber.model(chunk_input, training=False)
        chunk_onset_probs = outputs["onset_probs"].numpy()[0]
        chunk_velocity_values = outputs["velocity_values"].numpy()[0]

        if onset_probs is None or velocity_values is None:
            onset_probs = np.zeros((total_frames, chunk_onset_probs.shape[1]), dtype=np.float32)
            velocity_values = np.zeros(
                (total_frames, chunk_velocity_values.shape[1]),
                dtype=np.float32,
            )

        local_take_start = central_start - input_start
        take_length = min(INFERENCE_CENTRAL_FRAMES, total_frames - central_start)
        local_take_end = local_take_start + take_length

        onset_probs[central_start : central_start + take_length] = chunk_onset_probs[
            local_take_start:local_take_end
        ]
        velocity_values[central_start : central_start + take_length] = chunk_velocity_values[
            local_take_start:local_take_end
        ]
        print(
            f"Inference {chart_id}: chunk {chunk_index + 1}/{total_chunks}",
            flush=True,
        )

    if onset_probs is None or velocity_values is None:
        raise RuntimeError(f"model inference produced no outputs for {chart_id}")

    return onset_probs, velocity_values


def score_bin_class_pairs(
    chart_data: list[ChartCalibrationData],
    tolerance_sec: float,
) -> dict[str, dict[int, dict[str, Any]]]:
    classes = sorted(
        {event.canonical_class for chart in chart_data for event in chart.ground_truth},
    )
    bins = sorted({bin_index for chart in chart_data for bin_index in chart.predictions_by_bin})
    pair_scores: dict[str, dict[int, dict[str, Any]]] = {}
    for canonical_class in classes:
        pair_scores[canonical_class] = {}
        for bin_index in bins:
            summaries = []
            for chart in chart_data:
                ground_truth = [
                    event
                    for event in chart.ground_truth
                    if event.canonical_class == canonical_class
                ]
                predictions = chart.predictions_by_bin.get(bin_index, [])
                summaries.append(fast_pair_summary(ground_truth, predictions, tolerance_sec))
            pair_scores[canonical_class][bin_index] = aggregate_summaries(summaries)
    return pair_scores


def fast_pair_summary(
    ground_truth: list[BenchmarkEvent],
    predictions: list[BenchmarkEvent],
    tolerance_sec: float,
) -> ScoreSummary:
    gt_times = [event.time_sec for event in ground_truth]
    pred_times = [event.time_sec for event in predictions]
    gt_index = 0
    pred_index = 0
    true_positives = 0

    while gt_index < len(gt_times) and pred_index < len(pred_times):
        delta = pred_times[pred_index] - gt_times[gt_index]
        if abs(delta) <= tolerance_sec:
            true_positives += 1
            gt_index += 1
            pred_index += 1
        elif delta < -tolerance_sec:
            pred_index += 1
        else:
            gt_index += 1

    return ScoreSummary(
        true_positives=true_positives,
        false_positives=len(pred_times) - true_positives,
        false_negatives=len(gt_times) - true_positives,
    )


def corpus_class_order(chart_data: list[ChartCalibrationData]) -> list[str]:
    counts = Counter()
    for chart in chart_data:
        counts.update(class_counts(chart.ground_truth))
    return [
        class_name for class_name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_candidate_mappings(
    pair_scores: dict[str, dict[int, dict[str, Any]]],
    class_order: list[str],
) -> dict[str, dict[int, list[str]]]:
    candidates: dict[str, dict[int, list[str]]] = {}
    best_class_per_bin = ranked_best_class_per_bin(pair_scores)

    for top_n in DEFAULT_TOP_N:
        mapping: dict[int, list[str]] = {}
        for bin_index, class_name, metrics in best_class_per_bin[:top_n]:
            if metrics["true_positives"] == 0 or metrics["f1"] <= 0.0:
                continue
            mapping[bin_index] = [class_name]
        candidates[f"global_top_{top_n}"] = mapping

    for per_class_k in DEFAULT_PER_CLASS_K:
        used_bins: set[int] = set()
        mapping = {}
        for class_name in class_order:
            ranked_bins = sorted(
                pair_scores[class_name].items(),
                key=lambda item: strategy_sort_key(item[1]),
                reverse=True,
            )
            selected = 0
            for bin_index, metrics in ranked_bins:
                if bin_index in used_bins:
                    continue
                if metrics["true_positives"] == 0 or metrics["f1"] <= 0.0:
                    continue
                mapping[bin_index] = [class_name]
                used_bins.add(bin_index)
                selected += 1
                if selected >= per_class_k:
                    break
        candidates[f"per_class_top_{per_class_k}"] = mapping

    return candidates


def ranked_best_class_per_bin(
    pair_scores: dict[str, dict[int, dict[str, Any]]],
) -> list[tuple[int, str, dict[str, Any]]]:
    bins = sorted({bin_index for scores in pair_scores.values() for bin_index in scores})
    best_items = []
    for bin_index in bins:
        class_name, metrics = max(
            (
                (candidate_class, class_scores[bin_index])
                for candidate_class, class_scores in pair_scores.items()
                if bin_index in class_scores
            ),
            key=lambda item: strategy_sort_key(item[1]),
        )
        best_items.append((bin_index, class_name, metrics))
    best_items.sort(key=lambda item: strategy_sort_key(item[2]), reverse=True)
    return best_items


def current_production_mapping() -> dict[int, list[str]]:
    bin_to_classes: dict[int, list[str]] = defaultdict(list)
    drum_key_ranges = {
        36: range(35, 37),
        38: range(37, 41),
        42: range(41, 45),
        46: range(44, 46),
        49: range(45, 52),
        51: range(50, 53),
        41: range(52, 55),
        47: range(55, 58),
        50: range(58, 60),
    }
    for midi_note, key_range in drum_key_ranges.items():
        canonical_class = DEFAULT_MIDI_NOTE_MAP[midi_note]
        for bin_index in key_range:
            bin_to_classes[bin_index].append(canonical_class)
    return dict(sorted(bin_to_classes.items()))


def evaluate_mapping(
    chart_data: list[ChartCalibrationData],
    mapping: dict[int, list[str]],
    tolerance_sec: float,
) -> dict[str, Any]:
    raw_summaries: list[ScoreSummary] = []
    aligned_summaries: list[ScoreSummary] = []
    per_chart: dict[str, Any] = {}

    for chart in chart_data:
        predictions = assigned_predictions(chart.predictions_by_bin, mapping)
        result = score_events_with_alignment(chart.ground_truth, predictions, tolerance_sec)
        raw_summaries.append(result.raw.summary)
        aligned_summaries.append(result.aligned.summary)
        per_chart[chart.chart_id] = {
            "prediction_count": len(predictions),
            "prediction_class_counts": class_counts(predictions),
            "raw": score_summary_dict(result.raw.summary),
            "aligned": score_summary_dict(result.aligned.summary),
        }

    return {
        "raw": aggregate_summaries(raw_summaries),
        "aligned": aggregate_summaries(aligned_summaries),
        "per_chart": per_chart,
    }


def assigned_predictions(
    predictions_by_bin: dict[int, list[BenchmarkEvent]],
    mapping: dict[int, list[str]],
) -> list[BenchmarkEvent]:
    predictions: list[BenchmarkEvent] = []
    for bin_index, assigned_classes in mapping.items():
        for canonical_class in assigned_classes:
            for event in predictions_by_bin.get(bin_index, []):
                predictions.append(
                    replace(
                        event,
                        canonical_class=canonical_class,
                        metadata={
                            **event.metadata,
                            "assigned_class": canonical_class,
                        },
                    )
                )
    return sorted(predictions)


def aggregate_summaries(summaries: list[ScoreSummary]) -> dict[str, Any]:
    true_positives = sum(summary.true_positives for summary in summaries)
    false_positives = sum(summary.false_positives for summary in summaries)
    false_negatives = sum(summary.false_negatives for summary in summaries)
    offsets = [summary.offset_sec for summary in summaries]
    summary = ScoreSummary(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        median_abs_error_sec=None,
        p95_abs_error_sec=None,
        offset_sec=(float(np.mean(offsets)) if offsets else 0.0),
    )
    result = score_summary_dict(summary)
    result["offsets_sec"] = offsets
    return result


def score_summary_dict(summary: ScoreSummary) -> dict[str, Any]:
    return {
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "false_negatives": summary.false_negatives,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "offset_sec": summary.offset_sec,
    }


def strategy_sort_key(summary: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        float(summary["f1"]),
        int(summary["true_positives"]),
        float(summary["precision"]),
        float(summary["recall"]),
    )


def class_counts(events: list[BenchmarkEvent]) -> dict[str, int]:
    counts = Counter(event.canonical_class for event in events)
    return {class_name: counts[class_name] for class_name in sorted(counts)}


def corpus_ground_truth_class_counts(chart_data: list[ChartCalibrationData]) -> dict[str, int]:
    counts = Counter()
    for chart in chart_data:
        counts.update(class_counts(chart.ground_truth))
    return {class_name: counts[class_name] for class_name in sorted(counts)}


def top_bins_by_class(
    pair_scores: dict[str, dict[int, dict[str, Any]]],
    top_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    top_bins: dict[str, list[dict[str, Any]]] = {}
    for class_name, class_scores in sorted(pair_scores.items()):
        ranked = sorted(
            class_scores.items(),
            key=lambda item: strategy_sort_key(item[1]),
            reverse=True,
        )
        top_bins[class_name] = [
            {"model_bin": bin_index, **metrics}
            for bin_index, metrics in ranked[:top_n]
            if metrics["true_positives"] > 0
        ]
    return top_bins


def serialize_mapping(
    mapping: dict[int, list[str]],
    pair_scores: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    by_bin = {}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bin_index, classes in sorted(mapping.items()):
        by_bin[str(bin_index)] = []
        for canonical_class in classes:
            metrics = pair_scores.get(canonical_class, {}).get(
                bin_index,
                {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "offset_sec": 0.0,
                    "offsets_sec": [],
                },
            )
            entry = {
                "model_bin": bin_index,
                "canonical_class": canonical_class,
                "pair_score": metrics,
            }
            by_bin[str(bin_index)].append(entry)
            by_class[canonical_class].append(entry)
    return {
        "by_bin": by_bin,
        "by_class": {class_name: entries for class_name, entries in sorted(by_class.items())},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
