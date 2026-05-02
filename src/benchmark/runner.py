from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from src.benchmark.corpus import validate_score_midi_corpus
from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.mapping import map_dtx_events, map_midi_events
from src.benchmark.midi_io import parse_prediction_midi, write_reference_midi
from src.benchmark.prepare import CHART_SUFFIXES
from src.benchmark.reports import ChartReport, write_reports
from src.benchmark.scoring import score_events, score_events_with_alignment
from src.benchmark.timing import dtx_events_to_timed_events

logger = logging.getLogger(__name__)


def run_score_midi(
    charts_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    tolerance_ms: list[int],
    align: bool = True,
    export_reference_midi: bool = False,
) -> list[ChartReport]:
    validation = validate_score_midi_corpus(charts_dir, predictions_dir)
    if validation.errors:
        error_list = "; ".join(validation.errors)
        raise ValueError(f"corpus validation failed: {error_list}")

    reports: list[ChartReport] = []
    failed_charts: list[str] = []
    for item in validation.valid_items:
        try:
            chart = parse_dtx_file(item.dtx_path, chart_id=item.chart_id)
            ground_truth, gt_diag = map_dtx_events(dtx_events_to_timed_events(chart))
            predictions, pred_diag = map_midi_events(
                parse_prediction_midi(item.prediction_midi_path, item.chart_id)
            )

            if gt_diag.unmapped:
                logger.warning(
                    "chart %r has %d unmapped ground-truth events: %s",
                    item.chart_id,
                    sum(gt_diag.unmapped.values()),
                    gt_diag.unmapped,
                )
            if pred_diag.unmapped:
                logger.warning(
                    "chart %r has %d unmapped prediction events: %s",
                    item.chart_id,
                    sum(pred_diag.unmapped.values()),
                    pred_diag.unmapped,
                )

            if export_reference_midi:
                write_reference_midi(
                    ground_truth, output_dir / "reference_midi" / f"{item.chart_id}.mid"
                )

            for tolerance in tolerance_ms:
                tolerance_sec = tolerance / 1000
                if align:
                    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec)
                    reports.append(ChartReport(item.chart_id, tolerance, "raw", result.raw.summary))
                    reports.append(
                        ChartReport(item.chart_id, tolerance, "aligned", result.aligned.summary)
                    )
                else:
                    result = score_events(ground_truth, predictions, tolerance_sec)
                    reports.append(ChartReport(item.chart_id, tolerance, "raw", result.summary))
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to score chart %r; skipping", item.chart_id)
            failed_charts.append(item.chart_id)

    if failed_charts:
        raise RuntimeError(
            f"Scoring failed for {len(failed_charts)} chart(s): {', '.join(failed_charts)}"
        )
    write_reports(reports, output_dir)
    return reports


def export_reference_midis(charts_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    failed = 0
    for dtx_path in sorted(
        p for p in charts_dir.iterdir() if p.is_file() and p.suffix.lower() in CHART_SUFFIXES
    ):
        try:
            chart = parse_dtx_file(dtx_path, chart_id=dtx_path.stem)
            ground_truth, gt_diag = map_dtx_events(dtx_events_to_timed_events(chart))
            if gt_diag.unmapped:
                logger.warning(
                    "chart %r has %d unmapped ground-truth events: %s",
                    dtx_path.stem,
                    sum(gt_diag.unmapped.values()),
                    gt_diag.unmapped,
                )
            write_reference_midi(ground_truth, output_dir / f"{dtx_path.stem}.mid")
            count += 1
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to export reference MIDI for %s; skipping", dtx_path.name)
            failed += 1

    if failed:
        logger.warning("Export failed for %d chart(s)", failed)
    return count


def run_transcribe_and_score(
    charts_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    tolerance_ms: list[int],
    transcribe: Callable[[Path], bytes] | None = None,
) -> list[ChartReport]:
    if transcribe is None:
        transcriber = _create_shared_transcriber()
        transcribe = _make_transcribe_fn(transcriber)

    predictions_dir = output_dir / "predictions"
    matched_charts_dir = output_dir / "_matched_charts"

    # Clear stale artefacts from previous runs so that removed or skipped
    # charts don't silently contaminate the new results.
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
    if matched_charts_dir.exists():
        shutil.rmtree(matched_charts_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    matched_charts_dir.mkdir(parents=True, exist_ok=True)

    missing_audio: list[str] = []
    failed_charts: list[str] = []
    for dtx_path in sorted(
        p for p in charts_dir.iterdir() if p.is_file() and p.suffix.lower() in CHART_SUFFIXES
    ):
        try:
            audio_path = _find_audio(audio_dir, dtx_path.stem)
        except FileNotFoundError:
            missing_audio.append(dtx_path.stem)
            continue
        try:
            midi_bytes = transcribe(audio_path)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Transcription failed for chart %r; skipping", dtx_path.stem)
            failed_charts.append(dtx_path.stem)
            continue
        try:
            (predictions_dir / f"{dtx_path.stem}.mid").write_bytes(midi_bytes)
            shutil.copy2(dtx_path, matched_charts_dir / dtx_path.name)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to write output for chart %r; skipping", dtx_path.stem)
            failed_charts.append(dtx_path.stem)

    if missing_audio:
        logger.warning(
            "Skipping %d chart(s) with missing audio: %s",
            len(missing_audio),
            ", ".join(missing_audio),
        )
    if failed_charts:
        logger.warning(
            "Transcription failed for %d chart(s): %s",
            len(failed_charts),
            ", ".join(failed_charts),
        )

    if not any(matched_charts_dir.iterdir()):
        reasons: list[str] = []
        if missing_audio:
            reasons.append(f"{len(missing_audio)} chart(s) with missing audio")
        if failed_charts:
            reasons.append(f"{len(failed_charts)} chart(s) failed transcription")
        raise RuntimeError(f"No charts available for scoring ({'; '.join(reasons)})")

    return run_score_midi(matched_charts_dir, predictions_dir, output_dir, tolerance_ms, align=True)


def _find_audio(audio_dir: Path, chart_id: str) -> Path:
    for suffix in (".wav", ".mp3", ".m4a", ".flac"):
        path = audio_dir / f"{chart_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing audio for chart_id {chart_id}")


def _create_shared_transcriber():
    """Construct a single DrumTranscriber instance to reuse across all charts."""
    from src.app.transcriber import DrumTranscriber

    return DrumTranscriber()


def _make_transcribe_fn(transcriber):
    """Return a callable that reuses *transcriber* with a fresh jobs_store per call."""

    def _transcribe(audio_path: Path) -> bytes:
        job_id = audio_path.stem
        jobs_store = {job_id: {"progress": 0}}
        return asyncio.run(transcriber.transcribe(str(audio_path), job_id, jobs_store))

    return _transcribe
