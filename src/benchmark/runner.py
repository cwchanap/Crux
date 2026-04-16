from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from src.benchmark.corpus import discover_score_midi_items
from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.mapping import map_dtx_events, map_midi_events
from src.benchmark.midi_io import parse_prediction_midi, write_reference_midi
from src.benchmark.reports import ChartReport, write_reports
from src.benchmark.scoring import score_events, score_events_with_alignment
from src.benchmark.timing import dtx_events_to_timed_events


def run_score_midi(
    charts_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    tolerance_ms: list[int],
    align: bool = True,
    export_reference_midi: bool = False,
) -> list[ChartReport]:
    reports: list[ChartReport] = []
    for item in discover_score_midi_items(charts_dir, predictions_dir):
        chart = parse_dtx_file(item.dtx_path, chart_id=item.chart_id)
        ground_truth, _ = map_dtx_events(dtx_events_to_timed_events(chart))
        predictions, _ = map_midi_events(
            parse_prediction_midi(item.prediction_midi_path, item.chart_id)
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

    write_reports(reports, output_dir)
    return reports


def export_reference_midis(charts_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dtx_path in sorted(charts_dir.glob("*.dtx")):
        chart = parse_dtx_file(dtx_path, chart_id=dtx_path.stem)
        ground_truth, _ = map_dtx_events(dtx_events_to_timed_events(chart))
        write_reference_midi(ground_truth, output_dir / f"{dtx_path.stem}.mid")
        count += 1
    return count


def run_transcribe_and_score(
    charts_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    tolerance_ms: list[int],
    transcribe: Callable[[Path], bytes] | None = None,
) -> list[ChartReport]:
    if transcribe is None:
        transcribe = _default_transcribe_audio

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for dtx_path in sorted(charts_dir.glob("*.dtx")):
        audio_path = _find_audio(audio_dir, dtx_path.stem)
        midi_bytes = transcribe(audio_path)
        (predictions_dir / f"{dtx_path.stem}.mid").write_bytes(midi_bytes)

    return run_score_midi(charts_dir, predictions_dir, output_dir, tolerance_ms, align=True)


def _find_audio(audio_dir: Path, chart_id: str) -> Path:
    for suffix in (".wav", ".mp3", ".m4a", ".flac"):
        path = audio_dir / f"{chart_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing audio for chart_id {chart_id}")


def _default_transcribe_audio(audio_path: Path) -> bytes:
    from src.app.transcriber import DrumTranscriber

    job_id = audio_path.stem
    jobs_store = {job_id: {"progress": 0}}
    transcriber = DrumTranscriber()
    return asyncio.run(transcriber.transcribe(str(audio_path), job_id, jobs_store))
