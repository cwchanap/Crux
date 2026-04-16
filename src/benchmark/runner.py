from __future__ import annotations

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
