import json
from pathlib import Path

from src.benchmark.models import ScoreSummary
from src.benchmark.reports import ChartReport, write_reports


def test_write_reports_outputs_json_csv_and_markdown(tmp_path: Path):
    report = ChartReport(
        chart_id="song",
        tolerance_ms=50,
        mode="raw",
        summary=ScoreSummary(true_positives=1, false_positives=0, false_negatives=1),
    )

    write_reports([report], tmp_path)

    assert json.loads((tmp_path / "summary.json").read_text())["charts"][0]["chart_id"] == "song"
    assert "chart_id,tolerance_ms,mode" in (tmp_path / "per_chart.csv").read_text()
    assert "# DTX MIDI Benchmark Summary" in (tmp_path / "summary.md").read_text()
