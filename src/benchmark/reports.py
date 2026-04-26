from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.benchmark.models import ScoreSummary


@dataclass(frozen=True)
class ChartReport:
    chart_id: str
    tolerance_ms: int
    mode: str
    summary: ScoreSummary


def write_reports(reports: list[ChartReport], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_json(reports, output_dir / "summary.json")
    _write_per_chart_csv(reports, output_dir / "per_chart.csv")
    _write_summary_markdown(reports, output_dir / "summary.md")


def _report_row(report: ChartReport) -> dict[str, object]:
    summary = report.summary
    return {
        "chart_id": report.chart_id,
        "tolerance_ms": report.tolerance_ms,
        "mode": report.mode,
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "false_negatives": summary.false_negatives,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "median_abs_error_ms": (
            summary.median_abs_error_sec * 1000
            if summary.median_abs_error_sec is not None
            else None
        ),
        "p95_abs_error_ms": (
            summary.p95_abs_error_sec * 1000 if summary.p95_abs_error_sec is not None else None
        ),
        "offset_ms": summary.offset_sec * 1000,
    }


def _write_summary_json(reports: list[ChartReport], path: Path) -> None:
    payload = {"charts": [_report_row(report) for report in reports]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_per_chart_csv(reports: list[ChartReport], path: Path) -> None:
    rows = [_report_row(report) for report in reports]
    fieldnames = list(rows[0]) if rows else ["chart_id", "tolerance_ms", "mode"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(reports: list[ChartReport], path: Path) -> None:
    lines = ["# DTX MIDI Benchmark Summary", ""]
    for report in reports:
        row = _report_row(report)
        f1_str = f"{row['f1']:.3f}" if row["f1"] is not None else "N/A"
        p_str = f"{row['precision']:.3f}" if row["precision"] is not None else "N/A"
        r_str = f"{row['recall']:.3f}" if row["recall"] is not None else "N/A"
        lines.append(
            f"- `{report.chart_id}` {report.mode} @{report.tolerance_ms}ms: "
            f"F1={f1_str}, P={p_str}, R={r_str}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
