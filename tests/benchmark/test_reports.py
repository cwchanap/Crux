from __future__ import annotations

import csv
import math
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import StrictJsonError, strict_json_loads
from src.benchmark.cohort_scoring import (
    CohortCoverage,
    CohortIdentity,
    CohortItem,
    score_cohort,
)
from src.benchmark.models import BenchmarkEvent
from src.benchmark.reports import (
    REPORT_SCHEMA,
    _csv_decimal,
    _report_decimal,
    write_cohort_reports,
)


def _identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version="crux.dtx-taxonomy/v1",
        lane_map_version="crux.dtx-lane-map/v1",
        backend_id="backend-v1",
        model_id="model-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version="map-v1",
        input_view_id="full-mix-v1",
    )


def _item(
    simfile_id: str,
    *,
    prediction_events: tuple[BenchmarkEvent, ...] | None,
    status: str = "success",
    failure_reason: str | None = None,
    warnings: tuple[str, ...] = (),
    prediction_native_event_count: int | None = None,
    prediction_mapped_event_count: int | None = None,
    prediction_unmapped_event_count: int | None = None,
    prediction_native_class_counts: tuple[tuple[str, int], ...] = (),
) -> CohortItem:
    reference_events = (BenchmarkEvent(simfile_id, 0.5, "kick", "ground_truth"),)
    return CohortItem(
        simfile_id=simfile_id,
        status=status,  # type: ignore[arg-type]
        reference_events=reference_events,
        prediction_events=prediction_events,
        coverage=CohortCoverage(
            reference_native_event_count=1,
            reference_common_event_count=1,
            reference_ignored_event_count=0,
            reference_unmapped_event_count=0,
            reference_duplicate_collapsed_count=0,
            prediction_native_event_count=prediction_native_event_count,
            prediction_mapped_event_count=prediction_mapped_event_count,
            prediction_unmapped_event_count=prediction_unmapped_event_count,
            prediction_native_class_counts=prediction_native_class_counts,
        ),
        warnings=warnings,
        failure_reason=failure_reason,  # type: ignore[arg-type]
    )


def _prediction(simfile_id: str, time_sec: float) -> BenchmarkEvent:
    return BenchmarkEvent(
        simfile_id,
        time_sec,
        "kick",
        "prediction",
        metadata={
            "input_view_id": "full-mix-v1",
            "prediction_map_version": "map-v1",
        },
    )


def _result(*, diagnostics_for: tuple[str, ...] = (), reverse_items: bool = False):
    success = _item(
        "1",
        prediction_events=(_prediction("1", 0.5), _prediction("1", 1.0)),
        warnings=("z-warning", "a-warning"),
        prediction_native_event_count=2,
        prediction_mapped_event_count=2,
        prediction_unmapped_event_count=0,
        prediction_native_class_counts=(("midi_46", 1), ("midi_36", 1)),
    )
    empty = _item(
        "2",
        prediction_events=(),
        prediction_native_event_count=0,
        prediction_mapped_event_count=0,
        prediction_unmapped_event_count=0,
    )
    failed = _item(
        "3",
        prediction_events=None,
        status="failed",
        failure_reason="prediction_missing",
    )
    items = (failed, empty, success)
    if reverse_items:
        items = tuple(reversed(items))
    return score_cohort(_identity(), items, tolerances_ms=(50,), diagnostics_for=diagnostics_for)


def test_report_decimal_boundary() -> None:
    assert _report_decimal(0.5) == Decimal("0.500000")
    assert _report_decimal(None) is None
    assert _csv_decimal(0.5) == "0.5"
    assert _csv_decimal(None) == ""
    with pytest.raises(StrictJsonError):
        _report_decimal(math.nan)


def test_write_cohort_reports_outputs_six_contract_files(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)

    assert artifacts.summary_json == tmp_path / "summary.json"
    assert artifacts.items_csv == tmp_path / "items.csv"
    assert artifacts.per_song_csv == tmp_path / "per_song.csv"
    assert artifacts.per_class_csv == tmp_path / "per_class.csv"
    assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
    assert artifacts.summary_markdown == tmp_path / "summary.md"

    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    assert set(summary) == {"schema", "identity", "tolerances_ms", "population", "aggregates"}
    assert summary["schema"] == REPORT_SCHEMA
    assert "items" not in summary
    assert summary["aggregates"][0]["event_micro"]["precision"] == Decimal("0.5")
    assert summary["aggregates"][0]["song_macro_f1"] is not None

    with artifacts.items_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["simfile_id"] == "1"
    assert rows[0]["warnings"] == "a-warning|z-warning"
    assert rows[0]["prediction_mapping_coverage"] == "1"
    empty_row = next(row for row in rows if row["simfile_id"] == "2")
    assert empty_row["prediction_mapping_coverage"] == ""
    assert all(row["prediction_mapping_coverage"] == "" for row in rows if row["simfile_id"] == "3")


def test_per_song_and_per_class_are_song_scoped(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    with artifacts.per_song_csv.open(newline="", encoding="utf-8") as handle:
        song_rows = list(csv.DictReader(handle))
    with artifacts.per_class_csv.open(newline="", encoding="utf-8") as handle:
        class_rows = list(csv.DictReader(handle))

    assert song_rows
    assert song_rows[0]["precision"] == "0.5"
    assert "backend_descriptor_sha256" not in song_rows[0]
    assert class_rows
    assert "scope" not in class_rows[0]
    assert {row["simfile_id"] for row in class_rows} == {"1", "2"}


def test_event_diagnostics_are_slim_canonical_jsonl(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path)
    lines = artifacts.event_diagnostics_jsonl.read_bytes().splitlines()
    assert lines
    expected = (
        b'{"cohort_id":"oaf-full-mix-v1","common_class":"kick","mode":"raw",'
        b'"outcome":"matched","prediction_time_sec":0.5,"reference_time_sec":0.5,'
        b'"scored_prediction_time_sec":0.5,"simfile_id":"1","timing_error_sec":0,'
        b'"tolerance_ms":50}'
    )
    assert expected in lines
    assert set(strict_json_loads(lines[0], require_canonical=True)) == {
        "cohort_id",
        "simfile_id",
        "tolerance_ms",
        "mode",
        "outcome",
        "common_class",
        "reference_time_sec",
        "prediction_time_sec",
        "scored_prediction_time_sec",
        "timing_error_sec",
    }


def test_default_diagnostics_are_empty(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    assert artifacts.event_diagnostics_jsonl.read_bytes() == b""


def test_report_bytes_are_deterministic_for_same_result(tmp_path: Path) -> None:
    first = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "first")
    second = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "second")
    for first_path, second_path in zip(
        (
            first.summary_json,
            first.items_csv,
            first.per_song_csv,
            first.per_class_csv,
            first.event_diagnostics_jsonl,
            first.summary_markdown,
        ),
        (
            second.summary_json,
            second.items_csv,
            second.per_song_csv,
            second.per_class_csv,
            second.event_diagnostics_jsonl,
            second.summary_markdown,
        ),
    ):
        assert first_path.read_bytes() == second_path.read_bytes()


def test_reversed_input_order_keeps_all_report_bytes_identical(tmp_path: Path) -> None:
    first = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "first")
    second = write_cohort_reports(
        _result(diagnostics_for=("1",), reverse_items=True), tmp_path / "second"
    )
    for first_path, second_path in zip(
        (
            first.summary_json,
            first.items_csv,
            first.per_song_csv,
            first.per_class_csv,
            first.event_diagnostics_jsonl,
            first.summary_markdown,
        ),
        (
            second.summary_json,
            second.items_csv,
            second.per_song_csv,
            second.per_class_csv,
            second.event_diagnostics_jsonl,
            second.summary_markdown,
        ),
    ):
        assert first_path.read_bytes() == second_path.read_bytes()
