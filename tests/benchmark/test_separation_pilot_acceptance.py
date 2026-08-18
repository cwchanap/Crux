"""Fake-runtime acceptance coverage for HPA-328 Task 6."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.benchmark.separators import SeparatorExecutionError
from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
from tests.benchmark.test_separation_pilot import (
    _install_fixture_locks,
    _request,
    _subset_path,
    _task6_seams,
)


def test_separator_timeout_is_native_failure_and_other_view_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(
        tmp_path,
        fixture,
        monkeypatch,
        spleeter_error=SeparatorExecutionError("separator_timeout", "fake timeout"),
    )

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 1
    assert set(calls["transcribe"]) == {
        "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    }
    assert outcome.run_path is not None
    snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
    assert {item["spleeter"]["status"] for item in snapshot["items"]} == {
        "separation_failed",
    }
    assert {item["spleeter"]["failure_code"] for item in snapshot["items"]} == {
        "separator_timeout",
    }
    assert {item["htdemucs"]["status"] for item in snapshot["items"]} == {"success"}


def test_separation_failure_to_cohort_reason_is_frozen() -> None:
    from src.benchmark.separation_pilot import SEPARATION_FAILURE_TO_COHORT_REASON

    assert SEPARATION_FAILURE_TO_COHORT_REASON == {
        "separation_failed": "inference_failed",
        "stem_invalid": "inference_failed",
        "canonical_input_failed": "inference_failed",
        "inference_failed": "inference_failed",
        "prediction_invalid": "prediction_artifact_invalid",
        "prediction_output_conflict": "prediction_artifact_invalid",
        "prediction_publish_failed": "prediction_artifact_invalid",
    }


def test_derived_hpa325_reports_retain_reviewed_population_on_mixed_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(
        tmp_path,
        fixture,
        monkeypatch,
        spleeter_error=SeparatorExecutionError("separator_timeout", "fake timeout"),
    )

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 1
    assert outcome.run_path is not None
    snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
    expected_ids = [str(item["simfile_id"]) for item in snapshot["items"]]

    for view_name, expected_status, expected_reason in (
        ("spleeter", "failed", "inference_failed"),
        ("htdemucs", "success", ""),
    ):
        report_dir = outcome.run_path.parent / "views" / view_name / "reports"
        with (report_dir / "items.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["simfile_id"] for row in rows] == expected_ids
        assert {row["status"] for row in rows} == {expected_status}
        assert {row["failure_reason"] for row in rows} == {expected_reason}

    assert {item["spleeter"]["failure_code"] for item in snapshot["items"]} == {"separator_timeout"}
