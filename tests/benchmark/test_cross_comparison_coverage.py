from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.benchmark import cross_comparison
from src.benchmark.published_comparison import ComparisonIntegrityError
from tests.benchmark.test_cross_comparison import (
    _EXPECTED_ARTIFACTS,
    _request,
    _task3_summaries,
)


def test_cross_comparison_request_rejects_non_path_cache(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(TypeError, match="separation_cache_dir"):
        cross_comparison.CrossComparisonRequest(
            **{**request.__dict__, "separation_cache_dir": "cache"}  # type: ignore[arg-type]
        )


def test_cross_comparison_outcome_rejects_malformed_fields() -> None:
    base: dict[str, Any] = {
        "output_dir": Path("published"),
        "headline_matrix_path": Path("published/headline_matrix.csv"),
        "comparison_paths": {"comparison": Path("published/comparison")},
        "pairable_success_counts": {"comparison": 0},
    }

    invalid_cases = (
        ("output_dir", "published", "output_dir"),
        ("headline_matrix_path", "headline.csv", "headline_matrix_path"),
        ("comparison_paths", {"": Path("comparison")}, "comparison_paths keys"),
        ("comparison_paths", {"comparison": "comparison"}, "comparison_paths values"),
        ("pairable_success_counts", [], "pairable_success_counts must be a dict"),
        ("pairable_success_counts", {"": 0}, "pairable_success_counts keys"),
    )
    for field, value, message in invalid_cases:
        values = {**base, field: value}
        with pytest.raises((TypeError, ValueError), match=message):
            cross_comparison.CrossComparisonOutcome(**values)


def test_cross_comparison_model_rejects_malformed_model_value() -> None:
    summary = _task3_summaries()["oaf_muscriptor_full_mix"]
    summary["models"]["oaf"] = None

    with pytest.raises(ComparisonIntegrityError, match=r"models\['oaf'\]"):
        cross_comparison._model(summary, "oaf")


def test_cross_comparison_oaf_identity_rejects_non_string_lock() -> None:
    summaries = _task3_summaries()
    summaries["oaf_idm_htdemucs"]["models"]["oaf"]["model_lock_sha256"] = None

    with pytest.raises(ComparisonIntegrityError, match="IDM OaF model_lock_sha256"):
        cross_comparison._validate_oaf_identity(summaries)


def test_cross_comparison_population_rejects_malformed_values() -> None:
    with pytest.raises(ComparisonIntegrityError, match="population is malformed"):
        cross_comparison._validate_population({}, comparison_id="comparison", model_key="model")

    with pytest.raises(ComparisonIntegrityError, match="population fields are invalid"):
        cross_comparison._validate_population(
            {"population": {"total_count": 1}},
            comparison_id="comparison",
            model_key="model",
        )

    population = {field: 0 for field in cross_comparison._POPULATION_FIELDS}
    population["success_count"] = True
    with pytest.raises(ComparisonIntegrityError, match="population.success_count is invalid"):
        cross_comparison._validate_population(
            {"population": population},
            comparison_id="comparison",
            model_key="model",
        )


def test_cross_comparison_headline_rows_reject_input_view_mismatch() -> None:
    summaries = _task3_summaries()
    summaries["oaf_muscriptor_full_mix"]["models"]["oaf"]["input_view_id"] = "wrong-view"

    with pytest.raises(ComparisonIntegrityError, match="input_view_id mismatch"):
        cross_comparison._headline_rows(summaries)


def test_cross_comparison_pairable_counts_reject_missing_path() -> None:
    with pytest.raises(ComparisonIntegrityError, match="missing pairing"):
        cross_comparison._pairable_success_counts({"oaf_muscriptor_full_mix": {}})


def _artifact_stage(tmp_path: Path, comparison_id: str) -> Path:
    stage = tmp_path / comparison_id
    for relative in _EXPECTED_ARTIFACTS[comparison_id]:
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact\n", encoding="utf-8")
    return stage


def test_cross_comparison_artifact_index_rejects_directory_walk_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _artifact_stage(tmp_path, "oaf_muscriptor_full_mix")

    def fail_walk(_root, onerror=None):
        if onerror is not None:
            onerror(OSError("walk denied"))
        raise AssertionError("os.walk must surface failures through onerror")

    monkeypatch.setattr(cross_comparison.os, "walk", fail_walk)
    with pytest.raises(ComparisonIntegrityError, match="cannot inspect comparison artifacts"):
        cross_comparison._artifact_index(stage, "oaf_muscriptor_full_mix")


def test_cross_comparison_artifact_index_rejects_symlinked_directory(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    stage = _artifact_stage(tmp_path, "oaf_muscriptor_full_mix")
    (stage / "intruder").symlink_to(stage)

    with pytest.raises(ComparisonIntegrityError, match="unexpected comparison artifact"):
        cross_comparison._artifact_index(stage, "oaf_muscriptor_full_mix")


def test_cross_comparison_artifact_index_rejects_artifact_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _artifact_stage(tmp_path, "oaf_muscriptor_full_mix")

    def fail_read(_path: Path) -> bytes:
        raise OSError("read denied")

    monkeypatch.setattr(cross_comparison, "read_regular_file_no_follow", fail_read)
    with pytest.raises(ComparisonIntegrityError, match="invalid comparison artifact"):
        cross_comparison._artifact_index(stage, "oaf_muscriptor_full_mix")


@pytest.mark.parametrize("value", (123, "not-a-sha"))
def test_cross_comparison_scope_identity_rejects_malformed_subset_hash(value: object) -> None:
    summaries = _task3_summaries()
    summaries["oaf_separation_pilot"]["identity"]["reviewed_subset_manifest_sha256"] = value

    with pytest.raises(ComparisonIntegrityError, match="reviewed_subset_manifest_sha256"):
        cross_comparison._scope_identities(summaries)


def test_publish_cross_comparisons_rejects_wrong_request_type() -> None:
    with pytest.raises(TypeError, match="request must be CrossComparisonRequest"):
        cross_comparison.publish_cross_comparisons(None)  # type: ignore[arg-type]


def test_publish_cross_comparisons_wraps_driver_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_driver(_request: Any) -> None:
        raise ValueError("driver failure")

    monkeypatch.setattr(cross_comparison, "compare_oaf_muscriptor", fail_driver)
    with pytest.raises(ComparisonIntegrityError, match="driver failure"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))
