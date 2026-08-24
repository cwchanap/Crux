from __future__ import annotations

import csv
import json
import os
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.benchmark import cross_comparison
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.idm_pilot_run import IDM_STEM_INPUT_VIEW_ID
from src.benchmark.muscriptor_comparison import ComparisonOutcome
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.published_comparison import ComparisonIntegrityError
from src.benchmark.separation_comparison import (
    HTDEMUCS_INPUT_VIEW_ID,
    SeparationComparisonOutcome,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP_VERSION,
    OAF_PREDICTION_MAP_ID,
    TAXONOMY_VERSION,
)

_MODEL_LOCK = "a" * 64
_IDENTITY_FIELDS = (
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "taxonomy_version",
    "lane_map_version",
    "scoring_version",
)


def _request(
    tmp_path: Path, *, output_dir: Path | None = None
) -> cross_comparison.CrossComparisonRequest:
    return cross_comparison.CrossComparisonRequest(
        oaf_run_path=tmp_path / "oaf-run.json",
        muscriptor_run_path=tmp_path / "muscriptor-run.json",
        separation_run_path=tmp_path / "separation-run.json",
        idm_run_path=tmp_path / "idm-run.json",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        subset_manifest_path=tmp_path / "subset.jsonl",
        output_dir=output_dir or tmp_path / "published",
        separation_cache_dir=tmp_path / "cache",
    )


def _identity() -> dict[str, str]:
    return {
        "reference_manifest_sha256": "b" * 64,
        "reference_manifest_version": "reference-v1",
        "reference_timing_manifest_sha256": "c" * 64,
        "reference_timing_version": "timing-v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "scoring_version": SCORING_VERSION,
    }


def _summaries() -> dict[str, dict[str, Any]]:
    identity = _identity()
    return {
        "oaf_muscriptor_full_mix": {
            "schema": cross_comparison.COMPARISON_SCHEMA,
            "identity": deepcopy(identity),
            "models": {
                "oaf": {
                    "model_lock_sha256": _MODEL_LOCK,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                },
                "muscriptor": {
                    "model_lock_sha256": "b" * 64,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                },
            },
        },
        "oaf_separation_pilot": {
            "schema": cross_comparison.SEPARATION_COMPARISON_SCHEMA,
            "identity": {
                **deepcopy(identity),
                "reviewed_subset_manifest_sha256": "d" * 64,
            },
            "models": {
                "full_mix": {
                    "model_lock_sha256": _MODEL_LOCK,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                },
                "spleeter": {
                    "model_lock_sha256": _MODEL_LOCK,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
                },
                "htdemucs": {
                    "model_lock_sha256": _MODEL_LOCK,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": HTDEMUCS_INPUT_VIEW_ID,
                },
            },
        },
        "oaf_idm_htdemucs": {
            "schema": cross_comparison.IDM_COMPARISON_SCHEMA,
            "identity": deepcopy(identity),
            "models": {
                "oaf": {
                    "model_lock_sha256": _MODEL_LOCK,
                    "prediction_map_version": OAF_PREDICTION_MAP_ID,
                    "input_view_id": IDM_STEM_INPUT_VIEW_ID,
                },
                "idm": {
                    "model_lock_sha256": "e" * 64,
                    "input_view_id": IDM_STEM_INPUT_VIEW_ID,
                },
            },
        },
    }


def _task3_summaries() -> dict[str, dict[str, Any]]:
    summaries = _summaries()
    summaries["oaf_muscriptor_full_mix"]["models"]["oaf"]["population"] = {
        "total_count": 101,
        "eligible_count": 91,
        "success_count": 81,
        "failed_count": 7,
        "skipped_count": 2,
        "quarantined_count": 1,
    }
    summaries["oaf_muscriptor_full_mix"]["models"]["muscriptor"]["population"] = {
        "total_count": 102,
        "eligible_count": 92,
        "success_count": 82,
        "failed_count": 8,
        "skipped_count": 1,
        "quarantined_count": 1,
    }
    summaries["oaf_muscriptor_full_mix"]["pairing"] = {
        "pairable_success_intersection": 11,
    }
    summaries["oaf_separation_pilot"]["models"]["full_mix"]["population"] = {
        "total_count": 201,
        "eligible_count": 191,
        "success_count": 181,
        "failed_count": 9,
        "skipped_count": 1,
        "quarantined_count": 10,
    }
    summaries["oaf_separation_pilot"]["models"]["spleeter"]["population"] = {
        "total_count": 202,
        "eligible_count": 192,
        "success_count": 182,
        "failed_count": 8,
        "skipped_count": 2,
        "quarantined_count": 10,
    }
    summaries["oaf_separation_pilot"]["models"]["htdemucs"]["population"] = {
        "total_count": 203,
        "eligible_count": 193,
        "success_count": 183,
        "failed_count": 7,
        "skipped_count": 3,
        "quarantined_count": 10,
    }
    summaries["oaf_separation_pilot"]["pairing"] = {
        "spleeter": {"pairable_success_intersection": 12},
        "htdemucs": {"pairable_success_intersection": 13},
    }
    summaries["oaf_idm_htdemucs"]["models"]["oaf"]["population"] = {
        "total_count": 999,
        "eligible_count": 999,
        "success_count": 999,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
    }
    summaries["oaf_idm_htdemucs"]["models"]["idm"]["population"] = {
        "total_count": 301,
        "eligible_count": 291,
        "success_count": 281,
        "failed_count": 9,
        "skipped_count": 1,
        "quarantined_count": 10,
    }
    summaries["oaf_idm_htdemucs"]["pairing"] = {
        "pairable_success_intersection": 14,
    }
    return summaries


def _write_summary(path: Path, value: object, *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value)
    path.write_bytes(content if canonical else content + b"\n")


def _patch_drivers(
    monkeypatch: pytest.MonkeyPatch,
    summaries: dict[str, dict[str, Any]],
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    def fake_muscriptor(request: Any) -> ComparisonOutcome:
        calls.append(("muscriptor", request))
        _write_summary(request.output_dir / "summary.json", summaries["oaf_muscriptor_full_mix"])
        _write_non_summary_artifacts(request.output_dir, "oaf_muscriptor_full_mix")
        return ComparisonOutcome(output_dir=request.output_dir)

    def fake_separation(request: Any) -> SeparationComparisonOutcome:
        calls.append(("separation", request))
        _write_summary(request.output_dir / "summary.json", summaries["oaf_separation_pilot"])
        _write_non_summary_artifacts(request.output_dir, "oaf_separation_pilot")
        return SeparationComparisonOutcome(
            output_dir=request.output_dir,
            pairable_success_counts={},
            paired_song_counts={},
            paired_class_counts={},
        )

    def fake_idm(request: Any) -> Path:
        calls.append(("idm", request))
        _write_summary(request.output_dir / "summary.json", summaries["oaf_idm_htdemucs"])
        _write_non_summary_artifacts(request.output_dir, "oaf_idm_htdemucs")
        return request.output_dir

    monkeypatch.setattr(cross_comparison, "compare_oaf_muscriptor", fake_muscriptor)
    monkeypatch.setattr(cross_comparison, "compare_oaf_separation", fake_separation)
    monkeypatch.setattr(cross_comparison, "compare_oaf_idm", fake_idm)
    return calls


_EXPECTED_ARTIFACTS = {
    "oaf_muscriptor_full_mix": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
    "oaf_separation_pilot": (
        "summary.json",
        "summary.md",
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
    ),
    "oaf_idm_htdemucs": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
}


def _write_task3_artifacts(
    output_dir: Path, comparison_id: str, summary: Mapping[str, Any]
) -> None:
    _write_non_summary_artifacts(output_dir, comparison_id)
    _write_summary(output_dir / "summary.json", summary)


def _write_non_summary_artifacts(output_dir: Path, comparison_id: str) -> None:
    for relative in _EXPECTED_ARTIFACTS[comparison_id]:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "summary.json":
            continue
        if relative == "summary.md":
            path.write_text(f"# {comparison_id}\n", encoding="utf-8")
        else:
            path.write_text("placeholder\n", encoding="utf-8")


def _patch_task3_drivers(
    monkeypatch: pytest.MonkeyPatch,
    summaries: dict[str, dict[str, Any]],
    *,
    mutate: Any | None = None,
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    def fake_muscriptor(request: Any) -> ComparisonOutcome:
        calls.append(("muscriptor", request))
        _write_task3_artifacts(
            request.output_dir,
            "oaf_muscriptor_full_mix",
            summaries["oaf_muscriptor_full_mix"],
        )
        if mutate is not None:
            mutate("oaf_muscriptor_full_mix", request.output_dir)
        return ComparisonOutcome(output_dir=request.output_dir)

    def fake_separation(request: Any) -> SeparationComparisonOutcome:
        calls.append(("separation", request))
        _write_task3_artifacts(
            request.output_dir,
            "oaf_separation_pilot",
            summaries["oaf_separation_pilot"],
        )
        if mutate is not None:
            mutate("oaf_separation_pilot", request.output_dir)
        return SeparationComparisonOutcome(
            output_dir=request.output_dir,
            pairable_success_counts={"spleeter": 999, "htdemucs": 998},
            paired_song_counts={},
            paired_class_counts={},
        )

    def fake_idm(request: Any) -> Path:
        calls.append(("idm", request))
        _write_task3_artifacts(
            request.output_dir,
            "oaf_idm_htdemucs",
            summaries["oaf_idm_htdemucs"],
        )
        if mutate is not None:
            mutate("oaf_idm_htdemucs", request.output_dir)
        return request.output_dir

    monkeypatch.setattr(cross_comparison, "compare_oaf_muscriptor", fake_muscriptor)
    monkeypatch.setattr(cross_comparison, "compare_oaf_separation", fake_separation)
    monkeypatch.setattr(cross_comparison, "compare_oaf_idm", fake_idm)
    return calls


def test_cross_comparison_request_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="oaf_run_path"):
        cross_comparison.CrossComparisonRequest(
            oaf_run_path="run.json",  # type: ignore[arg-type]
            muscriptor_run_path=tmp_path / "muscriptor.json",
            separation_run_path=tmp_path / "separation.json",
            idm_run_path=tmp_path / "idm.json",
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "field",
    (
        "muscriptor_run_path",
        "separation_run_path",
        "idm_run_path",
        "reference_manifest_path",
        "timing_manifest_path",
        "subset_manifest_path",
        "output_dir",
    ),
)
def test_cross_comparison_request_rejects_non_path_fields(tmp_path: Path, field: str) -> None:
    values: dict[str, object] = {
        "oaf_run_path": tmp_path / "oaf.json",
        "muscriptor_run_path": tmp_path / "muscriptor.json",
        "separation_run_path": tmp_path / "separation.json",
        "idm_run_path": tmp_path / "idm.json",
        "reference_manifest_path": tmp_path / "reference.jsonl",
        "timing_manifest_path": tmp_path / "timing.jsonl",
        "subset_manifest_path": tmp_path / "subset.jsonl",
        "output_dir": tmp_path / "out",
    }
    values[field] = "not-a-path"
    with pytest.raises(TypeError, match=field):
        cross_comparison.CrossComparisonRequest(**values)  # type: ignore[arg-type]


def test_cross_comparison_request_accepts_optional_cache_none(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = cross_comparison.CrossComparisonRequest(
        **{**request.__dict__, "separation_cache_dir": None}
    )
    assert request.separation_cache_dir is None


def test_cross_comparison_outcome_requires_nonempty_paths_and_counts() -> None:
    outcome = cross_comparison.CrossComparisonOutcome(
        output_dir=Path("published"),
        headline_matrix_path=Path("published/headline_matrix.csv"),
        comparison_paths={"oaf_muscriptor_full_mix": Path("published/comparison")},
        pairable_success_counts={"oaf_muscriptor_full_mix": 0},
    )
    assert outcome.comparison_paths

    with pytest.raises(ValueError, match="comparison_paths"):
        cross_comparison.CrossComparisonOutcome(
            output_dir=Path("published"),
            headline_matrix_path=Path("published/headline_matrix.csv"),
            comparison_paths={},
            pairable_success_counts={},
        )

    for count in (True, -1):
        with pytest.raises(ValueError, match="pairable_success_counts"):
            cross_comparison.CrossComparisonOutcome(
                output_dir=Path("published"),
                headline_matrix_path=Path("published/headline_matrix.csv"),
                comparison_paths={"comparison": Path("published/comparison")},
                pairable_success_counts={"comparison": count},  # type: ignore[dict-item]
            )


def test_publish_cross_comparisons_routes_each_driver_once(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    calls = _patch_task3_drivers(monkeypatch, _task3_summaries())

    outcome = cross_comparison.publish_cross_comparisons(request)

    assert [name for name, _ in calls] == ["muscriptor", "separation", "idm"]
    muscriptor_request = calls[0][1]
    separation_request = calls[1][1]
    idm_request = calls[2][1]
    assert muscriptor_request.subset_manifest_path is None
    assert separation_request.subset_manifest_path == request.subset_manifest_path
    assert separation_request.cache_dir == request.separation_cache_dir
    assert idm_request.run_path == request.idm_run_path
    assert muscriptor_request.output_dir.parts[-2:] == ("comparisons", "oaf-muscriptor")
    assert separation_request.output_dir.parts[-2:] == ("comparisons", "oaf-separation")
    assert idm_request.output_dir.parts[-2:] == ("comparisons", "oaf-idm")
    assert muscriptor_request.output_dir.parent.parent.parent == request.output_dir.parent
    assert separation_request.output_dir.parent.parent.parent == request.output_dir.parent
    assert idm_request.output_dir.parent.parent.parent == request.output_dir.parent
    assert outcome.output_dir == request.output_dir
    assert outcome.comparison_paths == {
        "oaf_muscriptor_full_mix": request.output_dir / "comparisons/oaf-muscriptor",
        "oaf_separation_pilot": request.output_dir / "comparisons/oaf-separation",
        "oaf_idm_htdemucs": request.output_dir / "comparisons/oaf-idm",
    }
    assert request.output_dir.exists()


def test_publish_cross_comparisons_rejects_existing_output_directory(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "published"
    output_dir.mkdir()
    marker = output_dir / "untouched"
    marker.write_text("existing", encoding="utf-8")
    request = _request(tmp_path, output_dir=output_dir)
    calls = _patch_drivers(monkeypatch, _summaries())

    with pytest.raises(ComparisonIntegrityError, match="output_dir"):
        cross_comparison.publish_cross_comparisons(request)

    assert calls == []
    assert marker.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("comparison_id", ["oaf_muscriptor_full_mix", "oaf_separation_pilot"])
def test_publish_cross_comparisons_rejects_noncanonical_or_wrong_schema(
    tmp_path: Path, monkeypatch, comparison_id: str
) -> None:
    summaries = _summaries()
    summaries[comparison_id]["schema"] = "wrong.schema/v0"
    calls = _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="schema"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert len(calls) == 3
    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_rejects_noncanonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    summaries = _summaries()
    calls: list[tuple[str, Any]] = []

    def fake_muscriptor(request: Any) -> ComparisonOutcome:
        calls.append(("muscriptor", request))
        _write_summary(
            request.output_dir / "summary.json",
            summaries["oaf_muscriptor_full_mix"],
            canonical=False,
        )
        return ComparisonOutcome(output_dir=request.output_dir)

    def fake_separation(request: Any) -> SeparationComparisonOutcome:
        calls.append(("separation", request))
        _write_summary(request.output_dir / "summary.json", summaries["oaf_separation_pilot"])
        return SeparationComparisonOutcome(
            output_dir=request.output_dir,
            pairable_success_counts={},
            paired_song_counts={},
            paired_class_counts={},
        )

    def fake_idm(request: Any) -> Path:
        calls.append(("idm", request))
        _write_summary(request.output_dir / "summary.json", summaries["oaf_idm_htdemucs"])
        return request.output_dir

    monkeypatch.setattr(cross_comparison, "compare_oaf_muscriptor", fake_muscriptor)
    monkeypatch.setattr(cross_comparison, "compare_oaf_separation", fake_separation)
    monkeypatch.setattr(cross_comparison, "compare_oaf_idm", fake_idm)

    with pytest.raises(ComparisonIntegrityError, match="invalid comparison summary"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert len(calls) == 3
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("field", _IDENTITY_FIELDS)
def test_publish_cross_comparisons_rejects_shared_identity_mismatch(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    summaries = _summaries()
    summaries["oaf_idm_htdemucs"]["identity"][field] = "mismatch"
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match=field):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_rejects_missing_shared_identity(
    tmp_path: Path, monkeypatch
) -> None:
    summaries = _summaries()
    del summaries["oaf_separation_pilot"]["identity"]["taxonomy_version"]
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="taxonomy_version"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("lock", ("not-a-sha", "b" * 64))
@pytest.mark.parametrize(
    "comparison_id,model_key",
    [
        ("oaf_muscriptor_full_mix", "oaf"),
        ("oaf_separation_pilot", "full_mix"),
        ("oaf_idm_htdemucs", "oaf"),
    ],
)
def test_publish_cross_comparisons_rejects_malformed_or_mismatched_oaf_lock(
    tmp_path: Path, monkeypatch, comparison_id: str, model_key: str, lock: str
) -> None:
    summaries = _summaries()
    summaries[comparison_id]["models"][model_key]["model_lock_sha256"] = lock
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="model_lock_sha256"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_allows_distinct_separation_model_locks(
    tmp_path: Path, monkeypatch
) -> None:
    summaries = _task3_summaries()
    summaries["oaf_separation_pilot"]["models"]["spleeter"]["model_lock_sha256"] = "f" * 64
    summaries["oaf_separation_pilot"]["models"]["htdemucs"]["model_lock_sha256"] = "9" * 64
    _patch_task3_drivers(monkeypatch, summaries)

    outcome = cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert set(outcome.comparison_paths) == {
        "oaf_muscriptor_full_mix",
        "oaf_separation_pilot",
        "oaf_idm_htdemucs",
    }


@pytest.mark.parametrize(
    "comparison_id,model_key",
    [
        ("oaf_muscriptor_full_mix", "oaf"),
        ("oaf_separation_pilot", "full_mix"),
        ("oaf_separation_pilot", "spleeter"),
        ("oaf_separation_pilot", "htdemucs"),
        ("oaf_idm_htdemucs", "oaf"),
    ],
)
def test_publish_cross_comparisons_rejects_oaf_prediction_map_mismatch(
    tmp_path: Path, monkeypatch, comparison_id: str, model_key: str
) -> None:
    summaries = _summaries()
    summaries[comparison_id]["models"][model_key]["prediction_map_version"] = (
        "crux.prediction-map/oaf-egmd-8hit-v0"
    )
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="prediction_map_version mismatch"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("bad_map", (None, ""))
def test_publish_cross_comparisons_rejects_malformed_oaf_prediction_map(
    tmp_path: Path, monkeypatch, bad_map: object
) -> None:
    summaries = _summaries()
    summaries["oaf_muscriptor_full_mix"]["models"]["oaf"]["prediction_map_version"] = bad_map
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="prediction_map_version is malformed"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_rejects_htdemucs_view_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    summaries = _summaries()
    summaries["oaf_separation_pilot"]["models"]["htdemucs"]["input_view_id"] = "wrong-view"
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="HTDemucs input_view_id mismatch"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_rejects_idm_view_mismatch(tmp_path: Path, monkeypatch) -> None:
    summaries = _summaries()
    summaries["oaf_idm_htdemucs"]["models"]["oaf"]["input_view_id"] = "wrong-view"
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="HTDemucs input_view_id mismatch"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "comparison_id,models",
    [
        ("oaf_muscriptor_full_mix", []),
        ("oaf_muscriptor_full_mix", {}),
        ("oaf_separation_pilot", []),
        ("oaf_separation_pilot", {}),
        ("oaf_idm_htdemucs", []),
        ("oaf_idm_htdemucs", {}),
    ],
)
def test_publish_cross_comparisons_rejects_malformed_models_mapping(
    tmp_path: Path, monkeypatch, comparison_id: str, models: object
) -> None:
    summaries = _summaries()
    summaries[comparison_id]["models"] = models
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="models"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_publish_cross_comparisons_renders_closed_headline_matrix_and_index(
    tmp_path: Path, monkeypatch
) -> None:
    summaries = _task3_summaries()
    _patch_task3_drivers(monkeypatch, summaries)

    outcome = cross_comparison.publish_cross_comparisons(_request(tmp_path))

    with outcome.headline_matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["scope"], row["model"], int(row["total_count"])) for row in rows] == [
        ("broad_full_mix", "oaf", 101),
        ("broad_full_mix", "muscriptor", 102),
        ("reviewed_pilot", "oaf", 201),
        ("reviewed_pilot", "oaf", 202),
        ("reviewed_pilot", "oaf", 203),
        ("reviewed_pilot", "idm", 301),
    ]
    assert rows[0]["input_view_id"] == OAF_FULL_MIX_INPUT_VIEW_ID
    assert rows[-1]["input_view_id"] == IDM_STEM_INPUT_VIEW_ID

    summary = json.loads((tmp_path / "published" / "summary.json").read_text())
    assert summary["schema"] == cross_comparison.PAIRED_BENCHMARK_PUBLICATION_SCHEMA
    assert summary["pairable_success_counts"] == {
        "oaf_muscriptor_full_mix": 11,
        "oaf_separation_pilot.spleeter": 12,
        "oaf_separation_pilot.htdemucs": 13,
        "oaf_idm_htdemucs": 14,
    }
    assert summary["identity"]["oaf_model_lock_sha256"] == _MODEL_LOCK
    assert "reviewed_subset" not in summary
    assert summary["comparisons"]["oaf_separation_pilot"]["scope_identity"] == {
        "reviewed_subset_manifest_sha256": "d" * 64,
        "reviewed_subset_cross_verified": True,
    }
    assert summary["comparisons"]["oaf_idm_htdemucs"]["scope_identity"] == {
        "pilot_lineage": "validated_hpa396_run",
        "reviewed_subset_cross_verified": False,
    }
    assert summary["headline_matrix"] == {
        "path": "headline_matrix.csv",
        "sha256": sha256_hex((tmp_path / "published" / "headline_matrix.csv").read_bytes()),
    }
    assert outcome.pairable_success_counts == summary["pairable_success_counts"]
    assert (
        "Broad full-mix and reviewed-pilot rows have different populations"
        in (tmp_path / "published" / "summary.md").read_text()
    )
    assert (
        "The IDM pilot lineage is validated inside its HPA-396 run"
        in (tmp_path / "published" / "summary.md").read_text()
    )

    for comparison_id, relative_files in _EXPECTED_ARTIFACTS.items():
        actual = summary["comparisons"][comparison_id]["artifacts"]
        assert [entry["path"] for entry in actual] == [
            f"{cross_comparison._COMPARISON_DIRS[comparison_id]}/{relative}".replace("\\", "/")
            for relative in relative_files
        ]
        assert all(
            not Path(entry["path"]).is_absolute() and str(tmp_path) not in entry["path"]
            for entry in actual
        )


def test_cross_publication_rejects_missing_expected_artifact(tmp_path: Path, monkeypatch) -> None:
    summaries = _task3_summaries()

    def mutate(comparison_id: str, output_dir: Path) -> None:
        if comparison_id == "oaf_muscriptor_full_mix":
            (output_dir / "paired_per_class.csv").unlink()

    _patch_task3_drivers(monkeypatch, summaries, mutate=mutate)

    with pytest.raises(ComparisonIntegrityError, match="paired_per_class.csv"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_cross_publication_rejects_unexpected_artifact(tmp_path: Path, monkeypatch) -> None:
    summaries = _task3_summaries()

    def mutate(comparison_id: str, output_dir: Path) -> None:
        if comparison_id == "oaf_separation_pilot":
            (output_dir / "extra.csv").write_text("unexpected\n", encoding="utf-8")

    _patch_task3_drivers(monkeypatch, summaries, mutate=mutate)

    with pytest.raises(ComparisonIntegrityError, match="unexpected comparison artifact"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("entry_kind", ["symlink", "directory", "fifo", "dotfile"])
def test_cross_publication_rejects_non_regular_stage_entries(
    tmp_path: Path, monkeypatch, entry_kind: str
) -> None:
    if entry_kind == "symlink" and not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    if entry_kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs unavailable")

    summaries = _task3_summaries()

    def mutate(comparison_id: str, output_dir: Path) -> None:
        if comparison_id == "oaf_muscriptor_full_mix":
            intruder = output_dir / "intruder"
            if entry_kind == "symlink":
                intruder.symlink_to(output_dir / "summary.md")
            elif entry_kind == "directory":
                intruder.mkdir()
                (intruder / "nested.txt").write_text("unexpected\n", encoding="utf-8")
            elif entry_kind == "dotfile":
                intruder = output_dir / ".hidden.csv"
                intruder.write_text("unexpected\n", encoding="utf-8")
            else:
                os.mkfifo(intruder)

    _patch_task3_drivers(monkeypatch, summaries, mutate=mutate)

    with pytest.raises(ComparisonIntegrityError, match="unexpected comparison artifact"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "count_key,path,value",
    [
        (
            "oaf_muscriptor_full_mix",
            ("pairing", "pairable_success_intersection"),
            True,
        ),
        (
            "oaf_separation_pilot.spleeter",
            ("pairing", "spleeter", "pairable_success_intersection"),
            -1,
        ),
        (
            "oaf_separation_pilot.htdemucs",
            ("pairing", "htdemucs", "pairable_success_intersection"),
            None,
        ),
        (
            "oaf_idm_htdemucs",
            ("pairing", "pairable_success_intersection"),
            "14",
        ),
    ],
)
def test_cross_publication_rejects_invalid_pairable_success_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count_key: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    summaries = _task3_summaries()
    comparison_id = count_key.split(".", 1)[0]
    target: Any = summaries[comparison_id]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _patch_task3_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="pairable_success_intersection"):
        cross_comparison.publish_cross_comparisons(_request(tmp_path))

    assert not (tmp_path / "published").exists()


def test_cross_publication_is_root_independent(tmp_path: Path, monkeypatch) -> None:
    summaries = _task3_summaries()
    left = tmp_path / "left" / "published"
    right = tmp_path / "right" / "published"

    _patch_task3_drivers(monkeypatch, summaries)
    cross_comparison.publish_cross_comparisons(_request(tmp_path, output_dir=left))
    _patch_task3_drivers(monkeypatch, summaries)
    cross_comparison.publish_cross_comparisons(_request(tmp_path, output_dir=right))

    for name in ("summary.json", "summary.md", "headline_matrix.csv"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    summary = json.loads((left / "summary.json").read_text())
    for comparison in summary["comparisons"].values():
        assert not Path(comparison["path"]).is_absolute()
        for artifact in comparison["artifacts"]:
            assert not Path(artifact["path"]).is_absolute()
    assert not Path(summary["headline_matrix"]["path"]).is_absolute()


def _rewrite_report_identity(
    report_dir: Path,
    *,
    reference_manifest_sha256: str,
    reference_timing_version: str,
    identity: Mapping[str, str],
) -> None:
    summary_path = report_dir / "summary.json"
    summary = strict_json_loads(summary_path.read_bytes(), require_canonical=True)
    assert isinstance(summary, dict)
    summary_identity = summary["identity"]
    assert isinstance(summary_identity, dict)
    summary_identity.update(
        {
            "reference_manifest_sha256": reference_manifest_sha256,
            "reference_timing_version": reference_timing_version,
            **identity,
        }
    )
    summary_path.write_bytes(canonical_json_bytes(summary))

    for name in ("items.csv", "per_song.csv", "per_class.csv"):
        path = report_dir / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                for key, value in identity.items():
                    if key in row:
                        row[key] = value
                writer.writerow(row)


def _build_real_driver_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> cross_comparison.CrossComparisonRequest:
    """Build one tiny raw fixture while leaving the three comparison drivers real."""
    from src.benchmark.idm_pilot_run import build_run_id, render_idm_pilot_run
    from src.benchmark.muscriptor_corpus_run import render_muscriptor_corpus_run
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run
    from src.benchmark.reference_set_manifest import load_reference_set_manifest
    from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_idm_comparison import (
        _idm_prediction_bytes,
    )
    from tests.benchmark.test_idm_comparison import (
        _snapshot as idm_snapshot,
    )
    from tests.benchmark.test_idm_comparison import (
        _write_reports as write_idm_reports,
    )
    from tests.benchmark.test_muscriptor_comparison import (
        _write_run as write_muscriptor_run,
    )
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _subset_path,
        _task6_seams,
    )
    from tests.benchmark.test_separation_pilot import (
        _request as separation_pilot_request,
    )

    pilot_root = tmp_path / "pilot"
    fixture = build_reviewed_subset_oaf_fixture(
        pilot_root,
        eligible_count=20,
        failed_count=0,
    )
    subset_path = _subset_path(pilot_root, fixture)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    parent_items = tuple(parent["items"])
    assert all(isinstance(item, Mapping) for item in parent_items)
    simfile_ids = tuple(int(item["simfile_id"]) for item in parent_items)
    source_audio_sha256 = str(parent_items[0]["source_audio_sha256"])
    input_audio_sha256 = str(parent_items[0]["input_audio_sha256"])
    oaf_model_lock_sha256 = str(parent["model_lock_sha256"])
    oaf_model_id = str(parent["model_id"])
    oaf_descriptor_sha256 = str(parent["backend_descriptor_sha256"])
    oaf_prediction_map = str(parent["inference_config"]["prediction_map_version"])

    _install_fixture_locks(monkeypatch)
    seams = _task6_seams(pilot_root, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    # The pilot's own optional comparison is not the HPA-562 call; the later
    # coordinator invokes the real separation driver under test.
    monkeypatch.setattr(pilot, "compare_oaf_separation", lambda _request: None)
    from src.benchmark.reviewed_subset import score_oaf_reviewed_subset

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", score_oaf_reviewed_subset)
    separation_outcome = run_oaf_separation_pilot(
        separation_pilot_request(pilot_root, fixture, subset_path),
        backend_factory=seams["factory"][0],  # type: ignore[arg-type]
    )
    assert separation_outcome.exit_code == 0
    assert separation_outcome.run_path is not None
    full_mix_reports = separation_outcome.run_path.parent / "views" / "full_mix" / "reports"
    oaf_reports = fixture.run_path.parent / "reports"
    shutil.copytree(full_mix_reports, oaf_reports)
    _rewrite_report_identity(
        oaf_reports,
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        identity={"cohort_id": str(parent["run_id"])},
    )

    muscriptor_root = tmp_path / "muscriptor"
    muscriptor_run = write_muscriptor_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-real-fixture",
        schema="crux.muscriptor-corpus-run/v1",
        lock="f" * 64,
        prediction_map="muscriptor-map",
        input_view=OAF_FULL_MIX_INPUT_VIEW_ID,
        item_ids=simfile_ids,
        source_audio=source_audio_sha256,
        input_audio=input_audio_sha256,
    )
    muscriptor_snapshot = strict_json_loads(muscriptor_run.read_bytes(), require_canonical=True)
    assert isinstance(muscriptor_snapshot, dict)
    muscriptor_snapshot.update(
        {
            "reference_manifest_sha256": reference.manifest_sha256,
            "reference_manifest_version": reference.corpus_version,
            "reference_timing_manifest_sha256": timing.manifest_sha256,
            "reference_timing_version": timing.corpus_version,
        }
    )
    muscriptor_run.write_bytes(render_muscriptor_corpus_run(muscriptor_snapshot))
    shutil.copytree(full_mix_reports, muscriptor_root / "reports")
    _rewrite_report_identity(
        muscriptor_root / "reports",
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        identity={
            "cohort_id": "muscriptor-real-fixture",
            "model_id": "muscriptor-model",
            "model_lock_sha256": "f" * 64,
            "backend_descriptor_sha256": "a" * 64,
            "prediction_map_version": "muscriptor-map",
            "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
            "backend_id": "muscriptor-v0.3.0-drums-v1",
        },
    )

    idm_root = tmp_path / "idm-output"
    idm_payload = strict_json_loads(idm_snapshot(), require_canonical=True)
    assert isinstance(idm_payload, dict)
    idm_payload.update(
        {
            "reference_manifest_sha256": reference.manifest_sha256,
            "reference_manifest_version": reference.corpus_version,
            "reference_timing_manifest_sha256": timing.manifest_sha256,
            "reference_timing_version": timing.corpus_version,
            "oaf_model_id": oaf_model_id,
            "oaf_model_lock_sha256": oaf_model_lock_sha256,
            "oaf_backend_descriptor_sha256": oaf_descriptor_sha256,
            "oaf_prediction_map_version": oaf_prediction_map,
        }
    )
    idm_payload["items"][0]["prediction_artifact_sha256"] = sha256_hex(_idm_prediction_bytes())
    idm_payload["run_id"] = build_run_id(
        idm_payload["handoff_manifest_sha256"],
        idm_payload["handoff_manifest_version"],
        idm_payload["reference_manifest_sha256"],
        idm_payload["reference_manifest_version"],
        idm_payload["reference_timing_manifest_sha256"],
        idm_payload["reference_timing_version"],
        idm_payload["backend_descriptor_sha256"],
        idm_payload["model_lock_sha256"],
        idm_payload["inference_config_sha256"],
        idm_payload["input_view_id"],
        idm_payload.get("crux_commit"),
    )
    idm_run = idm_root / "runs" / str(idm_payload["run_id"]) / "run.json"
    idm_run.parent.mkdir(parents=True, exist_ok=True)
    idm_run.write_bytes(render_idm_pilot_run(idm_payload))
    write_idm_reports(idm_run.parent / "reports" / "oaf", label="oaf", run_id=idm_payload["run_id"])
    write_idm_reports(idm_run.parent / "reports" / "idm", label="idm", run_id=idm_payload["run_id"])
    _rewrite_report_identity(
        idm_run.parent / "reports" / "oaf",
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        identity={
            "model_id": oaf_model_id,
            "model_lock_sha256": oaf_model_lock_sha256,
            "backend_descriptor_sha256": oaf_descriptor_sha256,
            "prediction_map_version": oaf_prediction_map,
        },
    )
    _rewrite_report_identity(
        idm_run.parent / "reports" / "idm",
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        identity={},
    )
    prediction_path = idm_root / "predictions" / "1.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_bytes(_idm_prediction_bytes())

    return cross_comparison.CrossComparisonRequest(
        oaf_run_path=fixture.run_path,
        muscriptor_run_path=muscriptor_run,
        separation_run_path=separation_outcome.run_path,
        idm_run_path=idm_run,
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        subset_manifest_path=subset_path,
        output_dir=tmp_path / "published-real",
        separation_cache_dir=pilot_root / "cache",
    )


def test_publish_cross_comparisons_executes_real_driver_summary_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _build_real_driver_request(tmp_path, monkeypatch)

    outcome = cross_comparison.publish_cross_comparisons(request)

    assert outcome.pairable_success_counts.keys() == {
        "oaf_muscriptor_full_mix",
        "oaf_separation_pilot.spleeter",
        "oaf_separation_pilot.htdemucs",
        "oaf_idm_htdemucs",
    }
    for comparison_id, schema in (
        ("oaf_muscriptor_full_mix", cross_comparison.COMPARISON_SCHEMA),
        ("oaf_separation_pilot", cross_comparison.SEPARATION_COMPARISON_SCHEMA),
        ("oaf_idm_htdemucs", cross_comparison.IDM_COMPARISON_SCHEMA),
    ):
        nested = json.loads((outcome.comparison_paths[comparison_id] / "summary.json").read_text())
        assert nested["schema"] == schema
        assert nested["models"]
