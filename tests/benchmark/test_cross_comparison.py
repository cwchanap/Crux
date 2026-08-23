from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.benchmark import cross_comparison
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.idm_pilot_run import IDM_STEM_INPUT_VIEW_ID
from src.benchmark.muscriptor_comparison import ComparisonOutcome
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.published_comparison import ComparisonIntegrityError
from src.benchmark.separation_comparison import (
    HTDEMUCS_INPUT_VIEW_ID,
    SeparationComparisonOutcome,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

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
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                },
                "muscriptor": {
                    "model_lock_sha256": "b" * 64,
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
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                },
                "spleeter": {
                    "model_lock_sha256": "c" * 64,
                    "input_view_id": "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
                },
                "htdemucs": {
                    "model_lock_sha256": "d" * 64,
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
                    "input_view_id": IDM_STEM_INPUT_VIEW_ID,
                },
                "idm": {
                    "model_lock_sha256": "e" * 64,
                    "input_view_id": IDM_STEM_INPUT_VIEW_ID,
                },
            },
        },
    }


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
    calls = _patch_drivers(monkeypatch, _summaries())

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


@pytest.mark.parametrize(
    "comparison_id,field",
    [("oaf_muscriptor_full_mix", "schema"), ("oaf_separation_pilot", "schema")],
)
def test_publish_cross_comparisons_rejects_noncanonical_or_wrong_schema(
    tmp_path: Path, monkeypatch, comparison_id: str, field: str
) -> None:
    summaries = _summaries()
    if comparison_id == "oaf_muscriptor_full_mix":
        summaries[comparison_id][field] = "wrong.schema/v0"
    else:
        summaries[comparison_id][field] = "wrong.schema/v0"
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
def test_publish_cross_comparisons_rejects_malformed_or_mismatched_oaf_lock(
    tmp_path: Path, monkeypatch, lock: str
) -> None:
    summaries = _summaries()
    summaries["oaf_separation_pilot"]["models"]["full_mix"]["model_lock_sha256"] = lock
    _patch_drivers(monkeypatch, summaries)

    with pytest.raises(ComparisonIntegrityError, match="model_lock_sha256"):
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
        ("oaf_separation_pilot", []),
        ("oaf_idm_htdemucs", []),
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
