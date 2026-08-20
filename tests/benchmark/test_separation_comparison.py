"""Focused HPA-328 published separation-comparison coverage."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.backend_identity import OAF_BACKEND_ID, canonical_json_bytes
from src.benchmark.cohort_scoring import CohortIdentity
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.published_comparison import (
    ComparisonIntegrityError,
    PublishedRunEvidence,
    PublishedRunItem,
    pairable_success_ids,
)
from src.benchmark.separation_comparison import (
    SeparationComparisonOutcome,
    SeparationComparisonRequest,
    _artifact_bytes,
    _durations,
    _expected_cohort_id,
    _failure_histogram,
    _finite_seconds,
    _resolve_artifact,
    _resources,
    _status,
    _strict_report_identity,
    _validated_identity,
    _view_evidence,
    aggregate_paired_event_micro,
    compare_oaf_separation,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION


def _song(
    simfile_id: str,
    *,
    tp: int,
    fp: int,
    fn: int,
    f1: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        simfile_id=simfile_id,
        tolerance_ms=50,
        mode="raw",
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=Decimal("0.5"),
        recall=Decimal("0.5"),
        f1=Decimal(f1),
    )


def test_paired_event_micro_sums_published_counts_and_uses_authoritative_minutes() -> None:
    left = SimpleNamespace(
        songs=(_song("1", tp=1, fp=1, fn=3, f1="0.2"), _song("2", tp=3, fp=0, fn=1, f1="0.75")),
    )
    right = SimpleNamespace(
        songs=(_song("1", tp=2, fp=0, fn=2, f1="0.5"), _song("2", tp=4, fp=1, fn=0, f1="0.8")),
    )

    rows = aggregate_paired_event_micro(
        left,
        right,
        {"1", "2"},
        {"1": Decimal("60"), "2": Decimal("120")},
        left_label="full_mix",
        right_label="spleeter",
    )

    row = rows[0]
    assert row["full_mix"]["tp"] == 4
    assert row["full_mix"]["fp"] == 1
    assert row["full_mix"]["fn"] == 4
    assert row["full_mix"]["f1"] == Decimal("0.615385")
    assert row["spleeter"]["tp"] == 6
    assert row["spleeter"]["fp"] == 1
    assert row["spleeter"]["fn"] == 2
    assert row["spleeter"]["f1"] == Decimal("0.8")
    assert row["spleeter"]["fp_per_minute"] == Decimal("0.333333")
    assert row["spleeter"]["fn_per_minute"] == Decimal("0.666667")


@pytest.mark.parametrize(
    "source_durations",
    (
        {"1": Decimal("60")},
        {"1": Decimal("60"), "2": None},
        {"1": Decimal("60"), "2": Decimal("NaN")},
        {"1": Decimal("60"), "2": Decimal("0")},
        {"1": Decimal("60"), "2": Decimal("-1")},
    ),
    ids=("missing", "null", "nonfinite", "zero", "negative"),
)
def test_paired_event_micro_rejects_pairable_song_without_positive_finite_duration(
    source_durations: dict[str, object],
) -> None:
    reports = SimpleNamespace(
        songs=(_song("1", tp=1, fp=0, fn=0, f1="1"), _song("2", tp=1, fp=0, fn=0, f1="1")),
    )

    with pytest.raises(
        ComparisonIntegrityError,
        match="positive finite authoritative duration",
    ):
        aggregate_paired_event_micro(
            reports,
            reports,
            {"1", "2"},
            source_durations,
        )


def test_separation_pairing_allows_distinct_derived_input_hashes() -> None:
    left = PublishedRunEvidence(
        identity=SimpleNamespace(),
        items={"1": PublishedRunItem("1", "success", "a" * 64, "c" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )
    right = PublishedRunEvidence(
        identity=SimpleNamespace(),
        items={"1": PublishedRunItem("1", "success", "a" * 64, "d" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )

    pairable, exclusions = pairable_success_ids(
        left,
        right,
        None,
        require_identical_input_hash=False,
        left_label="full_mix",
        right_label="spleeter",
    )

    assert pairable == {"1"}
    assert exclusions["source_audio_mismatch"] == 0


def test_pilot_comparison_call_uses_run_scoped_output_after_scoring(
    tmp_path,
    monkeypatch,
) -> None:
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _request,
        _subset_path,
        _task6_seams,
    )

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    comparison_requests = []

    def fake_compare(comparison_request):
        comparison_requests.append(comparison_request)
        return None

    import src.benchmark.separation_pilot as pilot

    monkeypatch.setattr(pilot, "compare_oaf_separation", fake_compare)
    monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 0
    assert len(comparison_requests) == 1
    comparison_request = comparison_requests[0]
    assert comparison_request.run_path == outcome.run_path
    assert comparison_request.output_dir == outcome.run_path.parent / "comparison"
    assert comparison_request.subset_manifest_path == subset


def test_comparison_publishes_paired_csvs_summary_and_native_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _request,
        _subset_path,
        _task6_seams,
    )

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    import src.benchmark.separation_pilot as pilot
    from src.benchmark.reviewed_subset import score_oaf_reviewed_subset

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", score_oaf_reviewed_subset)

    from dataclasses import replace

    base_factory = calls["factory"][0]

    def aligned_backend_factory(**kwargs):
        backend = base_factory(**kwargs)

        class AlignedBackend:
            def descriptor(self):
                return backend.descriptor()

            def transcribe(self, audio):
                prediction = backend.transcribe(audio)
                return replace(
                    prediction,
                    events=tuple(replace(event, time_sec=0.5) for event in prediction.events),
                )

            def close(self):
                return backend.close()

        return AlignedBackend()

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=aligned_backend_factory,
    )

    assert outcome.exit_code == 0
    assert outcome.run_path is not None
    comparison_dir = outcome.run_path.parent / "comparison"
    expected_files = {
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
        "summary.json",
        "summary.md",
    }
    actual_files = {
        path.relative_to(comparison_dir).as_posix()
        for path in comparison_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files

    import json

    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "crux.oaf-separation-comparison/v1"
    for view_name in ("spleeter", "htdemucs"):
        assert summary["pairing"][view_name]["pairable_success_intersection"] == 20
        assert summary["models"][view_name]["failure_code_histogram"] == {}
        assert summary["models"][view_name]["reason_counts"] == {}
        resources = summary["models"][view_name]["resources"]
        assert resources["retained_stem_bytes"] > 0
        assert resources["retained_prediction_bytes"] > 0
        assert resources["retained_report_bytes"] > 0
        assert summary["comparisons"][view_name]["event_micro"]
    assert "cost" not in summary
    assert "reason_counts" in (comparison_dir / "summary.md").read_text(encoding="utf-8")


def _identity(
    *,
    cohort_id: str = "cohort-1",
    backend_id: str = OAF_BACKEND_ID,
    model_id: str = "magenta-egmd-ckpt-569400-v1",
    input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID,
) -> CohortIdentity:
    return CohortIdentity(
        cohort_id=cohort_id,
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=backend_id,
        model_id=model_id,
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version="crux.prediction-map/oaf-egmd-8hit-v1",
        input_view_id=input_view_id,
    )


def _identity_payload(identity: CohortIdentity) -> dict[str, object]:
    fields = (
        "cohort_id",
        "reference_manifest_sha256",
        "reference_timing_version",
        "taxonomy_version",
        "lane_map_version",
        "backend_id",
        "model_id",
        "model_lock_sha256",
        "backend_descriptor_sha256",
        "prediction_map_version",
        "input_view_id",
        "scoring_version",
    )
    return {field: getattr(identity, field) for field in fields}


def _write_summary(report_dir: Path, identity: CohortIdentity) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_bytes(
        canonical_json_bytes({"identity": _identity_payload(identity)})
    )


def _manifest(manifest_sha256: str = "a" * 64, corpus_version: str = "sha256:" + "b" * 64):
    return SimpleNamespace(
        manifest_sha256=manifest_sha256,
        corpus_version=corpus_version,
    )


def test_request_post_init_rejects_non_path_fields() -> None:
    with pytest.raises(TypeError, match="run_path must be a Path"):
        SeparationComparisonRequest(
            run_path="not a path",  # type: ignore[arg-type]
            reference_manifest_path=Path("."),
            timing_manifest_path=Path("."),
            subset_manifest_path=Path("."),
            output_dir=Path("."),
        )


def test_request_post_init_rejects_non_path_cache_dir() -> None:
    with pytest.raises(TypeError, match="cache_dir must be a Path or None"):
        SeparationComparisonRequest(
            run_path=Path("."),
            reference_manifest_path=Path("."),
            timing_manifest_path=Path("."),
            subset_manifest_path=Path("."),
            output_dir=Path("."),
            cache_dir="not a path",  # type: ignore[arg-type]
        )


def test_outcome_post_init_rejects_non_path_output_dir() -> None:
    with pytest.raises(TypeError, match="output_dir must be a Path"):
        SeparationComparisonOutcome(
            output_dir="not a path",  # type: ignore[arg-type]
            pairable_success_counts={},
            paired_song_counts={},
            paired_class_counts={},
        )


def test_outcome_post_init_rejects_non_dict_counts() -> None:
    with pytest.raises(TypeError, match="pairable_success_counts must be a dict"):
        SeparationComparisonOutcome(
            output_dir=Path("."),
            pairable_success_counts=[],  # type: ignore[arg-type]
            paired_song_counts={},
            paired_class_counts={},
        )


def test_outcome_post_init_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="contains an invalid count"):
        SeparationComparisonOutcome(
            output_dir=Path("."),
            pairable_success_counts={"spleeter": -1},
            paired_song_counts={},
            paired_class_counts={},
        )


def test_outcome_post_init_rejects_boolean_count() -> None:
    with pytest.raises(ValueError, match="contains an invalid count"):
        SeparationComparisonOutcome(
            output_dir=Path("."),
            pairable_success_counts={},
            paired_song_counts={"spleeter": True},  # type: ignore[arg-type]
            paired_class_counts={},
        )


def test_aggregate_paired_event_micro_rejects_empty_left_label() -> None:
    reports = SimpleNamespace(songs=(_song("1", tp=1, fp=0, fn=0, f1="1"),))
    with pytest.raises(ValueError, match="left_label must be a nonempty string"):
        aggregate_paired_event_micro(reports, reports, {"1"}, {"1": Decimal("60")}, left_label="")


def test_aggregate_paired_event_micro_rejects_empty_right_label() -> None:
    reports = SimpleNamespace(songs=(_song("1", tp=1, fp=0, fn=0, f1="1"),))
    with pytest.raises(ValueError, match="right_label must be a nonempty string"):
        aggregate_paired_event_micro(reports, reports, {"1"}, {"1": Decimal("60")}, right_label="")


def test_aggregate_paired_event_micro_rejects_score_key_grid_mismatch() -> None:
    left = SimpleNamespace(songs=(_song("1", tp=1, fp=0, fn=0, f1="1"),))
    right = SimpleNamespace(
        songs=(_song("1", tp=1, fp=0, fn=0, f1="1"), _song("2", tp=1, fp=0, fn=0, f1="1")),
    )
    with pytest.raises(ComparisonIntegrityError, match="score key grid mismatch"):
        aggregate_paired_event_micro(
            left,
            right,
            {"1", "2"},
            {"1": Decimal("60"), "2": Decimal("60")},
        )


def test_strict_report_identity_rejects_missing_summary(tmp_path: Path) -> None:
    with pytest.raises(ComparisonIntegrityError, match="cannot read HPA-325 identity"):
        _strict_report_identity(tmp_path)


def test_strict_report_identity_rejects_non_canonical_summary(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"identity": {"a": 1}}\n', encoding="utf-8")
    with pytest.raises(ComparisonIntegrityError, match="cannot read HPA-325 identity"):
        _strict_report_identity(tmp_path)


def test_strict_report_identity_rejects_malformed_identity_mapping(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_bytes(canonical_json_bytes({"identity": "not a mapping"}))
    with pytest.raises(ComparisonIntegrityError, match="summary identity is malformed"):
        _strict_report_identity(tmp_path)


def test_strict_report_identity_rejects_unconstructable_identity(tmp_path: Path) -> None:
    payload = _identity_payload(_identity())
    payload["backend_id"] = ""
    (tmp_path / "summary.json").write_bytes(canonical_json_bytes({"identity": payload}))
    with pytest.raises(ComparisonIntegrityError, match="summary identity is malformed"):
        _strict_report_identity(tmp_path)


def test_expected_cohort_id_rejects_missing_parent_run_id() -> None:
    snapshot = {"parent_oaf_run_id": ""}
    with pytest.raises(ComparisonIntegrityError, match="parent OaF run identity is unavailable"):
        _expected_cohort_id(snapshot, _manifest(), "full_mix")


def test_expected_cohort_id_rejects_missing_separation_run_id() -> None:
    snapshot = {"run_id": ""}
    with pytest.raises(ComparisonIntegrityError, match="separation run identity is unavailable"):
        _expected_cohort_id(snapshot, _manifest(), "spleeter")


def test_validated_identity_rejects_non_oaf_backend(tmp_path: Path) -> None:
    identity = _identity(backend_id="other-backend")
    _write_summary(tmp_path, identity)
    snapshot = {
        "parent_oaf_run_id": "parent-run-1",
        "oaf_model_lock_sha256": "c" * 64,
        "oaf_backend_descriptor_sha256": "d" * 64,
    }
    with pytest.raises(ComparisonIntegrityError, match="requires OaF reports"):
        _validated_identity(tmp_path, snapshot, _manifest(), _manifest(), _manifest(), "full_mix")


def test_validated_identity_rejects_field_mismatch(tmp_path: Path) -> None:
    from hashlib import sha256

    expected_cohort_id = sha256(
        canonical_json_bytes(
            {"parent_run_id": "parent-run-1", "reviewed_subset_manifest_sha256": "a" * 64}
        )
    ).hexdigest()
    identity = _identity(cohort_id=expected_cohort_id)
    _write_summary(tmp_path, identity)
    snapshot = {
        "parent_oaf_run_id": "parent-run-1",
        "oaf_model_lock_sha256": "z" * 64,
        "oaf_backend_descriptor_sha256": "d" * 64,
    }
    with pytest.raises(ComparisonIntegrityError, match="identity mismatch for model_lock_sha256"):
        _validated_identity(tmp_path, snapshot, _manifest(), _manifest(), _manifest(), "full_mix")


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("inferred", "success"),
        ("resumed", "success"),
        ("success", "success"),
        ("failed", "failed"),
        ("skipped", "skipped"),
        ("quarantined", "quarantined"),
        ("unknown", "failed"),
    ),
)
def test_status_maps_known_and_unknown_values(value: str, expected: str) -> None:
    assert _status(value) == expected


def _reports(items: dict[str, PublishedRunItem]) -> SimpleNamespace:
    return SimpleNamespace(identity=_identity(), items=tuple(items.values()))


def test_view_evidence_rejects_non_list_items() -> None:
    snapshot = {"items": "not a list"}
    with pytest.raises(ComparisonIntegrityError, match="run items are unavailable"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_malformed_item() -> None:
    snapshot = {"items": ["not a mapping"]}
    with pytest.raises(ComparisonIntegrityError, match="run item is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_malformed_simfile_id() -> None:
    snapshot = {"items": [{"spleeter": {"status": "success"}, "simfile_id": True}]}
    with pytest.raises(ComparisonIntegrityError, match="simfile_id is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_non_positive_simfile_id() -> None:
    snapshot = {"items": [{"spleeter": {"status": "success"}, "simfile_id": 0}]}
    with pytest.raises(ComparisonIntegrityError, match="simfile_id is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_malformed_view_mapping() -> None:
    snapshot = {"items": [{"spleeter": "not a mapping", "simfile_id": 1}]}
    with pytest.raises(ComparisonIntegrityError, match="spleeter run evidence is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_malformed_source_hash() -> None:
    snapshot = {
        "items": [{"spleeter": {"status": "success"}, "simfile_id": 1, "source_audio_sha256": 123}]
    }
    with pytest.raises(ComparisonIntegrityError, match="source_audio_sha256 is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_malformed_input_hash() -> None:
    snapshot = {
        "items": [
            {
                "spleeter": {"status": "success", "input": {"input_audio_sha256": 123}},
                "simfile_id": 1,
            }
        ]
    }
    with pytest.raises(ComparisonIntegrityError, match="input_audio_sha256 is malformed"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_population_mismatch() -> None:
    snapshot = {"items": [{"spleeter": {"status": "success"}, "simfile_id": 1}]}
    with pytest.raises(ComparisonIntegrityError, match="report population does not match"):
        _view_evidence(snapshot, _reports({}), "spleeter")


def test_view_evidence_rejects_status_mismatch() -> None:
    snapshot = {"items": [{"spleeter": {"status": "failed"}, "simfile_id": 1}]}
    reports = _reports({"1": PublishedRunItem("1", "success", None, None)})
    with pytest.raises(ComparisonIntegrityError, match="report status does not match"):
        _view_evidence(snapshot, reports, "spleeter")


def test_durations_rejects_non_list_items() -> None:
    with pytest.raises(ComparisonIntegrityError, match="run items are unavailable"):
        _durations({"items": "not a list"})


def test_durations_skips_non_mapping_items() -> None:
    snapshot = {"items": ["not a mapping", {"simfile_id": 1, "source_duration_sec": "60"}]}
    assert _durations(snapshot) == {"1": Decimal("60")}


def test_finite_seconds_handles_invalid_and_nonfinite_values() -> None:
    assert _finite_seconds("not a number") == Decimal(0)
    assert _finite_seconds("NaN") == Decimal(0)
    assert _finite_seconds("-1") == Decimal(0)
    assert _finite_seconds("60") == Decimal("60")


def test_resolve_artifact_rejects_non_string_path() -> None:
    with pytest.raises(ComparisonIntegrityError, match="retained artifact path is unavailable"):
        _resolve_artifact(123, roots=(Path("."),))  # type: ignore[arg-type]


def test_resolve_artifact_resolves_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "stem.wav"
    target.write_bytes(b"data")
    assert _resolve_artifact(str(target), roots=(tmp_path,)) == target


def test_resolve_artifact_rejects_unresolved_relative_path() -> None:
    with pytest.raises(ComparisonIntegrityError, match="retained artifact is unresolved"):
        _resolve_artifact("missing.wav", roots=(Path("/nonexistent-root"),))


def test_artifact_bytes_rejects_unreadable_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(ComparisonIntegrityError, match="retained artifact is unreadable"):
        _artifact_bytes(missing)


def test_artifact_bytes_rejects_non_regular_file(tmp_path: Path) -> None:
    link = tmp_path / "link.wav"
    target = tmp_path / "target.wav"
    target.write_bytes(b"data")
    link.symlink_to(target)
    with pytest.raises(ComparisonIntegrityError, match="retained artifact is not a regular file"):
        _artifact_bytes(link)


def test_resources_rejects_non_list_items(tmp_path: Path) -> None:
    snapshot = {"items": "not a list"}
    with pytest.raises(ComparisonIntegrityError, match="run items are unavailable"):
        _resources(snapshot, tmp_path, None, "spleeter", tmp_path)


def _write_report_files(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.json",
        "items.csv",
        "per_song.csv",
        "per_class.csv",
        "event_diagnostics.jsonl",
        "summary.md",
    ):
        (report_dir / name).write_bytes(b"x")


def test_resources_skips_non_mapping_items_and_views(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_report_files(report_dir)
    snapshot = {
        "items": [
            "not a mapping",
            {"spleeter": "not a mapping", "simfile_id": 1},
            {"spleeter": {"runtime": {"separator_wall_time_sec": "10"}}, "simfile_id": 2},
        ]
    }
    result = _resources(snapshot, tmp_path, None, "spleeter", report_dir)
    assert result["separator_wall_time_sec"] == Decimal("10.000000")
    assert result["oaf_wall_time_sec"] == Decimal("0.000000")


def test_resources_uses_inference_elapsed_seconds_fallback(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    _write_report_files(report_dir)
    snapshot = {
        "items": [
            {
                "spleeter": {"runtime": {"inference_elapsed_seconds": "5"}},
                "simfile_id": 1,
            }
        ]
    }
    result = _resources(snapshot, tmp_path, None, "spleeter", report_dir)
    assert result["oaf_wall_time_sec"] == Decimal("5.000000")


def test_failure_histogram_rejects_non_list_items() -> None:
    with pytest.raises(ComparisonIntegrityError, match="run items are unavailable"):
        _failure_histogram({"items": "not a list"}, "spleeter")


def test_failure_histogram_skips_non_mapping_items_and_views() -> None:
    snapshot = {
        "items": [
            "not a mapping",
            {"spleeter": "not a mapping", "simfile_id": 1},
            {"spleeter": {"failure_code": "boom"}, "simfile_id": 2},
            {"spleeter": {"failure_code": "boom"}, "simfile_id": 3},
            {"spleeter": {"failure_code": "kaput"}, "simfile_id": 4},
            {"spleeter": {"failure_code": ""}, "simfile_id": 5},
        ]
    }
    assert _failure_histogram(snapshot, "spleeter") == {"boom": 2, "kaput": 1}


def test_compare_oaf_separation_rejects_non_request() -> None:
    with pytest.raises(TypeError, match="request must be SeparationComparisonRequest"):
        compare_oaf_separation("not a request")  # type: ignore[arg-type]


def _run_pilot_for_comparison(tmp_path, monkeypatch):
    from src.benchmark.separation_comparison import SeparationComparisonRequest
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _request,
        _subset_path,
        _task6_seams,
    )

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    import src.benchmark.separation_pilot as pilot
    from src.benchmark.reviewed_subset import score_oaf_reviewed_subset

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", score_oaf_reviewed_subset)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(request, backend_factory=calls["factory"][0])
    assert outcome.exit_code == 0
    assert outcome.run_path is not None
    comparison_request = SeparationComparisonRequest(
        run_path=outcome.run_path,
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        subset_manifest_path=subset,
        output_dir=tmp_path / "comparison-out",
        cache_dir=tmp_path / "cache",
    )
    return comparison_request, outcome.run_path


def test_compare_oaf_separation_wraps_unparseable_run_snapshot(tmp_path, monkeypatch) -> None:
    request, run_path = _run_pilot_for_comparison(tmp_path, monkeypatch)
    run_path.write_bytes(b"not json")
    with pytest.raises(ComparisonIntegrityError, match="invalid JSON"):
        compare_oaf_separation(request)


def test_compare_oaf_separation_rejects_invalid_run_schema(tmp_path, monkeypatch) -> None:
    request, _ = _run_pilot_for_comparison(tmp_path, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    original = pilot.parse_oaf_separation_run

    def fake_parse(content, *, expected_run_id=None):
        snapshot = original(content, expected_run_id=expected_run_id)
        snapshot = dict(snapshot)
        snapshot["schema"] = "crux.wrong/v1"
        return snapshot

    monkeypatch.setattr(pilot, "parse_oaf_separation_run", fake_parse)
    with pytest.raises(ComparisonIntegrityError, match="run snapshot schema is invalid"):
        compare_oaf_separation(request)


def test_compare_oaf_separation_rejects_reference_manifest_lineage_mismatch(
    tmp_path, monkeypatch
) -> None:
    request, _ = _run_pilot_for_comparison(tmp_path, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    original = pilot.parse_oaf_separation_run

    def fake_parse(content, *, expected_run_id=None):
        snapshot = original(content, expected_run_id=expected_run_id)
        snapshot = dict(snapshot)
        snapshot["reference_manifest_sha256"] = "z" * 64
        return snapshot

    monkeypatch.setattr(pilot, "parse_oaf_separation_run", fake_parse)
    with pytest.raises(
        ComparisonIntegrityError, match="run/reference manifest lineage does not match"
    ):
        compare_oaf_separation(request)


def test_compare_oaf_separation_rejects_timing_manifest_lineage_mismatch(
    tmp_path, monkeypatch
) -> None:
    request, _ = _run_pilot_for_comparison(tmp_path, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    original = pilot.parse_oaf_separation_run

    def fake_parse(content, *, expected_run_id=None):
        snapshot = original(content, expected_run_id=expected_run_id)
        snapshot = dict(snapshot)
        snapshot["reference_timing_manifest_sha256"] = "z" * 64
        return snapshot

    monkeypatch.setattr(pilot, "parse_oaf_separation_run", fake_parse)
    with pytest.raises(
        ComparisonIntegrityError, match="run/timing manifest lineage does not match"
    ):
        compare_oaf_separation(request)


def test_compare_oaf_separation_rejects_subset_manifest_lineage_mismatch(
    tmp_path, monkeypatch
) -> None:
    request, _ = _run_pilot_for_comparison(tmp_path, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    original = pilot.parse_oaf_separation_run

    def fake_parse(content, *, expected_run_id=None):
        snapshot = original(content, expected_run_id=expected_run_id)
        snapshot = dict(snapshot)
        snapshot["reviewed_subset_manifest_sha256"] = "z" * 64
        return snapshot

    monkeypatch.setattr(pilot, "parse_oaf_separation_run", fake_parse)
    with pytest.raises(
        ComparisonIntegrityError, match="run/subset manifest lineage does not match"
    ):
        compare_oaf_separation(request)
