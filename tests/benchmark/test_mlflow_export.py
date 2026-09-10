"""Tests for the pure deterministic MLflow projection of cohort reports."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.artifact_io import PublishedArtifact
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    canonical_json_bytes,
    sha256_hex,
)
from src.benchmark.cohort_scoring import CohortIdentity, PopulationSummary
from src.benchmark.mlflow_export import (
    PROJECTION_VERSION,
    REPORT_ARTIFACT_NAMES,
    REQUIRED_BASELINE_VIEWS,
    MlflowProjectionError,
    build_mlflow_projection,
)
from src.benchmark.reports import (
    PublishedAggregate,
    PublishedAggregateClass,
    PublishedCohortReports,
    PublishedF1Distribution,
    PublishedMetric,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

SIX_COMMON_CLASSES = ("kick", "snare", "hihat", "crash", "ride", "tom")
EXTRA_ARTIFACT_NAME = "event_diagnostics.jsonl"
ARTIFACT_BYTES = {
    "summary.json": b'{"schema":"crux.single-cohort-report/v1"}\n',
    "summary.md": b"# Single-Cohort Drum Benchmark\n",
    "items.csv": b"cohort_id,simfile_id\n",
    "per_song.csv": b"cohort_id,simfile_id\n",
    "per_class.csv": b"cohort_id,simfile_id\n",
    EXTRA_ARTIFACT_NAME: b"{}\n",
}


def _identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version="crux.prediction-map/oaf-egmd-8hit-v1",
        input_view_id="full-mix-v1",
    )


def _metric(precision: str = "0.95") -> PublishedMetric:
    return PublishedMetric(
        true_positives=10,
        false_positives=2,
        false_negatives=1,
        precision=Decimal(precision),
        recall=Decimal("0.90"),
        f1=Decimal("0.925"),
    )


def _class_row(common_class: str, precision: str = "0.95") -> PublishedAggregateClass:
    return PublishedAggregateClass(
        common_class=common_class,
        metric=_metric(precision),
        reference_support=12,
        prediction_support=10,
    )


def _aggregate(tolerance_ms: int, mode: str, *, precision: str = "0.95") -> PublishedAggregate:
    # The 30ms raw view carries undefined optional metrics so the tests can pin
    # that ``None`` values are omitted instead of becoming zeros.
    sparse = (tolerance_ms, mode) == (30, "raw")
    per_class = (
        [_class_row(common, precision) for common in SIX_COMMON_CLASSES]
        if (tolerance_ms, mode) == (50, "aligned")
        else [_class_row("kick", precision)]
    )
    return PublishedAggregate(
        tolerance_ms=tolerance_ms,
        mode=mode,  # type: ignore[arg-type]
        event_micro=PublishedMetric(
            true_positives=10,
            false_positives=2,
            false_negatives=1,
            precision=Decimal(precision),
            recall=Decimal("0.90"),
            f1=None if sparse else Decimal("0.925"),
        ),
        song_macro_f1=None if sparse else Decimal("0.88"),
        class_macro_f1=Decimal("0.86"),
        song_f1_distribution=PublishedF1Distribution(
            minimum=Decimal("0.5"),
            p10=None if sparse else Decimal("0.6"),
            p25=Decimal("0.7"),
            median=Decimal("0.8"),
            p75=Decimal("0.9"),
            p90=Decimal("0.95"),
            maximum=Decimal("1.0"),
        ),
        per_class=tuple(per_class),
        successful_song_count=3,
    )


def _reports(extra_tolerance_ms: int | None = None, *, precision: str = "0.95"):
    tolerances_ms = [30, 50, 100]
    if extra_tolerance_ms is not None:
        tolerances_ms.append(extra_tolerance_ms)
    return PublishedCohortReports(
        identity=_identity(),
        population=PopulationSummary(
            total_count=5,
            success_count=3,
            failed_count=1,
            skipped_count=1,
            quarantined_count=0,
            reason_counts=(("prediction_missing", 1),),
        ),
        aggregates=tuple(
            _aggregate(tolerance_ms, mode, precision=precision)
            for tolerance_ms in tolerances_ms
            for mode in ("raw", "aligned")
        ),
        items=(),
        songs=(),
        classes=(),
    )


def _report_dir(tmp_path: Path, *, write_extra: bool = True) -> Path:
    for name, content in ARTIFACT_BYTES.items():
        if name == EXTRA_ARTIFACT_NAME and not write_extra:
            continue
        (tmp_path / name).write_bytes(content)
    return tmp_path


def test_required_baseline_views_are_the_six_v1_views() -> None:
    assert REQUIRED_BASELINE_VIEWS == frozenset(
        {
            (30, "raw"),
            (30, "aligned"),
            (50, "raw"),
            (50, "aligned"),
            (100, "raw"),
            (100, "aligned"),
        }
    )


def test_projection_covers_every_metric_family(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="broad")
    metrics = projection.metrics
    for name, value in (
        ("event_micro.precision.50ms.aligned", "0.95"),
        ("event_micro.recall.50ms.aligned", "0.90"),
        ("event_micro.f1.50ms.aligned", "0.925"),
        ("song_macro.f1.50ms.aligned", "0.88"),
        ("class_macro.f1.50ms.aligned", "0.86"),
        ("song_f1.minimum.50ms.aligned", "0.5"),
        ("song_f1.p10.50ms.aligned", "0.6"),
        ("song_f1.p25.50ms.aligned", "0.7"),
        ("song_f1.median.50ms.aligned", "0.8"),
        ("song_f1.p75.50ms.aligned", "0.9"),
        ("song_f1.p90.50ms.aligned", "0.95"),
        ("song_f1.maximum.50ms.aligned", "1.0"),
    ):
        assert metrics[name] == Decimal(value), name
    for common in SIX_COMMON_CLASSES:
        assert metrics[f"class.{common}.precision.50ms.aligned"] == Decimal("0.95")
        assert metrics[f"class.{common}.recall.50ms.aligned"] == Decimal("0.90")
        assert metrics[f"class.{common}.f1.50ms.aligned"] == Decimal("0.925")
        assert metrics[f"class.{common}.reference_support.50ms.aligned"] == 12
        assert metrics[f"class.{common}.prediction_support.50ms.aligned"] == 10
    assert metrics["population.total"] == 5
    assert metrics["population.success"] == 3
    assert metrics["population.failed"] == 1
    assert metrics["population.skipped"] == 1
    assert metrics["population.quarantined"] == 0
    assert metrics["population.reason.prediction_missing"] == 1


def test_projection_omits_undefined_metrics(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="broad")
    metrics = projection.metrics
    assert "event_micro.f1.30ms.raw" not in metrics
    assert "song_macro.f1.30ms.raw" not in metrics
    assert "song_f1.p10.30ms.raw" not in metrics
    assert metrics["event_micro.precision.30ms.raw"] == Decimal("0.95")


def test_projection_projects_extra_tolerance_views(tmp_path: Path) -> None:
    projection = build_mlflow_projection(
        _reports(extra_tolerance_ms=200), _report_dir(tmp_path), scope="broad"
    )
    assert projection.metrics["event_micro.precision.200ms.raw"] == Decimal("0.95")
    assert projection.metrics["event_micro.precision.200ms.aligned"] == Decimal("0.95")


def test_projection_requires_all_six_baseline_views(tmp_path: Path) -> None:
    reports = _reports()
    aggregates = tuple(
        row for row in reports.aggregates if (row.tolerance_ms, row.mode) != (100, "aligned")
    )
    with pytest.raises(MlflowProjectionError, match="required benchmark views are missing"):
        build_mlflow_projection(
            replace(reports, aggregates=aggregates), _report_dir(tmp_path), scope="broad"
        )


def test_projection_rejects_unknown_common_class(tmp_path: Path) -> None:
    reports = _reports()
    extra = PublishedAggregateClass("cowbell", _metric(), 1, 1)
    aggregates = tuple(
        replace(row, per_class=row.per_class + (extra,))
        if (row.tolerance_ms, row.mode) == (50, "aligned")
        else row
        for row in reports.aggregates
    )
    with pytest.raises(MlflowProjectionError, match="cowbell"):
        build_mlflow_projection(
            replace(reports, aggregates=aggregates), _report_dir(tmp_path), scope="broad"
        )


def test_projection_artifacts_cover_only_allowlisted_files(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert [artifact.path.name for artifact in projection.artifacts] == list(REPORT_ARTIFACT_NAMES)
    for artifact in projection.artifacts:
        assert isinstance(artifact, PublishedArtifact)
        assert artifact.path == report_dir / artifact.path.name
        assert artifact.sha256 == sha256_hex(ARTIFACT_BYTES[artifact.path.name])
    assert EXTRA_ARTIFACT_NAME not in {artifact.path.name for artifact in projection.artifacts}


def test_projection_wraps_unreadable_artifact_error(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    (report_dir / "summary.md").unlink()
    with pytest.raises(MlflowProjectionError, match="summary.md"):
        build_mlflow_projection(_reports(), report_dir, scope="broad")


def test_projection_is_stable_for_same_input(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    first = build_mlflow_projection(_reports(), report_dir, scope="broad")
    second = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert first == second


def test_fingerprint_changes_with_scope(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    broad = build_mlflow_projection(_reports(), report_dir, scope="broad")
    reviewed = build_mlflow_projection(_reports(), report_dir, scope="reviewed")
    assert broad.tags["crux.projection_sha256"] != reviewed.tags["crux.projection_sha256"]


def test_fingerprint_changes_with_metric_value(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    baseline = build_mlflow_projection(_reports(), report_dir, scope="broad")
    changed = build_mlflow_projection(_reports(precision="0.80"), report_dir, scope="broad")
    assert baseline.tags["crux.projection_sha256"] != changed.tags["crux.projection_sha256"]


def test_fingerprint_changes_with_artifact_bytes(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    baseline = build_mlflow_projection(_reports(), report_dir, scope="broad")
    (report_dir / "summary.md").write_bytes(b"# Changed\n")
    changed = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert baseline.tags["crux.projection_sha256"] != changed.tags["crux.projection_sha256"]


def test_projection_sha256_hashes_base_payload(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    base_tags = {
        name: value for name, value in projection.tags.items() if name != "crux.projection_sha256"
    }
    payload = {
        "projection_version": PROJECTION_VERSION,
        "scope": "broad",
        "run_name": projection.run_name,
        "tags": base_tags,
        "metrics": dict(projection.metrics),
        "artifacts": [
            {"path": artifact.path.name, "sha256": artifact.sha256}
            for artifact in projection.artifacts
        ],
    }
    assert projection.tags["crux.projection_sha256"] == sha256_hex(canonical_json_bytes(payload))


def test_projection_tags_carry_scope_and_identity(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="pilot")
    identity = _identity()
    assert projection.tags["crux.projection_version"] == PROJECTION_VERSION
    assert projection.tags["crux.scope"] == "pilot"
    assert projection.tags["crux.cohort_id"] == identity.cohort_id
    assert projection.tags["crux.backend_id"] == identity.backend_id
    assert projection.tags["crux.model_id"] == identity.model_id
    assert projection.tags["crux.model_lock_sha256"] == identity.model_lock_sha256
    assert projection.tags["crux.backend_descriptor_sha256"] == identity.backend_descriptor_sha256
    assert projection.tags["crux.input_view_id"] == identity.input_view_id
    assert projection.tags["crux.reference_manifest_sha256"] == identity.reference_manifest_sha256
    assert projection.tags["crux.reference_timing_version"] == identity.reference_timing_version
    assert projection.tags["crux.taxonomy_version"] == identity.taxonomy_version
    assert projection.tags["crux.lane_map_version"] == identity.lane_map_version
    assert projection.tags["crux.prediction_map_version"] == identity.prediction_map_version
    assert projection.tags["crux.scoring_version"] == identity.scoring_version


def test_run_name_is_deterministic_from_scope_model_view_and_cohort(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    identity = _identity()
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    again = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert projection.run_name == again.run_name
    assert projection.run_name == (
        f"broad-{identity.model_id}-{identity.input_view_id}-{identity.cohort_id}"
    )


def test_import_does_not_import_mlflow() -> None:
    code = "\n".join(
        [
            "import sys",
            "import src.benchmark.mlflow_export",
            "loaded = [n for n in sys.modules if n == 'mlflow' or n.startswith('mlflow.')]",
            "assert not loaded, loaded",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
