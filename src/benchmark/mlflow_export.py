"""Pure deterministic projection of canonical cohort reports to MLflow data.

This module never imports ``mlflow``.  It projects one validated
``PublishedCohortReports`` plus its report directory into the complete
MLflow-ready payload (tags, run name, canonical metrics, artifact digests,
fingerprint); the publisher in a later task only converts it at the MLflow
API boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, get_args

from src.benchmark.artifact_io import PublishedArtifact, read_regular_file_no_follow
from src.benchmark.backend_identity import JsonValue, canonical_json_bytes, sha256_hex
from src.benchmark.cohort_scoring import CohortIdentity, PopulationSummary
from src.benchmark.reports import (
    PublishedAggregate,
    PublishedAggregateClass,
    PublishedCohortReports,
)
from src.benchmark.taxonomy import CommonDrumClass

PROJECTION_VERSION = "crux.mlflow-projection/v1"
REPORT_ARTIFACT_PATH = "crux-reports"
REPORT_ARTIFACT_NAMES = (
    "summary.json",
    "summary.md",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
)
REQUIRED_BASELINE_VIEWS = frozenset(
    {
        (30, "raw"),
        (30, "aligned"),
        (50, "raw"),
        (50, "aligned"),
        (100, "raw"),
        (100, "aligned"),
    }
)
COMMON_CLASSES = frozenset(get_args(CommonDrumClass))

MlflowScope: TypeAlias = Literal["broad", "reviewed", "pilot"]
MlflowMetricValue: TypeAlias = int | Decimal

_FINGERPRINT_TAG = "crux.projection_sha256"
_DISTRIBUTION_STATS = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")


class MlflowProjectionError(ValueError):
    """Raised when canonical cohort reports cannot be projected for MLflow."""


@dataclass(frozen=True)
class MlflowProjection:
    """One complete MLflow-ready projection of a published cohort report."""

    run_name: str
    tags: Mapping[str, str]
    metrics: Mapping[str, MlflowMetricValue]
    artifacts: tuple[PublishedArtifact, ...]


def _view_suffix(tolerance_ms: int, mode: str) -> str:
    return f"{tolerance_ms}ms.{mode}"


def _project_decimal_metrics(
    metrics: dict[str, MlflowMetricValue],
    prefix: str,
    suffix: str,
    values: Mapping[str, Decimal | None],
) -> None:
    for name, value in values.items():
        if value is not None:
            metrics[f"{prefix}.{name}.{suffix}"] = value


def _project_class_row(
    metrics: dict[str, MlflowMetricValue],
    row: PublishedAggregateClass,
    suffix: str,
) -> None:
    if row.common_class not in COMMON_CLASSES:
        raise MlflowProjectionError(f"unknown common class: {row.common_class}")
    _project_decimal_metrics(
        metrics,
        f"class.{row.common_class}",
        suffix,
        {
            "precision": row.metric.precision,
            "recall": row.metric.recall,
            "f1": row.metric.f1,
        },
    )
    metrics[f"class.{row.common_class}.reference_support.{suffix}"] = row.reference_support
    metrics[f"class.{row.common_class}.prediction_support.{suffix}"] = row.prediction_support


def _project_aggregate(
    metrics: dict[str, MlflowMetricValue],
    row: PublishedAggregate,
) -> None:
    suffix = _view_suffix(row.tolerance_ms, row.mode)
    _project_decimal_metrics(
        metrics,
        "event_micro",
        suffix,
        {name: getattr(row.event_micro, name) for name in ("precision", "recall", "f1")},
    )
    _project_decimal_metrics(metrics, "song_macro", suffix, {"f1": row.song_macro_f1})
    _project_decimal_metrics(metrics, "class_macro", suffix, {"f1": row.class_macro_f1})
    _project_decimal_metrics(
        metrics,
        "song_f1",
        suffix,
        {stat: getattr(row.song_f1_distribution, stat) for stat in _DISTRIBUTION_STATS},
    )
    for class_row in row.per_class:
        _project_class_row(metrics, class_row, suffix)


def _project_population(
    metrics: dict[str, MlflowMetricValue],
    population: PopulationSummary,
) -> None:
    metrics["population.total"] = population.total_count
    metrics["population.success"] = population.success_count
    metrics["population.failed"] = population.failed_count
    metrics["population.skipped"] = population.skipped_count
    metrics["population.quarantined"] = population.quarantined_count
    for reason, count in population.reason_counts:
        metrics[f"population.reason.{reason}"] = count


def _identity_tags(identity: CohortIdentity, scope: MlflowScope) -> dict[str, str]:
    return {
        "crux.projection_version": PROJECTION_VERSION,
        "crux.scope": scope,
        "crux.cohort_id": identity.cohort_id,
        "crux.backend_id": identity.backend_id,
        "crux.model_id": identity.model_id,
        "crux.model_lock_sha256": identity.model_lock_sha256,
        "crux.backend_descriptor_sha256": identity.backend_descriptor_sha256,
        "crux.input_view_id": identity.input_view_id,
        "crux.reference_manifest_sha256": identity.reference_manifest_sha256,
        "crux.reference_timing_version": identity.reference_timing_version,
        "crux.taxonomy_version": identity.taxonomy_version,
        "crux.lane_map_version": identity.lane_map_version,
        "crux.prediction_map_version": identity.prediction_map_version,
        "crux.scoring_version": identity.scoring_version,
    }


def _read_report_artifacts(report_dir: Path) -> tuple[PublishedArtifact, ...]:
    artifacts: list[PublishedArtifact] = []
    for name in REPORT_ARTIFACT_NAMES:
        path = report_dir / name
        try:
            content = read_regular_file_no_follow(path)
        except OSError as error:
            raise MlflowProjectionError(f"cannot read report artifact {name}") from error
        artifacts.append(PublishedArtifact(path=path, sha256=sha256_hex(content)))
    return tuple(artifacts)


def build_mlflow_projection(
    reports: PublishedCohortReports,
    report_dir: Path,
    *,
    scope: MlflowScope,
) -> MlflowProjection:
    """Project one validated cohort report into deterministic MLflow data."""
    if not isinstance(reports, PublishedCohortReports):
        raise TypeError("reports must be PublishedCohortReports")
    if not isinstance(report_dir, Path):
        raise TypeError("report_dir must be a Path")
    actual = {(row.tolerance_ms, row.mode) for row in reports.aggregates}
    if not REQUIRED_BASELINE_VIEWS <= actual:
        raise MlflowProjectionError("required benchmark views are missing")

    metrics: dict[str, MlflowMetricValue] = {}
    for row in reports.aggregates:
        _project_aggregate(metrics, row)
    _project_population(metrics, reports.population)

    artifacts = _read_report_artifacts(report_dir)
    base_tags = _identity_tags(reports.identity, scope)
    run_name = f"{scope}-{reports.identity.model_id}-{reports.identity.input_view_id}-{reports.identity.cohort_id}"
    payload: JsonValue = {
        "projection_version": PROJECTION_VERSION,
        "scope": scope,
        "run_name": run_name,
        "tags": base_tags,
        "metrics": metrics,
        "artifacts": [
            {"path": artifact.path.name, "sha256": artifact.sha256} for artifact in artifacts
        ],
    }
    tags = dict(base_tags)
    tags[_FINGERPRINT_TAG] = sha256_hex(canonical_json_bytes(payload))
    return MlflowProjection(
        run_name=run_name,
        tags=MappingProxyType(tags),
        metrics=MappingProxyType(metrics),
        artifacts=artifacts,
    )


__all__ = [
    "PROJECTION_VERSION",
    "REPORT_ARTIFACT_NAMES",
    "REPORT_ARTIFACT_PATH",
    "REQUIRED_BASELINE_VIEWS",
    "MlflowMetricValue",
    "MlflowProjection",
    "MlflowProjectionError",
    "MlflowScope",
    "build_mlflow_projection",
]
