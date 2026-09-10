"""Projection of canonical cohort reports to MLflow data, plus its publisher.

The ``mlflow`` package is never imported at module scope: this module projects
one validated ``PublishedCohortReports`` plus its report directory into the
complete MLflow-ready payload (tags, run name, canonical metrics, artifact
digests, fingerprint), and publishes it through an injected low-level
``MlflowClient`` with only function-local mlflow imports.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, get_args
from urllib.parse import urlsplit

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
    "MATCHING_RUN_LIMIT",
    "PROJECTION_VERSION",
    "REPORT_ARTIFACT_NAMES",
    "REPORT_ARTIFACT_PATH",
    "REQUIRED_BASELINE_VIEWS",
    "MlflowErrorCode",
    "MlflowMetricValue",
    "MlflowProjection",
    "MlflowProjectionError",
    "MlflowPublication",
    "MlflowScope",
    "build_matching_runs_filter",
    "publish_mlflow_projection",
    "validate_tracking_uri",
]


# --- Publication against a real MLflow client (all mlflow imports are lazy) ---

MATCHING_RUN_LIMIT = 1000

MlflowErrorCode: TypeAlias = Literal[
    "missing_optional_dependency",
    "invalid_config",
    "experiment_access_failed",
    "run_search_failed",
    "run_create_failed",
    "metric_log_failed",
    "artifact_upload_failed",
    "run_finalize_failed",
    "run_conflict",
]

_MLFLOW_ERROR_MESSAGES: Mapping[MlflowErrorCode, str] = MappingProxyType(
    {
        "missing_optional_dependency": "mlflow is not installed; install the 'mlflow' extra",
        "invalid_config": (
            "tracking URI must be an http(s) URL with a hostname and no credentials,"
            " query, or fragment"
        ),
        "experiment_access_failed": "could not resolve or create the MLflow experiment",
        "run_search_failed": "could not search for matching MLflow runs",
        "run_create_failed": "could not create the MLflow run",
        "metric_log_failed": "could not log MLflow metrics",
        "artifact_upload_failed": "could not upload report artifacts to MLflow",
        "run_finalize_failed": "could not finalize the MLflow run",
        "run_conflict": "conflicting MLflow runs exist for this projection identity",
    }
)

_SEARCH_TAG_KEYS = ("crux.cohort_id", "crux.projection_version", "crux.scope")
_RETRIABLE_RUN_STATUSES = frozenset({"FAILED", "KILLED"})

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

_T = TypeVar("_T")


class MlflowPublicationError(Exception):
    """Publication failure with a bounded code; vendor detail stays in the cause."""

    def __init__(self, code: MlflowErrorCode) -> None:
        self.code: MlflowErrorCode = code
        self.message = _MLFLOW_ERROR_MESSAGES[code]
        super().__init__(self.message)


@dataclass(frozen=True)
class MlflowPublication:
    """Outcome of one publish call, shaped for direct CLI printing."""

    created: bool
    experiment_id: str
    experiment_name: str
    projection_sha256: str
    run_id: str
    status: Literal["published", "already_published"]


def validate_tracking_uri(raw_uri: str) -> str:
    """Accept only a public http(s) MLflow/DagsHub tracking URI for CLI use."""
    try:
        parts = urlsplit(raw_uri)
    except ValueError as error:
        raise MlflowPublicationError("invalid_config") from error
    invalid = (
        parts.scheme.lower() not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or bool(parts.query)
        or bool(parts.fragment)
    )
    if invalid:
        raise MlflowPublicationError("invalid_config")
    return raw_uri


def build_matching_runs_filter(tags: Mapping[str, str]) -> str:
    """Build an MLflow filter over the search-identity tags with quoted keys."""
    clauses = []
    for name in _SEARCH_TAG_KEYS:
        value = tags[name].replace("\\", "\\\\").replace("'", "\\'")
        clauses.append(f"tags.\"{name}\" = '{value}'")
    return " and ".join(clauses)


def _bounded(code: MlflowErrorCode, operation: Callable[[], _T]) -> _T:
    """Run one vendor call and bound any failure into the fixed error code."""
    try:
        return operation()
    except MlflowPublicationError:
        raise
    except Exception as error:
        raise MlflowPublicationError(code) from error


def _classify_matching_runs(runs: Iterable[Run], fingerprint: str) -> Run | None:
    finished: list[Run] = []
    for run in runs:
        if run.data.tags.get(_FINGERPRINT_TAG) != fingerprint:
            raise MlflowPublicationError("run_conflict")
        if run.info.status == "FINISHED":
            finished.append(run)
        elif run.info.status not in _RETRIABLE_RUN_STATUSES:
            raise MlflowPublicationError("run_conflict")
    if len(finished) > 1:
        raise MlflowPublicationError("run_conflict")
    return finished[0] if finished else None


def _resolve_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    def existing() -> str | None:
        experiment = client.get_experiment_by_name(experiment_name)
        return experiment.experiment_id if experiment is not None else None

    experiment_id = _bounded("experiment_access_failed", existing)
    if experiment_id is not None:
        return experiment_id
    return _bounded("experiment_access_failed", lambda: client.create_experiment(experiment_name))


def _find_published_run(
    client: MlflowClient, experiment_id: str, projection: MlflowProjection
) -> Run | None:
    filter_string = build_matching_runs_filter(projection.tags)

    def search():
        return client.search_runs(
            [experiment_id], filter_string=filter_string, max_results=MATCHING_RUN_LIMIT
        )

    runs = _bounded("run_search_failed", search)
    if runs.token:
        raise MlflowPublicationError("run_conflict")
    return _classify_matching_runs(runs, projection.tags[_FINGERPRINT_TAG])


def _create_run(client: MlflowClient, experiment_id: str, projection: MlflowProjection) -> str:
    def create():
        return client.create_run(
            experiment_id, tags=dict(projection.tags), run_name=projection.run_name
        )

    return _bounded("run_create_failed", create).info.run_id


def _log_metrics(
    client: MlflowClient, run_id: str, metrics: Mapping[str, MlflowMetricValue]
) -> None:
    def log() -> None:
        from mlflow.entities import Metric

        timestamp_ms = int(time.time() * 1000)
        entries = []
        for name, value in metrics.items():
            number = float(value)
            if not math.isfinite(number):
                raise MlflowPublicationError("metric_log_failed")
            entries.append(Metric(name, number, timestamp_ms, 0))
        client.log_batch(run_id, metrics=entries)

    _bounded("metric_log_failed", log)


def _log_artifacts(
    client: MlflowClient, run_id: str, artifacts: tuple[PublishedArtifact, ...]
) -> None:
    for artifact in artifacts:
        _bounded(
            "artifact_upload_failed",
            lambda path=artifact.path: client.log_artifact(run_id, str(path), REPORT_ARTIFACT_PATH),
        )


def _finalize_run(client: MlflowClient, run_id: str) -> None:
    _bounded("run_finalize_failed", lambda: client.set_terminated(run_id, "FINISHED"))


def _terminate_new_run_as_failed(client: MlflowClient, run_id: str) -> None:
    # Best-effort cleanup only; the original bounded error must win.
    try:
        client.set_terminated(run_id, "FAILED")
    except Exception:
        pass


def publish_mlflow_projection(
    projection: MlflowProjection,
    *,
    client: MlflowClient,
    experiment_name: str,
) -> MlflowPublication:
    """Publish or confirm one projection; idempotent on the search-identity tags."""
    if not isinstance(projection, MlflowProjection):
        raise TypeError("projection must be MlflowProjection")
    fingerprint = projection.tags[_FINGERPRINT_TAG]
    experiment_id = _resolve_experiment_id(client, experiment_name)
    published = _find_published_run(client, experiment_id, projection)
    if published is not None:
        return MlflowPublication(
            created=False,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            projection_sha256=fingerprint,
            run_id=published.info.run_id,
            status="already_published",
        )
    run_id = _create_run(client, experiment_id, projection)
    try:
        _log_metrics(client, run_id, projection.metrics)
        _log_artifacts(client, run_id, projection.artifacts)
        _finalize_run(client, run_id)
    except MlflowPublicationError:
        _terminate_new_run_as_failed(client, run_id)
        raise
    return MlflowPublication(
        created=True,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        projection_sha256=fingerprint,
        run_id=run_id,
        status="published",
    )
