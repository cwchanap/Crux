"""Deterministic rendering for one canonical cohort score result."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    strict_json_loads,
)
from src.benchmark.cohort_scoring import (
    COHORT_FAILURE_REASONS,
    ClassScore,
    CohortAggregate,
    CohortExecutionStatus,
    CohortFailureReason,
    CohortIdentity,
    CohortItem,
    CohortScoreResult,
    EventDiagnostic,
    PopulationSummary,
    SongScore,
)
from src.benchmark.models import ScoreSummary

REPORT_SCHEMA = "crux.single-cohort-report/v1"


class ReportIntegrityError(ValueError):
    """Raised when a published cohort report is malformed or inconsistent."""


@dataclass(frozen=True)
class PublishedMetric:
    """One persisted metric object, retaining its canonical decimal values."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


@dataclass(frozen=True)
class PublishedAggregateClass:
    common_class: str
    metric: PublishedMetric
    reference_support: int
    prediction_support: int

    @property
    def summary(self) -> PublishedMetric:
        return self.metric


@dataclass(frozen=True)
class PublishedF1Distribution:
    minimum: Decimal | None
    p10: Decimal | None
    p25: Decimal | None
    median: Decimal | None
    p75: Decimal | None
    p90: Decimal | None
    maximum: Decimal | None


@dataclass(frozen=True)
class PublishedItemRow:
    cohort_id: str
    simfile_id: str
    status: CohortExecutionStatus
    failure_reason: CohortFailureReason | None
    warnings: tuple[str, ...]
    reference_native_event_count: int
    reference_common_event_count: int
    reference_ignored_event_count: int
    reference_unmapped_event_count: int
    reference_duplicate_collapsed_count: int
    prediction_native_event_count: int | None
    prediction_mapped_event_count: int | None
    prediction_unmapped_event_count: int | None
    prediction_mapping_coverage: Decimal | None
    prediction_native_class_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PublishedSongRow:
    cohort_id: str
    model_id: str
    model_lock_sha256: str
    prediction_map_version: str
    input_view_id: str
    scoring_version: str
    simfile_id: str
    tolerance_ms: int
    mode: Literal["raw", "aligned"]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None
    prediction_to_reference_ratio: Decimal | None
    median_abs_error_ms: Decimal | None
    p95_abs_error_ms: Decimal | None
    offset_ms: Decimal | None
    warnings: tuple[str, ...]

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


@dataclass(frozen=True)
class PublishedClassRow:
    cohort_id: str
    model_id: str
    model_lock_sha256: str
    prediction_map_version: str
    input_view_id: str
    scoring_version: str
    simfile_id: str
    tolerance_ms: int
    mode: Literal["raw", "aligned"]
    common_class: str
    true_positives: int
    false_positives: int
    false_negatives: int
    reference_support: int
    prediction_support: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


@dataclass(frozen=True)
class PublishedAggregate:
    tolerance_ms: int
    mode: Literal["raw", "aligned"]
    event_micro: PublishedMetric
    song_macro_f1: Decimal | None
    class_macro_f1: Decimal | None
    song_f1_distribution: PublishedF1Distribution
    per_class: tuple[PublishedAggregateClass, ...]
    successful_song_count: int


@dataclass(frozen=True)
class PublishedCohortReports:
    identity: CohortIdentity
    population: PopulationSummary
    aggregates: tuple[PublishedAggregate, ...]
    items: tuple[PublishedItemRow, ...]
    songs: tuple[PublishedSongRow, ...]
    classes: tuple[PublishedClassRow, ...]


@dataclass(frozen=True)
class ReportArtifacts:
    summary_json: Path
    items_csv: Path
    per_song_csv: Path
    per_class_csv: Path
    event_diagnostics_jsonl: Path
    summary_markdown: Path


_ITEM_FIELDNAMES = (
    "cohort_id",
    "simfile_id",
    "status",
    "failure_reason",
    "warnings",
    "reference_native_event_count",
    "reference_common_event_count",
    "reference_ignored_event_count",
    "reference_unmapped_event_count",
    "reference_duplicate_collapsed_count",
    "prediction_native_event_count",
    "prediction_mapped_event_count",
    "prediction_unmapped_event_count",
    "prediction_mapping_coverage",
    "prediction_native_class_counts",
)

_PER_SONG_FIELDNAMES = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
    "simfile_id",
    "tolerance_ms",
    "mode",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "prediction_to_reference_ratio",
    "median_abs_error_ms",
    "p95_abs_error_ms",
    "offset_ms",
    "warnings",
)

_PER_CLASS_FIELDNAMES = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
    "simfile_id",
    "tolerance_ms",
    "mode",
    "common_class",
    "tp",
    "fp",
    "fn",
    "reference_support",
    "prediction_support",
    "precision",
    "recall",
    "f1",
)


def _report_decimal(value: float | None) -> Decimal | None:
    """Quantize one float-derived report value at the canonical boundary."""
    return None if value is None else quantize_six(value)


def _csv_decimal(value: float | None) -> str:
    """Render one float-derived value with the exact canonical JSON token."""
    decimal = _report_decimal(value)
    return "" if decimal is None else canonical_json_bytes(decimal).decode("ascii")


def _metric(summary: ScoreSummary) -> dict[str, JsonValue]:
    return {
        "tp": summary.true_positives,
        "fp": summary.false_positives,
        "fn": summary.false_negatives,
        "precision": _report_decimal(summary.precision),
        "recall": _report_decimal(summary.recall),
        "f1": _report_decimal(summary.f1),
    }


def _identity_payload(result: CohortScoreResult) -> dict[str, JsonValue]:
    identity = result.identity
    return {
        "cohort_id": identity.cohort_id,
        "reference_manifest_sha256": identity.reference_manifest_sha256,
        "reference_timing_version": identity.reference_timing_version,
        "taxonomy_version": identity.taxonomy_version,
        "lane_map_version": identity.lane_map_version,
        "backend_id": identity.backend_id,
        "model_id": identity.model_id,
        "model_lock_sha256": identity.model_lock_sha256,
        "backend_descriptor_sha256": identity.backend_descriptor_sha256,
        "prediction_map_version": identity.prediction_map_version,
        "input_view_id": identity.input_view_id,
        "scoring_version": identity.scoring_version,
    }


def _population_payload(result: CohortScoreResult) -> dict[str, JsonValue]:
    population = result.population
    return {
        "total_count": population.total_count,
        "success_count": population.success_count,
        "failed_count": population.failed_count,
        "skipped_count": population.skipped_count,
        "quarantined_count": population.quarantined_count,
        "reason_counts": dict(population.reason_counts),
    }


def _class_payload(row: ClassScore) -> dict[str, JsonValue]:
    return {
        "common_class": row.common_class,
        **_metric(row.summary),
        "reference_support": row.reference_support,
        "prediction_support": row.prediction_support,
    }


def _distribution_payload(aggregate: CohortAggregate) -> dict[str, JsonValue]:
    distribution = aggregate.song_f1_distribution
    return {
        "minimum": _report_decimal(distribution.minimum),
        "p10": _report_decimal(distribution.p10),
        "p25": _report_decimal(distribution.p25),
        "median": _report_decimal(distribution.median),
        "p75": _report_decimal(distribution.p75),
        "p90": _report_decimal(distribution.p90),
        "maximum": _report_decimal(distribution.maximum),
    }


def _aggregate_payload(aggregate: CohortAggregate) -> dict[str, JsonValue]:
    return {
        "tolerance_ms": aggregate.tolerance_ms,
        "mode": aggregate.mode,
        "event_micro": _metric(aggregate.event_micro),
        "song_macro_f1": _report_decimal(aggregate.song_macro_f1),
        "class_macro_f1": _report_decimal(aggregate.class_macro_f1),
        "song_f1_distribution": _distribution_payload(aggregate),
        "per_class": [_class_payload(row) for row in aggregate.per_class],
        "successful_song_count": aggregate.successful_song_count,
    }


def _summary_payload(result: CohortScoreResult) -> dict[str, JsonValue]:
    return {
        "schema": REPORT_SCHEMA,
        "identity": _identity_payload(result),
        "tolerances_ms": list(result.tolerances_ms),
        "population": _population_payload(result),
        "aggregates": [_aggregate_payload(row) for row in result.aggregates],
    }


def _optional_count(value: int | None) -> str:
    return "" if value is None else str(value)


def _warnings(value: tuple[str, ...]) -> str:
    return "|".join(sorted(value))


def _native_class_counts(value: tuple[tuple[str, int], ...]) -> str:
    return "|".join(f"{native_class}={count}" for native_class, count in value)


def _mapping_coverage(item: CohortItem) -> str:
    native = item.coverage.prediction_native_event_count
    mapped = item.coverage.prediction_mapped_event_count
    if native is None or mapped is None or native <= 0:
        return ""
    return _csv_decimal(mapped / native)


def _item_row(result: CohortScoreResult, item: CohortItem) -> dict[str, str | int]:
    coverage = item.coverage
    return {
        "cohort_id": result.identity.cohort_id,
        "simfile_id": item.simfile_id,
        "status": item.status,
        "failure_reason": item.failure_reason or "",
        "warnings": _warnings(item.warnings),
        "reference_native_event_count": coverage.reference_native_event_count,
        "reference_common_event_count": coverage.reference_common_event_count,
        "reference_ignored_event_count": coverage.reference_ignored_event_count,
        "reference_unmapped_event_count": coverage.reference_unmapped_event_count,
        "reference_duplicate_collapsed_count": coverage.reference_duplicate_collapsed_count,
        "prediction_native_event_count": _optional_count(coverage.prediction_native_event_count),
        "prediction_mapped_event_count": _optional_count(coverage.prediction_mapped_event_count),
        "prediction_unmapped_event_count": _optional_count(
            coverage.prediction_unmapped_event_count
        ),
        "prediction_mapping_coverage": _mapping_coverage(item),
        "prediction_native_class_counts": _native_class_counts(
            coverage.prediction_native_class_counts
        ),
    }


def _score_identity(result: CohortScoreResult) -> dict[str, str]:
    identity = result.identity
    return {
        "cohort_id": identity.cohort_id,
        "model_id": identity.model_id,
        "model_lock_sha256": identity.model_lock_sha256,
        "prediction_map_version": identity.prediction_map_version,
        "input_view_id": identity.input_view_id,
        "scoring_version": identity.scoring_version,
    }


def _song_row(result: CohortScoreResult, row: SongScore) -> dict[str, str | int]:
    summary = row.summary
    return {
        **_score_identity(result),
        "simfile_id": row.simfile_id,
        "tolerance_ms": row.tolerance_ms,
        "mode": row.mode,
        "tp": summary.true_positives,
        "fp": summary.false_positives,
        "fn": summary.false_negatives,
        "precision": _csv_decimal(summary.precision),
        "recall": _csv_decimal(summary.recall),
        "f1": _csv_decimal(summary.f1),
        "prediction_to_reference_ratio": _csv_decimal(row.prediction_to_reference_ratio),
        "median_abs_error_ms": _csv_decimal(
            summary.median_abs_error_sec * 1000
            if summary.median_abs_error_sec is not None
            else None
        ),
        "p95_abs_error_ms": _csv_decimal(
            summary.p95_abs_error_sec * 1000 if summary.p95_abs_error_sec is not None else None
        ),
        "offset_ms": _csv_decimal(summary.offset_sec * 1000),
        "warnings": _warnings(row.warnings),
    }


def _class_row(
    result: CohortScoreResult,
    song: SongScore,
    row: ClassScore,
) -> dict[str, str | int]:
    return {
        **_score_identity(result),
        "simfile_id": song.simfile_id,
        "tolerance_ms": song.tolerance_ms,
        "mode": song.mode,
        "common_class": row.common_class,
        "tp": row.summary.true_positives,
        "fp": row.summary.false_positives,
        "fn": row.summary.false_negatives,
        "reference_support": row.reference_support,
        "prediction_support": row.prediction_support,
        "precision": _csv_decimal(row.summary.precision),
        "recall": _csv_decimal(row.summary.recall),
        "f1": _csv_decimal(row.summary.f1),
    }


def _diagnostic_payload(
    result: CohortScoreResult,
    diagnostic: EventDiagnostic,
) -> dict[str, JsonValue]:
    return {
        "cohort_id": result.identity.cohort_id,
        "simfile_id": diagnostic.simfile_id,
        "tolerance_ms": diagnostic.tolerance_ms,
        "mode": diagnostic.mode,
        "outcome": diagnostic.outcome,
        "common_class": diagnostic.common_class,
        "reference_time_sec": _report_decimal(diagnostic.reference_time_sec),
        "prediction_time_sec": _report_decimal(diagnostic.prediction_time_sec),
        "scored_prediction_time_sec": _report_decimal(diagnostic.scored_prediction_time_sec),
        "timing_error_sec": _report_decimal(diagnostic.timing_error_sec),
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_value(value: float | None) -> str:
    return _csv_decimal(value) or "N/A"


def _markdown_summary(result: CohortScoreResult) -> str:
    identity = result.identity
    population = result.population
    lines = [
        "# Single-Cohort Drum Benchmark",
        "",
        "## Identity",
        "",
        f"- cohort_id: `{identity.cohort_id}`",
        f"- reference_manifest_sha256: `{identity.reference_manifest_sha256}`",
        f"- reference_timing_version: `{identity.reference_timing_version}`",
        f"- taxonomy_version: `{identity.taxonomy_version}`",
        f"- lane_map_version: `{identity.lane_map_version}`",
        f"- backend_id: `{identity.backend_id}`",
        f"- model_id: `{identity.model_id}`",
        f"- model_lock_sha256: `{identity.model_lock_sha256}`",
        f"- backend_descriptor_sha256: `{identity.backend_descriptor_sha256}`",
        f"- prediction_map_version: `{identity.prediction_map_version}`",
        f"- input_view_id: `{identity.input_view_id}`",
        f"- scoring_version: `{identity.scoring_version}`",
        "",
        "## Population",
        "",
        f"- total_count: {population.total_count}",
        f"- success_count: {population.success_count}",
        f"- failed_count: {population.failed_count}",
        f"- skipped_count: {population.skipped_count}",
        f"- quarantined_count: {population.quarantined_count}",
        "- reason_counts: "
        + (", ".join(f"{reason}={count}" for reason, count in population.reason_counts) or "none"),
        "",
        "## Aggregate Results",
        "",
        "| tolerance_ms | mode | tp | fp | fn | precision | recall | f1 | song_macro_f1 | "
        "class_macro_f1 | successful_song_count |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for aggregate in result.aggregates:
        metric = aggregate.event_micro
        lines.append(
            f"| {aggregate.tolerance_ms} | {aggregate.mode} | {metric.true_positives} | "
            f"{metric.false_positives} | {metric.false_negatives} | "
            f"{_markdown_value(metric.precision)} | {_markdown_value(metric.recall)} | "
            f"{_markdown_value(metric.f1)} | {_markdown_value(aggregate.song_macro_f1)} | "
            f"{_markdown_value(aggregate.class_macro_f1)} | {aggregate.successful_song_count} |"
        )

    lines.extend(
        [
            "",
            "## Per-Class Aggregate Results",
            "",
        ]
    )
    for aggregate in result.aggregates:
        lines.extend(
            [
                f"### {aggregate.tolerance_ms}ms {aggregate.mode}",
                "",
                "| common_class | tp | fp | fn | reference_support | prediction_support | "
                "precision | recall | f1 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        if not aggregate.per_class:
            lines.append("No supported classes.")
        else:
            for row in aggregate.per_class:
                lines.append(
                    f"| {row.common_class} | {row.summary.true_positives} | "
                    f"{row.summary.false_positives} | {row.summary.false_negatives} | "
                    f"{row.reference_support} | {row.prediction_support} | "
                    f"{_markdown_value(row.summary.precision)} | "
                    f"{_markdown_value(row.summary.recall)} | {_markdown_value(row.summary.f1)} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Song F1 Distribution",
            "",
            "| tolerance_ms | mode | minimum | p10 | p25 | median | p75 | p90 | maximum |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for aggregate in result.aggregates:
        distribution = aggregate.song_f1_distribution
        lines.append(
            f"| {aggregate.tolerance_ms} | {aggregate.mode} | "
            f"{_markdown_value(distribution.minimum)} | {_markdown_value(distribution.p10)} | "
            f"{_markdown_value(distribution.p25)} | {_markdown_value(distribution.median)} | "
            f"{_markdown_value(distribution.p75)} | {_markdown_value(distribution.p90)} | "
            f"{_markdown_value(distribution.maximum)} |"
        )

    lines.extend(["", "## Song Extremes", "", "### Lowest raw-50ms F1", ""])
    raw_50 = [
        row
        for row in result.song_scores
        if row.tolerance_ms == 50 and row.mode == "raw" and row.summary.f1 is not None
    ]
    raw_50.sort(key=lambda row: (row.summary.f1, row.simfile_id))
    if not raw_50:
        lines.append("No successful songs.")
    else:
        for row in raw_50[:5]:
            lines.append(f"- `{row.simfile_id}`: F1={_markdown_value(row.summary.f1)}")

    lines.extend(["", "### Highest raw-50ms F1", ""])
    if not raw_50:
        lines.append("No successful songs.")
    else:
        for row in reversed(raw_50[-5:]):
            lines.append(f"- `{row.simfile_id}`: F1={_markdown_value(row.summary.f1)}")
    return "\n".join(lines) + "\n"


def write_cohort_reports(
    result: CohortScoreResult,
    output_dir: Path,
) -> ReportArtifacts:
    """Write the six deterministic report artifacts for one scored cohort."""
    if not isinstance(result, CohortScoreResult):
        raise TypeError("result must be CohortScoreResult")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = ReportArtifacts(
        summary_json=output_dir / "summary.json",
        items_csv=output_dir / "items.csv",
        per_song_csv=output_dir / "per_song.csv",
        per_class_csv=output_dir / "per_class.csv",
        event_diagnostics_jsonl=output_dir / "event_diagnostics.jsonl",
        summary_markdown=output_dir / "summary.md",
    )

    artifact_fields = (
        "summary_json",
        "items_csv",
        "per_song_csv",
        "per_class_csv",
        "event_diagnostics_jsonl",
        "summary_markdown",
    )
    with tempfile.TemporaryDirectory(prefix=".cohort-report-stage-", dir=output_dir) as stage_name:
        staged_dir = Path(stage_name)
        staged = {field: staged_dir / getattr(artifacts, field).name for field in artifact_fields}
        staged["summary_json"].write_bytes(canonical_json_bytes(_summary_payload(result)))
        _write_csv(
            staged["items_csv"],
            _ITEM_FIELDNAMES,
            [_item_row(result, item) for item in result.items],
        )
        _write_csv(
            staged["per_song_csv"],
            _PER_SONG_FIELDNAMES,
            [_song_row(result, row) for row in result.song_scores],
        )
        _write_csv(
            staged["per_class_csv"],
            _PER_CLASS_FIELDNAMES,
            [
                _class_row(result, song, class_row)
                for song in result.song_scores
                for class_row in song.per_class
            ],
        )
        with staged["event_diagnostics_jsonl"].open("wb") as handle:
            for diagnostic in result.event_diagnostics:
                handle.write(
                    canonical_json_bytes(
                        _diagnostic_payload(result, diagnostic), trailing_newline=True
                    )
                )
        staged["summary_markdown"].write_text(_markdown_summary(result), encoding="utf-8")

        for field in artifact_fields:
            os.replace(staged[field], getattr(artifacts, field))
    return artifacts


_SUMMARY_FIELDS = ("schema", "identity", "tolerances_ms", "population", "aggregates")
_IDENTITY_FIELDS = (
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
_POPULATION_FIELDS = (
    "total_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
    "reason_counts",
)
_METRIC_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")
_DISTRIBUTION_FIELDS = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
_AGGREGATE_FIELDS = (
    "tolerance_ms",
    "mode",
    "event_micro",
    "song_macro_f1",
    "class_macro_f1",
    "song_f1_distribution",
    "per_class",
    "successful_song_count",
)
_AGGREGATE_CLASS_FIELDS = (
    "common_class",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "reference_support",
    "prediction_support",
)
_REPORT_IDENTITY_FIELDS = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
)
_SCORE_COUNT_FIELDS = ("tp", "fp", "fn")
_MODES = {"raw", "aligned"}


def _report_error(message: str) -> None:
    raise ReportIntegrityError(message)


def _require_keys(value: object, fields: tuple[str, ...], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        _report_error(f"{context} has an invalid schema")
    return value


def _read_report_json(path: Path) -> Mapping[str, object]:
    try:
        value = strict_json_loads(read_regular_file_no_follow(path), require_canonical=True)
    except (OSError, StrictJsonError, UnicodeDecodeError) as error:
        _report_error(f"cannot read report {path.name}: {error}")
    if not isinstance(value, Mapping):
        _report_error(f"{path.name} must contain an object")
    return value


def _read_report_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        content = read_regular_file_no_follow(path).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        _report_error(f"cannot read report {path.name}: {error}")
    reader = csv.DictReader(io.StringIO(content, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
        _report_error(f"{path.name} has an invalid column schema")
    rows: list[dict[str, str]] = []
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                _report_error(f"{path.name} contains a malformed row")
            rows.append({key: value for key, value in row.items() if key is not None})
    except csv.Error as error:
        _report_error(f"{path.name} contains malformed CSV: {error}")
    return rows


def _json_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _report_error(f"{field} must be a nonempty string")
    return value


def _json_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _report_error(f"{field} numeric field is malformed")
    return value


def _json_positive_int(value: object, field: str) -> int:
    parsed = _json_nonnegative_int(value, field)
    if parsed <= 0:
        _report_error(f"{field} numeric field is malformed")
    return parsed


def _json_decimal(value: object, field: str, *, optional: bool = True) -> Decimal | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        _report_error(f"{field} numeric field is malformed")
    parsed = Decimal(value)
    if not parsed.is_finite():
        _report_error(f"{field} numeric field is malformed")
    return parsed


def _bounded_metric(value: object, field: str) -> Decimal | None:
    parsed = _json_decimal(value, field)
    if parsed is not None and not 0 <= parsed <= 1:
        _report_error(f"{field} numeric field is out of range")
    return parsed


def _csv_int(value: object, field: str, *, positive: bool = False) -> int:
    if not isinstance(value, str) or not value or value != value.strip() or not value.isascii():
        _report_error(f"{field} numeric field is malformed")
    if not value.isdigit():
        _report_error(f"{field} numeric field is malformed")
    parsed = int(value)
    if (positive and parsed <= 0) or parsed < 0:
        _report_error(f"{field} numeric field is malformed")
    return parsed


def _parse_csv_decimal(value: object, field: str, *, optional: bool = False) -> Decimal | None:
    if value == "" and optional:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        _report_error(f"{field} numeric field is malformed")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        _report_error(f"{field} numeric field is malformed")
    if not parsed.is_finite():
        _report_error(f"{field} numeric field is malformed")
    return parsed


def _bounded_csv_metric(value: object, field: str) -> Decimal | None:
    parsed = _parse_csv_decimal(value, field, optional=True)
    if parsed is not None and not 0 <= parsed <= 1:
        _report_error(f"{field} numeric field is out of range")
    return parsed


def _nonnegative_csv_metric(value: object, field: str) -> Decimal | None:
    parsed = _parse_csv_decimal(value, field, optional=True)
    if parsed is not None and parsed < 0:
        _report_error(f"{field} numeric field is out of range")
    return parsed


def _parse_warnings(value: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        _report_error(f"{field} must be a string")
    if value == "":
        return ()
    warnings = tuple(value.split("|"))
    if any(not warning for warning in warnings) or len(set(warnings)) != len(warnings):
        _report_error(f"{field} is malformed")
    return warnings


def _parse_simfile_id(value: str, field: str = "simfile_id") -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        _report_error(f"{field} is malformed")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        _report_error(f"{field} is malformed")
    return value


def _parse_native_class_counts(value: str) -> tuple[tuple[str, int], ...]:
    if value == "":
        return ()
    parsed: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in value.split("|"):
        if "=" not in entry:
            _report_error("prediction_native_class_counts numeric field is malformed")
        native_class, count_text = entry.rsplit("=", 1)
        if not native_class or native_class in seen:
            _report_error("prediction_native_class_counts numeric field is malformed")
        seen.add(native_class)
        parsed.append((native_class, _csv_int(count_text, "prediction_native_class_counts")))
    return tuple(parsed)


def _parse_summary_identity(value: object, expected: CohortIdentity) -> CohortIdentity:
    payload = _require_keys(value, _IDENTITY_FIELDS, "summary identity")
    for field in _IDENTITY_FIELDS:
        _json_text(payload[field], f"identity.{field}")
    try:
        identity = CohortIdentity(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError, StrictJsonError) as error:
        _report_error(f"summary identity is malformed: {error}")
    if identity != expected:
        _report_error("summary identity mismatch")
    return identity


def _parse_json_metric(value: object, context: str) -> PublishedMetric:
    payload = _require_keys(value, _METRIC_FIELDS, context)
    counts = tuple(
        _json_nonnegative_int(payload[field], f"{context}.{field}") for field in _SCORE_COUNT_FIELDS
    )
    return PublishedMetric(
        *counts,
        _bounded_metric(payload["precision"], f"{context}.precision"),
        _bounded_metric(payload["recall"], f"{context}.recall"),
        _bounded_metric(payload["f1"], f"{context}.f1"),
    )


def _parse_population(value: object) -> PopulationSummary:
    payload = _require_keys(value, _POPULATION_FIELDS, "summary population")
    counts = {
        field: _json_nonnegative_int(payload[field], f"population.{field}")
        for field in (
            "total_count",
            "success_count",
            "failed_count",
            "skipped_count",
            "quarantined_count",
        )
    }
    reason_payload = payload["reason_counts"]
    if not isinstance(reason_payload, Mapping):
        _report_error("population.reason_counts must be an object")
    reason_counts: list[tuple[CohortFailureReason, int]] = []
    for reason, count in reason_payload.items():
        if reason not in COHORT_FAILURE_REASONS:
            _report_error("population.reason_counts contains an invalid reason")
        reason_counts.append((reason, _json_nonnegative_int(count, f"reason_counts.{reason}")))  # type: ignore[arg-type]
    if counts["total_count"] != sum(counts[field] for field in counts if field != "total_count"):
        _report_error("population counts do not balance")
    return PopulationSummary(
        **counts,
        reason_counts=tuple(sorted(reason_counts)),
    )


def _parse_aggregate_class(value: object, context: str) -> PublishedAggregateClass:
    payload = _require_keys(value, _AGGREGATE_CLASS_FIELDS, context)
    common_class = _json_text(payload["common_class"], f"{context}.common_class")
    metric = _parse_json_metric(
        {field: payload[field] for field in _METRIC_FIELDS},
        context,
    )
    return PublishedAggregateClass(
        common_class=common_class,
        metric=metric,
        reference_support=_json_nonnegative_int(
            payload["reference_support"], f"{context}.reference_support"
        ),
        prediction_support=_json_nonnegative_int(
            payload["prediction_support"], f"{context}.prediction_support"
        ),
    )


def _parse_aggregates(value: object) -> tuple[PublishedAggregate, ...]:
    if not isinstance(value, list):
        _report_error("summary aggregates must be an array")
    aggregates: list[PublishedAggregate] = []
    seen: set[tuple[int, str]] = set()
    for index, raw in enumerate(value):
        context = f"aggregates[{index}]"
        payload = _require_keys(raw, _AGGREGATE_FIELDS, context)
        tolerance_ms = _json_positive_int(payload["tolerance_ms"], f"{context}.tolerance_ms")
        mode = _json_text(payload["mode"], f"{context}.mode")
        if mode not in _MODES:
            _report_error(f"{context}.mode is invalid")
        key = (tolerance_ms, mode)
        if key in seen:
            _report_error("summary aggregates contain duplicate score key")
        seen.add(key)
        distribution_payload = _require_keys(
            payload["song_f1_distribution"], _DISTRIBUTION_FIELDS, f"{context}.song_f1_distribution"
        )
        distribution = PublishedF1Distribution(
            *(
                _bounded_metric(
                    distribution_payload[field], f"{context}.song_f1_distribution.{field}"
                )
                for field in _DISTRIBUTION_FIELDS
            )
        )
        classes = payload["per_class"]
        if not isinstance(classes, list):
            _report_error(f"{context}.per_class must be an array")
        parsed_classes = tuple(
            _parse_aggregate_class(row, f"{context}.per_class[{class_index}]")
            for class_index, row in enumerate(classes)
        )
        class_names = [row.common_class for row in parsed_classes]
        if len(set(class_names)) != len(class_names):
            _report_error(f"{context}.per_class contains duplicate common_class")
        aggregates.append(
            PublishedAggregate(
                tolerance_ms=tolerance_ms,
                mode=mode,  # type: ignore[arg-type]
                event_micro=_parse_json_metric(payload["event_micro"], f"{context}.event_micro"),
                song_macro_f1=_bounded_metric(payload["song_macro_f1"], f"{context}.song_macro_f1"),
                class_macro_f1=_bounded_metric(
                    payload["class_macro_f1"], f"{context}.class_macro_f1"
                ),
                song_f1_distribution=distribution,
                per_class=parsed_classes,
                successful_song_count=_json_nonnegative_int(
                    payload["successful_song_count"], f"{context}.successful_song_count"
                ),
            )
        )
    return tuple(aggregates)


def _validate_report_identity(row: Mapping[str, str], identity: CohortIdentity) -> None:
    expected = {
        "cohort_id": identity.cohort_id,
        "model_id": identity.model_id,
        "model_lock_sha256": identity.model_lock_sha256,
        "prediction_map_version": identity.prediction_map_version,
        "input_view_id": identity.input_view_id,
        "scoring_version": identity.scoring_version,
    }
    for field in _REPORT_IDENTITY_FIELDS:
        if row.get(field) != expected[field]:
            _report_error(f"report identity mismatch for {field}")


def _parse_item_rows(
    path: Path,
    identity: CohortIdentity,
    *,
    strict_semantics: bool = True,
) -> tuple[PublishedItemRow, ...]:
    rows = _read_report_csv(path, _ITEM_FIELDNAMES)
    parsed: list[PublishedItemRow] = []
    seen: set[str] = set()
    for row in rows:
        if row["cohort_id"] != identity.cohort_id:
            _report_error("items report identity mismatch for cohort_id")
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id in seen:
            _report_error("items report contains duplicate simfile_id")
        seen.add(simfile_id)
        status = row["status"]
        if status not in {"success", "failed", "skipped", "quarantined"}:
            _report_error("items report contains an invalid status")
        failure_reason = row["failure_reason"]
        if failure_reason and failure_reason not in COHORT_FAILURE_REASONS:
            _report_error("items report contains an invalid failure_reason")
        reference_counts = tuple(
            _csv_int(row[field], field)
            for field in (
                "reference_native_event_count",
                "reference_common_event_count",
                "reference_ignored_event_count",
                "reference_unmapped_event_count",
                "reference_duplicate_collapsed_count",
            )
        )
        prediction_counts = tuple(
            _csv_int(row[field], field) if row[field] != "" else None
            for field in (
                "prediction_native_event_count",
                "prediction_mapped_event_count",
                "prediction_unmapped_event_count",
            )
        )
        native_class_counts = _parse_native_class_counts(row["prediction_native_class_counts"])
        if strict_semantics and status == "success":
            if failure_reason:
                _report_error("successful item contains a failure_reason")
            if any(value is None for value in prediction_counts):
                _report_error("successful item is missing prediction coverage")
        elif strict_semantics:
            expected_reason = {
                "failed": {
                    "backend_unavailable",
                    "inference_failed",
                    "prediction_artifact_invalid",
                    "prediction_missing",
                },
                "skipped": {"explicitly_skipped"},
                "quarantined": {"reference_quarantined"},
            }[status]
            if failure_reason not in expected_reason:
                _report_error(f"{status} item has an invalid failure_reason")
            if any(value is not None for value in prediction_counts) or native_class_counts:
                _report_error(f"{status} item contains prediction coverage")
        parsed.append(
            PublishedItemRow(
                cohort_id=row["cohort_id"],
                simfile_id=simfile_id,
                status=status,  # type: ignore[arg-type]
                failure_reason=failure_reason or None,  # type: ignore[arg-type]
                warnings=_parse_warnings(row["warnings"], "warnings"),
                reference_native_event_count=reference_counts[0],
                reference_common_event_count=reference_counts[1],
                reference_ignored_event_count=reference_counts[2],
                reference_unmapped_event_count=reference_counts[3],
                reference_duplicate_collapsed_count=reference_counts[4],
                prediction_native_event_count=prediction_counts[0],
                prediction_mapped_event_count=prediction_counts[1],
                prediction_unmapped_event_count=prediction_counts[2],
                prediction_mapping_coverage=_bounded_csv_metric(
                    row["prediction_mapping_coverage"], "prediction_mapping_coverage"
                ),
                prediction_native_class_counts=native_class_counts,
            )
        )
    return tuple(parsed)


def _parse_song_rows(
    path: Path, identity: CohortIdentity, successful_ids: set[str]
) -> tuple[PublishedSongRow, ...]:
    rows = _read_report_csv(path, _PER_SONG_FIELDNAMES)
    parsed: list[PublishedSongRow] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        _validate_report_identity(row, identity)
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id not in successful_ids:
            _report_error("per_song report contains a non-success item")
        tolerance_ms = _csv_int(row["tolerance_ms"], "tolerance_ms", positive=True)
        mode = row["mode"]
        if mode not in _MODES:
            _report_error("per_song report contains an invalid mode")
        key = (simfile_id, tolerance_ms, mode)
        if key in seen:
            _report_error("per_song report contains duplicate score key")
        seen.add(key)
        parsed.append(
            PublishedSongRow(
                cohort_id=row["cohort_id"],
                model_id=row["model_id"],
                model_lock_sha256=row["model_lock_sha256"],
                prediction_map_version=row["prediction_map_version"],
                input_view_id=row["input_view_id"],
                scoring_version=row["scoring_version"],
                simfile_id=simfile_id,
                tolerance_ms=tolerance_ms,
                mode=mode,  # type: ignore[arg-type]
                true_positives=_csv_int(row["tp"], "tp"),
                false_positives=_csv_int(row["fp"], "fp"),
                false_negatives=_csv_int(row["fn"], "fn"),
                precision=_bounded_csv_metric(row["precision"], "precision"),
                recall=_bounded_csv_metric(row["recall"], "recall"),
                f1=_bounded_csv_metric(row["f1"], "f1"),
                prediction_to_reference_ratio=_nonnegative_csv_metric(
                    row["prediction_to_reference_ratio"], "prediction_to_reference_ratio"
                ),
                median_abs_error_ms=_nonnegative_csv_metric(
                    row["median_abs_error_ms"], "median_abs_error_ms"
                ),
                p95_abs_error_ms=_nonnegative_csv_metric(
                    row["p95_abs_error_ms"], "p95_abs_error_ms"
                ),
                offset_ms=_parse_csv_decimal(row["offset_ms"], "offset_ms", optional=True),
                warnings=_parse_warnings(row["warnings"], "warnings"),
            )
        )
    return tuple(parsed)


def _parse_class_rows(
    path: Path, identity: CohortIdentity, successful_ids: set[str]
) -> tuple[PublishedClassRow, ...]:
    rows = _read_report_csv(path, _PER_CLASS_FIELDNAMES)
    parsed: list[PublishedClassRow] = []
    seen: set[tuple[str, int, str, str]] = set()
    for row in rows:
        _validate_report_identity(row, identity)
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id not in successful_ids:
            _report_error("per_class report contains a non-success item")
        tolerance_ms = _csv_int(row["tolerance_ms"], "tolerance_ms", positive=True)
        mode = row["mode"]
        if mode not in _MODES:
            _report_error("per_class report contains an invalid mode")
        common_class = row["common_class"]
        if not common_class or common_class != common_class.strip():
            _report_error("per_class report contains an invalid common_class")
        key = (simfile_id, tolerance_ms, mode, common_class)
        if key in seen:
            _report_error("per_class report contains duplicate score key")
        seen.add(key)
        parsed.append(
            PublishedClassRow(
                cohort_id=row["cohort_id"],
                model_id=row["model_id"],
                model_lock_sha256=row["model_lock_sha256"],
                prediction_map_version=row["prediction_map_version"],
                input_view_id=row["input_view_id"],
                scoring_version=row["scoring_version"],
                simfile_id=simfile_id,
                tolerance_ms=tolerance_ms,
                mode=mode,  # type: ignore[arg-type]
                common_class=common_class,
                true_positives=_csv_int(row["tp"], "tp"),
                false_positives=_csv_int(row["fp"], "fp"),
                false_negatives=_csv_int(row["fn"], "fn"),
                reference_support=_csv_int(row["reference_support"], "reference_support"),
                prediction_support=_csv_int(row["prediction_support"], "prediction_support"),
                precision=_bounded_csv_metric(row["precision"], "precision"),
                recall=_bounded_csv_metric(row["recall"], "recall"),
                f1=_bounded_csv_metric(row["f1"], "f1"),
            )
        )
    return tuple(parsed)


def read_cohort_reports(
    report_dir: Path,
    *,
    expected_identity: CohortIdentity,
) -> PublishedCohortReports:
    """Read and validate the six published HPA-325 cohort report artifacts."""
    if not isinstance(report_dir, Path):
        raise TypeError("report_dir must be a Path")
    if not isinstance(expected_identity, CohortIdentity):
        raise TypeError("expected_identity must be CohortIdentity")

    summary = _read_report_json(report_dir / "summary.json")
    _require_keys(summary, _SUMMARY_FIELDS, "summary")
    if summary["schema"] != REPORT_SCHEMA:
        _report_error("summary schema is invalid")
    identity = _parse_summary_identity(summary["identity"], expected_identity)
    tolerances = summary["tolerances_ms"]
    if not isinstance(tolerances, list) or not tolerances:
        _report_error("summary tolerances_ms must be a nonempty array")
    parsed_tolerances = tuple(_json_positive_int(value, "tolerances_ms") for value in tolerances)
    if parsed_tolerances != tuple(sorted(set(parsed_tolerances))):
        _report_error("summary tolerances_ms must be sorted and unique")
    population = _parse_population(summary["population"])
    aggregates = _parse_aggregates(summary["aggregates"])
    if {aggregate.tolerance_ms for aggregate in aggregates} != set(parsed_tolerances):
        _report_error("summary aggregate tolerances do not match tolerances_ms")

    items = _parse_item_rows(report_dir / "items.csv", identity)
    status_counts = Counter(row.status for row in items)
    if (
        population.total_count,
        population.success_count,
        population.failed_count,
        population.skipped_count,
        population.quarantined_count,
    ) != (
        len(items),
        status_counts["success"],
        status_counts["failed"],
        status_counts["skipped"],
        status_counts["quarantined"],
    ):
        _report_error("summary population does not match items report")
    actual_reasons = Counter(row.failure_reason for row in items if row.failure_reason is not None)
    if dict(population.reason_counts) != dict(actual_reasons):
        _report_error("summary reason_counts does not match items report")

    successful_ids = {row.simfile_id for row in items if row.status == "success"}
    songs = _parse_song_rows(report_dir / "per_song.csv", identity, successful_ids)
    classes = _parse_class_rows(report_dir / "per_class.csv", identity, successful_ids)
    return PublishedCohortReports(
        identity=identity,
        population=population,
        aggregates=aggregates,
        items=items,
        songs=songs,
        classes=classes,
    )


__all__ = [
    "REPORT_SCHEMA",
    "ReportIntegrityError",
    "ReportArtifacts",
    "PublishedMetric",
    "PublishedAggregateClass",
    "PublishedF1Distribution",
    "PublishedItemRow",
    "PublishedSongRow",
    "PublishedClassRow",
    "PublishedAggregate",
    "PublishedCohortReports",
    "read_cohort_reports",
    "write_cohort_reports",
]
