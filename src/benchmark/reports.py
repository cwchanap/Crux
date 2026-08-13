"""Deterministic rendering for one canonical cohort score result."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src.benchmark.backend_identity import (
    JsonValue,
    canonical_json_bytes,
    quantize_six,
)
from src.benchmark.cohort_scoring import (
    ClassScore,
    CohortAggregate,
    CohortItem,
    CohortScoreResult,
    EventDiagnostic,
    SongScore,
)
from src.benchmark.models import ScoreSummary

REPORT_SCHEMA = "crux.single-cohort-report/v1"


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
        for row in raw_50[-5:]:
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

    artifacts.summary_json.write_bytes(canonical_json_bytes(_summary_payload(result)))
    _write_csv(
        artifacts.items_csv,
        _ITEM_FIELDNAMES,
        [_item_row(result, item) for item in result.items],
    )
    _write_csv(
        artifacts.per_song_csv,
        _PER_SONG_FIELDNAMES,
        [_song_row(result, row) for row in result.song_scores],
    )
    _write_csv(
        artifacts.per_class_csv,
        _PER_CLASS_FIELDNAMES,
        [
            _class_row(result, song, class_row)
            for song in result.song_scores
            for class_row in song.per_class
        ],
    )
    with artifacts.event_diagnostics_jsonl.open("wb") as handle:
        for diagnostic in result.event_diagnostics:
            handle.write(
                canonical_json_bytes(_diagnostic_payload(result, diagnostic), trailing_newline=True)
            )
    artifacts.summary_markdown.write_text(_markdown_summary(result), encoding="utf-8")
    return artifacts


__all__ = [
    "REPORT_SCHEMA",
    "ReportArtifacts",
    "write_cohort_reports",
]
