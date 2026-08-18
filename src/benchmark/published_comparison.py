"""Model-neutral joins and rendering for published cohort reports."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from statistics import mean, median

from src.benchmark.backend_identity import StrictJsonError, canonical_json_bytes, require_sha256
from src.benchmark.reports import (
    PublishedClassRow,
    PublishedCohortReports,
    PublishedSongRow,
)

_MODES = {"raw": 0, "aligned": 1}
_SIX_PLACES = Decimal("0.000001")
COMPARISON_SCHEMA = "crux.oaf-muscriptor-comparison/v1"
COMPARISON_TITLE = "OaF/MuScriptor Published Report Comparison"


class ComparisonIntegrityError(ValueError):
    """Raised when published comparison evidence cannot be joined safely."""


@dataclass(frozen=True)
class PublishedRunItem:
    """One run-snapshot row needed to establish a paired input population."""

    simfile_id: str
    status: str
    source_audio_sha256: str | None
    input_audio_sha256: str | None


@dataclass(frozen=True)
class PublishedRunEvidence:
    """Published reports plus the run rows that establish input provenance."""

    identity: object
    items: Mapping[str, PublishedRunItem]
    reports: PublishedCohortReports
    snapshot: Mapping[str, object] = field(default_factory=dict)
    label: str = "model"


def _fail(message: str) -> None:
    raise ComparisonIntegrityError(message)


def _label(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _items(evidence: PublishedRunEvidence | object) -> Mapping[str, object]:
    items = getattr(evidence, "items", None)
    if not isinstance(items, Mapping):
        _fail("published run evidence items must be a mapping")
    return items


def _item_status(item: object) -> str:
    status = getattr(item, "status", None)
    if not isinstance(status, str):
        _fail("published run item status is malformed")
    return status


def _item_hash(item: object, field_name: str) -> str | None:
    value = getattr(item, field_name, None)
    if value is not None and not isinstance(value, str):
        _fail(f"{field_name} is malformed")
    if field_name == "source_audio_sha256":
        if value is None:
            _fail("source_audio_sha256 is missing")
        try:
            require_sha256(value, field_name)
        except StrictJsonError as error:
            _fail(str(error))
    return value


def pairable_success_ids(
    left: PublishedRunEvidence,
    right: PublishedRunEvidence,
    selected_ids: set[str] | None,
    *,
    require_identical_input_hash: bool = True,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> tuple[set[str], dict[str, int]]:
    """Return successful rows with matching source provenance.

    The source-audio hash is always authoritative.  Canonical input hashes are
    required to match by default, with the explicit opt-out used by derived
    input-view experiments.
    """
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    if not isinstance(require_identical_input_hash, bool):
        raise TypeError("require_identical_input_hash must be a bool")
    left_items = _items(left)
    right_items = _items(right)
    left_success = {
        str(item_id)
        for item_id, item in left_items.items()
        if _item_status(item) == "success"
        and (selected_ids is None or str(item_id) in selected_ids)
    }
    right_success = {
        str(item_id)
        for item_id, item in right_items.items()
        if _item_status(item) == "success"
        and (selected_ids is None or str(item_id) in selected_ids)
    }
    common = left_success & right_success
    source_mismatch = 0
    pairable: set[str] = set()
    for simfile_id in sorted(common, key=int):
        left_item = left_items[simfile_id]
        right_item = right_items[simfile_id]
        left_source = _item_hash(left_item, "source_audio_sha256")
        right_source = _item_hash(right_item, "source_audio_sha256")
        if left_source != right_source:
            source_mismatch += 1
            continue
        if require_identical_input_hash and _item_hash(
            left_item, "input_audio_sha256"
        ) != _item_hash(right_item, "input_audio_sha256"):
            _fail(
                "canonical-input integrity error: input_audio_sha256 mismatch for "
                f"simfile_id={simfile_id} source_audio_sha256={left_source}"
            )
        pairable.add(simfile_id)
    exclusions = {
        f"{left_label}_only_success": len(left_success - right_success),
        f"{right_label}_only_success": len(right_success - left_success),
        "source_audio_mismatch": source_mismatch,
    }
    return pairable, exclusions


def selected_rows[T](rows: Mapping[T, object], selected_ids: set[str] | None) -> dict[T, object]:
    """Select score-keyed rows without changing their deterministic order later."""
    if selected_ids is None:
        return dict(rows)
    return {key: row for key, row in rows.items() if str(key[0]) in selected_ids}  # type: ignore[index]


def metric_delta(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    """Return right-minus-left at the report's six-decimal boundary."""
    if left is None or right is None:
        return None
    value = (right - left).quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN)
    return Decimal(0) if value.is_zero() else value


def csv_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    if value.is_zero():
        return "0"
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _row_value(row: object, field: str) -> object:
    return getattr(row, field)


def _song_key_sort_key(value: tuple[str, int, str]) -> tuple[int, int, int]:
    return int(value[0]), value[1], _MODES[value[2]]


def paired_song_rows(
    left: Mapping[tuple[str, int, str], PublishedSongRow | object],
    right: Mapping[tuple[str, int, str], PublishedSongRow | object],
    pairable_ids: set[str],
    *,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> list[dict[str, str]]:
    """Join per-song score rows and render deterministic metric deltas."""
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    left_keys = {key for key in left if str(key[0]) in pairable_ids}
    right_keys = {key for key in right if str(key[0]) in pairable_ids}
    if left_keys != right_keys:
        missing_right = sorted(left_keys - right_keys, key=_song_key_sort_key)
        missing_left = sorted(right_keys - left_keys, key=_song_key_sort_key)
        _fail(
            "per_song score key grid mismatch for pairable songs"
            f" (missing from {right_label}: {missing_right},"
            f" missing from {left_label}: {missing_left})"
        )
    rows: list[dict[str, str]] = []
    for key in sorted(left_keys, key=_song_key_sort_key):
        left_row, right_row = left[key], right[key]
        rows.append(
            {
                "simfile_id": key[0],
                "tolerance_ms": str(key[1]),
                "mode": key[2],
                f"{left_label}_precision": csv_decimal(_row_value(left_row, "precision")),
                f"{right_label}_precision": csv_decimal(_row_value(right_row, "precision")),
                "delta_precision": csv_decimal(
                    metric_delta(
                        _row_value(left_row, "precision"), _row_value(right_row, "precision")
                    )
                ),
                f"{left_label}_recall": csv_decimal(_row_value(left_row, "recall")),
                f"{right_label}_recall": csv_decimal(_row_value(right_row, "recall")),
                "delta_recall": csv_decimal(
                    metric_delta(_row_value(left_row, "recall"), _row_value(right_row, "recall"))
                ),
                f"{left_label}_f1": csv_decimal(_row_value(left_row, "f1")),
                f"{right_label}_f1": csv_decimal(_row_value(right_row, "f1")),
                "delta_f1": csv_decimal(
                    metric_delta(_row_value(left_row, "f1"), _row_value(right_row, "f1"))
                ),
            }
        )
    return rows


def paired_class_rows(
    left: Mapping[tuple[str, int, str, str], PublishedClassRow | object],
    right: Mapping[tuple[str, int, str, str], PublishedClassRow | object],
    pairable_ids: set[str],
    *,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> list[dict[str, str]]:
    """Join per-class score rows and render deterministic metric deltas."""
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    rows: list[dict[str, str]] = []
    for key in sorted(
        set(left) & set(right),
        key=lambda value: (int(value[0]), value[1], _MODES[value[2]], value[3]),
    ):
        if str(key[0]) not in pairable_ids:
            continue
        left_row, right_row = left[key], right[key]
        rows.append(
            {
                "simfile_id": key[0],
                "tolerance_ms": str(key[1]),
                "mode": key[2],
                "common_class": key[3],
                f"{left_label}_reference_support": str(_row_value(left_row, "reference_support")),
                f"{right_label}_reference_support": str(_row_value(right_row, "reference_support")),
                f"{left_label}_prediction_support": str(_row_value(left_row, "prediction_support")),
                f"{right_label}_prediction_support": str(
                    _row_value(right_row, "prediction_support")
                ),
                f"{left_label}_precision": csv_decimal(_row_value(left_row, "precision")),
                f"{right_label}_precision": csv_decimal(_row_value(right_row, "precision")),
                "delta_precision": csv_decimal(
                    metric_delta(
                        _row_value(left_row, "precision"), _row_value(right_row, "precision")
                    )
                ),
                f"{left_label}_recall": csv_decimal(_row_value(left_row, "recall")),
                f"{right_label}_recall": csv_decimal(_row_value(right_row, "recall")),
                "delta_recall": csv_decimal(
                    metric_delta(_row_value(left_row, "recall"), _row_value(right_row, "recall"))
                ),
                f"{left_label}_f1": csv_decimal(_row_value(left_row, "f1")),
                f"{right_label}_f1": csv_decimal(_row_value(right_row, "f1")),
                "delta_f1": csv_decimal(
                    metric_delta(_row_value(left_row, "f1"), _row_value(right_row, "f1"))
                ),
            }
        )
    return rows


def aggregate_delta_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["tolerance_ms"]), row["mode"])].append(row)
    result: list[dict[str, object]] = []
    for tolerance_ms, mode in sorted(grouped, key=lambda value: (value[0], _MODES[value[1]])):
        group = grouped[(tolerance_ms, mode)]
        aggregate: dict[str, object] = {
            "tolerance_ms": tolerance_ms,
            "mode": mode,
            "row_count": len(group),
        }
        for metric in ("precision", "recall", "f1"):
            values = [
                Decimal(row[f"delta_{metric}"]) for row in group if row[f"delta_{metric}"] != ""
            ]
            aggregate[f"mean_delta_{metric}"] = (
                Decimal(str(mean(values))).quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN)
                if values
                else None
            )
            aggregate[f"median_delta_{metric}"] = (
                Decimal(str(median(values))).quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN)
                if values
                else None
            )
        result.append(aggregate)
    return result


def population(
    evidence: PublishedRunEvidence | object,
    selected_ids: set[str] | None,
) -> dict[str, int]:
    statuses = [
        _item_status(item)
        for item_id, item in _items(evidence).items()
        if selected_ids is None or str(item_id) in selected_ids
    ]
    counts = Counter(statuses)
    return {
        "total_count": len(statuses),
        "eligible_count": len(statuses) - counts["quarantined"],
        "success_count": counts["success"],
        "failed_count": counts["failed"],
        "skipped_count": counts["skipped"],
        "quarantined_count": counts["quarantined"],
    }


def runtime(snapshot: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "aggregate_rtf",
        "projected_full_wall_time_sec",
        "measured_wall_time_sec",
        "measured_audio_duration_sec",
        "peak_process_rss_bytes",
        "device_peak_memory_bytes",
        "device",
        "dtype",
    )
    return {field_name: snapshot[field_name] for field_name in fields if field_name in snapshot}


def _identity_values(identity: object) -> dict[str, object]:
    report_values = getattr(identity, "report_values", None)
    if callable(report_values):
        return dict(report_values())
    return {
        field_name: getattr(identity, field_name)
        for field_name in (
            "cohort_id",
            "model_id",
            "model_lock_sha256",
            "prediction_map_version",
            "input_view_id",
            "scoring_version",
        )
    }


def comparison_summary(
    left: PublishedRunEvidence,
    right: PublishedRunEvidence,
    pairable_ids: set[str],
    exclusions: Mapping[str, int],
    song_rows: list[dict[str, str]],
    class_rows: list[dict[str, str]],
    reference_manifest: object,
    timing_manifest: object,
    subset_path: Path | None,
    subset_manifest: object | None,
    *,
    schema: str = COMPARISON_SCHEMA,
    identity: Mapping[str, object] | None = None,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> dict[str, object]:
    schema = _label(schema, "schema")
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    if identity is None:
        summary_identity: Mapping[str, object] = {
            "reference_manifest_sha256": getattr(reference_manifest, "manifest_sha256"),
            "reference_manifest_version": getattr(reference_manifest, "corpus_version"),
            "reference_timing_manifest_sha256": getattr(timing_manifest, "manifest_sha256"),
            "reference_timing_version": getattr(timing_manifest, "corpus_version"),
            "input_view_id": getattr(left.identity, "input_view_id"),
        }
    elif isinstance(identity, Mapping):
        summary_identity = dict(identity)
    else:
        raise TypeError("identity must be a mapping")
    return {
        "schema": schema,
        "identity": summary_identity,
        "subset_manifest": (
            None
            if subset_path is None
            else {
                "path": str(subset_path),
                "manifest_sha256": getattr(subset_manifest, "manifest_sha256"),
                "corpus_version": getattr(subset_manifest, "corpus_version"),
                "review_policy_version": getattr(subset_manifest, "review_policy_version"),
                "review_ledger_sha256": getattr(subset_manifest, "review_ledger_sha256"),
            }
        ),
        "models": {
            left_label: {
                **_identity_values(left.identity),
                "population": population(left, None),
                "runtime": runtime(left.snapshot),
            },
            right_label: {
                **_identity_values(right.identity),
                "population": population(right, None),
                "runtime": runtime(right.snapshot),
            },
        },
        "pairing": {
            "pairable_success_intersection": len(pairable_ids),
            "paired_song_row_count": len(song_rows),
            "paired_class_row_count": len(class_rows),
            "exclusions": dict(exclusions),
        },
        "aggregates": {
            "song": aggregate_delta_rows(song_rows),
            "class": aggregate_delta_rows(class_rows),
        },
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Decimal):
        return csv_decimal(value)
    return str(value)


def write_markdown(
    path: Path,
    summary: Mapping[str, object],
    *,
    title: str = COMPARISON_TITLE,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> None:
    title = _label(title, "title")
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    identity = summary["identity"]
    models = summary["models"]
    pairing = summary["pairing"]
    aggregates = summary["aggregates"]
    assert isinstance(identity, Mapping)
    assert isinstance(models, Mapping)
    assert isinstance(pairing, Mapping)
    assert isinstance(aggregates, Mapping)
    lines = [
        f"# {title}",
        "",
        "## Identity",
        "",
        *[f"- {field_name}: `{identity[field_name]}`" for field_name in identity],
        "",
        "## Population",
        "",
        "| model | total | eligible | success | failed | skipped | quarantined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in (left_label, right_label):
        model = models[model_name]
        population_value = model["population"]
        lines.append(
            f"| {model_name} | {population_value['total_count']} | "
            f"{population_value['eligible_count']} | {population_value['success_count']} | "
            f"{population_value['failed_count']} | {population_value['skipped_count']} | "
            f"{population_value['quarantined_count']} |"
        )
    lines.extend(
        [
            "",
            "## Pairing",
            "",
            f"- pairable_success_intersection: {pairing['pairable_success_intersection']}",
            f"- paired_song_row_count: {pairing['paired_song_row_count']}",
            f"- paired_class_row_count: {pairing['paired_class_row_count']}",
            "",
            "### Exclusions",
            "",
        ]
    )
    exclusions = pairing["exclusions"]
    lines.extend(f"- {key}: {value}" for key, value in exclusions.items())
    for label in ("song", "class"):
        lines.extend(
            [
                "",
                f"## {label.title()} Delta Aggregates",
                "",
                "| tolerance_ms | mode | rows | mean Δ precision | median Δ precision | "
                "mean Δ recall | median Δ recall | mean Δ F1 | median Δ F1 |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregates[label]:
            lines.append(
                f"| {row['tolerance_ms']} | {row['mode']} | {row['row_count']} | "
                f"{markdown_metric(row['mean_delta_precision'])} | "
                f"{markdown_metric(row['median_delta_precision'])} | "
                f"{markdown_metric(row['mean_delta_recall'])} | "
                f"{markdown_metric(row['median_delta_recall'])} | "
                f"{markdown_metric(row['mean_delta_f1'])} | "
                f"{markdown_metric(row['median_delta_f1'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_artifacts(
    output_dir: Path,
    song_rows: list[dict[str, str]],
    class_rows: list[dict[str, str]],
    summary: Mapping[str, object],
    *,
    title: str = COMPARISON_TITLE,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> None:
    """Persist the established deterministic comparison artifact names."""
    left_label = _label(left_label, "left_label")
    right_label = _label(right_label, "right_label")
    song_fields = (
        "simfile_id",
        "tolerance_ms",
        "mode",
        f"{left_label}_precision",
        f"{right_label}_precision",
        "delta_precision",
        f"{left_label}_recall",
        f"{right_label}_recall",
        "delta_recall",
        f"{left_label}_f1",
        f"{right_label}_f1",
        "delta_f1",
    )
    class_fields = (
        "simfile_id",
        "tolerance_ms",
        "mode",
        "common_class",
        f"{left_label}_reference_support",
        f"{right_label}_reference_support",
        f"{left_label}_prediction_support",
        f"{right_label}_prediction_support",
        f"{left_label}_precision",
        f"{right_label}_precision",
        "delta_precision",
        f"{left_label}_recall",
        f"{right_label}_recall",
        "delta_recall",
        f"{left_label}_f1",
        f"{right_label}_f1",
        "delta_f1",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("paired_per_song.csv", "paired_per_class.csv", "summary.json", "summary.md")
    with tempfile.TemporaryDirectory(prefix=".comparison-stage-", dir=output_dir) as stage_name:
        staged = Path(stage_name)
        write_csv(staged / names[0], song_fields, song_rows)
        write_csv(staged / names[1], class_fields, class_rows)
        (staged / names[2]).write_bytes(canonical_json_bytes(summary))
        write_markdown(
            staged / names[3],
            summary,
            title=title,
            left_label=left_label,
            right_label=right_label,
        )
        for name in names:
            os.replace(staged / name, output_dir / name)


# Private names retained as a narrow migration seam for the existing
# MuScriptor coverage suite.  The implementation lives in this module.
_metric_delta = metric_delta
_csv_decimal = csv_decimal
_paired_song_rows = paired_song_rows
_paired_class_rows = paired_class_rows
_aggregate_rows = aggregate_delta_rows
_population = population
_runtime = runtime
_summary = comparison_summary
_write_csv = write_csv
_markdown_metric = markdown_metric
_write_markdown = write_markdown


__all__ = [
    "COMPARISON_SCHEMA",
    "COMPARISON_TITLE",
    "ComparisonIntegrityError",
    "PublishedRunItem",
    "PublishedRunEvidence",
    "pairable_success_ids",
    "selected_rows",
    "metric_delta",
    "csv_decimal",
    "paired_song_rows",
    "paired_class_rows",
    "aggregate_delta_rows",
    "population",
    "runtime",
    "comparison_summary",
    "write_csv",
    "markdown_metric",
    "write_markdown",
    "write_comparison_artifacts",
]
