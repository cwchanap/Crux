"""Join published HPA-325 OaF and MuScriptor reports."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median
from typing import Literal

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    MUSCRIPTOR_BACKEND_ID,
    OAF_BACKEND_ID,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest


def load_reviewed_subset_manifest(path: Path):
    """Load HPA-327 only when an optional subset filter is requested."""
    from src.benchmark.reviewed_subset import (  # pylint: disable=import-outside-toplevel
        load_reviewed_subset_manifest as load_manifest,
    )

    return load_manifest(path)


COMPARISON_SCHEMA = "crux.oaf-muscriptor-comparison/v1"

_REPORT_IDENTITY_FIELDS = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
)
_ITEM_FIELDS = (
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
_SONG_FIELDS = (
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
_CLASS_FIELDS = (
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
_SONG_OUTPUT_FIELDS = (
    "simfile_id",
    "tolerance_ms",
    "mode",
    "oaf_precision",
    "muscriptor_precision",
    "delta_precision",
    "oaf_recall",
    "muscriptor_recall",
    "delta_recall",
    "oaf_f1",
    "muscriptor_f1",
    "delta_f1",
)
_CLASS_OUTPUT_FIELDS = (
    "simfile_id",
    "tolerance_ms",
    "mode",
    "common_class",
    "oaf_reference_support",
    "muscriptor_reference_support",
    "oaf_prediction_support",
    "muscriptor_prediction_support",
    "oaf_precision",
    "muscriptor_precision",
    "delta_precision",
    "oaf_recall",
    "muscriptor_recall",
    "delta_recall",
    "oaf_f1",
    "muscriptor_f1",
    "delta_f1",
)
_MODES = {"raw": 0, "aligned": 1}
_STATUS_BY_DISPOSITION = {
    "inferred": "success",
    "resumed": "success",
    "failed": "failed",
    "skipped": "skipped",
    "quarantined": "quarantined",
}
_SCORE_COUNT_FIELDS = ("tp", "fp", "fn")
_SIX_PLACES = Decimal("0.000001")


class ComparisonIntegrityError(ValueError):
    """Raised when published comparison evidence cannot be joined safely."""


@dataclass(frozen=True)
class ComparisonRequest:
    """Inputs for one published-report comparison."""

    oaf_run_path: Path
    muscriptor_run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    output_dir: Path
    subset_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        for field in (
            "oaf_run_path",
            "muscriptor_run_path",
            "reference_manifest_path",
            "timing_manifest_path",
            "output_dir",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if self.subset_manifest_path is not None and not isinstance(
            self.subset_manifest_path, Path
        ):
            raise TypeError("subset_manifest_path must be a Path or None")


@dataclass(frozen=True)
class ComparisonOutcome:
    """Published comparison paths and small CLI-friendly counts."""

    exit_code: Literal[0] = 0
    output_dir: Path = Path()
    pairable_success_count: int = 0
    paired_song_count: int = 0
    paired_class_count: int = 0

    def __post_init__(self) -> None:
        if self.exit_code != 0:
            raise ValueError("successful comparison exit_code must be 0")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path")
        for field in (
            "pairable_success_count",
            "paired_song_count",
            "paired_class_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")


@dataclass(frozen=True)
class _RunIdentity:
    cohort_id: str
    model_id: str
    backend_id: str
    model_lock_sha256: str
    prediction_map_version: str
    input_view_id: str
    reference_manifest_sha256: str
    reference_manifest_version: str
    reference_timing_manifest_sha256: str
    reference_timing_version: str
    scoring_version: str

    def report_values(self) -> dict[str, str]:
        return {
            "cohort_id": self.cohort_id,
            "model_id": self.model_id,
            "model_lock_sha256": self.model_lock_sha256,
            "prediction_map_version": self.prediction_map_version,
            "input_view_id": self.input_view_id,
            "scoring_version": self.scoring_version,
        }


@dataclass(frozen=True)
class _RunItem:
    simfile_id: str
    status: str
    source_audio_sha256: str | None
    input_audio_sha256: str | None


@dataclass(frozen=True)
class _SongRow:
    simfile_id: str
    tolerance_ms: int
    mode: str
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None


@dataclass(frozen=True)
class _ClassRow:
    simfile_id: str
    tolerance_ms: int
    mode: str
    common_class: str
    reference_support: int
    prediction_support: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None


@dataclass(frozen=True)
class _Reports:
    items: dict[str, str]
    songs: dict[tuple[str, int, str], _SongRow]
    classes: dict[tuple[str, int, str, str], _ClassRow]


@dataclass(frozen=True)
class _RunEvidence:
    identity: _RunIdentity
    items: dict[str, _RunItem]
    reports: _Reports
    snapshot: Mapping[str, object]


def _fail(message: str) -> None:
    raise ComparisonIntegrityError(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a nonempty string")
    return value


def _hash(value: object, field: str) -> str:
    try:
        return require_sha256(_text(value, field), field)
    except (StrictJsonError, ValueError) as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _parse_simfile_id(value: object, field: str = "simfile_id") -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        _fail(f"{field} is malformed")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        _fail(f"{field} is malformed")
    return value


def _parse_int(value: object, field: str, *, positive: bool = False) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        _fail(f"{field} numeric field is malformed")
    parsed = int(value)
    if (positive and parsed <= 0) or parsed < 0:
        _fail(f"{field} numeric field is malformed")
    return parsed


def _parse_decimal(value: object, field: str, *, optional: bool = False) -> Decimal | None:
    if value == "" and optional:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{field} numeric field is malformed")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        _fail(f"{field} numeric field is malformed")
    if not parsed.is_finite():
        _fail(f"{field} numeric field is malformed")
    return parsed


def _parse_metric(value: object, field: str, *, optional: bool = True) -> Decimal | None:
    parsed = _parse_decimal(value, field, optional=optional)
    if parsed is not None and field in {"precision", "recall", "f1"} and not 0 <= parsed <= 1:
        _fail(f"{field} numeric field is out of range")
    if (
        parsed is not None
        and field
        in {
            "prediction_to_reference_ratio",
            "median_abs_error_ms",
            "p95_abs_error_ms",
        }
        and parsed < 0
    ):
        _fail(f"{field} numeric field is out of range")
    return parsed


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        content = read_regular_file_no_follow(path).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        _fail(f"cannot read report {path.name}: {error}")
    reader = csv.DictReader(io.StringIO(content, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
        _fail(f"{path.name} has an invalid column schema")
    rows: list[dict[str, str]] = []
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                _fail(f"{path.name} contains a malformed row")
            rows.append({key: value for key, value in row.items() if key is not None})
    except csv.Error as error:
        _fail(f"{path.name} contains malformed CSV: {error}")
    return rows


def _validate_report_identity(row: Mapping[str, str], identity: _RunIdentity) -> None:
    expected = identity.report_values()
    for field in _REPORT_IDENTITY_FIELDS:
        if row.get(field) != expected[field]:
            _fail(f"report identity mismatch for {field}")


def _parse_items(path: Path, identity: _RunIdentity) -> dict[str, str]:
    items: dict[str, str] = {}
    for row in _read_csv(path, _ITEM_FIELDS):
        if row["cohort_id"] != identity.cohort_id:
            _fail("items report identity mismatch for cohort_id")
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id in items:
            _fail("items report contains duplicate simfile_id")
        status = row["status"]
        if status not in {"success", "failed", "skipped", "quarantined"}:
            _fail("items report contains an invalid status")
        for field in (
            "reference_native_event_count",
            "reference_common_event_count",
            "reference_ignored_event_count",
            "reference_unmapped_event_count",
            "reference_duplicate_collapsed_count",
        ):
            _parse_int(row[field], field)
        for field in (
            "prediction_native_event_count",
            "prediction_mapped_event_count",
            "prediction_unmapped_event_count",
        ):
            if row[field] != "":
                _parse_int(row[field], field)
        _parse_metric(row["prediction_mapping_coverage"], "prediction_mapping_coverage")
        if row["prediction_native_class_counts"]:
            for value in row["prediction_native_class_counts"].split("|"):
                if "=" not in value:
                    _fail("prediction_native_class_counts numeric field is malformed")
                native_class, count = value.rsplit("=", 1)
                if not native_class:
                    _fail("prediction_native_class_counts numeric field is malformed")
                _parse_int(count, "prediction_native_class_counts")
        items[simfile_id] = status
    return items


def _parse_song_rows(
    path: Path,
    identity: _RunIdentity,
    successful_ids: set[str],
) -> dict[tuple[str, int, str], _SongRow]:
    rows: dict[tuple[str, int, str], _SongRow] = {}
    for row in _read_csv(path, _SONG_FIELDS):
        _validate_report_identity(row, identity)
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id not in successful_ids:
            _fail("per_song report contains a non-success item")
        tolerance_ms = _parse_int(row["tolerance_ms"], "tolerance_ms", positive=True)
        mode = row["mode"]
        if mode not in _MODES:
            _fail("per_song report contains an invalid mode")
        for field in _SCORE_COUNT_FIELDS:
            _parse_int(row[field], field)
        precision = _parse_metric(row["precision"], "precision")
        recall = _parse_metric(row["recall"], "recall")
        f1 = _parse_metric(row["f1"], "f1")
        _parse_metric(row["prediction_to_reference_ratio"], "prediction_to_reference_ratio")
        _parse_metric(row["median_abs_error_ms"], "median_abs_error_ms")
        _parse_metric(row["p95_abs_error_ms"], "p95_abs_error_ms")
        _parse_metric(row["offset_ms"], "offset_ms")
        key = (simfile_id, tolerance_ms, mode)
        if key in rows:
            _fail("per_song report contains duplicate score key")
        rows[key] = _SongRow(simfile_id, tolerance_ms, mode, precision, recall, f1)
    return rows


def _parse_class_rows(
    path: Path,
    identity: _RunIdentity,
    successful_ids: set[str],
) -> dict[tuple[str, int, str, str], _ClassRow]:
    rows: dict[tuple[str, int, str, str], _ClassRow] = {}
    for row in _read_csv(path, _CLASS_FIELDS):
        _validate_report_identity(row, identity)
        simfile_id = _parse_simfile_id(row["simfile_id"])
        if simfile_id not in successful_ids:
            _fail("per_class report contains a non-success item")
        tolerance_ms = _parse_int(row["tolerance_ms"], "tolerance_ms", positive=True)
        mode = row["mode"]
        if mode not in _MODES:
            _fail("per_class report contains an invalid mode")
        common_class = row["common_class"]
        if not common_class:
            _fail("per_class report contains an invalid common_class")
        for field in _SCORE_COUNT_FIELDS + ("reference_support", "prediction_support"):
            _parse_int(row[field], field)
        precision = _parse_metric(row["precision"], "precision")
        recall = _parse_metric(row["recall"], "recall")
        f1 = _parse_metric(row["f1"], "f1")
        key = (simfile_id, tolerance_ms, mode, common_class)
        if key in rows:
            _fail("per_class report contains duplicate score key")
        rows[key] = _ClassRow(
            simfile_id,
            tolerance_ms,
            mode,
            common_class,
            _parse_int(row["reference_support"], "reference_support"),
            _parse_int(row["prediction_support"], "prediction_support"),
            precision,
            recall,
            f1,
        )
    return rows


def _parse_run_identity(snapshot: Mapping[str, object]) -> _RunIdentity:
    cohort_id = _text(snapshot.get("run_id"), "run_id")
    descriptor = snapshot.get("backend_descriptor")
    descriptor_model = descriptor.get("model_id") if isinstance(descriptor, Mapping) else None
    descriptor_backend = descriptor.get("backend_id") if isinstance(descriptor, Mapping) else None
    model_id = _text(snapshot.get("model_id", descriptor_model), "model_id")
    backend_id = _text(descriptor_backend, "backend_descriptor.backend_id")
    inference_config = snapshot.get("inference_config")
    configured_map = (
        inference_config.get("prediction_map_version")
        if isinstance(inference_config, Mapping)
        else None
    )
    prediction_map = snapshot.get("prediction_map_version", configured_map)
    if configured_map is not None and prediction_map != configured_map:
        _fail("run snapshot prediction_map_version is internally inconsistent")
    configured_view = (
        inference_config.get("input_view_id") if isinstance(inference_config, Mapping) else None
    )
    input_view = snapshot.get("input_view_id", configured_view)
    if configured_view is not None and input_view != configured_view:
        _fail("run snapshot input_view_id is internally inconsistent")
    scoring_version = snapshot.get("scoring_version", SCORING_VERSION)
    if scoring_version != SCORING_VERSION:
        _fail("run snapshot scoring_version is invalid")
    reference_manifest_version = snapshot.get("reference_manifest_version", "")
    if not isinstance(reference_manifest_version, str):
        _fail("reference_manifest_version must be a string")
    return _RunIdentity(
        cohort_id,
        model_id,
        backend_id,
        _hash(snapshot.get("model_lock_sha256"), "model_lock_sha256"),
        _text(prediction_map, "prediction_map_version"),
        _text(input_view, "input_view_id"),
        _hash(snapshot.get("reference_manifest_sha256"), "reference_manifest_sha256"),
        reference_manifest_version,
        _hash(
            snapshot.get("reference_timing_manifest_sha256"),
            "reference_timing_manifest_sha256",
        ),
        _text(snapshot.get("reference_timing_version"), "reference_timing_version"),
        SCORING_VERSION,
    )


def _parse_run(path: Path) -> tuple[Mapping[str, object], _RunIdentity, dict[str, _RunItem]]:
    snapshot: Mapping[str, object] = {}
    try:
        content = read_regular_file_no_follow(path)
        value = strict_json_loads(content, require_canonical=True)
        if not isinstance(value, dict):
            _fail("run snapshot must be an object")
        schema = value.get("schema")
        if schema == "crux.oaf-corpus-run/v1":
            from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run

            snapshot = parse_oaf_corpus_run(content)
        elif schema == "crux.muscriptor-corpus-run/v1":
            from src.benchmark.muscriptor_corpus_run import parse_muscriptor_corpus_run

            snapshot = parse_muscriptor_corpus_run(content)
        else:
            _fail("run snapshot schema is unsupported")
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        _fail(f"invalid run snapshot: {error}")
    identity = _parse_run_identity(snapshot)
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        _fail("run snapshot items must be an array")
    parsed_items: dict[str, _RunItem] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            _fail("run snapshot item must be an object")
        raw_id = raw.get("simfile_id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id <= 0:
            _fail("run snapshot item simfile_id is malformed")
        simfile_id = str(raw_id)
        if simfile_id in parsed_items:
            _fail("run snapshot contains duplicate simfile_id")
        disposition = raw.get("execution_disposition")
        status = _STATUS_BY_DISPOSITION.get(disposition)
        if status is None:
            _fail("run snapshot item execution disposition is missing or invalid")
        source_hash = raw.get("source_audio_sha256")
        input_hash = raw.get("input_audio_sha256")
        if status == "success":
            source_hash = _hash(source_hash, "source_audio_sha256")
            input_hash = _hash(input_hash, "input_audio_sha256")
        elif source_hash is not None and not isinstance(source_hash, str):
            _fail("source_audio_sha256 is malformed")
        elif input_hash is not None and not isinstance(input_hash, str):
            _fail("input_audio_sha256 is malformed")
        parsed_items[simfile_id] = _RunItem(simfile_id, status, source_hash, input_hash)
    return snapshot, identity, parsed_items


def _load_evidence(run_path: Path) -> _RunEvidence:
    snapshot, identity, run_items = _parse_run(run_path)
    reports_path = run_path.parent / "reports"
    items_report = _parse_items(reports_path / "items.csv", identity)
    if set(items_report) != set(run_items):
        _fail("items report population does not match run snapshot")
    if any(items_report[item_id] != item.status for item_id, item in run_items.items()):
        _fail("items report status does not match run snapshot")
    successful_ids = {item_id for item_id, item in run_items.items() if item.status == "success"}
    reports = _Reports(
        items_report,
        _parse_song_rows(reports_path / "per_song.csv", identity, successful_ids),
        _parse_class_rows(reports_path / "per_class.csv", identity, successful_ids),
    )
    return _RunEvidence(identity, run_items, reports, snapshot)


def _validate_manifest_lineage(
    evidence: _RunEvidence,
    reference_manifest: object,
    timing_manifest: object,
) -> None:
    reference_sha = getattr(reference_manifest, "manifest_sha256", None)
    reference_version = getattr(reference_manifest, "corpus_version", None)
    timing_sha = getattr(timing_manifest, "manifest_sha256", None)
    timing_version = getattr(timing_manifest, "corpus_version", None)
    identity = evidence.identity
    if identity.reference_manifest_sha256 != reference_sha:
        _fail("run snapshot reference manifest identity does not match supplied manifest")
    if (
        identity.reference_manifest_version
        and identity.reference_manifest_version != reference_version
    ):
        _fail("run snapshot reference manifest identity does not match supplied manifest")
    if (
        identity.reference_timing_manifest_sha256,
        identity.reference_timing_version,
    ) != (timing_sha, timing_version):
        _fail("run snapshot reference timing identity does not match supplied manifest")


def _validate_backend_family(
    identity: _RunIdentity, *, expected_backend_id: str, argument: str
) -> None:
    if identity.backend_id != expected_backend_id:
        family = "OaF" if expected_backend_id == OAF_BACKEND_ID else "MuScriptor"
        _fail(f"{argument} must contain {family} backend identity")


def _validate_pair_run_identity(oaf: _RunIdentity, muscriptor: _RunIdentity) -> None:
    for field in (
        "reference_manifest_sha256",
        "reference_timing_manifest_sha256",
        "reference_timing_version",
        "input_view_id",
    ):
        if getattr(oaf, field) != getattr(muscriptor, field):
            _fail(f"canonical-input run identity mismatch for {field}")


def _subset_ids(
    subset_path: Path | None,
    reference_manifest: object,
    timing_manifest: object,
    evidence: Iterable[_RunEvidence],
) -> set[str] | None:
    if subset_path is None:
        return None
    try:
        subset = load_reviewed_subset_manifest(subset_path)
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        _fail(f"invalid subset manifest: {error}")
    if (
        subset.source_reference_manifest_sha256
        != getattr(reference_manifest, "manifest_sha256", None)
        or subset.source_reference_manifest_version
        != getattr(reference_manifest, "corpus_version", None)
        or subset.source_timing_manifest_sha256 != getattr(timing_manifest, "manifest_sha256", None)
        or subset.source_timing_manifest_version != getattr(timing_manifest, "corpus_version", None)
    ):
        _fail("subset manifest lineage does not match supplied manifests")
    ids = {str(row.view.simfile_id) for row in subset.rows}
    for run in evidence:
        if not ids <= set(run.items):
            _fail("subset manifest member is absent from a run population")
    return ids


def _pairable_ids(
    oaf: _RunEvidence,
    muscriptor: _RunEvidence,
    selected_ids: set[str] | None,
) -> tuple[set[str], dict[str, int]]:
    oaf_success = {
        item_id
        for item_id, item in oaf.items.items()
        if item.status == "success" and (selected_ids is None or item_id in selected_ids)
    }
    muscriptor_success = {
        item_id
        for item_id, item in muscriptor.items.items()
        if item.status == "success" and (selected_ids is None or item_id in selected_ids)
    }
    common = oaf_success & muscriptor_success
    source_mismatch = 0
    pairable: set[str] = set()
    for simfile_id in sorted(common, key=int):
        oaf_item = oaf.items[simfile_id]
        muscriptor_item = muscriptor.items[simfile_id]
        if oaf_item.source_audio_sha256 != muscriptor_item.source_audio_sha256:
            source_mismatch += 1
            continue
        if oaf_item.input_audio_sha256 != muscriptor_item.input_audio_sha256:
            _fail(
                "canonical-input integrity error: input_audio_sha256 mismatch for "
                f"simfile_id={simfile_id} source_audio_sha256={oaf_item.source_audio_sha256}"
            )
        pairable.add(simfile_id)
    exclusions = {
        "oaf_only_success": len(oaf_success - muscriptor_success),
        "muscriptor_only_success": len(muscriptor_success - oaf_success),
        "source_audio_mismatch": source_mismatch,
    }
    return pairable, exclusions


def _selected_rows[T](rows: Mapping[T, object], selected_ids: set[str] | None) -> dict[T, object]:
    if selected_ids is None:
        return dict(rows)
    return {key: row for key, row in rows.items() if key[0] in selected_ids}  # type: ignore[index]


def _metric_delta(oaf: Decimal | None, muscriptor: Decimal | None) -> Decimal | None:
    if oaf is None or muscriptor is None:
        return None
    value = (muscriptor - oaf).quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN)
    return Decimal(0) if value.is_zero() else value


def _csv_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    if value.is_zero():
        return "0"
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _paired_song_rows(
    oaf: Mapping[tuple[str, int, str], _SongRow],
    muscriptor: Mapping[tuple[str, int, str], _SongRow],
    pairable_ids: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(
        set(oaf) & set(muscriptor),
        key=lambda value: (int(value[0]), value[1], _MODES[value[2]]),
    ):
        if key[0] not in pairable_ids:
            continue
        left, right = oaf[key], muscriptor[key]
        rows.append(
            {
                "simfile_id": key[0],
                "tolerance_ms": str(key[1]),
                "mode": key[2],
                "oaf_precision": _csv_decimal(left.precision),
                "muscriptor_precision": _csv_decimal(right.precision),
                "delta_precision": _csv_decimal(_metric_delta(left.precision, right.precision)),
                "oaf_recall": _csv_decimal(left.recall),
                "muscriptor_recall": _csv_decimal(right.recall),
                "delta_recall": _csv_decimal(_metric_delta(left.recall, right.recall)),
                "oaf_f1": _csv_decimal(left.f1),
                "muscriptor_f1": _csv_decimal(right.f1),
                "delta_f1": _csv_decimal(_metric_delta(left.f1, right.f1)),
            }
        )
    return rows


def _paired_class_rows(
    oaf: Mapping[tuple[str, int, str, str], _ClassRow],
    muscriptor: Mapping[tuple[str, int, str, str], _ClassRow],
    pairable_ids: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(
        set(oaf) & set(muscriptor),
        key=lambda value: (int(value[0]), value[1], _MODES[value[2]], value[3]),
    ):
        if key[0] not in pairable_ids:
            continue
        left, right = oaf[key], muscriptor[key]
        rows.append(
            {
                "simfile_id": key[0],
                "tolerance_ms": str(key[1]),
                "mode": key[2],
                "common_class": key[3],
                "oaf_reference_support": str(left.reference_support),
                "muscriptor_reference_support": str(right.reference_support),
                "oaf_prediction_support": str(left.prediction_support),
                "muscriptor_prediction_support": str(right.prediction_support),
                "oaf_precision": _csv_decimal(left.precision),
                "muscriptor_precision": _csv_decimal(right.precision),
                "delta_precision": _csv_decimal(_metric_delta(left.precision, right.precision)),
                "oaf_recall": _csv_decimal(left.recall),
                "muscriptor_recall": _csv_decimal(right.recall),
                "delta_recall": _csv_decimal(_metric_delta(left.recall, right.recall)),
                "oaf_f1": _csv_decimal(left.f1),
                "muscriptor_f1": _csv_decimal(right.f1),
                "delta_f1": _csv_decimal(_metric_delta(left.f1, right.f1)),
            }
        )
    return rows


def _aggregate_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
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


def _population(run: _RunEvidence, selected_ids: set[str] | None) -> dict[str, int]:
    statuses = [
        item.status
        for item_id, item in run.items.items()
        if selected_ids is None or item_id in selected_ids
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


def _runtime(snapshot: Mapping[str, object]) -> dict[str, object]:
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
    return {field: snapshot[field] for field in fields if field in snapshot}


def _summary(
    oaf: _RunEvidence,
    muscriptor: _RunEvidence,
    pairable_ids: set[str],
    exclusions: Mapping[str, int],
    song_rows: list[dict[str, str]],
    class_rows: list[dict[str, str]],
    reference_manifest: object,
    timing_manifest: object,
    subset_path: Path | None,
) -> dict[str, object]:
    return {
        "schema": COMPARISON_SCHEMA,
        "identity": {
            "reference_manifest_sha256": getattr(reference_manifest, "manifest_sha256"),
            "reference_manifest_version": getattr(reference_manifest, "corpus_version"),
            "reference_timing_manifest_sha256": getattr(timing_manifest, "manifest_sha256"),
            "reference_timing_version": getattr(timing_manifest, "corpus_version"),
            "input_view_id": oaf.identity.input_view_id,
        },
        "subset_manifest": None if subset_path is None else str(subset_path),
        "models": {
            "oaf": {
                **oaf.identity.report_values(),
                "population": _population(oaf, None),
                "runtime": _runtime(oaf.snapshot),
            },
            "muscriptor": {
                **muscriptor.identity.report_values(),
                "population": _population(muscriptor, None),
                "runtime": _runtime(muscriptor.snapshot),
            },
        },
        "pairing": {
            "pairable_success_intersection": len(pairable_ids),
            "paired_song_row_count": len(song_rows),
            "paired_class_row_count": len(class_rows),
            "exclusions": dict(exclusions),
        },
        "aggregates": {
            "song": _aggregate_rows(song_rows),
            "class": _aggregate_rows(class_rows),
        },
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Decimal):
        return _csv_decimal(value)
    return str(value)


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    identity = summary["identity"]
    models = summary["models"]
    pairing = summary["pairing"]
    aggregates = summary["aggregates"]
    assert isinstance(identity, Mapping)
    assert isinstance(models, Mapping)
    assert isinstance(pairing, Mapping)
    assert isinstance(aggregates, Mapping)
    lines = [
        "# OaF/MuScriptor Published Report Comparison",
        "",
        "## Identity",
        "",
        *[f"- {field}: `{identity[field]}`" for field in identity],
        "",
        "## Population",
        "",
        "| model | total | eligible | success | failed | skipped | quarantined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ("oaf", "muscriptor"):
        model = models[model_name]
        population = model["population"]
        lines.append(
            f"| {model_name} | {population['total_count']} | {population['eligible_count']} | "
            f"{population['success_count']} | {population['failed_count']} | "
            f"{population['skipped_count']} | {population['quarantined_count']} |"
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
                f"{_markdown_metric(row['mean_delta_precision'])} | "
                f"{_markdown_metric(row['median_delta_precision'])} | "
                f"{_markdown_metric(row['mean_delta_recall'])} | "
                f"{_markdown_metric(row['median_delta_recall'])} | "
                f"{_markdown_metric(row['mean_delta_f1'])} | "
                f"{_markdown_metric(row['median_delta_f1'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_oaf_muscriptor(request: ComparisonRequest) -> ComparisonOutcome:
    """Join two persisted HPA-325 report directories without scoring."""
    if not isinstance(request, ComparisonRequest):
        raise TypeError("request must be ComparisonRequest")
    try:
        reference_manifest = load_reference_set_manifest(request.reference_manifest_path)
        timing_manifest = load_reference_timing_manifest(request.timing_manifest_path)
        oaf = _load_evidence(request.oaf_run_path)
        muscriptor = _load_evidence(request.muscriptor_run_path)
        _validate_backend_family(
            oaf.identity,
            expected_backend_id=OAF_BACKEND_ID,
            argument="--oaf-run",
        )
        _validate_backend_family(
            muscriptor.identity,
            expected_backend_id=MUSCRIPTOR_BACKEND_ID,
            argument="--muscriptor-run",
        )
        _validate_manifest_lineage(oaf, reference_manifest, timing_manifest)
        _validate_manifest_lineage(muscriptor, reference_manifest, timing_manifest)
        _validate_pair_run_identity(oaf.identity, muscriptor.identity)
        selected_ids = _subset_ids(
            request.subset_manifest_path,
            reference_manifest,
            timing_manifest,
            (oaf, muscriptor),
        )
        pairable_ids, exclusions = _pairable_ids(oaf, muscriptor, selected_ids)
        oaf_songs = _selected_rows(oaf.reports.songs, selected_ids)
        muscriptor_songs = _selected_rows(muscriptor.reports.songs, selected_ids)
        oaf_classes = _selected_rows(oaf.reports.classes, selected_ids)
        muscriptor_classes = _selected_rows(muscriptor.reports.classes, selected_ids)
        song_rows = _paired_song_rows(oaf_songs, muscriptor_songs, pairable_ids)  # type: ignore[arg-type]
        class_rows = _paired_class_rows(oaf_classes, muscriptor_classes, pairable_ids)  # type: ignore[arg-type]
        # No prediction or scorer is loaded by this module.
        summary = _summary(
            oaf,
            muscriptor,
            pairable_ids,
            exclusions,
            song_rows,
            class_rows,
            reference_manifest,
            timing_manifest,
            request.subset_manifest_path,
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        names = ("paired_per_song.csv", "paired_per_class.csv", "summary.json", "summary.md")
        with tempfile.TemporaryDirectory(
            prefix=".comparison-stage-", dir=request.output_dir
        ) as stage_name:
            staged = Path(stage_name)
            _write_csv(staged / "paired_per_song.csv", _SONG_OUTPUT_FIELDS, song_rows)
            _write_csv(staged / "paired_per_class.csv", _CLASS_OUTPUT_FIELDS, class_rows)
            (staged / "summary.json").write_bytes(canonical_json_bytes(summary))
            _write_markdown(staged / "summary.md", summary)
            for name in names:
                os.replace(staged / name, request.output_dir / name)
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError(str(error)) from error
    return ComparisonOutcome(
        output_dir=request.output_dir,
        pairable_success_count=len(pairable_ids),
        paired_song_count=len(song_rows),
        paired_class_count=len(class_rows),
    )


__all__ = [
    "COMPARISON_SCHEMA",
    "ComparisonIntegrityError",
    "ComparisonOutcome",
    "ComparisonRequest",
    "compare_oaf_muscriptor",
]
