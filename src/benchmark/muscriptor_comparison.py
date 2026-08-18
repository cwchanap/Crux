"""Join published HPA-325 OaF and MuScriptor reports."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
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
from src.benchmark.cohort_scoring import SCORING_VERSION, CohortIdentity
from src.benchmark.published_comparison import (
    COMPARISON_SCHEMA,
    ComparisonIntegrityError,
    _aggregate_rows,
    _csv_decimal,
    _markdown_metric,
    _metric_delta,
    _paired_class_rows,
    _paired_song_rows,
    _population,
    _runtime,
    _summary,
    _write_csv,
    _write_markdown,
    pairable_success_ids,
    selected_rows,
)
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reports import (
    _ITEM_FIELDNAMES,
    _PER_CLASS_FIELDNAMES,
    _PER_SONG_FIELDNAMES,
    ReportIntegrityError,
    _bounded_csv_metric,
    _csv_int,
    _json_text,
    _parse_csv_decimal,
    _parse_item_rows,
    _read_report_csv,
    read_cohort_reports,
)
from src.benchmark.reports import (
    _parse_class_rows as _parse_published_class_rows,
)
from src.benchmark.reports import (
    _parse_song_rows as _parse_published_song_rows,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION


def load_reviewed_subset_manifest(path: Path):
    """Load HPA-327 only when an optional subset filter is requested."""
    from src.benchmark.reviewed_subset import (  # pylint: disable=import-outside-toplevel
        load_reviewed_subset_manifest as load_manifest,
    )

    return load_manifest(path)


_REPORT_IDENTITY_FIELDS = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
)
# Compatibility aliases for the existing tests; report.py owns the schemas.
_ITEM_FIELDS = _ITEM_FIELDNAMES
_SONG_FIELDS = _PER_SONG_FIELDNAMES
_CLASS_FIELDS = _PER_CLASS_FIELDNAMES
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


def _compat_report_identity(identity: _RunIdentity) -> CohortIdentity:
    """Build the report-reader identity for legacy private-parser callers."""
    return CohortIdentity(
        cohort_id=identity.cohort_id,
        reference_manifest_sha256=identity.reference_manifest_sha256,
        reference_timing_version=identity.reference_timing_version,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=identity.backend_id,
        model_id=identity.model_id,
        model_lock_sha256=identity.model_lock_sha256,
        backend_descriptor_sha256="0" * 64,
        prediction_map_version=identity.prediction_map_version,
        input_view_id=identity.input_view_id,
        scoring_version=identity.scoring_version,
    )


def _text(value: object, field: str) -> str:
    try:
        return _json_text(value, field)
    except ReportIntegrityError as error:
        _fail(str(error))
    raise AssertionError("unreachable")


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
    try:
        return _csv_int(value, field, positive=positive)
    except ReportIntegrityError as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _parse_decimal(value: object, field: str, *, optional: bool = False) -> Decimal | None:
    try:
        return _parse_csv_decimal(value, field, optional=optional)
    except ReportIntegrityError as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _parse_metric(value: object, field: str, *, optional: bool = True) -> Decimal | None:
    try:
        if field in {"precision", "recall", "f1"}:
            return _bounded_csv_metric(value, field)
        parsed = _parse_csv_decimal(value, field, optional=optional)
        if (
            parsed is not None
            and field
            in {"prediction_to_reference_ratio", "median_abs_error_ms", "p95_abs_error_ms"}
            and parsed < 0
        ):
            _fail(f"{field} numeric field is out of range")
        return parsed
    except ReportIntegrityError as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        return _read_report_csv(path, fields)
    except ReportIntegrityError as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _validate_report_identity(row: Mapping[str, str], identity: _RunIdentity) -> None:
    expected = identity.report_values()
    for field in _REPORT_IDENTITY_FIELDS:
        if row.get(field) != expected[field]:
            _fail(f"report identity mismatch for {field}")


def _parse_items(path: Path, identity: _RunIdentity) -> dict[str, str]:
    try:
        rows = _parse_item_rows(path, _compat_report_identity(identity))
    except ReportIntegrityError as error:
        _fail(str(error))
    return {row.simfile_id: row.status for row in rows}


def _parse_song_rows(
    path: Path,
    identity: _RunIdentity,
    successful_ids: set[str],
) -> dict[tuple[str, int, str], _SongRow]:
    try:
        published = _parse_published_song_rows(
            path, _compat_report_identity(identity), successful_ids
        )
    except ReportIntegrityError as error:
        _fail(str(error))
    return {
        (row.simfile_id, row.tolerance_ms, row.mode): _SongRow(
            row.simfile_id, row.tolerance_ms, row.mode, row.precision, row.recall, row.f1
        )
        for row in published
    }


def _parse_class_rows(
    path: Path,
    identity: _RunIdentity,
    successful_ids: set[str],
) -> dict[tuple[str, int, str, str], _ClassRow]:
    try:
        published = _parse_published_class_rows(
            path, _compat_report_identity(identity), successful_ids
        )
    except ReportIntegrityError as error:
        _fail(str(error))
    return {
        (row.simfile_id, row.tolerance_ms, row.mode, row.common_class): _ClassRow(
            row.simfile_id,
            row.tolerance_ms,
            row.mode,
            row.common_class,
            row.reference_support,
            row.prediction_support,
            row.precision,
            row.recall,
            row.f1,
        )
        for row in published
    }


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


def _report_identity_from_snapshot(snapshot: Mapping[str, object]) -> CohortIdentity:
    schema = snapshot.get("schema")
    try:
        if schema == "crux.oaf-corpus-run/v1":
            from src.benchmark.oaf_corpus_run import _cohort_identity_from_snapshot

            return _cohort_identity_from_snapshot(snapshot)
        elif schema == "crux.muscriptor-corpus-run/v1":
            from src.benchmark.muscriptor_corpus_run import _cohort_identity_from_snapshot

            return _cohort_identity_from_snapshot(snapshot)
        else:
            _fail("run snapshot schema is unsupported")
    except (TypeError, ValueError, StrictJsonError) as error:
        _fail(f"run snapshot cohort identity is invalid: {error}")
    raise AssertionError("unreachable")


def _load_evidence(
    run_path: Path,
    *,
    expected_backend_id: str | None = None,
    argument: str | None = None,
) -> _RunEvidence:
    snapshot, identity, run_items = _parse_run(run_path)
    if expected_backend_id is not None:
        _validate_backend_family(
            identity,
            expected_backend_id=expected_backend_id,
            argument=argument or "run",
        )
    reports_path = run_path.parent / "reports"
    try:
        published = read_cohort_reports(
            reports_path,
            expected_identity=_report_identity_from_snapshot(snapshot),
        )
    except ReportIntegrityError as error:
        _fail(str(error))
    items_report = {row.simfile_id: row.status for row in published.items}
    if set(items_report) != set(run_items):
        _fail("items report population does not match run snapshot")
    if any(items_report[item_id] != item.status for item_id, item in run_items.items()):
        _fail("items report status does not match run snapshot")
    songs_report = {
        (row.simfile_id, row.tolerance_ms, row.mode): _SongRow(
            row.simfile_id, row.tolerance_ms, row.mode, row.precision, row.recall, row.f1
        )
        for row in published.songs
    }
    classes_report = {
        (row.simfile_id, row.tolerance_ms, row.mode, row.common_class): _ClassRow(
            row.simfile_id,
            row.tolerance_ms,
            row.mode,
            row.common_class,
            row.reference_support,
            row.prediction_support,
            row.precision,
            row.recall,
            row.f1,
        )
        for row in published.classes
    }
    reports = _Reports(
        items_report,
        songs_report,
        classes_report,
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
) -> tuple[set[str] | None, object | None]:
    if subset_path is None:
        return None, None
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
    return ids, subset


def _pairable_ids(
    oaf: _RunEvidence,
    muscriptor: _RunEvidence,
    selected_ids: set[str] | None,
) -> tuple[set[str], dict[str, int]]:
    return pairable_success_ids(
        oaf,
        muscriptor,
        selected_ids,
        require_identical_input_hash=True,
        left_label="oaf",
        right_label="muscriptor",
    )


def _selected_rows[T](rows: Mapping[T, object], selected_ids: set[str] | None) -> dict[T, object]:
    return selected_rows(rows, selected_ids)


# The model-neutral comparison implementation lives in published_comparison.py.
_metric_delta = _metric_delta
_csv_decimal = _csv_decimal
_paired_song_rows = _paired_song_rows
_paired_class_rows = _paired_class_rows
_aggregate_rows = _aggregate_rows
_population = _population
_runtime = _runtime
_summary = _summary
_write_csv = _write_csv
_markdown_metric = _markdown_metric
_write_markdown = _write_markdown


def compare_oaf_muscriptor(request: ComparisonRequest) -> ComparisonOutcome:
    """Join two persisted HPA-325 report directories without scoring."""
    if not isinstance(request, ComparisonRequest):
        raise TypeError("request must be ComparisonRequest")
    try:
        reference_manifest = load_reference_set_manifest(request.reference_manifest_path)
        timing_manifest = load_reference_timing_manifest(request.timing_manifest_path)
        oaf = _load_evidence(
            request.oaf_run_path,
            expected_backend_id=OAF_BACKEND_ID,
            argument="--oaf-run",
        )
        muscriptor = _load_evidence(
            request.muscriptor_run_path,
            expected_backend_id=MUSCRIPTOR_BACKEND_ID,
            argument="--muscriptor-run",
        )
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
        selected_ids, subset_manifest = _subset_ids(
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
            subset_manifest,
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
