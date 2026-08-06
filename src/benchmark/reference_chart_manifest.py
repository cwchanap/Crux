from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.corpus_cache import _validate_relative_cache_path, is_chart_key, is_set_def_key
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    ManifestRowView,
    manifest_row_view_from_row,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.r2_corpus_models import MANIFEST_SCHEMA, PublishedManifest, RemoteObject
from src.benchmark.reference_chart_selection import (
    ChartSelection,
    SelectionMethod,
    SelectionOverride,
    SelectionReasonCode,
    load_selection_overrides,
    select_reference_chart,
)

REFERENCE_CHART_MANIFEST_SCHEMA = "crux.reference-chart-manifest/v1"
_HPA_321_ROW_KEYS = frozenset(
    {
        "schema_version",
        "corpus_version",
        "cache_profile",
        "simfile_id",
        "object_prefix",
        "source_endpoint_sha256",
        "source_bucket",
        "source_discovery_method",
        "objects",
        "sync_status",
        "sync_errors",
        "source_origin",
        "source_author_or_pack",
        "source_reference",
        "rights_status",
        "redistribution_allowed",
        "provenance_notes",
    }
)
_SELECTION_ROW_KEYS = frozenset(
    {
        "source_manifest_sha256",
        "source_corpus_version",
        "selection_status",
        "selection_method",
        "selection_reason_codes",
        "selection_warnings",
        "set_def_key",
        "set_def_content_hash",
        "selected_chart_key",
        "selected_chart_content_hash",
        "selected_chart_cache_path",
        "selected_level_slot",
        "selected_level_label",
        "dlevel_raw",
        "dlevel_normalized",
        "title",
        "artist",
        "override_document_sha256",
        "selection_override",
    }
)
_REFERENCE_ROW_KEYS = _HPA_321_ROW_KEYS | _SELECTION_ROW_KEYS
_SELECTION_METHODS = frozenset(get_args(SelectionMethod))
_SELECTION_REASON_CODES = frozenset(get_args(SelectionReasonCode))
_SELECTION_STATUSES = frozenset({"selected", "quarantined"})
_OVERRIDE_KEYS = frozenset({"chart_key", "reason"})
_SELECTED_IDENTITY_FIELDS = (
    "selected_chart_key",
    "selected_chart_content_hash",
    "selected_chart_cache_path",
)
_QUARANTINED_NULL_FIELDS = (
    *_SELECTED_IDENTITY_FIELDS,
    "selected_level_slot",
    "selected_level_label",
    "dlevel_raw",
    "dlevel_normalized",
    "title",
    "artist",
)


@dataclass(frozen=True)
class _ValidatedSourceRow:
    source_row: Mapping[str, object]
    view: ManifestRowView


@dataclass(frozen=True)
class _LoadedSourceManifest:
    source_manifest_sha256: str
    rows: tuple[_ValidatedSourceRow, ...]


@dataclass(frozen=True)
class SelectionRequest:
    manifest_path: Path
    cache_dir: Path
    overrides_file: Path | None
    output_dir: Path
    default_overrides_missing_ok: bool = False


@dataclass(frozen=True)
class SelectionOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    selected_count: int
    quarantined_count: int


def _load_source_manifest(path: Path) -> _LoadedSourceManifest:
    try:
        content = path.read_bytes()
    except OSError:
        raise ValueError("source manifest is unavailable") from None

    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("source manifest must contain canonical JSONL records")

    rows: list[_ValidatedSourceRow] = []
    simfile_ids: set[int] = set()
    source_identity: tuple[str, str, str, str, str] | None = None
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError("source manifest must contain canonical JSONL records")
        try:
            source_row = strict_json_loads(line[:-1], require_canonical=True)
        except StrictJsonError:
            raise ValueError("source manifest must contain canonical JSONL records") from None
        if not isinstance(source_row, dict) or source_row.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("source manifest contains an unsupported row")
        try:
            view = manifest_row_view_from_row(source_row)
        except ValueError:
            raise ValueError("source manifest contains an invalid HPA-321 row") from None

        identity = (
            view.corpus_version,
            view.source_endpoint_sha256,
            view.source_bucket,
            view.cache_profile,
            view.source_discovery_method,
        )
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise ValueError("source manifest contains mixed source identity")
        if view.inventory.simfile_id in simfile_ids:
            raise ValueError("source manifest contains duplicate simfile IDs")
        simfile_ids.add(view.inventory.simfile_id)
        rows.append(_ValidatedSourceRow(MappingProxyType(source_row), view))

    if not rows:
        raise ValueError("source manifest contains no records")
    return _LoadedSourceManifest(
        source_manifest_sha256=sha256(content).hexdigest(),
        rows=tuple(rows),
    )


def select_reference_manifest(request: SelectionRequest) -> SelectionOutcome:
    try:
        source = _load_source_manifest(request.manifest_path)
        overrides = load_selection_overrides(
            request.overrides_file,
            missing_ok=request.default_overrides_missing_ok,
        )
    except ValueError:
        return _failed_outcome()

    selected_count = 0
    quarantined_count = 0
    try:
        rows: list[dict[str, object]] = []
        for validated in source.rows:
            selection = select_reference_chart(
                validated.view,
                cache_dir=request.cache_dir,
                overrides=overrides,
            )
            if selection.status == "selected":
                selected_count += 1
            else:
                quarantined_count += 1
            rows.append(
                _build_selection_row(
                    validated,
                    source_manifest_sha256=source.source_manifest_sha256,
                    override_document_sha256=overrides.document_sha256,
                    selection=selection,
                )
            )
        rendered = render_manifest(tuple(rows))
        published = publish_manifest(request.output_dir, rendered)
        status: Literal["complete", "partial"] = "complete" if quarantined_count == 0 else "partial"
        publish_latest_manifest(request.output_dir, published, status, datetime.now(timezone.utc))
    except (ManifestPublicationError, OSError, ValueError):
        return _failed_outcome(selected_count, quarantined_count)

    return SelectionOutcome(
        status=status,
        exit_code=0 if quarantined_count == 0 else 1,
        manifest=published,
        selected_count=selected_count,
        quarantined_count=quarantined_count,
    )


def _failed_outcome(selected_count: int = 0, quarantined_count: int = 0) -> SelectionOutcome:
    return SelectionOutcome(
        status="failed",
        exit_code=2,
        manifest=None,
        selected_count=selected_count,
        quarantined_count=quarantined_count,
    )


def _build_selection_row(
    source: _ValidatedSourceRow,
    *,
    source_manifest_sha256: str,
    override_document_sha256: str,
    selection: ChartSelection,
) -> dict[str, object]:
    row = dict(source.source_row)
    source_corpus_version = row.pop("corpus_version")
    row["schema_version"] = REFERENCE_CHART_MANIFEST_SCHEMA
    row.update(
        {
            "source_manifest_sha256": source_manifest_sha256,
            "source_corpus_version": source_corpus_version,
            "selection_status": selection.status,
            "selection_method": selection.method,
            "selection_reason_codes": list(selection.reason_codes),
            "selection_warnings": list(selection.warnings),
            "set_def_key": _set_def_key(selection.set_def),
            "set_def_content_hash": _set_def_content_hash(selection.set_def),
            "selected_chart_key": _selected_chart_key(selection),
            "selected_chart_content_hash": _selected_chart_content_hash(selection),
            "selected_chart_cache_path": _selected_chart_cache_path(selection),
            "selected_level_slot": selection.selected_level_slot,
            "selected_level_label": selection.selected_level_label,
            "dlevel_raw": selection.dlevel_raw,
            "dlevel_normalized": selection.dlevel_normalized,
            "title": selection.title,
            "artist": selection.artist,
            "override_document_sha256": override_document_sha256,
            "selection_override": _selection_override(selection.override),
        }
    )
    _validate_selection_row_shape(row, selection)
    return row


def _set_def_key(remote: RemoteObject | None) -> str | None:
    if remote is None:
        return None
    _require_remote_hash(remote)
    return remote.key


def _set_def_content_hash(remote: RemoteObject | None) -> str | None:
    if remote is None:
        return None
    return _require_remote_hash(remote)


def _selected_chart_key(selection: ChartSelection) -> str | None:
    if selection.selected_chart is None:
        return None
    _require_selected_chart_identity(selection.selected_chart)
    return selection.selected_chart.key


def _selected_chart_content_hash(selection: ChartSelection) -> str | None:
    if selection.selected_chart is None:
        return None
    return _require_selected_chart_identity(selection.selected_chart)[0]


def _selected_chart_cache_path(selection: ChartSelection) -> str | None:
    if selection.selected_chart is None:
        return None
    return _require_selected_chart_identity(selection.selected_chart)[1]


def _require_remote_hash(remote: RemoteObject) -> str:
    if not isinstance(remote.key, str) or not remote.key or not isinstance(remote.sha256, str):
        raise ValueError("selection remote identity is invalid")
    return remote.sha256


def _require_selected_chart_identity(remote: RemoteObject) -> tuple[str, str]:
    digest = _require_remote_hash(remote)
    if not isinstance(remote.cache_path, str) or not remote.cache_path:
        raise ValueError("selected chart identity is invalid")
    return digest, remote.cache_path


def _selection_override(override: SelectionOverride | None) -> dict[str, str] | None:
    if override is None:
        return None
    return {"chart_key": override.chart_key, "reason": override.reason}


def _validate_selection_row_shape(row: Mapping[str, object], selection: ChartSelection) -> None:
    if selection.status == "selected":
        selected_values = (
            row["selected_chart_key"],
            row["selected_chart_content_hash"],
            row["selected_chart_cache_path"],
            row["title"],
            row["artist"],
        )
        if (
            selection.method is None
            or selection.reason_codes
            or any(value is None for value in selected_values)
        ):
            raise ValueError("selected reference chart row is invalid")
        return
    if selection.status == "quarantined":
        quarantined_fields = (
            "selected_chart_key",
            "selected_chart_content_hash",
            "selected_chart_cache_path",
            "selected_level_slot",
            "selected_level_label",
            "dlevel_raw",
            "dlevel_normalized",
            "title",
            "artist",
        )
        if (
            selection.method is not None
            or not selection.reason_codes
            or any(row[field] is not None for field in quarantined_fields)
        ):
            raise ValueError("quarantined reference chart row is invalid")
        return
    raise ValueError("reference chart selection status is invalid")


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != REFERENCE_CHART_MANIFEST_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference chart manifest golden must be canonical JSONL")

    lines = content.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reference chart manifest golden must contain exactly two records")
    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reference chart manifest golden must be canonical JSONL") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("reference chart manifest golden rows must be objects")

    validated = tuple(_validate_reference_row(row) for row in rows)
    statuses = [row["selection_status"] for row in rows]
    if sorted(statuses) != ["quarantined", "selected"]:
        raise ValueError(
            "reference chart manifest golden requires one selected and one quarantined row"
        )

    source_identities = {
        (
            row["source_manifest_sha256"],
            view.corpus_version,
            view.source_endpoint_sha256,
            view.source_bucket,
            view.cache_profile,
            view.source_discovery_method,
        )
        for row, view in zip(rows, validated, strict=True)
    }
    if len(source_identities) != 1:
        raise ValueError("reference chart manifest golden contains mixed source identity")

    derived_versions = {row["corpus_version"] for row in rows}
    if len(derived_versions) != 1:
        raise ValueError("reference chart manifest golden contains mixed corpus version")
    (derived_version,) = derived_versions
    if not _is_corpus_version(derived_version):
        raise ValueError("reference chart manifest golden has an invalid corpus version")
    normalized_rows = tuple(
        {key: value for key, value in row.items() if key != "corpus_version"} for row in rows
    )
    rendered = render_manifest(normalized_rows)
    if rendered.corpus_version != derived_version or rendered.content != content:
        raise ValueError("reference chart manifest golden has an invalid derived corpus version")


def _validate_reference_row(row: Mapping[str, object]) -> ManifestRowView:
    if set(row) != _REFERENCE_ROW_KEYS:
        raise ValueError("reference chart manifest row has an invalid key set")
    if row["schema_version"] != REFERENCE_CHART_MANIFEST_SCHEMA:
        raise ValueError("reference chart manifest row has an unsupported schema")
    _require_sha256_value(row["source_manifest_sha256"], "source_manifest_sha256")
    _require_sha256_value(row["override_document_sha256"], "override_document_sha256")
    if not _is_corpus_version(row["source_corpus_version"]):
        raise ValueError("reference chart manifest row has an invalid source corpus version")
    if not _is_corpus_version(row["corpus_version"]):
        raise ValueError("reference chart manifest row has an invalid corpus version")

    source_row = {key: row[key] for key in _HPA_321_ROW_KEYS}
    source_row["schema_version"] = MANIFEST_SCHEMA
    source_row["corpus_version"] = row["source_corpus_version"]
    try:
        view = manifest_row_view_from_row(source_row)
    except ValueError:
        raise ValueError("reference chart manifest row has an invalid HPA-321 payload") from None

    _validate_cached_objects(view)
    _validate_selection_fields(row, view)
    return view


def _validate_cached_objects(view: ManifestRowView) -> None:
    for remote in view.inventory.objects:
        if remote.cache_status != "verified":
            continue
        if remote.sha256 is None or remote.cache_path is None:
            raise ValueError("verified source object must have a cache identity")
        try:
            _validate_relative_cache_path(remote.cache_path, remote.sha256)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("reference chart manifest row has an invalid cache path") from None


def _validate_selection_fields(row: Mapping[str, object], view: ManifestRowView) -> None:
    status = row["selection_status"]
    method = row["selection_method"]
    reasons = row["selection_reason_codes"]
    warnings = row["selection_warnings"]
    if not isinstance(status, str) or status not in _SELECTION_STATUSES:
        raise ValueError("reference chart manifest row has an invalid selection status")
    if not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings):
        raise ValueError("reference chart manifest row has invalid selection warnings")
    _validate_reason_codes(reasons)
    _validate_selection_override(row["selection_override"])
    _validate_optional_level_metadata(row)
    _validate_set_def_identity(row, view)

    if status == "selected":
        if not isinstance(method, str) or method not in _SELECTION_METHODS or reasons:
            raise ValueError("selected reference chart row is invalid")
        _validate_selected_chart_identity(row, view)
        if not isinstance(row["title"], str) or not isinstance(row["artist"], str):
            raise ValueError("selected reference chart metadata is invalid")
        return

    if method is not None or not reasons:
        raise ValueError("quarantined reference chart row is invalid")
    if any(row[field] is not None for field in _QUARANTINED_NULL_FIELDS):
        raise ValueError("quarantined reference chart row contains selected metadata")


def _validate_reason_codes(value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or reason not in _SELECTION_REASON_CODES for reason in value
    ):
        raise ValueError("reference chart manifest row has invalid reason codes")


def _validate_selection_override(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _OVERRIDE_KEYS:
        raise ValueError("reference chart manifest row has an invalid selection override")
    chart_key = value["chart_key"]
    reason = value["reason"]
    if not isinstance(chart_key, str) or not chart_key:
        raise ValueError("reference chart manifest row has an invalid selection override")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reference chart manifest row has an invalid selection override")


def _validate_optional_level_metadata(row: Mapping[str, object]) -> None:
    slot = row["selected_level_slot"]
    label = row["selected_level_label"]
    dlevel_raw = row["dlevel_raw"]
    dlevel_normalized = row["dlevel_normalized"]
    if slot is not None and slot not in {"L1", "L2", "L3", "L4", "L5"}:
        raise ValueError("reference chart manifest row has an invalid selected level slot")
    if label is not None and not isinstance(label, str):
        raise ValueError("reference chart manifest row has an invalid selected level label")
    if dlevel_raw is not None and not isinstance(dlevel_raw, str):
        raise ValueError("reference chart manifest row has an invalid raw DLEVEL")
    if dlevel_normalized is not None and (
        not isinstance(dlevel_normalized, int)
        or isinstance(dlevel_normalized, bool)
        or not 0 <= dlevel_normalized <= 100
    ):
        raise ValueError("reference chart manifest row has an invalid normalized DLEVEL")


def _validate_set_def_identity(row: Mapping[str, object], view: ManifestRowView) -> None:
    key = row["set_def_key"]
    digest = row["set_def_content_hash"]
    if key is None and digest is None:
        return
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or not is_set_def_key(key)
    ):
        raise ValueError("reference chart manifest row has an invalid set.def identity")
    _require_sha256_value(digest, "set_def_content_hash")
    remote = _remote_by_key(view, key)
    if remote is None or remote.cache_status != "verified" or remote.sha256 != digest:
        raise ValueError("reference chart manifest row has an inconsistent set.def identity")


def _validate_selected_chart_identity(row: Mapping[str, object], view: ManifestRowView) -> None:
    key, digest, cache_path = (row[field] for field in _SELECTED_IDENTITY_FIELDS)
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or not isinstance(cache_path, str)
        or not is_chart_key(key)
    ):
        raise ValueError("selected reference chart identity is invalid")
    _require_sha256_value(digest, "selected_chart_content_hash")
    try:
        _validate_relative_cache_path(cache_path, digest)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("selected reference chart cache path is invalid") from None
    remote = _remote_by_key(view, key)
    if (
        remote is None
        or remote.cache_status != "verified"
        or remote.sha256 != digest
        or remote.cache_path != cache_path
    ):
        raise ValueError("selected reference chart identity is inconsistent")


def _remote_by_key(view: ManifestRowView, key: str) -> RemoteObject | None:
    return next((remote for remote in view.inventory.objects if remote.key == key), None)


def _require_sha256_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be lowercase SHA-256")
    try:
        return require_sha256(value, field)
    except StrictJsonError:
        raise ValueError(f"{field} must be lowercase SHA-256") from None


def _is_corpus_version(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    try:
        _require_sha256_value(value.removeprefix("sha256:"), "corpus_version")
    except ValueError:
        return False
    return True
