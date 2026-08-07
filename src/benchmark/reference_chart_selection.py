from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from src.benchmark.backend_identity import StrictJsonError, strict_json_loads
from src.benchmark.chart_names import CHART_FILENAME_PRIORITY
from src.benchmark.corpus_cache import is_chart_key, is_set_def_key, read_verified_cache_body
from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.dtx_parser import ParsedDtxChart, parse_dtx_bytes
from src.benchmark.inventory_object_keys import resolve_inventory_object_key
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, RemoteObject
from src.benchmark.set_def_parser import ParsedSetDef, parse_set_def_bytes

SelectionStatus = Literal["selected", "quarantined"]

SelectionMethod = Literal[
    "override",
    "set_def_slot",
    "single_candidate_fallback",
    "dlevel_fallback",
    "filename_tiebreak_fallback",
]

SelectionReasonCode = Literal[
    "source_inventory_unusable",
    "cached_body_unavailable",
    "no_verified_chart",
    "ambiguous_set_def",
    "invalid_set_def",
    "referenced_chart_missing",
    "invalid_chart_reference",
    "ambiguous_chart_key",
    "selected_chart_parse_failed",
    "override_invalid",
    "ambiguous_fallback",
]

_OVERRIDE_SCHEMA = "crux.reference-chart-overrides/v1"
_EMPTY_OVERRIDE_DOCUMENT = (
    b'{"overrides":{},"schema_version":"crux.reference-chart-overrides/v1"}\n'
)
_OVERRIDE_DOCUMENT_FIELDS = frozenset({"overrides", "schema_version"})
_OVERRIDE_ENTRY_FIELDS = frozenset({"chart_key", "reason"})
_MAX_SIMFILE_ID_TEXT = str(MAX_SIMFILE_ID)


@dataclass(frozen=True)
class SelectionOverride:
    chart_key: str
    reason: str


@dataclass(frozen=True)
class LoadedOverrides:
    document_sha256: str
    by_simfile_id: Mapping[int, SelectionOverride]


@dataclass(frozen=True)
# The downstream manifest contract requires every one of these fields.
# pylint: disable=too-many-instance-attributes
class ChartSelection:
    status: SelectionStatus
    method: SelectionMethod | None
    reason_codes: tuple[SelectionReasonCode, ...]
    warnings: tuple[str, ...]
    set_def: RemoteObject | None
    selected_chart: RemoteObject | None
    selected_level_slot: str | None
    selected_level_label: str | None
    dlevel_raw: str | None
    dlevel_normalized: int | None
    title: str | None
    artist: str | None
    override: SelectionOverride | None

    def __post_init__(self) -> None:
        if self.status == "selected":
            if (
                self.method is None
                or self.selected_chart is None
                or self.reason_codes
                or not isinstance(self.title, str)
                or not isinstance(self.artist, str)
            ):
                raise ValueError("selected chart selection is invalid")
            return
        if self.status == "quarantined":
            if self.method is not None or self.selected_chart is not None or not self.reason_codes:
                raise ValueError("quarantined chart selection is invalid")
            return
        raise ValueError("chart selection status is invalid")


def load_selection_overrides(
    path: Path | None,
    *,
    missing_ok: bool,
) -> LoadedOverrides:
    content = _read_override_document(path, missing_ok=missing_ok)
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference chart override document is not canonical JSON")
    try:
        document = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError:
        raise ValueError("reference chart override document is not canonical JSON") from None
    if not isinstance(document, dict) or set(document) != _OVERRIDE_DOCUMENT_FIELDS:
        raise ValueError("reference chart override document has an invalid shape")
    if document.get("schema_version") != _OVERRIDE_SCHEMA:
        raise ValueError("reference chart override document has an unsupported schema_version")

    overrides = document.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("reference chart overrides must be an object")

    normalized_entries = [
        (_normalize_simfile_id(raw_id), raw_id, raw_entry)
        for raw_id, raw_entry in overrides.items()
    ]
    normalized_ids = [simfile_id for simfile_id, _, _ in normalized_entries]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("duplicate override simfile ID after numeric normalization")

    parsed: dict[int, SelectionOverride] = {}
    for simfile_id, raw_id, raw_entry in normalized_entries:
        if raw_id != str(simfile_id):
            raise ValueError("override simfile ID must be canonical decimal")
        parsed[simfile_id] = _parse_override_entry(raw_entry)

    return LoadedOverrides(
        document_sha256=sha256(content).hexdigest(),
        by_simfile_id=MappingProxyType(parsed),
    )


def select_reference_chart(
    row: ManifestRowView,
    *,
    cache_dir: Path,
    overrides: LoadedOverrides,
) -> ChartSelection:
    inventory = row.inventory
    warnings = _inventory_warnings(row)
    override = overrides.by_simfile_id.get(inventory.simfile_id)
    if inventory.sync_status == "empty" or not any(
        remote.cache_status == "verified" for remote in inventory.objects
    ):
        return _quarantined(
            "source_inventory_unusable",
            warnings=(),
            override=override,
        )

    if override is not None:
        return _select_override(row, cache_dir, override, warnings)

    set_def, discovery_reason = _discover_set_def(row)
    if discovery_reason is not None:
        return _quarantined(discovery_reason, warnings=warnings)
    if set_def is None:
        return _select_fallback(row, cache_dir, warnings)

    parsed_set_def, parse_reason = _parse_set_def(row, cache_dir, set_def)
    if parse_reason is not None:
        return _quarantined(parse_reason, warnings=warnings)
    assert parsed_set_def is not None
    warnings += parsed_set_def.warnings
    return _select_authored_slots(row, cache_dir, set_def, parsed_set_def, warnings)


def _select_authored_slots(
    row: ManifestRowView,
    cache_dir: Path,
    set_def: RemoteObject,
    parsed_set_def: ParsedSetDef,
    warnings: tuple[str, ...],
) -> ChartSelection:
    inventory = row.inventory
    for slot in reversed(parsed_set_def.slots):
        if slot.file is None:
            continue
        resolution = resolve_inventory_object_key(
            slot.file,
            base_object_key_dir=_object_key_directory(set_def.key),
            object_prefix=inventory.object_prefix,
            objects=inventory.objects,
        )
        if resolution.status == "missing" and _set_def_is_nested(set_def, inventory.object_prefix):
            root_resolution = resolve_inventory_object_key(
                slot.file,
                base_object_key_dir=inventory.object_prefix,
                object_prefix=inventory.object_prefix,
                objects=inventory.objects,
            )
            if root_resolution.status in {"exact", "casefold"}:
                warnings += ("set_def_root_fallback",)
            resolution = root_resolution
        if resolution.status == "missing":
            continue
        if resolution.status == "invalid_path":
            return _quarantined("invalid_chart_reference", warnings=warnings, set_def=set_def)
        if resolution.status == "ambiguous":
            return _quarantined("ambiguous_chart_key", warnings=warnings, set_def=set_def)
        remote = resolution.remote
        assert remote is not None
        if not is_chart_key(remote.key):
            return _quarantined("invalid_chart_reference", warnings=warnings, set_def=set_def)
        chart, chart_reason = _parse_chart(row, cache_dir, remote)
        if chart_reason is not None:
            return _quarantined(chart_reason, warnings=warnings, set_def=set_def)
        assert chart is not None
        return _selected(
            "set_def_slot",
            remote,
            chart,
            warnings=warnings,
            set_def=set_def,
            selected_level_slot=f"L{slot.level}",
            selected_level_label=slot.label,
        )

    return _quarantined("referenced_chart_missing", warnings=warnings, set_def=set_def)


def _read_override_document(path: Path | None, *, missing_ok: bool) -> bytes:
    if path is None:
        if missing_ok:
            return _EMPTY_OVERRIDE_DOCUMENT
        raise ValueError("reference chart override document is required")
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return _EMPTY_OVERRIDE_DOCUMENT
        raise ValueError("reference chart override document is required") from None
    except OSError:
        raise ValueError("reference chart override document is unavailable") from None


def _normalize_simfile_id(raw_id: object) -> int:
    if not isinstance(raw_id, str) or not raw_id.isascii() or not raw_id.isdecimal():
        raise ValueError("override simfile ID must be a decimal integer in supported range")
    normalized = raw_id.lstrip("0") or "0"
    if len(normalized) > len(_MAX_SIMFILE_ID_TEXT) or (
        len(normalized) == len(_MAX_SIMFILE_ID_TEXT) and normalized > _MAX_SIMFILE_ID_TEXT
    ):
        raise ValueError("override simfile ID must be a decimal integer in supported range")
    return int(normalized)


def _parse_override_entry(value: object) -> SelectionOverride:
    if not isinstance(value, dict) or set(value) != _OVERRIDE_ENTRY_FIELDS:
        raise ValueError("reference chart override entry has an invalid shape")
    chart_key = value["chart_key"]
    reason = value["reason"]
    if not isinstance(chart_key, str) or not chart_key:
        raise ValueError("reference chart override chart_key must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reference chart override reason must be a non-empty string")
    return SelectionOverride(chart_key=chart_key, reason=reason)


def _inventory_warnings(row: ManifestRowView) -> tuple[str, ...]:
    inventory = row.inventory
    if inventory.sync_status != "complete" or inventory.sync_errors:
        return ("partial_inventory",)
    return ()


def _select_override(
    row: ManifestRowView,
    cache_dir: Path,
    override: SelectionOverride,
    warnings: tuple[str, ...],
) -> ChartSelection:
    exact_matches = tuple(
        remote for remote in row.inventory.objects if remote.key == override.chart_key
    )
    if (
        len(exact_matches) != 1
        or exact_matches[0].cache_status != "verified"
        or not is_chart_key(exact_matches[0].key)
    ):
        return _quarantined("override_invalid", warnings=warnings, override=override)
    chart, chart_reason = _parse_chart(row, cache_dir, exact_matches[0])
    if chart_reason is not None:
        return _quarantined("override_invalid", warnings=warnings, override=override)
    assert chart is not None
    return _selected(
        "override",
        exact_matches[0],
        chart,
        warnings=warnings,
        set_def=None,
        selected_level_slot=None,
        selected_level_label=None,
        override=override,
    )


# Each terminal outcome is deliberately visible in the ordered fallback policy.
# pylint: disable-next=too-many-return-statements
def _select_fallback(
    row: ManifestRowView,
    cache_dir: Path,
    warnings: tuple[str, ...],
) -> ChartSelection:
    parsed_candidates: list[tuple[RemoteObject, ParsedDtxChart]] = []
    for remote in row.inventory.objects:
        if remote.cache_status != "verified" or not is_chart_key(remote.key):
            continue
        chart, chart_reason = _parse_chart(row, cache_dir, remote)
        if chart_reason in {"selected_chart_parse_failed", "cached_body_unavailable"}:
            continue
        if chart_reason is not None:
            return _quarantined(chart_reason, warnings=warnings)
        assert chart is not None
        has_note_evidence = any(event.lane_id != "01" for event in chart.events)
        if has_note_evidence:
            parsed_candidates.append((remote, chart))

    if not parsed_candidates:
        return _quarantined("no_verified_chart", warnings=warnings)
    if len(parsed_candidates) == 1:
        remote, chart = parsed_candidates[0]
        return _selected(
            "single_candidate_fallback",
            remote,
            chart,
            warnings=warnings,
            set_def=None,
            selected_level_slot=None,
            selected_level_label=None,
        )

    numeric_candidates = tuple(
        (remote, chart)
        for remote, chart in parsed_candidates
        if chart.dlevel_normalized is not None
    )
    if not numeric_candidates:
        return _quarantined("ambiguous_fallback", warnings=warnings)
    highest_dlevel = max(chart.dlevel_normalized for _, chart in numeric_candidates)
    assert highest_dlevel is not None
    highest_candidates = tuple(
        (remote, chart)
        for remote, chart in numeric_candidates
        if chart.dlevel_normalized == highest_dlevel
    )
    if len(highest_candidates) == 1:
        remote, chart = highest_candidates[0]
        return _selected(
            "dlevel_fallback",
            remote,
            chart,
            warnings=warnings,
            set_def=None,
            selected_level_slot=None,
            selected_level_label=None,
        )

    ranked_candidates = tuple(
        (CHART_FILENAME_PRIORITY.index(_chart_basename(remote.key)), remote, chart)
        for remote, chart in highest_candidates
        if _chart_basename(remote.key) in CHART_FILENAME_PRIORITY
    )
    if not ranked_candidates:
        return _quarantined("ambiguous_fallback", warnings=warnings)
    winning_rank = min(rank for rank, _, _ in ranked_candidates)
    ranked_winners = tuple(
        (remote, chart) for rank, remote, chart in ranked_candidates if rank == winning_rank
    )
    if len(ranked_winners) != 1:
        return _quarantined("ambiguous_fallback", warnings=warnings)
    remote, chart = ranked_winners[0]
    return _selected(
        "filename_tiebreak_fallback",
        remote,
        chart,
        warnings=warnings,
        set_def=None,
        selected_level_slot=None,
        selected_level_label=None,
    )


def _discover_set_def(
    row: ManifestRowView,
) -> tuple[RemoteObject | None, SelectionReasonCode | None]:
    inventory = row.inventory
    candidates = tuple(remote for remote in inventory.objects if is_set_def_key(remote.key))
    root_candidates = tuple(
        remote
        for remote in candidates
        if remote.key == inventory.object_prefix + remote.key.rsplit("/", 1)[-1]
    )
    if root_candidates:
        if len(root_candidates) == 1:
            return root_candidates[0], None
        lowercase_candidates = tuple(
            remote for remote in root_candidates if remote.key.rsplit("/", 1)[-1] == "set.def"
        )
        if len(lowercase_candidates) == 1:
            return lowercase_candidates[0], None
        return None, "ambiguous_set_def"

    if not candidates:
        return None, None
    shallowest_depth = min(
        _nested_set_def_depth(inventory.object_prefix, remote) for remote in candidates
    )
    shallowest = tuple(
        remote
        for remote in candidates
        if _nested_set_def_depth(inventory.object_prefix, remote) == shallowest_depth
    )
    if len(shallowest) != 1:
        return None, "ambiguous_set_def"
    return shallowest[0], None


def _nested_set_def_depth(object_prefix: str, remote: RemoteObject) -> int:
    return remote.key.removeprefix(object_prefix).count("/")


def _parse_set_def(
    row: ManifestRowView,
    cache_dir: Path,
    remote: RemoteObject,
) -> tuple[ParsedSetDef | None, SelectionReasonCode | None]:
    try:
        body = read_verified_cache_body(
            cache_dir,
            remote,
            source_endpoint_sha256=row.source_endpoint_sha256,
            bucket=row.source_bucket,
        )
    except ValueError:
        return None, "cached_body_unavailable"
    try:
        return parse_set_def_bytes(body, source_name=remote.key), None
    except ValueError:
        return None, "invalid_set_def"


def _parse_chart(
    row: ManifestRowView,
    cache_dir: Path,
    remote: RemoteObject,
) -> tuple[ParsedDtxChart | None, SelectionReasonCode | None]:
    try:
        body = read_verified_cache_body(
            cache_dir,
            remote,
            source_endpoint_sha256=row.source_endpoint_sha256,
            bucket=row.source_bucket,
        )
    except ValueError:
        return None, "cached_body_unavailable"
    try:
        return parse_dtx_bytes(body, chart_id=remote.key, source_name=remote.key), None
    except ValueError:
        return None, "selected_chart_parse_failed"


def _object_key_directory(key: str) -> str:
    return key.rsplit("/", 1)[0]


def _set_def_is_nested(set_def: RemoteObject, object_prefix: str) -> bool:
    return _object_key_directory(set_def.key) != object_prefix.rstrip("/")


def _chart_basename(key: str) -> str:
    return key.rsplit("/", 1)[-1].rsplit(".", 1)[0].casefold()


# This maps the selector's explicit policy result to the fixed manifest-facing contract.
# pylint: disable-next=too-many-arguments
def _selected(
    method: SelectionMethod,
    remote: RemoteObject,
    chart: ParsedDtxChart,
    *,
    warnings: tuple[str, ...],
    set_def: RemoteObject | None,
    selected_level_slot: str | None,
    selected_level_label: str | None,
    override: SelectionOverride | None = None,
) -> ChartSelection:
    return ChartSelection(
        status="selected",
        method=method,
        reason_codes=(),
        warnings=warnings,
        set_def=set_def,
        selected_chart=remote,
        selected_level_slot=selected_level_slot,
        selected_level_label=selected_level_label,
        dlevel_raw=chart.dlevel_raw,
        dlevel_normalized=chart.dlevel_normalized,
        title=chart.title,
        artist=chart.artist,
        override=override,
    )


def _quarantined(
    reason: SelectionReasonCode,
    *,
    warnings: tuple[str, ...],
    set_def: RemoteObject | None = None,
    override: SelectionOverride | None = None,
) -> ChartSelection:
    return ChartSelection(
        status="quarantined",
        method=None,
        reason_codes=(reason,),
        warnings=warnings,
        set_def=set_def,
        selected_chart=None,
        selected_level_slot=None,
        selected_level_label=None,
        dlevel_raw=None,
        dlevel_normalized=None,
        title=None,
        artist=None,
        override=override,
    )
