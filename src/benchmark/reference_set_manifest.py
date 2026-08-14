"""Publish model-independent eligibility for native HPA-323 references.

HPA-324 deliberately keeps the native HPA-323 event artifact as the only
persisted event representation.  This module reads that artifact, applies the
frozen taxonomy/map in :mod:`src.benchmark.reference_set`, and publishes one
small derived manifest row containing the decision and its accounting.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.r2_corpus_models import PublishedManifest
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_timing import NativeReferenceEvent, read_reference_events
from src.benchmark.reference_timing_manifest import (
    REFERENCE_TIMING_MANIFEST_SCHEMA,
    LoadedReferenceTimingRow,
    _validate_timing_manifest_row,
    load_reference_timing_manifest,
    read_canonical_manifest_core,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

BENCHMARK_REFERENCE_MANIFEST_SCHEMA = "crux.benchmark-reference-manifest/v1"

ReferenceEligibilityStatus = Literal["eligible", "quarantined"]
EligibilityReasonCode = Literal[
    "upstream_reference_unavailable",
    "reference_event_artifact_invalid",
    "unclassified_reference_lane",
    "no_scored_drum_events",
]

_ELIGIBILITY_REASON_CODES: frozenset[str] = frozenset(get_args(EligibilityReasonCode))
_ELIGIBILITY_KEYS: frozenset[str] = frozenset(
    {
        "source_reference_timing_manifest_sha256",
        "source_reference_timing_version",
        "taxonomy_version",
        "lane_map_version",
        "reference_eligibility_status",
        "reference_eligibility_reason_codes",
        "reference_eligibility_warnings",
        "mapped_event_count",
        "common_scored_event_count",
        "ignored_event_count",
        "unmapped_event_count",
        "duplicate_common_event_count",
    }
)
_EVENT_ARTIFACT_PATH_RE = re.compile(r"events/([0-9a-f]{64})\.jsonl\Z")
_IGNORED_WARNING_RE = re.compile(r"ignored_reference_lane:([0-9A-Z]{2}):count=([1-9][0-9]*)\Z")
_DUPLICATE_WARNING_RE = re.compile(r"duplicate_common_projection:count=([1-9][0-9]*)\Z")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReferenceSetRequest:
    manifest_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ReferenceSetOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    eligible_count: int
    quarantined_count: int


@dataclass(frozen=True)
class ReferenceSetRowView:
    simfile_id: int
    eligibility_status: ReferenceEligibilityStatus
    eligibility_reason_codes: tuple[EligibilityReasonCode, ...]
    eligibility_warnings: tuple[str, ...]
    mapped_event_count: int
    common_scored_event_count: int
    ignored_event_count: int
    unmapped_event_count: int
    duplicate_common_event_count: int


@dataclass(frozen=True)
class LoadedReferenceSetRow:
    source_row: Mapping[str, object]
    view: ReferenceSetRowView


@dataclass(frozen=True)
class LoadedReferenceSetManifest:
    manifest_sha256: str
    corpus_version: str
    source_reference_timing_manifest_sha256: str
    source_reference_timing_version: str
    rows: tuple[LoadedReferenceSetRow, ...]


@dataclass(frozen=True)
class _ReferenceSetResolution:
    status: ReferenceEligibilityStatus
    reason_codes: tuple[EligibilityReasonCode, ...]
    warnings: tuple[str, ...]
    mapped_event_count: int
    common_scored_event_count: int
    ignored_event_count: int
    unmapped_event_count: int
    duplicate_common_event_count: int


def _zero_resolution(
    reason_code: EligibilityReasonCode,
) -> _ReferenceSetResolution:
    return _ReferenceSetResolution(
        status="quarantined",
        reason_codes=(reason_code,),
        warnings=(),
        mapped_event_count=0,
        common_scored_event_count=0,
        ignored_event_count=0,
        unmapped_event_count=0,
        duplicate_common_event_count=0,
    )


def read_native_reference_events(
    loaded: LoadedReferenceTimingRow,
    *,
    timing_output_root: Path,
) -> tuple[NativeReferenceEvent, ...]:
    relative = loaded.view.reference_events_cache_path
    if not isinstance(relative, str):
        raise ValueError("reference event artifact path is unavailable")
    match = _EVENT_ARTIFACT_PATH_RE.fullmatch(relative)
    if match is None:
        raise ValueError("reference event artifact path is unsafe")

    # The HPA-323 manifest stores paths relative to its output directory (the
    # parent of ``manifests/``), not relative to the manifest file itself.
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("reference event artifact path is unsafe")
    artifact_path = timing_output_root.joinpath(*relative_path.parts)
    try:
        artifact_path.relative_to(timing_output_root)
    except ValueError:
        raise ValueError("reference event artifact path is unsafe") from None

    content = read_regular_file_no_follow(artifact_path)
    expected_sha256 = match.group(1)
    if sha256(content).hexdigest() != expected_sha256:
        raise ValueError("reference event artifact hash mismatch")
    events = read_reference_events(content)
    expected_identity = (
        loaded.view.simfile_id,
        loaded.source_row["selected_chart_key"],
        loaded.source_row["selected_chart_content_hash"],
        loaded.view.source_audio_key,
        loaded.view.source_audio_content_hash,
    )
    if any(
        (
            event.simfile_id,
            event.selected_chart_key,
            event.selected_chart_content_hash,
            event.source_audio_key,
            event.source_audio_content_hash,
        )
        != expected_identity
        for event in events
    ):
        raise ValueError("reference event artifact identity does not match timing row")
    return events


def _evaluate_row(
    loaded: LoadedReferenceTimingRow,
    *,
    timing_output_root: Path,
) -> _ReferenceSetResolution:
    if loaded.view.timing_status != "ready":
        return _zero_resolution("upstream_reference_unavailable")

    try:
        events = read_native_reference_events(loaded, timing_output_root=timing_output_root)
    except (OSError, RuntimeError, ValueError):
        return _zero_resolution("reference_event_artifact_invalid")

    result = map_reference_events(events)
    ignored_event_count = sum(result.diagnostics.ignored.values())
    unmapped_event_count = sum(result.diagnostics.unmapped.values())
    mapped_event_count = len(result.mapped_events)
    common_scored_event_count = len(result.common_events)
    duplicate_count = result.diagnostics.duplicate_common_event_count

    if result.diagnostics.unmapped:
        return _ReferenceSetResolution(
            status="quarantined",
            reason_codes=("unclassified_reference_lane",),
            warnings=(),
            mapped_event_count=mapped_event_count,
            common_scored_event_count=common_scored_event_count,
            ignored_event_count=ignored_event_count,
            unmapped_event_count=unmapped_event_count,
            duplicate_common_event_count=duplicate_count,
        )
    if not result.mapped_events:
        return _ReferenceSetResolution(
            status="quarantined",
            reason_codes=("no_scored_drum_events",),
            warnings=(),
            mapped_event_count=0,
            common_scored_event_count=0,
            ignored_event_count=ignored_event_count,
            unmapped_event_count=0,
            duplicate_common_event_count=0,
        )

    warnings = [
        f"ignored_reference_lane:{lane}:count={count}"
        for lane, count in sorted(result.diagnostics.ignored.items())
    ]
    if duplicate_count:
        warnings.append(f"duplicate_common_projection:count={duplicate_count}")
    return _ReferenceSetResolution(
        status="eligible",
        reason_codes=(),
        warnings=tuple(sorted(warnings)),
        mapped_event_count=mapped_event_count,
        common_scored_event_count=common_scored_event_count,
        ignored_event_count=ignored_event_count,
        unmapped_event_count=0,
        duplicate_common_event_count=duplicate_count,
    )


def build_reference_set_row(
    source_row: Mapping[str, object],
    *,
    source_reference_timing_manifest_sha256: str,
    source_reference_timing_version: str,
    resolution: _ReferenceSetResolution,
) -> dict[str, object]:
    """Derive one HPA-324 row from a validated HPA-323 timing row."""
    row = dict(source_row)
    row.pop("corpus_version", None)
    row["schema_version"] = BENCHMARK_REFERENCE_MANIFEST_SCHEMA
    row.update(
        {
            "source_reference_timing_manifest_sha256": source_reference_timing_manifest_sha256,
            "source_reference_timing_version": source_reference_timing_version,
            "taxonomy_version": TAXONOMY_VERSION,
            "lane_map_version": DTX_LANE_MAP_VERSION,
            "reference_eligibility_status": resolution.status,
            "reference_eligibility_reason_codes": list(resolution.reason_codes),
            "reference_eligibility_warnings": list(resolution.warnings),
            "mapped_event_count": resolution.mapped_event_count,
            "common_scored_event_count": resolution.common_scored_event_count,
            "ignored_event_count": resolution.ignored_event_count,
            "unmapped_event_count": resolution.unmapped_event_count,
            "duplicate_common_event_count": resolution.duplicate_common_event_count,
        }
    )
    return row


def build_reference_set_outcome(
    *,
    manifest: PublishedManifest,
    total_input_rows: int,
    eligible_count: int,
    quarantined_count: int,
) -> ReferenceSetOutcome:
    if eligible_count + quarantined_count != total_input_rows:
        raise ValueError("reference set outcome must balance input rows")
    if eligible_count < 0 or quarantined_count < 0:
        raise ValueError("reference set outcome counts must be nonnegative")
    if quarantined_count:
        return ReferenceSetOutcome(
            status="partial",
            exit_code=1,
            manifest=manifest,
            eligible_count=eligible_count,
            quarantined_count=quarantined_count,
        )
    return ReferenceSetOutcome(
        status="complete",
        exit_code=0,
        manifest=manifest,
        eligible_count=eligible_count,
        quarantined_count=0,
    )


def failed_reference_set_outcome(
    eligible_count: int = 0,
    quarantined_count: int = 0,
) -> ReferenceSetOutcome:
    return ReferenceSetOutcome(
        status="failed",
        exit_code=2,
        manifest=None,
        eligible_count=eligible_count,
        quarantined_count=quarantined_count,
    )


def run_reference_set(
    request: ReferenceSetRequest,
    *,
    clock: Clock = _utc_now,
) -> ReferenceSetOutcome:
    """Read HPA-323 timing rows and publish the HPA-324 eligibility manifest."""
    try:
        loaded = load_reference_timing_manifest(request.manifest_path)
    except (OSError, ValueError):
        return failed_reference_set_outcome()

    timing_output_root = request.manifest_path.parent.parent
    output_rows: list[dict[str, object]] = []
    eligible_count = 0
    quarantined_count = 0
    try:
        for loaded_row in loaded.rows:
            resolution = _evaluate_row(
                loaded_row,
                timing_output_root=timing_output_root,
            )
            if resolution.status == "eligible":
                eligible_count += 1
            else:
                quarantined_count += 1
            output_rows.append(
                build_reference_set_row(
                    loaded_row.source_row,
                    source_reference_timing_manifest_sha256=loaded.manifest_sha256,
                    source_reference_timing_version=loaded.corpus_version,
                    resolution=resolution,
                )
            )

        rendered = render_manifest(tuple(output_rows))
        for row in rendered.rows:
            _validate_reference_set_row(row)
        published = publish_manifest(request.output_dir, rendered)
        overall_status: Literal["complete", "partial"] = (
            "complete" if quarantined_count == 0 else "partial"
        )
        publish_latest_manifest(request.output_dir, published, overall_status, clock())
    except (ManifestPublicationError, OSError, RuntimeError, ValueError):
        return failed_reference_set_outcome(eligible_count, quarantined_count)

    return build_reference_set_outcome(
        manifest=published,
        total_input_rows=len(loaded.rows),
        eligible_count=eligible_count,
        quarantined_count=quarantined_count,
    )


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate the two-row HPA-324 schema golden."""
    if schema != BENCHMARK_REFERENCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference set manifest golden must be canonical JSONL")
    lines = content.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reference set manifest golden must contain exactly two records")
    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reference set manifest golden must be canonical JSONL") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("reference set manifest golden rows must be objects")
    typed_rows = tuple(row for row in rows if isinstance(row, dict))
    for row in typed_rows:
        _validate_reference_set_row(row)
    statuses = sorted(row["reference_eligibility_status"] for row in typed_rows)
    if statuses != ["eligible", "quarantined"]:
        raise ValueError(
            "reference set manifest golden requires one eligible and one quarantined row"
        )
    source_identities = {
        (
            row["source_reference_timing_manifest_sha256"],
            row["source_reference_timing_version"],
            row["taxonomy_version"],
            row["lane_map_version"],
        )
        for row in typed_rows
    }
    if len(source_identities) != 1:
        raise ValueError("reference set manifest golden contains mixed source identity")
    versions = {row["corpus_version"] for row in typed_rows}
    if len(versions) != 1:
        raise ValueError("reference set manifest golden contains mixed corpus version")
    (version,) = versions
    if not _is_corpus_version(version):
        raise ValueError("reference set manifest golden has an invalid corpus version")
    normalized_rows = tuple(
        {key: value for key, value in row.items() if key != "corpus_version"} for row in typed_rows
    )
    rendered = render_manifest(normalized_rows)
    if rendered.content != content or rendered.corpus_version != version:
        raise ValueError("reference set manifest golden has an invalid derived corpus version")


def _validate_reference_set_row(row: Mapping[str, object]) -> None:
    required = _ELIGIBILITY_KEYS | {"schema_version", "corpus_version"}
    if not required <= set(row):
        raise ValueError("reference set manifest row has an invalid key set")
    if row["schema_version"] != BENCHMARK_REFERENCE_MANIFEST_SCHEMA:
        raise ValueError("reference set manifest row has an unsupported schema")

    source_manifest_sha256 = row["source_reference_timing_manifest_sha256"]
    if not isinstance(source_manifest_sha256, str):
        raise ValueError("reference set manifest row has an invalid source timing hash")
    try:
        require_sha256(source_manifest_sha256, "source_reference_timing_manifest_sha256")
    except StrictJsonError:
        raise ValueError("reference set manifest row has an invalid source timing hash") from None
    source_version = row["source_reference_timing_version"]
    if not _is_corpus_version(source_version):
        raise ValueError("reference set manifest row has an invalid source timing version")
    if (
        row["taxonomy_version"] != TAXONOMY_VERSION
        or row["lane_map_version"] != DTX_LANE_MAP_VERSION
    ):
        raise ValueError("reference set manifest row has an unsupported taxonomy version")
    if not _is_corpus_version(row["corpus_version"]):
        raise ValueError("reference set manifest row has an invalid corpus version")

    status = row["reference_eligibility_status"]
    if status not in {"eligible", "quarantined"}:
        raise ValueError("reference set manifest row has an invalid eligibility status")
    reasons = row["reference_eligibility_reason_codes"]
    if (
        not isinstance(reasons, list)
        or any(
            not isinstance(reason, str) or reason not in _ELIGIBILITY_REASON_CODES
            for reason in reasons
        )
        or reasons != sorted(set(reasons))
    ):
        raise ValueError("reference set manifest row has invalid eligibility reason codes")
    warnings = row["reference_eligibility_warnings"]
    _validate_warnings(warnings)
    warning_counts = _warning_counts(warnings)

    count_fields = (
        "mapped_event_count",
        "common_scored_event_count",
        "ignored_event_count",
        "unmapped_event_count",
        "duplicate_common_event_count",
    )
    counts = {field: row[field] for field in count_fields}
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("reference set manifest row has invalid event counts")
    if counts["common_scored_event_count"] > counts["mapped_event_count"]:
        raise ValueError("reference set manifest row has invalid common event count")
    if counts["duplicate_common_event_count"] > counts["mapped_event_count"]:
        raise ValueError("reference set manifest row has invalid duplicate event count")

    # Reconstruct the HPA-323 row and delegate its complete key-set and domain
    # validation to the source-owner validator.  Unknown HPA-324 keys remain in
    # this payload and are rejected there rather than silently dropped.
    timing_row = {
        key: value
        for key, value in row.items()
        if key not in _ELIGIBILITY_KEYS and key != "corpus_version"
    }
    timing_row["schema_version"] = REFERENCE_TIMING_MANIFEST_SCHEMA
    timing_row["corpus_version"] = source_version
    try:
        _validate_timing_manifest_row(timing_row)
    except (KeyError, TypeError, ValueError):
        raise ValueError("reference set manifest row has an invalid timing payload") from None

    timing_status = row["timing_status"]
    if status == "eligible":
        if timing_status != "ready" or reasons or counts["mapped_event_count"] <= 0:
            raise ValueError("eligible reference set row has an invalid status shape")
        if counts["unmapped_event_count"]:
            raise ValueError("eligible reference set row contains unmapped events")
        if counts["common_scored_event_count"] <= 0:
            raise ValueError("eligible reference set row has no common events")
        if warning_counts["ignored"] != counts["ignored_event_count"]:
            raise ValueError("eligible reference set row has inconsistent ignored warnings")
        if warning_counts["duplicate"] != counts["duplicate_common_event_count"]:
            raise ValueError("eligible reference set row has inconsistent duplicate warnings")
    elif len(reasons) != 1:
        raise ValueError("quarantined reference set row must carry exactly one reason code")

    reason = reasons[0] if reasons else None
    if reason == "upstream_reference_unavailable":
        if timing_status == "ready" or any(counts.values()):
            raise ValueError("upstream-quarantined reference set row is inconsistent")
    elif reason == "reference_event_artifact_invalid":
        if timing_status != "ready" or any(counts.values()):
            raise ValueError("invalid-artifact reference set row is inconsistent")
    elif reason == "unclassified_reference_lane":
        if timing_status != "ready" or counts["unmapped_event_count"] <= 0:
            raise ValueError("unclassified reference set row is inconsistent")
    elif reason == "no_scored_drum_events":
        if timing_status != "ready" or counts["mapped_event_count"] != 0:
            raise ValueError("empty reference set row is inconsistent")
    if status == "quarantined" and warnings:
        raise ValueError("quarantined reference set row must not carry warnings")


def _reference_set_row_view_from_row(row: Mapping[str, object]) -> ReferenceSetRowView:
    """Build the narrow HPA-326 view after HPA-324 row validation."""
    simfile_id = row["simfile_id"]
    status = row["reference_eligibility_status"]
    reasons = row["reference_eligibility_reason_codes"]
    warnings = row["reference_eligibility_warnings"]
    if (
        isinstance(simfile_id, bool)
        or not isinstance(simfile_id, int)
        or not isinstance(status, str)
        or not isinstance(reasons, list)
        or not isinstance(warnings, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or any(not isinstance(warning, str) for warning in warnings)
    ):
        raise ValueError("reference set manifest contains an invalid eligibility row")
    counts = tuple(
        row[field]
        for field in (
            "mapped_event_count",
            "common_scored_event_count",
            "ignored_event_count",
            "unmapped_event_count",
            "duplicate_common_event_count",
        )
    )
    if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
        raise ValueError("reference set manifest contains invalid event counts")
    return ReferenceSetRowView(
        simfile_id=simfile_id,
        eligibility_status=status,  # type: ignore[arg-type]
        eligibility_reason_codes=tuple(reasons),  # type: ignore[arg-type]
        eligibility_warnings=tuple(warnings),
        mapped_event_count=counts[0],
        common_scored_event_count=counts[1],
        ignored_event_count=counts[2],
        unmapped_event_count=counts[3],
        duplicate_common_event_count=counts[4],
    )


def load_reference_set_manifest(path: Path) -> LoadedReferenceSetManifest:
    """Load an immutable HPA-324 manifest through the shared JSONL core."""
    rows: list[LoadedReferenceSetRow] = []
    simfile_ids: set[int] = set()
    source_manifest_sha256: str | None = None
    source_version: str | None = None

    def validate_rows(source_rows: tuple[Mapping[str, object], ...]) -> None:
        nonlocal source_manifest_sha256, source_version
        for source_row in source_rows:
            _validate_reference_set_row(source_row)
            view = _reference_set_row_view_from_row(source_row)
            row_manifest_sha256 = source_row["source_reference_timing_manifest_sha256"]
            row_version = source_row["source_reference_timing_version"]
            if not isinstance(row_manifest_sha256, str) or not isinstance(row_version, str):
                raise ValueError(
                    "reference set manifest contains an invalid source timing identity"
                )
            if source_manifest_sha256 is None and source_version is None:
                source_manifest_sha256 = row_manifest_sha256
                source_version = row_version
            elif row_manifest_sha256 != source_manifest_sha256 or row_version != source_version:
                raise ValueError("reference set manifest contains mixed source timing identity")
            if view.simfile_id in simfile_ids:
                raise ValueError("reference set manifest contains duplicate simfile IDs")
            simfile_ids.add(view.simfile_id)
            rows.append(LoadedReferenceSetRow(source_row=source_row, view=view))

    canonical = read_canonical_manifest_core(
        path,
        schema_version=BENCHMARK_REFERENCE_MANIFEST_SCHEMA,
        validate_rows=validate_rows,
    )
    if source_manifest_sha256 is None or source_version is None:
        raise ValueError("reference set manifest contains no records")
    return LoadedReferenceSetManifest(
        manifest_sha256=canonical.manifest_sha256,
        corpus_version=canonical.corpus_version,
        source_reference_timing_manifest_sha256=source_manifest_sha256,
        source_reference_timing_version=source_version,
        rows=tuple(rows),
    )


def _validate_warnings(value: object) -> None:
    if not isinstance(value, list) or any(not isinstance(warning, str) for warning in value):
        raise ValueError("reference set manifest row has invalid eligibility warnings")
    if value != sorted(set(value)):
        raise ValueError("reference set manifest row has nondeterministic eligibility warnings")
    for warning in value:
        if _IGNORED_WARNING_RE.fullmatch(warning) or _DUPLICATE_WARNING_RE.fullmatch(warning):
            continue
        raise ValueError("reference set manifest row has an unsupported eligibility warning")


def _warning_counts(value: object) -> dict[str, int]:
    assert isinstance(value, list)
    ignored = 0
    duplicate = 0
    for warning in value:
        ignored_match = _IGNORED_WARNING_RE.fullmatch(warning)
        if ignored_match is not None:
            ignored += int(ignored_match.group(2))
            continue
        duplicate_match = _DUPLICATE_WARNING_RE.fullmatch(warning)
        if duplicate_match is not None:
            duplicate += int(duplicate_match.group(1))
    return {"ignored": ignored, "duplicate": duplicate}


def _is_corpus_version(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    try:
        require_sha256(value.removeprefix("sha256:"), "corpus_version")
    except StrictJsonError:
        return False
    return True
