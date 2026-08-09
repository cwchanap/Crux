"""Pure timing-manifest contract (HPA-323 Task 6a).

This module is intentionally offline: it defines the *contract* for the
reference-timing manifest — loading a validated HPA-322 reference-chart
manifest, remapping each row into a timing-lineage row, rendering that row
through the shared canonical-JSONL publisher, accounting for the run outcome,
and validating the ``crux.reference-timing-manifest/v1`` schema golden.

No R2/boto3/cache orchestration lives here.  Cache fills, source-audio
downloads, and the actual publication of the derived events artifact are the
responsibility of Task 6b; this task only models the events-reference field and
leaves its population to that follow-up.

Lineage (Brief Step 2):  every HPA-322 field is carried through verbatim
except the top-level ``corpus_version`` (re-derived by ``render_manifest``).
``source_manifest_sha256`` / ``source_corpus_version`` — which already point at
the upstream HPA-321 corpus manifest inside the HPA-322 row — are preserved
unchanged.  Two new fields record the HPA-322 manifest identity itself:

* ``source_reference_chart_manifest_sha256`` — exact input-byte SHA-256;
* ``source_reference_chart_version``         — the HPA-322 input ``corpus_version``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.r2_corpus_models import PublishedManifest
from src.benchmark.reference_chart_manifest import (
    REFERENCE_CHART_MANIFEST_SCHEMA,
    ReferenceChartRowView,
    reference_chart_row_view_from_row,
)
from src.benchmark.reference_timing import TimingReasonCode

#: Canonical schema id for the derived reference-timing manifest rows.
REFERENCE_TIMING_MANIFEST_SCHEMA = "crux.reference-timing-manifest/v1"

#: The timing-semantics family the rows were produced under.
TIMING_SEMANTICS_VERSION = "crux.dtx-audio-timing/v1"

#: Every stable timing reason code (mirrors :data:`TimingReasonCode` via ``get_args``).
_TIMING_REASON_CODES: frozenset[str] = frozenset(get_args(TimingReasonCode))

#: Timing-specific keys added to every derived row.
_TIMING_SPECIFIC_KEYS: frozenset[str] = frozenset(
    {
        "timing_semantics_version",
        "timing_status",
        "timing_reason_codes",
        "timing_warnings",
        "source_audio_key",
        "source_audio_content_hash",
        "reference_events_cache_path",
    }
)

#: Lineage keys added to every derived row (the HPA-322 manifest identity).
_TIMING_LINEAGE_KEYS: frozenset[str] = frozenset(
    {
        "source_reference_chart_manifest_sha256",
        "source_reference_chart_version",
    }
)

_TIMING_STATUSES: frozenset[str] = frozenset({"ready", "quarantined"})


@dataclass(frozen=True)
class ReferenceTimingRequest:
    manifest_path: Path
    cache_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ReferenceTimingOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    ready_count: int
    quarantined_count: int
    upstream_quarantined_count: int
    events_published: int


@dataclass(frozen=True)
class TimingRowResolution:
    """The per-row timing decision consumed by :func:`build_timing_row`.

    ``status`` is the timing-layer verdict (independent of the upstream HPA-322
    ``selection_status``): an HPA-322 *selected* row may still be timing
    *quarantined* when timing analysis fails.  ``reason_codes`` only ever holds
    values from :data:`TimingReasonCode`.  The source-audio / events identity is
    populated for ``ready`` rows and ``None`` for ``quarantined`` rows.
    """

    status: Literal["ready", "quarantined"]
    reason_codes: tuple[TimingReasonCode, ...]
    warnings: tuple[str, ...]
    source_audio_key: str | None
    source_audio_content_hash: str | None
    reference_events_cache_path: str | None


def upstream_chart_unavailable_resolution() -> TimingRowResolution:
    """The canonical timing resolution for an upstream-quarantined HPA-322 row.

    When the HPA-322 row was already quarantined at chart selection, timing
    analysis cannot run: there is no selected chart to time.  The row carries
    the single reason code ``upstream_chart_selection_unavailable`` and null
    source-audio / events identity.
    """
    return TimingRowResolution(
        status="quarantined",
        reason_codes=("upstream_chart_selection_unavailable",),
        warnings=(),
        source_audio_key=None,
        source_audio_content_hash=None,
        reference_events_cache_path=None,
    )


@dataclass(frozen=True)
class _ValidatedReferenceChartRow:
    source_row: Mapping[str, object]
    view: ReferenceChartRowView


@dataclass(frozen=True)
class _LoadedReferenceChartManifest:
    source_reference_chart_manifest_sha256: str
    source_reference_chart_version: str
    rows: tuple[_ValidatedReferenceChartRow, ...]


def load_reference_chart_manifest(path: Path) -> _LoadedReferenceChartManifest:
    """Load and validate a canonical HPA-322 reference-chart manifest.

    Mirrors the canonical-JSONL loader used for HPA-321 source manifests but
    validates each row through :func:`reference_chart_row_view_from_row` (the
    merged HPA-322 reference-chart validator) — never the HPA-321-only
    :func:`manifest_row_view_from_row`.  Records the exact input-byte SHA-256
    and the shared HPA-322 ``corpus_version`` for downstream lineage.
    """
    try:
        content = path.read_bytes()
    except OSError:
        raise ValueError("reference chart manifest is unavailable") from None

    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference chart manifest must contain canonical JSONL records")

    rows: list[_ValidatedReferenceChartRow] = []
    simfile_ids: set[int] = set()
    source_identity: tuple[str, str, str, str, str] | None = None
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError("reference chart manifest must contain canonical JSONL records")
        try:
            source_row = strict_json_loads(line[:-1], require_canonical=True)
        except StrictJsonError:
            raise ValueError(
                "reference chart manifest must contain canonical JSONL records"
            ) from None
        if (
            not isinstance(source_row, dict)
            or source_row.get("schema_version") != REFERENCE_CHART_MANIFEST_SCHEMA
        ):
            raise ValueError("reference chart manifest contains an unsupported row")
        try:
            view = reference_chart_row_view_from_row(source_row)
        except ValueError:
            raise ValueError(
                "reference chart manifest contains an invalid reference chart row"
            ) from None

        identity = (
            view.corpus_version,
            view.source.source_endpoint_sha256,
            view.source.source_bucket,
            view.source.cache_profile,
            view.source.source_discovery_method,
        )
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise ValueError("reference chart manifest contains mixed source identity")
        if view.simfile_id in simfile_ids:
            raise ValueError("reference chart manifest contains duplicate simfile IDs")
        simfile_ids.add(view.simfile_id)
        rows.append(_ValidatedReferenceChartRow(MappingProxyType(source_row), view))

    if not rows:
        raise ValueError("reference chart manifest contains no records")
    normalized_rows = tuple(
        {key: value for key, value in validated.source_row.items() if key != "corpus_version"}
        for validated in rows
    )
    rendered = render_manifest(normalized_rows)
    assert source_identity is not None
    if rendered.content != content or rendered.corpus_version != source_identity[0]:
        raise ValueError("reference chart manifest has an invalid derived corpus version")
    return _LoadedReferenceChartManifest(
        source_reference_chart_manifest_sha256=sha256(content).hexdigest(),
        source_reference_chart_version=source_identity[0],
        rows=tuple(rows),
    )


def build_timing_row(
    validated: _ValidatedReferenceChartRow,
    *,
    source_reference_chart_manifest_sha256: str,
    source_reference_chart_version: str,
    timing: TimingRowResolution,
) -> dict[str, object]:
    """Remap one validated HPA-322 row into a derived timing-manifest row.

    Lineage remap (Brief Step 2):

    * ``schema_version`` is set to :data:`REFERENCE_TIMING_MANIFEST_SCHEMA`;
    * the HPA-322 top-level ``corpus_version`` is removed (``render_manifest``
      re-derives it);
    * ``source_manifest_sha256`` / ``source_corpus_version`` are carried through
      unchanged from the HPA-322 row;
    * ``source_reference_chart_manifest_sha256`` /
      ``source_reference_chart_version`` record the HPA-322 manifest identity.

    Every other HPA-322 field is passed through verbatim.  Reason codes and
    warnings are sorted so canonical rendering is byte-stable.
    """
    row = dict(validated.source_row)
    row.pop("corpus_version")
    row["schema_version"] = REFERENCE_TIMING_MANIFEST_SCHEMA
    row["source_reference_chart_manifest_sha256"] = source_reference_chart_manifest_sha256
    row["source_reference_chart_version"] = source_reference_chart_version
    row["timing_semantics_version"] = TIMING_SEMANTICS_VERSION
    row["timing_status"] = timing.status
    row["timing_reason_codes"] = sorted(timing.reason_codes)
    row["timing_warnings"] = sorted(timing.warnings)
    row["source_audio_key"] = timing.source_audio_key
    row["source_audio_content_hash"] = timing.source_audio_content_hash
    row["reference_events_cache_path"] = timing.reference_events_cache_path
    _validate_timing_status_shape(row)
    return row


def build_reference_timing_outcome(
    *,
    manifest: PublishedManifest,
    total_input_rows: int,
    ready_count: int,
    quarantined_count: int,
    upstream_quarantined_count: int,
    events_published: int,
) -> ReferenceTimingOutcome:
    """Build a successful :class:`ReferenceTimingOutcome`, enforcing invariants.

    Raises ``ValueError`` if the pure accounting invariants do not hold:

    * ``ready_count + quarantined_count == total_input_rows``;
    * ``upstream_quarantined_count <= quarantined_count``;
    * ``events_published == ready_count``.

    The exit convention is preserved: no quarantines -> ``0``; any quarantine
    with a published manifest -> ``1``.  Fatal loading / publication failures
    use :func:`failed_reference_timing_outcome` (exit ``2``).
    """
    if ready_count + quarantined_count != total_input_rows:
        raise ValueError("reference timing outcome must balance input rows")
    if upstream_quarantined_count > quarantined_count:
        raise ValueError("reference timing upstream quarantine exceeds total quarantine")
    if events_published != ready_count:
        raise ValueError("reference timing events published must equal ready rows")
    if quarantined_count == 0:
        status: Literal["complete", "partial"] = "complete"
        exit_code: Literal[0, 1, 2] = 0
    else:
        status = "partial"
        exit_code = 1
    return ReferenceTimingOutcome(
        status=status,
        exit_code=exit_code,
        manifest=manifest,
        ready_count=ready_count,
        quarantined_count=quarantined_count,
        upstream_quarantined_count=upstream_quarantined_count,
        events_published=events_published,
    )


def failed_reference_timing_outcome() -> ReferenceTimingOutcome:
    """The exit-2 outcome for a fatal loading / publication-prep failure."""
    return ReferenceTimingOutcome(
        status="failed",
        exit_code=2,
        manifest=None,
        ready_count=0,
        quarantined_count=0,
        upstream_quarantined_count=0,
        events_published=0,
    )


# ---------------------------------------------------------------------------
# Schema-golden validation for crux.reference-timing-manifest/v1
# ---------------------------------------------------------------------------


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate a ``crux.reference-timing-manifest/v1`` canonical JSONL golden.

    Asserts canonical JSONL (one final newline, no blank lines, each line
    canonical), exactly two records (one ready and one quarantined), a single
    shared HPA-322 source identity, a single derived ``corpus_version`` that
    round-trips through :func:`render_manifest`, and — per row — a valid
    timing payload whose underlying HPA-322 reference-chart fields re-validate
    through :func:`reference_chart_row_view_from_row`.  Reason codes are
    validated against ``get_args(TimingReasonCode)``.
    """
    if schema != REFERENCE_TIMING_MANIFEST_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference timing manifest golden must be canonical JSONL")

    lines = content.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reference timing manifest golden must contain exactly two records")
    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reference timing manifest golden must be canonical JSONL") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("reference timing manifest golden rows must be objects")

    for row in rows:
        _validate_timing_manifest_row(row)

    statuses = [row["timing_status"] for row in rows]
    if sorted(statuses) != ["quarantined", "ready"]:
        raise ValueError(
            "reference timing manifest golden requires one ready and one quarantined row"
        )

    source_identities = {
        (
            row["source_reference_chart_manifest_sha256"],
            row["source_reference_chart_version"],
            row["source_endpoint_sha256"],
            row["source_bucket"],
            row["cache_profile"],
            row["source_discovery_method"],
        )
        for row in rows
    }
    if len(source_identities) != 1:
        raise ValueError("reference timing manifest golden contains mixed source identity")

    derived_versions = {row["corpus_version"] for row in rows}
    if len(derived_versions) != 1:
        raise ValueError("reference timing manifest golden contains mixed corpus version")
    (derived_version,) = derived_versions
    if not _is_corpus_version(derived_version):
        raise ValueError("reference timing manifest golden has an invalid corpus version")
    normalized_rows = tuple(
        {key: value for key, value in row.items() if key != "corpus_version"} for row in rows
    )
    rendered = render_manifest(normalized_rows)
    if rendered.corpus_version != derived_version or rendered.content != content:
        raise ValueError("reference timing manifest golden has an invalid derived corpus version")


def _validate_timing_manifest_row(row: Mapping[str, object]) -> None:
    # Timing-specific + lineage keys must be present before the HPA-322
    # reconstruction below reads ``source_reference_chart_version``.
    if not (_TIMING_SPECIFIC_KEYS | _TIMING_LINEAGE_KEYS) <= set(row):
        raise ValueError("reference timing manifest row has an invalid key set")
    if "schema_version" not in row or "corpus_version" not in row:
        raise ValueError("reference timing manifest row has an invalid key set")

    # Reconstruct the HPA-322 row and delegate the complete key-set, schema,
    # digest, cache-path, DLEVEL, and selected/nullability contract to the
    # merged reference-chart validator.  This both validates the pass-through
    # payload and rejects any unknown / missing key.
    hpa322_row = {
        key: value
        for key, value in row.items()
        if key not in _TIMING_SPECIFIC_KEYS
        and key not in _TIMING_LINEAGE_KEYS
        and key != "corpus_version"
    }
    hpa322_row["schema_version"] = REFERENCE_CHART_MANIFEST_SCHEMA
    hpa322_row["corpus_version"] = row["source_reference_chart_version"]
    try:
        reference_chart_row_view_from_row(hpa322_row)
    except ValueError:
        raise ValueError(
            "reference timing manifest row has an invalid reference chart payload"
        ) from None

    if row["schema_version"] != REFERENCE_TIMING_MANIFEST_SCHEMA:
        raise ValueError("reference timing manifest row has an unsupported schema")
    if row["timing_semantics_version"] != TIMING_SEMANTICS_VERSION:
        raise ValueError(
            "reference timing manifest row has an unsupported timing semantics version"
        )
    _require_sha256_value(
        row["source_reference_chart_manifest_sha256"],
        "source_reference_chart_manifest_sha256",
    )
    if not _is_corpus_version(row["source_reference_chart_version"]):
        raise ValueError(
            "reference timing manifest row has an invalid source reference chart version"
        )
    if not _is_corpus_version(row["corpus_version"]):
        raise ValueError("reference timing manifest row has an invalid corpus version")
    warnings = row["timing_warnings"]
    if not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings):
        raise ValueError("reference timing manifest row has invalid timing warnings")
    _validate_timing_status_shape(row)


def _validate_timing_status_shape(row: Mapping[str, object]) -> None:
    """Validate the ready/quarantined shape shared by builder and validator."""
    status = row["timing_status"]
    if not isinstance(status, str) or status not in _TIMING_STATUSES:
        raise ValueError("reference timing row has an invalid timing status")
    _validate_timing_reason_codes(row["timing_reason_codes"])
    source_audio_fields = (
        "source_audio_key",
        "source_audio_content_hash",
        "reference_events_cache_path",
    )
    if status == "ready":
        if row["timing_reason_codes"]:
            raise ValueError("ready reference timing row must not carry reason codes")
        for field in source_audio_fields:
            value = row[field]
            if not isinstance(value, str) or not value:
                raise ValueError("ready reference timing row is missing source audio identity")
        _require_sha256_value(row["source_audio_content_hash"], "source_audio_content_hash")
        return
    if not row["timing_reason_codes"]:
        raise ValueError("quarantined reference timing row must carry reason codes")
    for field in source_audio_fields:
        if row[field] is not None:
            raise ValueError("quarantined reference timing row must null source audio identity")


def _validate_timing_reason_codes(value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or reason not in _TIMING_REASON_CODES for reason in value
    ):
        raise ValueError("reference timing row has invalid reason codes")


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
