"""Reviewed-reference-subset manifest contract (HPA-327 Task 3).

Defines the frozen ``crux.reviewed-reference-subset/v1`` row contract — closed
band/fidelity/character/reason enums, exact key sets, and canonical row
validation — plus the canonical loader
(:func:`load_reviewed_subset_manifest`) and the schema-golden validator
(:func:`validate_schema_golden`).  The loader requires a shared
source-reference/timing identity, review-policy version, and review-ledger
hashes across every row, unique positive candidate ranks and simfile IDs, and
a 20–30 row population (``REVIEW_MIN_COUNT``..``REVIEW_MAX_COUNT``); candidate
ranks need not be contiguous because excluded review candidates are absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, get_args

from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, parse_manifest_timestamp
from src.benchmark.reference_timing_manifest import read_canonical_manifest_core

#: Canonical schema id for the reviewed-reference-subset rows.
REVIEWED_REFERENCE_SUBSET_SCHEMA = "crux.reviewed-reference-subset/v1"

#: Frozen review policy family that produced the rows.
REVIEW_POLICY_VERSION = "hpa327-v1"

#: Target / minimum / maximum accepted population sizes.
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 30

#: Frozen deterministic candidate-stream seed.
REVIEW_SELECTION_SEED = "crux-hpa327-v1"

Band = Literal["low", "medium", "high"]
MusicalFidelity = Literal["close", "usable_with_limits", "not_representative"]
DrumCharacter = Literal["acoustic", "electronic", "hybrid", "unknown"]
ReviewReasonCode = Literal[
    "chart_selection_mismatch",
    "audio_revision_mismatch",
    "bgm_alignment_problem",
    "chart_audio_drift",
    "chart_simplification",
    "chart_authored_error",
    "unusual_lane_convention",
    "not_representative",
    "other",
]

_BANDS: frozenset[str] = frozenset(get_args(Band))
_MUSICAL_FIDELITIES: frozenset[str] = frozenset(get_args(MusicalFidelity))
_DRUM_CHARACTERS: frozenset[str] = frozenset(get_args(DrumCharacter))
_REASON_CODES: frozenset[str] = frozenset(get_args(ReviewReasonCode))

_REVIEWED_ROW_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "corpus_version",
        "review_policy_version",
        "review_ledger_sha256",
        "prior_review_ledger_sha256",
        "candidate_rank",
        "simfile_id",
        "source_reference_manifest_sha256",
        "source_reference_manifest_version",
        "source_timing_manifest_sha256",
        "source_timing_manifest_version",
        "source_row_sha256",
        "selected_chart_key",
        "selected_chart_content_hash",
        "source_audio_key",
        "source_audio_content_hash",
        "common_event_count",
        "reference_event_span_sec",
        "common_event_density_per_sec",
        "common_class_count",
        "density_band",
        "class_richness_band",
        "has_timing_warning",
        "selects_real_or_full_chart",
        "reviewer",
        "reviewed_at",
        "musical_fidelity",
        "drum_character",
        "known_limitations",
        "reason_codes",
        "notes",
    }
)


@dataclass(frozen=True)
class ReviewedSubsetRowView:
    simfile_id: int
    candidate_rank: int
    common_event_count: int
    reference_event_span_sec: float
    common_event_density_per_sec: float
    common_class_count: int
    density_band: Band
    class_richness_band: Band
    has_timing_warning: bool
    selects_real_or_full_chart: bool
    musical_fidelity: MusicalFidelity
    drum_character: DrumCharacter
    reason_codes: tuple[ReviewReasonCode, ...]


@dataclass(frozen=True)
class LoadedReviewedSubsetRow:
    source_row: Mapping[str, object]
    view: ReviewedSubsetRowView


@dataclass(frozen=True)
class LoadedReviewedSubsetManifest:
    manifest_sha256: str
    corpus_version: str
    review_policy_version: str
    review_ledger_sha256: str
    prior_review_ledger_sha256: str | None
    source_reference_manifest_sha256: str
    source_reference_manifest_version: str
    source_timing_manifest_sha256: str
    source_timing_manifest_version: str
    rows: tuple[LoadedReviewedSubsetRow, ...]


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


def _is_metric_number(value: object) -> bool:
    # The canonical renderer collapses whole-number Decimals to integer tokens
    # and strict_json_loads parses integer tokens back as ``int``; both are
    # legitimate for a metric field.
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, Decimal))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _validate_reviewed_subset_row(row: Mapping[str, object]) -> None:
    if set(row) != _REVIEWED_ROW_KEYS:
        raise ValueError("reviewed subset manifest row has an invalid key set")
    if row["schema_version"] != REVIEWED_REFERENCE_SUBSET_SCHEMA:
        raise ValueError("reviewed subset manifest row has an unsupported schema")
    if row["review_policy_version"] != REVIEW_POLICY_VERSION:
        raise ValueError("reviewed subset manifest row has an unsupported review policy version")

    _require_sha256_value(row["review_ledger_sha256"], "review_ledger_sha256")
    prior = row["prior_review_ledger_sha256"]
    if prior is not None:
        _require_sha256_value(prior, "prior_review_ledger_sha256")
    for field in (
        "source_reference_manifest_sha256",
        "source_timing_manifest_sha256",
        "source_row_sha256",
    ):
        _require_sha256_value(row[field], field)
    if not _is_corpus_version(row["source_reference_manifest_version"]):
        raise ValueError(
            "reviewed subset manifest row has an invalid source reference manifest version"
        )
    if not _is_corpus_version(row["source_timing_manifest_version"]):
        raise ValueError(
            "reviewed subset manifest row has an invalid source timing manifest version"
        )

    candidate_rank = row["candidate_rank"]
    simfile_id = row["simfile_id"]
    if (
        isinstance(candidate_rank, bool)
        or not isinstance(candidate_rank, int)
        or candidate_rank <= 0
    ):
        raise ValueError("reviewed subset manifest row has an invalid candidate rank")
    if (
        isinstance(simfile_id, bool)
        or not isinstance(simfile_id, int)
        or not 0 <= simfile_id <= MAX_SIMFILE_ID
    ):
        raise ValueError("reviewed subset manifest row has an invalid simfile ID")

    for field in ("selected_chart_key", "source_audio_key"):
        value = row[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"reviewed subset manifest row has an invalid {field}")
    for field in ("selected_chart_content_hash", "source_audio_content_hash"):
        _require_sha256_value(row[field], field)

    for field in ("common_event_count", "common_class_count"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("reviewed subset manifest row has invalid event counts")
    for field in ("reference_event_span_sec", "common_event_density_per_sec"):
        if not _is_metric_number(row[field]):
            raise ValueError("reviewed subset manifest row has invalid selection metrics")

    for field in ("density_band", "class_richness_band"):
        if row[field] not in _BANDS:
            raise ValueError(f"reviewed subset manifest row has an invalid {field}")
    for field in ("has_timing_warning", "selects_real_or_full_chart"):
        if not isinstance(row[field], bool):
            raise ValueError(f"reviewed subset manifest row has an invalid {field}")

    reviewer = row["reviewer"]
    reviewed_at = row["reviewed_at"]
    if not isinstance(reviewer, str) or not reviewer:
        raise ValueError("reviewed subset manifest row has an invalid reviewer")
    if not isinstance(reviewed_at, str):
        raise ValueError("reviewed subset manifest row has an invalid reviewed_at")
    try:
        parse_manifest_timestamp(reviewed_at)
    except ValueError:
        raise ValueError("reviewed subset manifest row has an invalid reviewed_at") from None

    if row["musical_fidelity"] not in _MUSICAL_FIDELITIES:
        raise ValueError("reviewed subset manifest row has an invalid musical fidelity")
    if row["drum_character"] not in _DRUM_CHARACTERS:
        raise ValueError("reviewed subset manifest row has an invalid drum character")
    if not isinstance(row["known_limitations"], str) or not isinstance(row["notes"], str):
        raise ValueError("reviewed subset manifest row has invalid audit text")

    reasons = row["reason_codes"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or reason not in _REASON_CODES for reason in reasons)
        or reasons != sorted(set(reasons))
    ):
        raise ValueError("reviewed subset manifest row has invalid reason codes")
    if "other" in reasons and not row["notes"]:
        raise ValueError("reviewed subset manifest row requires notes for the other reason code")


def _reviewed_subset_row_view_from_row(row: Mapping[str, object]) -> ReviewedSubsetRowView:
    """Build the typed view after reviewed-subset row validation."""
    return ReviewedSubsetRowView(
        simfile_id=row["simfile_id"],  # type: ignore[arg-type]
        candidate_rank=row["candidate_rank"],  # type: ignore[arg-type]
        common_event_count=row["common_event_count"],  # type: ignore[arg-type]
        reference_event_span_sec=float(row["reference_event_span_sec"]),  # type: ignore[arg-type]
        common_event_density_per_sec=float(row["common_event_density_per_sec"]),  # type: ignore[arg-type]
        common_class_count=row["common_class_count"],  # type: ignore[arg-type]
        density_band=row["density_band"],  # type: ignore[arg-type]
        class_richness_band=row["class_richness_band"],  # type: ignore[arg-type]
        has_timing_warning=row["has_timing_warning"],  # type: ignore[arg-type]
        selects_real_or_full_chart=row["selects_real_or_full_chart"],  # type: ignore[arg-type]
        musical_fidelity=row["musical_fidelity"],  # type: ignore[arg-type]
        drum_character=row["drum_character"],  # type: ignore[arg-type]
        reason_codes=tuple(row["reason_codes"]),  # type: ignore[arg-type]
    )


def load_reviewed_subset_manifest(path: Path) -> LoadedReviewedSubsetManifest:
    """Load an immutable HPA-327 manifest through the shared JSONL core."""
    rows: list[LoadedReviewedSubsetRow] = []
    simfile_ids: set[int] = set()
    candidate_ranks: set[int] = set()
    review_ledger_sha256: str | None = None
    prior_review_ledger_sha256: str | None = None
    source_reference_manifest_sha256: str | None = None
    source_reference_manifest_version: str | None = None
    source_timing_manifest_sha256: str | None = None
    source_timing_manifest_version: str | None = None

    def validate_rows(source_rows: tuple[Mapping[str, object], ...]) -> None:
        nonlocal review_ledger_sha256, prior_review_ledger_sha256
        nonlocal source_reference_manifest_sha256, source_reference_manifest_version
        nonlocal source_timing_manifest_sha256, source_timing_manifest_version
        for source_row in source_rows:
            _validate_reviewed_subset_row(source_row)
            view = _reviewed_subset_row_view_from_row(source_row)

            ledger = source_row["review_ledger_sha256"]
            prior = source_row["prior_review_ledger_sha256"]
            source_identity = (
                source_row["source_reference_manifest_sha256"],
                source_row["source_reference_manifest_version"],
                source_row["source_timing_manifest_sha256"],
                source_row["source_timing_manifest_version"],
            )
            if review_ledger_sha256 is None:
                assert source_reference_manifest_sha256 is None
                review_ledger_sha256 = ledger
                prior_review_ledger_sha256 = prior  # type: ignore[assignment]
                (
                    source_reference_manifest_sha256,
                    source_reference_manifest_version,
                    source_timing_manifest_sha256,
                    source_timing_manifest_version,
                ) = source_identity  # type: ignore[assignment]
            else:
                if ledger != review_ledger_sha256:
                    raise ValueError("reviewed subset manifest contains mixed review ledger hashes")
                if prior != prior_review_ledger_sha256:
                    raise ValueError(
                        "reviewed subset manifest contains mixed prior review ledger hashes"
                    )
                if source_identity != (
                    source_reference_manifest_sha256,
                    source_reference_manifest_version,
                    source_timing_manifest_sha256,
                    source_timing_manifest_version,
                ):
                    raise ValueError("reviewed subset manifest contains mixed source identity")
            if view.simfile_id in simfile_ids:
                raise ValueError("reviewed subset manifest contains duplicate simfile IDs")
            if view.candidate_rank in candidate_ranks:
                raise ValueError("reviewed subset manifest contains duplicate candidate ranks")
            simfile_ids.add(view.simfile_id)
            candidate_ranks.add(view.candidate_rank)
            rows.append(LoadedReviewedSubsetRow(source_row=source_row, view=view))

    try:
        canonical = read_canonical_manifest_core(
            path,
            schema_version=REVIEWED_REFERENCE_SUBSET_SCHEMA,
            validate_rows=validate_rows,
        )
    except TypeError:
        # The shared core re-renders parsed rows through ``render_manifest``,
        # whose canonical line writer cannot serialize the Decimal values that
        # ``strict_json_loads`` produces for fractional JSON numbers.  Surface
        # that round-trip ceiling as a load error instead of a crash.
        # ponytail: whole-number metrics only until render_manifest grows a
        # Decimal-aware line writer (canonical_json_bytes already has one).
        raise ValueError("reviewed subset manifest contains unsupported numeric values") from None

    row_count = len(rows)
    if not REVIEW_MIN_COUNT <= row_count <= REVIEW_MAX_COUNT:
        raise ValueError("reviewed subset manifest must contain 20 to 30 rows")
    if review_ledger_sha256 is None or source_reference_manifest_sha256 is None:
        raise ValueError("reviewed subset manifest contains no records")
    return LoadedReviewedSubsetManifest(
        manifest_sha256=canonical.manifest_sha256,
        corpus_version=canonical.corpus_version,
        review_policy_version=REVIEW_POLICY_VERSION,
        review_ledger_sha256=review_ledger_sha256,
        prior_review_ledger_sha256=prior_review_ledger_sha256,
        source_reference_manifest_sha256=source_reference_manifest_sha256,
        source_reference_manifest_version=source_reference_manifest_version,
        source_timing_manifest_sha256=source_timing_manifest_sha256,
        source_timing_manifest_version=source_timing_manifest_version,
        rows=tuple(rows),
    )


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate the one-row ``crux.reviewed-reference-subset/v1`` schema golden.

    Asserts canonical JSONL (one final newline, no blank lines, each line
    canonical), at least one record, per-row reviewed-subset shape validation
    (without the normal 20-row minimum), and a single derived ``corpus_version``
    that round-trips through :func:`render_manifest`.
    """
    if schema != REVIEWED_REFERENCE_SUBSET_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reviewed subset manifest golden must be canonical JSONL")

    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reviewed subset manifest golden must be canonical JSONL")
    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reviewed subset manifest golden must be canonical JSONL") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("reviewed subset manifest golden rows must be objects")

    for row in rows:
        _validate_reviewed_subset_row(row)  # type: ignore[arg-type]

    versions = {row["corpus_version"] for row in rows}
    if len(versions) != 1:
        raise ValueError("reviewed subset manifest golden contains mixed corpus version")
    (version,) = versions
    if not _is_corpus_version(version):
        raise ValueError("reviewed subset manifest golden has an invalid corpus version")
    normalized_rows = tuple(
        {key: value for key, value in row.items() if key != "corpus_version"} for row in rows
    )
    rendered = render_manifest(normalized_rows)
    if rendered.content != content or rendered.corpus_version != version:
        raise ValueError("reviewed subset manifest golden has an invalid derived corpus version")
