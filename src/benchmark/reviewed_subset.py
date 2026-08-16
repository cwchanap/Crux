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

HPA-327 Task 4 adds the deterministic pre-score candidate selection
(:func:`build_candidate_stream`), the review-CSV preparation entrypoint
(:func:`prepare_reviewed_subset`), and the optional prior-ledger continuation
that preserves valid completed include reviews while replacing the rest from
the same deterministic stream.  HPA-327 Task 5 adds the finalization
entrypoint (:func:`finalize_reviewed_subset`) that re-runs the same slate
reproduction, validates the completed manual reviews, and publishes the
canonical complete review ledger plus the accepted subset manifest.

Selection metrics publish as canonical quantize-six string tokens (the same
CSV display token) because the shared manifest rail cannot round-trip
fractional JSON numbers.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Literal, get_args

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    SIX_PLACES,
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.cohort_scoring import score_cohort
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.durability import atomic_replace_bytes, ensure_durable_directory
from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    PublishedManifest,
    format_manifest_timestamp,
    parse_manifest_timestamp,
)
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    ReferenceMappingResult,
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
    read_canonical_manifest_core,
)
from src.benchmark.reports import write_cohort_reports

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

#: Generated review-CSV columns (24) followed by the manual audit columns (12).
REVIEW_CSV_FIELDS: tuple[str, ...] = (
    "review_policy_version",
    "selection_seed",
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
    "selected_chart_cache_path",
    "source_audio_key",
    "source_audio_content_hash",
    "source_audio_cache_path",
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
    "chart_selection_confirmed",
    "audio_revision_confirmed",
    "bgm_alignment_confirmed",
    "technical_mapping_confirmed",
    "musical_fidelity",
    "drum_character",
    "known_limitations",
    "decision",
    "reason_codes",
    "notes",
)

#: The 12 operator-authored review columns (a subset of :data:`REVIEW_CSV_FIELDS`).
REVIEW_MANUAL_FIELDS: tuple[str, ...] = REVIEW_CSV_FIELDS[REVIEW_CSV_FIELDS.index("reviewer") :]

#: Confirmation columns that must all be ``true`` for an included review.
_CONFIRMATION_FIELDS = (
    "chart_selection_confirmed",
    "audio_revision_confirmed",
    "bgm_alignment_confirmed",
    "technical_mapping_confirmed",
)

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
    reference_event_span_sec: Decimal
    common_event_density_per_sec: Decimal
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


@dataclass(frozen=True)
class ReviewCandidate:
    """One deterministic pre-score candidate derived from HPA-323/HPA-324.

    Metric fields stay numeric in memory; the CSV renderer quantizes them for
    display.  ``source_row_sha256`` is the exact HPA-324 row identity (minus
    ``corpus_version``) and is the only carry-forward guard for prior reviews.
    """

    simfile_id: int
    source_reference_manifest_sha256: str
    source_reference_manifest_version: str
    source_timing_manifest_sha256: str
    source_timing_manifest_version: str
    source_row_sha256: str
    selected_chart_key: str
    selected_chart_content_hash: str
    selected_chart_cache_path: str
    source_audio_key: str
    source_audio_content_hash: str
    source_audio_cache_path: str
    common_event_count: int
    reference_event_span_sec: float
    common_event_density_per_sec: float
    common_class_count: int
    density_band: Band
    class_richness_band: Band
    has_timing_warning: bool
    selects_real_or_full_chart: bool


@dataclass(frozen=True)
class PrepareReviewedSubsetRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    output_file: Path
    prior_ledger_path: Path | None = None


@dataclass(frozen=True)
class PrepareReviewedSubsetOutcome:
    exit_code: Literal[0, 2]
    output_file: Path | None
    candidate_count: int
    carried_include_count: int
    replacement_count: int


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


def _is_metric_token(value: object) -> bool:
    """Whether ``value`` is a canonical quantize-six metric string token.

    The shared manifest rail cannot round-trip fractional JSON numbers
    (``strict_json_loads`` parses them as ``Decimal`` and the canonical line
    writer is float-based), so span/density metrics publish as the exact CSV
    display token: parse as Decimal, quantize to six places, re-render, and
    require the byte-identical canonical form.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = Decimal(value)
        if not parsed.is_finite() or parsed < 0:
            return False
        return value == canonical_json_bytes(
            parsed.quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)
        ).decode("ascii")
    except (InvalidOperation, ValueError):
        return False


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
        if not _is_metric_token(row[field]):
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
        reference_event_span_sec=Decimal(row["reference_event_span_sec"]),  # type: ignore[arg-type]
        common_event_density_per_sec=Decimal(row["common_event_density_per_sec"]),  # type: ignore[arg-type]
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

    canonical = read_canonical_manifest_core(
        path,
        schema_version=REVIEWED_REFERENCE_SUBSET_SCHEMA,
        validate_rows=validate_rows,
    )

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


def _source_row_sha256(source_row: Mapping[str, object]) -> str:
    payload = {key: value for key, value in source_row.items() if key != "corpus_version"}
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _assign_bands(
    values: tuple[tuple[float | int, int], ...],
) -> dict[int, Band]:
    ordered = sorted(values)
    count = len(ordered)
    labels: tuple[Band, Band, Band] = ("low", "medium", "high")
    return {
        simfile_id: labels[min(2, (index * 3) // count)]
        for index, (_, simfile_id) in enumerate(ordered)
    }


def canonical_stratum_key(candidate: ReviewCandidate) -> str:
    return (
        f"{candidate.density_band}|{candidate.class_richness_band}|"
        f"{int(candidate.has_timing_warning)}|"
        f"{int(candidate.selects_real_or_full_chart)}"
    )


def _seeded_hash(value: str) -> str:
    return sha256(f"{REVIEW_SELECTION_SEED}:{value}".encode()).hexdigest()


def _csv_metric(value: float) -> str:
    return canonical_json_bytes(quantize_six(value)).decode("ascii")


def _csv_bool(value: bool) -> str:
    return canonical_json_bytes(value).decode("ascii")


def build_candidate_stream(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
) -> tuple[ReviewCandidate, ...]:
    """Build the deterministic seeded candidate stream over eligible rows.

    Only HPA-324 ``eligible`` rows participate, each with a non-``None``
    mapping whose reconstructed common-event count must match the published
    HPA-324 accounting.  Strata are ordered by the seeded stratum hash and
    rows within a stratum by their seeded source-row hash, then round-robined
    one row per stratum per round so the stream stays deterministic beyond the
    target count for continuation replacements.
    """
    timing_rows = {row.view.simfile_id: row for row in timing_manifest.rows}
    eligible: list[tuple[float, int, dict[str, object]]] = []
    for loaded in reference_manifest.rows:
        if loaded.view.eligibility_status != "eligible":
            continue
        simfile_id = loaded.view.simfile_id
        mapping = mappings.get(simfile_id)
        if mapping is None:
            raise ValueError("eligible reference row has no reference mapping")
        if len(mapping.common_events) != loaded.view.common_scored_event_count:
            raise ValueError("reference common event count does not match HPA-324")
        timing_row = timing_rows.get(simfile_id)
        if timing_row is None:
            raise ValueError("eligible reference row has no timing row")
        common_times = [float(event.canonical_audio_time) for event in mapping.common_events]
        span = common_times[-1] - common_times[0] if len(common_times) > 1 else 0.0
        density = len(common_times) / max(span, 1.0)
        class_count = len({event.common_class for event in mapping.common_events})
        source_row = loaded.source_row
        source_audio_content_hash = str(source_row["source_audio_content_hash"])
        selected_chart_key = str(source_row["selected_chart_key"])
        eligible.append(
            (
                density,
                simfile_id,
                {
                    "simfile_id": simfile_id,
                    "source_reference_manifest_sha256": reference_manifest.manifest_sha256,
                    "source_reference_manifest_version": reference_manifest.corpus_version,
                    "source_timing_manifest_sha256": timing_manifest.manifest_sha256,
                    "source_timing_manifest_version": timing_manifest.corpus_version,
                    "source_row_sha256": _source_row_sha256(source_row),
                    "selected_chart_key": selected_chart_key,
                    "selected_chart_content_hash": str(source_row["selected_chart_content_hash"]),
                    "selected_chart_cache_path": str(source_row["selected_chart_cache_path"]),
                    "source_audio_key": str(source_row["source_audio_key"]),
                    "source_audio_content_hash": source_audio_content_hash,
                    "source_audio_cache_path": (
                        f"sha256/{source_audio_content_hash[:2]}/{source_audio_content_hash}"
                    ),
                    "common_event_count": len(mapping.common_events),
                    "reference_event_span_sec": span,
                    "common_event_density_per_sec": density,
                    "common_class_count": class_count,
                    "has_timing_warning": bool(timing_row.view.timing_warnings),
                    "selects_real_or_full_chart": (
                        selected_chart_key.rsplit("/", 1)[-1].lower() in {"real.dtx", "full.dtx"}
                    ),
                },
            )
        )

    if not eligible:
        raise ValueError("no eligible reference rows are available for review selection")

    density_bands = _assign_bands(
        tuple((density, simfile_id) for density, simfile_id, _ in eligible)
    )
    richness_bands = _assign_bands(
        tuple((int(fields["common_class_count"]), simfile_id) for _, simfile_id, fields in eligible)
    )
    candidates = [
        ReviewCandidate(
            density_band=density_bands[simfile_id],  # type: ignore[index]
            class_richness_band=richness_bands[simfile_id],  # type: ignore[index]
            **fields,  # type: ignore[arg-type]
        )
        for _, simfile_id, fields in eligible
    ]

    strata: dict[str, list[ReviewCandidate]] = {}
    for candidate in candidates:
        strata.setdefault(canonical_stratum_key(candidate), []).append(candidate)
    for bucket in strata.values():
        bucket.sort(key=lambda candidate: _seeded_hash(candidate.source_row_sha256))
    stream: list[ReviewCandidate] = []
    for round_index in range(max(len(bucket) for bucket in strata.values())):
        for key in sorted(strata, key=_seeded_hash):
            bucket = strata[key]
            if round_index < len(bucket):
                stream.append(bucket[round_index])
    return tuple(stream)


def _candidate_csv_row(
    candidate: ReviewCandidate,
    *,
    rank: int,
    prior_ledger_sha256: str,
) -> dict[str, str]:
    return {
        "review_policy_version": REVIEW_POLICY_VERSION,
        "selection_seed": REVIEW_SELECTION_SEED,
        "prior_review_ledger_sha256": prior_ledger_sha256,
        "candidate_rank": str(rank),
        "simfile_id": str(candidate.simfile_id),
        "source_reference_manifest_sha256": candidate.source_reference_manifest_sha256,
        "source_reference_manifest_version": candidate.source_reference_manifest_version,
        "source_timing_manifest_sha256": candidate.source_timing_manifest_sha256,
        "source_timing_manifest_version": candidate.source_timing_manifest_version,
        "source_row_sha256": candidate.source_row_sha256,
        "selected_chart_key": candidate.selected_chart_key,
        "selected_chart_content_hash": candidate.selected_chart_content_hash,
        "selected_chart_cache_path": candidate.selected_chart_cache_path,
        "source_audio_key": candidate.source_audio_key,
        "source_audio_content_hash": candidate.source_audio_content_hash,
        "source_audio_cache_path": candidate.source_audio_cache_path,
        "common_event_count": str(candidate.common_event_count),
        "reference_event_span_sec": _csv_metric(candidate.reference_event_span_sec),
        "common_event_density_per_sec": _csv_metric(candidate.common_event_density_per_sec),
        "common_class_count": str(candidate.common_class_count),
        "density_band": candidate.density_band,
        "class_richness_band": candidate.class_richness_band,
        "has_timing_warning": _csv_bool(candidate.has_timing_warning),
        "selects_real_or_full_chart": _csv_bool(candidate.selects_real_or_full_chart),
    }


def _parse_prior_simfile_id(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError("prior ledger contains an invalid simfile ID")
    parsed = int(value)
    if value != str(parsed):
        raise ValueError("prior ledger contains a non-canonical simfile ID")
    return parsed


def _parse_prior_reasons(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    reasons = tuple(value.split(";"))
    if any(reason not in _REASON_CODES for reason in reasons):
        raise ValueError("prior ledger contains an invalid reason code")
    return reasons


def _validate_prior_manual(manual: Mapping[str, str]) -> None:
    decision = manual["decision"]
    if not decision:
        return  # unreviewed rows stay unreviewed and are re-offered
    if decision not in {"include", "exclude"}:
        raise ValueError("prior ledger contains an invalid decision")
    if not manual["reviewer"]:
        raise ValueError("prior ledger contains an incomplete reviewer")
    try:
        _parse_reviewed_at(manual["reviewed_at"])
    except ValueError:
        raise ValueError("prior ledger contains an invalid reviewed_at") from None
    if any(manual[field] not in {"true", "false"} for field in _CONFIRMATION_FIELDS):
        raise ValueError("prior ledger contains an invalid confirmation")
    if manual["musical_fidelity"] not in _MUSICAL_FIDELITIES:
        raise ValueError("prior ledger contains an invalid musical fidelity")
    if manual["drum_character"] not in _DRUM_CHARACTERS:
        raise ValueError("prior ledger contains an invalid drum character")
    reasons = _parse_prior_reasons(manual["reason_codes"])
    if decision == "include":
        if any(manual[field] != "true" for field in _CONFIRMATION_FIELDS):
            raise ValueError("prior ledger contains an include with failed confirmations")
        if manual["musical_fidelity"] not in {"close", "usable_with_limits"}:
            raise ValueError("prior ledger contains an include with unrepresentative fidelity")
    elif not reasons:
        raise ValueError("prior ledger contains an exclude without reason codes")
    if "other" in reasons and not manual["notes"]:
        raise ValueError("prior ledger requires notes for the other reason code")


@dataclass(frozen=True)
class _PriorReview:
    source_row_sha256: str
    decision: Literal["include", "exclude"] | None
    manual: dict[str, str]


def _parse_prior_ledger(content: bytes) -> dict[int, _PriorReview]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    if reader.fieldnames is None:
        raise ValueError("prior ledger has no header")
    missing = ({"simfile_id", "source_row_sha256"} | set(REVIEW_MANUAL_FIELDS)) - set(
        reader.fieldnames
    )
    if missing:
        raise ValueError("prior ledger is missing required columns")
    reviews: dict[int, _PriorReview] = {}
    for source in reader:
        if len(source) != len(reader.fieldnames) or None in source.values():
            raise ValueError("prior ledger contains a truncated row")
        simfile_id = _parse_prior_simfile_id(source["simfile_id"])
        if simfile_id in reviews:
            raise ValueError("prior ledger contains duplicate simfile IDs")
        manual = {field: source[field] for field in REVIEW_MANUAL_FIELDS}
        _validate_prior_manual(manual)
        decision: Literal["include", "exclude"] | None = None
        if manual["decision"]:
            decision = manual["decision"]  # type: ignore[assignment]
        reviews[simfile_id] = _PriorReview(
            source_row_sha256=source["source_row_sha256"],
            decision=decision,
            manual=manual,
        )
    return reviews


@dataclass(frozen=True)
class FinalizeReviewedSubsetRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    review_file: Path
    output_dir: Path
    prior_ledger_path: Path | None = None


@dataclass(frozen=True)
class FinalizeReviewedSubsetOutcome:
    exit_code: Literal[0, 2]
    manifest: PublishedManifest | None
    review_ledger_path: Path | None
    included_count: int
    excluded_count: int


@dataclass(frozen=True)
class ScoreReviewedSubsetRequest:
    """Local inputs for rescoring one persisted OaF run on a reviewed subset.

    ``run_path`` names the persisted ``crux.oaf-corpus-run/v1`` snapshot;
    ``output_dir`` receives the HPA-325 report artifacts for the subset cohort.
    """

    run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ScoreReviewedSubsetOutcome:
    exit_code: Literal[0, 1, 2]
    cohort_id: str | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int


def _fatal_score_outcome() -> ScoreReviewedSubsetOutcome:
    return ScoreReviewedSubsetOutcome(
        exit_code=2,
        cohort_id=None,
        reports_path=None,
        success_count=0,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
    )


def score_oaf_reviewed_subset(
    request: ScoreReviewedSubsetRequest,
) -> ScoreReviewedSubsetOutcome:
    """Rescore a persisted OaF run on the exact reviewed subset membership.

    Reconstructs the persisted cohort from the run snapshot (never constructs
    a backend), requires the run and reviewed-subset lineages to match the
    supplied HPA-323/HPA-324 manifests, filters to the subset's exact simfile
    IDs (keeping selected non-success rows), and scores through HPA-325 with
    event diagnostics retained for successful selected rows only.  Exit 0 when
    every selected row succeeds; exit 1 when any selected row is
    failed/skipped/quarantined (missing or invalid persisted prediction
    artifacts are item-level failures); exit 2 only for fatal
    run/lineage/membership/report errors.
    """
    # Local imports keep candidate preparation free of an import-time OaF
    # execution dependency (no backend, model, or TensorFlow at module load).
    from src.benchmark.oaf_corpus_run import (  # pylint: disable=import-outside-toplevel
        build_oaf_cohort_from_snapshot,
        parse_oaf_corpus_run,
    )

    try:
        if request.output_dir.resolve() == (request.run_path.parent / "reports").resolve():
            raise ValueError(
                "subset report output directory must not alias the parent run's broad reports"
            )
        reference = load_reference_set_manifest(request.reference_manifest_path)
        timing = load_reference_timing_manifest(request.timing_manifest_path)
        mappings = preflight_reference_mappings(
            reference,
            timing,
            timing_output_root=request.timing_manifest_path.parent.parent,
        )
        snapshot = parse_oaf_corpus_run(read_regular_file_no_follow(request.run_path))
        if snapshot.get("reference_manifest_sha256") != reference.manifest_sha256:
            raise ValueError("run snapshot reference manifest does not match the supplied manifest")
        if snapshot.get("reference_timing_manifest_sha256") != timing.manifest_sha256:
            raise ValueError("run snapshot timing manifest does not match the supplied manifest")
        if snapshot.get("reference_timing_version") != timing.corpus_version:
            raise ValueError("run snapshot timing version does not match the supplied manifest")

        subset = load_reviewed_subset_manifest(request.subset_manifest_path)
        if subset.source_reference_manifest_sha256 != reference.manifest_sha256 or (
            subset.source_reference_manifest_version != reference.corpus_version
        ):
            raise ValueError(
                "reviewed subset reference identity does not match the supplied manifest"
            )
        if subset.source_timing_manifest_sha256 != timing.manifest_sha256 or (
            subset.source_timing_manifest_version != timing.corpus_version
        ):
            raise ValueError("reviewed subset timing identity does not match the supplied manifest")

        parent_identity, parent_items = build_oaf_cohort_from_snapshot(
            snapshot,
            mappings=mappings,
            output_dir=request.run_path.parents[2],
        )
        subset_ids = {str(row.view.simfile_id) for row in subset.rows}
        missing = sorted(subset_ids - {item.simfile_id for item in parent_items})
        if missing:
            raise ValueError("reviewed subset members are absent from the parent population")
        selected_items = tuple(item for item in parent_items if item.simfile_id in subset_ids)
        subset_cohort_id = sha256(
            canonical_json_bytes(
                {
                    "parent_run_id": parent_identity.cohort_id,
                    "reviewed_subset_manifest_sha256": subset.manifest_sha256,
                }
            )
        ).hexdigest()
        subset_identity = replace(parent_identity, cohort_id=subset_cohort_id)
        diagnostics_for = tuple(
            sorted(item.simfile_id for item in selected_items if item.status == "success")
        )
        result = score_cohort(
            subset_identity,
            selected_items,
            diagnostics_for=diagnostics_for,
        )
        write_cohort_reports(result, request.output_dir)
    except (OSError, StrictJsonError, ValueError):
        return _fatal_score_outcome()
    population = result.population
    return ScoreReviewedSubsetOutcome(
        exit_code=(
            0
            if population.failed_count == 0
            and population.skipped_count == 0
            and population.quarantined_count == 0
            else 1
        ),
        cohort_id=subset_cohort_id,
        reports_path=request.output_dir,
        success_count=population.success_count,
        failed_count=population.failed_count,
        skipped_count=population.skipped_count,
        quarantined_count=population.quarantined_count,
    )


def _current_candidate_slate(
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    prior_ledger_path: Path | None,
) -> tuple[tuple[ReviewCandidate, ...], dict[int, dict[str, str]], str, int]:
    """Reproduce the current candidate slate exactly as preparation would.

    Loads HPA-323/HPA-324 through their canonical loaders, reconstructs the
    reference mappings once, builds the deterministic seeded stream, and
    applies the optional prior-ledger continuation (carry forward unchanged
    valid includes, consume unchanged excludes, replace from the stream).
    Returns the slate in rank order, the carried manual fields by simfile ID
    (empty when no prior ledger), the prior-ledger SHA-256 (``""`` when
    none), and the number of continuation replacements drawn from the stream
    (``0`` for a fresh preparation).  Both prepare and finalize call this so
    continuation membership and ranks come from a single implementation.
    """
    reference = load_reference_set_manifest(reference_manifest_path)
    timing = load_reference_timing_manifest(timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=timing_manifest_path.parent.parent,
    )
    stream = build_candidate_stream(reference, timing, mappings=mappings)
    if len(stream) < REVIEW_MIN_COUNT:
        raise ValueError("eligible population is below the review minimum")

    if prior_ledger_path is None:
        return stream[:REVIEW_TARGET_COUNT], {}, "", 0

    prior_bytes = prior_ledger_path.read_bytes()
    prior_ledger_sha256 = sha256(prior_bytes).hexdigest()
    prior_reviews = _parse_prior_ledger(prior_bytes)
    prior_unreviewed_ids = {
        prior_id for prior_id, prior in prior_reviews.items() if prior.decision is None
    }
    current = {candidate.simfile_id: candidate for candidate in stream}
    consumed: set[int] = set()
    carried: list[tuple[ReviewCandidate, dict[str, str]]] = []
    for prior_id, prior in prior_reviews.items():
        candidate = current.get(prior_id)
        if candidate is None or candidate.source_row_sha256 != prior.source_row_sha256:
            continue
        if prior.decision is None:
            continue
        consumed.add(prior_id)
        if prior.decision == "include":
            carried.append((candidate, prior.manual))
    selected = [candidate for candidate, _ in carried]
    for candidate in stream:
        if candidate.simfile_id in consumed:
            continue
        if len(selected) >= REVIEW_TARGET_COUNT:
            break
        selected.append(candidate)
    if len(selected) < REVIEW_MIN_COUNT:
        raise ValueError("continuation population is below the review minimum")
    carried_manual = {candidate.simfile_id: manual for candidate, manual in carried}
    replacement_count = sum(
        1
        for candidate in selected
        if candidate.simfile_id not in carried_manual
        and candidate.simfile_id not in prior_unreviewed_ids
    )
    return tuple(selected), carried_manual, prior_ledger_sha256, replacement_count


def prepare_reviewed_subset(
    request: PrepareReviewedSubsetRequest,
) -> PrepareReviewedSubsetOutcome:
    """Prepare the deterministic HPA-327 review CSV for the request.

    Delegates slate reproduction to :func:`_current_candidate_slate` (shared
    with finalization) and renders the current slate: fresh ranks ``1..N`` in
    slate order, prior-ledger SHA in every row, carried include reviews
    pre-filled, remaining manual cells empty.  Any
    lineage/population/continuation/write failure exits 2.
    """
    try:
        selected, carried_manual, prior_ledger_sha256, replacement_count = _current_candidate_slate(
            request.reference_manifest_path,
            request.timing_manifest_path,
            request.prior_ledger_path,
        )
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=REVIEW_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, candidate in enumerate(selected, start=1):
            row = _candidate_csv_row(
                candidate,
                rank=rank,
                prior_ledger_sha256=prior_ledger_sha256,
            )
            row.update(carried_manual.get(candidate.simfile_id, {}))
            writer.writerow(row)
        ensure_durable_directory(request.output_file.parent)
        atomic_replace_bytes(request.output_file, buffer.getvalue().encode("utf-8"))
    except (OSError, ValueError):
        return PrepareReviewedSubsetOutcome(
            exit_code=2,
            output_file=None,
            candidate_count=0,
            carried_include_count=0,
            replacement_count=0,
        )
    return PrepareReviewedSubsetOutcome(
        exit_code=0,
        output_file=request.output_file,
        candidate_count=len(selected),
        carried_include_count=len(carried_manual),
        replacement_count=replacement_count,
    )


def _parse_reviewed_at(value: str) -> datetime:
    """Parse an RFC3339 UTC timestamp, canonicalizing ``+00:00`` to ``Z``."""
    if value.endswith("+00:00"):
        value = value[:-6] + "Z"
    try:
        return parse_manifest_timestamp(value)
    except ValueError:
        raise ValueError("reviewed_at must be an RFC3339 UTC timestamp") from None


@dataclass(frozen=True)
class _CompletedReview:
    simfile_id: int
    reviewer: str
    reviewed_at: datetime
    confirmations: tuple[Literal["true", "false"], ...]
    musical_fidelity: MusicalFidelity
    drum_character: DrumCharacter
    known_limitations: str
    decision: Literal["include", "exclude"]
    reason_codes: tuple[ReviewReasonCode, ...]
    notes: str


def _parse_completed_reasons(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    reasons = tuple(value.split(";"))
    if any(reason not in _REASON_CODES for reason in reasons):
        raise ValueError("review file contains an invalid reason code")
    return tuple(sorted(set(reasons)))


def _parse_completed_review_row(
    simfile_id: int,
    manual: Mapping[str, str],
) -> _CompletedReview:
    reviewer = manual["reviewer"]
    if not reviewer:
        raise ValueError("review file contains an incomplete reviewer")
    try:
        reviewed_at = _parse_reviewed_at(manual["reviewed_at"])
    except ValueError:
        raise ValueError("review file contains an invalid reviewed_at") from None
    confirmations = tuple(manual[field] for field in _CONFIRMATION_FIELDS)
    if any(value not in {"true", "false"} for value in confirmations):
        raise ValueError("review file contains an invalid confirmation")
    fidelity = manual["musical_fidelity"]
    if fidelity not in _MUSICAL_FIDELITIES:
        raise ValueError("review file contains an invalid musical fidelity")
    character = manual["drum_character"]
    if character not in _DRUM_CHARACTERS:
        raise ValueError("review file contains an invalid drum character")
    decision = manual["decision"]
    if decision not in {"include", "exclude"}:
        raise ValueError("review file contains an invalid decision")
    reasons = _parse_completed_reasons(manual["reason_codes"])
    if decision == "include":
        if any(value != "true" for value in confirmations):
            raise ValueError("review file contains an include with failed confirmations")
        if fidelity not in {"close", "usable_with_limits"}:
            raise ValueError("review file contains an include with unrepresentative fidelity")
    elif not reasons:
        raise ValueError("review file contains an exclude without reason codes")
    if "other" in reasons and not manual["notes"]:
        raise ValueError("review file requires notes for the other reason code")
    return _CompletedReview(
        simfile_id=simfile_id,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        confirmations=confirmations,  # type: ignore[arg-type]
        musical_fidelity=fidelity,  # type: ignore[arg-type]
        drum_character=character,  # type: ignore[arg-type]
        known_limitations=manual["known_limitations"],
        decision=decision,  # type: ignore[arg-type]
        reason_codes=reasons,  # type: ignore[arg-type]
        notes=manual["notes"],
    )


def _parse_completed_review_file(content: bytes) -> dict[int, _CompletedReview]:
    """Parse a completed review CSV into per-simfile reviews.

    Requires the exact 36-column boundary and complete canonical rows; only
    ``simfile_id`` and the 12 manual columns are read as authority, so
    rewritten generated display cells never reject a review.
    """
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    if reader.fieldnames is None:
        raise ValueError("review file has no header")
    if set(reader.fieldnames) != set(REVIEW_CSV_FIELDS):
        raise ValueError("review file has an invalid column set")
    reviews: dict[int, _CompletedReview] = {}
    for source in reader:
        if len(source) != len(reader.fieldnames) or None in source.values():
            raise ValueError("review file contains a truncated row")
        simfile_id = _parse_prior_simfile_id(source["simfile_id"])
        if simfile_id in reviews:
            raise ValueError("review file contains duplicate simfile IDs")
        manual = {field: source[field] for field in REVIEW_MANUAL_FIELDS}
        reviews[simfile_id] = _parse_completed_review_row(simfile_id, manual)
    return reviews


def _render_complete_ledger(
    slate: tuple[ReviewCandidate, ...],
    reviews: Mapping[int, _CompletedReview],
    prior_ledger_sha256: str,
) -> bytes:
    """Re-render the canonical complete review ledger from fresh values.

    Generated columns are rebuilt from the reproduced slate; manual columns
    carry the validated review values with timestamps canonicalized to ``Z``
    and reasons re-joined in sorted unique order.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for rank, candidate in enumerate(slate, start=1):
        row = _candidate_csv_row(candidate, rank=rank, prior_ledger_sha256=prior_ledger_sha256)
        review = reviews[candidate.simfile_id]
        row.update(
            {
                "reviewer": review.reviewer,
                "reviewed_at": format_manifest_timestamp(review.reviewed_at),
            }
        )
        row.update(dict(zip(_CONFIRMATION_FIELDS, review.confirmations)))
        row.update(
            {
                "musical_fidelity": review.musical_fidelity,
                "drum_character": review.drum_character,
                "known_limitations": review.known_limitations,
                "decision": review.decision,
                "reason_codes": ";".join(review.reason_codes),
                "notes": review.notes,
            }
        )
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _subset_manifest_row(
    candidate: ReviewCandidate,
    review: _CompletedReview,
    *,
    rank: int,
    review_ledger_sha256: str,
    prior_review_ledger_sha256: str | None,
) -> dict[str, object]:
    """Build one accepted-row contract with fresh generated values bound to the ledger hash."""
    return {
        "schema_version": REVIEWED_REFERENCE_SUBSET_SCHEMA,
        "review_policy_version": REVIEW_POLICY_VERSION,
        "prior_review_ledger_sha256": prior_review_ledger_sha256,
        "review_ledger_sha256": review_ledger_sha256,
        "candidate_rank": rank,
        "simfile_id": candidate.simfile_id,
        "source_reference_manifest_sha256": candidate.source_reference_manifest_sha256,
        "source_reference_manifest_version": candidate.source_reference_manifest_version,
        "source_timing_manifest_sha256": candidate.source_timing_manifest_sha256,
        "source_timing_manifest_version": candidate.source_timing_manifest_version,
        "source_row_sha256": candidate.source_row_sha256,
        "selected_chart_key": candidate.selected_chart_key,
        "selected_chart_content_hash": candidate.selected_chart_content_hash,
        "source_audio_key": candidate.source_audio_key,
        "source_audio_content_hash": candidate.source_audio_content_hash,
        "common_event_count": candidate.common_event_count,
        "reference_event_span_sec": _csv_metric(candidate.reference_event_span_sec),
        "common_event_density_per_sec": _csv_metric(candidate.common_event_density_per_sec),
        "common_class_count": candidate.common_class_count,
        "density_band": candidate.density_band,
        "class_richness_band": candidate.class_richness_band,
        "has_timing_warning": candidate.has_timing_warning,
        "selects_real_or_full_chart": candidate.selects_real_or_full_chart,
        "reviewer": review.reviewer,
        "reviewed_at": format_manifest_timestamp(review.reviewed_at),
        "musical_fidelity": review.musical_fidelity,
        "drum_character": review.drum_character,
        "known_limitations": review.known_limitations,
        "reason_codes": list(review.reason_codes),
        "notes": review.notes,
    }


def finalize_reviewed_subset(
    request: FinalizeReviewedSubsetRequest,
) -> FinalizeReviewedSubsetOutcome:
    """Validate the completed review CSV and publish the canonical subset.

    Re-runs the same deterministic slate reproduction as preparation (same
    optional prior ledger), requires exactly one completed review per slate
    member, re-renders the canonical complete review ledger through the
    shared atomic byte-replacement helper, and publishes the accepted rows on
    the manifest rails.  Any lineage/population/review/publication failure
    exits 2.
    """
    try:
        slate, _, prior_ledger_sha256, _ = _current_candidate_slate(
            request.reference_manifest_path,
            request.timing_manifest_path,
            request.prior_ledger_path,
        )
        reviews = _parse_completed_review_file(request.review_file.read_bytes())
        candidates = {candidate.simfile_id: candidate for candidate in slate}
        if set(reviews) != set(candidates):
            raise ValueError("review file does not match the current candidate slate")

        included_count = sum(
            1 for candidate in slate if reviews[candidate.simfile_id].decision == "include"
        )
        if not REVIEW_MIN_COUNT <= included_count <= REVIEW_MAX_COUNT:
            raise ValueError("review file must include 20 to 30 candidates")

        review_ledger_bytes = _render_complete_ledger(slate, reviews, prior_ledger_sha256)
        review_ledger_sha256 = sha256(review_ledger_bytes).hexdigest()
        review_ledger_path = request.output_dir / "review-ledger.csv"
        ensure_durable_directory(request.output_dir)
        atomic_replace_bytes(review_ledger_path, review_ledger_bytes)

        rows = tuple(
            _subset_manifest_row(
                candidate,
                reviews[candidate.simfile_id],
                rank=rank,
                review_ledger_sha256=review_ledger_sha256,
                prior_review_ledger_sha256=prior_ledger_sha256 or None,
            )
            for rank, candidate in enumerate(slate, start=1)
            if reviews[candidate.simfile_id].decision == "include"
        )
        rendered = render_manifest(rows)
        published = publish_manifest(request.output_dir, rendered)
        publish_latest_manifest(
            request.output_dir,
            published,
            "complete",
            datetime.now(timezone.utc),
        )
    except (ManifestPublicationError, OSError, ValueError):
        return FinalizeReviewedSubsetOutcome(
            exit_code=2,
            manifest=None,
            review_ledger_path=None,
            included_count=0,
            excluded_count=0,
        )
    return FinalizeReviewedSubsetOutcome(
        exit_code=0,
        manifest=published,
        review_ledger_path=review_ledger_path,
        included_count=included_count,
        excluded_count=len(slate) - included_count,
    )
