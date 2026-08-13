"""In-memory identity, coverage, and validation contract for cohort scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from statistics import fmean, median
from typing import Literal, get_args

from src.benchmark import scoring
from src.benchmark.backend_identity import OAF_BACKEND_ID, require_sha256
from src.benchmark.models import BenchmarkEvent, ScoreSummary
from src.benchmark.prediction_artifact import PredictionArtifact, read_prediction_artifact
from src.benchmark.reference_set import (
    ReferenceMappingResult,
    project_common_reference_events,
)
from src.benchmark.scorer_input import (
    prediction_to_benchmark_events,
    reference_to_benchmark_events,
)
from src.benchmark.scoring import ScoreResult, score_events_with_alignment
from src.benchmark.taxonomy import (
    DTX_LANE_MAP,
    DTX_LANE_MAP_VERSION,
    OAF_PREDICTION_MAP_ID,
    TAXONOMY_VERSION,
    project_to_common,
)

SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)
SCORE_MODES = ("raw", "aligned")
_ORIGINAL_PREDICTION_TIME_METADATA_KEY = "_crux_original_prediction_time_sec"

CohortExecutionStatus = Literal["success", "failed", "skipped", "quarantined"]
CohortFailureReason = Literal[
    "reference_quarantined",
    "backend_unavailable",
    "inference_failed",
    "prediction_artifact_invalid",
    "prediction_missing",
    "explicitly_skipped",
]
COHORT_FAILURE_REASONS = frozenset(get_args(CohortFailureReason))
ScoreMode = Literal["raw", "aligned"]


@dataclass(frozen=True)
class ClassScore:
    common_class: str
    summary: ScoreSummary
    reference_support: int
    prediction_support: int


@dataclass(frozen=True)
class SongScore:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    summary: ScoreSummary
    prediction_to_reference_ratio: float
    per_class: tuple[ClassScore, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PopulationSummary:
    total_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    reason_counts: tuple[tuple[CohortFailureReason, int], ...]


@dataclass(frozen=True)
class F1Distribution:
    minimum: float | None
    p10: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    maximum: float | None


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class CohortAggregate:
    tolerance_ms: int
    mode: ScoreMode
    event_micro: ScoreSummary
    song_macro_f1: float | None
    class_macro_f1: float | None
    song_f1_distribution: F1Distribution
    per_class: tuple[ClassScore, ...]
    successful_song_count: int


# pylint: enable=too-many-instance-attributes


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class CohortScoreResult:
    identity: CohortIdentity
    tolerances_ms: tuple[int, ...]
    items: tuple[CohortItem, ...]
    song_scores: tuple[SongScore, ...]
    event_diagnostics: tuple[EventDiagnostic, ...]
    population: PopulationSummary
    aggregates: tuple[CohortAggregate, ...]


# pylint: enable=too-many-instance-attributes


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class EventDiagnostic:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    outcome: Literal["matched", "false_positive", "false_negative"]
    common_class: str
    reference_time_sec: float | None
    prediction_time_sec: float | None
    scored_prediction_time_sec: float | None
    timing_error_sec: float | None


# pylint: enable=too-many-instance-attributes


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class CohortIdentity:
    cohort_id: str
    reference_manifest_sha256: str
    reference_timing_version: str
    taxonomy_version: str
    lane_map_version: str
    backend_id: str
    model_id: str
    model_lock_sha256: str
    backend_descriptor_sha256: str
    prediction_map_version: str
    input_view_id: str
    scoring_version: str = SCORING_VERSION

    def __post_init__(self) -> None:
        for field in (
            "cohort_id",
            "reference_manifest_sha256",
            "reference_timing_version",
            "taxonomy_version",
            "lane_map_version",
            "backend_id",
            "model_id",
            "model_lock_sha256",
            "backend_descriptor_sha256",
            "prediction_map_version",
            "input_view_id",
            "scoring_version",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        if self.scoring_version != SCORING_VERSION:
            raise ValueError(f"scoring_version must be {SCORING_VERSION}")
        if self.taxonomy_version != TAXONOMY_VERSION:
            raise ValueError(f"taxonomy_version must be {TAXONOMY_VERSION}")
        if self.lane_map_version != DTX_LANE_MAP_VERSION:
            raise ValueError(f"lane_map_version must be {DTX_LANE_MAP_VERSION}")
        for field in (
            "reference_manifest_sha256",
            "model_lock_sha256",
            "backend_descriptor_sha256",
        ):
            require_sha256(getattr(self, field), field)


@dataclass(frozen=True)
class CohortCoverage:
    reference_native_event_count: int
    reference_common_event_count: int
    reference_ignored_event_count: int
    reference_unmapped_event_count: int
    reference_duplicate_collapsed_count: int
    prediction_native_event_count: int | None
    prediction_mapped_event_count: int | None
    prediction_unmapped_event_count: int | None
    prediction_native_class_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CohortArtifactIdentity:
    """The descriptor-level provenance bound to one scoreable cohort item."""

    simfile_id: str
    backend_id: str
    model_id: str
    backend_descriptor_sha256: str
    input_view_id: str
    prediction_map_version: str

    def __post_init__(self) -> None:
        for field in (
            "simfile_id",
            "backend_id",
            "model_id",
            "backend_descriptor_sha256",
            "input_view_id",
            "prediction_map_version",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        require_sha256(self.backend_descriptor_sha256, "backend_descriptor_sha256")


# pylint: enable=too-many-instance-attributes


@dataclass(frozen=True)
class CohortItem:
    simfile_id: str
    status: CohortExecutionStatus
    reference_events: tuple[BenchmarkEvent, ...]
    prediction_events: tuple[BenchmarkEvent, ...] | None
    coverage: CohortCoverage
    warnings: tuple[str, ...] = ()
    failure_reason: CohortFailureReason | None = None
    artifact_identity: CohortArtifactIdentity | None = None
    reference_artifact: ReferenceMappingResult | None = None
    prediction_artifact: PredictionArtifact | None = None


def cohort_item_from_artifacts(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem:
    """Build one successful item while retaining artifact identity in memory.

    The persisted artifacts remain the source of descriptor, model, backend,
    input-view, map, and song identity.  Empty prediction artifacts use the
    existing OaF backend map identity because their event records carry no map
    field of their own.
    """
    if not isinstance(identity, CohortIdentity):
        raise TypeError("identity must be CohortIdentity")
    if not isinstance(simfile_id, str) or not simfile_id:
        raise ValueError("simfile_id must be a nonempty string")
    if not isinstance(reference, ReferenceMappingResult):
        raise TypeError("reference must be ReferenceMappingResult")
    if not isinstance(prediction, PredictionArtifact):
        raise TypeError("prediction must be PredictionArtifact")
    if not isinstance(warnings, tuple) or any(not isinstance(value, str) for value in warnings):
        raise TypeError("warnings must be a tuple of strings")

    artifact_identity = _artifact_identity_from_artifacts(simfile_id, reference, prediction)
    _validate_artifact_identity(identity, artifact_identity, simfile_id)

    item = CohortItem(
        simfile_id=simfile_id,
        status="success",
        reference_events=reference_to_benchmark_events(simfile_id, reference.common_events),
        prediction_events=prediction_to_benchmark_events(prediction),
        coverage=coverage_from_artifacts(reference, prediction),
        warnings=warnings,
        artifact_identity=artifact_identity,
        reference_artifact=reference,
        prediction_artifact=prediction,
    )
    validate_cohort_items(identity, (item,))
    return item


def coverage_from_artifacts(
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact | None,
) -> CohortCoverage:
    """Derive native/common coverage directly from validated domain artifacts."""
    if not isinstance(reference, ReferenceMappingResult):
        raise TypeError("reference must be ReferenceMappingResult")
    if prediction is not None and not isinstance(prediction, PredictionArtifact):
        raise TypeError("prediction must be PredictionArtifact or None")

    diagnostics = reference.diagnostics
    reference_ignored_event_count = sum(diagnostics.ignored.values())
    reference_unmapped_event_count = sum(diagnostics.unmapped.values())
    reference_native_event_count = (
        len(reference.mapped_events)
        + reference_ignored_event_count
        + reference_unmapped_event_count
    )

    if prediction is None:
        return CohortCoverage(
            reference_native_event_count=reference_native_event_count,
            reference_common_event_count=len(reference.common_events),
            reference_ignored_event_count=reference_ignored_event_count,
            reference_unmapped_event_count=reference_unmapped_event_count,
            reference_duplicate_collapsed_count=diagnostics.duplicate_common_event_count,
            prediction_native_event_count=None,
            prediction_mapped_event_count=None,
            prediction_unmapped_event_count=None,
            prediction_native_class_counts=(),
        )

    native_events = prediction.prediction.events
    prediction_mapped_event_count = sum(item.mapping_status == "mapped" for item in native_events)
    prediction_unmapped_event_count = sum(
        item.mapping_status == "unmapped" for item in native_events
    )
    native_class_counts = tuple(
        sorted(Counter(item.native.native_class_id for item in native_events).items())
    )
    return CohortCoverage(
        reference_native_event_count=reference_native_event_count,
        reference_common_event_count=len(reference.common_events),
        reference_ignored_event_count=reference_ignored_event_count,
        reference_unmapped_event_count=reference_unmapped_event_count,
        reference_duplicate_collapsed_count=diagnostics.duplicate_common_event_count,
        prediction_native_event_count=len(native_events),
        prediction_mapped_event_count=prediction_mapped_event_count,
        prediction_unmapped_event_count=prediction_unmapped_event_count,
        prediction_native_class_counts=native_class_counts,
    )


# pylint: disable-next=too-many-branches
def validate_cohort_items(
    identity: CohortIdentity,
    items: Iterable[CohortItem],
) -> None:
    """Validate row shapes and ensure successful predictions share one identity."""
    if not isinstance(identity, CohortIdentity):
        raise TypeError("identity must be CohortIdentity")

    seen_simfile_ids: set[str] = set()
    for item in items:
        if not isinstance(item, CohortItem):
            raise TypeError("cohort items must be CohortItem")
        if not isinstance(item.simfile_id, str) or not item.simfile_id:
            raise ValueError("simfile_id must be a nonempty string")
        if item.simfile_id in seen_simfile_ids:
            raise ValueError("simfile_id values must be unique")
        seen_simfile_ids.add(item.simfile_id)

        if item.status not in {"success", "failed", "skipped", "quarantined"}:
            raise ValueError("cohort item has an invalid status")
        _validate_event_sources(item)
        _validate_coverage(item.coverage, item.reference_events)

        if item.status == "success":
            if not item.reference_events:
                raise ValueError("success item requires nonempty reference_events")
            if item.prediction_events is None:
                raise ValueError("success item requires prediction_events")
            if item.failure_reason is not None:
                raise ValueError("success item requires failure_reason to be None")
            if item.artifact_identity is None:
                raise ValueError("success item requires artifact_identity")
            _validate_artifact_identity(identity, item.artifact_identity, item.simfile_id)
            _validate_success_artifact_binding(identity, item)
            if item.coverage.prediction_native_event_count is None:
                raise ValueError("success item requires prediction coverage")
            if item.coverage.prediction_mapped_event_count != len(item.prediction_events):
                raise ValueError("prediction mapped count must match prediction_events")
            for event in item.prediction_events:
                if event.metadata.get("input_view_id") != identity.input_view_id:
                    raise ValueError("prediction event input_view_id does not match cohort")
                if event.metadata.get("prediction_map_version") != (
                    identity.prediction_map_version
                ):
                    raise ValueError(
                        "prediction event prediction_map_version does not match cohort"
                    )
            continue

        if any(
            evidence is not None
            for evidence in (
                item.artifact_identity,
                item.reference_artifact,
                item.prediction_artifact,
            )
        ):
            raise ValueError(f"{item.status} item must not have artifact evidence")
        if item.prediction_events is not None:
            raise ValueError(f"{item.status} item must not have prediction_events")
        if any(
            value is not None
            for value in (
                item.coverage.prediction_native_event_count,
                item.coverage.prediction_mapped_event_count,
                item.coverage.prediction_unmapped_event_count,
            )
        ):
            raise ValueError(f"{item.status} item must not have prediction coverage")
        if item.coverage.prediction_native_class_counts:
            raise ValueError(f"{item.status} item must not have prediction native class counts")

        expected_reason: set[str]
        if item.status == "failed":
            expected_reason = {
                "backend_unavailable",
                "inference_failed",
                "prediction_artifact_invalid",
                "prediction_missing",
            }
            if item.failure_reason not in expected_reason:
                raise ValueError("failed item requires a prediction failure reason")
        elif item.status == "skipped":
            if item.failure_reason != "explicitly_skipped":
                raise ValueError("skipped item requires explicitly_skipped")
        else:
            if item.failure_reason != "reference_quarantined":
                raise ValueError("quarantined item requires reference_quarantined")


def _validate_artifact_identity(
    identity: CohortIdentity,
    artifact_identity: CohortArtifactIdentity,
    simfile_id: str,
) -> None:
    if not isinstance(artifact_identity, CohortArtifactIdentity):
        raise TypeError("artifact_identity must be CohortArtifactIdentity")
    if artifact_identity.simfile_id != simfile_id:
        raise ValueError("artifact simfile_id does not match item")
    expected_fields = (
        "backend_id",
        "model_id",
        "backend_descriptor_sha256",
        "input_view_id",
        "prediction_map_version",
    )
    for field in expected_fields:
        expected = getattr(identity, field)
        actual = getattr(artifact_identity, field)
        if actual != expected:
            raise ValueError(f"artifact {field} does not match cohort")


def _artifact_identity_from_artifacts(
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
) -> CohortArtifactIdentity:
    """Derive item provenance from the persisted artifact objects."""
    for mapped_event in reference.mapped_events:
        if str(mapped_event.native.simfile_id) != simfile_id:
            raise ValueError("reference artifact simfile_id does not match item")

    audio = prediction.prediction.audio
    if audio.source_audio_id != simfile_id:
        raise ValueError("prediction artifact source_audio_id does not match item")

    if reference.mapped_events:
        source_hashes = {
            mapped_event.native.source_audio_content_hash
            for mapped_event in reference.mapped_events
        }
        if len(source_hashes) > 1:
            raise ValueError("reference events have mixed source_audio_content_hash values")
        reference_source_hash = next(iter(source_hashes))
        if reference_source_hash != audio.source_audio_sha256:
            raise ValueError(
                "reference source_audio_content_hash does not match prediction source_audio_sha256"
            )

    descriptor = prediction.prediction.descriptor
    descriptor_payload = descriptor.payload
    backend_id = descriptor_payload.get("backend_id")
    model_id = descriptor_payload.get("model_id")
    if not isinstance(backend_id, str) or not backend_id:
        raise ValueError("prediction descriptor backend_id is invalid")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("prediction descriptor model_id is invalid")

    event_map_versions = {event.prediction_map_version for event in prediction.prediction.events}
    if len(event_map_versions) > 1:
        raise ValueError("prediction artifact has mixed prediction_map_version values")
    if event_map_versions:
        prediction_map_version = next(iter(event_map_versions))
    elif backend_id == OAF_BACKEND_ID:
        prediction_map_version = OAF_PREDICTION_MAP_ID
    else:
        raise ValueError("empty prediction artifact has no prediction_map_version")

    return CohortArtifactIdentity(
        simfile_id=simfile_id,
        backend_id=backend_id,
        model_id=model_id,
        backend_descriptor_sha256=descriptor.sha256,
        input_view_id=audio.input_view_id,
        prediction_map_version=prediction_map_version,
    )


# pylint: disable-next=too-many-branches
def _validate_success_artifact_binding(identity: CohortIdentity, item: CohortItem) -> None:
    """Recompute successful item evidence from the persisted artifacts."""
    if not isinstance(item.reference_artifact, ReferenceMappingResult):
        raise ValueError("success item requires reference artifact evidence")
    if not isinstance(item.prediction_artifact, PredictionArtifact):
        raise ValueError("success item requires prediction artifact evidence")

    canonical_prediction = read_prediction_artifact(item.prediction_artifact.content)
    if canonical_prediction != item.prediction_artifact:
        raise ValueError("prediction artifact fields do not match canonical content")

    projected_common_events = project_common_reference_events(item.reference_artifact.mapped_events)
    if item.reference_artifact.common_events != projected_common_events:
        raise ValueError("reference common_events do not match mapped event projection")
    expected_duplicate_count = sum(
        len(event.source_events) - 1 for event in projected_common_events
    )
    if item.reference_artifact.diagnostics.duplicate_common_event_count != expected_duplicate_count:
        raise ValueError("reference duplicate diagnostics do not match mapped event projection")

    for mapped_event in item.reference_artifact.mapped_events:
        lane_id = mapped_event.native.lane_id
        expected_mapping = DTX_LANE_MAP.get(lane_id)
        if expected_mapping is None or expected_mapping.canonical_class is None:
            raise ValueError("mapped reference event was not mapped using the frozen DTX lane map")
        if mapped_event.canonical_class != expected_mapping.canonical_class:
            raise ValueError(
                "mapped reference event canonical_class does not match frozen DTX lane map"
            )
        expected_common = expected_mapping.common_class or project_to_common(
            expected_mapping.canonical_class
        )
        if mapped_event.common_class != expected_common:
            raise ValueError(
                "mapped reference event common_class does not match frozen DTX lane map"
            )

    expected_identity = _artifact_identity_from_artifacts(
        item.simfile_id,
        item.reference_artifact,
        canonical_prediction,
    )
    if item.artifact_identity != expected_identity:
        raise ValueError("artifact_identity does not match persisted artifact evidence")
    _validate_artifact_identity(identity, expected_identity, item.simfile_id)
    _validate_prediction_event_metadata(identity, item.prediction_events or ())

    expected_reference_events = reference_to_benchmark_events(
        item.simfile_id,
        item.reference_artifact.common_events,
    )
    if item.reference_events != expected_reference_events:
        raise ValueError("reference_events do not match reference artifact evidence")

    expected_prediction_events = prediction_to_benchmark_events(canonical_prediction)
    if item.prediction_events != expected_prediction_events:
        raise ValueError("prediction_events do not match prediction artifact evidence")

    expected_coverage = coverage_from_artifacts(item.reference_artifact, canonical_prediction)
    if item.coverage != expected_coverage:
        raise ValueError("coverage does not match persisted artifact evidence")


def _validate_prediction_event_metadata(
    identity: CohortIdentity,
    prediction_events: tuple[BenchmarkEvent, ...],
) -> None:
    for event in prediction_events:
        if event.metadata.get("input_view_id") != identity.input_view_id:
            raise ValueError("prediction event input_view_id does not match cohort")
        if event.metadata.get("prediction_map_version") != identity.prediction_map_version:
            raise ValueError("prediction event prediction_map_version does not match cohort")


def _validate_event_sources(item: CohortItem) -> None:
    for event in item.reference_events:
        if not isinstance(event, BenchmarkEvent):
            raise TypeError("reference_events must contain BenchmarkEvent values")
        if event.chart_id != item.simfile_id:
            raise ValueError("reference event chart_id does not match item simfile_id")
        if event.source != "ground_truth":
            raise ValueError("reference event source must be ground_truth")
    if item.prediction_events is None:
        return
    for event in item.prediction_events:
        if not isinstance(event, BenchmarkEvent):
            raise TypeError("prediction_events must contain BenchmarkEvent values")
        if event.chart_id != item.simfile_id:
            raise ValueError("prediction event chart_id does not match item simfile_id")
        if event.source != "prediction":
            raise ValueError("prediction event source must be prediction")


def _validate_coverage(
    coverage: CohortCoverage,
    reference_events: tuple[BenchmarkEvent, ...],
) -> None:
    if not isinstance(coverage, CohortCoverage):
        raise TypeError("coverage must be CohortCoverage")
    reference_counts = (
        coverage.reference_native_event_count,
        coverage.reference_common_event_count,
        coverage.reference_ignored_event_count,
        coverage.reference_unmapped_event_count,
        coverage.reference_duplicate_collapsed_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in reference_counts
    ):
        raise ValueError("reference coverage counts must be nonnegative integers")
    if coverage.reference_native_event_count != (
        coverage.reference_common_event_count
        + coverage.reference_duplicate_collapsed_count
        + coverage.reference_ignored_event_count
        + coverage.reference_unmapped_event_count
    ):
        raise ValueError("reference coverage counts do not balance")
    if coverage.reference_common_event_count != len(reference_events):
        raise ValueError("reference common count must match reference_events")

    prediction_counts = (
        coverage.prediction_native_event_count,
        coverage.prediction_mapped_event_count,
        coverage.prediction_unmapped_event_count,
    )
    if any(value is None for value in prediction_counts):
        if not all(value is None for value in prediction_counts):
            raise ValueError("prediction coverage counts must be all present or all absent")
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in prediction_counts
        ):
            raise ValueError("prediction coverage counts must be nonnegative integers")
        assert (
            coverage.prediction_native_event_count is not None
            and coverage.prediction_mapped_event_count is not None
            and coverage.prediction_unmapped_event_count is not None
        )
        if coverage.prediction_native_event_count != (
            coverage.prediction_mapped_event_count + coverage.prediction_unmapped_event_count
        ):
            raise ValueError("prediction coverage counts do not balance")

    for native_class_id, count in coverage.prediction_native_class_counts:
        if not isinstance(native_class_id, str) or not native_class_id:
            raise ValueError("prediction native class ids must be nonempty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("prediction native class counts must be nonnegative integers")


def score_cohort(
    identity: CohortIdentity,
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...] = DEFAULT_TOLERANCES_MS,
    diagnostics_for: tuple[str, ...] = (),
) -> CohortScoreResult:
    """Validate, score, and aggregate a cohort in canonical order."""
    if not isinstance(identity, CohortIdentity):
        raise TypeError("identity must be CohortIdentity")

    canonical_tolerances = _validate_tolerances(tolerances_ms)
    canonical_items = tuple(items)
    validate_cohort_items(identity, canonical_items)
    canonical_diagnostics = _validate_diagnostics_for(diagnostics_for, canonical_items)
    canonical_items = tuple(sorted(canonical_items, key=lambda item: item.simfile_id))

    song_scores, event_diagnostics = _score_success_items(
        canonical_items,
        canonical_tolerances,
        frozenset(canonical_diagnostics),
    )
    population = _population_summary(canonical_items)
    aggregates = _cohort_aggregates(song_scores, canonical_tolerances)
    return CohortScoreResult(
        identity=identity,
        tolerances_ms=canonical_tolerances,
        items=canonical_items,
        song_scores=song_scores,
        event_diagnostics=event_diagnostics,
        population=population,
        aggregates=aggregates,
    )


def _validate_tolerances(tolerances_ms: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(tolerances_ms, tuple):
        raise TypeError("tolerances_ms must be a tuple")
    if not tolerances_ms:
        raise ValueError("tolerances_ms must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in tolerances_ms):
        raise ValueError("tolerances_ms must contain positive integers")
    if any(value <= 0 for value in tolerances_ms):
        raise ValueError("tolerances_ms must contain positive integers")
    if len(set(tolerances_ms)) != len(tolerances_ms):
        raise ValueError("tolerances_ms must contain unique values")
    if tolerances_ms != tuple(sorted(tolerances_ms)):
        raise ValueError("tolerances_ms must be sorted")
    return tolerances_ms


def _validate_diagnostics_for(
    diagnostics_for: tuple[str, ...],
    items: tuple[CohortItem, ...],
) -> tuple[str, ...]:
    if not isinstance(diagnostics_for, tuple):
        raise TypeError("diagnostics_for must be a tuple")
    if any(not isinstance(simfile_id, str) for simfile_id in diagnostics_for):
        raise ValueError("diagnostics_for IDs must be strings")
    if any(not simfile_id for simfile_id in diagnostics_for):
        raise ValueError("diagnostics_for IDs must be nonempty")
    if len(set(diagnostics_for)) != len(diagnostics_for):
        raise ValueError("diagnostics_for IDs must be unique")

    by_id = {item.simfile_id: item for item in items}
    for simfile_id in diagnostics_for:
        item = by_id.get(simfile_id)
        if item is None:
            raise ValueError("diagnostics_for ID names an input item")
        if item.status != "success":
            raise ValueError("diagnostics_for IDs must name successful items")
    return diagnostics_for


def _population_summary(items: tuple[CohortItem, ...]) -> PopulationSummary:
    status_counts = Counter(item.status for item in items)
    reason_counts = Counter(
        item.failure_reason for item in items if item.failure_reason is not None
    )
    return PopulationSummary(
        total_count=len(items),
        success_count=status_counts["success"],
        failed_count=status_counts["failed"],
        skipped_count=status_counts["skipped"],
        quarantined_count=status_counts["quarantined"],
        reason_counts=tuple(sorted(reason_counts.items())),
    )


def _cohort_aggregates(
    song_scores: tuple[SongScore, ...],
    tolerances_ms: tuple[int, ...],
) -> tuple[CohortAggregate, ...]:
    aggregates: list[CohortAggregate] = []
    for tolerance_ms in tolerances_ms:
        for mode in SCORE_MODES:
            rows = tuple(
                row for row in song_scores if row.tolerance_ms == tolerance_ms and row.mode == mode
            )
            event_micro = ScoreSummary(
                true_positives=sum(row.summary.true_positives for row in rows),
                false_positives=sum(row.summary.false_positives for row in rows),
                false_negatives=sum(row.summary.false_negatives for row in rows),
            )
            song_f1_values = tuple(row.summary.f1 for row in rows)
            per_class = _aggregate_class_scores(rows)
            aggregates.append(
                CohortAggregate(
                    tolerance_ms=tolerance_ms,
                    mode=mode,
                    event_micro=event_micro,
                    song_macro_f1=_mean_f1(song_f1_values),
                    class_macro_f1=_mean_f1(
                        tuple(row.summary.f1 for row in per_class if row.reference_support > 0)
                    ),
                    song_f1_distribution=_f1_distribution(song_f1_values),
                    per_class=per_class,
                    successful_song_count=len(rows),
                )
            )
    return tuple(aggregates)


def _mean_f1(values: tuple[float | None, ...]) -> float | None:
    scored = tuple(value for value in values if value is not None)
    return fmean(scored) if scored else None


def _f1_distribution(values: tuple[float | None, ...]) -> F1Distribution:
    scored = tuple(sorted(value for value in values if value is not None))
    if not scored:
        return F1Distribution(None, None, None, None, None, None, None)
    scored_list = list(scored)
    return F1Distribution(
        minimum=scored[0],
        p10=scoring.percentile(scored_list, 0.10),
        p25=scoring.percentile(scored_list, 0.25),
        median=median(scored),
        p75=scoring.percentile(scored_list, 0.75),
        p90=scoring.percentile(scored_list, 0.90),
        maximum=scored[-1],
    )


def _aggregate_class_scores(rows: tuple[SongScore, ...]) -> tuple[ClassScore, ...]:
    totals: dict[str, tuple[int, int, int, int, int]] = {}
    for row in rows:
        for class_row in row.per_class:
            previous = totals.get(class_row.common_class, (0, 0, 0, 0, 0))
            totals[class_row.common_class] = (
                previous[0] + class_row.summary.true_positives,
                previous[1] + class_row.summary.false_positives,
                previous[2] + class_row.summary.false_negatives,
                previous[3] + class_row.reference_support,
                previous[4] + class_row.prediction_support,
            )

    aggregate_rows: list[ClassScore] = []
    for common_class in sorted(totals):
        true_positives, false_positives, false_negatives, reference_support, prediction_support = (
            totals[common_class]
        )
        aggregate_rows.append(
            ClassScore(
                common_class=common_class,
                summary=ScoreSummary(
                    true_positives=true_positives,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                ),
                reference_support=reference_support,
                prediction_support=prediction_support,
            )
        )
    return tuple(aggregate_rows)


def _score_success_items(
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...],
    diagnostics_for: frozenset[str],
) -> tuple[tuple[SongScore, ...], tuple[EventDiagnostic, ...]]:
    """Score successful cohort items and optionally retain event diagnostics."""
    song_scores: list[SongScore] = []
    diagnostics: list[EventDiagnostic] = []

    for item in sorted(items, key=lambda item: item.simfile_id):
        if item.status != "success" or item.prediction_events is None:
            continue

        prediction_events = _prediction_events_with_provenance(item.prediction_events)
        for tolerance_ms in sorted(tolerances_ms):
            aligned = score_events_with_alignment(
                list(item.reference_events),
                list(prediction_events),
                tolerance_sec=tolerance_ms / 1000.0,
            )
            for mode in SCORE_MODES:
                score_result = aligned.raw if mode == "raw" else aligned.aligned
                per_class = _class_scores(score_result)
                song_scores.append(
                    SongScore(
                        simfile_id=item.simfile_id,
                        tolerance_ms=tolerance_ms,
                        mode=mode,
                        summary=score_result.summary,
                        prediction_to_reference_ratio=(
                            len(item.prediction_events) / len(item.reference_events)
                        ),
                        per_class=per_class,
                        warnings=tuple(item.warnings),
                    )
                )
                if item.simfile_id in diagnostics_for:
                    diagnostics.extend(_event_diagnostics(item, tolerance_ms, mode, score_result))

    song_scores.sort(key=_song_score_sort_key)
    diagnostics.sort(key=_event_diagnostic_sort_key)
    return tuple(song_scores), tuple(diagnostics)


def _class_scores(score_result: ScoreResult) -> tuple[ClassScore, ...]:
    classes = {match.ground_truth.canonical_class for match in score_result.matches}
    classes.update(match.prediction.canonical_class for match in score_result.matches)
    classes.update(event.canonical_class for event in score_result.unmatched_ground_truth)
    classes.update(event.canonical_class for event in score_result.unmatched_predictions)

    rows: list[ClassScore] = []
    for common_class in sorted(classes):
        true_positives = sum(
            match.ground_truth.canonical_class == common_class for match in score_result.matches
        )
        false_negatives = sum(
            event.canonical_class == common_class for event in score_result.unmatched_ground_truth
        )
        false_positives = sum(
            event.canonical_class == common_class for event in score_result.unmatched_predictions
        )
        summary = ScoreSummary(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
        rows.append(
            ClassScore(
                common_class=common_class,
                summary=summary,
                reference_support=true_positives + false_negatives,
                prediction_support=true_positives + false_positives,
            )
        )
    return tuple(rows)


def _event_diagnostics(
    item: CohortItem,
    tolerance_ms: int,
    mode: ScoreMode,
    score_result: ScoreResult,
) -> list[EventDiagnostic]:
    diagnostics: list[EventDiagnostic] = []

    for match in score_result.matches:
        prediction_time = _original_prediction_time(match.prediction, mode)
        diagnostics.append(
            EventDiagnostic(
                simfile_id=item.simfile_id,
                tolerance_ms=tolerance_ms,
                mode=mode,
                outcome="matched",
                common_class=match.ground_truth.canonical_class,
                reference_time_sec=match.ground_truth.time_sec,
                prediction_time_sec=prediction_time,
                scored_prediction_time_sec=match.prediction.time_sec,
                timing_error_sec=match.timing_error_sec,
            )
        )

    for event in score_result.unmatched_predictions:
        prediction_time = _original_prediction_time(event, mode)
        diagnostics.append(
            EventDiagnostic(
                simfile_id=item.simfile_id,
                tolerance_ms=tolerance_ms,
                mode=mode,
                outcome="false_positive",
                common_class=event.canonical_class,
                reference_time_sec=None,
                prediction_time_sec=prediction_time,
                scored_prediction_time_sec=event.time_sec,
                timing_error_sec=None,
            )
        )

    for event in score_result.unmatched_ground_truth:
        diagnostics.append(
            EventDiagnostic(
                simfile_id=item.simfile_id,
                tolerance_ms=tolerance_ms,
                mode=mode,
                outcome="false_negative",
                common_class=event.canonical_class,
                reference_time_sec=event.time_sec,
                prediction_time_sec=None,
                scored_prediction_time_sec=None,
                timing_error_sec=None,
            )
        )

    return diagnostics


def _original_prediction_time(
    event: BenchmarkEvent,
    mode: ScoreMode,
) -> float:
    if mode == "raw":
        return event.time_sec
    provenance = event.metadata.get(_ORIGINAL_PREDICTION_TIME_METADATA_KEY)
    if not isinstance(provenance, (int, float)) or isinstance(provenance, bool):
        raise ValueError("aligned prediction is missing original time provenance")
    return float(provenance)


def _prediction_events_with_provenance(
    events: tuple[BenchmarkEvent, ...],
) -> tuple[BenchmarkEvent, ...]:
    return tuple(
        replace(
            event,
            metadata={
                **event.metadata,
                _ORIGINAL_PREDICTION_TIME_METADATA_KEY: event.time_sec,
            },
        )
        for event in events
    )


def _song_score_sort_key(score: SongScore) -> tuple[str, int, int]:
    return score.simfile_id, score.tolerance_ms, _mode_rank(score.mode)


def _event_diagnostic_sort_key(
    diagnostic: EventDiagnostic,
) -> tuple[str, int, int, str, str, float, float]:
    return (
        diagnostic.simfile_id,
        diagnostic.tolerance_ms,
        _mode_rank(diagnostic.mode),
        diagnostic.outcome,
        diagnostic.common_class,
        -1.0 if diagnostic.reference_time_sec is None else diagnostic.reference_time_sec,
        -1.0 if diagnostic.prediction_time_sec is None else diagnostic.prediction_time_sec,
    )


def _mode_rank(mode: ScoreMode) -> int:
    return 0 if mode == "raw" else 1


__all__ = [
    "SCORING_VERSION",
    "DEFAULT_TOLERANCES_MS",
    "SCORE_MODES",
    "CohortExecutionStatus",
    "CohortFailureReason",
    "COHORT_FAILURE_REASONS",
    "ScoreMode",
    "ClassScore",
    "SongScore",
    "PopulationSummary",
    "F1Distribution",
    "CohortAggregate",
    "CohortScoreResult",
    "EventDiagnostic",
    "CohortIdentity",
    "CohortCoverage",
    "CohortArtifactIdentity",
    "CohortItem",
    "coverage_from_artifacts",
    "cohort_item_from_artifacts",
    "validate_cohort_items",
    "score_cohort",
    "_score_success_items",
]
