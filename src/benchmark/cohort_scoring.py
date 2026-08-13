"""In-memory identity, coverage, and validation contract for cohort scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from src.benchmark.backend_identity import require_sha256
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prediction_artifact import PredictionArtifact
from src.benchmark.reference_set import ReferenceMappingResult

SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)
SCORE_MODES = ("raw", "aligned")

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
        _validate_coverage(item.coverage, item.reference_events)

        if item.status == "success":
            if not item.reference_events:
                raise ValueError("success item requires nonempty reference_events")
            if item.prediction_events is None:
                raise ValueError("success item requires prediction_events")
            if item.failure_reason is not None:
                raise ValueError("success item requires failure_reason to be None")
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


__all__ = [
    "SCORING_VERSION",
    "DEFAULT_TOLERANCES_MS",
    "SCORE_MODES",
    "CohortExecutionStatus",
    "CohortFailureReason",
    "COHORT_FAILURE_REASONS",
    "ScoreMode",
    "CohortIdentity",
    "CohortCoverage",
    "CohortItem",
    "coverage_from_artifacts",
    "validate_cohort_items",
]
