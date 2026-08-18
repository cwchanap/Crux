"""Identity, input, snapshot, and sequential execution contracts.

This module owns the OaF corpus run boundary from validated inputs through one
persistent-worker inference lifecycle.
"""

# The pure identity, input, snapshot, and execution contracts intentionally live
# together so each run uses one canonical module-level boundary.
# pylint: disable=too-many-lines

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal, TypeAlias

from runtime.oaf_tf1.model import OafModelConfig, load_model_config
from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    PublishedArtifact,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    OAF_DESCRIPTOR_IDENTITIES,
    OAF_DESCRIPTOR_KEYS,
    OAF_DESCRIPTOR_SCHEMA,
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    build_descriptor,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio, NativePrediction
from src.benchmark.backends.oaf import (
    OAF_ADAPTER_REVISION,
    OafBackend,
    OafBackendError,
    create_backend,
)
from src.benchmark.cohort_scoring import (
    COHORT_FAILURE_REASONS,
    CohortIdentity,
    CohortItem,
    cohort_item_from_validated_prediction_artifact,
    cohort_item_without_prediction,
    score_cohort,
)
from src.benchmark.corpus_cache import CacheIndexStore, ResolvedSourceAudio, resolve_source_audio
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.input_view import materialize_full_mix_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.prediction_artifact import (
    PredictionArtifactError,
    prediction_artifact_matches_audio,
    prediction_artifact_matches_run_row,
    prediction_path,
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.reference_set import ReferenceMappingResult
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
)
from src.benchmark.reports import write_cohort_reports
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, OAF_PREDICTION_MAP_ID, TAXONOMY_VERSION

OAF_CORPUS_RUN_SCHEMA = "crux.oaf-corpus-run/v1"
OAF_INFERENCE_CONFIG_SCHEMA = "crux.oaf-inference-config/v1"
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
OAF_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0


OafRunStatus: TypeAlias = Literal["complete", "partial", "failed"]
OafRunExitCode: TypeAlias = Literal[0, 1, 2]


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OafCorpusRunRequest:
    """The immutable local inputs and scope for one OaF corpus run."""

    reference_manifest_path: Path
    timing_manifest_path: Path
    cache_dir: Path
    output_dir: Path
    include_simfile_ids: tuple[int, ...] = ()
    exclude_simfile_ids: tuple[int, ...] = ()
    resume: bool = False
    crux_commit: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "reference_manifest_path",
            "timing_manifest_path",
            "cache_dir",
            "output_dir",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        include_ids, exclude_ids = _normalize_scope(
            self.include_simfile_ids, self.exclude_simfile_ids
        )
        object.__setattr__(self, "include_simfile_ids", include_ids)
        object.__setattr__(self, "exclude_simfile_ids", exclude_ids)
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool")
        if self.crux_commit is not None:
            _require_commit(self.crux_commit)


@dataclass(frozen=True)
class OafCorpusRunOutcome:
    """Stable summary returned by the later execution/CLI layer."""

    overall_status: OafRunStatus
    exit_code: OafRunExitCode
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    aggregate_rtf: float | None
    projected_full_wall_time_sec: float | None

    def __post_init__(self) -> None:
        if self.overall_status not in {"complete", "partial", "failed"}:
            raise ValueError("overall_status is invalid")
        if self.exit_code not in {0, 1, 2} or isinstance(self.exit_code, bool):
            raise ValueError("exit_code is invalid")
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("run_id must be a nonempty string or None")
        for field in ("run_path", "reports_path"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{field} must be a Path or None")
        for field in ("success_count", "failed_count", "skipped_count", "quarantined_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in ("aggregate_rtf", "projected_full_wall_time_sec"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (float, int))
            ):
                raise TypeError(f"{field} must be a float or None")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")


# pylint: enable=too-many-instance-attributes


OAF_BACKEND_ERROR_POLICY: dict[str, tuple[str | None, str]] = {
    "inference_failed": ("inference_failed", "item_local"),
    "invalid_request": ("inference_failed", "item_local"),
    "input_path_invalid": ("canonical_input_failed", "item_local"),
    "native_event_invalid": ("inference_failed", "item_local"),
    "worker_error": ("worker_protocol_failed", "poison"),
    "worker_start_failed": ("backend_unavailable", "poison"),
    "worker_ready_invalid": ("backend_unavailable", "poison"),
    "worker_identity_invalid": ("backend_unavailable", "poison"),
    "worker_response_invalid": ("worker_protocol_failed", "poison"),
    "backend_closed": ("worker_protocol_failed", "poison"),
    "descriptor_invalid": (None, "fatal_preflight"),
    "worker_close_failed": (None, "finalization"),
}

RUNNER_FAILURE_TO_COHORT_REASON: dict[str, str] = {
    "source_audio_unavailable": "inference_failed",
    "source_audio_decode_failed": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "backend_unavailable": "backend_unavailable",
    "worker_protocol_failed": "backend_unavailable",
    "inference_failed": "inference_failed",
    "prediction_artifact_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
    "prediction_missing": "prediction_missing",
    "explicitly_skipped": "explicitly_skipped",
}

if not set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS:
    raise RuntimeError(  # pragma: no cover
        "runner failure mapping contains an unknown cohort reason"
    )


_SHA256_FIELDS = frozenset(
    {
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
        "reference_manifest_sha256",
        "reference_timing_manifest_sha256",
        "inference_config_sha256",
    }
)
_INFERENCE_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "canonicalization_revision",
    }
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SNAPSHOT_DISPOSITIONS = frozenset({"inferred", "resumed", "skipped", "quarantined", "failed"})
_SNAPSHOT_COUNTS = ("success_count", "failed_count", "skipped_count", "quarantined_count")


def _require_commit(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise StrictJsonError("crux_commit must be a lowercase 40-character commit")
    return value


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise StrictJsonError(f"{field} must be lowercase SHA-256")
    return require_sha256(value, field)


def compute_model_lock_sha256(path: Path) -> str:
    """Return the exact SHA-256 of one checked-in model-lock file."""
    if not isinstance(path, Path):
        raise TypeError("model lock path must be a Path")
    return sha256_hex(read_regular_file_no_follow(path))


def build_inference_config(
    config: OafModelConfig,
    descriptor: BackendDescriptor,
    model_lock_sha256: str,
    *,
    input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID,
) -> dict[str, str]:
    """Build the closed, inference-semantic identity payload."""
    if not isinstance(config, OafModelConfig):
        raise TypeError("config must be OafModelConfig")
    if not isinstance(descriptor, BackendDescriptor):
        raise TypeError("descriptor must be BackendDescriptor")
    descriptor_sha256 = _require_hash(descriptor.sha256, "backend_descriptor_sha256")
    model_lock = _require_hash(model_lock_sha256, "model_lock_sha256")
    if not isinstance(input_view_id, str) or not input_view_id:
        raise ValueError("input_view_id must be a nonempty string")
    checkpoint = _require_hash(
        config.checkpoint.archive_sha256,
        "checkpoint_archive_sha256",
    )
    return {
        "schema": OAF_INFERENCE_CONFIG_SCHEMA,
        "backend_descriptor_sha256": descriptor_sha256,
        "model_lock_sha256": model_lock,
        "checkpoint_archive_sha256": checkpoint,
        "adapter_revision": OAF_ADAPTER_REVISION,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "input_view_id": input_view_id,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
    }


def inference_config_sha256(payload: Mapping[str, object]) -> str:
    """Hash exactly one canonical inference configuration payload."""
    if not isinstance(payload, Mapping) or set(payload) != _INFERENCE_CONFIG_FIELDS:
        raise StrictJsonError("inference config must contain the exact key set")
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not value:
            raise StrictJsonError(f"inference config {key} must be a nonempty string")
        normalized[key] = value
    if normalized["schema"] != OAF_INFERENCE_CONFIG_SCHEMA:
        raise StrictJsonError(f"inference config schema must be {OAF_INFERENCE_CONFIG_SCHEMA}")
    for field in _SHA256_FIELDS & _INFERENCE_CONFIG_FIELDS:
        _require_hash(normalized[field], field)
    return sha256_hex(canonical_json_bytes(normalized))


def _normalize_scope(
    include_simfile_ids: Iterable[int], exclude_simfile_ids: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def normalize(values: Iterable[int], field: str) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{field} must contain integer simfile IDs")
        try:
            values_tuple = tuple(values)
        except TypeError as error:
            raise ValueError(f"{field} must contain integer simfile IDs") from error
        for value in values_tuple:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must contain positive integer simfile IDs")
        return tuple(sorted(set(values_tuple)))

    include = normalize(include_simfile_ids, "include_simfile_ids")
    exclude = normalize(exclude_simfile_ids, "exclude_simfile_ids")
    overlap = sorted(set(include) & set(exclude))
    if overlap:
        raise ValueError(f"include/exclude scope overlap: {overlap}")
    return include, exclude


def _validate_scope(
    include_simfile_ids: Iterable[int],
    exclude_simfile_ids: Iterable[int],
    loaded_manifest_ids: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalize and validate scope membership before any backend work."""
    include, exclude = _normalize_scope(include_simfile_ids, exclude_simfile_ids)
    try:
        loaded = set(loaded_manifest_ids)
    except TypeError as error:
        raise ValueError("loaded manifest IDs must be iterable") from error
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in loaded):
        raise ValueError("loaded manifest IDs must be positive integers")
    unknown_include = sorted(set(include) - loaded)
    unknown_exclude = sorted(set(exclude) - loaded)
    if unknown_include:
        raise ValueError(f"unknown include simfile IDs: {unknown_include}")
    if unknown_exclude:
        raise ValueError(f"unknown exclude simfile IDs: {unknown_exclude}")
    return include, exclude


# pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-outer-name
def build_run_id(
    reference_manifest_sha256: str,
    reference_timing_manifest_sha256: str,
    backend_descriptor_sha256: str,
    model_lock_sha256: str,
    checkpoint_archive_sha256: str,
    inference_config_sha256: str,
    include_simfile_ids: Iterable[int] = (),
    exclude_simfile_ids: Iterable[int] = (),
) -> str:
    """Derive a deterministic cohort-run identity from immutable inputs."""
    include, exclude = _normalize_scope(include_simfile_ids, exclude_simfile_ids)
    values = {
        "reference_manifest_sha256": _require_hash(
            reference_manifest_sha256, "reference_manifest_sha256"
        ),
        "reference_timing_manifest_sha256": _require_hash(
            reference_timing_manifest_sha256,
            "reference_timing_manifest_sha256",
        ),
        "backend_descriptor_sha256": _require_hash(
            backend_descriptor_sha256, "backend_descriptor_sha256"
        ),
        "model_lock_sha256": _require_hash(model_lock_sha256, "model_lock_sha256"),
        "checkpoint_archive_sha256": _require_hash(
            checkpoint_archive_sha256, "checkpoint_archive_sha256"
        ),
        "adapter_revision": OAF_ADAPTER_REVISION,
        "inference_config_sha256": _require_hash(
            inference_config_sha256, "inference_config_sha256"
        ),
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "include_simfile_ids": list(include),
        "exclude_simfile_ids": list(exclude),
    }
    return "oaf-" + sha256_hex(canonical_json_bytes(values))[:16]


# pylint: enable=too-many-arguments,too-many-positional-arguments,redefined-outer-name


def _materialize_oaf_full_mix(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
    config: OafModelConfig,
) -> CanonicalAudio:
    """Validate OaF config before calling the model-neutral materializer."""
    if not isinstance(config, OafModelConfig):
        raise TypeError("config must be OafModelConfig")
    return materialize_full_mix_audio(
        source_audio,
        output_path,
        input_root=input_root,
        input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
        max_input_audio_frames=config.max_input_audio_frames,
    )


def classify_oaf_backend_error(code: str) -> tuple[str | None, str]:
    """Map a backend code to runner failure family and lifecycle disposition."""
    if not isinstance(code, str):
        return "worker_protocol_failed", "poison"
    return OAF_BACKEND_ERROR_POLICY.get(code, ("worker_protocol_failed", "poison"))


def _normalize_snapshot_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str, Decimal)):
        return value  # type: ignore[return-value]
    if isinstance(value, float):
        return quantize_six(value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJsonError("run snapshot object keys must be strings")
            normalized[key] = _normalize_snapshot_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_snapshot_value(item) for item in value]
    raise StrictJsonError(f"unsupported run snapshot value: {type(value).__name__}")


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def _validate_snapshot(snapshot: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if snapshot.get("schema") != OAF_CORPUS_RUN_SCHEMA:
        raise StrictJsonError(f"run snapshot schema must be {OAF_CORPUS_RUN_SCHEMA}")
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise StrictJsonError("run snapshot run_id must be a nonempty string")

    include = snapshot.get("include_simfile_ids", [])
    exclude = snapshot.get("exclude_simfile_ids", [])
    if not isinstance(include, list) or not isinstance(exclude, list):
        raise StrictJsonError("run snapshot scope must be arrays")
    normalized_include, normalized_exclude = _normalize_scope(include, exclude)
    snapshot["include_simfile_ids"] = list(normalized_include)
    snapshot["exclude_simfile_ids"] = list(normalized_exclude)

    items = snapshot.get("items", [])
    if not isinstance(items, list):
        raise StrictJsonError("run snapshot items must be an array")
    item_rows: list[dict[str, JsonValue]] = []
    item_ids: set[int] = set()
    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise StrictJsonError("run snapshot item must be an object")
        simfile_id = raw_item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int) or simfile_id <= 0:
            raise StrictJsonError("run snapshot item simfile_id is invalid")
        if simfile_id in item_ids:
            raise StrictJsonError("run snapshot items must have unique simfile IDs")
        item_ids.add(simfile_id)
        disposition = raw_item.get("execution_disposition")
        if disposition is not None:
            if not isinstance(disposition, str) or disposition not in _SNAPSHOT_DISPOSITIONS:
                raise StrictJsonError("run snapshot execution disposition is invalid")
            if disposition == "skipped" and simfile_id not in set(normalized_exclude):
                raise StrictJsonError("skipped item requires explicit filter exclusion")
        item_rows.append(raw_item)
    item_rows.sort(key=lambda item: int(item["simfile_id"]))
    snapshot["items"] = item_rows

    overall_status = snapshot.get("overall_status")
    if overall_status is not None and overall_status not in {"complete", "partial", "failed"}:
        raise StrictJsonError("run snapshot overall_status is invalid")
    if overall_status == "complete":
        missing_counts = [field for field in _SNAPSHOT_COUNTS if field not in snapshot]
        if missing_counts:
            raise StrictJsonError("complete run snapshot requires all counts")
        missing_dispositions = [
            item["simfile_id"]
            for item in item_rows
            if item.get("execution_disposition") not in _SNAPSHOT_DISPOSITIONS
        ]
        if missing_dispositions:
            raise StrictJsonError("complete run snapshot requires item dispositions")

    if all(field in snapshot for field in _SNAPSHOT_COUNTS):
        expected = {field: snapshot[field] for field in _SNAPSHOT_COUNTS}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected.values()
        ):
            raise StrictJsonError("run snapshot counts must be nonnegative integers")
        actual = {field: 0 for field in _SNAPSHOT_COUNTS}
        for item in item_rows:
            disposition = item.get("execution_disposition")
            if disposition in {"inferred", "resumed"}:
                actual["success_count"] += 1
            elif disposition == "failed":
                actual["failed_count"] += 1
            elif disposition == "skipped":
                actual["skipped_count"] += 1
            elif disposition == "quarantined":
                actual["quarantined_count"] += 1
        if (
            overall_status == "complete"
            or all("execution_disposition" in item for item in item_rows)
        ) and expected != actual:
            raise StrictJsonError("run snapshot counts do not reconcile with items")

    completed_at = snapshot.get("completed_at")
    if completed_at is not None and overall_status != "complete":
        raise StrictJsonError("completed timestamp is only valid for a completed run")
    return snapshot


# pylint: enable=too-many-branches,too-many-locals,too-many-statements


def render_oaf_corpus_run(snapshot: Mapping[str, object]) -> bytes:
    """Normalize float-derived values and render one canonical run document."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("run snapshot must be a mapping")
    normalized = _normalize_snapshot_value(snapshot)
    if not isinstance(normalized, dict):
        raise StrictJsonError("run snapshot must be an object")  # pragma: no cover
    return canonical_json_bytes(_validate_snapshot(normalized))


def parse_oaf_corpus_run(
    content: bytes, *, expected_run_id: str | None = None
) -> dict[str, JsonValue]:
    """Parse and validate canonical run bytes, preserving Decimal numbers."""
    value = strict_json_loads(content, require_canonical=True)
    if not isinstance(value, dict):
        raise StrictJsonError("run snapshot must be an object")
    validated = _validate_snapshot(value)
    if canonical_json_bytes(validated) != content:
        raise StrictJsonError("run snapshot is not semantically canonical")
    if expected_run_id is not None and validated.get("run_id") != expected_run_id:
        raise StrictJsonError("run snapshot run_id does not match expected identity")
    return validated


def write_oaf_corpus_run(run_path: Path, snapshot: Mapping[str, object]) -> None:
    """Publish a run snapshot through the shared durable replacement helper."""
    if not isinstance(run_path, Path):
        raise TypeError("run_path must be a Path")
    content = render_oaf_corpus_run(snapshot)
    atomic_replace_bytes(run_path, content)


# The private spelling is retained for the execution layer's checkpoint calls.
_write_oaf_corpus_run = write_oaf_corpus_run


# pylint: disable=too-many-arguments,too-many-return-statements,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
@dataclass
class _OafExecutionState:
    loaded: LoadedReferenceSetRow
    mapping: ReferenceMappingResult | None
    snapshot: dict[str, object]
    source: ResolvedSourceAudio | None = None
    source_audio_kwargs: dict[str, str | None] | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _model_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime" / "oaf_tf1" / "model.json"


def _expected_oaf_descriptor(config: OafModelConfig) -> BackendDescriptor:
    payload = dict(OAF_DESCRIPTOR_IDENTITIES)
    payload.update(
        {
            "backend_id": config.backend_id,
            "model_id": config.model_id,
            "architecture_id": config.architecture_id,
            "upstream_source_commit": config.upstream_source_commit,
            "training_data_map_id": config.training_data_map_id,
            "native_output_space_id": config.native_output_space_id,
            "native_metadata_schema_id": config.native_metadata_schema_id,
            "prediction_schema": "crux.drum-prediction-events/v2",
            "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        }
    )
    return build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _bounded_error(error: BaseException, *, limit: int = 512) -> str:
    detail = str(error).replace("\x00", " ").strip()
    if not detail:
        detail = type(error).__name__
    return detail[:limit]


def _bounded_close_error(error: OafBackendError | BaseException) -> dict[str, str]:
    code = getattr(error, "code", "worker_close_failed")
    if not isinstance(code, str) or not code:
        code = "worker_close_failed"
    return {"code": code[:128], "message": _bounded_error(error)}


def _snapshot_item(loaded: LoadedReferenceSetRow) -> dict[str, object]:
    source = loaded.source_row
    row: dict[str, object] = {
        "simfile_id": loaded.view.simfile_id,
        "eligibility_status": loaded.view.eligibility_status,
        "eligibility_reason_codes": list(loaded.view.eligibility_reason_codes),
        "eligibility_warnings": list(loaded.view.eligibility_warnings),
    }
    for field in (
        "selected_chart_key",
        "selected_chart_content_hash",
        "source_audio_key",
        "source_audio_content_hash",
    ):
        value = source.get(field)
        if value is not None:
            row[field] = value
    return row


def _snapshot_counts(items: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = {field: 0 for field in _SNAPSHOT_COUNTS}
    for item in items:
        disposition = item.get("execution_disposition")
        if disposition in {"inferred", "resumed"}:
            counts["success_count"] += 1
        elif disposition == "failed":
            counts["failed_count"] += 1
        elif disposition == "skipped":
            counts["skipped_count"] += 1
        elif disposition == "quarantined":
            counts["quarantined_count"] += 1
    return counts


def _project_runtime(
    items: Iterable[Mapping[str, object]],
    *,
    eligible_audio_durations: Iterable[float | None],
) -> dict[str, object]:
    """Project full-corpus wall time from measured inference rows.

    Resumed rows are eligible for the measured aggregate only when their
    persisted snapshot still carries both timing values.  A resume hit with
    no retained timing therefore contributes neither a zero elapsed time nor a
    zero duration.  The full eligible-duration total is available only when
    every eligible source duration was measured successfully.
    """

    def finite_positive(value: object, *, allow_zero: bool = False) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
            return None
        return numeric

    rows = tuple(items)
    durations = tuple(eligible_audio_durations)
    available_durations = tuple(
        duration
        for duration in (finite_positive(value) for value in durations)
        if duration is not None
    )
    duration_count = len(durations)
    duration_coverage_count = len(available_durations)
    full_duration = (
        sum(available_durations)
        if duration_count > 0 and duration_coverage_count == duration_count
        else None
    )

    measured_wall = 0.0
    measured_duration = 0.0
    measured_rows = 0
    for row in rows:
        if row.get("execution_disposition") not in {"inferred", "resumed"}:
            continue
        wall_time = finite_positive(row.get("wall_time_sec"), allow_zero=True)
        audio_duration = finite_positive(row.get("source_duration_sec"))
        if wall_time is None or audio_duration is None:
            continue
        measured_wall += wall_time
        measured_duration += audio_duration
        measured_rows += 1

    aggregate_rtf = (
        measured_wall / measured_duration if measured_rows and measured_duration else None
    )
    projected = (
        aggregate_rtf * full_duration if aggregate_rtf is not None and full_duration else None
    )
    return {
        "measured_wall_time_sec": measured_wall if measured_rows else None,
        "measured_audio_duration_sec": measured_duration if measured_rows else None,
        "aggregate_rtf": aggregate_rtf,
        "eligible_audio_duration_sec": full_duration,
        "eligible_audio_duration_coverage_count": duration_coverage_count,
        "eligible_audio_duration_total_count": duration_count,
        "projected_full_wall_time_sec": projected,
    }


def _write_snapshot_checkpoint(
    run_path: Path,
    header: Mapping[str, object],
    items: Iterable[Mapping[str, object]],
    *,
    overall_status: OafRunStatus | None = None,
    completed_at: str | None = None,
    close_error: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
) -> None:
    snapshot: dict[str, object] = dict(header)
    snapshot["items"] = [
        dict(item) for item in sorted(items, key=lambda row: int(row["simfile_id"]))
    ]
    snapshot.update(_snapshot_counts(snapshot["items"]))  # type: ignore[arg-type]
    if overall_status is not None:
        snapshot["overall_status"] = overall_status
    if completed_at is not None:
        snapshot["completed_at"] = completed_at
    if close_error is not None:
        snapshot["close_error"] = dict(close_error)
    if runtime:
        snapshot.update(runtime)
    _write_oaf_corpus_run(run_path, snapshot)


def _source_failure_code(error: BaseException) -> str:
    detail = _bounded_error(error).lower()
    if isinstance(error, (OSError, RuntimeError)) or any(
        marker in detail for marker in ("decode", "unreadable", "invalid wav", "header")
    ):
        return "source_audio_decode_failed"
    return "source_audio_unavailable"


def _set_failed(item: dict[str, object], code: str, error: BaseException | str) -> None:
    item["execution_disposition"] = "failed"
    item["runner_failure_code"] = code
    item["failure_detail"] = (
        _bounded_error(error)
        if isinstance(error, BaseException)
        else _bounded_error(ValueError(error))
    )


def _set_skipped(item: dict[str, object]) -> None:
    item["execution_disposition"] = "skipped"
    item["runner_failure_code"] = "explicitly_skipped"


def _set_quarantined(item: dict[str, object]) -> None:
    item["execution_disposition"] = "quarantined"
    item["runner_failure_code"] = "reference_quarantined"


def _remove_temporary_input(path: Path, input_root: Path) -> None:
    try:
        path.resolve().relative_to(input_root.resolve())
    except (OSError, RuntimeError, ValueError):
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _read_existing_prediction(path: Path) -> tuple[bool, bytes | None]:
    try:
        return True, read_regular_file_no_follow(path)
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None


def _cohort_item_from_run_row(
    identity: CohortIdentity,
    row: Mapping[str, object],
    mapping: ReferenceMappingResult | None,
    *,
    output_dir: Path,
) -> CohortItem:
    """Adapt one persisted execution row to the HPA-325 item contract."""
    if not isinstance(identity, CohortIdentity):
        raise TypeError("identity must be CohortIdentity")
    if not isinstance(row, Mapping):
        raise TypeError("run row must be a mapping")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    raw_simfile_id = row.get("simfile_id")
    if isinstance(raw_simfile_id, bool) or not isinstance(raw_simfile_id, (int, str)):
        raise ValueError("run row simfile_id is invalid")
    simfile_id = str(raw_simfile_id)
    if not simfile_id:
        raise ValueError("run row simfile_id is invalid")
    warnings_value = row.get("eligibility_warnings", ())
    if isinstance(warnings_value, (list, tuple)) and all(
        isinstance(value, str) for value in warnings_value
    ):
        warnings = tuple(warnings_value)
    else:
        warnings = ()

    disposition = row.get("execution_disposition")
    if disposition in {"inferred", "resumed"}:
        prediction_path_value = row.get("prediction_path")
        if not isinstance(prediction_path_value, str) or not prediction_path_value:
            return cohort_item_without_prediction(
                identity,
                simfile_id,
                mapping,
                status="failed",
                failure_reason="prediction_missing",
                warnings=warnings,
            )
        prediction_artifact_path = Path(prediction_path_value)
        if not prediction_artifact_path.is_absolute():
            prediction_artifact_path = output_dir / prediction_artifact_path
        try:
            content = read_regular_file_no_follow(prediction_artifact_path)
        except FileNotFoundError:
            return cohort_item_without_prediction(
                identity,
                simfile_id,
                mapping,
                status="failed",
                failure_reason="prediction_missing",
                warnings=warnings,
            )
        except OSError:
            return cohort_item_without_prediction(
                identity,
                simfile_id,
                mapping,
                status="failed",
                failure_reason="prediction_artifact_invalid",
                warnings=warnings,
            )
        if mapping is None:
            return cohort_item_without_prediction(
                identity,
                simfile_id,
                mapping,
                status="failed",
                failure_reason="prediction_artifact_invalid",
                warnings=warnings,
            )
        try:
            prediction = read_prediction_artifact(content)
            if not prediction_artifact_matches_run_row(
                prediction,
                row,
                expected_input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            ):
                return cohort_item_without_prediction(
                    identity,
                    simfile_id,
                    mapping,
                    status="failed",
                    failure_reason="prediction_artifact_invalid",
                    warnings=warnings,
                )
            return cohort_item_from_validated_prediction_artifact(
                identity,
                simfile_id,
                mapping,
                prediction,
                warnings=warnings,
            )
        except (PredictionArtifactError, StrictJsonError, TypeError, ValueError):
            return cohort_item_without_prediction(
                identity,
                simfile_id,
                mapping,
                status="failed",
                failure_reason="prediction_artifact_invalid",
                warnings=warnings,
            )

    if disposition == "quarantined":
        return cohort_item_without_prediction(
            identity,
            simfile_id,
            mapping,
            status="quarantined",
            failure_reason="reference_quarantined",
            warnings=warnings,
        )
    if disposition == "skipped":
        runner_code = row.get("runner_failure_code", "explicitly_skipped")
        failure_reason = RUNNER_FAILURE_TO_COHORT_REASON.get(
            runner_code if isinstance(runner_code, str) else "explicitly_skipped",
            "explicitly_skipped",
        )
        return cohort_item_without_prediction(
            identity,
            simfile_id,
            mapping,
            status="skipped",
            failure_reason=failure_reason,
            warnings=warnings,
        )

    runner_code = row.get("runner_failure_code")
    if not isinstance(runner_code, str) or not runner_code:
        runner_code = "backend_unavailable"
    failure_reason = RUNNER_FAILURE_TO_COHORT_REASON.get(runner_code, "backend_unavailable")
    return cohort_item_without_prediction(
        identity,
        simfile_id,
        mapping,
        status="failed",
        failure_reason=failure_reason,
        warnings=warnings,
    )


def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    """Reconstruct the HPA-325 cohort from one persisted execution snapshot."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        raise ValueError("run snapshot items must be a list")
    identity = _cohort_identity_from_snapshot(snapshot)
    cohort_items = tuple(
        _cohort_item_from_run_row(
            identity,
            row,
            mappings.get(int(row["simfile_id"])),
            output_dir=output_dir,
        )
        for row in items
        if isinstance(row, Mapping)
    )
    return identity, cohort_items


def _prediction_relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return path.as_posix()


def _build_run_header(
    *,
    request: OafCorpusRunRequest,
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    descriptor: BackendDescriptor,
    config: OafModelConfig,
    model_lock_sha256: str,
    inference_config: Mapping[str, str],
    inference_config_sha: str,
    run_id: str,
    started_at: str,
) -> dict[str, object]:
    return {
        "schema": OAF_CORPUS_RUN_SCHEMA,
        "run_id": run_id,
        "reference_manifest_sha256": reference_manifest.manifest_sha256,
        "reference_manifest_version": reference_manifest.corpus_version,
        "reference_timing_manifest_sha256": timing_manifest.manifest_sha256,
        "reference_timing_version": timing_manifest.corpus_version,
        "backend_descriptor_sha256": descriptor.sha256,
        "backend_descriptor": dict(descriptor.payload),
        "model_id": config.model_id,
        "model_lock_sha256": model_lock_sha256,
        "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
        "adapter_revision": OAF_ADAPTER_REVISION,
        "inference_config": dict(inference_config),
        "inference_config_sha256": inference_config_sha,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
        "crux_commit": request.crux_commit,
        "include_simfile_ids": list(request.include_simfile_ids),
        "exclude_simfile_ids": list(request.exclude_simfile_ids),
        "request_timeout_seconds": OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
        "worker_close_timeout_seconds": OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
        "started_at": started_at,
    }


def _fatal_outcome() -> OafCorpusRunOutcome:
    return OafCorpusRunOutcome(
        overall_status="failed",
        exit_code=2,
        run_id=None,
        run_path=None,
        reports_path=None,
        success_count=0,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
        aggregate_rtf=None,
        projected_full_wall_time_sec=None,
    )


def _cohort_identity_from_snapshot(snapshot: Mapping[str, object]) -> CohortIdentity:
    descriptor = snapshot.get("backend_descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("run snapshot backend descriptor is missing")
    backend_id = descriptor.get("backend_id")
    descriptor_model_id = descriptor.get("model_id")
    model_id = snapshot.get("model_id", descriptor_model_id)
    inference_config = snapshot.get("inference_config")
    prediction_map_version = OAF_PREDICTION_MAP_ID
    if isinstance(inference_config, Mapping):
        configured_map = inference_config.get("prediction_map_version")
        if isinstance(configured_map, str) and configured_map:
            prediction_map_version = configured_map
    values = {
        "cohort_id": snapshot.get("run_id"),
        "reference_manifest_sha256": snapshot.get("reference_manifest_sha256"),
        "reference_timing_version": snapshot.get("reference_timing_version"),
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "backend_id": backend_id,
        "model_id": model_id,
        "model_lock_sha256": snapshot.get("model_lock_sha256"),
        "backend_descriptor_sha256": snapshot.get("backend_descriptor_sha256"),
        "prediction_map_version": prediction_map_version,
        "input_view_id": snapshot.get("input_view_id"),
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("run snapshot does not contain a complete cohort identity")
    return CohortIdentity(**values)  # type: ignore[arg-type]


def _item_counts(items: Iterable[CohortItem]) -> dict[str, int]:
    counts = {field: 0 for field in _SNAPSHOT_COUNTS}
    for item in items:
        if item.status == "success":
            counts["success_count"] += 1
        elif item.status == "failed":
            counts["failed_count"] += 1
        elif item.status == "skipped":
            counts["skipped_count"] += 1
        elif item.status == "quarantined":
            counts["quarantined_count"] += 1
    return counts


def _finalize_scoring_and_outcome(
    snapshot: Mapping[str, object],
    *,
    run_id: str,
    run_path: Path,
    reports_path: Path,
    aggregate_rtf: float | None,
    projected_full_wall_time_sec: float | None,
    mappings: Mapping[int, ReferenceMappingResult | None] | None = None,
    output_dir: Path | None = None,
) -> OafCorpusRunOutcome:
    """Assemble HPA-325 items, score them, and publish the canonical reports."""
    reports_path.mkdir(parents=True, exist_ok=True)
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        return _fatal_outcome()
    resolved_output_dir = output_dir or run_path.parents[2]
    identity, cohort_items = build_oaf_cohort_from_snapshot(
        snapshot,
        mappings=mappings or {},
        output_dir=resolved_output_dir,
    )
    score_result = score_cohort(identity, cohort_items, diagnostics_for=())
    write_cohort_reports(score_result, reports_path)
    counts = (
        {
            "success_count": score_result.population.success_count,
            "failed_count": score_result.population.failed_count,
            "skipped_count": score_result.population.skipped_count,
            "quarantined_count": score_result.population.quarantined_count,
        }
        if hasattr(score_result, "population")
        else _item_counts(cohort_items)
    )
    status = snapshot.get("overall_status")
    if status not in {"complete", "partial", "failed"}:
        status = "partial"
    if status == "complete" and counts["failed_count"]:
        status = "partial"
    return OafCorpusRunOutcome(
        overall_status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "complete" else 2 if status == "failed" else 1,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        success_count=counts["success_count"],
        failed_count=counts["failed_count"],
        skipped_count=counts["skipped_count"],
        quarantined_count=counts["quarantined_count"],
        aggregate_rtf=aggregate_rtf,
        projected_full_wall_time_sec=projected_full_wall_time_sec,
    )


def run_oaf_corpus(
    request: OafCorpusRunRequest,
    *,
    backend_factory: Callable[..., OafBackend] = create_backend,
    perf_counter: Callable[[], float] = time.perf_counter,
    clock: Callable[[], datetime] = _utc_now,
) -> OafCorpusRunOutcome:
    """Run one OaF corpus sequentially through a single persistent backend."""
    if not isinstance(request, OafCorpusRunRequest):
        raise TypeError("request must be OafCorpusRunRequest")

    # Keep every operation that can reject scope or lineage above backend
    # construction.  In particular, a bad include/exclude set must never touch
    # the injected factory.
    try:
        reference_manifest = load_reference_set_manifest(request.reference_manifest_path)
        timing_manifest = load_reference_timing_manifest(request.timing_manifest_path)
        loaded_ids = {row.view.simfile_id for row in reference_manifest.rows}
        include_ids, exclude_ids = _validate_scope(
            request.include_simfile_ids,
            request.exclude_simfile_ids,
            loaded_ids,
        )
        mappings = preflight_reference_mappings(
            reference_manifest,
            timing_manifest,
            timing_output_root=request.timing_manifest_path.parent.parent,
        )
        config = load_model_config(_model_lock_path())
        model_lock_sha = compute_model_lock_sha256(_model_lock_path())
        descriptor = _expected_oaf_descriptor(config)
        inference_payload = build_inference_config(config, descriptor, model_lock_sha)
        config_sha = inference_config_sha256(inference_payload)
        run_id = build_run_id(
            reference_manifest.manifest_sha256,
            timing_manifest.manifest_sha256,
            descriptor.sha256,
            model_lock_sha,
            config.checkpoint.archive_sha256,
            config_sha,
            include_ids,
            exclude_ids,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    run_dir = request.output_dir / "runs" / run_id
    run_path = run_dir / "run.json"
    reports_path = run_dir / "reports"
    input_root = run_dir / "inputs"
    try:
        started_at = _timestamp(clock())
    except (OSError, TypeError, ValueError, RuntimeError):
        return _fatal_outcome()
    header = _build_run_header(
        request=request,
        reference_manifest=reference_manifest,
        timing_manifest=timing_manifest,
        descriptor=descriptor,
        config=config,
        model_lock_sha256=model_lock_sha,
        inference_config=inference_payload,
        inference_config_sha=config_sha,
        run_id=run_id,
        started_at=started_at,
    )

    resume_rows: dict[int, Mapping[str, object]] = {}
    if request.resume and run_path.exists():
        try:
            prior_snapshot = parse_oaf_corpus_run(
                read_regular_file_no_follow(run_path), expected_run_id=run_id
            )
            prior_items = prior_snapshot.get("items", [])
            if isinstance(prior_items, list):
                resume_rows = {
                    int(row["simfile_id"]): row
                    for row in prior_items
                    if isinstance(row, Mapping) and isinstance(row.get("simfile_id"), int)
                }
        except (OSError, StrictJsonError, ValueError):
            return _fatal_outcome()

    if run_path.exists() and not request.resume:
        return _fatal_outcome()
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        input_root.mkdir(parents=True, exist_ok=True)
        states: list[_OafExecutionState] = []
        include_set = set(include_ids)
        for loaded in sorted(reference_manifest.rows, key=lambda row: row.view.simfile_id):
            simfile_id = loaded.view.simfile_id
            if include_set and simfile_id not in include_set:
                continue
            row = _snapshot_item(loaded)
            state = _OafExecutionState(loaded, mappings.get(simfile_id), row)
            states.append(state)
            if simfile_id in set(exclude_ids):
                _set_skipped(row)
            elif loaded.view.eligibility_status != "eligible":
                _set_quarantined(row)

        _write_snapshot_checkpoint(run_path, header, (state.snapshot for state in states))
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    try:
        cache_index = CacheIndexStore.load(request.cache_dir)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _fatal_outcome()
    eligible_states = [
        state
        for state in states
        if state.loaded.view.eligibility_status == "eligible"
        and state.snapshot.get("execution_disposition") is None
    ]
    for state in eligible_states:
        source_row = state.loaded.source_row
        source_audio_key = source_row.get("source_audio_key")
        source_audio_content_hash = source_row.get("source_audio_content_hash")
        source_endpoint_sha256 = source_row.get("source_endpoint_sha256")
        source_bucket = source_row.get("source_bucket")
        source_audio_kwargs = {
            "source_audio_key": source_audio_key if isinstance(source_audio_key, str) else None,
            "source_audio_content_hash": (
                source_audio_content_hash if isinstance(source_audio_content_hash, str) else None
            ),
            "source_endpoint_sha256": (
                source_endpoint_sha256 if isinstance(source_endpoint_sha256, str) else None
            ),
            "source_bucket": source_bucket if isinstance(source_bucket, str) else None,
        }
        try:
            source = resolve_source_audio(
                source_row,
                request.cache_dir,
                cache_index,
                **source_audio_kwargs,
                load_body=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _set_failed(state.snapshot, _source_failure_code(error), error)
            continue
        state.source = source
        state.source_audio_kwargs = source_audio_kwargs
        state.snapshot["source_audio_id"] = source.source_audio_id
        state.snapshot["source_audio_sha256"] = source.source_audio_sha256
        state.snapshot["source_duration_sec"] = source.duration_sec

    try:
        _write_snapshot_checkpoint(run_path, header, (state.snapshot for state in states))
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    backend: OafBackend | None = None
    backend_descriptor = descriptor
    stop_after_poison = False
    fatal_backend_error = False
    fatal_run_error = False
    interrupted_error: BaseException | None = None
    close_error: dict[str, str] | None = None
    try:
        for state in eligible_states:
            item = state.snapshot
            if item.get("execution_disposition") == "failed" or state.source is None:
                continue
            assert state.source is not None
            assert state.source_audio_kwargs is not None
            canonical_path = input_root / str(state.loaded.view.simfile_id) / "full-mix.wav"
            prediction_target = prediction_path(
                request.output_dir,
                simfile_id=state.loaded.view.simfile_id,
                source_audio_sha256=state.source.source_audio_sha256,
                backend_descriptor_sha256=backend_descriptor.sha256,
                inference_config_sha256=config_sha,
            )
            # Re-pin the source body immediately before materialization so the
            # bytes inferred by librosa are guaranteed to match the HPA-323 digest
            # recorded in the run row. The bulk preflight only verified the cache
            # body and retained its path; a cache replacement between that
            # validation and inference must be rejected rather than silently
            # inferred against the original digest.
            try:
                state.source = resolve_source_audio(
                    state.loaded.source_row,
                    request.cache_dir,
                    cache_index,
                    **state.source_audio_kwargs,
                    load_body=True,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                _set_failed(item, _source_failure_code(error), error)
                _remove_temporary_input(canonical_path, input_root)
                try:
                    _write_snapshot_checkpoint(
                        run_path, header, (state.snapshot for state in states)
                    )
                except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                    fatal_run_error = True
                    stop_after_poison = True
                    break
                continue
            try:
                audio = _materialize_oaf_full_mix(
                    state.source,
                    canonical_path,
                    input_root=input_root,
                    config=config,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                _set_failed(item, "canonical_input_failed", error)
                _remove_temporary_input(canonical_path, input_root)
                try:
                    _write_snapshot_checkpoint(
                        run_path, header, (state.snapshot for state in states)
                    )
                except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                    fatal_run_error = True
                    stop_after_poison = True
                    break
                continue
            # The canonical input now exists on disk; release the raw corpus body
            # so execution state does not retain audio bytes for the whole run.
            state.source = replace(state.source, content=None)
            try:
                exists, content = _read_existing_prediction(prediction_target)
                item["input_view_id"] = audio.input_view_id
                item["input_audio_sha256"] = audio.input_audio_sha256
                if exists and not request.resume:
                    _set_failed(
                        item, "prediction_output_conflict", "prediction artifact already exists"
                    )
                    continue
                if exists and request.resume:
                    if content is None:
                        _set_failed(
                            item, "prediction_artifact_invalid", "prediction artifact is unreadable"
                        )
                        continue
                    try:
                        artifact = read_prediction_artifact(content)
                    except (PredictionArtifactError, StrictJsonError, TypeError, ValueError):
                        _set_failed(
                            item, "prediction_artifact_invalid", "prediction artifact is invalid"
                        )
                        continue
                    prior_row = resume_rows.get(state.loaded.view.simfile_id) or {}
                    has_persisted_artifact_evidence = any(
                        field in prior_row
                        for field in (
                            "prediction_path",
                            "prediction_artifact_sha256",
                            "input_audio_sha256",
                        )
                    )
                    artifact_matches_run_row = prediction_artifact_matches_run_row(
                        artifact,
                        prior_row,
                        expected_input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
                    )
                    if has_persisted_artifact_evidence and not artifact_matches_run_row:
                        _set_failed(
                            item,
                            "prediction_artifact_invalid",
                            "prediction artifact does not match persisted run evidence",
                        )
                        continue
                    if not prediction_artifact_matches_audio(
                        artifact,
                        source_audio_id=state.source.source_audio_id,
                        source_audio_sha256=state.source.source_audio_sha256,
                        audio=audio,
                        descriptor=backend_descriptor,
                        prediction_map_version=OAF_PREDICTION_MAP_ID,
                    ):
                        _set_failed(
                            item,
                            "prediction_artifact_invalid",
                            "prediction artifact identity mismatch",
                        )
                        continue
                    for timing_field in ("wall_time_sec", "rtf"):
                        timing_value = prior_row.get(timing_field)
                        if (
                            isinstance(timing_value, (int, float, Decimal))
                            and not isinstance(timing_value, bool)
                            and math.isfinite(float(timing_value))
                            and timing_value >= 0
                        ):
                            item[timing_field] = timing_value
                    item["execution_disposition"] = "resumed"
                    item["prediction_path"] = _prediction_relative_path(
                        prediction_target, request.output_dir
                    )
                    item["prediction_artifact_sha256"] = artifact.artifact_sha256
                    continue

                if backend is None:
                    try:
                        backend = backend_factory(
                            input_root=input_root,
                            timeout_seconds=OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
                            close_timeout_seconds=OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
                        )
                        backend_descriptor = backend.descriptor()
                        if not isinstance(backend_descriptor, BackendDescriptor):
                            raise OafBackendError(
                                "backend descriptor is invalid", code="descriptor_invalid"
                            )
                        if backend_descriptor.sha256 != descriptor.sha256 or dict(
                            backend_descriptor.payload
                        ) != dict(descriptor.payload):
                            raise OafBackendError(
                                "backend descriptor identity changed", code="descriptor_invalid"
                            )
                    except OafBackendError:
                        raise
                    except (OSError, RuntimeError, TypeError, ValueError) as error:
                        raise OafBackendError(
                            _bounded_error(error), code="worker_start_failed"
                        ) from error

                started = perf_counter()
                try:
                    native = backend.transcribe(audio)
                finally:
                    elapsed = max(0.0, perf_counter() - started)
                if not isinstance(native, NativePrediction):
                    raise OafBackendError(
                        "native prediction is invalid", code="native_event_invalid"
                    )
                item["wall_time_sec"] = elapsed
                item["rtf"] = (
                    elapsed / state.source.duration_sec if state.source.duration_sec > 0 else None
                )
                try:
                    mapped, _ = map_oaf_prediction(native)
                except Exception as error:  # pylint: disable=broad-exception-caught
                    raise OafBackendError(
                        "worker response mapping failed", code="worker_response_invalid"
                    ) from error
                published: PublishedArtifact = publish_prediction_artifact(
                    prediction_target, mapped
                )
                item["execution_disposition"] = "inferred"
                item["prediction_path"] = _prediction_relative_path(
                    prediction_target, request.output_dir
                )
                item["prediction_artifact_sha256"] = published.sha256
            except OafBackendError as error:
                runner_code, disposition = classify_oaf_backend_error(error.code)
                if disposition == "fatal_preflight":
                    fatal_backend_error = True
                    stop_after_poison = True
                    _set_failed(item, "backend_unavailable", error)
                else:
                    _set_failed(item, runner_code or "worker_protocol_failed", error)
                    stop_after_poison = disposition == "poison"
            except (ArtifactPublicationError, PredictionArtifactError) as error:
                _set_failed(item, "prediction_publish_failed", error)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                _set_failed(item, "prediction_artifact_invalid", error)
            except Exception:  # pylint: disable=broad-exception-caught
                fatal_run_error = True
                stop_after_poison = True
            finally:
                _remove_temporary_input(canonical_path, input_root)
                try:
                    _write_snapshot_checkpoint(
                        run_path, header, (state.snapshot for state in states)
                    )
                except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                    fatal_run_error = True
                    stop_after_poison = True

            if stop_after_poison:
                break
    except BaseException as error:  # pylint: disable=broad-exception-caught
        interrupted_error = error
        fatal_run_error = True
    finally:
        if backend is not None:
            try:
                backend.close()
            except (OafBackendError, OSError, RuntimeError, TypeError, ValueError) as error:
                close_error = _bounded_close_error(error)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                close_error = _bounded_close_error(error)
                if interrupted_error is None:
                    interrupted_error = error

    if stop_after_poison:
        for state in eligible_states:
            if state.snapshot.get("execution_disposition") is None:
                _set_failed(
                    state.snapshot,
                    "worker_protocol_failed",
                    "inference was not attempted after a poison failure",
                )

    counts = _snapshot_counts(state.snapshot for state in states)
    has_pending = any("execution_disposition" not in state.snapshot for state in states)
    has_item_problem = counts["failed_count"] > 0
    has_close_problem = close_error is not None
    overall_status: OafRunStatus
    if fatal_backend_error or fatal_run_error:
        overall_status = "failed"
    elif not has_pending and not has_item_problem and not has_close_problem:
        overall_status = "complete"
    else:
        overall_status = "partial"
    completed_at: str | None = None
    if overall_status == "complete":
        try:
            completed_at = _timestamp(clock())
        except (OSError, TypeError, ValueError, RuntimeError):
            overall_status = "failed"
    runtime = _project_runtime(
        (state.snapshot for state in states),
        eligible_audio_durations=tuple(
            state.source.duration_sec if state.source is not None else None
            for state in eligible_states
        ),
    )
    try:
        _write_snapshot_checkpoint(
            run_path,
            header,
            (state.snapshot for state in states),
            overall_status=overall_status,
            completed_at=completed_at,
            close_error=close_error,
            runtime=runtime,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    try:
        final_snapshot = parse_oaf_corpus_run(
            read_regular_file_no_follow(run_path), expected_run_id=run_id
        )
    except (OSError, StrictJsonError, ValueError):
        return _fatal_outcome()
    try:
        outcome = _finalize_scoring_and_outcome(
            final_snapshot,
            run_id=run_id,
            run_path=run_path,
            reports_path=reports_path,
            aggregate_rtf=(
                runtime["aggregate_rtf"]
                if isinstance(runtime["aggregate_rtf"], (float, int))
                else None
            ),
            projected_full_wall_time_sec=(
                runtime["projected_full_wall_time_sec"]
                if isinstance(runtime["projected_full_wall_time_sec"], (float, int))
                else None
            ),
            mappings=mappings,
            output_dir=request.output_dir,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        if interrupted_error is not None:
            raise interrupted_error from error
        return _fatal_outcome()
    if interrupted_error is not None:
        raise interrupted_error
    return outcome


# pylint: enable=too-many-arguments,too-many-return-statements,too-many-locals
# pylint: enable=too-many-branches,too-many-statements


__all__ = [
    "OAF_ADAPTER_REVISION",
    "OAF_BACKEND_ERROR_POLICY",
    "OAF_CANONICALIZATION_REVISION",
    "OAF_CORPUS_REQUEST_TIMEOUT_SECONDS",
    "OAF_CORPUS_RUN_SCHEMA",
    "OAF_FULL_MIX_INPUT_VIEW_ID",
    "OAF_INFERENCE_CONFIG_SCHEMA",
    "OAF_PREDICTION_MAP_ID",
    "OAF_WORKER_CLOSE_TIMEOUT_SECONDS",
    "RUNNER_FAILURE_TO_COHORT_REASON",
    "OafCorpusRunOutcome",
    "OafCorpusRunRequest",
    "build_inference_config",
    "build_run_id",
    "classify_oaf_backend_error",
    "compute_model_lock_sha256",
    "inference_config_sha256",
    "parse_oaf_corpus_run",
    "render_oaf_corpus_run",
    "run_oaf_corpus",
    "write_oaf_corpus_run",
]
