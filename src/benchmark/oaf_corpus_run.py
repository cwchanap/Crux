"""Pure identity and run-snapshot contracts for OaF corpus inference.

The actual corpus lifecycle is intentionally owned by later HPA-326 slices.  This
module keeps the values that identify a run, prediction, and immutable model
configuration in one dependency-light place so those slices can compose them
without creating a second serialization or durability implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypeAlias

from runtime.oaf_tf1.model import OafModelConfig
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.oaf import OAF_ADAPTER_REVISION
from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID

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
}

if not set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS:
    raise RuntimeError("runner failure mapping contains an unknown cohort reason")


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
) -> dict[str, str]:
    """Build the closed, inference-semantic identity payload."""
    if not isinstance(config, OafModelConfig):
        raise TypeError("config must be OafModelConfig")
    if not isinstance(descriptor, BackendDescriptor):
        raise TypeError("descriptor must be BackendDescriptor")
    descriptor_sha256 = _require_hash(descriptor.sha256, "backend_descriptor_sha256")
    model_lock = _require_hash(model_lock_sha256, "model_lock_sha256")
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
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
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
# pylint: disable=redefined-outer-name
def prediction_path(
    output_dir: Path,
    *,
    simfile_id: int,
    source_audio_sha256: str,
    backend_descriptor_sha256: str,
    inference_config_sha256: str,
) -> Path:
    """Return the source-keyed immutable prediction-v2 location."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if isinstance(simfile_id, bool) or not isinstance(simfile_id, int) or simfile_id <= 0:
        raise ValueError("simfile_id must be a positive integer")
    source_sha = _require_hash(source_audio_sha256, "source_audio_sha256")
    descriptor_sha = _require_hash(backend_descriptor_sha256, "backend_descriptor_sha256")
    config_sha = _require_hash(inference_config_sha256, "inference_config_sha256")
    return (
        output_dir
        / "predictions"
        / str(simfile_id)
        / source_sha
        / descriptor_sha
        / f"{config_sha}.jsonl"
    )


# pylint: enable=redefined-outer-name


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
            item["simfile_id"] for item in item_rows if "execution_disposition" not in item
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
        raise StrictJsonError("run snapshot must be an object")
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
    "prediction_path",
    "render_oaf_corpus_run",
    "write_oaf_corpus_run",
]
