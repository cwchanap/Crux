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

import librosa
import soundfile

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
from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.corpus_cache import (
    CacheIndexStore,
    cache_entry_matches_remote,
    resolve_verified_cache_body,
    validate_cached_body,
)
from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.input_view import load_materialized_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    PredictionArtifactError,
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.r2_corpus_models import RemoteObject, parse_manifest_timestamp
from src.benchmark.reference_set import ReferenceMappingResult, map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    load_reference_set_manifest,
    read_native_reference_events,
)
from src.benchmark.reference_timing import inspect_source_audio
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
)
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID

OAF_CORPUS_RUN_SCHEMA = "crux.oaf-corpus-run/v1"
OAF_INFERENCE_CONFIG_SCHEMA = "crux.oaf-inference-config/v1"
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
OAF_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0


OafRunStatus: TypeAlias = Literal["complete", "partial", "failed"]
OafRunExitCode: TypeAlias = Literal[0, 1, 2]


@dataclass(frozen=True)
class ResolvedSourceAudio:
    """One locally verified source-audio body and its header duration."""

    path: Path
    source_audio_id: str
    source_audio_sha256: str
    duration_sec: float


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


def _remote_from_source_mapping(
    source: Mapping[str, object],
    *,
    source_audio_key: str,
) -> RemoteObject:
    raw_objects = source.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("source manifest does not contain an object inventory")
    for raw_object in raw_objects:
        if not isinstance(raw_object, Mapping) or raw_object.get("key") != source_audio_key:
            continue
        try:
            return RemoteObject(
                key=source_audio_key,
                size=raw_object["size"],  # type: ignore[arg-type]
                etag=raw_object["etag"],  # type: ignore[arg-type]
                etag_is_weak=raw_object["etag_is_weak"],  # type: ignore[arg-type]
                last_modified=parse_manifest_timestamp(raw_object["last_modified"]),
                content_type=raw_object["content_type"],  # type: ignore[arg-type]
                cache_status=raw_object["cache_status"],  # type: ignore[arg-type]
                sha256=raw_object["sha256"],  # type: ignore[arg-type]
                cache_path=raw_object["cache_path"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("source manifest contains an invalid audio object") from None
    raise ValueError("source audio key is absent from the source inventory")


def _source_audio_parts(
    source: RemoteObject | ManifestRowView | Mapping[str, object],
    *,
    source_audio_key: str | None,
    source_audio_content_hash: str | None,
    source_endpoint_sha256: str | None,
    source_bucket: str | None,
) -> tuple[RemoteObject, str, str, str]:
    if isinstance(source, ManifestRowView):
        endpoint = source_endpoint_sha256 or source.source_endpoint_sha256
        bucket = source_bucket or source.source_bucket
        key = source_audio_key
        if key is None:
            raise ValueError("source_audio_key is required")
        remote = next((item for item in source.inventory.objects if item.key == key), None)
        if remote is None:
            raise ValueError("source audio key is absent from the source inventory")
        expected = source_audio_content_hash
    elif isinstance(source, RemoteObject):
        endpoint = source_endpoint_sha256
        bucket = source_bucket
        key = source_audio_key or source.key
        remote = source
        expected = source_audio_content_hash
    elif isinstance(source, Mapping):
        key = source_audio_key or source.get("source_audio_key")
        endpoint = source_endpoint_sha256 or source.get("source_endpoint_sha256")
        bucket = source_bucket or source.get("source_bucket")
        expected = source_audio_content_hash
        if not isinstance(key, str):
            raise ValueError("source_audio_key is required")
        remote = _remote_from_source_mapping(source, source_audio_key=key)
    else:
        raise TypeError("source must be a RemoteObject, ManifestRowView, or mapping")

    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("source_endpoint_sha256 is required")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("source_bucket is required")
    if not isinstance(key, str) or not key:
        raise ValueError("source_audio_key is required")
    if not isinstance(expected, str):
        raise ValueError("source_audio_content_hash is required")
    require_sha256(expected, "source_audio_content_hash")
    return remote, endpoint, bucket, expected


# pylint: disable=too-many-arguments,too-many-locals
def _resolve_source_audio(
    source: RemoteObject | ManifestRowView | Mapping[str, object],
    cache_dir: Path,
    cache_index: CacheIndexStore | None = None,
    *,
    index: CacheIndexStore | None = None,
    source_audio_key: str | None = None,
    source_audio_content_hash: str | None = None,
    source_endpoint_sha256: str | None = None,
    source_bucket: str | None = None,
) -> ResolvedSourceAudio:
    """Resolve one source body from the verified local HPA-321 cache only."""
    if not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a Path")
    if cache_index is not None and index is not None and cache_index is not index:
        raise ValueError("cache index arguments disagree")
    cache_index = cache_index or index or CacheIndexStore.load(cache_dir)
    remote, endpoint, bucket, expected = _source_audio_parts(
        source,
        source_audio_key=source_audio_key,
        source_audio_content_hash=source_audio_content_hash,
        source_endpoint_sha256=source_endpoint_sha256,
        source_bucket=source_bucket,
    )

    verified_remote = remote
    if remote.cache_status != "verified":
        entry = cache_index.get(endpoint, bucket, remote.key)
        if not cache_entry_matches_remote(entry, remote, endpoint=endpoint, bucket=bucket):
            raise ValueError("verified source audio unavailable")
        validation = validate_cached_body(cache_dir, entry)
        if validation.state != "verified" or entry is None:
            raise ValueError("verified source audio unavailable")
        verified_remote = replace(
            remote,
            cache_status="verified",
            sha256=entry.sha256,
            cache_path=entry.cache_path,
        )

    if verified_remote.sha256 != expected:
        raise ValueError("source audio digest does not match reference timing manifest")
    path = resolve_verified_cache_body(
        cache_dir,
        verified_remote,
        source_endpoint_sha256=endpoint,
        bucket=bucket,
        expected_sha256=expected,
    )
    duration_sec = inspect_source_audio(path).duration_sec
    return ResolvedSourceAudio(
        path=path,
        source_audio_id=remote.key,
        source_audio_sha256=expected,
        duration_sec=duration_sec,
    )


# pylint: enable=too-many-arguments,too-many-locals


def _materialize_oaf_full_mix(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
    config: OafModelConfig,
) -> CanonicalAudio:
    """Materialize one temporary, canonical OaF full-mix input."""
    if not isinstance(source_audio, ResolvedSourceAudio):
        raise TypeError("source_audio must be ResolvedSourceAudio")
    if not isinstance(output_path, Path) or not isinstance(input_root, Path):
        raise TypeError("output_path and input_root must be Paths")
    if not isinstance(config, OafModelConfig):
        raise TypeError("config must be OafModelConfig")
    root = input_root.resolve()
    destination = output_path.resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise ValueError("canonical input must be beneath input_root") from None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples, _ = librosa.load(
        source_audio.path,
        sr=44100,
        mono=True,
        res_type="soxr_hq",
    )
    soundfile.write(
        output_path,
        samples,
        44100,
        format="WAV",
        subtype="PCM_16",
    )
    return load_materialized_audio(
        path=output_path,
        source_audio_id=source_audio.source_audio_id,
        source_audio_sha256=source_audio.source_audio_sha256,
        input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
        max_input_audio_frames=config.max_input_audio_frames,
    )


# pylint: disable=too-many-branches
def _preflight_reference_mappings(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    timing_output_root: Path,
) -> dict[int, ReferenceMappingResult | None]:
    """Reconstruct eligible native references before any backend work."""
    if not isinstance(reference_manifest, LoadedReferenceSetManifest):
        raise TypeError("reference_manifest must be LoadedReferenceSetManifest")
    if not isinstance(timing_manifest, LoadedReferenceTimingManifest):
        raise TypeError("timing_manifest must be LoadedReferenceTimingManifest")
    if not isinstance(timing_output_root, Path):
        raise TypeError("timing_output_root must be a Path")
    if (
        reference_manifest.source_reference_timing_manifest_sha256
        != timing_manifest.manifest_sha256
        or reference_manifest.source_reference_timing_version != timing_manifest.corpus_version
    ):
        raise ValueError("reference and timing manifests have different lineage")

    timing_rows = {row.view.simfile_id: row for row in timing_manifest.rows}
    mappings: dict[int, ReferenceMappingResult | None] = {}
    for loaded in reference_manifest.rows:
        simfile_id = loaded.view.simfile_id
        timing_row = timing_rows.get(simfile_id)
        reasons = set(loaded.view.eligibility_reason_codes)
        if loaded.view.eligibility_status != "eligible":
            if reasons & {"upstream_reference_unavailable", "reference_event_artifact_invalid"}:
                mappings[simfile_id] = None
                continue
            if timing_row is None or timing_row.view.timing_status != "ready":
                mappings[simfile_id] = None
                continue
        if timing_row is None or timing_row.view.timing_status != "ready":
            raise ValueError("eligible reference timing row is unavailable")
        for field in (
            "selected_chart_key",
            "selected_chart_content_hash",
            "source_audio_key",
            "source_audio_content_hash",
            "reference_events_cache_path",
        ):
            if loaded.source_row.get(field) != timing_row.source_row.get(field):
                raise ValueError("eligible reference identity does not match timing row")
        try:
            events = read_native_reference_events(
                timing_row,
                timing_output_root=timing_output_root,
            )
            mappings[simfile_id] = map_reference_events(events)
        except (OSError, RuntimeError, ValueError):
            if loaded.view.eligibility_status == "eligible":
                raise ValueError("eligible reference event artifact invalid") from None
            mappings[simfile_id] = None
    return mappings


# pylint: enable=too-many-branches


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


# pylint: disable=too-many-arguments,too-many-return-statements,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
@dataclass
class _OafExecutionState:
    loaded: LoadedReferenceSetRow
    mapping: ReferenceMappingResult | None
    snapshot: dict[str, object]
    source: ResolvedSourceAudio | None = None


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


def _prediction_artifact_matches(
    artifact: PredictionArtifact,
    *,
    source: ResolvedSourceAudio,
    audio: CanonicalAudio,
    descriptor: BackendDescriptor,
) -> bool:
    prediction = artifact.prediction
    if prediction.descriptor.sha256 != descriptor.sha256:
        return False
    if dict(prediction.descriptor.payload) != dict(descriptor.payload):
        return False
    if prediction.audio.source_audio_id != source.source_audio_id:
        return False
    if prediction.audio.source_audio_sha256 != source.source_audio_sha256:
        return False
    if prediction.audio.input_view_id != OAF_FULL_MIX_INPUT_VIEW_ID:
        return False
    if prediction.audio.input_view_id != audio.input_view_id:
        return False
    if prediction.audio.input_audio_sha256 != audio.input_audio_sha256:
        return False
    if prediction.audio.source_audio_id != audio.source_audio_id:
        return False
    if prediction.audio.source_audio_sha256 != audio.source_audio_sha256:
        return False
    if prediction.descriptor.payload.get("model_id") != descriptor.payload.get("model_id"):
        return False
    return all(event.prediction_map_version == OAF_PREDICTION_MAP_ID for event in prediction.events)


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


def _finalize_scoring_and_outcome(
    snapshot: Mapping[str, object],
    *,
    run_id: str,
    run_path: Path,
    reports_path: Path,
    aggregate_rtf: float | None,
    projected_full_wall_time_sec: float | None,
) -> OafCorpusRunOutcome:
    """Return the Task 6 execution outcome and reserve the Task 7 scorer seam."""
    reports_path.mkdir(parents=True, exist_ok=True)
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        return _fatal_outcome()
    counts = _snapshot_counts(item for item in items if isinstance(item, Mapping))
    status = snapshot.get("overall_status")
    if status not in {"complete", "partial", "failed"}:
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
        mappings = _preflight_reference_mappings(
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

    if request.resume and run_path.exists():
        try:
            parse_oaf_corpus_run(read_regular_file_no_follow(run_path), expected_run_id=run_id)
        except (OSError, StrictJsonError, ValueError):
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
        try:
            source = _resolve_source_audio(
                source_row,
                request.cache_dir,
                cache_index,
                source_audio_key=source_audio_key if isinstance(source_audio_key, str) else None,
                source_audio_content_hash=(
                    source_audio_content_hash
                    if isinstance(source_audio_content_hash, str)
                    else None
                ),
                source_endpoint_sha256=(
                    source_endpoint_sha256 if isinstance(source_endpoint_sha256, str) else None
                ),
                source_bucket=source_bucket if isinstance(source_bucket, str) else None,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _set_failed(state.snapshot, _source_failure_code(error), error)
            continue
        state.source = source
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

    for state in eligible_states:
        item = state.snapshot
        if item.get("execution_disposition") == "failed" or state.source is None:
            continue
        assert state.source is not None
        canonical_path = input_root / str(state.loaded.view.simfile_id) / "full-mix.wav"
        prediction_target = prediction_path(
            request.output_dir,
            simfile_id=state.loaded.view.simfile_id,
            source_audio_sha256=state.source.source_audio_sha256,
            backend_descriptor_sha256=backend_descriptor.sha256,
            inference_config_sha256=config_sha,
        )
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
                _write_snapshot_checkpoint(run_path, header, (state.snapshot for state in states))
            except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                fatal_run_error = True
                stop_after_poison = True
                break
            continue
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
                if not _prediction_artifact_matches(
                    artifact,
                    source=state.source,
                    audio=audio,
                    descriptor=backend_descriptor,
                ):
                    _set_failed(
                        item, "prediction_artifact_invalid", "prediction artifact identity mismatch"
                    )
                    continue
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
                raise OafBackendError("native prediction is invalid", code="native_event_invalid")
            item["wall_time_sec"] = elapsed
            item["rtf"] = (
                elapsed / state.source.duration_sec if state.source.duration_sec > 0 else None
            )
            mapped, _ = map_oaf_prediction(native)
            published: PublishedArtifact = publish_prediction_artifact(prediction_target, mapped)
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
            _write_snapshot_checkpoint(run_path, header, (state.snapshot for state in states))
        except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
            fatal_run_error = True
            stop_after_poison = True
        if stop_after_poison:
            break

    close_error: dict[str, str] | None = None
    if backend is not None:
        try:
            backend.close()
        except (OafBackendError, OSError, RuntimeError, TypeError, ValueError) as error:
            close_error = _bounded_close_error(error)

    counts = _snapshot_counts(state.snapshot for state in states)
    has_pending = any("execution_disposition" not in state.snapshot for state in states)
    has_item_problem = (
        counts["failed_count"] > 0 or counts["skipped_count"] > 0 or counts["quarantined_count"] > 0
    )
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
    try:
        _write_snapshot_checkpoint(
            run_path,
            header,
            (state.snapshot for state in states),
            overall_status=overall_status,
            completed_at=completed_at,
            close_error=close_error,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    try:
        final_snapshot = parse_oaf_corpus_run(
            read_regular_file_no_follow(run_path), expected_run_id=run_id
        )
    except (OSError, StrictJsonError, ValueError):
        return _fatal_outcome()
    return _finalize_scoring_and_outcome(
        final_snapshot,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        aggregate_rtf=None,
        projected_full_wall_time_sec=None,
    )


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
    "ResolvedSourceAudio",
    "build_inference_config",
    "build_run_id",
    "classify_oaf_backend_error",
    "compute_model_lock_sha256",
    "inference_config_sha256",
    "parse_oaf_corpus_run",
    "prediction_path",
    "render_oaf_corpus_run",
    "run_oaf_corpus",
    "write_oaf_corpus_run",
]
