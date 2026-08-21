"""Concrete sequential consumer for the immutable HPA-328 IDM pilot handoff.

The runner intentionally owns only the HPA-396 boundary.  HPA-328 stems and
OaF predictions are retained evidence: this module reads them from the two
explicit owner roots, checks their bytes and identities, and never attempts to
recreate either upstream artifact.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    IDM_BACKEND_ID,
    OAF_BACKEND_ID,
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio, NativePrediction
from src.benchmark.backends.idm import (
    IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
    IdmBackend,
    IdmBackendError,
    descriptor_for_lock,
)
from src.benchmark.cohort_scoring import (
    COHORT_FAILURE_REASONS,
    CohortIdentity,
    CohortItem,
    cohort_item_from_validated_prediction_artifact,
    cohort_item_without_prediction,
    score_cohort,
)
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.idm_model import (
    IDM_REQUEST_TIMEOUT_SECONDS,
    IdmModelLock,
    idm_inference_config,
    load_idm_model_lock,
)
from src.benchmark.input_view import parse_canonical_wav
from src.benchmark.mapping import map_idm_prediction
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    PredictionArtifactError,
    prediction_artifact_matches_audio,
    prediction_path,
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.reference_set import ReferenceMappingResult
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
)
from src.benchmark.reports import write_cohort_reports
from src.benchmark.separation_handoff import (
    LoadedSeparationPilotManifest,
    load_separation_pilot_manifest,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP_VERSION,
    IDM_PREDICTION_MAP_ID,
    TAXONOMY_VERSION,
)

IDM_PILOT_RUN_SCHEMA = "crux.idm-stem-pilot-run/v1"
IDM_STEM_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

IDM_FAILURE_TO_COHORT_REASON = {
    "worker_start_failed": "backend_unavailable",
    "worker_protocol_failed": "backend_unavailable",
    "inference_failed": "inference_failed",
    "native_event_invalid": "inference_failed",
    "upstream_stem_unavailable": "inference_failed",
    "retained_input_invalid": "prediction_artifact_invalid",
    "retained_oaf_prediction_invalid": "prediction_artifact_invalid",
    "prediction_artifact_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}

if not set(IDM_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS:
    raise RuntimeError("IDM failure mapping contains an unknown HPA-325 reason")

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_VERSION_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUCCESSFUL_HANDOFF_STATUSES = frozenset({"success", "resumed"})
_DISPOSITIONS = frozenset({"inferred", "resumed", "failed", "quarantined"})
_COUNT_FIELDS = ("success_count", "failed_count", "skipped_count", "quarantined_count")


class IdmPilotRunError(ValueError):
    """Raised for a bounded, machine-readable HPA-396 run failure."""

    def __init__(self, message: str, *, code: str = "inference_failed") -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class IdmPilotRunRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    separation_artifact_root: Path
    stem_cache_root: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    resume: bool = False
    crux_commit: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "separation_handoff_path",
            "reference_manifest_path",
            "timing_manifest_path",
            "separation_artifact_root",
            "stem_cache_root",
            "output_dir",
            "model_lock_path",
            "model_root",
            "runtime_python",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool")
        if self.crux_commit is not None and (
            not isinstance(self.crux_commit, str) or _COMMIT_RE.fullmatch(self.crux_commit) is None
        ):
            raise ValueError("crux_commit must be a lowercase 40-character commit")


@dataclass(frozen=True)
class IdmPilotRunOutcome:
    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    aggregate_rtf: float | None = None
    projected_full_wall_time_sec: float | None = None
    native_failure_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.overall_status not in {"complete", "partial", "failed"}:
            raise ValueError("overall_status is invalid")
        if self.exit_code not in {0, 1, 2}:
            raise ValueError("exit_code is invalid")
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("run_id must be a nonempty string or None")
        for field in ("run_path", "reports_path"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{field} must be a Path or None")
        for field in _COUNT_FIELDS:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in ("aggregate_rtf", "projected_full_wall_time_sec"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{field} must be a finite number or None")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class _PreparedHandoffRow:
    row: Mapping[str, object]
    audio: CanonicalAudio | None
    oaf_artifact: PredictionArtifact | None
    native_failure_code: str | None = None
    upstream_failure_code: str | None = None


def _fatal_outcome() -> IdmPilotRunOutcome:
    return IdmPilotRunOutcome(
        overall_status="failed",
        exit_code=2,
        run_id=None,
        run_path=None,
        reports_path=None,
        success_count=0,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _bounded_error(error: BaseException | str, *, limit: int = 512) -> str:
    detail = str(error).replace("\x00", " ").strip()
    if not detail:
        detail = type(error).__name__ if isinstance(error, BaseException) else "error"
    return detail[:limit]


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise IdmPilotRunError(f"{field} must be a lowercase SHA-256", code="preflight_invalid")
    try:
        return require_sha256(value, field)
    except StrictJsonError as error:
        raise IdmPilotRunError(str(error), code="preflight_invalid") from None


def _version(value: object, field: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise IdmPilotRunError(f"{field} must be a corpus version", code="preflight_invalid")
    return value


def _commit(value: object, field: str = "crux_commit") -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise IdmPilotRunError(
            f"{field} must be a lowercase 40-character commit", code="preflight_invalid"
        )
    return value


def build_run_id(
    handoff_manifest_sha256: str,
    handoff_manifest_version: str,
    reference_manifest_sha256: str,
    reference_manifest_version: str,
    timing_manifest_sha256: str,
    timing_manifest_version: str,
    backend_descriptor_sha256: str,
    model_lock_sha256: str,
    inference_config_sha256: str,
    input_view_id: str,
    crux_commit: str | None = None,
) -> str:
    """Derive the exact HPA-396 run identity from immutable inputs."""
    values = {
        "schema": IDM_PILOT_RUN_SCHEMA,
        "handoff_manifest_sha256": _hash(handoff_manifest_sha256, "handoff_manifest_sha256"),
        "handoff_manifest_version": _version(handoff_manifest_version, "handoff_manifest_version"),
        "reference_manifest_sha256": _hash(reference_manifest_sha256, "reference_manifest_sha256"),
        "reference_manifest_version": _version(
            reference_manifest_version, "reference_manifest_version"
        ),
        "timing_manifest_sha256": _hash(timing_manifest_sha256, "timing_manifest_sha256"),
        "timing_manifest_version": _version(timing_manifest_version, "timing_manifest_version"),
        "backend_descriptor_sha256": _hash(backend_descriptor_sha256, "backend_descriptor_sha256"),
        "model_lock_sha256": _hash(model_lock_sha256, "model_lock_sha256"),
        "inference_config_sha256": _hash(inference_config_sha256, "inference_config_sha256"),
        "input_view_id": input_view_id,
        "crux_commit": None if crux_commit is None else _commit(crux_commit),
    }
    if not isinstance(input_view_id, str) or not input_view_id:
        raise ValueError("input_view_id must be a nonempty string")
    return "idm-" + sha256_hex(canonical_json_bytes(values))[:16]


def compute_model_lock_sha256(path: Path) -> str:
    if not isinstance(path, Path):
        raise TypeError("model lock path must be a Path")
    return sha256_hex(read_regular_file_no_follow(path))


def build_idm_inference_config(
    model_lock_sha256: str,
    backend_descriptor_sha256: str,
    *,
    input_view_id: str = IDM_STEM_INPUT_VIEW_ID,
    timeout_seconds: int | float | None = None,
) -> dict[str, object]:
    """Build the closed IDM configuration used by the run identity."""
    if timeout_seconds is None:
        config = idm_inference_config(
            model_lock_sha256,
            backend_descriptor_sha256,
            input_view_id,
        )
    else:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be numeric")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        config = idm_inference_config(
            model_lock_sha256,
            backend_descriptor_sha256,
            input_view_id,
        )
        # The canonical IDM config stores integral seconds.  The frozen default
        # is integral; reject fractional overrides rather than silently changing
        # the persisted identity.
        if int(timeout_seconds) != timeout_seconds:
            raise ValueError("timeout_seconds must be an integer number of seconds")
        config["request_timeout_seconds"] = int(timeout_seconds)
    return config


def idm_inference_config_sha256(config: Mapping[str, object]) -> str:
    """Hash the closed IDM config while retaining timeout in run identity.

    The model module validates the frozen default timeout.  The pilot keeps
    that default, but this narrow wrapper also lets tests and offline callers
    characterize a deliberate timeout change without silently reusing a run
    identity from the default policy.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a Mapping")
    expected_keys = {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "request_timeout_seconds",
    }
    if set(config) != expected_keys:
        raise StrictJsonError("inference config must contain the exact key set")
    if config.get("schema") != "crux.idm-inference-config/v1":
        raise StrictJsonError("inference config schema is invalid")
    for field in ("backend_descriptor_sha256", "model_lock_sha256"):
        _hash(config[field], field)
    for field in ("adapter_revision", "prediction_map_version", "input_view_id"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise StrictJsonError(f"inference config {field} is invalid")
    timeout = config.get("request_timeout_seconds")
    if type(timeout) is not int or timeout <= 0:
        raise StrictJsonError("inference config request_timeout_seconds is invalid")
    return sha256_hex(canonical_json_bytes(dict(config)))


def _normalize_snapshot_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str, Decimal)):
        return value  # type: ignore[return-value]
    if isinstance(value, float):
        return quantize_six(value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJsonError("snapshot object keys must be strings")
            normalized[key] = _normalize_snapshot_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_snapshot_value(item) for item in value]
    raise StrictJsonError(f"unsupported snapshot value: {type(value).__name__}")


def _snapshot_counts(items: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = {field: 0 for field in _COUNT_FIELDS}
    for item in items:
        disposition = item.get("execution_disposition")
        if disposition in {"inferred", "resumed"}:
            counts["success_count"] += 1
        elif disposition == "failed":
            counts["failed_count"] += 1
        elif disposition == "quarantined":
            counts["quarantined_count"] += 1
    return counts


def _validate_snapshot(snapshot: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if snapshot.get("schema") != IDM_PILOT_RUN_SCHEMA:
        raise StrictJsonError(f"run snapshot schema must be {IDM_PILOT_RUN_SCHEMA}")
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise StrictJsonError("run snapshot run_id must be a nonempty string")
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        raise StrictJsonError("run snapshot items must be an array")
    item_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise StrictJsonError("run snapshot item must be an object")
        simfile_id = item.get("simfile_id")
        if type(simfile_id) is not int or simfile_id <= 0:
            raise StrictJsonError("run snapshot item simfile_id is invalid")
        if simfile_id in item_ids:
            raise StrictJsonError("run snapshot items must have unique simfile IDs")
        item_ids.add(simfile_id)
        disposition = item.get("execution_disposition")
        if disposition is not None and disposition not in _DISPOSITIONS:
            raise StrictJsonError("run snapshot execution disposition is invalid")
        if disposition == "failed" and (
            not isinstance(item.get("native_failure_code"), str)
            or not item.get("native_failure_code")
        ):
            raise StrictJsonError("failed run snapshot item must retain native_failure_code")
    items.sort(key=lambda item: int(item["simfile_id"]))
    snapshot["items"] = items
    overall_status = snapshot.get("overall_status")
    if overall_status is not None and overall_status not in {"complete", "partial", "failed"}:
        raise StrictJsonError("run snapshot overall_status is invalid")
    if all(field in snapshot for field in _COUNT_FIELDS):
        expected = {field: snapshot[field] for field in _COUNT_FIELDS}
        if any(type(value) is not int or value < 0 for value in expected.values()):
            raise StrictJsonError("run snapshot counts must be nonnegative integers")
        if expected != _snapshot_counts(items):
            raise StrictJsonError("run snapshot counts do not reconcile with items")
    if overall_status == "complete":
        if any(item.get("execution_disposition") not in _DISPOSITIONS for item in items):
            raise StrictJsonError("complete run snapshot requires item dispositions")
        if any(field not in snapshot for field in _COUNT_FIELDS):
            raise StrictJsonError("complete run snapshot requires all counts")
    if "completed_at" in snapshot and overall_status != "complete":
        raise StrictJsonError("completed timestamp is only valid for a completed run")
    return snapshot


def render_idm_pilot_run(snapshot: Mapping[str, object]) -> bytes:
    if not isinstance(snapshot, Mapping):
        raise TypeError("run snapshot must be a mapping")
    normalized = _normalize_snapshot_value(snapshot)
    if not isinstance(normalized, dict):
        raise StrictJsonError("run snapshot must be an object")
    return canonical_json_bytes(_validate_snapshot(normalized))


def parse_idm_pilot_run(
    content: bytes,
    *,
    expected_run_id: str | None = None,
) -> dict[str, JsonValue]:
    value = strict_json_loads(content, require_canonical=True)
    if not isinstance(value, dict):
        raise StrictJsonError("run snapshot must be an object")
    validated = _validate_snapshot(value)
    if canonical_json_bytes(validated) != content:
        raise StrictJsonError("run snapshot is not semantically canonical")
    if expected_run_id is not None and validated.get("run_id") != expected_run_id:
        raise StrictJsonError("run snapshot run_id does not match expected identity")
    return validated


def write_idm_pilot_run(run_path: Path, snapshot: Mapping[str, object]) -> None:
    if not isinstance(run_path, Path):
        raise TypeError("run_path must be a Path")
    atomic_replace_bytes(run_path, render_idm_pilot_run(snapshot))


def _root_resolved(root: Path, field: str, *, require_directory: bool = False) -> Path:
    if not isinstance(root, Path):
        raise TypeError(f"{field} must be a Path")
    try:
        resolved = root.resolve(strict=require_directory)
    except OSError as error:
        raise IdmPilotRunError(f"{field} is unavailable", code="preflight_invalid") from error
    if require_directory and (root.is_symlink() or not resolved.is_dir()):
        raise IdmPilotRunError(f"{field} is unavailable", code="preflight_invalid")
    return resolved


def _validate_output_roots(request: IdmPilotRunRequest) -> None:
    if request.output_dir.is_symlink():
        raise IdmPilotRunError("output_dir must not be a symlink", code="preflight_invalid")
    output = _root_resolved(request.output_dir, "output_dir")
    owner_roots = (
        _root_resolved(request.separation_artifact_root, "separation_artifact_root"),
        _root_resolved(request.stem_cache_root, "stem_cache_root"),
        _root_resolved(request.model_root, "model_root"),
    )
    for root in owner_roots:
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise IdmPilotRunError(
                "output_dir must not alias an input root", code="preflight_invalid"
            )


def _owned_path(root: Path, raw_path: object, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise IdmPilotRunError(f"{field} path is invalid", code="retained_input_invalid")
    parsed = PurePosixPath(raw_path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise IdmPilotRunError(f"{field} path is invalid", code="retained_input_invalid")
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise IdmPilotRunError(
            f"{field} owner root is unavailable", code="retained_input_invalid"
        ) from error
    if root.is_symlink() or not root_resolved.is_dir():
        raise IdmPilotRunError(f"{field} owner root is unavailable", code="retained_input_invalid")
    path = root.joinpath(*parsed.parts)
    cursor = root
    try:
        for part in parsed.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise IdmPilotRunError(
                    f"{field} path must not use symlinks", code="retained_input_invalid"
                )
        resolved = path.resolve(strict=True)
    except IdmPilotRunError:
        raise
    except OSError as error:
        raise IdmPilotRunError(
            f"{field} artifact is unavailable", code="retained_input_invalid"
        ) from error
    if not resolved.is_relative_to(root_resolved):
        raise IdmPilotRunError(f"{field} path escapes owner root", code="retained_input_invalid")
    return path


def _read_owned(root: Path, raw_path: object, field: str) -> tuple[Path, bytes]:
    path = _owned_path(root, raw_path, field)
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError) as error:
        raise IdmPilotRunError(
            f"{field} artifact is unreadable", code="retained_input_invalid"
        ) from error
    return path, content


def _read_output_prediction(path: Path, output_root: Path) -> bytes | None:
    try:
        if not isinstance(path, Path) or not isinstance(output_root, Path):
            raise TypeError("prediction path and output root must be Paths")
        relative = path.relative_to(output_root)
        if relative.is_absolute():
            raise IdmPilotRunError(
                "prediction path escapes output root", code="prediction_artifact_invalid"
            )
        cursor = output_root
        for part in relative.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise IdmPilotRunError(
                    "prediction path must not use symlinks", code="prediction_artifact_invalid"
                )
            if not cursor.exists() or not cursor.is_dir():
                return None
        if path.is_symlink():
            raise IdmPilotRunError(
                "prediction path must not use symlinks", code="prediction_artifact_invalid"
            )
        if not path.exists():
            return None
        # Resolve each component through the same no-symlink owner check.
        owned = _owned_path(output_root, relative.as_posix(), "prediction")
        return read_regular_file_no_follow(owned)
    except FileNotFoundError:
        return None
    except IdmPilotRunError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise IdmPilotRunError(
            "prediction artifact is unreadable", code="prediction_artifact_invalid"
        ) from error


def _check_sha(content: bytes, expected: object, field: str, code: str) -> None:
    try:
        expected_sha = _hash(expected, field)
    except IdmPilotRunError as error:
        raise IdmPilotRunError(str(error), code=code) from None
    if sha256_hex(content) != expected_sha:
        raise IdmPilotRunError(f"{field} bytes do not match handoff", code=code)


def _handoff_failure_code(view: Mapping[str, object]) -> tuple[str, str | None]:
    raw_code = view.get("failure_code")
    status = view.get("status")
    if isinstance(raw_code, str) and raw_code in IDM_FAILURE_TO_COHORT_REASON:
        return raw_code, raw_code
    if status == "prediction_invalid":
        return "retained_oaf_prediction_invalid", raw_code if isinstance(raw_code, str) else None
    if status == "inference_failed":
        return "inference_failed", raw_code if isinstance(raw_code, str) else None
    return "upstream_stem_unavailable", raw_code if isinstance(raw_code, str) else None


def _prepare_handoff_row(
    row: Mapping[str, object],
    *,
    separation_artifact_root: Path,
    stem_cache_root: Path,
) -> _PreparedHandoffRow:
    if not isinstance(row, Mapping):
        raise IdmPilotRunError("handoff row is invalid", code="preflight_invalid")
    view = row.get("htdemucs")
    if not isinstance(view, Mapping):
        raise IdmPilotRunError("handoff HTDemucs view is invalid", code="preflight_invalid")
    status = view.get("status")
    if status not in _SUCCESSFUL_HANDOFF_STATUSES:
        code, upstream = _handoff_failure_code(view)
        return _PreparedHandoffRow(
            row=row,
            audio=None,
            oaf_artifact=None,
            native_failure_code=code,
            upstream_failure_code=upstream,
        )

    try:
        source_audio_id = row["source_audio_id"]
        source_audio_sha256 = _hash(row["source_audio_sha256"], "source_audio_sha256")
        if not isinstance(source_audio_id, str) or not source_audio_id:
            raise ValueError("source_audio_id is invalid")
        input_evidence = view.get("input")
        stem_evidence = view.get("stem")
        prediction_evidence = view.get("prediction")
        if not all(
            isinstance(value, Mapping)
            for value in (input_evidence, stem_evidence, prediction_evidence)
        ):
            raise ValueError("successful handoff evidence is incomplete")
        assert isinstance(input_evidence, Mapping)
        assert isinstance(stem_evidence, Mapping)
        assert isinstance(prediction_evidence, Mapping)
        if view.get("input_view_id") != IDM_STEM_INPUT_VIEW_ID:
            raise ValueError("HTDemucs input view identity is invalid")
        if input_evidence.get("input_view_id") != IDM_STEM_INPUT_VIEW_ID:
            raise ValueError("retained input view identity is invalid")
        input_path, input_content = _read_owned(
            separation_artifact_root,
            input_evidence.get("path"),
            "htdemucs.input",
        )
        _check_sha(
            input_content,
            input_evidence.get("input_audio_sha256"),
            "htdemucs.input.input_audio_sha256",
            "retained_input_invalid",
        )
        if input_evidence.get("source_audio_id") != source_audio_id:
            raise ValueError("retained input source ID does not match")
        if input_evidence.get("source_audio_sha256") != source_audio_sha256:
            raise ValueError("retained input source hash does not match")
        try:
            wav = parse_canonical_wav(input_content, None)
        except (TypeError, ValueError) as error:
            raise IdmPilotRunError(
                "retained HTDemucs input is not canonical WAV", code="retained_input_invalid"
            ) from error

        _stem_path, stem_content = _read_owned(
            stem_cache_root,
            stem_evidence.get("path"),
            "htdemucs.stem",
        )
        _check_sha(
            stem_content,
            stem_evidence.get("sha256"),
            "htdemucs.stem.sha256",
            "retained_input_invalid",
        )
        if stem_evidence.get("source_audio_sha256") != source_audio_sha256:
            raise ValueError("retained stem source hash does not match")
        oaf_path, oaf_content = _read_owned(
            separation_artifact_root,
            prediction_evidence.get("path"),
            "htdemucs.prediction",
        )
        del oaf_path
        _check_sha(
            oaf_content,
            prediction_evidence.get("artifact_sha256"),
            "htdemucs.prediction.artifact_sha256",
            "retained_oaf_prediction_invalid",
        )
        try:
            oaf_artifact = read_prediction_artifact(oaf_content)
        except (PredictionArtifactError, StrictJsonError, TypeError, ValueError) as error:
            raise IdmPilotRunError(
                "retained OaF prediction is invalid", code="retained_oaf_prediction_invalid"
            ) from error
        oaf_audio = oaf_artifact.prediction.audio
        if (
            oaf_audio.source_audio_id != source_audio_id
            or oaf_audio.source_audio_sha256 != source_audio_sha256
            or oaf_audio.input_view_id != IDM_STEM_INPUT_VIEW_ID
            or oaf_audio.input_audio_sha256 != input_evidence.get("input_audio_sha256")
        ):
            raise IdmPilotRunError(
                "retained OaF prediction identity does not match handoff",
                code="retained_oaf_prediction_invalid",
            )
        descriptor = oaf_artifact.prediction.descriptor
        if descriptor.payload.get("backend_id") != OAF_BACKEND_ID:
            raise IdmPilotRunError(
                "retained OaF backend identity is invalid", code="retained_oaf_prediction_invalid"
            )
        if descriptor.sha256 != row.get("oaf_backend_descriptor_sha256"):
            raise IdmPilotRunError(
                "retained OaF descriptor identity does not match",
                code="retained_oaf_prediction_invalid",
            )
        if descriptor.payload.get("model_id") != row.get("oaf_model_id"):
            raise IdmPilotRunError(
                "retained OaF model identity does not match", code="retained_oaf_prediction_invalid"
            )
        if any(
            event.prediction_map_version != row.get("oaf_prediction_map_version")
            for event in oaf_artifact.prediction.events
        ):
            raise IdmPilotRunError(
                "retained OaF mapping identity does not match",
                code="retained_oaf_prediction_invalid",
            )
        return _PreparedHandoffRow(
            row=row,
            audio=CanonicalAudio(
                path=input_path,
                source_audio_id=source_audio_id,
                source_audio_sha256=source_audio_sha256,
                input_view_id=IDM_STEM_INPUT_VIEW_ID,
                input_audio_sha256=sha256_hex(input_content),
                byte_length=wav.byte_length,
                sample_rate=wav.sample_rate,
                channel_count=wav.channel_count,
                sample_width_bytes=wav.sample_width_bytes,
                audio_frame_count=wav.audio_frame_count,
            ),
            oaf_artifact=oaf_artifact,
        )
    except IdmPilotRunError:
        raise
    except (KeyError, TypeError, ValueError, StrictJsonError) as error:
        raise IdmPilotRunError(
            "retained HPA-328 evidence is invalid", code="retained_input_invalid"
        ) from error


def _validate_lineage(
    handoff: LoadedSeparationPilotManifest,
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
) -> None:
    if not isinstance(handoff, LoadedSeparationPilotManifest):
        raise TypeError("handoff must be LoadedSeparationPilotManifest")
    if not isinstance(reference, LoadedReferenceSetManifest):
        raise TypeError("reference must be LoadedReferenceSetManifest")
    if not isinstance(timing, LoadedReferenceTimingManifest):
        raise TypeError("timing must be LoadedReferenceTimingManifest")
    if len(handoff.rows) < 20 or len(handoff.rows) > 30:
        raise IdmPilotRunError(
            "handoff population must contain 20-30 rows", code="preflight_invalid"
        )
    raw_ids = [row.get("simfile_id") for row in handoff.rows]
    if any(type(simfile_id) is not int or simfile_id <= 0 for simfile_id in raw_ids):
        raise IdmPilotRunError("handoff simfile IDs are invalid", code="preflight_invalid")
    handoff_ids = {int(simfile_id) for simfile_id in raw_ids}
    if len(handoff_ids) != len(handoff.rows):
        raise IdmPilotRunError(
            "handoff membership contains duplicate IDs", code="preflight_invalid"
        )
    reference_ids = {row.view.simfile_id for row in reference.rows}
    if not handoff_ids <= reference_ids:
        raise IdmPilotRunError("handoff and reference membership differ", code="preflight_invalid")
    for row in handoff.rows:
        if row.get("reference_manifest_sha256") != reference.manifest_sha256:
            raise IdmPilotRunError(
                "reference manifest SHA differs from handoff", code="preflight_invalid"
            )
        if row.get("reference_manifest_version") != reference.corpus_version:
            raise IdmPilotRunError(
                "reference manifest version differs from handoff", code="preflight_invalid"
            )
        if row.get("reference_timing_manifest_sha256") != timing.manifest_sha256:
            raise IdmPilotRunError(
                "timing manifest SHA differs from handoff", code="preflight_invalid"
            )
        if row.get("reference_timing_version") != timing.corpus_version:
            raise IdmPilotRunError(
                "timing manifest version differs from handoff", code="preflight_invalid"
            )


def _backend_failure_code(error: BaseException) -> str:
    raw = getattr(error, "code", None)
    if raw in {"input_path_invalid", "input_audio_invalid", "runtime_artifact_invalid"}:
        return "retained_input_invalid"
    if isinstance(raw, str) and raw:
        return raw
    return "worker_protocol_failed"


def classify_idm_backend_error(code: str) -> tuple[str, Literal["item_local", "poison"]]:
    """Map an IDM host error to its persisted native code and disposition."""
    if code in {"worker_start_failed", "worker_protocol_failed", "native_event_invalid"}:
        return code, "poison"
    if code in IDM_FAILURE_TO_COHORT_REASON:
        return code, "item_local"
    return "worker_protocol_failed", "poison"


def _set_failed(item: dict[str, object], code: str, detail: BaseException | str) -> None:
    item["execution_disposition"] = "failed"
    item["native_failure_code"] = code
    item["cohort_failure_reason"] = IDM_FAILURE_TO_COHORT_REASON[code]
    item["failure_detail"] = _bounded_error(detail)


def _clear_failure_fields(item: dict[str, object]) -> None:
    for field in ("native_failure_code", "cohort_failure_reason", "failure_detail"):
        item.pop(field, None)


def _native_failure_counts(items: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        code = item.get("native_failure_code")
        if isinstance(code, str):
            counts[code] += 1
    return counts


def _set_quarantined(item: dict[str, object], detail: str = "reference is quarantined") -> None:
    item["execution_disposition"] = "quarantined"
    item["cohort_failure_reason"] = "reference_quarantined"
    item["failure_detail"] = detail


def _prediction_relative_path(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def _prediction_matches(
    content: bytes,
    *,
    audio: CanonicalAudio,
    descriptor: BackendDescriptor,
) -> PredictionArtifact:
    try:
        artifact = read_prediction_artifact(content)
    except (PredictionArtifactError, StrictJsonError, TypeError, ValueError) as error:
        raise IdmPilotRunError(
            "IDM prediction artifact is invalid", code="prediction_artifact_invalid"
        ) from error
    if not prediction_artifact_matches_audio(
        artifact,
        source_audio_id=audio.source_audio_id,
        source_audio_sha256=audio.source_audio_sha256,
        audio=audio,
        descriptor=descriptor,
        prediction_map_version=IDM_PREDICTION_MAP_ID,
    ):
        raise IdmPilotRunError(
            "IDM prediction artifact identity mismatch", code="prediction_artifact_invalid"
        )
    return artifact


def _cohort_identity(
    snapshot: Mapping[str, object],
    *,
    backend_id: str,
    model_id: str,
    input_view_id: str,
    prediction_map_version: str,
    cohort_suffix: str,
) -> CohortIdentity:
    return CohortIdentity(
        cohort_id=f"{snapshot['run_id']}:{cohort_suffix}",
        reference_manifest_sha256=str(snapshot["reference_manifest_sha256"]),
        reference_timing_version=str(snapshot["reference_timing_version"]),
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=backend_id,
        model_id=model_id,
        model_lock_sha256=str(
            snapshot["model_lock_sha256"]
            if backend_id == IDM_BACKEND_ID
            else snapshot["oaf_model_lock_sha256"]
        ),
        backend_descriptor_sha256=str(
            snapshot["backend_descriptor_sha256"]
            if backend_id == IDM_BACKEND_ID
            else snapshot["oaf_backend_descriptor_sha256"]
        ),
        prediction_map_version=prediction_map_version,
        input_view_id=input_view_id,
    )


def _row_reference(
    row: Mapping[str, object], mappings: Mapping[int, ReferenceMappingResult | None]
) -> ReferenceMappingResult | None:
    simfile_id = row.get("simfile_id")
    if type(simfile_id) is not int:
        raise ValueError("handoff simfile_id is invalid")
    return mappings.get(simfile_id)


def build_idm_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    """Re-read IDM artifacts and reconstruct the complete HPA-325 cohort."""
    descriptor_payload = snapshot.get("backend_descriptor")
    if not isinstance(descriptor_payload, Mapping):
        raise ValueError("IDM snapshot backend descriptor is missing")
    model_id = snapshot.get("model_id")
    descriptor_sha = snapshot.get("backend_descriptor_sha256")
    model_lock_sha = snapshot.get("model_lock_sha256")
    if not all(
        isinstance(value, str) and value for value in (model_id, descriptor_sha, model_lock_sha)
    ):
        raise ValueError("IDM snapshot identity is incomplete")
    identity = _cohort_identity(
        snapshot,
        backend_id=IDM_BACKEND_ID,
        model_id=model_id,
        input_view_id=IDM_STEM_INPUT_VIEW_ID,
        prediction_map_version=IDM_PREDICTION_MAP_ID,
        cohort_suffix="idm",
    )
    items: list[CohortItem] = []
    for raw_item in snapshot.get("items", []):
        if not isinstance(raw_item, Mapping):
            raise ValueError("IDM snapshot item is invalid")
        simfile_id = raw_item.get("simfile_id")
        if type(simfile_id) is not int:
            raise ValueError("IDM snapshot simfile_id is invalid")
        reference = mappings.get(simfile_id)
        if raw_item.get("execution_disposition") in {"inferred", "resumed"}:
            prediction_path_value = raw_item.get("prediction_path")
            if reference is None:
                items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        None,
                        status="quarantined",
                        failure_reason="reference_quarantined",
                    )
                )
                continue
            if not isinstance(prediction_path_value, str):
                raise ValueError("successful IDM item has no prediction path")
            path = _owned_path(output_dir, prediction_path_value, "IDM prediction")
            content = read_regular_file_no_follow(path)
            expected_sha = raw_item.get("prediction_artifact_sha256")
            if sha256_hex(content) != expected_sha:
                raise ValueError("IDM prediction artifact hash differs from snapshot")
            artifact_audio = read_prediction_artifact(content).prediction.audio
            if (
                artifact_audio.source_audio_id != raw_item.get("source_audio_id")
                or artifact_audio.source_audio_sha256 != raw_item.get("source_audio_sha256")
                or artifact_audio.input_view_id != IDM_STEM_INPUT_VIEW_ID
                or artifact_audio.input_audio_sha256 != raw_item.get("input_audio_sha256")
            ):
                raise ValueError("IDM prediction artifact identity differs from snapshot")
            artifact = _prediction_matches(
                content,
                audio=artifact_audio,
                descriptor=BackendDescriptor(
                    payload=descriptor_payload, sha256=str(descriptor_sha)
                ),
            )
            items.append(
                cohort_item_from_validated_prediction_artifact(
                    identity,
                    str(simfile_id),
                    reference,
                    artifact,
                )
            )
        else:
            code = raw_item.get("native_failure_code")
            reason = raw_item.get("cohort_failure_reason")
            if reference is None:
                items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        None,
                        status="quarantined",
                        failure_reason="reference_quarantined",
                    )
                )
            else:
                if not isinstance(reason, str) or reason not in COHORT_FAILURE_REASONS:
                    if isinstance(code, str):
                        reason = IDM_FAILURE_TO_COHORT_REASON.get(code, "inference_failed")
                    else:
                        reason = "inference_failed"
                items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        reference,
                        status="failed",
                        failure_reason=reason,  # type: ignore[arg-type]
                    )
                )
    return identity, tuple(items)


def build_oaf_cohort_from_handoff(
    snapshot: Mapping[str, object],
    *,
    handoff_rows: Iterable[Mapping[str, object]],
    mappings: Mapping[int, ReferenceMappingResult | None],
    separation_artifact_root: Path,
    stem_cache_root: Path | None = None,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    """Re-read retained OaF artifacts and build the same fixed population."""
    rows = tuple(handoff_rows)
    if not rows:
        raise ValueError("handoff rows are unavailable")
    first = rows[0]
    identity = _cohort_identity(
        {
            **snapshot,
            "oaf_model_lock_sha256": first.get("oaf_model_lock_sha256"),
            "oaf_backend_descriptor_sha256": first.get("oaf_backend_descriptor_sha256"),
        },
        backend_id=OAF_BACKEND_ID,
        model_id=str(first.get("oaf_model_id")),
        input_view_id=IDM_STEM_INPUT_VIEW_ID,
        prediction_map_version=str(first.get("oaf_prediction_map_version")),
        cohort_suffix="oaf",
    )
    items: list[CohortItem] = []
    for row in rows:
        simfile_id = row.get("simfile_id")
        if type(simfile_id) is not int:
            raise ValueError("handoff simfile_id is invalid")
        reference = mappings.get(simfile_id)
        view = row.get("htdemucs")
        if not isinstance(view, Mapping) or view.get("status") not in _SUCCESSFUL_HANDOFF_STATUSES:
            code, _ = _handoff_failure_code(view if isinstance(view, Mapping) else {})
            reason = IDM_FAILURE_TO_COHORT_REASON.get(code, "inference_failed")
            items.append(
                cohort_item_without_prediction(
                    identity,
                    str(simfile_id),
                    reference,
                    status="quarantined" if reference is None else "failed",
                    failure_reason="reference_quarantined" if reference is None else reason,  # type: ignore[arg-type]
                )
            )
            continue
        try:
            if reference is None:
                items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        None,
                        status="quarantined",
                        failure_reason="reference_quarantined",
                    )
                )
                continue
            if stem_cache_root is None:
                raise ValueError("stem_cache_root is required to revalidate retained evidence")
            prepared = _prepare_handoff_row(
                row,
                separation_artifact_root=separation_artifact_root,
                stem_cache_root=stem_cache_root,
            )
            if prepared.audio is None or prepared.oaf_artifact is None:
                raise IdmPilotRunError(
                    "retained OaF evidence is unavailable",
                    code=prepared.native_failure_code or "retained_oaf_prediction_invalid",
                )
            items.append(
                cohort_item_from_validated_prediction_artifact(
                    identity,
                    str(simfile_id),
                    reference,
                    prepared.oaf_artifact,
                )
            )
        except (IdmPilotRunError, PredictionArtifactError, StrictJsonError, TypeError, ValueError):
            items.append(
                cohort_item_without_prediction(
                    identity,
                    str(simfile_id),
                    reference,
                    status="quarantined" if reference is None else "failed",
                    failure_reason=(
                        "reference_quarantined"
                        if reference is None
                        else "prediction_artifact_invalid"
                    ),
                )
            )
    return identity, tuple(items)


def _build_header(
    request: IdmPilotRunRequest,
    *,
    handoff: LoadedSeparationPilotManifest,
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    lock: IdmModelLock,
    descriptor: BackendDescriptor,
    model_lock_sha256: str,
    inference_config: Mapping[str, object],
    inference_config_sha256_value: str,
    run_id: str,
    started_at: str,
) -> dict[str, object]:
    first = handoff.rows[0]
    return {
        "schema": IDM_PILOT_RUN_SCHEMA,
        "run_id": run_id,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "handoff_manifest_version": handoff.corpus_version,
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_manifest_version": reference.corpus_version,
        "reference_timing_manifest_sha256": timing.manifest_sha256,
        "reference_timing_version": timing.corpus_version,
        "backend_descriptor_sha256": descriptor.sha256,
        "backend_descriptor": dict(descriptor.payload),
        "model_id": lock.model_id,
        "model_lock_sha256": model_lock_sha256,
        "inference_config": dict(inference_config),
        "inference_config_sha256": inference_config_sha256_value,
        "input_view_id": IDM_STEM_INPUT_VIEW_ID,
        "crux_commit": request.crux_commit,
        "request_timeout_seconds": inference_config.get("request_timeout_seconds"),
        "worker_close_timeout_seconds": IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
        "oaf_model_id": first.get("oaf_model_id"),
        "oaf_model_lock_sha256": first.get("oaf_model_lock_sha256"),
        "oaf_backend_descriptor_sha256": first.get("oaf_backend_descriptor_sha256"),
        "oaf_prediction_map_version": first.get("oaf_prediction_map_version"),
        "started_at": started_at,
    }


def _initial_item(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "simfile_id": row.get("simfile_id"),
        "source_row_sha256": row.get("source_row_sha256"),
        "source_audio_id": row.get("source_audio_id"),
        "source_audio_sha256": row.get("source_audio_sha256"),
        "source_duration_sec": row.get("source_duration_sec"),
        "upstream_htdemucs_status": (
            row.get("htdemucs", {}).get("status")
            if isinstance(row.get("htdemucs"), Mapping)
            else None
        ),
        "upstream_htdemucs_failure_code": (
            row.get("htdemucs", {}).get("failure_code")
            if isinstance(row.get("htdemucs"), Mapping)
            else None
        ),
    }


def _write_checkpoint(
    run_path: Path,
    header: Mapping[str, object],
    items: Iterable[Mapping[str, object]],
    *,
    overall_status: str | None = None,
    completed_at: str | None = None,
    close_error: Mapping[str, object] | None = None,
    native_failure_counts: Mapping[str, int] | None = None,
) -> None:
    snapshot: dict[str, object] = dict(header)
    snapshot["items"] = [
        dict(item) for item in sorted(items, key=lambda item: int(item["simfile_id"]))
    ]
    snapshot.update(_snapshot_counts(snapshot["items"]))  # type: ignore[arg-type]
    if overall_status is not None:
        snapshot["overall_status"] = overall_status
    if completed_at is not None:
        snapshot["completed_at"] = completed_at
    if close_error is not None:
        snapshot["close_error"] = dict(close_error)
    if native_failure_counts is not None:
        snapshot["native_failure_counts"] = dict(sorted(native_failure_counts.items()))
    write_idm_pilot_run(run_path, snapshot)


def _build_report_cohorts(
    snapshot: Mapping[str, object],
    *,
    handoff: LoadedSeparationPilotManifest,
    mappings: Mapping[int, ReferenceMappingResult | None],
    separation_artifact_root: Path,
    stem_cache_root: Path,
    output_dir: Path,
) -> tuple[
    tuple[CohortIdentity, tuple[CohortItem, ...]], tuple[CohortIdentity, tuple[CohortItem, ...]]
]:
    oaf = build_oaf_cohort_from_handoff(
        snapshot,
        handoff_rows=handoff.rows,
        mappings=mappings,
        separation_artifact_root=separation_artifact_root,
        stem_cache_root=stem_cache_root,
    )
    idm = build_idm_cohort_from_snapshot(snapshot, mappings=mappings, output_dir=output_dir)
    return oaf, idm


def _outcome_from_scores(
    idm_score: object,
    *,
    run_id: str,
    run_path: Path,
    reports_path: Path,
    snapshot: Mapping[str, object],
) -> IdmPilotRunOutcome:
    population = getattr(idm_score, "population", None)
    if population is None:
        counts = _snapshot_counts(
            item for item in snapshot.get("items", []) if isinstance(item, Mapping)
        )
    else:
        counts = {
            "success_count": population.success_count,
            "failed_count": population.failed_count,
            "skipped_count": population.skipped_count,
            "quarantined_count": population.quarantined_count,
        }
    status = snapshot.get("overall_status")
    if status not in {"complete", "partial", "failed"}:
        status = "partial"
    if status == "complete" and (
        counts["failed_count"] or counts["quarantined_count"] or counts["skipped_count"]
    ):
        status = "partial"
    native_counts = tuple(
        sorted(
            (str(key), int(value))
            for key, value in (snapshot.get("native_failure_counts", {}) or {}).items()
            if isinstance(key, str) and isinstance(value, int)
        )
    )
    return IdmPilotRunOutcome(
        overall_status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "complete" else 2 if status == "failed" else 1,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        success_count=counts["success_count"],
        failed_count=counts["failed_count"],
        skipped_count=counts["skipped_count"],
        quarantined_count=counts["quarantined_count"],
        native_failure_counts=native_counts,
    )


def run_idm_pilot(
    request: IdmPilotRunRequest,
    *,
    backend_factory: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] = _utc_now,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> IdmPilotRunOutcome:
    """Run the fixed handoff population through one persistent IDM backend."""
    if not isinstance(request, IdmPilotRunRequest):
        raise TypeError("request must be IdmPilotRunRequest")
    try:
        _validate_output_roots(request)
        handoff = load_separation_pilot_manifest(request.separation_handoff_path)
        reference = load_reference_set_manifest(request.reference_manifest_path)
        timing = load_reference_timing_manifest(request.timing_manifest_path)
        _validate_lineage(handoff, reference, timing)
        mappings = preflight_reference_mappings(
            reference,
            timing,
            timing_output_root=request.timing_manifest_path.parent.parent,
        )
        if not {int(row["simfile_id"]) for row in handoff.rows} <= set(mappings):
            raise IdmPilotRunError("reference mapping membership differs", code="preflight_invalid")
        lock = load_idm_model_lock(request.model_lock_path)
        model_lock_sha256 = compute_model_lock_sha256(request.model_lock_path)
        descriptor = descriptor_for_lock(lock)
        inference_config = build_idm_inference_config(
            model_lock_sha256,
            descriptor.sha256,
            input_view_id=IDM_STEM_INPUT_VIEW_ID,
            timeout_seconds=IDM_REQUEST_TIMEOUT_SECONDS,
        )
        inference_config_sha = idm_inference_config_sha256(inference_config)
        run_id = build_run_id(
            handoff.manifest_sha256,
            handoff.corpus_version,
            reference.manifest_sha256,
            reference.corpus_version,
            timing.manifest_sha256,
            timing.corpus_version,
            descriptor.sha256,
            model_lock_sha256,
            inference_config_sha,
            IDM_STEM_INPUT_VIEW_ID,
            request.crux_commit,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, IdmPilotRunError):
        return _fatal_outcome()

    run_dir = request.output_dir / "runs" / run_id
    run_path = run_dir / "run.json"
    reports_path = run_dir / "reports"
    if run_path.exists() and not request.resume:
        return _fatal_outcome()
    try:
        started_at = _timestamp(clock())
        run_dir.mkdir(parents=True, exist_ok=True)
        header = _build_header(
            request,
            handoff=handoff,
            reference=reference,
            timing=timing,
            lock=lock,
            descriptor=descriptor,
            model_lock_sha256=model_lock_sha256,
            inference_config=inference_config,
            inference_config_sha256_value=inference_config_sha,
            run_id=run_id,
            started_at=started_at,
        )
        prior_items: dict[int, Mapping[str, object]] = {}
        if request.resume and run_path.exists():
            prior_snapshot = parse_idm_pilot_run(
                read_regular_file_no_follow(run_path), expected_run_id=run_id
            )
            raw_prior_items = prior_snapshot.get("items", [])
            if not isinstance(raw_prior_items, list):
                raise ValueError("prior run items are invalid")
            prior_items = {
                int(item["simfile_id"]): item
                for item in raw_prior_items
                if isinstance(item, Mapping) and type(item.get("simfile_id")) is int
            }
        items: list[dict[str, object]] = []
        for row in handoff.rows:
            item = _initial_item(row)
            prior = prior_items.get(int(row["simfile_id"]))
            if prior is not None:
                item.update(dict(prior))
            items.append(item)
        native_failure_counts = _native_failure_counts(items)
        _write_checkpoint(
            run_path,
            header,
            items,
            native_failure_counts=native_failure_counts,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()

    backend: Any | None = None
    backend_poisoned = False
    fatal_run_error = False
    close_error: dict[str, str] | None = None
    native_failure_counts: Counter[str] = _native_failure_counts(items)
    try:
        for index, row in enumerate(handoff.rows):
            item = items[index]
            try:
                mapping = mappings.get(int(row["simfile_id"]))
                if mapping is None:
                    _set_quarantined(item)
                    continue
                prepared = _prepare_handoff_row(
                    row,
                    separation_artifact_root=request.separation_artifact_root,
                    stem_cache_root=request.stem_cache_root,
                )
                if prepared.native_failure_code is not None or prepared.audio is None:
                    _set_failed(
                        item,
                        prepared.native_failure_code or "upstream_stem_unavailable",
                        prepared.upstream_failure_code
                        or "retained HPA-328 HTDemucs row is unavailable",
                    )
                    native_failure_counts[item["native_failure_code"]] += 1  # type: ignore[index]
                    continue
                audio = prepared.audio
                item["input_view_id"] = audio.input_view_id
                item["input_audio_sha256"] = audio.input_audio_sha256
                target = prediction_path(
                    request.output_dir,
                    simfile_id=int(row["simfile_id"]),
                    source_audio_sha256=audio.source_audio_sha256,
                    backend_descriptor_sha256=descriptor.sha256,
                    inference_config_sha256=inference_config_sha,
                )
                existing = _read_output_prediction(target, request.output_dir)
                prior = prior_items.get(int(row["simfile_id"]))
                if existing is not None:
                    if not request.resume:
                        _set_failed(
                            item, "prediction_output_conflict", "prediction artifact already exists"
                        )
                        native_failure_counts[item["native_failure_code"]] += 1  # type: ignore[index]
                        continue
                    if prior is None or not isinstance(
                        prior.get("prediction_artifact_sha256"), str
                    ):
                        _set_failed(
                            item,
                            "prediction_output_conflict",
                            "prediction artifact has no exact-run ledger evidence",
                        )
                        native_failure_counts[item["native_failure_code"]] += 1  # type: ignore[index]
                        continue
                    try:
                        artifact = _prediction_matches(existing, audio=audio, descriptor=descriptor)
                        if prior["prediction_artifact_sha256"] != artifact.artifact_sha256:
                            raise IdmPilotRunError(
                                "resumed prediction hash differs",
                                code="prediction_artifact_invalid",
                            )
                    except IdmPilotRunError as error:
                        _set_failed(item, error.code, error)
                        native_failure_counts[item["native_failure_code"]] += 1  # type: ignore[index]
                        continue
                    _clear_failure_fields(item)
                    item.update(
                        {
                            "execution_disposition": "resumed",
                            "prediction_path": _prediction_relative_path(
                                target, request.output_dir
                            ),
                            "prediction_artifact_sha256": artifact.artifact_sha256,
                        }
                    )
                    for field in ("wall_time_sec", "rtf"):
                        if field in prior:
                            item[field] = prior[field]
                    continue
                if backend_poisoned:
                    _set_failed(
                        item,
                        "worker_protocol_failed",
                        "inference was not attempted after a poison failure",
                    )
                    native_failure_counts[item["native_failure_code"]] += 1  # type: ignore[index]
                    continue
                if backend is None:
                    factory = backend_factory or IdmBackend
                    try:
                        backend = factory(
                            runtime_python=request.runtime_python,
                            model_lock_path=request.model_lock_path,
                            model_root=request.model_root,
                            input_root=request.separation_artifact_root,
                            timeout_seconds=IDM_REQUEST_TIMEOUT_SECONDS,
                            close_timeout_seconds=IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
                        )
                        backend_descriptor = backend.descriptor()
                        if (
                            not isinstance(backend_descriptor, BackendDescriptor)
                            or backend_descriptor != descriptor
                        ):
                            raise IdmBackendError(
                                "backend descriptor identity changed", code="worker_start_failed"
                            )
                    except BaseException as error:  # pylint: disable=broad-exception-caught
                        raw_code = getattr(error, "code", None)
                        startup_code = (
                            raw_code
                            if isinstance(raw_code, str)
                            and raw_code
                            in {
                                "worker_start_failed",
                                "worker_protocol_failed",
                                "native_event_invalid",
                            }
                            else "worker_start_failed"
                        )
                        code, disposition = classify_idm_backend_error(startup_code)
                        _set_failed(item, code, error)
                        native_failure_counts[code] += 1
                        if disposition == "poison":
                            backend_poisoned = True
                        continue
                started = perf_counter()
                try:
                    native = backend.transcribe(audio)
                except BaseException as error:  # pylint: disable=broad-exception-caught
                    code, disposition = classify_idm_backend_error(_backend_failure_code(error))
                    _set_failed(item, code, error)
                    native_failure_counts[code] += 1
                    if disposition == "poison":
                        backend_poisoned = True
                    continue
                elapsed = max(0.0, perf_counter() - started)
                if (
                    not isinstance(native, NativePrediction)
                    or native.audio != audio
                    or native.descriptor != descriptor
                ):
                    _set_failed(
                        item, "native_event_invalid", "IDM backend returned an invalid prediction"
                    )
                    native_failure_counts["native_event_invalid"] += 1
                    backend_poisoned = True
                    continue
                item["wall_time_sec"] = elapsed
                duration = row.get("source_duration_sec")
                if isinstance(duration, (int, float, Decimal)) and float(duration) > 0:
                    item["rtf"] = elapsed / float(duration)
                try:
                    mapped, _diagnostics = map_idm_prediction(native)
                except BaseException as error:  # pylint: disable=broad-exception-caught
                    _set_failed(item, "native_event_invalid", error)
                    native_failure_counts["native_event_invalid"] += 1
                    backend_poisoned = True
                    continue
                try:
                    published = publish_prediction_artifact(target, mapped)
                except ArtifactPublicationError as error:
                    _set_failed(item, "prediction_publish_failed", error)
                    native_failure_counts["prediction_publish_failed"] += 1
                    continue
                except (PredictionArtifactError, OSError, TypeError, ValueError) as error:
                    _set_failed(item, "prediction_artifact_invalid", error)
                    native_failure_counts["prediction_artifact_invalid"] += 1
                    continue
                _clear_failure_fields(item)
                item.update(
                    {
                        "execution_disposition": "inferred",
                        "prediction_path": _prediction_relative_path(target, request.output_dir),
                        "prediction_artifact_sha256": published.sha256,
                    }
                )
            except IdmPilotRunError as error:
                code = (
                    error.code if error.code in IDM_FAILURE_TO_COHORT_REASON else "inference_failed"
                )
                _set_failed(item, code, error)
                native_failure_counts[code] += 1
            except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError) as error:
                _set_failed(item, "retained_input_invalid", error)
                native_failure_counts["retained_input_invalid"] += 1
            finally:
                try:
                    native_failure_counts = _native_failure_counts(items)
                    _write_checkpoint(
                        run_path,
                        header,
                        items,
                        native_failure_counts=native_failure_counts,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                    fatal_run_error = True
            if fatal_run_error:
                break
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:  # pylint: disable=broad-exception-caught
                close_error = {
                    "code": str(getattr(error, "code", "worker_close_failed")),
                    "message": _bounded_error(error),
                }

    for item in items:
        if "execution_disposition" not in item:
            _set_failed(
                item,
                "worker_protocol_failed",
                "inference was not attempted after a fatal run error",
            )
            native_failure_counts["worker_protocol_failed"] += 1

    native_failure_counts = _native_failure_counts(items)
    counts = _snapshot_counts(items)
    all_done = all(item.get("execution_disposition") in _DISPOSITIONS for item in items)
    if fatal_run_error:
        overall_status = "failed"
    elif (
        all_done
        and counts["failed_count"] == 0
        and counts["quarantined_count"] == 0
        and close_error is None
    ):
        overall_status = "complete"
    else:
        overall_status = "partial"
    completed_at: str | None = None
    if overall_status == "complete":
        try:
            completed_at = _timestamp(clock())
        except (OSError, RuntimeError, TypeError, ValueError):
            overall_status = "failed"
    try:
        _write_checkpoint(
            run_path,
            header,
            items,
            overall_status=overall_status,
            completed_at=completed_at,
            close_error=close_error,
            native_failure_counts=native_failure_counts,
        )
        final_snapshot = parse_idm_pilot_run(
            read_regular_file_no_follow(run_path), expected_run_id=run_id
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        return _fatal_outcome()
    if fatal_run_error:
        return _fatal_outcome()
    try:
        (oaf_identity, oaf_items), (idm_identity, idm_items) = _build_report_cohorts(
            final_snapshot,
            handoff=handoff,
            mappings=mappings,
            separation_artifact_root=request.separation_artifact_root,
            stem_cache_root=request.stem_cache_root,
            output_dir=request.output_dir,
        )
        oaf_score = score_cohort(oaf_identity, oaf_items, diagnostics_for=())
        idm_score = score_cohort(idm_identity, idm_items, diagnostics_for=())
        oaf_reports = reports_path / "oaf"
        idm_reports = reports_path / "idm"
        write_cohort_reports(oaf_score, oaf_reports)
        write_cohort_reports(idm_score, idm_reports)
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, PredictionArtifactError):
        return _fatal_outcome()
    outcome = _outcome_from_scores(
        idm_score,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        snapshot=final_snapshot,
    )
    if close_error is not None and outcome.overall_status == "complete":
        return IdmPilotRunOutcome(
            **{
                **outcome.__dict__,
                "overall_status": "partial",
                "exit_code": 1,
            }
        )
    return outcome


__all__ = [
    "IDM_FAILURE_TO_COHORT_REASON",
    "IDM_PILOT_RUN_SCHEMA",
    "IDM_STEM_INPUT_VIEW_ID",
    "IdmPilotRunError",
    "IdmPilotRunOutcome",
    "IdmPilotRunRequest",
    "build_idm_cohort_from_snapshot",
    "build_idm_inference_config",
    "build_oaf_cohort_from_handoff",
    "build_run_id",
    "classify_idm_backend_error",
    "compute_model_lock_sha256",
    "idm_inference_config_sha256",
    "parse_idm_pilot_run",
    "render_idm_pilot_run",
    "run_idm_pilot",
    "write_idm_pilot_run",
]
