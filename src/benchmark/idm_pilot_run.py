"""Concrete sequential consumer for the immutable HPA-328 IDM pilot handoff.

The runner intentionally owns only the HPA-396 boundary.  HPA-328 stems and
OaF predictions are retained evidence: this module reads them from the two
explicit owner roots, checks their bytes and identities, and never attempts to
recreate either upstream artifact.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    PublishedArtifact,
    publish_immutable_file_at,
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
from src.benchmark.corpus_cache import CacheIndexStore, ResolvedSourceAudio, resolve_source_audio
from src.benchmark.durability import atomic_replace_bytes, ensure_durable_directory
from src.benchmark.idm_model import (
    IDM_REQUEST_TIMEOUT_SECONDS,
    IdmModelLock,
    idm_inference_config,
    load_idm_model_lock,
)
from src.benchmark.input_view import materialize_full_mix_audio, parse_canonical_wav
from src.benchmark.mapping import map_idm_prediction
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    PredictionArtifactError,
    prediction_artifact_matches_audio,
    prediction_path,
    publish_prediction_artifact,
    read_prediction_artifact,
    render_prediction_artifact,
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
IDM_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
IDM_SMOKE_SCHEMA = "crux.idm-smoke/v1"
IDM_FULL_MIX_SMOKE_RUN_SCHEMA = "crux.idm-full-mix-smoke-run/v1"
IDM_FULL_MIX_SMOKE_DIRNAME = "full-mix-smoke"
IDM_FULL_MIX_SMOKE_REPORT_DIRNAME = "reports"
IDM_SMOKE_CASE_ORDER = (
    "short",
    "long",
    "sparse",
    "dense",
    "median_duration",
)
_IDM_SMOKE_CASE_REASONS = frozenset(IDM_SMOKE_CASE_ORDER)

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
    "output_integrity_failed": "prediction_artifact_invalid",
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


class IdmSmokeManifestError(ValueError):
    """Raised when the offline/production five-song smoke contract is invalid."""

    def __init__(self, message: str, *, code: str = "preflight_invalid") -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class IdmSmokeCase:
    """One reason-labelled member of the fixed full-mix smoke population."""

    reason: str
    simfile_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or self.reason not in _IDM_SMOKE_CASE_REASONS:
            raise IdmSmokeManifestError("smoke case reason is invalid")
        if type(self.simfile_id) is not int or self.simfile_id <= 0:
            raise IdmSmokeManifestError("smoke case simfile_id must be a positive integer")


@dataclass(frozen=True)
class IdmSmokeManifest:
    """Canonical, ordered five-song smoke membership."""

    schema: str
    cases: tuple[IdmSmokeCase, ...]

    def __post_init__(self) -> None:
        if self.schema != IDM_SMOKE_SCHEMA:
            raise IdmSmokeManifestError("smoke manifest schema is invalid")
        if not isinstance(self.cases, tuple):
            raise TypeError("smoke manifest cases must be a tuple")
        if len(self.cases) != len(IDM_SMOKE_CASE_ORDER):
            raise IdmSmokeManifestError("smoke manifest must contain exactly five cases")
        if any(not isinstance(case, IdmSmokeCase) for case in self.cases):
            raise TypeError("smoke manifest cases must be IdmSmokeCase values")
        reasons = tuple(case.reason for case in self.cases)
        if reasons != IDM_SMOKE_CASE_ORDER:
            raise IdmSmokeManifestError("smoke case reasons must use the canonical order")
        ids = tuple(case.simfile_id for case in self.cases)
        if len(set(ids)) != len(ids):
            raise IdmSmokeManifestError("smoke case simfile IDs must be unique")


@dataclass(frozen=True)
class IdmFullMixSmokeRequest:
    """Inputs for the separate five-song full-mix IDM diagnostic."""

    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    smoke_manifest_path: Path
    source_cache_dir: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    crux_commit: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "separation_handoff_path",
            "reference_manifest_path",
            "timing_manifest_path",
            "smoke_manifest_path",
            "source_cache_dir",
            "output_dir",
            "model_lock_path",
            "model_root",
            "runtime_python",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if self.crux_commit is not None and (
            not isinstance(self.crux_commit, str) or _COMMIT_RE.fullmatch(self.crux_commit) is None
        ):
            raise ValueError("crux_commit must be a lowercase 40-character commit")


@dataclass(frozen=True)
class IdmFullMixSmokeOutcome:
    """Stable result for one offline or production full-mix smoke run."""

    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int

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
        for field in ("success_count", "failed_count", "skipped_count", "quarantined_count"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")


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
    input_content: bytes | None = None
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


def _smoke_handoff_rows(
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if isinstance(handoff, LoadedSeparationPilotManifest):
        return tuple(handoff.rows)
    if isinstance(handoff, Mapping):
        raw_rows = handoff.get("rows")
        if not isinstance(raw_rows, Iterable) or isinstance(raw_rows, (str, bytes)):
            raise IdmSmokeManifestError("handoff rows are unavailable")
        rows = tuple(raw_rows)
    else:
        try:
            rows = tuple(handoff)
        except TypeError as error:
            raise IdmSmokeManifestError("handoff rows are unavailable") from error
    if any(not isinstance(row, Mapping) for row in rows):
        raise IdmSmokeManifestError("handoff row is invalid")
    return rows  # type: ignore[return-value]


def _coerce_smoke_case(value: object) -> IdmSmokeCase:
    if isinstance(value, IdmSmokeCase):
        return value
    if not isinstance(value, Mapping) or set(value) != {"reason", "simfile_id"}:
        raise IdmSmokeManifestError("smoke case must contain reason and simfile_id")
    return IdmSmokeCase(
        reason=value["reason"],  # type: ignore[arg-type]
        simfile_id=value["simfile_id"],  # type: ignore[arg-type]
    )


def _coerce_smoke_manifest(value: object) -> IdmSmokeManifest:
    if isinstance(value, IdmSmokeManifest):
        return value
    if not isinstance(value, Mapping) or set(value) != {"cases", "schema"}:
        raise IdmSmokeManifestError("smoke manifest must contain the exact schema/cases keys")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list):
        raise IdmSmokeManifestError("smoke manifest cases must be an array")
    try:
        cases = tuple(_coerce_smoke_case(case) for case in raw_cases)
        return IdmSmokeManifest(schema=value.get("schema"), cases=cases)  # type: ignore[arg-type]
    except IdmSmokeManifestError:
        raise
    except (TypeError, ValueError) as error:
        raise IdmSmokeManifestError("smoke manifest is invalid") from error


def validate_idm_smoke_manifest(
    manifest: IdmSmokeManifest | Mapping[str, object],
    *,
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]] | None = None,
) -> IdmSmokeManifest:
    """Validate exact five-case shape and, when supplied, handoff membership."""
    normalized = _coerce_smoke_manifest(manifest)
    if handoff is not None:
        handoff_ids: set[int] = set()
        for row in _smoke_handoff_rows(handoff):
            simfile_id = row.get("simfile_id")
            if type(simfile_id) is int and simfile_id > 0:
                handoff_ids.add(simfile_id)
        missing = [
            case.simfile_id for case in normalized.cases if case.simfile_id not in handoff_ids
        ]
        if missing:
            raise IdmSmokeManifestError(
                "smoke case simfile IDs are not all present in the loaded handoff"
            )
    return normalized


def render_idm_smoke_manifest(
    cases: IdmSmokeManifest | Iterable[IdmSmokeCase | Mapping[str, object]],
) -> bytes:
    """Render canonical smoke JSON without inventing production membership."""
    if isinstance(cases, IdmSmokeManifest):
        manifest = cases
    else:
        try:
            manifest = IdmSmokeManifest(
                schema=IDM_SMOKE_SCHEMA,
                cases=tuple(_coerce_smoke_case(case) for case in cases),
            )
        except TypeError as error:
            raise TypeError("cases must be iterable") from error
    payload = {
        "cases": [
            {"reason": case.reason, "simfile_id": case.simfile_id} for case in manifest.cases
        ],
        "schema": manifest.schema,
    }
    return canonical_json_bytes(payload, trailing_newline=True)


def parse_idm_smoke_manifest(
    content: bytes,
    *,
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]] | None = None,
) -> IdmSmokeManifest:
    """Parse a canonical smoke manifest and optionally bind it to handoff IDs."""
    if not isinstance(content, bytes):
        raise TypeError("smoke manifest content must be bytes")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise IdmSmokeManifestError("smoke manifest must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise IdmSmokeManifestError(str(error)) from None
    try:
        manifest = validate_idm_smoke_manifest(value, handoff=handoff)  # type: ignore[arg-type]
    except IdmSmokeManifestError:
        raise
    if render_idm_smoke_manifest(manifest) != content:
        raise IdmSmokeManifestError("smoke manifest is not semantically canonical")
    return manifest


def load_idm_smoke_manifest(
    path: Path,
    *,
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]] | None = None,
) -> IdmSmokeManifest:
    if not isinstance(path, Path):
        raise TypeError("smoke manifest path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError) as error:
        raise IdmSmokeManifestError("smoke manifest is unavailable") from error
    return parse_idm_smoke_manifest(content, handoff=handoff)


def write_idm_smoke_manifest(
    path: Path,
    manifest: IdmSmokeManifest | Mapping[str, object],
    *,
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]] | None = None,
) -> None:
    if not isinstance(path, Path):
        raise TypeError("smoke manifest path must be a Path")
    normalized = validate_idm_smoke_manifest(manifest, handoff=handoff)
    atomic_replace_bytes(path, render_idm_smoke_manifest(normalized))


def _smoke_mapping_event_count(mapping: object, simfile_id: int) -> int:
    if mapping is None:
        raise IdmSmokeManifestError("reference mapping is unavailable")
    common_events = (
        mapping.get("common_events")
        if isinstance(mapping, Mapping)
        else getattr(mapping, "common_events", None)
    )
    if not isinstance(common_events, (tuple, list)):
        raise IdmSmokeManifestError(f"reference mapping for {simfile_id} is invalid")
    return len(common_events)


def _smoke_duration(value: object, simfile_id: int) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise IdmSmokeManifestError(f"source duration for {simfile_id} is invalid")
    try:
        duration = Decimal(str(value))
    except (ArithmeticError, ValueError) as error:
        raise IdmSmokeManifestError(f"source duration for {simfile_id} is invalid") from error
    if not duration.is_finite() or duration <= 0:
        raise IdmSmokeManifestError(f"source duration for {simfile_id} is invalid")
    return duration


def _smoke_median(values: Iterable[Decimal]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise IdmSmokeManifestError("smoke candidates are unavailable")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def select_idm_smoke_cases(
    handoff: LoadedSeparationPilotManifest | Iterable[Mapping[str, object]],
    reference_mappings: Mapping[int, object],
) -> tuple[IdmSmokeCase, ...]:
    """Select five pre-IDM cases from loaded HPA-328 evidence only."""
    if not isinstance(reference_mappings, Mapping):
        raise TypeError("reference_mappings must be a Mapping")
    rows = _smoke_handoff_rows(handoff)
    seen_ids: set[int] = set()
    candidates: list[tuple[int, Decimal, int]] = []
    for row in rows:
        simfile_id = row.get("simfile_id")
        if type(simfile_id) is not int or simfile_id <= 0:
            raise IdmSmokeManifestError("handoff simfile IDs are invalid")
        if simfile_id in seen_ids:
            raise IdmSmokeManifestError("handoff membership contains duplicate IDs")
        seen_ids.add(simfile_id)
        view = row.get("htdemucs")
        if not isinstance(view, Mapping) or view.get("status") not in _SUCCESSFUL_HANDOFF_STATUSES:
            continue
        mapping = reference_mappings.get(simfile_id)
        if mapping is None:
            continue
        duration = _smoke_duration(row.get("source_duration_sec"), simfile_id)
        event_count = _smoke_mapping_event_count(mapping, simfile_id)
        candidates.append((simfile_id, duration, event_count))

    if len(candidates) < len(IDM_SMOKE_CASE_ORDER):
        raise IdmSmokeManifestError("fewer than five eligible smoke candidates remain")

    remaining = list(candidates)
    all_durations = [duration for _simfile_id, duration, _event_count in candidates]

    def take(key: Callable[[tuple[int, Decimal, int]], object], *, reverse: bool = False):
        if reverse:
            best_value = max(key(candidate) for candidate in remaining)
            eligible = [candidate for candidate in remaining if key(candidate) == best_value]
            chosen = min(eligible, key=lambda candidate: candidate[0])
        else:
            best_value = min(key(candidate) for candidate in remaining)
            eligible = [candidate for candidate in remaining if key(candidate) == best_value]
            chosen = min(eligible, key=lambda candidate: candidate[0])
        remaining.remove(chosen)
        return chosen

    selected: list[IdmSmokeCase] = []
    short = take(lambda candidate: candidate[1])
    selected.append(IdmSmokeCase("short", short[0]))
    long = take(lambda candidate: candidate[1], reverse=True)
    selected.append(IdmSmokeCase("long", long[0]))
    sparse = take(lambda candidate: candidate[2])
    selected.append(IdmSmokeCase("sparse", sparse[0]))
    dense = take(lambda candidate: candidate[2], reverse=True)
    selected.append(IdmSmokeCase("dense", dense[0]))
    median_duration = _smoke_median(all_durations)
    median_case = take(lambda candidate: abs(candidate[1] - median_duration))
    selected.append(IdmSmokeCase("median_duration", median_case[0]))
    return tuple(selected)


def materialize_idm_full_mix_audio(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
) -> CanonicalAudio:
    """Use the historical OaF full-mix canonicalization for smoke inputs."""
    if not isinstance(source_audio, ResolvedSourceAudio):
        raise TypeError("source_audio must be ResolvedSourceAudio")
    audio = materialize_full_mix_audio(
        source_audio,
        output_path,
        input_root=input_root,
        input_view_id=IDM_FULL_MIX_INPUT_VIEW_ID,
        max_input_audio_frames=None,
    )
    if not isinstance(audio, CanonicalAudio):
        raise ValueError("full-mix materializer returned invalid audio")
    if audio.input_view_id != IDM_FULL_MIX_INPUT_VIEW_ID:
        raise ValueError("full-mix materializer returned the wrong input view")
    if audio.path.resolve() != output_path.resolve():
        raise ValueError("full-mix materializer returned the wrong path")
    if (
        audio.source_audio_id != source_audio.source_audio_id
        or audio.source_audio_sha256 != source_audio.source_audio_sha256
    ):
        raise ValueError("full-mix materializer changed source identity")
    return audio


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


_PRIMARY_REPORT_FILENAMES = (
    "summary.json",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
    "event_diagnostics.jsonl",
    "summary.md",
)


@dataclass
class _PrimaryNamespace:
    """Held descriptors for the primary run-owned output namespace."""

    run_id: str
    output_dir: Path
    output_fd: int
    runs_fd: int
    predictions_fd: int
    run_fd: int
    input_fd: int
    reports_fd: int
    oaf_reports_fd: int
    idm_reports_fd: int
    prediction_directory_identities: dict[str, tuple[tuple[int, int], ...]]


def _close_primary_namespace(namespace: _PrimaryNamespace | None) -> None:
    if namespace is None:
        return
    for descriptor in (
        namespace.idm_reports_fd,
        namespace.oaf_reports_fd,
        namespace.reports_fd,
        namespace.input_fd,
        namespace.run_fd,
        namespace.predictions_fd,
        namespace.runs_fd,
        namespace.output_fd,
    ):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _close_primary_prediction_parents(parents: Mapping[str, int]) -> None:
    for descriptor in set(parents.values()):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _primary_directory_identity(directory_fd: int) -> tuple[int, int]:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError("primary namespace component is not a directory")
    return int(metadata.st_dev), int(metadata.st_ino)


def _verify_primary_directory_chain(
    output_dir: Path,
    components: tuple[str, ...],
    expected_identities: tuple[tuple[int, int], ...],
    field: str,
) -> None:
    """Re-walk one primary path without following a swapped component."""
    if not isinstance(output_dir, Path) or not isinstance(field, str) or not field:
        raise TypeError("primary identity guard arguments are invalid")
    if len(expected_identities) != len(components) + 1:
        raise ValueError("primary identity guard chain is incomplete")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("primary directory no-follow support is unavailable")
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        current_fd = os.open(output_dir, flags)
        descriptors.append(current_fd)
        for index, expected in enumerate(expected_identities):
            if _primary_directory_identity(current_fd) != expected:
                raise IdmPilotRunError(
                    f"{field} directory identity changed", code="output_integrity_failed"
                )
            if index == len(components):
                break
            component = components[index]
            if not component or "/" in component or component in {".", ".."}:
                raise ValueError("primary identity guard component is invalid")
            current_fd = os.open(component, flags, dir_fd=current_fd)
            descriptors.append(current_fd)
    except IdmPilotRunError:
        raise
    except OSError as error:
        raise IdmPilotRunError(
            f"{field} directory chain is unavailable", code="output_integrity_failed"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _primary_run_directory_identities(namespace: _PrimaryNamespace) -> None:
    _verify_primary_directory_chain(
        namespace.output_dir,
        ("runs", namespace.run_id),
        tuple(
            _primary_directory_identity(descriptor)
            for descriptor in (namespace.output_fd, namespace.runs_fd, namespace.run_fd)
        ),
        "IDM run",
    )


def _primary_input_root_identities(namespace: _PrimaryNamespace) -> None:
    _verify_primary_directory_chain(
        namespace.output_dir,
        ("runs", namespace.run_id, "inputs"),
        tuple(
            _primary_directory_identity(descriptor)
            for descriptor in (
                namespace.output_fd,
                namespace.runs_fd,
                namespace.run_fd,
                namespace.input_fd,
            )
        ),
        "IDM input",
    )


def _primary_input_directory_identities(
    namespace: _PrimaryNamespace, simfile_id: int, directory_fd: int
) -> None:
    _verify_primary_directory_chain(
        namespace.output_dir,
        ("runs", namespace.run_id, "inputs", str(simfile_id)),
        tuple(
            _primary_directory_identity(descriptor)
            for descriptor in (
                namespace.output_fd,
                namespace.runs_fd,
                namespace.run_fd,
                namespace.input_fd,
                directory_fd,
            )
        ),
        "IDM verified input",
    )


def _primary_report_directory_identities(
    namespace: _PrimaryNamespace, cohort: str, directory_fd: int
) -> None:
    if cohort not in {"oaf", "idm"}:
        raise ValueError("primary report cohort is invalid")
    _verify_primary_directory_chain(
        namespace.output_dir,
        ("runs", namespace.run_id, "reports", cohort),
        tuple(
            _primary_directory_identity(descriptor)
            for descriptor in (
                namespace.output_fd,
                namespace.runs_fd,
                namespace.run_fd,
                namespace.reports_fd,
                directory_fd,
            )
        ),
        f"{cohort} report",
    )


def _open_primary_child_directory(parent_fd: int, name: str) -> int:
    """Open or create one private directory relative to a held parent."""
    if not isinstance(parent_fd, int) or not isinstance(name, str):
        raise TypeError("primary directory parent and name are invalid")
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("primary directory name is invalid")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("primary directory no-follow support is unavailable")
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        return os.open(name, flags, dir_fd=parent_fd)


def _open_primary_report_directories(
    reports_path: Path, namespace: _PrimaryNamespace
) -> tuple[int, int]:
    """Open the two report roots relative to the held run report directory."""
    if not isinstance(reports_path, Path):
        raise TypeError("reports_path must be a Path")
    oaf_fd: int | None = None
    try:
        oaf_fd = _open_primary_child_directory(namespace.reports_fd, "oaf")
        idm_fd = _open_primary_child_directory(namespace.reports_fd, "idm")
    except BaseException:
        if oaf_fd is not None:
            os.close(oaf_fd)
        raise
    return oaf_fd, idm_fd


def _open_primary_namespace(
    request: IdmPilotRunRequest,
    *,
    run_id: str,
    run_dir: Path,
    reports_path: Path,
) -> _PrimaryNamespace:
    """Hold every primary output directory used after preflight."""
    ensure_durable_directory(request.output_dir)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("primary output no-follow support is unavailable")
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        output_fd = os.open(request.output_dir, flags)
        descriptors.append(output_fd)
        runs_fd = _open_primary_child_directory(output_fd, "runs")
        descriptors.append(runs_fd)
        predictions_fd = _open_primary_child_directory(output_fd, "predictions")
        descriptors.append(predictions_fd)
        run_fd = _open_primary_child_directory(runs_fd, run_id)
        descriptors.append(run_fd)
        input_fd = _open_primary_child_directory(run_fd, "inputs")
        descriptors.append(input_fd)
        reports_fd = _open_primary_child_directory(run_fd, "reports")
        descriptors.append(reports_fd)
        namespace = _PrimaryNamespace(
            run_id=run_id,
            output_dir=request.output_dir.absolute(),
            output_fd=output_fd,
            runs_fd=runs_fd,
            predictions_fd=predictions_fd,
            run_fd=run_fd,
            input_fd=input_fd,
            reports_fd=reports_fd,
            oaf_reports_fd=-1,
            idm_reports_fd=-1,
            prediction_directory_identities={},
        )
        oaf_fd, idm_fd = _open_primary_report_directories(reports_path, namespace)
        namespace.oaf_reports_fd = oaf_fd
        namespace.idm_reports_fd = idm_fd
        descriptors.extend((oaf_fd, idm_fd))
        return namespace
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _read_primary_file_at(directory_fd: int, name: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("primary file no-follow support is unavailable")
    descriptor = os.open(
        name,
        os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("primary output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _primary_atomic_replace_at(
    directory_fd: int,
    name: str,
    content: bytes,
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    """Atomically replace one primary leaf relative to a held directory."""
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("primary output leaf name is invalid")
    if not isinstance(content, bytes):
        raise TypeError("primary output content must be bytes")
    temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short primary output write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if before_replace is not None:
            before_replace()
        os.replace(temporary_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _primary_existing_leaf(directory_fd: int, name: str) -> bytes | None:
    try:
        return _read_primary_file_at(directory_fd, name)
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class _PrimaryLeafSnapshot:
    content: bytes | None
    mode: int | None


def _primary_snapshot_leaf(directory_fd: int, name: str) -> _PrimaryLeafSnapshot:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("primary file no-follow support is unavailable")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return _PrimaryLeafSnapshot(content=None, mode=None)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("primary output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return _PrimaryLeafSnapshot(
                    content=b"".join(chunks),
                    mode=stat.S_IMODE(metadata.st_mode),
                )
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _primary_restore_leaf(directory_fd: int, name: str, snapshot: _PrimaryLeafSnapshot) -> None:
    if snapshot.content is None:
        _primary_remove_leaf(directory_fd, name)
        return
    if snapshot.mode is None:
        raise ValueError("primary leaf snapshot mode is missing")
    _primary_atomic_replace_at(directory_fd, name, snapshot.content)
    descriptor = os.open(
        name,
        os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("primary output is not a regular file")
        os.fchmod(descriptor, snapshot.mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _primary_remove_leaf(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    os.fsync(directory_fd)


def _primary_replace_with_identity_guard(
    directory_fd: int,
    name: str,
    content: bytes,
    verify: Callable[[], None],
) -> None:
    """Replace one primary leaf only while its lexical directory is unchanged."""
    previous = _primary_existing_leaf(directory_fd, name)
    verify()
    replaced = False
    try:
        _primary_atomic_replace_at(
            directory_fd,
            name,
            content,
            before_replace=verify,
        )
        replaced = True
        verify()
    except BaseException:
        if replaced:
            try:
                if previous is None:
                    _primary_remove_leaf(directory_fd, name)
                else:
                    _primary_atomic_replace_at(directory_fd, name, previous)
            except OSError:
                pass
        raise


def _primary_publish_immutable_with_identity_guard(
    directory_fd: int,
    name: str,
    content: bytes,
    verify: Callable[[], None],
) -> str:
    """Publish one primary immutable leaf only while its directory is unchanged."""
    previous = _primary_existing_leaf(directory_fd, name)
    verify()
    published = False
    try:
        digest = publish_immutable_file_at(directory_fd, name, content)
        published = True
        verify()
    except (ArtifactPublicationError, OSError, TypeError, ValueError):
        if published and previous is None:
            try:
                _primary_remove_leaf(directory_fd, name)
            except OSError:
                pass
        raise
    return digest


def _open_primary_prediction_parent(
    target: Path, namespace: _PrimaryNamespace, output_dir: Path
) -> int:
    """Hold the dynamic prediction parent through inference publication."""
    try:
        relative = target.relative_to(output_dir)
    except ValueError as error:
        raise IdmPilotRunError(
            "prediction path escapes output root", code="prediction_artifact_invalid"
        ) from error
    if len(relative.parts) < 3 or relative.parts[0] != "predictions":
        raise IdmPilotRunError(
            "prediction path is outside the primary namespace", code="prediction_artifact_invalid"
        )
    prediction_key = relative.as_posix()
    expected_identities = namespace.prediction_directory_identities.get(prediction_key)
    if expected_identities is not None:
        _verify_primary_directory_chain(
            namespace.output_dir,
            relative.parts[:-1],
            expected_identities,
            "IDM prediction",
        )
    else:
        _verify_primary_directory_chain(
            namespace.output_dir,
            ("predictions",),
            tuple(
                _primary_directory_identity(descriptor)
                for descriptor in (namespace.output_fd, namespace.predictions_fd)
            ),
            "IDM prediction root",
        )
    parent_fd = namespace.predictions_fd
    opened: list[int] = []
    opened_identities: list[tuple[int, int]] = []
    try:
        for component in relative.parts[1:-1]:
            child_fd = _open_primary_child_directory(parent_fd, component)
            opened.append(child_fd)
            opened_identities.append(_primary_directory_identity(child_fd))
            if parent_fd != namespace.predictions_fd:
                os.close(parent_fd)
            parent_fd = child_fd
        namespace.prediction_directory_identities.setdefault(
            prediction_key,
            tuple(
                _primary_directory_identity(descriptor)
                for descriptor in (namespace.output_fd, namespace.predictions_fd)
            )
            + tuple(opened_identities),
        )
        return parent_fd
    except BaseException:
        for descriptor in reversed(opened):
            if descriptor != parent_fd:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if parent_fd != namespace.predictions_fd:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        raise


def _primary_verify_prediction_parent(namespace: _PrimaryNamespace, target: Path) -> None:
    try:
        relative = target.absolute().relative_to(namespace.output_dir)
    except ValueError as error:
        raise IdmPilotRunError(
            "prediction path escapes output root", code="output_integrity_failed"
        ) from error
    if len(relative.parts) < 3 or relative.parts[0] != "predictions":
        raise IdmPilotRunError(
            "prediction path is outside the primary namespace", code="output_integrity_failed"
        )
    expected = namespace.prediction_directory_identities.get(relative.as_posix())
    if expected is None:
        raise IdmPilotRunError(
            "prediction directory identity is not held", code="output_integrity_failed"
        )
    _verify_primary_directory_chain(
        namespace.output_dir,
        relative.parts[:-1],
        expected,
        "IDM prediction",
    )


def _publish_primary_prediction_at(
    directory_fd: int,
    target: Path,
    prediction: object,
    *,
    namespace: _PrimaryNamespace | None = None,
) -> PublishedArtifact:
    content = render_prediction_artifact(prediction)  # type: ignore[arg-type]
    verify = (
        (lambda: _primary_verify_prediction_parent(namespace, target))
        if namespace is not None
        else None
    )
    try:
        if verify is None:
            digest = publish_immutable_file_at(directory_fd, target.name, content)
        else:
            digest = _primary_publish_immutable_with_identity_guard(
                directory_fd,
                target.name,
                content,
                verify,
            )
        persisted = _read_primary_file_at(directory_fd, target.name)
        artifact = read_prediction_artifact(persisted)
    except IdmPilotRunError:
        raise
    except (
        ArtifactPublicationError,
        OSError,
        PredictionArtifactError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ArtifactPublicationError):
            raise
        raise PredictionArtifactError("published prediction bytes are invalid") from error
    if artifact.content != content or artifact.artifact_sha256 != digest:
        raise PredictionArtifactError("published prediction bytes changed")
    return PublishedArtifact(path=target, sha256=digest)


def _validate_primary_namespace_path(
    path: Path,
    output_dir: Path,
    output_root: Path,
    field: str,
    *,
    leaf_kind: Literal["directory", "file"],
) -> None:
    """Validate one primary output path without following preseeded components."""
    if not isinstance(path, Path) or not isinstance(output_dir, Path):
        raise TypeError("primary namespace paths must be Paths")
    try:
        relative = path.relative_to(output_dir)
    except ValueError as error:
        raise IdmPilotRunError(f"{field} escapes output_dir", code="preflight_invalid") from error

    cursor = output_dir
    components = (cursor, *relative.parts)
    for index, part in enumerate(components):
        if index:
            cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IdmPilotRunError(f"{field} is unavailable", code="preflight_invalid") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise IdmPilotRunError(f"{field} must not use symlinks", code="preflight_invalid")
        is_leaf = index == len(components) - 1
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise IdmPilotRunError(f"{field} parent is not a directory", code="preflight_invalid")
        if is_leaf:
            if leaf_kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
                raise IdmPilotRunError(f"{field} is not a directory", code="preflight_invalid")
            if leaf_kind == "file" and not stat.S_ISREG(metadata.st_mode):
                raise IdmPilotRunError(f"{field} is not a regular file", code="preflight_invalid")

    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise IdmPilotRunError(f"{field} is unavailable", code="preflight_invalid") from error
    if not resolved.is_relative_to(output_root):
        raise IdmPilotRunError(f"{field} escapes output_dir", code="preflight_invalid")


def _validate_primary_namespace(
    request: IdmPilotRunRequest,
    *,
    run_id: str,
    allow_existing_run: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Validate the primary run-owned namespace before any mutable writes."""
    output_dir = request.output_dir
    output_root = _root_resolved(output_dir, "output_dir")
    _validate_primary_namespace_path(
        output_dir,
        output_dir,
        output_root,
        "output_dir",
        leaf_kind="directory",
    )
    runs_root = output_dir / "runs"
    predictions_root = output_dir / "predictions"
    run_dir = runs_root / run_id
    run_path = run_dir / "run.json"
    input_root = run_dir / "inputs"
    reports_path = run_dir / "reports"
    derived_directories = (
        (runs_root, "runs"),
        (predictions_root, "predictions"),
        (run_dir, "IDM run directory"),
        (input_root, "IDM input root"),
        (reports_path, "IDM reports root"),
        (reports_path / "oaf", "OaF reports root"),
        (reports_path / "idm", "IDM reports root"),
    )
    for path, field in derived_directories:
        _validate_primary_namespace_path(
            path,
            output_dir,
            output_root,
            field,
            leaf_kind="directory",
        )

    _validate_primary_namespace_path(
        run_path,
        output_dir,
        output_root,
        "IDM run snapshot",
        leaf_kind="file",
    )
    run_exists = False
    try:
        run_exists = stat.S_ISDIR(run_dir.lstat().st_mode)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise IdmPilotRunError(
            "IDM run directory is unavailable", code="preflight_invalid"
        ) from error
    if run_exists and not request.resume and not allow_existing_run:
        raise IdmPilotRunError("IDM run directory already exists", code="preflight_invalid")
    if request.resume:
        try:
            metadata = run_path.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise IdmPilotRunError(
                "resume run snapshot is not a regular file", code="preflight_invalid"
            )

    for cohort in ("oaf", "idm"):
        report_root = reports_path / cohort
        for filename in _PRIMARY_REPORT_FILENAMES:
            _validate_primary_namespace_path(
                report_root / filename,
                output_dir,
                output_root,
                f"{cohort} report {filename}",
                leaf_kind="file",
            )
    return run_dir, run_path, reports_path, input_root


def _validate_primary_prediction_target(
    target: Path,
    *,
    output_dir: Path,
) -> None:
    output_root = _root_resolved(output_dir, "output_dir")
    _validate_primary_namespace_path(
        target,
        output_dir,
        output_root,
        "IDM prediction path",
        leaf_kind="file",
    )


def _validate_primary_reports_namespace(reports_path: Path, output_dir: Path) -> None:
    output_root = _root_resolved(output_dir, "output_dir")
    for cohort in ("oaf", "idm"):
        report_root = reports_path / cohort
        _validate_primary_namespace_path(
            report_root,
            output_dir,
            output_root,
            f"{cohort} reports root",
            leaf_kind="directory",
        )
        for filename in _PRIMARY_REPORT_FILENAMES:
            _validate_primary_namespace_path(
                report_root / filename,
                output_dir,
                output_root,
                f"{cohort} report {filename}",
                leaf_kind="file",
            )


def _stage_primary_verified_input(
    run_dir: Path,
    output_dir: Path,
    simfile_id: int,
    content: bytes,
    *,
    input_root_fd: int | None = None,
    namespace: _PrimaryNamespace | None = None,
) -> Path:
    """Persist verified handoff bytes in a private, run-owned regular file."""
    if type(simfile_id) is not int or simfile_id <= 0:
        raise IdmPilotRunError("simfile_id is invalid", code="retained_input_invalid")
    if not isinstance(content, bytes):
        raise TypeError("verified input content must be bytes")
    output_root = _root_resolved(output_dir, "output_dir")
    input_root = run_dir / "inputs"
    song_root = input_root / str(simfile_id)
    staged_path = song_root / "verified.wav"
    for path, field in (
        (input_root, "IDM input root"),
        (song_root, "IDM song input root"),
        (staged_path, "IDM verified input"),
    ):
        _validate_primary_namespace_path(
            path,
            output_dir,
            output_root,
            field,
            leaf_kind="file" if path == staged_path else "directory",
        )
    directory_fd: int | None = None
    try:
        if namespace is not None and input_root_fd is not None:
            _primary_input_root_identities(namespace)
        if input_root_fd is None:
            ensure_durable_directory(song_root)
            directory_fd = os.open(
                song_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        else:
            directory_fd = _open_primary_child_directory(input_root_fd, str(simfile_id))
    except OSError as error:
        raise IdmPilotRunError(
            "verified IDM input directory is unavailable", code="retained_input_invalid"
        ) from error
    assert directory_fd is not None
    try:
        if namespace is not None:
            _primary_input_root_identities(namespace)

            def verify_input_directory() -> None:
                _primary_input_directory_identities(namespace, simfile_id, directory_fd)

            try:
                _primary_publish_immutable_with_identity_guard(
                    directory_fd,
                    staged_path.name,
                    content,
                    verify_input_directory,
                )
            except (ArtifactPublicationError, OSError) as error:
                raise IdmPilotRunError(
                    "verified IDM input could not be staged", code="retained_input_invalid"
                ) from error
            persisted = _read_primary_file_at(directory_fd, staged_path.name)
            metadata = os.stat(staged_path.name, dir_fd=directory_fd, follow_symlinks=False)
            regular = stat.S_ISREG(metadata.st_mode)
            private = not stat.S_IMODE(metadata.st_mode) & 0o077
            verify_input_directory()
        elif input_root_fd is not None:
            try:
                publish_immutable_file_at(directory_fd, staged_path.name, content)
            except (ArtifactPublicationError, OSError) as error:
                raise IdmPilotRunError(
                    "verified IDM input could not be staged", code="retained_input_invalid"
                ) from error
            persisted = _read_primary_file_at(directory_fd, staged_path.name)
            metadata = os.stat(staged_path.name, dir_fd=directory_fd, follow_symlinks=False)
            regular = stat.S_ISREG(metadata.st_mode)
            private = not stat.S_IMODE(metadata.st_mode) & 0o077
        else:
            try:
                publish_immutable_file_at(directory_fd, staged_path.name, content)
            except (ArtifactPublicationError, OSError) as error:
                raise IdmPilotRunError(
                    "verified IDM input could not be staged", code="retained_input_invalid"
                ) from error
            persisted = read_regular_file_no_follow(staged_path)
            metadata = staged_path.lstat()
            regular = stat.S_ISREG(metadata.st_mode)
            private = not stat.S_IMODE(metadata.st_mode) & 0o077
    except OSError as error:
        raise IdmPilotRunError(
            "verified IDM input could not be re-read", code="retained_input_invalid"
        ) from error
    finally:
        os.close(directory_fd)
    if persisted != content or not regular or not private:
        raise IdmPilotRunError(
            "verified IDM input bytes or permissions changed", code="retained_input_invalid"
        )
    return staged_path


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
            input_content=input_content,
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
    if raw in {
        "input_path_invalid",
        "input_audio_invalid",
        "invalid_request",
        "runtime_artifact_invalid",
    }:
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


def _primary_prediction_relative_path(path: Path, output_dir: Path) -> str:
    """Return the lexical primary path without resolving a swapped directory."""
    try:
        relative = path.relative_to(output_dir)
    except ValueError as error:
        raise IdmPilotRunError(
            "prediction path escapes output root", code="prediction_artifact_invalid"
        ) from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise IdmPilotRunError(
            "prediction path contains invalid components", code="prediction_artifact_invalid"
        )
    return relative.as_posix()


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
    prediction_parent_fds: Mapping[str, int] | None = None,
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
            held_parent_fd = (
                prediction_parent_fds.get(prediction_path_value)
                if prediction_parent_fds is not None
                else None
            )
            if held_parent_fd is not None:
                content = _read_primary_file_at(held_parent_fd, Path(prediction_path_value).name)
            else:
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
    run_fd: int | None = None,
    namespace: _PrimaryNamespace | None = None,
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
    content = render_idm_pilot_run(snapshot)
    if run_fd is None:
        write_idm_pilot_run(run_path, snapshot)
    elif namespace is None:
        _primary_atomic_replace_at(run_fd, run_path.name, content)
    else:
        _primary_replace_with_identity_guard(
            run_fd,
            run_path.name,
            content,
            lambda: _primary_run_directory_identities(namespace),
        )


def _build_report_cohorts(
    snapshot: Mapping[str, object],
    *,
    handoff: LoadedSeparationPilotManifest,
    mappings: Mapping[int, ReferenceMappingResult | None],
    separation_artifact_root: Path,
    stem_cache_root: Path,
    output_dir: Path,
    prediction_parent_fds: Mapping[str, int] | None = None,
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
    idm = build_idm_cohort_from_snapshot(
        snapshot,
        mappings=mappings,
        output_dir=output_dir,
        prediction_parent_fds=prediction_parent_fds,
    )
    return oaf, idm


def _publish_primary_cohort_reports(
    result: object,
    reports_fd: int,
    *,
    namespace: _PrimaryNamespace | None = None,
    cohort: str | None = None,
) -> None:
    """Render reports privately, then publish their fixed leaves by descriptor."""
    snapshots = {
        filename: _primary_snapshot_leaf(reports_fd, filename)
        for filename in _PRIMARY_REPORT_FILENAMES
    }
    touched: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".idm-primary-report-stage-") as stage_name:
            staged_dir = Path(stage_name)
            write_cohort_reports(result, staged_dir)  # type: ignore[arg-type]
            for filename in _PRIMARY_REPORT_FILENAMES:
                content = read_regular_file_no_follow(staged_dir / filename)
                touched.append(filename)
                if namespace is None:
                    _primary_atomic_replace_at(reports_fd, filename, content)
                else:
                    if cohort is None:
                        raise ValueError("primary report cohort is required")
                    _primary_replace_with_identity_guard(
                        reports_fd,
                        filename,
                        content,
                        lambda: _primary_report_directory_identities(namespace, cohort, reports_fd),
                    )
                    _primary_report_directory_identities(namespace, cohort, reports_fd)
            if namespace is not None:
                if cohort is None:
                    raise ValueError("primary report cohort is required")
                _primary_report_directory_identities(namespace, cohort, reports_fd)
    except BaseException:
        for filename in reversed(touched):
            try:
                _primary_restore_leaf(reports_fd, filename, snapshots[filename])
            except OSError:
                pass
        raise


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

    try:
        run_dir, run_path, reports_path, input_root = _validate_primary_namespace(
            request,
            run_id=run_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError, IdmPilotRunError):
        return _fatal_outcome()
    namespace: _PrimaryNamespace | None = None
    try:
        started_at = _timestamp(clock())
        ensure_durable_directory(request.output_dir)
        _validate_primary_namespace(
            request,
            run_id=run_id,
            allow_existing_run=True,
        )
        namespace = _open_primary_namespace(
            request,
            run_id=run_id,
            run_dir=run_dir,
            reports_path=reports_path,
        )
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
        if request.resume:
            try:
                prior_content = _read_primary_file_at(namespace.run_fd, run_path.name)
            except FileNotFoundError:
                prior_content = None
            if prior_content is not None:
                prior_snapshot = parse_idm_pilot_run(prior_content, expected_run_id=run_id)
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
            run_fd=namespace.run_fd,
            namespace=namespace,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        _close_primary_namespace(namespace)
        return _fatal_outcome()

    assert namespace is not None
    held_prediction_parents: dict[str, int] = {}

    backend: Any | None = None
    backend_poisoned = False
    fatal_run_error = False
    close_error: dict[str, str] | None = None
    native_failure_counts: Counter[str] = _native_failure_counts(items)
    try:
        for index, row in enumerate(handoff.rows):
            item = items[index]
            prediction_parent_fd: int | None = None
            prediction_parent_key: str | None = None
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
                try:
                    _validate_primary_prediction_target(target, output_dir=request.output_dir)
                except (OSError, RuntimeError, TypeError, ValueError, IdmPilotRunError) as error:
                    failure_code = (
                        "output_integrity_failed"
                        if getattr(error, "code", None) == "output_integrity_failed"
                        else "prediction_artifact_invalid"
                    )
                    _set_failed(item, failure_code, error)
                    native_failure_counts[failure_code] += 1
                    if failure_code == "output_integrity_failed":
                        fatal_run_error = True
                    continue
                try:
                    prediction_parent_fd = _open_primary_prediction_parent(
                        target, namespace, request.output_dir
                    )
                    prediction_parent_key = _primary_prediction_relative_path(
                        target, request.output_dir
                    )
                    previous_parent_fd = held_prediction_parents.get(prediction_parent_key)
                    if previous_parent_fd is None:
                        held_prediction_parents[prediction_parent_key] = prediction_parent_fd
                    else:
                        os.close(prediction_parent_fd)
                        prediction_parent_fd = previous_parent_fd
                    _primary_verify_prediction_parent(namespace, target)
                except (OSError, RuntimeError, TypeError, ValueError, IdmPilotRunError) as error:
                    failure_code = (
                        "output_integrity_failed"
                        if isinstance(error, IdmPilotRunError)
                        and error.code == "output_integrity_failed"
                        else "prediction_artifact_invalid"
                    )
                    _set_failed(item, failure_code, error)
                    native_failure_counts[failure_code] += 1
                    if failure_code == "output_integrity_failed":
                        fatal_run_error = True
                    continue
                try:
                    if prepared.input_content is None:
                        raise IdmPilotRunError(
                            "verified IDM input bytes are unavailable",
                            code="retained_input_invalid",
                        )
                    staged_path = _stage_primary_verified_input(
                        run_dir,
                        request.output_dir,
                        int(row["simfile_id"]),
                        prepared.input_content,
                        input_root_fd=namespace.input_fd,
                        namespace=namespace,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, IdmPilotRunError) as error:
                    failure_code = (
                        "output_integrity_failed"
                        if isinstance(error, IdmPilotRunError)
                        and error.code == "output_integrity_failed"
                        else "retained_input_invalid"
                    )
                    _set_failed(item, failure_code, error)
                    native_failure_counts[failure_code] += 1
                    if failure_code == "output_integrity_failed":
                        fatal_run_error = True
                    continue
                audio = replace(audio, path=staged_path)
                try:
                    existing = _read_primary_file_at(prediction_parent_fd, target.name)
                except FileNotFoundError:
                    existing = None
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
                            "prediction_path": prediction_parent_key
                            if prediction_parent_key is not None
                            else _primary_prediction_relative_path(target, request.output_dir),
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
                            input_root=input_root,
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
                    if prediction_parent_fd is None:
                        raise OSError("prediction parent is not held")
                    published = _publish_primary_prediction_at(
                        prediction_parent_fd,
                        target,
                        mapped,
                        namespace=namespace,
                    )
                except ArtifactPublicationError as error:
                    _set_failed(item, "prediction_publish_failed", error)
                    native_failure_counts["prediction_publish_failed"] += 1
                    continue
                except (PredictionArtifactError, OSError, TypeError, ValueError) as error:
                    failure_code = (
                        "output_integrity_failed"
                        if getattr(error, "code", None) == "output_integrity_failed"
                        else "prediction_artifact_invalid"
                    )
                    _set_failed(item, failure_code, error)
                    native_failure_counts[failure_code] += 1
                    if failure_code == "output_integrity_failed":
                        fatal_run_error = True
                    continue
                _clear_failure_fields(item)
                item.update(
                    {
                        "execution_disposition": "inferred",
                        "prediction_path": prediction_parent_key
                        if prediction_parent_key is not None
                        else _primary_prediction_relative_path(target, request.output_dir),
                        "prediction_artifact_sha256": published.sha256,
                    }
                )
            except IdmPilotRunError as error:
                code = (
                    error.code if error.code in IDM_FAILURE_TO_COHORT_REASON else "inference_failed"
                )
                _set_failed(item, code, error)
                native_failure_counts[code] += 1
                if error.code == "output_integrity_failed":
                    fatal_run_error = True
            except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError) as error:
                _set_failed(item, "retained_input_invalid", error)
                native_failure_counts["retained_input_invalid"] += 1
            finally:
                if (
                    prediction_parent_fd is not None
                    and prediction_parent_key not in held_prediction_parents
                ):
                    try:
                        os.close(prediction_parent_fd)
                    except OSError:
                        pass
                try:
                    native_failure_counts = _native_failure_counts(items)
                    _write_checkpoint(
                        run_path,
                        header,
                        items,
                        native_failure_counts=native_failure_counts,
                        run_fd=namespace.run_fd,
                        namespace=namespace,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
                    fatal_run_error = True
                if fatal_run_error:
                    break
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
            run_fd=namespace.run_fd,
            namespace=namespace,
        )
        final_snapshot = parse_idm_pilot_run(
            _read_primary_file_at(namespace.run_fd, run_path.name), expected_run_id=run_id
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError):
        _close_primary_prediction_parents(held_prediction_parents)
        _close_primary_namespace(namespace)
        return _fatal_outcome()
    if fatal_run_error:
        _close_primary_prediction_parents(held_prediction_parents)
        _close_primary_namespace(namespace)
        return _fatal_outcome()
    try:
        (oaf_identity, oaf_items), (idm_identity, idm_items) = _build_report_cohorts(
            final_snapshot,
            handoff=handoff,
            mappings=mappings,
            separation_artifact_root=request.separation_artifact_root,
            stem_cache_root=request.stem_cache_root,
            output_dir=request.output_dir,
            prediction_parent_fds=held_prediction_parents,
        )
        oaf_score = score_cohort(oaf_identity, oaf_items, diagnostics_for=())
        idm_score = score_cohort(idm_identity, idm_items, diagnostics_for=())
        _publish_primary_cohort_reports(
            oaf_score,
            namespace.oaf_reports_fd,
            namespace=namespace,
            cohort="oaf",
        )
        _publish_primary_cohort_reports(
            idm_score,
            namespace.idm_reports_fd,
            namespace=namespace,
            cohort="idm",
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, PredictionArtifactError):
        _close_primary_prediction_parents(held_prediction_parents)
        _close_primary_namespace(namespace)
        return _fatal_outcome()
    outcome = _outcome_from_scores(
        idm_score,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        snapshot=final_snapshot,
    )
    if close_error is not None and outcome.overall_status == "complete":
        final_outcome = IdmPilotRunOutcome(
            **{
                **outcome.__dict__,
                "overall_status": "partial",
                "exit_code": 1,
            }
        )
    else:
        final_outcome = outcome
    _close_primary_prediction_parents(held_prediction_parents)
    _close_primary_namespace(namespace)
    return final_outcome


def _fatal_full_mix_smoke() -> IdmFullMixSmokeOutcome:
    return IdmFullMixSmokeOutcome(
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


def _validate_full_mix_smoke_roots(request: IdmFullMixSmokeRequest) -> None:
    if request.output_dir.is_symlink():
        raise IdmSmokeManifestError("output_dir must not be a symlink")
    output = request.output_dir.resolve()
    for field in ("source_cache_dir", "model_root"):
        root = getattr(request, field).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise IdmSmokeManifestError("full-mix smoke output aliases an input root")


def _validate_full_mix_smoke_namespace(
    request: IdmFullMixSmokeRequest, *, run_id: str
) -> tuple[Path, Path, Path, Path]:
    """Validate and derive the smoke-only output paths before creating them."""
    output_root = request.output_dir.resolve()
    namespace = request.output_dir / IDM_FULL_MIX_SMOKE_DIRNAME
    runs_root = namespace / "runs"
    run_dir = runs_root / run_id
    run_path = run_dir / "run.json"
    input_root = run_dir / "inputs"
    reports_path = run_dir / IDM_FULL_MIX_SMOKE_REPORT_DIRNAME
    derived_paths = (
        ("full-mix smoke namespace", namespace),
        ("full-mix smoke runs root", runs_root),
        ("full-mix smoke run directory", run_dir),
        ("full-mix smoke run snapshot", run_path),
        ("full-mix smoke input root", input_root),
        ("full-mix smoke reports root", reports_path),
    )
    for field, path in derived_paths:
        if path.is_symlink():
            raise IdmSmokeManifestError(f"{field} must not be a symlink")
        try:
            resolved = path.resolve()
        except OSError as error:
            raise IdmSmokeManifestError(f"{field} is unavailable") from error
        if not resolved.is_relative_to(output_root):
            raise IdmSmokeManifestError(f"{field} escapes output_dir")
    if run_dir.exists():
        raise IdmSmokeManifestError("full-mix smoke run directory already exists")
    return run_dir, run_path, reports_path, input_root


def _validate_full_mix_smoke_prediction_target(
    run_dir: Path, output_root: Path, target: Path
) -> None:
    """Reject dynamic prediction parents that are symlinked or escape output."""
    if not isinstance(run_dir, Path) or not isinstance(output_root, Path):
        raise TypeError("prediction roots must be Paths")
    if not isinstance(target, Path):
        raise TypeError("prediction target must be a Path")
    try:
        relative = target.relative_to(run_dir)
        run_root = run_dir.resolve(strict=True)
        output_resolved = output_root.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise IdmSmokeManifestError("full-mix smoke prediction path is unavailable") from error
    if not relative.parts or not run_root.is_relative_to(output_resolved):
        raise IdmSmokeManifestError("full-mix smoke prediction path escapes output_dir")
    cursor = run_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise IdmSmokeManifestError("full-mix smoke prediction path must not use symlinks")
        try:
            resolved = cursor.resolve()
        except OSError as error:
            raise IdmSmokeManifestError("full-mix smoke prediction path is unavailable") from error
        if not resolved.is_relative_to(output_resolved):
            raise IdmSmokeManifestError("full-mix smoke prediction path escapes output_dir")
        if cursor != target and cursor.exists() and not cursor.is_dir():
            raise IdmSmokeManifestError("full-mix smoke prediction parent is not a directory")


def _smoke_run_id(
    *,
    handoff: LoadedSeparationPilotManifest,
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    smoke_manifest_sha256: str,
    descriptor_sha256: str,
    model_lock_sha256: str,
    inference_config_sha256_value: str,
    crux_commit: str | None,
) -> str:
    values = {
        "schema": IDM_FULL_MIX_SMOKE_RUN_SCHEMA,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "handoff_manifest_version": handoff.corpus_version,
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_manifest_version": reference.corpus_version,
        "timing_manifest_sha256": timing.manifest_sha256,
        "timing_manifest_version": timing.corpus_version,
        "smoke_manifest_sha256": smoke_manifest_sha256,
        "backend_descriptor_sha256": descriptor_sha256,
        "model_lock_sha256": model_lock_sha256,
        "inference_config_sha256": inference_config_sha256_value,
        "input_view_id": IDM_FULL_MIX_INPUT_VIEW_ID,
        "crux_commit": crux_commit,
    }
    return "idm-full-mix-" + sha256_hex(canonical_json_bytes(values))[:16]


def _write_full_mix_smoke_snapshot(path: Path, snapshot: Mapping[str, object]) -> None:
    if not isinstance(path, Path):
        raise TypeError("run path must be a Path")
    normalized = _normalize_snapshot_value(snapshot)
    atomic_replace_bytes(path, canonical_json_bytes(normalized, trailing_newline=True))


def _smoke_source_kwargs(source_row: Mapping[str, object]) -> dict[str, str | None]:
    return {
        "source_audio_key": (
            source_row.get("source_audio_key")
            if isinstance(source_row.get("source_audio_key"), str)
            else None
        ),
        "source_audio_content_hash": (
            source_row.get("source_audio_content_hash")
            if isinstance(source_row.get("source_audio_content_hash"), str)
            else None
        ),
        "source_endpoint_sha256": (
            source_row.get("source_endpoint_sha256")
            if isinstance(source_row.get("source_endpoint_sha256"), str)
            else None
        ),
        "source_bucket": (
            source_row.get("source_bucket")
            if isinstance(source_row.get("source_bucket"), str)
            else None
        ),
    }


def _set_full_mix_smoke_failed(
    item: dict[str, object], code: str, detail: BaseException | str
) -> None:
    if code not in IDM_FAILURE_TO_COHORT_REASON:
        code = "inference_failed"
    item.update(
        {
            "execution_disposition": "failed",
            "native_failure_code": code,
            "cohort_failure_reason": IDM_FAILURE_TO_COHORT_REASON[code],
            "failure_detail": _bounded_error(detail),
        }
    )


def run_idm_full_mix_smoke(
    request: IdmFullMixSmokeRequest,
    *,
    backend_factory: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] = _utc_now,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> IdmFullMixSmokeOutcome:
    """Run exactly the validated five-song smoke through the frozen IDM backend.

    This path intentionally owns its cache, input-view/config identity, run
    directory, and report directory.  It never calls the stem runner or the
    headline comparison module.
    """
    if not isinstance(request, IdmFullMixSmokeRequest):
        raise TypeError("request must be IdmFullMixSmokeRequest")
    try:
        _validate_full_mix_smoke_roots(request)
        handoff = load_separation_pilot_manifest(request.separation_handoff_path)
        reference = load_reference_set_manifest(request.reference_manifest_path)
        timing = load_reference_timing_manifest(request.timing_manifest_path)
        _validate_lineage(handoff, reference, timing)
        smoke_content = read_regular_file_no_follow(request.smoke_manifest_path)
        smoke_manifest = parse_idm_smoke_manifest(smoke_content, handoff=handoff)
        lock = load_idm_model_lock(request.model_lock_path)
        model_lock_sha256 = compute_model_lock_sha256(request.model_lock_path)
        descriptor = descriptor_for_lock(lock)
        inference_config = build_idm_inference_config(
            model_lock_sha256,
            descriptor.sha256,
            input_view_id=IDM_FULL_MIX_INPUT_VIEW_ID,
            timeout_seconds=IDM_REQUEST_TIMEOUT_SECONDS,
        )
        inference_config_sha = idm_inference_config_sha256(inference_config)
        run_id = _smoke_run_id(
            handoff=handoff,
            reference=reference,
            timing=timing,
            smoke_manifest_sha256=sha256_hex(smoke_content),
            descriptor_sha256=descriptor.sha256,
            model_lock_sha256=model_lock_sha256,
            inference_config_sha256_value=inference_config_sha,
            crux_commit=request.crux_commit,
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, IdmSmokeManifestError):
        return _fatal_full_mix_smoke()

    try:
        run_dir, run_path, reports_path, input_root = _validate_full_mix_smoke_namespace(
            request, run_id=run_id
        )
        mappings = preflight_reference_mappings(
            reference,
            timing,
            timing_output_root=request.timing_manifest_path.parent.parent,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        input_root.mkdir(parents=True, exist_ok=True)
        header: dict[str, object] = {
            "schema": IDM_FULL_MIX_SMOKE_RUN_SCHEMA,
            "run_id": run_id,
            "handoff_manifest_sha256": handoff.manifest_sha256,
            "handoff_manifest_version": handoff.corpus_version,
            "reference_manifest_sha256": reference.manifest_sha256,
            "reference_manifest_version": reference.corpus_version,
            "reference_timing_manifest_sha256": timing.manifest_sha256,
            "reference_timing_version": timing.corpus_version,
            "smoke_manifest_sha256": sha256_hex(smoke_content),
            "smoke_manifest": {
                "schema": smoke_manifest.schema,
                "cases": [
                    {"reason": case.reason, "simfile_id": case.simfile_id}
                    for case in smoke_manifest.cases
                ],
            },
            "backend_descriptor_sha256": descriptor.sha256,
            "backend_descriptor": dict(descriptor.payload),
            "model_id": lock.model_id,
            "model_lock_sha256": model_lock_sha256,
            "inference_config": dict(inference_config),
            "inference_config_sha256": inference_config_sha,
            "input_view_id": IDM_FULL_MIX_INPUT_VIEW_ID,
            "request_timeout_seconds": inference_config.get("request_timeout_seconds"),
            "worker_close_timeout_seconds": int(IDM_WORKER_CLOSE_TIMEOUT_SECONDS),
            "crux_commit": request.crux_commit,
            "started_at": _timestamp(clock()),
        }
        _write_full_mix_smoke_snapshot(
            run_path,
            {**header, "items": [], "overall_status": "partial"},
        )
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, IdmSmokeManifestError):
        return _fatal_full_mix_smoke()

    handoff_by_id = {int(row["simfile_id"]): row for row in handoff.rows}
    reference_rows = {row.view.simfile_id: row.source_row for row in reference.rows}
    cache_index: CacheIndexStore
    try:
        cache_index = CacheIndexStore.load(request.source_cache_dir)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _fatal_full_mix_smoke()

    items: list[dict[str, object]] = [
        {
            "simfile_id": case.simfile_id,
            "reason": case.reason,
            "execution_disposition": None,
        }
        for case in smoke_manifest.cases
    ]
    audio_by_id: dict[int, CanonicalAudio] = {}
    backend: Any | None = None
    backend_poisoned = False
    close_error: str | None = None

    try:
        for item in items:
            simfile_id = int(item["simfile_id"])
            row = handoff_by_id.get(simfile_id)
            mapping = mappings.get(simfile_id)
            if row is None or mapping is None:
                item["execution_disposition"] = "quarantined"
                item["cohort_failure_reason"] = "reference_quarantined"
                item["failure_detail"] = "smoke member has no loaded handoff/reference row"
                continue
            source_row = reference_rows.get(simfile_id)
            if source_row is None:
                _set_full_mix_smoke_failed(
                    item,
                    "retained_input_invalid",
                    "smoke member has no authoritative source row",
                )
                continue
            try:
                source = resolve_source_audio(
                    source_row,
                    request.source_cache_dir,
                    cache_index,
                    **_smoke_source_kwargs(source_row),
                    load_body=True,
                )
                if source.source_audio_id != row.get(
                    "source_audio_id"
                ) or source.source_audio_sha256 != row.get("source_audio_sha256"):
                    raise IdmSmokeManifestError("resolved source identity differs from handoff")
                canonical_path = input_root / str(simfile_id) / "full-mix.wav"
                audio = materialize_idm_full_mix_audio(
                    source,
                    canonical_path,
                    input_root=input_root,
                )
                item.update(
                    {
                        "source_audio_id": source.source_audio_id,
                        "source_audio_sha256": source.source_audio_sha256,
                        "source_duration_sec": source.duration_sec,
                        "input_view_id": audio.input_view_id,
                        "input_audio_sha256": audio.input_audio_sha256,
                    }
                )
                audio_by_id[simfile_id] = audio
            except (OSError, RuntimeError, TypeError, ValueError, IdmSmokeManifestError) as error:
                _set_full_mix_smoke_failed(item, "retained_input_invalid", error)
                continue

            target = prediction_path(
                run_dir,
                simfile_id=simfile_id,
                source_audio_sha256=audio.source_audio_sha256,
                backend_descriptor_sha256=descriptor.sha256,
                inference_config_sha256=inference_config_sha,
            )
            try:
                _validate_full_mix_smoke_prediction_target(run_dir, request.output_dir, target)
            except (OSError, RuntimeError, TypeError, ValueError, IdmSmokeManifestError):
                return _fatal_full_mix_smoke()
            if target.exists():
                _set_full_mix_smoke_failed(
                    item,
                    "prediction_output_conflict",
                    "full-mix smoke prediction already exists",
                )
                continue
            if backend_poisoned:
                _set_full_mix_smoke_failed(
                    item,
                    "worker_protocol_failed",
                    "inference was not attempted after a poison failure",
                )
                continue
            if backend is None:
                factory = backend_factory or IdmBackend
                try:
                    backend = factory(
                        runtime_python=request.runtime_python,
                        model_lock_path=request.model_lock_path,
                        model_root=request.model_root,
                        input_root=input_root,
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
                    code, disposition = classify_idm_backend_error(
                        getattr(error, "code", "worker_start_failed")
                    )
                    _set_full_mix_smoke_failed(item, code, error)
                    if disposition == "poison":
                        backend_poisoned = True
                    continue
            try:
                started = perf_counter()
                native = backend.transcribe(audio)
                elapsed = max(0.0, perf_counter() - started)
                if (
                    not isinstance(native, NativePrediction)
                    or native.audio != audio
                    or native.descriptor != descriptor
                ):
                    raise IdmBackendError(
                        "IDM backend returned an invalid prediction", code="native_event_invalid"
                    )
                mapped, _diagnostics = map_idm_prediction(native)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                code, disposition = classify_idm_backend_error(_backend_failure_code(error))
                _set_full_mix_smoke_failed(item, code, error)
                if disposition == "poison":
                    backend_poisoned = True
                continue
            try:
                published = publish_prediction_artifact(target, mapped)
            except ArtifactPublicationError as error:
                _set_full_mix_smoke_failed(item, "prediction_publish_failed", error)
                continue
            except BaseException as error:  # pylint: disable=broad-exception-caught
                _set_full_mix_smoke_failed(item, "prediction_artifact_invalid", error)
                continue
            item.update(
                {
                    "execution_disposition": "inferred",
                    "prediction_path": target.relative_to(run_dir).as_posix(),
                    "prediction_artifact_sha256": published.sha256,
                    "wall_time_sec": elapsed,
                }
            )
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:  # pylint: disable=broad-exception-caught
                close_error = _bounded_error(error)

    identity = CohortIdentity(
        cohort_id=f"{run_id}:full-mix-smoke",
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=IDM_BACKEND_ID,
        model_id=lock.model_id,
        model_lock_sha256=model_lock_sha256,
        backend_descriptor_sha256=descriptor.sha256,
        prediction_map_version=IDM_PREDICTION_MAP_ID,
        input_view_id=IDM_FULL_MIX_INPUT_VIEW_ID,
    )
    cohort_items: list[CohortItem] = []
    try:
        for item in items:
            simfile_id = int(item["simfile_id"])
            reference_mapping = mappings.get(simfile_id)
            if item.get("execution_disposition") == "inferred":
                prediction_path_value = item.get("prediction_path")
                audio = audio_by_id.get(simfile_id)
                if not isinstance(prediction_path_value, str) or audio is None:
                    raise ValueError("full-mix smoke success item is incomplete")
                artifact_path = _owned_path(
                    run_dir, prediction_path_value, "full-mix smoke prediction"
                )
                content = read_regular_file_no_follow(artifact_path)
                artifact = read_prediction_artifact(content)
                if not prediction_artifact_matches_audio(
                    artifact,
                    source_audio_id=audio.source_audio_id,
                    source_audio_sha256=audio.source_audio_sha256,
                    audio=audio,
                    descriptor=descriptor,
                    prediction_map_version=IDM_PREDICTION_MAP_ID,
                ):
                    raise ValueError("full-mix smoke prediction identity mismatch")
                cohort_items.append(
                    cohort_item_from_validated_prediction_artifact(
                        identity,
                        str(simfile_id),
                        reference_mapping,
                        artifact,
                    )
                )
            elif item.get("execution_disposition") == "quarantined":
                cohort_items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        None,
                        status="quarantined",
                        failure_reason="reference_quarantined",
                    )
                )
            else:
                reason = item.get("cohort_failure_reason", "inference_failed")
                if not isinstance(reason, str) or reason not in COHORT_FAILURE_REASONS:
                    reason = "inference_failed"
                cohort_items.append(
                    cohort_item_without_prediction(
                        identity,
                        str(simfile_id),
                        reference_mapping,
                        status="failed",
                        failure_reason=reason,  # type: ignore[arg-type]
                    )
                )
        score = score_cohort(identity, tuple(cohort_items), diagnostics_for=())
        write_cohort_reports(score, reports_path)
        counts = score.population
        status = (
            "complete" if counts.failed_count == 0 and counts.quarantined_count == 0 else "partial"
        )
        if close_error is not None and status == "complete":
            status = "partial"
        snapshot = {
            **header,
            "items": sorted(items, key=lambda item: int(item["simfile_id"])),
            "success_count": counts.success_count,
            "failed_count": counts.failed_count,
            "skipped_count": counts.skipped_count,
            "quarantined_count": counts.quarantined_count,
            "overall_status": status,
            "completed_at": _timestamp(clock()) if status == "complete" else None,
        }
        if close_error is not None:
            snapshot["close_error"] = close_error
        snapshot = {key: value for key, value in snapshot.items() if value is not None}
        _write_full_mix_smoke_snapshot(run_path, snapshot)
    except (OSError, RuntimeError, TypeError, ValueError, StrictJsonError, PredictionArtifactError):
        return _fatal_full_mix_smoke()
    return IdmFullMixSmokeOutcome(
        overall_status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "complete" else 1,
        run_id=run_id,
        run_path=run_path,
        reports_path=reports_path,
        success_count=counts.success_count,
        failed_count=counts.failed_count,
        skipped_count=counts.skipped_count,
        quarantined_count=counts.quarantined_count,
    )


__all__ = [
    "IDM_FAILURE_TO_COHORT_REASON",
    "IDM_FULL_MIX_INPUT_VIEW_ID",
    "IDM_FULL_MIX_SMOKE_DIRNAME",
    "IDM_FULL_MIX_SMOKE_REPORT_DIRNAME",
    "IDM_FULL_MIX_SMOKE_RUN_SCHEMA",
    "IDM_PILOT_RUN_SCHEMA",
    "IDM_SMOKE_CASE_ORDER",
    "IDM_SMOKE_SCHEMA",
    "IDM_STEM_INPUT_VIEW_ID",
    "IdmFullMixSmokeOutcome",
    "IdmFullMixSmokeRequest",
    "IdmPilotRunError",
    "IdmPilotRunOutcome",
    "IdmPilotRunRequest",
    "IdmSmokeCase",
    "IdmSmokeManifest",
    "IdmSmokeManifestError",
    "build_idm_cohort_from_snapshot",
    "build_idm_inference_config",
    "build_oaf_cohort_from_handoff",
    "build_run_id",
    "classify_idm_backend_error",
    "compute_model_lock_sha256",
    "idm_inference_config_sha256",
    "load_idm_smoke_manifest",
    "materialize_idm_full_mix_audio",
    "parse_idm_pilot_run",
    "parse_idm_smoke_manifest",
    "render_idm_pilot_run",
    "render_idm_smoke_manifest",
    "run_idm_full_mix_smoke",
    "run_idm_pilot",
    "select_idm_smoke_cases",
    "validate_idm_smoke_manifest",
    "write_idm_smoke_manifest",
    "write_idm_pilot_run",
]
