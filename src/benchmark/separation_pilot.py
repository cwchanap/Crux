"""HPA-328 fixed-subset preflight and separation-run ledger.

Task 5 deliberately keeps this boundary concrete.  It validates the four
already-published input artifacts, attests the two fixed separator runtimes,
binds the derived views to the persisted OaF control, writes the local HPA-328
run identity, and obtains the full-mix control through the public reviewed-
subset scorer.
"""

# This concrete task boundary intentionally keeps request/ledger state and
# preflight checks together; splitting it into a framework would violate the
# HPA-328 scope.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-branches,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines,redefined-outer-name,too-many-return-statements

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    PublishedArtifact,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
    quantize_six,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio, NativePrediction
from src.benchmark.backends.oaf import OafBackendError, create_backend
from src.benchmark.cohort_scoring import (
    SCORING_VERSION,
    CohortIdentity,
    cohort_item_from_validated_prediction_artifact,
    cohort_item_without_prediction,
    score_cohort,
)
from src.benchmark.corpus_cache import CacheIndexStore, ResolvedSourceAudio, resolve_source_audio
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.input_view import materialize_derived_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_PREDICTION_MAP_ID,
    OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
    _model_lock_path,
    build_run_id,
    classify_oaf_backend_error,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_oaf_corpus_run,
)
from src.benchmark.prediction_artifact import (
    PredictionArtifactError,
    prediction_artifact_matches_audio,
    prediction_artifact_matches_run_row,
    prediction_path,
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.r2_corpus_models import format_manifest_timestamp, parse_manifest_timestamp
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
from src.benchmark.reviewed_subset import (
    LoadedReviewedSubsetManifest,
    ScoreReviewedSubsetOutcome,
    ScoreReviewedSubsetRequest,
    _source_row_sha256,
    load_reviewed_subset_manifest,
    score_oaf_reviewed_subset,
)
from src.benchmark.separation_comparison import (
    SeparationComparisonRequest,
    compare_oaf_separation,
)
from src.benchmark.separators import (
    ATTESTATION_FAILURE_CODES,
    HTDEMUCS_SEPARATOR_ID,
    SPLEETER_SEPARATOR_ID,
    AttestedSeparatorRuntime,
    SeparatedStem,
    SeparatorExecutionError,
    SeparatorLock,
    attest_separator_runtime,
    load_separator_lock,
    revalidate_separator_model_root,
    run_htdemucs_drums,
    run_spleeter_drums,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

SEPARATION_RUN_SCHEMA = "crux.oaf-separation-run/v1"
SPLEETER_INPUT_VIEW_ID = "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
HTDEMUCS_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

SEPARATION_FAILURE_TO_COHORT_REASON = {
    "separation_failed": "inference_failed",
    "stem_invalid": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "inference_failed": "inference_failed",
    "prediction_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RUN_ID_RE = re.compile(r"oaf-separation-[0-9a-f]{16}\Z")
_SEPARATION_STATUSES = frozenset(
    {
        "pending",
        "separation_failed",
        "stem_invalid",
        "inference_failed",
        "prediction_invalid",
        "success",
        "resumed",
    }
)
_PARENT_STATUSES = frozenset({"inferred", "resumed", "failed", "skipped", "quarantined"})
_HASH_FIELDS = (
    "reviewed_subset_manifest_sha256",
    "reference_manifest_sha256",
    "reference_timing_manifest_sha256",
    "parent_oaf_run_id",
    "oaf_backend_descriptor_sha256",
    "oaf_model_lock_sha256",
    "oaf_checkpoint_archive_sha256",
    "spleeter_lock_sha256",
    "htdemucs_lock_sha256",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
SEPARATOR_LOCK_PATHS: dict[str, Path] = {
    SPLEETER_SEPARATOR_ID: _REPO_ROOT / "runtime" / "separators" / "spleeter" / "model.json",
    HTDEMUCS_SEPARATOR_ID: _REPO_ROOT / "runtime" / "separators" / "htdemucs" / "model.json",
}

_COMPARISON_REPORT_NAMES = (
    "summary.json",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
    "event_diagnostics.jsonl",
    "summary.md",
)


PilotStatus = Literal["complete", "partial", "failed"]
PilotExitCode = Literal[0, 1, 2]


@dataclass(frozen=True)
class OafSeparationPilotRequest:
    """Fixed HPA-328 inputs.

    The complete HPA-327 subset is the population.  There are intentionally no
    sample, seed, count, include, or exclude controls here.
    """

    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    oaf_run_path: Path
    cache_dir: Path
    output_dir: Path
    spleeter_python: Path
    demucs_python: Path
    spleeter_model_root: Path
    demucs_model_root: Path
    resume: bool = False
    crux_commit: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "reference_manifest_path",
            "timing_manifest_path",
            "subset_manifest_path",
            "oaf_run_path",
            "cache_dir",
            "output_dir",
            "spleeter_python",
            "demucs_python",
            "spleeter_model_root",
            "demucs_model_root",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool")
        if self.crux_commit is not None and _COMMIT_RE.fullmatch(self.crux_commit) is None:
            raise ValueError("crux_commit must be a lowercase 40-character commit")


@dataclass(frozen=True)
class OafSeparationPilotOutcome:
    """Stable outcome for the preflight/control boundary.

    The derived-view counts remain zero until Task 6 populates the per-view
    ledgers.  The control counts are retained so a partial persisted OaF
    control remains visible to callers.
    """

    overall_status: PilotStatus
    exit_code: PilotExitCode
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    full_mix_reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    failure_code: str | None

    def __post_init__(self) -> None:
        if self.overall_status not in {"complete", "partial", "failed"}:
            raise ValueError("overall_status is invalid")
        if self.exit_code not in {0, 1, 2}:
            raise ValueError("exit_code is invalid")
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise TypeError("run_id must be a string or None")
        for field in ("run_path", "reports_path", "full_mix_reports_path"):
            if getattr(self, field) is not None and not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path or None")
        for field in ("success_count", "failed_count", "skipped_count", "quarantined_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str)
            or self.failure_code not in ATTESTATION_FAILURE_CODES
        ):
            raise ValueError("failure_code is invalid")


class SeparationRunError(ValueError):
    """A malformed or inconsistent HPA-328 run snapshot."""


def _fatal_outcome(failure_code: str | None = None) -> OafSeparationPilotOutcome:
    return OafSeparationPilotOutcome(
        overall_status="failed",
        exit_code=2,
        run_id=None,
        run_path=None,
        reports_path=None,
        full_mix_reports_path=None,
        success_count=0,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
        failure_code=failure_code,
    )


def _comparison_reports_ready(run_dir: Path) -> bool:
    """Gate Task 8B on the six published reports for every fixed view."""
    return all(
        (run_dir / "views" / view_name / "reports" / report_name).is_file()
        for view_name in ("full_mix", "spleeter", "htdemucs")
        for report_name in _COMPARISON_REPORT_NAMES
    )


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise SeparationRunError(f"{field} must be a lowercase SHA-256")
    try:
        return require_sha256(value, field)
    except StrictJsonError:
        raise SeparationRunError(f"{field} must be a lowercase SHA-256") from None


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SeparationRunError(f"{field} must be a nonempty string")
    return value


def _require_absolute_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SeparationRunError(f"{field} must be an absolute path")
    path = Path(value)
    try:
        if not path.is_absolute() or path != path.resolve():
            raise SeparationRunError(f"{field} must be an absolute path")
    except (OSError, RuntimeError) as error:
        raise SeparationRunError(f"{field} must be an absolute path") from error
    return value


def _require_crux_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise SeparationRunError("crux_commit must be a lowercase 40-character commit")
    return value


def _normalize_snapshot_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str, Decimal)):
        return value  # type: ignore[return-value]
    if isinstance(value, float):
        return quantize_six(value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJsonError("separation snapshot object keys must be strings")
            normalized[key] = _normalize_snapshot_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_snapshot_value(item) for item in value]
    raise StrictJsonError(f"unsupported separation snapshot value: {type(value).__name__}")


def _validate_evidence(value: object, field: str) -> None:
    if value is not None and not isinstance(value, Mapping):
        raise SeparationRunError(f"{field} evidence must be an object or null")


def _validate_view_row(value: object, *, field: str, derived: bool) -> None:
    if not isinstance(value, Mapping):
        raise SeparationRunError(f"{field} view evidence must be an object")
    status = value.get("status")
    allowed = _SEPARATION_STATUSES if derived else _PARENT_STATUSES
    if not isinstance(status, str) or status not in allowed:
        raise SeparationRunError(f"{field} view status is invalid")
    failure_code = value.get("failure_code")
    if failure_code is not None and (not isinstance(failure_code, str) or not failure_code):
        raise SeparationRunError(f"{field} failure_code is invalid")
    for evidence_name in ("stem", "input", "prediction", "runtime"):
        _validate_evidence(value.get(evidence_name), f"{field}.{evidence_name}")
    input_view_id = value.get("input_view_id")
    if derived:
        _require_hash(value.get("separator_lock_sha256"), f"{field}.separator_lock_sha256")
        expected = {
            "spleeter": SPLEETER_INPUT_VIEW_ID,
            "htdemucs": HTDEMUCS_INPUT_VIEW_ID,
        }.get(field)
        if expected is not None and input_view_id != expected:
            raise SeparationRunError(f"{field} input view identity is invalid")
    elif field == "full_mix" and input_view_id != OAF_FULL_MIX_INPUT_VIEW_ID:
        raise SeparationRunError("full_mix input view identity is invalid")

    input_evidence = value.get("input")
    if isinstance(input_evidence, Mapping):
        if input_evidence.get("input_view_id") != input_view_id:
            raise SeparationRunError(f"{field} input evidence view identity is invalid")
        input_sha = input_evidence.get("input_audio_sha256")
        if input_sha is not None:
            _require_hash(input_sha, f"{field}.input.input_audio_sha256")
    prediction_evidence = value.get("prediction")
    if isinstance(prediction_evidence, Mapping):
        prediction_path_value = prediction_evidence.get("path")
        if prediction_path_value is not None and (
            not isinstance(prediction_path_value, str) or not prediction_path_value
        ):
            raise SeparationRunError(f"{field} prediction path is invalid")
        prediction_sha = prediction_evidence.get("artifact_sha256")
        if prediction_sha is not None:
            _require_hash(prediction_sha, f"{field}.prediction.artifact_sha256")


def _validate_snapshot(snapshot: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if snapshot.get("schema") != SEPARATION_RUN_SCHEMA:
        raise SeparationRunError(f"run snapshot schema must be {SEPARATION_RUN_SCHEMA}")
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise SeparationRunError("run snapshot run_id is invalid")
    for field in _HASH_FIELDS:
        if field == "parent_oaf_run_id":
            _require_nonempty_string(snapshot.get(field), field)
        else:
            _require_hash(snapshot.get(field), field)
    for field in (
        "reference_manifest_version",
        "reference_timing_version",
        "spleeter_input_view_id",
        "htdemucs_input_view_id",
        "scoring_version",
    ):
        _require_nonempty_string(snapshot.get(field), field)
    if snapshot["spleeter_input_view_id"] != SPLEETER_INPUT_VIEW_ID:
        raise SeparationRunError("spleeter input view identity is invalid")
    if snapshot["htdemucs_input_view_id"] != HTDEMUCS_INPUT_VIEW_ID:
        raise SeparationRunError("htdemucs input view identity is invalid")
    if snapshot["scoring_version"] != SCORING_VERSION:
        raise SeparationRunError("scoring_version is invalid")
    _require_absolute_path(snapshot.get("parent_oaf_run_path"), "parent_oaf_run_path")
    _require_crux_commit(snapshot.get("crux_commit"))
    started_at = snapshot.get("started_at")
    try:
        parse_manifest_timestamp(started_at)
    except ValueError as error:
        raise SeparationRunError("started_at is invalid") from error

    identity = snapshot.get("run_identity")
    if not isinstance(identity, Mapping):
        raise SeparationRunError("run_identity is missing")
    expected_identity = {
        key: snapshot[key]
        for key in (
            "schema",
            "reviewed_subset_manifest_sha256",
            "reference_manifest_sha256",
            "reference_timing_manifest_sha256",
            "parent_oaf_run_id",
            "parent_oaf_run_path",
            "oaf_backend_descriptor_sha256",
            "oaf_model_lock_sha256",
            "oaf_checkpoint_archive_sha256",
            "spleeter_lock_sha256",
            "htdemucs_lock_sha256",
            "spleeter_input_view_id",
            "htdemucs_input_view_id",
            "scoring_version",
            "crux_commit",
        )
    }
    if dict(identity) != expected_identity:
        raise SeparationRunError("run_identity does not match snapshot header")
    expected_run_id = (
        "oaf-separation-" + sha256(canonical_json_bytes(expected_identity)).hexdigest()[:16]
    )
    if run_id != expected_run_id:
        raise SeparationRunError("run_id does not match frozen identity")

    full_mix_reports_path = snapshot.get("full_mix_reports_path")
    if full_mix_reports_path != "views/full_mix/reports":
        raise SeparationRunError("full_mix_reports_path is invalid")
    overall_status = snapshot.get("overall_status")
    if overall_status not in {"pending", "complete", "partial", "failed"}:
        raise SeparationRunError("overall_status is invalid")

    items = snapshot.get("items")
    if not isinstance(items, list):
        raise SeparationRunError("run snapshot items must be an array")
    if not 20 <= len(items) <= 30:
        raise SeparationRunError("run snapshot must contain the exact HPA-327 population")
    previous_id = 0
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise SeparationRunError("run snapshot item must be an object")
        simfile_id = item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int) or simfile_id <= 0:
            raise SeparationRunError("run snapshot item simfile_id is invalid")
        if simfile_id in seen or simfile_id < previous_id:
            raise SeparationRunError("run snapshot items must be unique and sorted")
        seen.add(simfile_id)
        previous_id = simfile_id
        _require_hash(item.get("source_row_sha256"), "source_row_sha256")
        _require_nonempty_string(item.get("source_audio_id"), "source_audio_id")
        _require_hash(item.get("source_audio_sha256"), "source_audio_sha256")
        duration = item.get("source_duration_sec")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float, Decimal))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise SeparationRunError("source_duration_sec is invalid")
        _validate_view_row(item.get("full_mix"), field="full_mix", derived=False)
        _validate_view_row(item.get("spleeter"), field="spleeter", derived=True)
        _validate_view_row(item.get("htdemucs"), field="htdemucs", derived=True)
        if not isinstance(item["spleeter"], Mapping) or not isinstance(item["htdemucs"], Mapping):
            raise SeparationRunError("derived view evidence is invalid")
        if item["spleeter"].get("separator_lock_sha256") != snapshot["spleeter_lock_sha256"]:
            raise SeparationRunError("Spleeter lock identity is inconsistent")
        if item["htdemucs"].get("separator_lock_sha256") != snapshot["htdemucs_lock_sha256"]:
            raise SeparationRunError("HTDemucs lock identity is inconsistent")
    return snapshot


def render_oaf_separation_run(snapshot: Mapping[str, object]) -> bytes:
    """Render one strict canonical HPA-328 run snapshot."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("run snapshot must be a mapping")
    normalized = _normalize_snapshot_value(snapshot)
    if not isinstance(normalized, dict):
        raise SeparationRunError("run snapshot must be an object")
    return canonical_json_bytes(_validate_snapshot(normalized))


def parse_oaf_separation_run(
    content: bytes,
    *,
    expected_run_id: str | None = None,
) -> dict[str, JsonValue]:
    """Parse one canonical HPA-328 run snapshot without executing anything."""
    value = strict_json_loads(content, require_canonical=True)
    if not isinstance(value, dict):
        raise SeparationRunError("run snapshot must be an object")
    validated = _validate_snapshot(value)
    if canonical_json_bytes(validated) != content:
        raise SeparationRunError("run snapshot is not semantically canonical")
    if expected_run_id is not None and validated.get("run_id") != expected_run_id:
        raise SeparationRunError("run snapshot run_id does not match expected identity")
    return validated


def write_oaf_separation_run(run_path: Path, snapshot: Mapping[str, object]) -> None:
    """Durably replace one HPA-328 run snapshot."""
    if not isinstance(run_path, Path):
        raise TypeError("run_path must be a Path")
    run_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(run_path, render_oaf_separation_run(snapshot))


# Short aliases make the local snapshot seam convenient to call in later HPA-328
# tasks without introducing a generic run-snapshot module.
render_separation_run = render_oaf_separation_run
parse_separation_run = parse_oaf_separation_run
write_separation_run = write_oaf_separation_run


def _load_separator_locks() -> tuple[SeparatorLock, SeparatorLock]:
    try:
        spleeter_path = SEPARATOR_LOCK_PATHS[SPLEETER_SEPARATOR_ID]
        htdemucs_path = SEPARATOR_LOCK_PATHS[HTDEMUCS_SEPARATOR_ID]
    except (KeyError, TypeError):
        raise SeparationRunError("separator lock paths are incomplete") from None
    try:
        spleeter = load_separator_lock(spleeter_path)
        htdemucs = load_separator_lock(htdemucs_path)
    except (OSError, TypeError, ValueError) as error:
        raise SeparationRunError("separator lock is invalid") from error
    if spleeter.separator_id != SPLEETER_SEPARATOR_ID:
        raise SeparationRunError("Spleeter separator lock identity is invalid")
    if htdemucs.separator_id != HTDEMUCS_SEPARATOR_ID:
        raise SeparationRunError("HTDemucs separator lock identity is invalid")
    if spleeter.sha256 == htdemucs.sha256:
        raise SeparationRunError("separator locks must be distinct")
    return spleeter, htdemucs


def _read_parent_run(path: Path) -> dict[str, JsonValue]:
    try:
        content = read_regular_file_no_follow(path)
        return parse_oaf_corpus_run(content)
    except (OSError, TypeError, StrictJsonError, ValueError) as error:
        raise SeparationRunError("parent OaF run snapshot is invalid") from error


def _validate_parent_identity(
    parent: Mapping[str, object],
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
) -> None:
    if parent.get("schema") != OAF_CORPUS_RUN_SCHEMA:
        raise SeparationRunError("parent run schema is invalid")
    if parent.get("reference_manifest_sha256") != reference.manifest_sha256:
        raise SeparationRunError("parent run reference manifest does not match")
    if parent.get("reference_manifest_version") != reference.corpus_version:
        raise SeparationRunError("parent run reference version does not match")
    if parent.get("reference_timing_manifest_sha256") != timing.manifest_sha256:
        raise SeparationRunError("parent run timing manifest does not match")
    if parent.get("reference_timing_version") != timing.corpus_version:
        raise SeparationRunError("parent run timing version does not match")
    _require_nonempty_string(parent.get("run_id"), "parent run_id")
    for field in (
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
        "inference_config_sha256",
    ):
        _require_hash(parent.get(field), f"parent {field}")
    descriptor = parent.get("backend_descriptor")
    if not isinstance(descriptor, Mapping):
        raise SeparationRunError("parent OaF descriptor is unavailable")
    try:
        normalized_descriptor = normalize_known_backend_descriptor(descriptor)
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SeparationRunError("parent OaF descriptor is invalid") from error
    if (
        sha256(canonical_json_bytes(normalized_descriptor)).hexdigest()
        != parent["backend_descriptor_sha256"]
    ):
        raise SeparationRunError("parent OaF descriptor hash is invalid")
    if parent.get("input_view_id") != OAF_FULL_MIX_INPUT_VIEW_ID:
        raise SeparationRunError("parent OaF input view is not the full-mix control")
    config = parent.get("inference_config")
    if not isinstance(config, Mapping) or config.get("input_view_id") != OAF_FULL_MIX_INPUT_VIEW_ID:
        raise SeparationRunError("parent inference config has mixed OaF identity")
    for field in (
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
    ):
        if config.get(field) != parent.get(field):
            raise SeparationRunError("parent inference config has mixed OaF identity")
    config_sha = parent.get("inference_config_sha256")
    if not isinstance(config_sha, str):
        raise SeparationRunError("parent inference config identity is invalid")
    try:
        if inference_config_sha256(config) != config_sha:
            raise SeparationRunError("parent inference config hash is invalid")
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SeparationRunError("parent inference config is invalid") from error
    try:
        expected_parent_run_id = build_run_id(
            reference.manifest_sha256,
            timing.manifest_sha256,
            parent["backend_descriptor_sha256"],
            parent["model_lock_sha256"],
            parent["checkpoint_archive_sha256"],
            config_sha,
            parent.get("include_simfile_ids", ()),
            parent.get("exclude_simfile_ids", ()),
        )
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SeparationRunError("parent run identity inputs are invalid") from error
    if parent["run_id"] != expected_parent_run_id:
        raise SeparationRunError("parent run_id does not match deterministic OaF identity")
    for item in parent.get("items", []):
        if not isinstance(item, Mapping):
            raise SeparationRunError("parent run item is invalid")
        row_view = item.get("input_view_id")
        if row_view is not None and row_view != OAF_FULL_MIX_INPUT_VIEW_ID:
            raise SeparationRunError("parent run contains mixed OaF input views")
        row_config = item.get("inference_config_sha256")
        if row_config is not None and row_config != config_sha:
            raise SeparationRunError("parent run contains mixed OaF inference identity")
        row_descriptor = item.get("backend_descriptor_sha256")
        if row_descriptor is not None and row_descriptor != parent["backend_descriptor_sha256"]:
            raise SeparationRunError("parent run contains mixed OaF descriptor identity")
        row_model = item.get("model_lock_sha256")
        if row_model is not None and row_model != parent["model_lock_sha256"]:
            raise SeparationRunError("parent run contains mixed OaF model identity")
        row_checkpoint = item.get("checkpoint_archive_sha256")
        if row_checkpoint is not None and row_checkpoint != parent["checkpoint_archive_sha256"]:
            raise SeparationRunError("parent run contains mixed OaF checkpoint identity")


def _validate_output_paths(request: OafSeparationPilotRequest) -> None:
    output = request.output_dir.resolve()
    run_parent = request.oaf_run_path.resolve().parent
    protected = {
        request.oaf_run_path.resolve(),
        run_parent,
        run_parent / "reports",
        request.reference_manifest_path.resolve(),
        request.timing_manifest_path.resolve(),
        request.subset_manifest_path.resolve(),
        request.cache_dir.resolve(),
    }
    if output in protected:
        raise SeparationRunError("HPA-328 output directory aliases an input or parent run")
    # Keep this containment check lexical: Task 4 treats model roots as caller-
    # supplied absolute paths and must not follow symlinks while comparing them.
    lexical_output = Path(os.path.abspath(os.fspath(request.output_dir)))
    for model_root in (request.spleeter_model_root, request.demucs_model_root):
        lexical_model_root = Path(os.path.abspath(os.fspath(model_root)))
        if lexical_output == lexical_model_root or lexical_model_root in lexical_output.parents:
            raise SeparationRunError("HPA-328 output directory aliases a separator model root")


def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> None:
    """Bind every HPA-327 row to the corresponding HPA-324 reference row."""
    reference_rows = {loaded.view.simfile_id: loaded.source_row for loaded in reference.rows}
    for loaded in subset.rows:
        reference_row = reference_rows.get(loaded.view.simfile_id)
        if reference_row is None:
            raise SeparationRunError("reviewed subset member is absent from reference manifest")
        for field in (
            "selected_chart_key",
            "selected_chart_content_hash",
            "source_audio_key",
            "source_audio_content_hash",
        ):
            if loaded.source_row[field] != reference_row[field]:
                raise SeparationRunError("reviewed subset member reference identity is invalid")
        if loaded.source_row["source_row_sha256"] != _source_row_sha256(reference_row):
            raise SeparationRunError("reviewed subset source row identity is invalid")


def _subset_parent_rows(
    subset: LoadedReviewedSubsetManifest,
    parent: Mapping[str, object],
    *,
    spleeter_lock_sha256: str,
    htdemucs_lock_sha256: str,
) -> tuple[dict[str, object], ...]:
    raw_items = parent.get("items")
    if not isinstance(raw_items, list):
        raise SeparationRunError("parent run items are unavailable")
    parent_by_id: dict[int, Mapping[str, object]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise SeparationRunError("parent run item is invalid")
        simfile_id = raw_item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int):
            raise SeparationRunError("parent run item simfile_id is invalid")
        if simfile_id in parent_by_id:
            raise SeparationRunError("parent run contains duplicate simfile IDs")
        parent_by_id[simfile_id] = raw_item

    rows: list[dict[str, object]] = []
    for loaded in sorted(subset.rows, key=lambda row: row.view.simfile_id):
        simfile_id = loaded.view.simfile_id
        parent_row = parent_by_id.get(simfile_id)
        if parent_row is None:
            raise SeparationRunError("reviewed subset member is absent from parent run")
        source_id = parent_row.get("source_audio_id", loaded.source_row["source_audio_key"])
        reviewed_source_sha = loaded.source_row["source_audio_content_hash"]
        _require_hash(reviewed_source_sha, "reviewed source_audio_content_hash")
        if "source_audio_sha256" in parent_row:
            source_sha = parent_row["source_audio_sha256"]
            if source_sha != reviewed_source_sha:
                raise SeparationRunError("parent source audio identity does not match subset")
        else:
            source_sha = reviewed_source_sha
        if not isinstance(source_id, str) or not source_id:
            raise SeparationRunError("source audio ID is unavailable for subset member")
        _require_hash(source_sha, "source_audio_sha256")
        source_duration = parent_row.get(
            "source_audio_duration_sec", parent_row.get("source_duration_sec")
        )
        if source_duration is not None and (
            isinstance(source_duration, bool)
            or not isinstance(source_duration, (int, float, Decimal))
            or not math.isfinite(float(source_duration))
            or float(source_duration) <= 0
        ):
            raise SeparationRunError("source duration is invalid for subset member")

        parent_status = parent_row.get("execution_disposition")
        if not isinstance(parent_status, str) or parent_status not in _PARENT_STATUSES:
            raise SeparationRunError("parent run item status is invalid")
        parent_input_view_id = parent_row.get("input_view_id") or OAF_FULL_MIX_INPUT_VIEW_ID
        full_mix = {
            "status": parent_status,
            "failure_code": parent_row.get("runner_failure_code"),
            "input_view_id": parent_input_view_id,
            "stem": None,
            "input": (
                {
                    "input_view_id": parent_input_view_id,
                    "input_audio_sha256": parent_row.get("input_audio_sha256"),
                }
                if parent_row.get("input_audio_sha256") is not None
                else None
            ),
            "prediction": (
                {
                    "path": parent_row.get("prediction_path"),
                    "artifact_sha256": parent_row.get("prediction_artifact_sha256"),
                }
                if parent_row.get("prediction_path") is not None
                or parent_row.get("prediction_artifact_sha256") is not None
                else None
            ),
            "runtime": {
                key: parent_row[key]
                for key in (
                    "inference_elapsed_seconds",
                    "real_time_factor",
                )
                if key in parent_row
            },
        }
        rows.append(
            {
                "simfile_id": simfile_id,
                "source_row_sha256": loaded.source_row["source_row_sha256"],
                "source_audio_id": source_id,
                "source_audio_sha256": source_sha,
                "source_duration_sec": source_duration,
                "full_mix": full_mix,
                "spleeter": {
                    "status": "pending",
                    "failure_code": None,
                    "separator_lock_sha256": spleeter_lock_sha256,
                    "input_view_id": SPLEETER_INPUT_VIEW_ID,
                    "stem": None,
                    "input": None,
                    "prediction": None,
                    "runtime": None,
                },
                "htdemucs": {
                    "status": "pending",
                    "failure_code": None,
                    "separator_lock_sha256": htdemucs_lock_sha256,
                    "input_view_id": HTDEMUCS_INPUT_VIEW_ID,
                    "stem": None,
                    "input": None,
                    "prediction": None,
                    "runtime": None,
                },
            }
        )
    return tuple(rows)


def _identity_payload(
    request: OafSeparationPilotRequest,
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    subset: LoadedReviewedSubsetManifest,
    parent: Mapping[str, object],
    spleeter: SeparatorLock,
    htdemucs: SeparatorLock,
) -> dict[str, object]:
    return {
        "schema": SEPARATION_RUN_SCHEMA,
        "reviewed_subset_manifest_sha256": subset.manifest_sha256,
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_timing_manifest_sha256": timing.manifest_sha256,
        "parent_oaf_run_id": parent["run_id"],
        "parent_oaf_run_path": str(request.oaf_run_path.resolve()),
        "oaf_backend_descriptor_sha256": parent["backend_descriptor_sha256"],
        "oaf_model_lock_sha256": parent["model_lock_sha256"],
        "oaf_checkpoint_archive_sha256": parent["checkpoint_archive_sha256"],
        "spleeter_lock_sha256": spleeter.sha256,
        "htdemucs_lock_sha256": htdemucs.sha256,
        "spleeter_input_view_id": SPLEETER_INPUT_VIEW_ID,
        "htdemucs_input_view_id": HTDEMUCS_INPUT_VIEW_ID,
        "scoring_version": SCORING_VERSION,
        "crux_commit": request.crux_commit,
    }


def _build_snapshot(
    request: OafSeparationPilotRequest,
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    subset: LoadedReviewedSubsetManifest,
    parent: Mapping[str, object],
    spleeter: SeparatorLock,
    htdemucs: SeparatorLock,
    *,
    clock: Callable[[], datetime],
    run_id: str,
    rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    started_at = clock()
    if not isinstance(started_at, datetime):
        raise SeparationRunError("clock must return a datetime")
    identity = _identity_payload(request, reference, timing, subset, parent, spleeter, htdemucs)
    return {
        "schema": SEPARATION_RUN_SCHEMA,
        "run_id": run_id,
        "run_identity": identity,
        "reviewed_subset_manifest_sha256": subset.manifest_sha256,
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_manifest_version": reference.corpus_version,
        "reference_timing_manifest_sha256": timing.manifest_sha256,
        "reference_timing_version": timing.corpus_version,
        "parent_oaf_run_id": parent["run_id"],
        "parent_oaf_run_path": str(request.oaf_run_path.resolve()),
        "oaf_backend_descriptor_sha256": parent["backend_descriptor_sha256"],
        "oaf_model_lock_sha256": parent["model_lock_sha256"],
        "oaf_checkpoint_archive_sha256": parent["checkpoint_archive_sha256"],
        "spleeter_lock_sha256": spleeter.sha256,
        "htdemucs_lock_sha256": htdemucs.sha256,
        "spleeter_input_view_id": SPLEETER_INPUT_VIEW_ID,
        "htdemucs_input_view_id": HTDEMUCS_INPUT_VIEW_ID,
        "scoring_version": SCORING_VERSION,
        "crux_commit": request.crux_commit,
        "full_mix_reports_path": "views/full_mix/reports",
        "started_at": format_manifest_timestamp(started_at),
        "overall_status": "pending",
        "items": list(rows),
    }


def _recover_prior_derived_evidence(
    snapshot: dict[str, object], prior_snapshot: Mapping[str, object]
) -> None:
    """Carry the prior per-view ledger into the next durable snapshot."""
    current_items = snapshot.get("items")
    prior_items = prior_snapshot.get("items")
    if not isinstance(current_items, list) or not isinstance(prior_items, list):
        raise SeparationRunError("run snapshot items are unavailable")
    prior_by_id: dict[int, Mapping[str, object]] = {}
    for prior_item in prior_items:
        if not isinstance(prior_item, Mapping) or not isinstance(prior_item.get("simfile_id"), int):
            raise SeparationRunError("prior run snapshot item is invalid")
        prior_by_id[prior_item["simfile_id"]] = prior_item
    for item in current_items:
        if not isinstance(item, dict) or not isinstance(item.get("simfile_id"), int):
            raise SeparationRunError("run snapshot item is invalid")
        prior_item = prior_by_id.get(item["simfile_id"])
        if prior_item is None:
            raise SeparationRunError("prior run snapshot membership is incomplete")
        for field in ("source_row_sha256", "source_audio_id", "source_audio_sha256"):
            if prior_item.get(field) != item.get(field):
                raise SeparationRunError("prior run source identity does not match")
        for view_name in ("spleeter", "htdemucs"):
            prior_view = prior_item.get(view_name)
            if not isinstance(prior_view, Mapping):
                raise SeparationRunError("prior derived view evidence is invalid")
            recovered_view = dict(prior_view)
            for evidence_name in ("stem", "input", "prediction", "runtime"):
                evidence = prior_view.get(evidence_name)
                if isinstance(evidence, Mapping):
                    recovered_view[evidence_name] = dict(evidence)
            item[view_name] = recovered_view


def _capture_derived_view_preimages(
    snapshot: Mapping[str, object],
) -> dict[tuple[int, str], dict[str, object]]:
    """Deep-copy each mutable derived view before this invocation can change it."""
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise SeparationRunError("run snapshot items are unavailable")
    preimages: dict[tuple[int, str], dict[str, object]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise SeparationRunError("run snapshot item is invalid")
        simfile_id = raw_item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int):
            raise SeparationRunError("run snapshot item simfile_id is invalid")
        for view_name in ("spleeter", "htdemucs"):
            raw_view = raw_item.get(view_name)
            if not isinstance(raw_view, Mapping):
                raise SeparationRunError("derived view evidence is invalid")
            preimages[(simfile_id, view_name)] = deepcopy(dict(raw_view))
    return preimages


def _restore_derived_view_preimages(
    snapshot: dict[str, object],
    preimages: Mapping[tuple[int, str], Mapping[str, object]],
) -> None:
    """Restore only derived views, preserving all immutable artifacts and rows."""
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise SeparationRunError("run snapshot items are unavailable")
    items_by_id: dict[int, dict[str, object]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise SeparationRunError("run snapshot item is invalid")
        simfile_id = raw_item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int):
            raise SeparationRunError("run snapshot item simfile_id is invalid")
        items_by_id[simfile_id] = raw_item
    for (simfile_id, view_name), preimage in preimages.items():
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int):
            raise SeparationRunError("derived view preimage simfile_id is invalid")
        if view_name not in {"spleeter", "htdemucs"}:
            raise SeparationRunError("derived view preimage name is invalid")
        if not isinstance(preimage, Mapping):
            raise SeparationRunError("derived view preimage is invalid")
        item = items_by_id.get(simfile_id)
        if item is None:
            raise SeparationRunError("derived view preimage item is unavailable")
        item[view_name] = deepcopy(dict(preimage))


def _revalidate_separator_runtimes(
    runtimes: Mapping[str, AttestedSeparatorRuntime],
) -> None:
    """Revalidate every successful-preflight separator runtime."""
    first_error: SeparatorExecutionError | None = None
    for runtime in runtimes.values():
        try:
            revalidate_separator_model_root(runtime)
        except SeparatorExecutionError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _persist_restored_failed_snapshot(
    run_path: Path,
    snapshot: dict[str, object],
    preimages: Mapping[tuple[int, str], Mapping[str, object]],
) -> None:
    """Persist a model-root-drift snapshot without newly attributed evidence."""
    _restore_derived_view_preimages(snapshot, preimages)
    snapshot["overall_status"] = "failed"
    write_oaf_separation_run(run_path, snapshot)


def _score_full_mix_control(
    request: OafSeparationPilotRequest,
    run_dir: Path,
) -> ScoreReviewedSubsetOutcome:
    reports_path = run_dir / "views" / "full_mix" / "reports"
    outcome = score_oaf_reviewed_subset(
        ScoreReviewedSubsetRequest(
            reference_manifest_path=request.reference_manifest_path,
            timing_manifest_path=request.timing_manifest_path,
            subset_manifest_path=request.subset_manifest_path,
            run_path=request.oaf_run_path,
            output_dir=reports_path,
        )
    )
    if not isinstance(outcome, ScoreReviewedSubsetOutcome):
        raise SeparationRunError("full-mix control returned an invalid outcome")
    if outcome.exit_code not in {0, 1}:
        raise SeparationRunError("full-mix reviewed-subset control failed fatally")
    return outcome


def _derived_cohort_identity(
    snapshot: Mapping[str, object],
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    descriptor: BackendDescriptor,
    *,
    input_view_id: str,
) -> CohortIdentity:
    """Build one HPA-325 identity for a fixed derived input view."""
    backend_id = descriptor.payload.get("backend_id")
    model_id = descriptor.payload.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        model_id = snapshot.get("model_id")
    if not isinstance(backend_id, str) or not backend_id:
        raise SeparationRunError("derived cohort backend identity is unavailable")
    if not isinstance(model_id, str) or not model_id:
        raise SeparationRunError("derived cohort model identity is unavailable")
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SeparationRunError("derived cohort parent run identity is unavailable")
    cohort_id = sha256(
        canonical_json_bytes(
            {
                "parent_oaf_run_id": run_id,
                "input_view_id": input_view_id,
            }
        )
    ).hexdigest()
    return CohortIdentity(
        cohort_id=cohort_id,
        reference_manifest_sha256=reference.manifest_sha256,
        reference_timing_version=timing.corpus_version,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=backend_id,
        model_id=model_id,
        model_lock_sha256=_require_hash(
            snapshot.get("oaf_model_lock_sha256"), "oaf_model_lock_sha256"
        ),
        backend_descriptor_sha256=_require_hash(
            snapshot.get("oaf_backend_descriptor_sha256"),
            "oaf_backend_descriptor_sha256",
        ),
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id=input_view_id,
    )


def _derived_failure_reason(status: object, failure_code: object) -> str:
    if isinstance(failure_code, str):
        reason = SEPARATION_FAILURE_TO_COHORT_REASON.get(failure_code)
        if reason is not None:
            return reason
    if isinstance(status, str):
        reason = SEPARATION_FAILURE_TO_COHORT_REASON.get(status)
        if reason is not None:
            return reason
        if status in {"pending", "separation_failed", "stem_invalid", "inference_failed"}:
            return SEPARATION_FAILURE_TO_COHORT_REASON["inference_failed"]
        if status == "prediction_invalid":
            return SEPARATION_FAILURE_TO_COHORT_REASON["prediction_invalid"]
    raise SeparationRunError("derived view has no supported HPA-325 failure reason")


def _read_derived_prediction_artifact(
    request: OafSeparationPilotRequest,
    item: Mapping[str, object],
    view: Mapping[str, object],
    *,
    input_view_id: str,
) -> object:
    prediction = view.get("prediction")
    if not isinstance(prediction, Mapping):
        raise PredictionArtifactError("derived prediction evidence is missing")
    path_value = prediction.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PredictionArtifactError("derived prediction path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        path = request.output_dir / path
    artifact = read_prediction_artifact(read_regular_file_no_follow(path))
    if not prediction_artifact_matches_run_row(
        artifact,
        _prediction_row_from_view(item, view),
        expected_input_view_id=input_view_id,
    ):
        raise PredictionArtifactError("derived prediction does not match run evidence")
    return artifact


def _derived_cohort_item(
    identity: CohortIdentity,
    request: OafSeparationPilotRequest,
    item: Mapping[str, object],
    view: Mapping[str, object],
    mapping: ReferenceMappingResult | None,
    *,
    input_view_id: str,
) -> object:
    raw_simfile_id = item.get("simfile_id")
    if isinstance(raw_simfile_id, bool) or not isinstance(raw_simfile_id, int):
        raise SeparationRunError("derived cohort item simfile_id is invalid")
    simfile_id = str(raw_simfile_id)
    status = view.get("status")
    if status in {"success", "resumed"} and mapping is not None:
        try:
            artifact = _read_derived_prediction_artifact(
                request,
                item,
                view,
                input_view_id=input_view_id,
            )
            return cohort_item_from_validated_prediction_artifact(
                identity,
                simfile_id,
                mapping,
                artifact,  # type: ignore[arg-type]
            )
        except (OSError, PredictionArtifactError, StrictJsonError, TypeError, ValueError):
            failure_reason = SEPARATION_FAILURE_TO_COHORT_REASON["prediction_invalid"]
    elif status in {"success", "resumed"}:
        failure_reason = SEPARATION_FAILURE_TO_COHORT_REASON["prediction_invalid"]
    else:
        failure_reason = _derived_failure_reason(status, view.get("failure_code"))
    return cohort_item_without_prediction(
        identity,
        simfile_id,
        mapping,
        status="failed",
        failure_reason=failure_reason,  # type: ignore[arg-type]
    )


def _score_derived_cohort(
    request: OafSeparationPilotRequest,
    run_dir: Path,
    snapshot: Mapping[str, object],
    rows: tuple[dict[str, object], ...],
    mappings: Mapping[int, ReferenceMappingResult | None],
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    descriptor: BackendDescriptor,
    *,
    view_name: str,
    input_view_id: str,
) -> None:
    """Rebuild one complete derived population and publish HPA-325 reports."""
    identity = _derived_cohort_identity(
        snapshot,
        reference,
        timing,
        descriptor,
        input_view_id=input_view_id,
    )
    items: list[object] = []
    for item in rows:
        view = item.get(view_name)
        if not isinstance(view, Mapping):
            raise SeparationRunError("derived cohort view evidence is unavailable")
        simfile_id = item.get("simfile_id")
        if isinstance(simfile_id, bool) or not isinstance(simfile_id, int):
            raise SeparationRunError("derived cohort item simfile_id is invalid")
        items.append(
            _derived_cohort_item(
                identity,
                request,
                item,
                view,
                mappings.get(simfile_id),
                input_view_id=input_view_id,
            )
        )
    cohort_items = tuple(items)  # type: ignore[arg-type]
    successful_ids = tuple(item.simfile_id for item in cohort_items if item.status == "success")
    result = score_cohort(identity, cohort_items, diagnostics_for=successful_ids)
    write_cohort_reports(result, run_dir / "views" / view_name / "reports")


def _source_audio_kwargs(source_row: Mapping[str, object]) -> dict[str, str | None]:
    """Pass only the HPA-321 source identity fields to the cache resolver."""
    return {
        "source_audio_key": (
            source_row["source_audio_key"]
            if isinstance(source_row.get("source_audio_key"), str)
            else None
        ),
        "source_audio_content_hash": (
            source_row["source_audio_content_hash"]
            if isinstance(source_row.get("source_audio_content_hash"), str)
            else None
        ),
        "source_endpoint_sha256": (
            source_row["source_endpoint_sha256"]
            if isinstance(source_row.get("source_endpoint_sha256"), str)
            else None
        ),
        "source_bucket": (
            source_row["source_bucket"]
            if isinstance(source_row.get("source_bucket"), str)
            else None
        ),
    }


def _resolve_pilot_sources(
    request: OafSeparationPilotRequest,
    subset: LoadedReviewedSubsetManifest,
    rows: tuple[dict[str, object], ...],
) -> dict[int, ResolvedSourceAudio]:
    """Resolve every fixed member's authoritative source before execution."""
    cache_index = CacheIndexStore.load(request.cache_dir)
    loaded_by_id = {loaded.view.simfile_id: loaded for loaded in subset.rows}
    sources: dict[int, ResolvedSourceAudio] = {}
    for row in rows:
        simfile_id = row["simfile_id"]
        if not isinstance(simfile_id, int):
            raise SeparationRunError("pilot row simfile_id is invalid")
        loaded = loaded_by_id.get(simfile_id)
        if loaded is None:
            raise SeparationRunError("pilot row source member is unavailable")
        source = resolve_source_audio(
            loaded.source_row,
            request.cache_dir,
            cache_index,
            **_source_audio_kwargs(loaded.source_row),
            load_body=False,
        )
        if not isinstance(source, ResolvedSourceAudio):
            raise SeparationRunError("source resolver returned an invalid result")
        if source.source_audio_id != row.get(
            "source_audio_id"
        ) or source.source_audio_sha256 != row.get("source_audio_sha256"):
            raise SeparationRunError("resolved source identity does not match fixed membership")
        row["source_audio_id"] = source.source_audio_id
        row["source_audio_sha256"] = source.source_audio_sha256
        row["source_duration_sec"] = source.duration_sec
        sources[simfile_id] = source
    return sources


def _view_inference_config(
    parent: Mapping[str, object],
    input_view_id: str,
) -> tuple[dict[str, str], str]:
    """Derive one view config with the sole semantic change being its view ID."""
    full_mix = parent.get("inference_config")
    if not isinstance(full_mix, Mapping):
        raise SeparationRunError("parent inference config is unavailable")
    config = dict(full_mix)
    if not isinstance(input_view_id, str) or not input_view_id:
        raise SeparationRunError("derived input view identity is invalid")
    config["input_view_id"] = input_view_id
    try:
        config_sha = inference_config_sha256(config)
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SeparationRunError("derived inference config is invalid") from error
    return config, config_sha


def _relative_artifact_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _stem_evidence(stem: SeparatedStem, cache_root: Path) -> dict[str, object]:
    if not isinstance(stem, SeparatedStem):
        raise SeparationRunError("separator returned an invalid stem")
    return {
        "separator_id": stem.separator_id,
        "source_audio_sha256": stem.source_audio_sha256,
        "separator_lock_sha256": stem.separator_lock_sha256,
        "owner_root": str(cache_root.resolve()),
        "path": _relative_artifact_path(stem.path, cache_root),
        "sha256": stem.sha256,
        "cache_hit": stem.cache_hit,
        "qc": asdict(stem.qc),
        "warnings": list(stem.warnings),
    }


def _resolve_retained_stem(
    raw_stem: Mapping[str, object],
    *,
    cache_root: Path,
    source: ResolvedSourceAudio,
    lock: SeparatorLock,
) -> Path:
    path_value = raw_stem.get("path")
    digest = raw_stem.get("sha256", raw_stem.get("artifact_sha256"))
    if not isinstance(path_value, str) or not path_value:
        raise SeparatorExecutionError("stem_identity_invalid", "retained stem path is missing")
    if not isinstance(digest, str) or not digest:
        raise SeparatorExecutionError("stem_identity_invalid", "retained stem hash is missing")
    path = Path(path_value)
    if not path.is_absolute():
        path = cache_root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(cache_root.resolve())
    except (OSError, ValueError):
        raise SeparatorExecutionError(
            "stem_identity_invalid", "retained stem is outside the native cache"
        ) from None
    try:
        content = read_regular_file_no_follow(resolved)
    except (OSError, TypeError) as error:
        raise SeparatorExecutionError(
            "stem_identity_invalid", "retained stem is unreadable"
        ) from error
    if sha256(content).hexdigest() != digest:
        raise SeparatorExecutionError(
            "stem_identity_invalid", "retained stem bytes do not match checkpoint evidence"
        )
    if raw_stem.get("source_audio_sha256") != source.source_audio_sha256:
        raise SeparatorExecutionError(
            "stem_identity_invalid", "retained stem source identity does not match"
        )
    if raw_stem.get("separator_lock_sha256") != lock.sha256:
        raise SeparatorExecutionError(
            "stem_identity_invalid", "retained stem separator identity does not match"
        )
    return resolved


def _prediction_row_from_view(
    item: Mapping[str, object], view: Mapping[str, object]
) -> dict[str, object]:
    input_evidence = view.get("input")
    prediction_evidence = view.get("prediction")
    input_sha = (
        input_evidence.get("input_audio_sha256") if isinstance(input_evidence, Mapping) else None
    )
    prediction_sha = (
        prediction_evidence.get("artifact_sha256")
        if isinstance(prediction_evidence, Mapping)
        else None
    )
    return {
        "prediction_artifact_sha256": prediction_sha,
        "source_audio_id": item.get("source_audio_id"),
        "source_audio_sha256": item.get("source_audio_sha256"),
        "input_audio_sha256": input_sha,
        "input_view_id": view.get("input_view_id"),
    }


def _set_view_failure(
    view: dict[str, object],
    *,
    status: str,
    failure_code: str,
    runtime: Mapping[str, object] | None = None,
) -> None:
    view["status"] = status
    view["failure_code"] = failure_code
    if runtime is not None:
        view["runtime"] = dict(runtime)


def _separator_failure_status(code: str) -> str:
    if code in {
        "stem_decode_failed",
        "stem_channel_count",
        "stem_nonfinite",
        "stem_duration_invalid",
        "stem_duration_mismatch",
        "stem_near_silent",
        "stem_identity_invalid",
    }:
        return "stem_invalid"
    return "separation_failed"


def _close_backend(backend: object | None) -> None:
    if backend is None:
        return
    try:
        backend.close()  # type: ignore[attr-defined]
    except (OSError, RuntimeError, TypeError, ValueError):
        pass


def _validate_frozen_oaf_binding(parent: Mapping[str, object]) -> Path:
    """Bind OaF construction to the parent run's exact local model identity."""
    model_lock_sha256 = parent.get("model_lock_sha256")
    checkpoint_archive_sha256 = parent.get("checkpoint_archive_sha256")
    if not isinstance(model_lock_sha256, str) or not isinstance(checkpoint_archive_sha256, str):
        raise OafBackendError("parent OaF model identity is unavailable", code="descriptor_invalid")
    model_lock_path = _model_lock_path()
    try:
        current_lock_sha256 = compute_model_lock_sha256(model_lock_path)
        current_config = load_model_config(model_lock_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise OafBackendError(
            "parent OaF model lock cannot be verified", code="descriptor_invalid"
        ) from error
    if current_lock_sha256 != model_lock_sha256:
        raise OafBackendError(
            "current OaF model lock differs from the parent run",
            code="descriptor_invalid",
        )
    if current_config.checkpoint.archive_sha256 != checkpoint_archive_sha256:
        raise OafBackendError(
            "current OaF checkpoint differs from the parent run",
            code="descriptor_invalid",
        )
    cache_root = Path(
        os.environ.get("CRUX_OAF_CHECKPOINT_CACHE", "artifacts/benchmark/model-cache")
    )
    return cache_root / "sha256" / checkpoint_archive_sha256


def _bind_oaf_backend_factory(
    backend_factory: Callable[..., object],
    parent: Mapping[str, object],
    descriptor: BackendDescriptor,
) -> Callable[..., object]:
    """Forward frozen checkpoint and descriptor identity to one backend factory."""

    def bound_factory(**kwargs: object) -> object:
        bound_kwargs = dict(kwargs)
        bound_kwargs["checkpoint_dir"] = _validate_frozen_oaf_binding(parent)
        bound_kwargs["descriptor"] = descriptor
        return backend_factory(**bound_kwargs)

    return bound_factory


def _mark_outstanding_derived_views(snapshot: dict[str, object]) -> None:
    """Persist the native poison disposition for all views not yet attempted."""
    items = snapshot.get("items")
    if not isinstance(items, list):
        raise SeparationRunError("run snapshot items are unavailable")
    for item in items:
        if not isinstance(item, dict):
            raise SeparationRunError("run snapshot item is invalid")
        for view_name in ("spleeter", "htdemucs"):
            view = item.get(view_name)
            if not isinstance(view, dict):
                raise SeparationRunError("derived view evidence is invalid")
            if view.get("status") == "pending":
                runtime = view.get("runtime")
                _set_view_failure(
                    view,
                    status="inference_failed",
                    failure_code="worker_protocol_failed",
                    runtime=runtime if isinstance(runtime, Mapping) else None,
                )


def _execute_derived_view(
    request: OafSeparationPilotRequest,
    run_path: Path,
    run_dir: Path,
    snapshot: dict[str, object],
    item: dict[str, object],
    source: ResolvedSourceAudio,
    *,
    view_name: str,
    input_view_id: str,
    runtime: AttestedSeparatorRuntime,
    inference_config: Mapping[str, str],
    inference_config_sha: str,
    prior_item: Mapping[str, object] | None,
    separator_runner: Callable[..., object],
    backend_factory: Callable[..., object],
    backend: object | None,
    backend_ref: list[object | None],
    descriptor: BackendDescriptor,
    perf_counter: Callable[[], float],
    stop_disposition: list[str],
    separator_invocation_attempted: list[bool],
) -> object | None:
    """Execute one fixed view for one member and checkpoint each boundary."""
    backend_ref[0] = backend
    view = item.get(view_name)
    if not isinstance(view, dict):
        raise SeparationRunError("derived view row is invalid")
    view["input_view_id"] = input_view_id
    view["inference_config"] = dict(inference_config)
    view["inference_config_sha256"] = inference_config_sha
    input_root = run_dir / "inputs"
    canonical_path = input_root / str(item["simfile_id"]) / f"{view_name}.wav"
    prior_view = (
        prior_item.get(view_name)
        if isinstance(prior_item, Mapping) and isinstance(prior_item.get(view_name), Mapping)
        else None
    )
    runtime_evidence: dict[str, object] = {}
    stem_path: Path | None = None

    separator_started = perf_counter()
    try:
        if (
            request.resume
            and isinstance(prior_view, Mapping)
            and prior_view.get("stem") is not None
        ):
            raw_stem = prior_view.get("stem")
            if not isinstance(raw_stem, Mapping):
                raise SeparatorExecutionError("stem_identity_invalid", "stem evidence is invalid")
            stem_path = _resolve_retained_stem(
                raw_stem,
                cache_root=request.cache_dir,
                source=source,
                lock=runtime.lock,
            )
            view["stem"] = dict(raw_stem)
            runtime_evidence["separator_cache_hit"] = True
        else:
            separator_invocation_attempted[0] = True
            stem = separator_runner(
                source.path,
                source_audio_sha256=source.source_audio_sha256,
                source_duration_sec=source.duration_sec,
                runtime=runtime,
                cache_root=request.cache_dir,
            )
            if not isinstance(stem, SeparatedStem):
                raise SeparatorExecutionError(
                    "stem_identity_invalid", "separator returned invalid data"
                )
            if stem.source_audio_sha256 != source.source_audio_sha256:
                raise SeparatorExecutionError(
                    "stem_identity_invalid", "separator source identity changed"
                )
            if stem.separator_lock_sha256 != runtime.lock.sha256:
                raise SeparatorExecutionError(
                    "stem_identity_invalid", "separator lock identity changed"
                )
            stem_path = stem.path
            view["stem"] = _stem_evidence(stem, request.cache_dir)
            runtime_evidence["separator_cache_hit"] = stem.cache_hit
    except SeparatorExecutionError as error:
        elapsed = max(0.0, perf_counter() - separator_started)
        runtime_evidence.update(
            {
                "separator_wall_time_sec": elapsed,
                "separator_rtf": elapsed / source.duration_sec if source.duration_sec > 0 else None,
            }
        )
        _set_view_failure(
            view,
            status=_separator_failure_status(error.code),
            failure_code=error.code,
            runtime=runtime_evidence,
        )
        write_oaf_separation_run(run_path, snapshot)
        return backend
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        elapsed = max(0.0, perf_counter() - separator_started)
        runtime_evidence.update({"separator_wall_time_sec": elapsed})
        _set_view_failure(
            view,
            status="separation_failed",
            failure_code=type(error).__name__,
            runtime=runtime_evidence,
        )
        write_oaf_separation_run(run_path, snapshot)
        return backend

    runtime_evidence["separator_wall_time_sec"] = max(0.0, perf_counter() - separator_started)
    runtime_evidence["separator_rtf"] = (
        runtime_evidence["separator_wall_time_sec"] / source.duration_sec
    )
    if isinstance(prior_view, Mapping):
        prior_runtime = prior_view.get("runtime")
        if isinstance(prior_runtime, Mapping):
            for field in ("separator_wall_time_sec", "separator_rtf"):
                if field in prior_runtime:
                    runtime_evidence[field] = prior_runtime[field]
    view["runtime"] = runtime_evidence
    write_oaf_separation_run(run_path, snapshot)

    assert stem_path is not None
    try:
        audio = materialize_derived_audio(
            source,
            stem_path,
            canonical_path,
            input_root=input_root,
            input_view_id=input_view_id,
            max_input_audio_frames=None,
        )
        if not isinstance(audio, CanonicalAudio):
            raise TypeError("derived materializer returned invalid audio")
        if (
            audio.source_audio_id != source.source_audio_id
            or audio.source_audio_sha256 != source.source_audio_sha256
            or audio.input_view_id != input_view_id
        ):
            raise ValueError("derived audio identity is invalid")
        view["input"] = {
            "path": _relative_artifact_path(audio.path, input_root),
            "input_view_id": audio.input_view_id,
            "input_audio_sha256": audio.input_audio_sha256,
            "source_audio_id": audio.source_audio_id,
            "source_audio_sha256": audio.source_audio_sha256,
            "byte_length": audio.byte_length,
            "sample_rate": audio.sample_rate,
            "channel_count": audio.channel_count,
            "sample_width_bytes": audio.sample_width_bytes,
            "audio_frame_count": audio.audio_frame_count,
        }
        write_oaf_separation_run(run_path, snapshot)
    except (OSError, RuntimeError, TypeError, ValueError):
        _set_view_failure(
            view,
            status="inference_failed",
            failure_code="canonical_input_failed",
            runtime=runtime_evidence,
        )
        write_oaf_separation_run(run_path, snapshot)
        canonical_path.unlink(missing_ok=True)
        return backend

    prediction_target = prediction_path(
        run_dir.parent.parent,
        simfile_id=item["simfile_id"],  # type: ignore[arg-type]
        source_audio_sha256=source.source_audio_sha256,
        backend_descriptor_sha256=descriptor.sha256,
        inference_config_sha256=inference_config_sha,
    )
    existing_content: bytes | None = None
    try:
        existing_content = read_regular_file_no_follow(prediction_target)
    except FileNotFoundError:
        existing_content = None
    except (OSError, TypeError):
        _set_view_failure(
            view,
            status="prediction_invalid",
            failure_code="prediction_artifact_invalid",
            runtime=runtime_evidence,
        )
        write_oaf_separation_run(run_path, snapshot)
        canonical_path.unlink(missing_ok=True)
        return backend

    if existing_content is not None:
        if not request.resume:
            _set_view_failure(
                view,
                status="prediction_invalid",
                failure_code="prediction_output_conflict",
                runtime=runtime_evidence,
            )
            write_oaf_separation_run(run_path, snapshot)
            canonical_path.unlink(missing_ok=True)
            return backend
        prior_prediction = prior_view.get("prediction") if isinstance(prior_view, Mapping) else None
        prior_artifact_sha = (
            prior_prediction.get("artifact_sha256")
            if isinstance(prior_prediction, Mapping)
            else None
        )
        if isinstance(prior_artifact_sha, str) and sha256(existing_content).hexdigest() != (
            prior_artifact_sha
        ):
            _set_view_failure(
                view,
                status="prediction_invalid",
                failure_code="prediction_output_conflict",
                runtime=runtime_evidence,
            )
            write_oaf_separation_run(run_path, snapshot)
            canonical_path.unlink(missing_ok=True)
            return backend
        try:
            artifact = read_prediction_artifact(existing_content)
        except (PredictionArtifactError, StrictJsonError, TypeError, ValueError):
            _set_view_failure(
                view,
                status="prediction_invalid",
                failure_code="prediction_artifact_invalid",
                runtime=runtime_evidence,
            )
            write_oaf_separation_run(run_path, snapshot)
            canonical_path.unlink(missing_ok=True)
            return backend
        prior_row = (
            _prediction_row_from_view(prior_item, prior_view)
            if isinstance(prior_item, Mapping) and isinstance(prior_view, Mapping)
            else {}
        )
        matches_run_row = prediction_artifact_matches_run_row(
            artifact,
            prior_row,
            expected_input_view_id=input_view_id,
        )
        if not matches_run_row:
            _set_view_failure(
                view,
                status="prediction_invalid",
                failure_code="prediction_output_conflict",
                runtime=runtime_evidence,
            )
            write_oaf_separation_run(run_path, snapshot)
            canonical_path.unlink(missing_ok=True)
            return backend
        if not prediction_artifact_matches_audio(
            artifact,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            audio=audio,
            descriptor=descriptor,
            prediction_map_version=OAF_PREDICTION_MAP_ID,
        ):
            _set_view_failure(
                view,
                status="prediction_invalid",
                failure_code="prediction_artifact_invalid",
                runtime=runtime_evidence,
            )
            write_oaf_separation_run(run_path, snapshot)
            canonical_path.unlink(missing_ok=True)
            return backend
        prior_runtime = prior_view.get("runtime") if isinstance(prior_view, Mapping) else None
        if isinstance(prior_runtime, Mapping):
            runtime_evidence.update(dict(prior_runtime))
        view["prediction"] = {
            "path": _relative_artifact_path(prediction_target, run_dir.parent.parent),
            "artifact_sha256": artifact.artifact_sha256,
        }
        view["status"] = "resumed"
        view["failure_code"] = None
        view["runtime"] = runtime_evidence
        write_oaf_separation_run(run_path, snapshot)
        canonical_path.unlink(missing_ok=True)
        return backend

    try:
        if backend is None:
            try:
                backend = backend_factory(
                    input_root=input_root,
                    timeout_seconds=OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
                    close_timeout_seconds=OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
                )
                backend_ref[0] = backend
                actual_descriptor = backend.descriptor()  # type: ignore[attr-defined]
                if not isinstance(actual_descriptor, BackendDescriptor) or (
                    actual_descriptor.sha256 != descriptor.sha256
                    or dict(actual_descriptor.payload) != dict(descriptor.payload)
                ):
                    raise OafBackendError(
                        "backend descriptor identity changed",
                        code="descriptor_invalid",
                    )
            except OafBackendError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as error:
                raise OafBackendError(
                    "backend could not be started", code="worker_start_failed"
                ) from error
        started = perf_counter()
        try:
            native = backend.transcribe(audio)  # type: ignore[attr-defined]
        except OafBackendError:
            raise
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise OafBackendError("worker request failed", code="worker_error") from error
        elapsed = max(0.0, perf_counter() - started)
        if not isinstance(native, NativePrediction):
            raise OafBackendError("native prediction is invalid", code="native_event_invalid")
        try:
            mapped, _ = map_oaf_prediction(native)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise OafBackendError(
                "worker response mapping failed", code="worker_response_invalid"
            ) from error
        published: PublishedArtifact = publish_prediction_artifact(prediction_target, mapped)
        runtime_evidence["wall_time_sec"] = elapsed
        runtime_evidence["rtf"] = elapsed / source.duration_sec if source.duration_sec > 0 else None
        view["prediction"] = {
            "path": _relative_artifact_path(prediction_target, run_dir.parent.parent),
            "artifact_sha256": published.sha256,
        }
        view["status"] = "success"
        view["failure_code"] = None
        view["runtime"] = runtime_evidence
    except OafBackendError as error:
        runner_code, disposition = classify_oaf_backend_error(error.code)
        _set_view_failure(
            view,
            status="inference_failed",
            failure_code=runner_code or error.code,
            runtime=runtime_evidence,
        )
        if disposition in {"poison", "fatal_preflight"}:
            backend_to_close = backend
            backend = None
            backend_ref[0] = None
            _close_backend(backend_to_close)
            stop_disposition.append(disposition)
    except (ArtifactPublicationError, PredictionArtifactError):
        _set_view_failure(
            view,
            status="prediction_invalid",
            failure_code="prediction_publish_failed",
            runtime=runtime_evidence,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        _set_view_failure(
            view,
            status="prediction_invalid",
            failure_code="prediction_artifact_invalid",
            runtime=runtime_evidence,
        )
    finally:
        write_oaf_separation_run(run_path, snapshot)
        canonical_path.unlink(missing_ok=True)
    backend_ref[0] = backend
    return backend


def run_oaf_separation_pilot(
    request: OafSeparationPilotRequest,
    *,
    backend_factory: object | None = None,
    spleeter_runner: object | None = None,
    htdemucs_runner: object | None = None,
    perf_counter: object | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OafSeparationPilotOutcome:
    """Run the frozen full-mix control plus both fixed derived OaF views."""
    if not isinstance(request, OafSeparationPilotRequest):
        raise TypeError("request must be OafSeparationPilotRequest")
    selected_clock = clock or (lambda: datetime.now(timezone.utc))
    selected_perf_counter = perf_counter or time.perf_counter
    selected_backend_factory = backend_factory or create_backend
    selected_spleeter_runner = spleeter_runner or run_spleeter_drums
    selected_htdemucs_runner = htdemucs_runner or run_htdemucs_drums
    if not all(
        callable(seam)
        for seam in (
            selected_perf_counter,
            selected_backend_factory,
            selected_spleeter_runner,
            selected_htdemucs_runner,
        )
    ):
        raise TypeError("execution seams must be callable")
    backend: object | None = None
    backend_ref: list[object | None] = [None]
    runtimes: dict[str, AttestedSeparatorRuntime] = {}
    run_path: Path | None = None
    snapshot: dict[str, object] | None = None
    derived_view_preimages: dict[tuple[int, str], dict[str, object]] = {}
    separator_invocation_attempted = [False]
    postflight_finished = False
    try:
        _require_crux_commit(request.crux_commit)
        _validate_output_paths(request)
        reference = load_reference_set_manifest(request.reference_manifest_path)
        timing = load_reference_timing_manifest(request.timing_manifest_path)
        subset = load_reviewed_subset_manifest(request.subset_manifest_path)
        if (
            subset.source_reference_manifest_sha256 != reference.manifest_sha256
            or subset.source_reference_manifest_version != reference.corpus_version
        ):
            raise SeparationRunError("reviewed subset reference identity does not match")
        if (
            subset.source_timing_manifest_sha256 != timing.manifest_sha256
            or subset.source_timing_manifest_version != timing.corpus_version
        ):
            raise SeparationRunError("reviewed subset timing identity does not match")
        _validate_subset_population(subset, reference)
        # Reconstruct the exact HPA-323/HPA-324 mapping once at the fatal
        # boundary.  The returned mapping is intentionally not used to score
        # full mix: the public wrapper owns that operation.
        reference_mappings = preflight_reference_mappings(
            reference,
            timing,
            timing_output_root=request.timing_manifest_path.parent.parent,
        )
        parent = _read_parent_run(request.oaf_run_path)
        _validate_parent_identity(parent, reference, timing)
        spleeter, htdemucs = _load_separator_locks()
        rows = _subset_parent_rows(
            subset,
            parent,
            spleeter_lock_sha256=spleeter.sha256,
            htdemucs_lock_sha256=htdemucs.sha256,
        )
        identity = _identity_payload(request, reference, timing, subset, parent, spleeter, htdemucs)
        run_id = "oaf-separation-" + sha256(canonical_json_bytes(identity)).hexdigest()[:16]
        run_dir = request.output_dir / "runs" / run_id
        reports_path = run_dir / "views" / "full_mix" / "reports"
        if reports_path.resolve() == (request.oaf_run_path.resolve().parent / "reports").resolve():
            raise SeparationRunError("full-mix reports alias parent run reports")
        run_path = run_dir / "run.json"

        # Resolve all authoritative source identities before either report
        # scoring or an expensive separator/OaF operation.
        sources = _resolve_pilot_sources(request, subset, rows)
        raw_descriptor = parent.get("backend_descriptor")
        if not isinstance(raw_descriptor, Mapping):
            raise SeparationRunError("parent backend descriptor is unavailable")
        descriptor = BackendDescriptor(
            payload=dict(raw_descriptor),
            sha256=parent["backend_descriptor_sha256"],  # type: ignore[arg-type]
        )
        _validate_frozen_oaf_binding(parent)
        view_configs = {
            "spleeter": _view_inference_config(parent, SPLEETER_INPUT_VIEW_ID),
            "htdemucs": _view_inference_config(parent, HTDEMUCS_INPUT_VIEW_ID),
        }
        try:
            runtimes = {
                SPLEETER_SEPARATOR_ID: attest_separator_runtime(
                    SEPARATOR_LOCK_PATHS[SPLEETER_SEPARATOR_ID],
                    request.spleeter_python,
                    request.spleeter_model_root,
                ),
                HTDEMUCS_SEPARATOR_ID: attest_separator_runtime(
                    SEPARATOR_LOCK_PATHS[HTDEMUCS_SEPARATOR_ID],
                    request.demucs_python,
                    request.demucs_model_root,
                ),
            }
        except SeparatorExecutionError as error:
            failure_code = error.code if error.code in ATTESTATION_FAILURE_CODES else None
            return _fatal_outcome(failure_code)

        prior_snapshot: Mapping[str, object] | None = None
        if run_path.exists():
            if not request.resume:
                raise SeparationRunError("HPA-328 run already exists without resume")
            prior_snapshot = parse_oaf_separation_run(
                read_regular_file_no_follow(run_path),
                expected_run_id=run_id,
            )
        snapshot = _build_snapshot(
            request,
            reference,
            timing,
            subset,
            parent,
            spleeter,
            htdemucs,
            clock=selected_clock,
            run_id=run_id,
            rows=rows,
        )
        if prior_snapshot is not None:
            _recover_prior_derived_evidence(snapshot, prior_snapshot)
        derived_view_preimages = _capture_derived_view_preimages(snapshot)
        # Validate the complete local ledger before invoking the public
        # scorer.  A malformed parent evidence row is still fatal preflight;
        # it must not cause even the full-mix control wrapper to run.
        render_oaf_separation_run(snapshot)
        bound_backend_factory = _bind_oaf_backend_factory(
            selected_backend_factory, parent, descriptor
        )
        write_oaf_separation_run(run_path, snapshot)

        # Publish the control only after every fatal identity and source check
        # has passed.  This is the persisted full-mix report path; no full-mix
        # audio is ever sent through the injected backend below.
        control = _score_full_mix_control(request, run_dir)
        snapshot["overall_status"] = "pending" if control.exit_code == 0 else "partial"
        write_oaf_separation_run(run_path, snapshot)

        prior_items = prior_snapshot.get("items", []) if isinstance(prior_snapshot, Mapping) else []
        prior_by_id = {
            row["simfile_id"]: row
            for row in prior_items
            if isinstance(row, Mapping) and isinstance(row.get("simfile_id"), int)
        }
        stop_disposition: list[str] = []
        for item in rows:
            simfile_id = item.get("simfile_id")
            source = sources.get(simfile_id) if isinstance(simfile_id, int) else None
            if source is None:
                raise SeparationRunError("resolved source is unavailable")
            prior_item = prior_by_id.get(simfile_id)
            stop_disposition.clear()
            backend = _execute_derived_view(
                request,
                run_path,
                run_dir,
                snapshot,
                item,
                source,
                view_name="spleeter",
                input_view_id=SPLEETER_INPUT_VIEW_ID,
                runtime=runtimes[SPLEETER_SEPARATOR_ID],
                inference_config=view_configs["spleeter"][0],
                inference_config_sha=view_configs["spleeter"][1],
                prior_item=prior_item,
                separator_runner=selected_spleeter_runner,  # type: ignore[arg-type]
                backend_factory=bound_backend_factory,
                backend=backend,
                backend_ref=backend_ref,
                descriptor=descriptor,
                perf_counter=selected_perf_counter,  # type: ignore[arg-type]
                stop_disposition=stop_disposition,
                separator_invocation_attempted=separator_invocation_attempted,
            )
            if stop_disposition:
                break
            stop_disposition.clear()
            backend = _execute_derived_view(
                request,
                run_path,
                run_dir,
                snapshot,
                item,
                source,
                view_name="htdemucs",
                input_view_id=HTDEMUCS_INPUT_VIEW_ID,
                runtime=runtimes[HTDEMUCS_SEPARATOR_ID],
                inference_config=view_configs["htdemucs"][0],
                inference_config_sha=view_configs["htdemucs"][1],
                prior_item=prior_item,
                separator_runner=selected_htdemucs_runner,  # type: ignore[arg-type]
                backend_factory=bound_backend_factory,
                backend=backend,
                backend_ref=backend_ref,
                descriptor=descriptor,
                perf_counter=selected_perf_counter,  # type: ignore[arg-type]
                stop_disposition=stop_disposition,
                separator_invocation_attempted=separator_invocation_attempted,
            )
            if stop_disposition:
                break
        disposition = stop_disposition[0] if stop_disposition else None
        if disposition is not None:
            _mark_outstanding_derived_views(snapshot)
            snapshot["overall_status"] = "failed" if disposition == "fatal_preflight" else "partial"
            write_oaf_separation_run(run_path, snapshot)
        try:
            _revalidate_separator_runtimes(runtimes)
        except SeparatorExecutionError as error:
            postflight_finished = True
            if error.code == "separator_model_root_invalid":
                _persist_restored_failed_snapshot(run_path, snapshot, derived_view_preimages)
                return _fatal_outcome("separator_model_root_invalid")
            raise
        postflight_finished = True
        if disposition == "fatal_preflight":
            return _fatal_outcome()
        _score_derived_cohort(
            request,
            run_dir,
            snapshot,
            rows,
            reference_mappings,
            reference,
            timing,
            descriptor,
            view_name="spleeter",
            input_view_id=SPLEETER_INPUT_VIEW_ID,
        )
        _score_derived_cohort(
            request,
            run_dir,
            snapshot,
            rows,
            reference_mappings,
            reference,
            timing,
            descriptor,
            view_name="htdemucs",
            input_view_id=HTDEMUCS_INPUT_VIEW_ID,
        )
        if _comparison_reports_ready(run_dir):
            compare_oaf_separation(
                SeparationComparisonRequest(
                    run_path=run_path,
                    reference_manifest_path=request.reference_manifest_path,
                    timing_manifest_path=request.timing_manifest_path,
                    subset_manifest_path=request.subset_manifest_path,
                    output_dir=run_dir / "comparison",
                    cache_dir=request.cache_dir,
                )
            )
        derived_failed = any(
            isinstance(item.get(view_name), Mapping)
            and item[view_name].get("status")  # type: ignore[index]
            in {"separation_failed", "stem_invalid", "inference_failed", "prediction_invalid"}
            for item in rows
            for view_name in ("spleeter", "htdemucs")
        )
        status: PilotStatus = (
            "complete" if control.exit_code == 0 and not derived_failed else "partial"
        )
        exit_code: PilotExitCode = 0 if status == "complete" else 1
        snapshot["overall_status"] = status
        write_oaf_separation_run(run_path, snapshot)
        return OafSeparationPilotOutcome(
            overall_status=status,
            exit_code=exit_code,
            run_id=run_id,
            run_path=run_path,
            reports_path=reports_path,
            full_mix_reports_path=reports_path,
            success_count=control.success_count,
            failed_count=control.failed_count,
            skipped_count=control.skipped_count,
            quarantined_count=control.quarantined_count,
            failure_code=None,
        )
    except (OSError, RuntimeError, StrictJsonError, SeparationRunError, TypeError, ValueError):
        if separator_invocation_attempted[0] and not postflight_finished:
            cleanup_failure_code: str | None = None
            try:
                _revalidate_separator_runtimes(runtimes)
            except SeparatorExecutionError as error:
                if error.code == "separator_model_root_invalid":
                    try:
                        if snapshot is not None and run_path is not None:
                            _persist_restored_failed_snapshot(
                                run_path, snapshot, derived_view_preimages
                            )
                            cleanup_failure_code = "separator_model_root_invalid"
                    except BaseException:  # preserve the original fatal path
                        pass
            except BaseException:  # preserve the original fatal path
                pass
            if cleanup_failure_code is not None:
                return _fatal_outcome(cleanup_failure_code)
        return _fatal_outcome()
    except BaseException:
        if separator_invocation_attempted[0] and not postflight_finished:
            try:
                _revalidate_separator_runtimes(runtimes)
            except SeparatorExecutionError as error:
                if error.code == "separator_model_root_invalid":
                    try:
                        if snapshot is not None and run_path is not None:
                            _persist_restored_failed_snapshot(
                                run_path, snapshot, derived_view_preimages
                            )
                    except BaseException:  # preserve the original exception
                        pass
            except BaseException:  # preserve the original exception
                pass
        raise
    finally:
        _close_backend(backend_ref[0])


__all__ = [
    "HTDEMUCS_INPUT_VIEW_ID",
    "OafSeparationPilotOutcome",
    "OafSeparationPilotRequest",
    "SEPARATION_FAILURE_TO_COHORT_REASON",
    "SEPARATION_RUN_SCHEMA",
    "SEPARATOR_LOCK_PATHS",
    "SPLEETER_INPUT_VIEW_ID",
    "SeparationRunError",
    "parse_oaf_separation_run",
    "parse_separation_run",
    "render_oaf_separation_run",
    "render_separation_run",
    "run_oaf_separation_pilot",
    "write_oaf_separation_run",
    "write_separation_run",
]
