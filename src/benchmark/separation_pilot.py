"""HPA-328 fixed-subset preflight and separation-run ledger.

Task 5 deliberately keeps this boundary concrete.  It validates the four
already-published input artifacts, binds the two fixed derived views to the
persisted OaF control, writes the local HPA-328 run identity, and obtains the
full-mix control through the public reviewed-subset scorer.  Separator and OaF
execution are added in the following task; this module must not construct
either runtime during preflight.
"""

# This concrete task boundary intentionally keeps request/ledger state and
# preflight checks together; splitting it into a framework would violate the
# HPA-328 scope.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-branches,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
    quantize_six,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.durability import atomic_replace_bytes
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    build_run_id,
    inference_config_sha256,
    parse_oaf_corpus_run,
)
from src.benchmark.r2_corpus_models import format_manifest_timestamp, parse_manifest_timestamp
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
)
from src.benchmark.reviewed_subset import (
    LoadedReviewedSubsetManifest,
    ScoreReviewedSubsetOutcome,
    ScoreReviewedSubsetRequest,
    _source_row_sha256,
    load_reviewed_subset_manifest,
    score_oaf_reviewed_subset,
)
from src.benchmark.separators import (
    HTDEMUCS_SEPARATOR_ID,
    SPLEETER_SEPARATOR_ID,
    SeparatorLock,
    load_separator_lock,
)

SEPARATION_RUN_SCHEMA = "crux.oaf-separation-run/v1"
SPLEETER_INPUT_VIEW_ID = "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
HTDEMUCS_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

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


class SeparationRunError(ValueError):
    """A malformed or inconsistent HPA-328 run snapshot."""


def _fatal_outcome() -> OafSeparationPilotOutcome:
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
        prediction_path = prediction_evidence.get("path")
        if prediction_path is not None and (
            not isinstance(prediction_path, str) or not prediction_path
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


def run_oaf_separation_pilot(
    request: OafSeparationPilotRequest,
    *,
    backend_factory: object | None = None,
    spleeter_runner: object | None = None,
    htdemucs_runner: object | None = None,
    perf_counter: object | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OafSeparationPilotOutcome:
    """Preflight HPA-328 and publish the persisted full-mix control.

    The execution keyword parameters are accepted as the stable Task 6 seam,
    but are intentionally unused in this Task 5 implementation.  No
    separator process or OaF backend is touched here.
    """
    del backend_factory, spleeter_runner, htdemucs_runner, perf_counter
    if not isinstance(request, OafSeparationPilotRequest):
        raise TypeError("request must be OafSeparationPilotRequest")
    selected_clock = clock or (lambda: datetime.now(timezone.utc))
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
        preflight_reference_mappings(
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
        # Validate the complete local ledger before invoking the public
        # scorer.  A malformed parent evidence row is still fatal preflight;
        # it must not cause even the full-mix control wrapper to run.
        render_oaf_separation_run(snapshot)
        # Publish the control only after every fatal identity check has passed.
        control = _score_full_mix_control(request, run_dir)
        snapshot["overall_status"] = "complete" if control.exit_code == 0 else "partial"
        write_oaf_separation_run(run_dir / "run.json", snapshot)
        status: PilotStatus = "complete" if control.exit_code == 0 else "partial"
        exit_code: PilotExitCode = 0 if control.exit_code == 0 else 1
        return OafSeparationPilotOutcome(
            overall_status=status,
            exit_code=exit_code,
            run_id=run_id,
            run_path=run_dir / "run.json",
            reports_path=reports_path,
            full_mix_reports_path=reports_path,
            success_count=control.success_count,
            failed_count=control.failed_count,
            skipped_count=control.skipped_count,
            quarantined_count=control.quarantined_count,
        )
    except (OSError, RuntimeError, StrictJsonError, SeparationRunError, TypeError, ValueError):
        return _fatal_outcome()


__all__ = [
    "HTDEMUCS_INPUT_VIEW_ID",
    "OafSeparationPilotOutcome",
    "OafSeparationPilotRequest",
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
