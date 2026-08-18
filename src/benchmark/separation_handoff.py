"""Immutable HPA-396 handoff for the fixed HPA-328 separation pilot.

The pilot run is deliberately mutable while it executes and resumes.  This
module is the closeout boundary: it validates the frozen subset/run lineage,
re-reads every retained successful stem and prediction, hashes the comparison
files, and publishes self-contained JSONL rows through the normal manifest
rails.  Consumers can therefore use the handoff after the run directory is no
longer available.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    StrictJsonError,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    publish_manifest,
    render_manifest,
)
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.prediction_artifact import PredictionArtifactError, read_prediction_artifact
from src.benchmark.reviewed_subset import load_reviewed_subset_manifest
from src.benchmark.separation_pilot import (
    HTDEMUCS_INPUT_VIEW_ID,
    SEPARATION_RUN_SCHEMA,
    SPLEETER_INPUT_VIEW_ID,
    parse_oaf_separation_run,
)

SEPARATION_PILOT_SCHEMA = "crux.oaf-separation-pilot/v1"
SeparationDecision = Literal[
    "keep_full_mix",
    "use_spleeter",
    "use_htdemucs",
    "gather_more_evidence",
    "prioritize_another_model",
]

_DECISIONS = frozenset(get_args(SeparationDecision))
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RUN_ID_RE = re.compile(r"oaf-separation-[0-9a-f]{16}\Z")
_CORPUS_VERSION_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VIEW_IDS = {
    "full_mix": OAF_FULL_MIX_INPUT_VIEW_ID,
    "spleeter": SPLEETER_INPUT_VIEW_ID,
    "htdemucs": HTDEMUCS_INPUT_VIEW_ID,
}
_DERIVED_VIEW_NAMES = frozenset({"spleeter", "htdemucs"})
_FULL_MIX_STATUSES = frozenset({"inferred", "resumed", "failed", "skipped", "quarantined"})
_DERIVED_STATUSES = frozenset(
    {
        "success",
        "resumed",
        "separation_failed",
        "stem_invalid",
        "inference_failed",
        "prediction_invalid",
    }
)
_COMPARISON_ARTIFACT_KEYS = (
    "summary.json",
    "summary.md",
    "spleeter/paired_per_song.csv",
    "spleeter/paired_per_class.csv",
    "htdemucs/paired_per_song.csv",
    "htdemucs/paired_per_class.csv",
)

_ROW_KEYS = frozenset(
    {
        "schema_version",
        "separation_run_id",
        "simfile_id",
        "source_row_sha256",
        "reviewed_subset_manifest_sha256",
        "reviewed_subset_manifest_version",
        "reference_manifest_sha256",
        "reference_manifest_version",
        "reference_timing_manifest_sha256",
        "reference_timing_version",
        "source_audio_id",
        "source_audio_sha256",
        "source_duration_sec",
        "parent_oaf_run_id",
        "oaf_backend_descriptor_sha256",
        "oaf_model_id",
        "oaf_model_lock_sha256",
        "oaf_checkpoint_archive_sha256",
        "oaf_adapter_revision",
        "oaf_canonicalization_revision",
        "oaf_inference_config_sha256",
        "oaf_prediction_map_version",
        "crux_commit",
        "full_mix",
        "spleeter",
        "htdemucs",
        "comparison_artifacts",
        "decision",
        "rationale",
    }
)
_VIEW_KEYS = frozenset(
    {
        "status",
        "failure_code",
        "separator_lock_sha256",
        "input_view_id",
        "stem",
        "input",
        "prediction",
    }
)
_STEM_KEYS = frozenset({"path", "sha256", "source_audio_sha256", "separator_lock_sha256"})
_INPUT_KEYS = frozenset(
    {"path", "input_view_id", "input_audio_sha256", "source_audio_id", "source_audio_sha256"}
)
_PREDICTION_KEYS = frozenset(
    {
        "path",
        "artifact_sha256",
        "source_audio_id",
        "source_audio_sha256",
        "input_view_id",
        "input_audio_sha256",
    }
)
_COMPARISON_ARTIFACT_KEYS_SET = frozenset(_COMPARISON_ARTIFACT_KEYS)
_ARTIFACT_KEYS = frozenset({"path", "sha256"})


class SeparationHandoffError(ValueError):
    """Raised when a pilot cannot be closed into an immutable handoff."""


@dataclass(frozen=True)
class FinalizeSeparationPilotRequest:
    run_path: Path
    subset_manifest_path: Path
    output_manifest: Path
    decision: SeparationDecision
    rationale: str

    def __post_init__(self) -> None:
        for field in ("run_path", "subset_manifest_path", "output_manifest"):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if self.decision not in _DECISIONS:
            raise ValueError("decision is invalid")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be nonempty")


@dataclass(frozen=True)
class FinalizeSeparationPilotOutcome:
    exit_code: Literal[0, 2]
    manifest: object | None


@dataclass(frozen=True)
class LoadedSeparationPilotManifest:
    manifest_sha256: str
    corpus_version: str
    rows: tuple[Mapping[str, object], ...]


def _fail(message: str) -> None:
    raise SeparationHandoffError(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a nonempty string")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a lowercase SHA-256")
    try:
        return require_sha256(value, field)
    except StrictJsonError:
        _fail(f"{field} must be a lowercase SHA-256")
    raise AssertionError("unreachable")


def _version(value: object, field: str) -> str:
    if not isinstance(value, str) or _CORPUS_VERSION_RE.fullmatch(value) is None:
        _fail(f"{field} must be a corpus version")
    return value


def _commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        _fail("crux_commit must be a lowercase 40-character commit")
    return value


def _path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        _fail(f"{field} path is invalid")
    return value


def _owner_root(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{field} owner root is invalid")
    root = Path(value)
    if not root.is_absolute():
        _fail(f"{field} owner root is invalid")
    return root


def _positive_duration(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        _fail(f"{field} is invalid")
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError):
        _fail(f"{field} is invalid")
    if not parsed.is_finite() or parsed <= 0:
        _fail(f"{field} is invalid")
    return parsed


def _manifest_duration(value: object, field: str) -> int | float:
    parsed = _positive_duration(value, field)
    if parsed == parsed.to_integral_value():
        return int(parsed)
    return float(parsed)


def _manifest_json_value(value: object) -> object:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Mapping):
        return {key: _manifest_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_manifest_json_value(item) for item in value]
    return value


def _validate_stem(
    value: object,
    *,
    field: str,
    source_audio_sha256: str,
    separator_lock_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _STEM_KEYS:
        _fail(f"{field} stem evidence is invalid")
    _path(value["path"], f"{field}.stem")
    _hash(value["sha256"], f"{field}.stem.sha256")
    _hash(value["source_audio_sha256"], f"{field}.stem.source_audio_sha256")
    _hash(value["separator_lock_sha256"], f"{field}.stem.separator_lock_sha256")
    if value["source_audio_sha256"] != source_audio_sha256:
        _fail(f"{field} stem source identity does not match")
    if value["separator_lock_sha256"] != separator_lock_sha256:
        _fail(f"{field} stem separator identity does not match")


def _validate_input(
    value: object,
    *,
    field: str,
    input_view_id: str,
    source_audio_id: str,
    source_audio_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _INPUT_KEYS:
        _fail(f"{field} input evidence is invalid")
    if value["path"] is None:
        if field != "full_mix":
            _fail(f"{field}.input path is invalid")
    else:
        _path(value["path"], f"{field}.input")
    _text(value["input_view_id"], f"{field}.input.input_view_id")
    _hash(value["input_audio_sha256"], f"{field}.input.input_audio_sha256")
    _text(value["source_audio_id"], f"{field}.input.source_audio_id")
    _hash(value["source_audio_sha256"], f"{field}.input.source_audio_sha256")
    if value["input_view_id"] != input_view_id:
        _fail(f"{field} input view identity does not match")
    if value["source_audio_id"] != source_audio_id:
        _fail(f"{field} input source ID does not match")
    if value["source_audio_sha256"] != source_audio_sha256:
        _fail(f"{field} input source hash does not match")


def _validate_prediction(
    value: object,
    *,
    field: str,
    input_view_id: str,
    source_audio_id: str,
    source_audio_sha256: str,
    input_audio_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _PREDICTION_KEYS:
        _fail(f"{field} prediction evidence is invalid")
    _path(value["path"], f"{field}.prediction")
    _hash(value["artifact_sha256"], f"{field}.prediction.artifact_sha256")
    _text(value["source_audio_id"], f"{field}.prediction.source_audio_id")
    _hash(value["source_audio_sha256"], f"{field}.prediction.source_audio_sha256")
    _text(value["input_view_id"], f"{field}.prediction.input_view_id")
    _hash(value["input_audio_sha256"], f"{field}.prediction.input_audio_sha256")
    if value["input_view_id"] != input_view_id:
        _fail(f"{field} prediction view identity does not match")
    if value["source_audio_id"] != source_audio_id:
        _fail(f"{field} prediction source ID does not match")
    if value["source_audio_sha256"] != source_audio_sha256:
        _fail(f"{field} prediction source hash does not match")
    if value["input_audio_sha256"] != input_audio_sha256:
        _fail(f"{field} prediction input hash does not match")


def _validate_view(
    value: object,
    *,
    field: str,
    source_audio_id: str,
    source_audio_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _VIEW_KEYS:
        _fail(f"{field} view evidence is invalid")
    status = value["status"]
    allowed = _FULL_MIX_STATUSES if field == "full_mix" else _DERIVED_STATUSES
    if not isinstance(status, str) or status not in allowed:
        _fail(f"{field} status is invalid")
    failure_code = value["failure_code"]
    if failure_code is not None:
        _text(failure_code, f"{field}.failure_code")
    input_view_id = value["input_view_id"]
    if input_view_id != _VIEW_IDS[field]:
        _fail(f"{field} input view identity is invalid")

    lock = value["separator_lock_sha256"]
    if field == "full_mix":
        if lock is not None:
            _fail("full_mix separator lock must be null")
    else:
        _hash(lock, f"{field}.separator_lock_sha256")

    stem = value["stem"]
    input_evidence = value["input"]
    prediction = value["prediction"]
    successful = status in {"inferred", "success", "resumed"}
    if successful and failure_code is not None:
        _fail(f"successful {field} failure_code must be null")
    if field == "full_mix":
        if successful:
            if stem is not None or input_evidence is None or prediction is None:
                _fail("successful full_mix evidence has invalid nullability")
        elif stem is not None or prediction is not None:
            _fail("non-successful full_mix evidence has invalid nullability")
    elif successful:
        if stem is None:
            _fail(f"successful {field} stem evidence has invalid nullability")
        if input_evidence is None:
            _fail(f"successful {field} input evidence has invalid nullability")
        if prediction is None:
            _fail(f"successful {field} prediction evidence has invalid nullability")
    elif status in {"separation_failed", "stem_invalid"}:
        if any(evidence is not None for evidence in (stem, input_evidence, prediction)):
            _fail(f"{status} {field} evidence has invalid nullability")
    elif prediction is not None:
        _fail(f"failed {field} prediction evidence must be null")

    if stem is not None:
        _validate_stem(
            stem,
            field=field,
            source_audio_sha256=source_audio_sha256,
            separator_lock_sha256=(
                _hash(lock, f"{field}.separator_lock_sha256") if field != "full_mix" else ""
            ),
        )
    if input_evidence is not None:
        _validate_input(
            input_evidence,
            field=field,
            input_view_id=input_view_id,
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
        )
    if prediction is not None:
        if not isinstance(input_evidence, Mapping):
            _fail(f"{field} prediction evidence has no input identity")
        input_audio_sha256 = input_evidence["input_audio_sha256"]
        if not isinstance(input_audio_sha256, str):
            _fail(f"{field} input hash is invalid")
        _validate_prediction(
            prediction,
            field=field,
            input_view_id=input_view_id,
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
            input_audio_sha256=input_audio_sha256,
        )


def _validate_comparison_artifacts(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _COMPARISON_ARTIFACT_KEYS_SET:
        _fail("comparison_artifacts has an invalid key set")
    for key in _COMPARISON_ARTIFACT_KEYS:
        artifact = value[key]
        if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_KEYS:
            _fail(f"comparison_artifacts.{key} is invalid")
        expected_path = f"comparison/{key}"
        if artifact["path"] != expected_path:
            _fail(f"comparison_artifacts.{key} path is invalid")
        _path(artifact["path"], f"comparison_artifacts.{key}")
        _hash(artifact["sha256"], f"comparison_artifacts.{key}.sha256")


def _comparison_identity(value: object) -> tuple[tuple[str, str, str], ...]:
    """Return the canonical comparison-file identity for cross-row checks."""
    _validate_comparison_artifacts(value)
    if not isinstance(value, Mapping):  # pragma: no cover - validated above
        raise AssertionError("comparison artifacts must be a mapping")
    identity: list[tuple[str, str, str]] = []
    for key in _COMPARISON_ARTIFACT_KEYS:
        artifact = value[key]
        if not isinstance(artifact, Mapping):  # pragma: no cover - validated above
            raise AssertionError("comparison artifact must be a mapping")
        identity.append((key, artifact["path"], artifact["sha256"]))
    return tuple(identity)


def _validate_row(row: Mapping[str, object], *, allow_corpus_version: bool) -> dict[str, object]:
    keys = set(row)
    if allow_corpus_version:
        if keys != _ROW_KEYS | {"corpus_version"}:
            _fail("separation handoff row has an invalid key set")
        _version(row["corpus_version"], "corpus_version")
    elif keys != _ROW_KEYS:
        _fail("separation handoff row has an invalid key set")
    if row["schema_version"] != SEPARATION_PILOT_SCHEMA:
        _fail("separation handoff row has an unsupported schema")
    run_id = _text(row["separation_run_id"], "separation_run_id")
    if _RUN_ID_RE.fullmatch(run_id) is None:
        _fail("separation_run_id is invalid")
    simfile_id = row["simfile_id"]
    if isinstance(simfile_id, bool) or not isinstance(simfile_id, int) or simfile_id <= 0:
        _fail("simfile_id is invalid")
    for field in (
        "source_row_sha256",
        "reviewed_subset_manifest_sha256",
        "reference_manifest_sha256",
        "reference_timing_manifest_sha256",
        "source_audio_sha256",
        "oaf_backend_descriptor_sha256",
        "oaf_model_lock_sha256",
        "oaf_checkpoint_archive_sha256",
        "oaf_inference_config_sha256",
    ):
        _hash(row[field], field)
    for field in (
        "reviewed_subset_manifest_version",
        "reference_manifest_version",
        "reference_timing_version",
    ):
        _version(row[field], field)
    source_audio_id = _text(row["source_audio_id"], "source_audio_id")
    source_audio_sha256 = _hash(row["source_audio_sha256"], "source_audio_sha256")
    _positive_duration(row["source_duration_sec"], "source_duration_sec")
    _text(row["parent_oaf_run_id"], "parent_oaf_run_id")
    _text(row["oaf_model_id"], "oaf_model_id")
    _text(row["oaf_adapter_revision"], "oaf_adapter_revision")
    _text(row["oaf_canonicalization_revision"], "oaf_canonicalization_revision")
    _text(row["oaf_prediction_map_version"], "oaf_prediction_map_version")
    _commit(row["crux_commit"])
    decision = row["decision"]
    if decision not in _DECISIONS:
        _fail("decision is invalid")
    rationale = row["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        _fail("rationale must be nonempty")
    for field in ("full_mix", "spleeter", "htdemucs"):
        _validate_view(
            row[field],
            field=field,
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
        )
    _validate_comparison_artifacts(row["comparison_artifacts"])
    return dict(row)


def _parse_manifest_content(
    content: bytes, *, require_population: bool
) -> tuple[str, tuple[dict[str, object], ...]]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        _fail("separation handoff must contain canonical JSONL records")
    lines = content.splitlines(keepends=True)
    if not lines or any(line == b"\n" or not line.endswith(b"\n") for line in lines):
        _fail("separation handoff must contain canonical JSONL records")
    rows: list[dict[str, object]] = []
    try:
        for line in lines:
            value = strict_json_loads(line[:-1], require_canonical=True)
            if not isinstance(value, Mapping):
                _fail("separation handoff rows must be objects")
            rows.append(_validate_row(value, allow_corpus_version=True))
    except StrictJsonError as error:
        _fail(f"separation handoff is not canonical JSONL: {error}")
    versions = {row["corpus_version"] for row in rows}
    if len(versions) != 1:
        _fail("separation handoff rows contain mixed corpus versions")
    (corpus_version,) = versions
    normalized = tuple(
        _manifest_json_value({key: value for key, value in row.items() if key != "corpus_version"})
        for row in rows
    )
    rendered = render_manifest(normalized)
    if rendered.content != content or rendered.corpus_version != corpus_version:
        _fail("separation handoff has an invalid derived corpus version")
    if require_population:
        if not 20 <= len(rows) <= 30:
            _fail("separation handoff must contain the exact HPA-327 population")
        ids = [row["simfile_id"] for row in rows]
        if ids != sorted(set(ids)):
            _fail("separation handoff rows must be unique and sorted")
    # Decision and lineage are run-level facts, repeated on each row because a
    # manifest has no mutable/header protocol.
    identity_fields = (
        "separation_run_id",
        "reviewed_subset_manifest_sha256",
        "reviewed_subset_manifest_version",
        "reference_manifest_sha256",
        "reference_manifest_version",
        "reference_timing_manifest_sha256",
        "reference_timing_version",
        "parent_oaf_run_id",
        "oaf_backend_descriptor_sha256",
        "oaf_model_id",
        "oaf_model_lock_sha256",
        "oaf_checkpoint_archive_sha256",
        "oaf_adapter_revision",
        "oaf_canonicalization_revision",
        "oaf_inference_config_sha256",
        "oaf_prediction_map_version",
        "crux_commit",
        "decision",
        "rationale",
    )
    identities = {
        tuple(row[field] for field in identity_fields)
        + (_comparison_identity(row["comparison_artifacts"]),)
        for row in rows
    }
    if len(identities) != 1:
        _fail("separation handoff rows contain mixed run or decision identity")
    return corpus_version, tuple(rows)


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate a canonical schema golden without requiring 20 pilot rows."""
    if schema != SEPARATION_PILOT_SCHEMA:
        raise ValueError("unsupported schema golden")
    _parse_manifest_content(content, require_population=False)


def load_separation_pilot_manifest(path: Path) -> LoadedSeparationPilotManifest:
    """Load an immutable HPA-396 handoff and validate its complete population."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError) as error:
        raise SeparationHandoffError("separation handoff is unavailable") from error
    corpus_version, rows = _parse_manifest_content(content, require_population=True)
    return LoadedSeparationPilotManifest(
        manifest_sha256=sha256(content).hexdigest(),
        corpus_version=corpus_version,
        rows=tuple(rows),
    )


def _recorded_artifact_path(
    raw_path: object,
    *,
    field: str,
    run_path: Path,
    owner_root: Path | None = None,
) -> Path:
    """Resolve one persisted path from its fixed HPA-328 owner root.

    Each path has one owner: parent OaF predictions and retained stems carry
    their exact owner root in the mutable run evidence, derived predictions
    are relative to the separation output root, and comparisons are relative
    to the run directory. No directory discovery is permitted at finalization
    time.
    """
    path = _path(raw_path, field)
    parsed = PurePosixPath(path)
    if field.startswith("comparison_artifacts."):
        root = run_path.parent
    elif field in {"full_mix.prediction", "spleeter.stem", "htdemucs.stem"}:
        if owner_root is None:
            _fail(f"{field} owner root is unavailable")
            raise AssertionError("unreachable")
        root = owner_root
    elif field.endswith(".prediction"):
        root = run_path.parents[2]
    else:
        _fail(f"{field} has no retained artifact owner")
        raise AssertionError("unreachable")
    return root.joinpath(*parsed.parts)


def _read_matching_artifact(
    raw_path: object,
    expected_sha256: object,
    *,
    field: str,
    run_path: Path,
    subset_path: Path,
    owner_root: Path | None = None,
) -> tuple[Path, bytes]:
    del subset_path
    path = _recorded_artifact_path(
        raw_path,
        field=field,
        run_path=run_path,
        owner_root=owner_root,
    )
    expected = _hash(expected_sha256, f"{field}.sha256")
    try:
        content = read_regular_file_no_follow(path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SeparationHandoffError(f"{field} artifact is missing") from error
    except (OSError, TypeError) as error:
        raise SeparationHandoffError(f"{field} is unreadable") from error
    if sha256(content).hexdigest() != expected:
        _fail(f"{field} bytes do not match retained SHA-256")
    return path, content


def _prediction_payload(
    raw_prediction: Mapping[str, object],
    *,
    field: str,
    run_path: Path,
    subset_path: Path,
    owner_root: Path | None,
    input_view_id: str,
    source_audio_id: str,
    source_audio_sha256: str,
    input_audio_sha256: str,
) -> dict[str, object]:
    artifact_field = f"{field}.prediction"
    path, content = _read_matching_artifact(
        raw_prediction.get("path"),
        raw_prediction.get("artifact_sha256"),
        field=artifact_field,
        run_path=run_path,
        subset_path=subset_path,
        owner_root=owner_root,
    )
    del path
    try:
        artifact = read_prediction_artifact(content)
    except (PredictionArtifactError, StrictJsonError, TypeError, ValueError) as error:
        raise SeparationHandoffError(f"{field} prediction artifact is invalid") from error
    audio = artifact.prediction.audio
    if (
        audio.source_audio_id != source_audio_id
        or audio.source_audio_sha256 != source_audio_sha256
        or audio.input_view_id != input_view_id
        or audio.input_audio_sha256 != input_audio_sha256
    ):
        _fail(f"{field} prediction identity does not match run evidence")
    return {
        "path": _path(raw_prediction.get("path"), field),
        "artifact_sha256": _hash(
            raw_prediction.get("artifact_sha256"), f"{artifact_field}.artifact_sha256"
        ),
        "source_audio_id": audio.source_audio_id,
        "source_audio_sha256": audio.source_audio_sha256,
        "input_view_id": audio.input_view_id,
        "input_audio_sha256": audio.input_audio_sha256,
    }


def _input_payload(
    raw_input: Mapping[str, object],
    *,
    field: str,
    input_view_id: str,
    source_audio_id: str,
    source_audio_sha256: str,
) -> dict[str, object]:
    return {
        "path": (
            None
            if raw_input.get("path") is None and field == "full_mix"
            else _path(raw_input.get("path"), field)
        ),
        "input_view_id": input_view_id,
        "input_audio_sha256": _hash(
            raw_input.get("input_audio_sha256"), f"{field}.input_audio_sha256"
        ),
        "source_audio_id": source_audio_id,
        "source_audio_sha256": source_audio_sha256,
    }


def _stem_payload(
    raw_stem: Mapping[str, object],
    *,
    field: str,
    run_path: Path,
    subset_path: Path,
    source_audio_sha256: str,
    separator_lock_sha256: str,
) -> dict[str, object]:
    artifact_field = f"{field}.stem"
    owner_root = _owner_root(raw_stem.get("owner_root"), artifact_field)
    _read_matching_artifact(
        raw_stem.get("path"),
        raw_stem.get("sha256"),
        field=artifact_field,
        run_path=run_path,
        subset_path=subset_path,
        owner_root=owner_root,
    )
    return {
        "path": _path(raw_stem.get("path"), field),
        "sha256": _hash(raw_stem.get("sha256"), f"{artifact_field}.sha256"),
        "source_audio_sha256": source_audio_sha256,
        "separator_lock_sha256": separator_lock_sha256,
    }


def _view_payload(
    raw_view: Mapping[str, object],
    *,
    field: str,
    item: Mapping[str, object],
    run_path: Path,
    subset_path: Path,
    parent_prediction_root: Path,
) -> dict[str, object]:
    input_view_id = _VIEW_IDS[field]
    source_audio_id = _text(item["source_audio_id"], "source_audio_id")
    source_audio_sha256 = _hash(item["source_audio_sha256"], "source_audio_sha256")
    status = _text(raw_view.get("status"), f"{field}.status")
    lock = raw_view.get("separator_lock_sha256")
    if field == "full_mix":
        lock_value = None
    else:
        lock_value = _hash(lock, f"{field}.separator_lock_sha256")

    raw_input = raw_view.get("input")
    raw_prediction = raw_view.get("prediction")
    raw_stem = raw_view.get("stem")
    successful = status in {"inferred", "success", "resumed"}
    input_payload = (
        _input_payload(
            raw_input,
            field=field,
            input_view_id=input_view_id,
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
        )
        if isinstance(raw_input, Mapping)
        else None
    )
    prediction_payload = None
    if isinstance(raw_prediction, Mapping):
        if input_payload is None:
            _fail(f"{field} prediction has no input evidence")
        prediction_payload = _prediction_payload(
            raw_prediction,
            field=field,
            run_path=run_path,
            subset_path=subset_path,
            owner_root=parent_prediction_root if field == "full_mix" else None,
            input_view_id=input_view_id,
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
            input_audio_sha256=input_payload["input_audio_sha256"],  # type: ignore[arg-type]
        )
    stem_payload = None
    if isinstance(raw_stem, Mapping):
        if lock_value is None:
            _fail("full_mix cannot carry stem evidence")
        stem_payload = _stem_payload(
            raw_stem,
            field=field,
            run_path=run_path,
            subset_path=subset_path,
            source_audio_sha256=source_audio_sha256,
            separator_lock_sha256=lock_value,
        )
    view = {
        "status": status,
        "failure_code": raw_view.get("failure_code"),
        "separator_lock_sha256": lock_value,
        "input_view_id": input_view_id,
        "stem": stem_payload,
        "input": input_payload,
        "prediction": prediction_payload,
    }
    # Run-level statuses are checked again on the self-contained row.  This
    # keeps the finalizer from publishing a row that was merely pending.
    _validate_view(
        view,
        field=field,
        source_audio_id=source_audio_id,
        source_audio_sha256=source_audio_sha256,
    )
    if field in _DERIVED_VIEW_NAMES and status == "pending":
        _fail(f"{field} remains pending")
    if successful and prediction_payload is None:
        _fail(f"successful {field} prediction evidence is missing")
    return view


def _comparison_payload(run_path: Path, subset_path: Path) -> dict[str, object]:
    del subset_path
    comparison_root = run_path.parent / "comparison"
    result: dict[str, object] = {}
    for key in _COMPARISON_ARTIFACT_KEYS:
        path = comparison_root / key
        try:
            content = read_regular_file_no_follow(path)
        except (OSError, TypeError) as error:
            raise SeparationHandoffError(f"comparison artifact is unavailable: {key}") from error
        result[key] = {"path": f"comparison/{key}", "sha256": sha256(content).hexdigest()}
    return result


def _oaf_parent_identity(snapshot: Mapping[str, object]) -> tuple[dict[str, str], Path]:
    """Load the immutable OaF identity needed by the handoff row.

    HPA-328 persists the exact parent run path in its authenticated run
    identity. The parent OaF snapshot owns the model and inference
    configuration fields, while its output root owns inherited predictions.
    No run-id-based directory discovery is permitted at finalization time.
    """
    parent_run_id = _text(snapshot.get("parent_oaf_run_id"), "parent_oaf_run_id")
    parent_path = _owner_root(snapshot.get("parent_oaf_run_path"), "parent_oaf_run_path")
    try:
        parent_content = read_regular_file_no_follow(parent_path)
        parent_value = strict_json_loads(parent_content, require_canonical=True)
    except (OSError, TypeError, StrictJsonError) as error:
        raise SeparationHandoffError("parent OaF run identity is unavailable") from error
    if not isinstance(parent_value, Mapping) or parent_value.get("run_id") != parent_run_id:
        _fail("parent OaF run identity is unavailable")
    parent: Mapping[str, object] = parent_value
    try:
        parent_output_root = parent_path.parents[2]
    except IndexError:
        _fail("parent OaF output root is unavailable")
        raise AssertionError("unreachable")
    inference_config = parent.get("inference_config")
    if not isinstance(inference_config, Mapping):
        inference_config = {}

    def _field_value(field: str, *, config_field: str | None = None) -> object:
        result = parent.get(field)
        if result is None:
            result = snapshot.get(field)
        if result is None and config_field is not None:
            result = inference_config.get(config_field)
        return result

    return (
        {
            "parent_oaf_run_id": parent_run_id,
            "oaf_model_id": _text(_field_value("model_id"), "oaf_model_id"),
            "oaf_backend_descriptor_sha256": _hash(
                _field_value("backend_descriptor_sha256"), "oaf_backend_descriptor_sha256"
            ),
            "oaf_model_lock_sha256": _hash(
                _field_value("model_lock_sha256"), "oaf_model_lock_sha256"
            ),
            "oaf_checkpoint_archive_sha256": _hash(
                _field_value("checkpoint_archive_sha256"), "oaf_checkpoint_archive_sha256"
            ),
            "oaf_adapter_revision": _text(
                _field_value("adapter_revision", config_field="adapter_revision"),
                "oaf_adapter_revision",
            ),
            "oaf_canonicalization_revision": _text(
                _field_value("canonicalization_revision", config_field="canonicalization_revision"),
                "oaf_canonicalization_revision",
            ),
            "oaf_inference_config_sha256": _hash(
                _field_value("inference_config_sha256"), "oaf_inference_config_sha256"
            ),
            "oaf_prediction_map_version": _text(
                _field_value("prediction_map_version", config_field="prediction_map_version"),
                "oaf_prediction_map_version",
            ),
        },
        parent_output_root,
    )


def _validate_view_oaf_identity(raw_item: Mapping[str, object], parent: Mapping[str, str]) -> None:
    """Ensure every persisted OaF view uses the parent model identity."""
    for field in ("spleeter", "htdemucs"):
        raw_view = raw_item.get(field)
        if not isinstance(raw_view, Mapping):
            _fail(f"{field} view evidence is invalid")
        config = raw_view.get("inference_config")
        if not isinstance(config, Mapping):
            _fail(f"{field} inference configuration is unavailable")
        for config_field, parent_field in (
            ("backend_descriptor_sha256", "oaf_backend_descriptor_sha256"),
            ("model_lock_sha256", "oaf_model_lock_sha256"),
            ("checkpoint_archive_sha256", "oaf_checkpoint_archive_sha256"),
            ("adapter_revision", "oaf_adapter_revision"),
            ("canonicalization_revision", "oaf_canonicalization_revision"),
            ("prediction_map_version", "oaf_prediction_map_version"),
        ):
            if config.get(config_field) != parent[parent_field]:
                _fail(f"{field} OaF identity does not match parent run")
        config_sha = raw_view.get("inference_config_sha256")
        _hash(config_sha, f"{field}.inference_config_sha256")


def _build_rows(
    snapshot: Mapping[str, object],
    subset: object,
    *,
    run_path: Path,
    subset_path: Path,
    decision: SeparationDecision,
    rationale: str,
    artifact_roots: dict[str, Path],
) -> tuple[dict[str, object], ...]:
    if snapshot.get("schema") != SEPARATION_RUN_SCHEMA:
        _fail("run snapshot schema is invalid")
    if snapshot.get("overall_status") not in {"complete", "partial"}:
        _fail("run snapshot is not closed")
    if not isinstance(subset, object) or not hasattr(subset, "rows"):
        _fail("reviewed subset manifest is invalid")
    subset_rows = getattr(subset, "rows")
    subset_by_id: dict[int, Mapping[str, object]] = {}
    for loaded in subset_rows:
        source_row = loaded.source_row
        simfile_id = loaded.view.simfile_id
        subset_by_id[simfile_id] = source_row
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        _fail("run snapshot items are unavailable")
    if set(item.get("simfile_id") for item in raw_items if isinstance(item, Mapping)) != set(
        subset_by_id
    ):
        _fail("run snapshot membership does not match HPA-327 subset")
    expected_subset_sha = getattr(subset, "manifest_sha256")
    if snapshot.get("reviewed_subset_manifest_sha256") != expected_subset_sha:
        _fail("run/subset manifest lineage does not match")
    for snapshot_field, subset_field in (
        (
            "reference_manifest_sha256",
            "source_reference_manifest_sha256",
        ),
        (
            "reference_manifest_version",
            "source_reference_manifest_version",
        ),
        (
            "reference_timing_manifest_sha256",
            "source_timing_manifest_sha256",
        ),
        (
            "reference_timing_version",
            "source_timing_manifest_version",
        ),
    ):
        if snapshot.get(snapshot_field) != getattr(subset, subset_field):
            _fail("run/subset source lineage does not match")
    for field in ("reference_manifest_version", "reference_timing_version"):
        _version(snapshot[field], field)
    for field in (
        "reference_manifest_sha256",
        "reference_timing_manifest_sha256",
        "oaf_backend_descriptor_sha256",
        "oaf_model_lock_sha256",
        "oaf_checkpoint_archive_sha256",
    ):
        _hash(snapshot[field], field)
    _text(snapshot["parent_oaf_run_id"], "parent_oaf_run_id")
    _commit(snapshot["crux_commit"])
    parent_identity, parent_prediction_root = _oaf_parent_identity(snapshot)
    artifact_roots["parent_prediction"] = parent_prediction_root
    run_id = _text(snapshot.get("run_id"), "run_id")
    if _RUN_ID_RE.fullmatch(run_id) is None:
        _fail("run_id is invalid")
    comparison = _comparison_payload(run_path, subset_path)
    rows: list[dict[str, object]] = []
    for raw_item in sorted(raw_items, key=lambda item: item["simfile_id"]):
        if not isinstance(raw_item, Mapping):
            _fail("run snapshot item is invalid")
        simfile_id = raw_item.get("simfile_id")
        if not isinstance(simfile_id, int) or isinstance(simfile_id, bool):
            _fail("run snapshot item simfile_id is invalid")
        subset_row = subset_by_id.get(simfile_id)
        if subset_row is None:
            _fail("run snapshot member is absent from subset")
        for field in (
            "source_row_sha256",
            "source_audio_sha256",
        ):
            if (
                raw_item.get(field)
                != subset_row[
                    field if field == "source_row_sha256" else "source_audio_content_hash"
                ]
            ):
                _fail(f"{field} does not match reviewed subset")
        source_audio_id = _text(raw_item.get("source_audio_id"), "source_audio_id")
        source_audio_sha256 = _hash(raw_item.get("source_audio_sha256"), "source_audio_sha256")
        _validate_view_oaf_identity(raw_item, parent_identity)
        for view_name in ("spleeter", "htdemucs"):
            raw_view = raw_item[view_name]
            if isinstance(raw_view, Mapping) and isinstance(raw_view.get("stem"), Mapping):
                cache_root = _owner_root(raw_view["stem"].get("owner_root"), f"{view_name}.stem")
                previous_cache_root = artifact_roots.setdefault("cache", cache_root)
                if previous_cache_root != cache_root:
                    _fail("derived stem owner roots are inconsistent")
        row = {
            "schema_version": SEPARATION_PILOT_SCHEMA,
            "separation_run_id": run_id,
            "simfile_id": simfile_id,
            "source_row_sha256": subset_row["source_row_sha256"],
            "reviewed_subset_manifest_sha256": getattr(subset, "manifest_sha256"),
            "reviewed_subset_manifest_version": getattr(subset, "corpus_version"),
            "reference_manifest_sha256": snapshot["reference_manifest_sha256"],
            "reference_manifest_version": snapshot["reference_manifest_version"],
            "reference_timing_manifest_sha256": snapshot["reference_timing_manifest_sha256"],
            "reference_timing_version": snapshot["reference_timing_version"],
            "source_audio_id": source_audio_id,
            "source_audio_sha256": source_audio_sha256,
            "source_duration_sec": _manifest_duration(
                raw_item.get("source_duration_sec"), "source_duration_sec"
            ),
            "parent_oaf_run_id": parent_identity["parent_oaf_run_id"],
            "oaf_model_id": parent_identity["oaf_model_id"],
            "oaf_backend_descriptor_sha256": parent_identity["oaf_backend_descriptor_sha256"],
            "oaf_model_lock_sha256": parent_identity["oaf_model_lock_sha256"],
            "oaf_checkpoint_archive_sha256": parent_identity["oaf_checkpoint_archive_sha256"],
            "oaf_adapter_revision": parent_identity["oaf_adapter_revision"],
            "oaf_canonicalization_revision": parent_identity["oaf_canonicalization_revision"],
            "oaf_inference_config_sha256": parent_identity["oaf_inference_config_sha256"],
            "oaf_prediction_map_version": parent_identity["oaf_prediction_map_version"],
            "crux_commit": snapshot["crux_commit"],
            "full_mix": _view_payload(
                raw_item["full_mix"],
                field="full_mix",
                item=raw_item,
                run_path=run_path,
                subset_path=subset_path,
                parent_prediction_root=parent_prediction_root,
            ),
            "spleeter": _view_payload(
                raw_item["spleeter"],
                field="spleeter",
                item=raw_item,
                run_path=run_path,
                subset_path=subset_path,
                parent_prediction_root=parent_prediction_root,
            ),
            "htdemucs": _view_payload(
                raw_item["htdemucs"],
                field="htdemucs",
                item=raw_item,
                run_path=run_path,
                subset_path=subset_path,
                parent_prediction_root=parent_prediction_root,
            ),
            "comparison_artifacts": comparison,
            "decision": decision,
            "rationale": rationale,
        }
        _validate_row(row, allow_corpus_version=False)
        rows.append(row)
    return tuple(rows)


def _rehash_rows(
    rows: tuple[dict[str, object], ...],
    *,
    run_path: Path,
    subset_path: Path,
    artifact_roots: Mapping[str, Path],
) -> None:
    """Re-read retained evidence after rendering and immediately before publish."""
    parent_prediction_root = artifact_roots.get("parent_prediction")
    cache_root = artifact_roots.get("cache")
    for row in rows:
        for field in ("spleeter", "htdemucs"):
            view = row[field]
            if not isinstance(view, Mapping) or view["status"] not in {"success", "resumed"}:
                continue
            stem = view["stem"]
            prediction = view["prediction"]
            assert isinstance(stem, Mapping) and isinstance(prediction, Mapping)
            _read_matching_artifact(
                stem["path"],
                stem["sha256"],
                field=f"{field}.stem",
                run_path=run_path,
                subset_path=subset_path,
                owner_root=cache_root,
            )
            _read_matching_artifact(
                prediction["path"],
                prediction["artifact_sha256"],
                field=f"{field}.prediction",
                run_path=run_path,
                subset_path=subset_path,
            )
        full_mix = row["full_mix"]
        if isinstance(full_mix, Mapping) and full_mix["status"] in {"inferred", "resumed"}:
            prediction = full_mix["prediction"]
            assert isinstance(prediction, Mapping)
            _read_matching_artifact(
                prediction["path"],
                prediction["artifact_sha256"],
                field="full_mix.prediction",
                run_path=run_path,
                subset_path=subset_path,
                owner_root=parent_prediction_root,
            )
        comparison = row["comparison_artifacts"]
        assert isinstance(comparison, Mapping)
        for key in _COMPARISON_ARTIFACT_KEYS:
            artifact = comparison[key]
            assert isinstance(artifact, Mapping)
            _read_matching_artifact(
                artifact["path"],
                artifact["sha256"],
                field=f"comparison_artifacts.{key}",
                run_path=run_path,
                subset_path=subset_path,
            )


def _direct_manifest_alias(output_manifest: Path, published: object, content: bytes) -> None:
    """Expose the requested path while retaining the content-addressed rail."""
    published_path = getattr(published, "path")
    if published_path == output_manifest:
        return
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest.exists():
        try:
            if read_regular_file_no_follow(output_manifest) != content:
                _fail("output manifest already exists with different bytes")
        except (OSError, TypeError) as error:
            raise SeparationHandoffError("output manifest is unavailable") from error
        return
    try:
        os.link(published_path, output_manifest)
    except OSError as error:
        raise SeparationHandoffError("output manifest alias could not be published") from error


def finalize_separation_pilot(
    request: FinalizeSeparationPilotRequest,
) -> FinalizeSeparationPilotOutcome:
    """Validate and publish one closed HPA-396 handoff decision."""
    try:
        if not isinstance(request, FinalizeSeparationPilotRequest):
            raise TypeError("request must be FinalizeSeparationPilotRequest")
        subset = load_reviewed_subset_manifest(request.subset_manifest_path)
        try:
            run_content = read_regular_file_no_follow(request.run_path)
            snapshot = parse_oaf_separation_run(run_content)
        except (OSError, TypeError, StrictJsonError, ValueError) as error:
            raise SeparationHandoffError("run snapshot is invalid") from error
        artifact_roots: dict[str, Path] = {}
        rows = _build_rows(
            snapshot,
            subset,
            run_path=request.run_path,
            subset_path=request.subset_manifest_path,
            decision=request.decision,
            rationale=request.rationale,
            artifact_roots=artifact_roots,
        )
        rendered = render_manifest(rows)
        # The second read is intentionally after the manifest bytes are fixed,
        # closing the TOCTOU window before the immutable publication call.
        _rehash_rows(
            rows,
            run_path=request.run_path,
            subset_path=request.subset_manifest_path,
            artifact_roots=artifact_roots,
        )
        published = publish_manifest(request.output_manifest.parent, rendered)
        _direct_manifest_alias(request.output_manifest, published, rendered.content)
        return FinalizeSeparationPilotOutcome(exit_code=0, manifest=published)
    except (
        ManifestPublicationError,
        OSError,
        PredictionArtifactError,
        SeparationHandoffError,
        StrictJsonError,
        TypeError,
        ValueError,
    ):
        return FinalizeSeparationPilotOutcome(exit_code=2, manifest=None)


__all__ = [
    "FinalizeSeparationPilotOutcome",
    "FinalizeSeparationPilotRequest",
    "LoadedSeparationPilotManifest",
    "SEPARATION_PILOT_SCHEMA",
    "SeparationDecision",
    "SeparationHandoffError",
    "finalize_separation_pilot",
    "load_separation_pilot_manifest",
    "validate_schema_golden",
]
