from __future__ import annotations

# The v1 record shapes mandate the public field counts. Validation is deliberately
# branch-heavy because each backend and status has distinct nullability rules.
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-branches,too-many-lines
# pylint: disable=unidiomatic-typecheck,broad-exception-caught,duplicate-code
import fcntl
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Iterator, Literal, cast
from uuid import UUID, uuid4

from src.benchmark.backend_identity import (
    HEURISTIC_BACKEND_ID as _HEURISTIC_BACKEND_ID,
)
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID as _OAF_BACKEND_ID,
)
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_publication import (
    DirectoryAnchor,
    atomic_replace_bytes,
    open_lock_file_no_follow,
    publish_immutable_bytes,
    read_regular_file_no_follow,
    unlink_regular_file_no_follow,
)
from src.benchmark.backends import BackendError, PublishedArtifact

_VERIFICATION_SCHEMA = "crux.backend-verification-report/v1"
_EXECUTION_SCHEMA = "crux.backend-execution-report/v1"
_LEGACY_SCHEMA = "crux.legacy-score-report/v1"
_ITEM_ID_SCHEMA = "crux.backend-execution-item-id/v1"
_LEGACY_BACKEND_ID = "legacy-tf2-h5-v0"
UNAVAILABLE_BACKEND_REPORT_ID = "backend-unavailable"
_UNAVAILABLE_NAMESPACE_ERROR = "unavailable backend requires exact unavailable execution failure"
_TIMESTAMP_PATTERN = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")
_LATEST_UNKNOWN = object()
_LATEST_ABSENT = object()
_VERIFICATION_KEYS = frozenset(
    {
        "schema",
        "report_type",
        "run_id",
        "started_at",
        "finished_at",
        "status",
        "exit_code",
        "descriptor",
        "descriptor_sha256",
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "parameter_lock_sha256",
        "seal_evidence_sha256",
        "execution_attestation",
        "tensor_coverage",
        "smoke",
        "artifacts",
        "errors",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "schema",
        "report_type",
        "run_id",
        "started_at",
        "finished_at",
        "status",
        "exit_code",
        "descriptor",
        "descriptor_sha256",
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "parameter_lock_sha256",
        "seal_evidence_sha256",
        "execution_attestation",
        "verification_report",
        "items",
        "errors",
    }
)
_LEGACY_KEYS = frozenset(
    {
        "schema",
        "report_type",
        "run_id",
        "started_at",
        "finished_at",
        "workflow_mode",
        "backend_id",
        "backend_validation_status",
        "status",
        "exit_code",
        "score_report",
        "errors",
    }
)
_ITEM_KEYS = frozenset(
    {
        "item_id",
        "source_audio_id",
        "source_audio_sha256",
        "input_view_id",
        "input_audio_sha256",
        "status",
        "prediction",
        "midi",
        "errors",
    }
)
_TENSOR_KEYS = frozenset(
    {
        "status",
        "required_count",
        "restored_count",
        "non_inference_count",
        "required_inventory_sha256",
        "non_inference_inventory_sha256",
        "report_path",
        "report_sha256",
    }
)
_SMOKE_KEYS = frozenset(
    {
        "status",
        "audio_sha256",
        "oracle_sha256",
        "prediction_path",
        "prediction_sha256",
    }
)


class ReportValidationError(ValueError):
    pass


class OperationalReportPublicationError(OSError):
    pass


@dataclass(frozen=True)
class ExecutionItem:
    item_id: str
    source_audio_id: str
    source_audio_sha256: str
    input_view_id: str
    input_audio_sha256: str
    status: Literal["complete", "incomplete", "failed"]
    prediction: PublishedArtifact | None
    midi: PublishedArtifact | None
    errors: tuple[BackendError, ...]

    def __post_init__(self) -> None:
        _normalize_item(_execution_item_payload(self))


@dataclass(frozen=True)
class VerificationReport:
    payload: Mapping[str, JsonValue]
    _canonical_bytes: bytes = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _install_report_snapshot(self, _normalize_verification(self.payload))


@dataclass(frozen=True)
class ExecutionReport:
    payload: Mapping[str, JsonValue]
    _canonical_bytes: bytes = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _install_report_snapshot(self, _normalize_execution(self.payload))


@dataclass(frozen=True)
class LegacyScoreReport:
    payload: Mapping[str, JsonValue]
    _canonical_bytes: bytes = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _install_report_snapshot(self, _normalize_legacy(self.payload))


OperationalReport = VerificationReport | ExecutionReport | LegacyScoreReport


def derive_item_id(
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    input_view_id: str,
    input_audio_sha256: str,
) -> str:
    _require_string(source_audio_id, "source_audio_id")
    _require_hash(source_audio_sha256, "source_audio_sha256")
    _require_string(input_view_id, "input_view_id")
    _require_hash(input_audio_sha256, "input_audio_sha256")
    payload: dict[str, JsonValue] = {
        "input_audio_sha256": input_audio_sha256,
        "input_view_id": input_view_id,
        "schema": _ITEM_ID_SCHEMA,
        "source_audio_id": source_audio_id,
        "source_audio_sha256": source_audio_sha256,
    }
    return f"sha256:{sha256_hex(canonical_json_bytes(payload))}"


# The optional retained root anchor is part of the security boundary.
# pylint: disable-next=too-many-arguments
def publish_operational_report(
    reports_root: Path,
    *,
    backend_id: str,
    report: OperationalReport,
    now: datetime | None = None,
    run_id: UUID | None = None,
    anchor: DirectoryAnchor | None = None,
) -> PublishedArtifact:
    try:
        _validate_backend_id(backend_id)
        effective_now = datetime.now(UTC) if now is None else now
        effective_run_id = uuid4() if run_id is None else run_id
        timestamp = _filename_timestamp(effective_now)
        run_id_text = _uuid4_text(effective_run_id)
        snapshot = _report_snapshot(report)
        if snapshot["run_id"] != run_id_text:
            raise ReportValidationError("report run_id does not match publication run_id")
        _validate_publication_namespace(backend_id, report, snapshot)
        latest_name, role = _report_publication_kind(report)
        content = report._canonical_bytes  # pylint: disable=protected-access
        digest = sha256_hex(content)
        backend_root = reports_root / backend_id
        immutable_path = backend_root / "reports" / f"{timestamp}-{run_id_text}.json"
        published = publish_immutable_bytes(
            immutable_path,
            content,
            digest,
            role=role,
            anchor=anchor,
        )
    except Exception:
        raise OperationalReportPublicationError("operational_report_publication_failed") from None

    latest_path = backend_root / latest_name
    try:
        with _latest_namespace_lock(backend_root, anchor=anchor):
            previous_state: bytes | object = _LATEST_UNKNOWN
            replacement_may_have_happened = False
            try:
                previous_content = _read_optional_regular_file(latest_path, anchor=anchor)
                previous_state = _LATEST_ABSENT if previous_content is None else previous_content
                replacement_may_have_happened = True
                if anchor is None:
                    atomic_replace_bytes(latest_path, content)
                else:
                    atomic_replace_bytes(latest_path, content, anchor=anchor)
            except Exception:
                if replacement_may_have_happened and previous_state is not _LATEST_UNKNOWN:
                    try:
                        _restore_latest(latest_path, previous_state, anchor=anchor)
                    except Exception:
                        pass
                raise
    except Exception:
        raise OperationalReportPublicationError("operational_report_publication_failed") from None
    return published


def _normalize_verification(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = _copy_exact_mapping(payload, _VERIFICATION_KEYS, "verification report")
    _validate_common_report(
        normalized,
        schema=_VERIFICATION_SCHEMA,
        report_type="verification",
        statuses={"verified", "failed", "environment_unsupported"},
    )
    _require_status_exit_code(
        normalized,
        {
            "verified": 0,
            "failed": 2,
            "environment_unsupported": 1,
        },
    )
    _normalize_identity_fields(normalized)
    normalized["execution_attestation"] = _normalize_reference(
        normalized["execution_attestation"],
        "execution_attestation",
    )
    normalized["tensor_coverage"] = _normalize_tensor_coverage(normalized["tensor_coverage"])
    normalized["smoke"] = _normalize_smoke(normalized["smoke"])
    normalized["artifacts"] = _normalize_artifacts(normalized["artifacts"])
    normalized["errors"] = _normalize_errors(normalized["errors"])
    if normalized["status"] == "verified":
        if normalized["errors"]:
            raise ReportValidationError("complete verification cannot carry errors")
        _validate_verified_report(normalized)
    elif not normalized["errors"]:
        raise ReportValidationError(f"{normalized['status']} report requires errors")
    return _canonical_payload(normalized)


def _normalize_execution(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = _copy_exact_mapping(payload, _EXECUTION_KEYS, "execution report")
    _validate_common_report(
        normalized,
        schema=_EXECUTION_SCHEMA,
        report_type="execution",
        statuses={"complete", "partial", "failed", "environment_unsupported"},
    )
    _require_status_exit_code(
        normalized,
        {
            "complete": 0,
            "partial": 1,
            "failed": 2,
            "environment_unsupported": 1,
        },
    )
    _normalize_identity_fields(normalized)
    normalized["execution_attestation"] = _normalize_reference(
        normalized["execution_attestation"],
        "execution_attestation",
    )
    normalized["verification_report"] = _normalize_reference(
        normalized["verification_report"],
        "verification_report",
    )
    items = normalized["items"]
    if not isinstance(items, (list, tuple)):
        raise ReportValidationError("items must be an array")
    normalized["items"] = [
        _normalize_item(_execution_item_payload(item) if isinstance(item, ExecutionItem) else item)
        for item in items
    ]
    normalized["errors"] = _normalize_errors(normalized["errors"])
    if normalized["status"] == "complete":
        if normalized["errors"]:
            raise ReportValidationError("complete execution cannot carry errors")
        if normalized["exit_code"] != 0:
            raise ReportValidationError("complete execution exit_code must be zero")
        if any(
            item["status"] != "complete"
            for item in cast(list[dict[str, JsonValue]], normalized["items"])
        ):
            raise ReportValidationError("complete execution requires complete items")
        _validate_complete_execution(normalized)
    elif normalized["status"] in {"failed", "environment_unsupported"}:
        if not normalized["errors"]:
            raise ReportValidationError(f"{normalized['status']} report requires errors")
        if normalized["items"]:
            raise ReportValidationError(f"{normalized['status']} execution cannot carry items")
    elif not normalized["errors"] and all(
        item["status"] == "complete"
        for item in cast(list[dict[str, JsonValue]], normalized["items"])
    ):
        raise ReportValidationError("partial execution requires failure evidence")
    return _canonical_payload(normalized)


def _normalize_legacy(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = _copy_exact_mapping(payload, _LEGACY_KEYS, "legacy-score report")
    _validate_common_report(
        normalized,
        schema=_LEGACY_SCHEMA,
        report_type="legacy_score",
        statuses={"complete", "failed", "canonical_mapping_required"},
    )
    _require_status_exit_code(
        normalized,
        {
            "complete": 0,
            "failed": 1,
            "canonical_mapping_required": 1,
        },
    )
    if normalized["workflow_mode"] != "legacy_compatibility":
        raise ReportValidationError("workflow_mode must be legacy_compatibility")
    backend_id = normalized["backend_id"]
    _require_string(backend_id, "backend_id")
    validation_status = normalized["backend_validation_status"]
    expected_validation = {
        _LEGACY_BACKEND_ID: "unvalidated",
        _HEURISTIC_BACKEND_ID: "parameter_locked",
        _OAF_BACKEND_ID: "not_checked",
    }.get(cast(str, backend_id))
    if expected_validation is None or validation_status != expected_validation:
        raise ReportValidationError("backend_validation_status does not match backend_id")
    normalized["score_report"] = _normalize_reference(
        normalized["score_report"],
        "score_report",
    )
    normalized["errors"] = _normalize_errors(normalized["errors"])
    if normalized["status"] == "canonical_mapping_required":
        if backend_id != _OAF_BACKEND_ID:
            raise ReportValidationError("canonical mapping status requires the OaF backend")
        if normalized["exit_code"] != 1 or normalized["score_report"] is not None:
            raise ReportValidationError(
                "canonical mapping status requires exit one and null score_report"
            )
        if not normalized["errors"]:
            raise ReportValidationError("canonical mapping status requires errors")
    if normalized["status"] == "complete":
        if normalized["errors"]:
            raise ReportValidationError("complete legacy report cannot carry errors")
        if normalized["exit_code"] != 0 or normalized["score_report"] is None:
            raise ReportValidationError("complete legacy score requires a score report")
    elif normalized["status"] == "failed":
        if not normalized["errors"]:
            raise ReportValidationError("failed report requires errors")
        if normalized["score_report"] is not None:
            raise ReportValidationError("failed legacy report cannot carry score_report")
    return _canonical_payload(normalized)


def _validate_common_report(
    payload: dict[str, object],
    *,
    schema: str,
    report_type: str,
    statuses: set[str],
) -> None:
    if payload["schema"] != schema:
        raise ReportValidationError(f"schema must be {schema}")
    if payload["report_type"] != report_type:
        raise ReportValidationError(f"report_type must be {report_type}")
    _parse_uuid4(payload["run_id"], "run_id")
    started = _parse_timestamp(payload["started_at"], "started_at")
    finished = _parse_timestamp(payload["finished_at"], "finished_at")
    if finished < started:
        raise ReportValidationError("finished_at must not precede started_at")
    if payload["status"] not in statuses:
        raise ReportValidationError("status is invalid")
    if type(payload["exit_code"]) is not int:
        raise ReportValidationError("exit_code must be an integer")


def _normalize_identity_fields(payload: dict[str, object]) -> None:
    descriptor = payload["descriptor"]
    descriptor_sha256 = payload["descriptor_sha256"]
    if descriptor is None:
        if descriptor_sha256 is not None:
            raise ReportValidationError("descriptor_sha256 requires descriptor")
    else:
        normalized_descriptor = _normalize_descriptor(descriptor)
        descriptor_content = canonical_json_bytes(normalized_descriptor)
        _require_hash(descriptor_sha256, "descriptor_sha256")
        if descriptor_sha256 != sha256_hex(descriptor_content):
            raise ReportValidationError("descriptor_sha256 does not match descriptor")
        payload["descriptor"] = normalized_descriptor
    for field in (
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "parameter_lock_sha256",
        "seal_evidence_sha256",
    ):
        value = payload[field]
        if value is not None:
            _require_hash(value, field)
    normalized = payload["descriptor"]
    if isinstance(normalized, dict):
        backend_id = normalized["backend_id"]
        if backend_id == _OAF_BACKEND_ID:
            _require_matching_identity(payload, normalized, "backend_lock_sha256")
            _require_matching_identity(payload, normalized, "runtime_lock_sha256")
        elif backend_id == _HEURISTIC_BACKEND_ID:
            _require_matching_identity(payload, normalized, "parameter_lock_sha256")


def _normalize_descriptor(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ReportValidationError("descriptor must be an object or null")
    try:
        descriptor = normalize_known_backend_descriptor(value)
    except StrictJsonError as error:
        raise ReportValidationError(str(error)) from None
    return _canonical_payload(cast(dict[str, object], descriptor))


def _require_matching_identity(
    report: dict[str, object],
    descriptor: dict[str, JsonValue],
    field: str,
) -> None:
    if report[field] != descriptor[field]:
        raise ReportValidationError(f"{field} does not match descriptor")


def _require_status_exit_code(
    payload: dict[str, object],
    expected_codes: dict[str, int],
) -> None:
    status = cast(str, payload["status"])
    if payload["exit_code"] != expected_codes[status]:
        raise ReportValidationError(f"{status} status has an invalid exit_code")


def _normalize_tensor_coverage(value: object) -> dict[str, JsonValue]:
    tensor = _copy_exact_mapping(value, _TENSOR_KEYS, "tensor_coverage")
    if tensor["status"] not in {"passed", "failed", "not_run", "not_applicable"}:
        raise ReportValidationError("tensor_coverage status is invalid")
    for field in ("required_count", "restored_count", "non_inference_count"):
        if type(tensor[field]) is not int or cast(int, tensor[field]) < 0:
            raise ReportValidationError(f"tensor_coverage {field} must be nonnegative")
    for field in ("required_inventory_sha256", "non_inference_inventory_sha256"):
        if tensor[field] is not None:
            _require_hash(tensor[field], f"tensor_coverage {field}")
    _validate_path_hash_pair(tensor, "report_path", "report_sha256", "tensor_coverage")
    return _canonical_payload(tensor)


def _normalize_smoke(value: object) -> dict[str, JsonValue]:
    smoke = _copy_exact_mapping(value, _SMOKE_KEYS, "smoke")
    if smoke["status"] not in {"passed", "failed", "not_run", "not_applicable"}:
        raise ReportValidationError("smoke status is invalid")
    for field in ("audio_sha256", "oracle_sha256"):
        if smoke[field] is not None:
            _require_hash(smoke[field], f"smoke {field}")
    _validate_path_hash_pair(smoke, "prediction_path", "prediction_sha256", "smoke")
    return _canonical_payload(smoke)


def _normalize_item(value: object) -> dict[str, JsonValue]:
    item = _copy_exact_mapping(value, _ITEM_KEYS, "execution item")
    _require_string(item["source_audio_id"], "source_audio_id")
    _require_hash(item["source_audio_sha256"], "source_audio_sha256")
    _require_string(item["input_view_id"], "input_view_id")
    _require_hash(item["input_audio_sha256"], "input_audio_sha256")
    expected_item_id = derive_item_id(
        source_audio_id=cast(str, item["source_audio_id"]),
        source_audio_sha256=cast(str, item["source_audio_sha256"]),
        input_view_id=cast(str, item["input_view_id"]),
        input_audio_sha256=cast(str, item["input_audio_sha256"]),
    )
    if item["item_id"] != expected_item_id:
        raise ReportValidationError("item_id does not match item identity")
    if item["status"] not in {"complete", "incomplete", "failed"}:
        raise ReportValidationError("item status is invalid")
    item["prediction"] = _normalize_reference(item["prediction"], "prediction")
    item["midi"] = _normalize_reference(item["midi"], "midi")
    item["errors"] = _normalize_errors(item["errors"])
    if item["status"] == "complete":
        if item["prediction"] is None:
            raise ReportValidationError("complete item requires prediction")
        if item["errors"]:
            raise ReportValidationError("complete item cannot carry errors")
    elif item["status"] == "incomplete":
        if item["prediction"] is None:
            raise ReportValidationError("incomplete item requires prediction")
        if item["midi"] is not None:
            raise ReportValidationError("incomplete item cannot carry MIDI")
        if not item["errors"]:
            raise ReportValidationError("incomplete item requires errors")
    else:
        if item["prediction"] is not None or item["midi"] is not None:
            raise ReportValidationError("failed item cannot carry success artifacts")
        if not item["errors"]:
            raise ReportValidationError("failed item requires errors")
    return _canonical_payload(item)


def _normalize_reference(value: object, field: str) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    if isinstance(value, PublishedArtifact):
        value = {"path": value.path.as_posix(), "sha256": value.sha256}
    reference = _copy_exact_mapping(value, {"path", "sha256"}, field)
    _validate_persisted_path(reference["path"], f"{field} path")
    _require_hash(reference["sha256"], f"{field} sha256")
    return _canonical_payload(reference)


def _normalize_artifacts(value: object) -> list[JsonValue]:
    if not isinstance(value, (list, tuple)):
        raise ReportValidationError("artifacts must be an array")
    artifacts: list[dict[str, JsonValue]] = []
    for entry in value:
        if isinstance(entry, PublishedArtifact):
            entry = {
                "role": entry.role,
                "path": entry.path.as_posix(),
                "sha256": entry.sha256,
            }
        artifact = _copy_exact_mapping(entry, {"role", "path", "sha256"}, "artifact")
        _require_string(artifact["role"], "artifact role")
        _validate_persisted_path(artifact["path"], "artifact path")
        _require_hash(artifact["sha256"], "artifact sha256")
        artifacts.append(_canonical_payload(artifact))
    return sorted(
        artifacts,
        key=lambda artifact: (
            cast(str, artifact["role"]).encode("utf-8"),
            cast(str, artifact["path"]).encode("utf-8"),
        ),
    )


def _normalize_errors(value: object) -> list[JsonValue]:
    if not isinstance(value, (list, tuple)):
        raise ReportValidationError("errors must be an array")
    errors: list[dict[str, JsonValue]] = []
    for entry in value:
        if isinstance(entry, BackendError):
            entry = {"code": entry.code, "message": entry.message}
        error = _copy_exact_mapping(entry, {"code", "message"}, "error")
        _require_string(error["code"], "error code")
        _require_string(error["message"], "error message")
        try:
            validated = BackendError(
                code=cast(str, error["code"]),
                message=cast(str, error["message"]),
            )
        except (TypeError, ValueError) as failure:
            raise ReportValidationError(str(failure)) from None
        errors.append(
            _canonical_payload(
                {
                    "code": validated.code,
                    "message": validated.message,
                }
            )
        )
    return sorted(
        errors,
        key=lambda error: (
            cast(str, error["code"]).encode("utf-8"),
            cast(str, error["message"]).encode("utf-8"),
        ),
    )


def _validate_verified_report(payload: dict[str, JsonValue]) -> None:
    for field in ("descriptor", "descriptor_sha256", "execution_attestation"):
        if payload[field] is None:
            raise ReportValidationError(f"verified report requires {field}")
    descriptor = cast(dict[str, JsonValue], payload["descriptor"])
    backend_id = descriptor.get("backend_id")
    tensor = cast(dict[str, JsonValue], payload["tensor_coverage"])
    smoke = cast(dict[str, JsonValue], payload["smoke"])
    if payload["exit_code"] != 0:
        raise ReportValidationError("verified report exit_code must be zero")
    if backend_id == _OAF_BACKEND_ID:
        for field in (
            "backend_lock_sha256",
            "runtime_lock_sha256",
            "seal_evidence_sha256",
        ):
            if payload[field] is None:
                raise ReportValidationError(f"verified OaF report requires {field}")
        if tensor["status"] != "passed":
            raise ReportValidationError("verified OaF tensor_coverage must pass")
        if smoke["status"] != "passed":
            raise ReportValidationError("verified OaF smoke must pass")
        for field in (
            "required_inventory_sha256",
            "non_inference_inventory_sha256",
            "report_path",
            "report_sha256",
        ):
            if tensor[field] is None:
                raise ReportValidationError(f"verified OaF tensor_coverage requires {field}")
        for field in (
            "audio_sha256",
            "oracle_sha256",
            "prediction_path",
            "prediction_sha256",
        ):
            if smoke[field] is None:
                raise ReportValidationError(f"verified OaF smoke requires {field}")
    elif backend_id == _HEURISTIC_BACKEND_ID:
        if payload["parameter_lock_sha256"] is None:
            raise ReportValidationError("verified heuristic requires parameter_lock_sha256")
        for field in (
            "backend_lock_sha256",
            "runtime_lock_sha256",
            "seal_evidence_sha256",
        ):
            if payload[field] is not None:
                raise ReportValidationError(f"verified heuristic requires null {field}")
        if tensor["status"] != "not_applicable" or smoke["status"] != "not_applicable":
            raise ReportValidationError("heuristic checks must be not_applicable")
        if any(
            tensor[field] != 0
            for field in (
                "required_count",
                "restored_count",
                "non_inference_count",
            )
        ):
            raise ReportValidationError("heuristic tensor counts must be zero")
        if any(
            tensor[field] is not None
            for field in (
                "required_inventory_sha256",
                "non_inference_inventory_sha256",
                "report_path",
                "report_sha256",
            )
        ):
            raise ReportValidationError("heuristic tensor artifacts must be null")
        if any(
            smoke[field] is not None
            for field in (
                "audio_sha256",
                "oracle_sha256",
                "prediction_path",
                "prediction_sha256",
            )
        ):
            raise ReportValidationError("heuristic smoke artifacts must be null")


def _validate_complete_execution(payload: dict[str, JsonValue]) -> None:
    for field in (
        "descriptor",
        "descriptor_sha256",
        "execution_attestation",
    ):
        if payload[field] is None:
            raise ReportValidationError(f"complete execution requires {field}")
    descriptor = cast(dict[str, JsonValue], payload["descriptor"])
    backend_id = descriptor.get("backend_id")
    if backend_id == _OAF_BACKEND_ID:
        for field in (
            "backend_lock_sha256",
            "runtime_lock_sha256",
            "seal_evidence_sha256",
        ):
            if payload[field] is None:
                raise ReportValidationError(f"complete OaF execution requires {field}")
    elif backend_id == _HEURISTIC_BACKEND_ID:
        if payload["parameter_lock_sha256"] is None:
            raise ReportValidationError(
                "complete heuristic execution requires parameter_lock_sha256"
            )
        for field in (
            "backend_lock_sha256",
            "runtime_lock_sha256",
            "seal_evidence_sha256",
        ):
            if payload[field] is not None:
                raise ReportValidationError(f"complete heuristic execution requires null {field}")


def _execution_item_payload(item: ExecutionItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "source_audio_id": item.source_audio_id,
        "source_audio_sha256": item.source_audio_sha256,
        "input_view_id": item.input_view_id,
        "input_audio_sha256": item.input_audio_sha256,
        "status": item.status,
        "prediction": item.prediction,
        "midi": item.midi,
        "errors": item.errors,
    }


def _copy_exact_mapping(
    value: object,
    keys: set[str] | frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ReportValidationError(f"{name} must contain the exact key set")
    return dict(value)


def _canonical_payload(value: dict[str, object]) -> dict[str, JsonValue]:
    try:
        canonical_json_bytes(cast(dict[str, JsonValue], value))
    except (TypeError, ValueError):
        raise ReportValidationError("report payload must contain canonical JSON values") from None
    return cast(dict[str, JsonValue], value)


def _install_report_snapshot(
    report: VerificationReport | ExecutionReport | LegacyScoreReport,
    normalized: dict[str, JsonValue],
) -> None:
    content = canonical_json_bytes(normalized, trailing_newline=True)
    detached = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(detached, dict):
        raise ReportValidationError("report payload must be an object")
    object.__setattr__(report, "payload", MappingProxyType(detached))
    object.__setattr__(report, "_canonical_bytes", content)


def _report_snapshot(report: OperationalReport) -> dict[str, JsonValue]:
    snapshot = strict_json_loads(
        report._canonical_bytes[:-1],  # pylint: disable=protected-access
        require_canonical=True,
    )
    if not isinstance(snapshot, dict):
        raise ReportValidationError("report snapshot must be an object")
    return snapshot


def _validate_path_hash_pair(
    payload: dict[str, object],
    path_field: str,
    hash_field: str,
    name: str,
) -> None:
    path = payload[path_field]
    digest = payload[hash_field]
    if (path is None) != (digest is None):
        raise ReportValidationError(f"{name} path and sha256 must both be null or present")
    if path is not None:
        _validate_persisted_path(path, f"{name} {path_field}")
        _require_hash(digest, f"{name} {hash_field}")


def _validate_persisted_path(value: object, field: str) -> None:
    _require_string(value, field)
    path = cast(str, value)
    if path.startswith("/") or "\\" in path or path.endswith("/"):
        raise ReportValidationError(f"{field} must be repository-relative POSIX")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(path).is_absolute():
        raise ReportValidationError(f"{field} must be repository-relative POSIX")


def _require_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ReportValidationError(f"{field} must be a nonempty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ReportValidationError(f"{field} must be valid UTF-8") from None


def _require_hash(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ReportValidationError(f"{field} must be lowercase SHA-256")
    try:
        require_sha256(value, field)
    except ValueError:
        raise ReportValidationError(f"{field} must be lowercase SHA-256") from None


def _parse_uuid4(value: object, field: str) -> UUID:
    if not isinstance(value, str) or value != value.lower():
        raise ReportValidationError(f"{field} must be lowercase UUIDv4")
    try:
        parsed = UUID(value)
    except ValueError:
        raise ReportValidationError(f"{field} must be lowercase UUIDv4") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ReportValidationError(f"{field} must be lowercase UUIDv4")
    return parsed


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ReportValidationError(f"{field} must be a microsecond UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        raise ReportValidationError(f"{field} must be a valid UTC timestamp") from None


def _validate_backend_id(value: object) -> None:
    _require_string(value, "backend_id")
    backend_id = cast(str, value)
    if backend_id in {".", ".."} or "/" in backend_id or "\\" in backend_id:
        raise ReportValidationError("backend_id must be a single path component")


def _filename_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ReportValidationError("publication time must be UTC")
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _uuid4_text(value: UUID) -> str:
    if not isinstance(value, UUID) or value.version != 4:
        raise ReportValidationError("publication run_id must be UUIDv4")
    return str(value)


def _report_publication_kind(report: OperationalReport) -> tuple[str, str]:
    if isinstance(report, VerificationReport):
        return "latest-verification.json", "verification_report"
    if isinstance(report, ExecutionReport):
        return "latest-execution.json", "execution_report"
    if isinstance(report, LegacyScoreReport):
        return "latest-legacy-score.json", "legacy_score_report"
    raise ReportValidationError("unknown operational report type")


def _validate_publication_namespace(
    backend_id: str,
    report: OperationalReport,
    snapshot: dict[str, JsonValue],
) -> None:
    if backend_id == UNAVAILABLE_BACKEND_REPORT_ID:
        _validate_unavailable_execution_report(report, snapshot)
        return
    if backend_id == _LEGACY_BACKEND_ID and not isinstance(report, LegacyScoreReport):
        _validate_unavailable_execution_report(report, snapshot)
        return
    if isinstance(report, LegacyScoreReport):
        declared_backend_id = snapshot["backend_id"]
    else:
        descriptor = snapshot["descriptor"]
        declared_backend_id = descriptor["backend_id"] if isinstance(descriptor, dict) else None
        if declared_backend_id is None and backend_id not in {
            _OAF_BACKEND_ID,
            _HEURISTIC_BACKEND_ID,
        }:
            raise ReportValidationError("report namespace backend_id is unknown")
    if declared_backend_id is not None and backend_id != declared_backend_id:
        raise ReportValidationError("report backend_id does not match publication namespace")


def _validate_unavailable_execution_report(
    report: OperationalReport,
    snapshot: dict[str, JsonValue],
) -> None:
    expected_error: list[JsonValue] = [
        {
            "code": "backend_unavailable",
            "message": "Backend is unavailable.",
        }
    ]
    nullable_identity_fields = (
        "descriptor",
        "descriptor_sha256",
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "parameter_lock_sha256",
        "seal_evidence_sha256",
        "execution_attestation",
        "verification_report",
    )
    invalid = not isinstance(report, ExecutionReport)
    invalid = invalid or snapshot["status"] != "failed"
    invalid = invalid or snapshot["exit_code"] != 2
    invalid = invalid or snapshot["items"] != []
    invalid = invalid or snapshot["errors"] != expected_error
    invalid = invalid or any(snapshot[field] is not None for field in nullable_identity_fields)
    if invalid:
        raise ReportValidationError(_UNAVAILABLE_NAMESPACE_ERROR)


@contextmanager
def _latest_namespace_lock(
    backend_root: Path,
    *,
    anchor: DirectoryAnchor | None,
) -> Iterator[None]:
    with open_lock_file_no_follow(
        backend_root / ".latest-report.lock",
        anchor=anchor,
    ) as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)


def _read_optional_regular_file(
    path: Path,
    *,
    anchor: DirectoryAnchor | None,
) -> bytes | None:
    try:
        return read_regular_file_no_follow(path, anchor=anchor)
    except FileNotFoundError:
        return None


def _restore_latest(
    path: Path,
    previous_state: bytes | object,
    *,
    anchor: DirectoryAnchor | None,
) -> None:
    if isinstance(previous_state, bytes):
        if anchor is None:
            atomic_replace_bytes(path, previous_state)
        else:
            atomic_replace_bytes(path, previous_state, anchor=anchor)
        return
    if previous_state is not _LATEST_ABSENT:
        raise OSError("latest report rollback state is unknown")
    try:
        unlink_regular_file_no_follow(path, anchor=anchor)
    except FileNotFoundError:
        return
