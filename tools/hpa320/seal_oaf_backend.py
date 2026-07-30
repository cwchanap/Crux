#!/usr/bin/env python3
"""Validate and immutably publish the frozen OaF backend seal."""

# The seal gate deliberately keeps every cross-record check visible in one module.
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import os
import platform
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, cast

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_lock import (
    BackendLockError,
    LoadedBackendLock,
    LoadedConversionAudit,
    LoadedRuntimeLock,
    LoadedSealEvidence,
    load_backend_lock,
    load_conversion_audit,
    load_runtime_lock,
    load_seal_evidence,
    validate_oaf_lock_set,
)
from src.benchmark.backend_process import NativeHostEvidence
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    publish_immutable_bytes,
    read_regular_file_no_follow,
)
from src.benchmark.backends import PublishedArtifact
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    load_checkpoint_acquisition_evidence,
    load_checkpoint_acquisition_request,
)
from tools.hpa320.audit_legacy_tf2_conversion import (
    CANDIDATE_MANIFEST_NAME,
    CANDIDATE_MANIFEST_SCHEMA,
)
from tools.hpa320.oaf_system_packages import (
    ProbeResult,
    SystemPackage,
    SystemPackageError,
    inventory_sha256,
    load_base_system_package_evidence,
    load_base_system_package_request,
)

HOST_EVIDENCE_NAME = "native-host-evidence.json"
PROPOSED_BACKEND_LOCK_NAME = "proposed-backend-lock.json"
PROPOSED_RUNTIME_LOCK_NAME = "proposed-runtime-lock.json"
PROPOSED_SEAL_EVIDENCE_NAME = "proposed-seal-evidence.json"
TENSOR_COVERAGE_NAME = "tensor-coverage.json"
SECURITY_SCAN_NAME = "security-scan.json"
ADVISORY_SNAPSHOT_NAME = "advisory-snapshot.json"
OCI_LAYOUT_MANIFEST_NAME = "oci-layout-manifest.json"
SMOKE_ORACLE_NAME = "smoke-oracle.json"
SMOKE_AUDIO_NAME = "canonical.wav"
SMOKE_PREDICTION_NAME = "smoke-prediction.jsonl"
CALIBRATION_CANDIDATE_AUTHORITY_NAME = "calibration-candidate-authority.json"
CONVERSION_AUDIT_NAME = "conversion-audit.json"

HOST_ADAPTER_SCHEMA = "crux.oaf-host-adapter-source-manifest/v1"
SMOKE_ORACLE_SCHEMA = "crux.oaf-smoke-oracle/v1"

HOST_ADAPTER_SOURCE_PATHS = (
    "src/benchmark/backend_attestation.py",
    "src/benchmark/backend_identity.py",
    "src/benchmark/backend_lock.py",
    "src/benchmark/backend_process.py",
    "src/benchmark/backend_publication.py",
    "src/benchmark/backends/base.py",
    "src/benchmark/backends/oaf_tf1.py",
    "src/benchmark/input_view.py",
    "src/benchmark/prediction_artifact.py",
)

_HOST_EVIDENCE_KEYS = frozenset({"kind", "official_execution_allowed", "payload", "sha256"})
_CANDIDATE_MANIFEST_KEYS = frozenset(
    {
        "checkpoint_components",
        "checkpoint_prefix",
        "model_artifact_set_sha256",
        "required_inference_inventory_sha256",
        "schema",
    }
)
_SMOKE_ORACLE_KEYS = frozenset(
    {
        "input_audio_frame_count",
        "input_audio_sha256",
        "input_view_id",
        "native_events",
        "schema",
        "source_audio_id",
        "source_audio_sha256",
    }
)
_OCI_LAYOUT_KEYS = frozenset(
    {
        "archive",
        "config_digest",
        "image_manifest_digest",
        "layer_digests",
        "schema",
    }
)
_ARCHIVE_KEYS = frozenset({"name", "sha256", "size"})
_FINAL_LOCK_HASH_KEYS = frozenset({"backend_lock_sha256", "runtime_lock_sha256"})
CALIBRATION_MEASUREMENT_REQUEST_SCHEMA = "crux.oaf-calibration-measurement-request/v1"
CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA = "crux.oaf-calibration-measurement-evidence/v1"
SEAL_PROFILE_REQUEST_SCHEMA = "crux.oaf-seal-profile-request/v1"
SEAL_CANDIDATE_SCHEMA = "crux.oaf-seal-candidate/v1"
_MEASUREMENT_REQUEST_KEYS = frozenset(
    {
        "backend_id",
        "container_restrictions",
        "fixtures",
        "frame_counts",
        "output_schemas",
        "repetition_count",
        "required_metrics",
        "schema",
    }
)
_MEASUREMENT_EVIDENCE_KEYS = frozenset(
    {
        "base_system_package_evidence_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "image_manifest_digest",
        "measurement_rows",
        "native_host_evidence",
        "request_sha256",
        "schema",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "base_system_package_evidence_sha256",
        "base_system_package_request_sha256",
        "calibration_measurement_evidence_sha256",
        "calibration_measurement_request_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "checkpoint_acquisition_request_sha256",
        "cpu_limit_millis",
        "max_input_audio_frames",
        "memory_limit_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "runtime_gid",
        "runtime_uid",
        "schema",
        "shm_bytes",
        "startup_deadline_seconds",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
        "tmp_bytes",
    }
)
_MEASUREMENT_ROW_KEYS = frozenset(
    {
        "exit_code",
        "input_frame_count",
        "oom_killed",
        "peak_cpu_millis",
        "peak_pid_count",
        "peak_rss_bytes",
        "peak_shm_bytes",
        "peak_tmp_bytes",
        "prediction_sha256",
        "process_instance_id",
        "repetition",
        "request_millis",
        "signal",
        "startup_millis",
        "stderr_max_line_bytes",
        "stdout_max_line_bytes",
    }
)
_CHECKPOINT_REQUEST_PATH = Path(
    "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
)
_BASE_SYSTEM_REQUEST_PATH = Path("runtime/oaf_tf1/base-system-package-request.json")
_HASH_FIELDS = (
    (
        "runtime/oaf_tf1/source-manifest.json",
        "upstream_source_manifest_sha256",
    ),
    (
        "runtime/oaf_tf1/runner-source-manifest.json",
        "runner_source_manifest_sha256",
    ),
    (
        "runtime/oaf_tf1/distribution-build-manifest.json",
        "distribution_build_manifest_sha256",
    ),
    (
        "runtime/oaf_tf1/patches/capture-emitted-frame.patch",
        "instrumentation_patch_sha256",
    ),
)


class SealError(ValueError):
    """The candidate cannot be accepted or atomically published."""


class SealIntegrityError(SealError):
    """Authenticated inputs or immutable publication contradict the requested contract."""


@dataclass(frozen=True)
class PublishedSeal:
    """The immutable artifacts published by one successful seal operation."""

    artifacts: tuple[PublishedArtifact, ...]
    publication_order: tuple[str, ...]


@dataclass(frozen=True)
class MeasurementRow:
    input_frame_count: int
    repetition: int
    process_instance_id: str
    peak_cpu_millis: int
    peak_rss_bytes: int
    peak_tmp_bytes: int
    peak_shm_bytes: int
    peak_pid_count: int
    startup_millis: int
    request_millis: int
    stdout_max_line_bytes: int
    stderr_max_line_bytes: int
    exit_code: int
    signal: int | None
    oom_killed: bool
    prediction_sha256: str | None


@dataclass(frozen=True)
class CalibrationMeasurementRequest:
    frame_counts: tuple[int, ...]
    repetition_count: int
    sha256: str
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class SealProfileRequest:
    max_input_audio_frames: int
    sha256: str
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class PublishedSealCandidate:
    artifact: PublishedArtifact


@dataclass(frozen=True)
class ProducerOutcome:
    status: str
    exit_code: Literal[0, 1, 2]
    report_path: Path | None
    report_sha256: str | None


@dataclass(frozen=True)
class CalibrationProbeResult:
    """One runner result with explicit pre-inference rejection evidence."""

    row: MeasurementRow
    rejected_before_inference: bool


@dataclass(frozen=True)
class _Candidate:
    backend: LoadedBackendLock
    runtime: LoadedRuntimeLock
    seal: LoadedSealEvidence
    audit: LoadedConversionAudit
    host: NativeHostEvidence
    contents: Mapping[str, bytes]
    host_manifest_content: bytes


def load_native_host_evidence(path: Path) -> NativeHostEvidence:
    """Strict-read one existing NativeHostEvidence record on its native worker."""

    value, _ = _read_canonical_object(Path(path), "native host evidence")
    if set(value) != _HOST_EVIDENCE_KEYS:
        raise SealError("native host evidence fields must match the exact existing record")
    kind = value["kind"]
    payload = value["payload"]
    digest = value["sha256"]
    official = value["official_execution_allowed"]
    if (
        not isinstance(kind, str)
        or not isinstance(payload, dict)
        or not isinstance(digest, str)
        or not isinstance(official, bool)
    ):
        raise SealError("native host evidence field types are invalid")
    try:
        evidence = NativeHostEvidence(
            kind=cast(Any, kind),
            payload=payload,
            sha256=digest,
            official_execution_allowed=official,
        )
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SealError(f"native host evidence is invalid: {error}") from None
    _require_current_native_worker(evidence)
    return evidence


def build_host_adapter_source_manifest(repository_root: Path) -> dict[str, JsonValue]:
    """Hash the exact inference-relevant host adapter files selected by the design."""

    repository = _require_directory(Path(repository_root), "repository root")
    rows: list[JsonValue] = []
    for relative in HOST_ADAPTER_SOURCE_PATHS:
        content = _read_regular(repository / relative, f"host adapter source {relative}")
        rows.append({"path": relative, "sha256": sha256_hex(content)})
    return {
        "covered_roots": list(HOST_ADAPTER_SOURCE_PATHS),
        "files": rows,
        "schema": HOST_ADAPTER_SCHEMA,
    }


def materialize_system_packages(
    *,
    host_evidence: Path,
    bundle: Path,
    build_args_output: Path,
) -> None:
    """Enforce the native gate without inventing an unspecified acquisition protocol."""

    load_native_host_evidence(host_evidence)
    _require_absent(Path(bundle), "system package bundle")
    _require_absent(Path(build_args_output), "system package build arguments")
    raise SealError(
        "native system-package acquisition inputs are not specified; no output was written"
    )


def attest_base_system(
    *,
    request_path: Path,
    host_evidence_path: Path,
    image: str,
    output_path: Path,
) -> PublishedArtifact:
    """Publish immutable base-system evidence after exact native probes."""

    request = load_base_system_package_request(Path(request_path))
    host = load_native_host_evidence(Path(host_evidence_path))
    if (
        not isinstance(image, str)
        or not image
        or image.strip().lower() in {"auto", "none", "sentinel", "unlimited", "unset"}
    ):
        raise SealError("base-system image identity is invalid")
    base_image = f"{request.base_image}@{request.base_image_manifest_digest}"
    observed_manifest = (
        _docker_capture(
            ("docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", base_image),
            "pinned base image manifest",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if observed_manifest != base_image:
        raise SealError("pinned base image manifest does not match the request")

    base_inventory_bytes = _docker_run(
        base_image, _DPKG_INVENTORY_COMMAND, "base package inventory"
    )
    runtime_inventory_bytes = _docker_run(
        image, _DPKG_INVENTORY_COMMAND, "runtime package inventory"
    )
    if base_inventory_bytes != runtime_inventory_bytes:
        raise SealError("runtime package inventory differs from the pinned base image")
    inventory = _parse_dpkg_inventory(base_inventory_bytes)

    base_keyring = _parse_keyring_sha256(
        _docker_run(base_image, _KEYRING_SHA256_COMMAND, "base archive keyring")
    )
    runtime_keyring = _parse_keyring_sha256(
        _docker_run(image, _KEYRING_SHA256_COMMAND, "runtime archive keyring")
    )
    if base_keyring != request.base_image_archive_keyring_sha256 or runtime_keyring != base_keyring:
        raise SealError("base-image archive keyring hash does not match the request")

    probes = tuple(
        ProbeResult(name=name, value=_run_base_system_probe(name, base_image, image))
        for name in request.required_probes
    )
    payload: JsonValue = {
        "additional_system_packages": [],
        "base_image_archive_keyring_sha256": request.base_image_archive_keyring_sha256,
        "base_image_manifest_digest": request.base_image_manifest_digest,
        "native_host_evidence": {
            "kind": host.kind,
            "official_execution_allowed": host.official_execution_allowed,
            "payload": _plain_json(host.payload),
            "sha256": host.sha256,
        },
        "package_inventory": [
            {
                "architecture": package.architecture,
                "name": package.name,
                "version": package.version,
            }
            for package in inventory
        ],
        "package_inventory_sha256": inventory_sha256(inventory),
        "probes": [{"name": probe.name, "value": probe.value} for probe in probes],
        "request_sha256": request.sha256,
        "schema": "crux.oaf-base-system-package-evidence/v1",
    }
    content = canonical_json_bytes(payload, trailing_newline=True)
    try:
        published = publish_immutable_bytes(
            Path(output_path),
            content,
            sha256_hex(content),
            role="base_system_package_evidence",
        )
    except ArtifactPublicationError:
        raise SealError("base-system evidence publication failed") from None
    try:
        load_base_system_package_evidence(Path(output_path), request=request)
    except (OSError, ValueError) as error:
        raise SealError(f"published base-system evidence is invalid: {error}") from None
    return published


_DPKG_INVENTORY_COMMAND = "dpkg-query -W -f='${Package}\\t${Version}\\t${Architecture}\\n'"
_KEYRING_SHA256_COMMAND = "sha256sum /usr/share/keyrings/debian-archive-keyring.gpg"
_BASE_SYSTEM_PROBE_COMMANDS = {
    "base_python_version": ("base", "python --version"),
    "runtime_python_version": ("runtime", "/opt/crux/venv/bin/python --version"),
    "runtime_tensorflow_version": (
        "runtime",
        "/opt/crux/venv/bin/python -c 'import tensorflow as tf; print(tf.__version__)'",
    ),
    "runtime_smoke": (
        "runtime",
        "/opt/crux/venv/bin/python -s /opt/crux/runtime/entrypoint.py --help",
    ),
}


def _docker_capture(command: tuple[str, ...], label: str) -> bytes:
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise SealError(f"{label} command failed")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise SealError(f"{label} command produced no output")
    return result.stdout


def _docker_run(image: str, command: str, label: str) -> bytes:
    return _docker_capture(
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--platform",
            "linux/amd64",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            command,
        ),
        label,
    )


def _parse_dpkg_inventory(content: bytes) -> tuple[SystemPackage, ...]:
    try:
        lines = content.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        raise SealError("base package inventory is not UTF-8") from None
    if not lines or content != ("\n".join(lines) + "\n").encode("utf-8"):
        raise SealError("base package inventory is not canonical")
    inventory: list[SystemPackage] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 3 or not all(fields):
            raise SealError("base package inventory row is invalid")
        inventory.append(SystemPackage(name=fields[0], version=fields[1], architecture=fields[2]))
    try:
        inventory_sha256(tuple(inventory))
    except ValueError as error:
        raise SealError(str(error)) from None
    return tuple(inventory)


def _parse_keyring_sha256(content: bytes) -> str:
    try:
        line = content.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        raise SealError("base-image archive keyring hash is not ASCII") from None
    if not line.endswith("\n") or line.endswith("\n\n"):
        raise SealError("base-image archive keyring hash is not canonical")
    digest, separator, _ = line[:-1].partition("  ")
    if not separator or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SealError("base-image archive keyring hash is invalid")
    return digest


def _run_base_system_probe(name: str, base_image: str, runtime_image: str) -> str:
    try:
        target, command = _BASE_SYSTEM_PROBE_COMMANDS[name]
    except KeyError:
        raise SealError("base-system request contains an unapproved probe") from None
    output = _docker_run(base_image if target == "base" else runtime_image, command, name)
    try:
        value = output.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        raise SealError(f"{name} probe output is not UTF-8") from None
    if not value:
        raise SealError(f"{name} probe produced no result")
    return value


def load_calibration_measurement_request(path: Path) -> CalibrationMeasurementRequest:
    """Strict-load the reviewed diagnostic measurement authority."""

    payload, content = _read_canonical_object(path, "calibration measurement request")
    if (
        set(payload) != _MEASUREMENT_REQUEST_KEYS
        or payload["schema"] != CALIBRATION_MEASUREMENT_REQUEST_SCHEMA
    ):
        raise SealError("calibration measurement request fields are invalid")
    if not isinstance(payload["backend_id"], str) or not payload["backend_id"]:
        raise SealError("calibration measurement request backend is invalid")
    counts = payload["frame_counts"]
    repetitions = payload["repetition_count"]
    if (
        not isinstance(counts, list)
        or not counts
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in counts
        )
        or counts != sorted(counts)
        or len(set(counts)) != len(counts)
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
    ):
        raise SealError("calibration measurement request bounds are invalid")
    restrictions = payload["container_restrictions"]
    if (
        not isinstance(restrictions, dict)
        or set(restrictions)
        != {"drop_capabilities", "network", "no_new_privileges", "platform", "read_only_root"}
        or restrictions
        != {
            "drop_capabilities": ["ALL"],
            "network": "none",
            "no_new_privileges": True,
            "platform": "linux/amd64",
            "read_only_root": True,
        }
    ):
        raise SealError("calibration measurement restrictions are invalid")
    fixtures = payload["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise SealError("calibration measurement fixtures are invalid")
    for fixture in fixtures:
        if not isinstance(fixture, dict) or set(fixture) != {
            "input_audio_sha256",
            "input_view_id",
            "source_audio_id",
            "source_audio_sha256",
        }:
            raise SealError("calibration measurement fixture fields are invalid")
        for field in ("input_audio_sha256", "source_audio_sha256"):
            _require_sha256_value(fixture[field], "calibration measurement fixture hash")
        if not all(
            isinstance(fixture[field], str) and fixture[field]
            for field in ("input_view_id", "source_audio_id")
        ):
            raise SealError("calibration measurement fixture identities are invalid")
    if not _string_list(payload["required_metrics"]) or not _string_list(payload["output_schemas"]):
        raise SealError("calibration measurement output contract is invalid")
    return CalibrationMeasurementRequest(tuple(counts), repetitions, sha256_hex(content), payload)


def load_seal_profile_request(path: Path) -> SealProfileRequest:
    """Strict-load only a fully reviewed, non-sentinel profile authority."""

    payload, content = _read_canonical_object(path, "seal profile request")
    if set(payload) != _PROFILE_KEYS or payload["schema"] != SEAL_PROFILE_REQUEST_SCHEMA:
        raise SealError("seal profile request fields are invalid")
    for field in (
        "base_system_package_evidence_sha256",
        "base_system_package_request_sha256",
        "calibration_measurement_evidence_sha256",
        "calibration_measurement_request_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "checkpoint_acquisition_request_sha256",
    ):
        _require_sha256_value(payload[field], f"seal profile request {field}")
    for field in (
        "cpu_limit_millis",
        "max_input_audio_frames",
        "memory_limit_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "runtime_gid",
        "runtime_uid",
        "shm_bytes",
        "startup_deadline_seconds",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
        "tmp_bytes",
    ):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SealError(f"seal profile request {field} must be an explicit positive integer")
    return SealProfileRequest(
        max_input_audio_frames=cast(int, payload["max_input_audio_frames"]),
        sha256=sha256_hex(content),
        payload=payload,
    )


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Run isolated reviewed-authority fixtures through their production loaders."""
    loaders = {
        CALIBRATION_MEASUREMENT_REQUEST_SCHEMA: load_calibration_measurement_request,
        SEAL_PROFILE_REQUEST_SCHEMA: load_seal_profile_request,
    }
    try:
        if schema not in loaders:
            expected = {
                CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA: _MEASUREMENT_EVIDENCE_KEYS,
                SEAL_CANDIDATE_SCHEMA: {
                    "calibration_measurement_evidence_sha256",
                    "checkpoint_components",
                    "checkpoint_prefix",
                    "model_artifact_set_sha256",
                    "required_inference_inventory_sha256",
                    "schema",
                    "seal_profile_request_sha256",
                },
                "crux.oaf-oci-layout-manifest/v1": _OCI_LAYOUT_KEYS,
                HOST_ADAPTER_SCHEMA: {"covered_roots", "files", "schema"},
            }[schema]
            payload, _ = _read_canonical_object_from_content(content, "schema golden")
            if set(payload) != expected or payload["schema"] != schema:
                raise SealError("schema golden fields are invalid")
            first = min(expected)
            if first == "archive" and not isinstance(payload[first], dict):
                raise SealError("schema golden archive is invalid")
            if first == "calibration_measurement_evidence_sha256" and not _is_sha256(
                payload[first]
            ):
                raise SealError("schema golden hash is invalid")
            if first == "covered_roots" and not isinstance(payload[first], list):
                raise SealError("schema golden roots are invalid")
            if first == "base_system_package_evidence_sha256" and not _is_sha256(payload[first]):
                raise SealError("schema golden measurement evidence hash is invalid")
            return
        loader = loaders[schema]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "golden.json"
            path.write_bytes(content)
            loader(path)
    except (KeyError, OSError, SealError) as error:
        raise ValueError(str(error)) from None


def _read_canonical_object_from_content(
    content: bytes, label: str
) -> tuple[dict[str, JsonValue], bytes]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SealError(f"{label} must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, dict):
        raise SealError(f"{label} must be an object")
    return value, content


def _measurement_row_from_value(value: MeasurementRow | Mapping[str, object]) -> MeasurementRow:
    if isinstance(value, MeasurementRow):
        return value
    if not isinstance(value, Mapping) or set(value) != _MEASUREMENT_ROW_KEYS:
        raise SealError("measurement runner returned an invalid row")
    try:
        row = MeasurementRow(**cast(dict[str, Any], value))
    except TypeError:
        raise SealError("measurement runner returned an invalid row") from None
    _validate_measurement_row(row)
    return row


def _validate_measurement_row(row: MeasurementRow) -> None:
    positive = (
        row.input_frame_count,
        row.repetition,
        row.peak_cpu_millis,
        row.peak_rss_bytes,
        row.peak_tmp_bytes,
        row.peak_shm_bytes,
        row.peak_pid_count,
        row.startup_millis,
        row.request_millis,
        row.stdout_max_line_bytes,
        row.stderr_max_line_bytes,
    )
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in positive
        )
        or row.input_frame_count <= 0
        or row.repetition <= 0
        or not isinstance(row.process_instance_id, str)
        or not row.process_instance_id
        or not isinstance(row.exit_code, int)
        or isinstance(row.exit_code, bool)
        or (row.signal is not None and (not isinstance(row.signal, int) or row.signal <= 0))
        or not isinstance(row.oom_killed, bool)
        or (row.prediction_sha256 is not None and not _is_sha256(row.prediction_sha256))
    ):
        raise SealError("measurement row is invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256_value(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise SealError(f"{label} is invalid")
    return cast(str, value)


def _string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
        and len(set(cast(list[str], value))) == len(value)
    )


def _require_image(image: str, label: str) -> None:
    if (
        not isinstance(image, str)
        or not image
        or image.strip().lower() in {"auto", "none", "sentinel", "unlimited", "unset"}
    ):
        raise SealError(f"{label} image identity is invalid")


def _row_payload(row: MeasurementRow) -> dict[str, JsonValue]:
    return {
        "exit_code": row.exit_code,
        "input_frame_count": row.input_frame_count,
        "oom_killed": row.oom_killed,
        "peak_cpu_millis": row.peak_cpu_millis,
        "peak_pid_count": row.peak_pid_count,
        "peak_rss_bytes": row.peak_rss_bytes,
        "peak_shm_bytes": row.peak_shm_bytes,
        "peak_tmp_bytes": row.peak_tmp_bytes,
        "prediction_sha256": row.prediction_sha256,
        "process_instance_id": row.process_instance_id,
        "repetition": row.repetition,
        "request_millis": row.request_millis,
        "signal": row.signal,
        "startup_millis": row.startup_millis,
        "stderr_max_line_bytes": row.stderr_max_line_bytes,
        "stdout_max_line_bytes": row.stdout_max_line_bytes,
    }


def _require_headroom(profile: SealProfileRequest, rows: Sequence[MeasurementRow]) -> None:
    """Reject equality as well as underprovisioning after the profile's units convert."""

    observed = {
        "cpu_limit_millis": max(row.peak_cpu_millis for row in rows),
        "memory_limit_bytes": max(row.peak_rss_bytes for row in rows),
        "pid_limit": max(row.peak_pid_count for row in rows),
        "tmp_bytes": max(row.peak_tmp_bytes for row in rows),
        "shm_bytes": max(row.peak_shm_bytes for row in rows),
        "startup_deadline_seconds": max(row.startup_millis for row in rows),
        "request_deadline_seconds": max(row.request_millis for row in rows),
        "stdout_max_line_bytes": max(row.stdout_max_line_bytes for row in rows),
        "stderr_max_line_bytes": max(row.stderr_max_line_bytes for row in rows),
        "stderr_read_chunk_bytes": max(row.stderr_max_line_bytes for row in rows),
        "stderr_ring_buffer_bytes": max(row.stderr_max_line_bytes for row in rows),
    }
    for field, peak in observed.items():
        value = cast(int, profile.payload[field])
        comparable = value * 1000 if field.endswith("_deadline_seconds") else value
        if comparable <= peak:
            raise SealError(f"seal profile request {field} is not strictly above measured peak")


def measure(
    *,
    request_path: Path,
    host_evidence_path: Path,
    image: str,
    model_cache: Path,
    checkpoint_evidence_path: Path,
    base_system_evidence_path: Path,
    output_path: Path,
    runner: (
        Callable[[CalibrationMeasurementRequest, int, int], MeasurementRow | Mapping[str, object]]
        | None
    ) = None,
) -> PublishedArtifact:
    """Publish diagnostic measurements only; resource choices remain outside this command."""

    request = load_calibration_measurement_request(Path(request_path))
    load_native_host_evidence(Path(host_evidence_path))
    _require_image(image, "measurement")
    _require_directory(Path(model_cache), "model cache")
    try:
        checkpoint_request = load_checkpoint_acquisition_request(_CHECKPOINT_REQUEST_PATH)
        checkpoint = load_checkpoint_acquisition_evidence(
            Path(checkpoint_evidence_path), request=checkpoint_request
        )
        base_request = load_base_system_package_request(_BASE_SYSTEM_REQUEST_PATH)
        base = load_base_system_package_evidence(
            Path(base_system_evidence_path), request=base_request
        )
    except (CheckpointAcquisitionError, SystemPackageError) as error:
        raise SealIntegrityError(f"measurement authority is invalid: {error}") from None
    if runner is None:
        raise SealError("measurement runner is unavailable; no output was written")
    rows: list[MeasurementRow] = []
    for frame_count in request.frame_counts:
        for repetition in range(1, request.repetition_count + 1):
            row = _measurement_row_from_value(runner(request, frame_count, repetition))
            if row.input_frame_count != frame_count or row.repetition != repetition:
                raise SealError("measurement runner row does not match the requested probe")
            rows.append(row)
    rows.sort(key=lambda row: (row.input_frame_count, row.process_instance_id, row.repetition))
    host_payload, _ = _read_canonical_object(Path(host_evidence_path), "native host evidence")
    payload: JsonValue = {
        "base_system_package_evidence_sha256": base.sha256,
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "image_manifest_digest": image,
        "measurement_rows": [_row_payload(row) for row in rows],
        "native_host_evidence": host_payload,
        "request_sha256": request.sha256,
        "schema": CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA,
    }
    content = canonical_json_bytes(payload, trailing_newline=True)
    try:
        return publish_immutable_bytes(
            Path(output_path), content, sha256_hex(content), role="calibration_measurement_evidence"
        )
    except ArtifactPublicationError:
        raise SealError("measurement evidence publication failed") from None


def _load_measurement_evidence(
    path: Path, request: CalibrationMeasurementRequest
) -> tuple[dict[str, JsonValue], tuple[MeasurementRow, ...], str]:
    payload, content = _read_canonical_object(path, "calibration measurement evidence")
    if (
        set(payload) != _MEASUREMENT_EVIDENCE_KEYS
        or payload["schema"] != CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA
    ):
        raise SealError("calibration measurement evidence fields are invalid")
    if payload["request_sha256"] != request.sha256:
        raise SealError("calibration measurement evidence request hash does not match")
    rows_value = payload["measurement_rows"]
    if not isinstance(rows_value, list) or not rows_value:
        raise SealError("calibration measurement evidence rows are invalid")
    rows = tuple(_measurement_row_from_value(cast(Mapping[str, object], row)) for row in rows_value)
    if (
        tuple(
            sorted(
                rows,
                key=lambda row: (row.input_frame_count, row.process_instance_id, row.repetition),
            )
        )
        != rows
    ):
        raise SealError("calibration measurement evidence rows are not sorted")
    expected_pairs = {
        (frame_count, repetition)
        for frame_count in request.frame_counts
        for repetition in range(1, request.repetition_count + 1)
    }
    actual_pairs = {(row.input_frame_count, row.repetition) for row in rows}
    if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise SealIntegrityError(
            "calibration measurement evidence matrix is incomplete or duplicated"
        )
    return payload, rows, sha256_hex(content)


def _probe_result_from_value(
    value: CalibrationProbeResult | Mapping[str, object],
) -> CalibrationProbeResult:
    if isinstance(value, CalibrationProbeResult):
        _validate_measurement_row(value.row)
        return value
    if not isinstance(value, Mapping) or set(value) != {"row", "rejected_before_inference"}:
        raise SealError("calibration runner returned an invalid probe result")
    rejected = value["rejected_before_inference"]
    if not isinstance(rejected, bool):
        raise SealError("calibration runner rejection evidence is invalid")
    return CalibrationProbeResult(
        row=_measurement_row_from_value(cast(Mapping[str, object], value["row"])),
        rejected_before_inference=rejected,
    )


def _load_candidate_authority(
    directory: Path,
    *,
    profile: SealProfileRequest,
    measurement_sha256: str,
    checkpoint_artifact_set_sha256: str,
    model_cache: Path,
    repository_root: Path,
) -> _Candidate:
    """Authenticate a complete directory that can be handed directly to ``seal``."""

    authority, _ = _read_canonical_object(
        directory / CALIBRATION_CANDIDATE_AUTHORITY_NAME, "calibration candidate authority"
    )
    expected_keys = {
        "candidate_manifest_sha256",
        "calibration_measurement_evidence_sha256",
        "schema",
        "seal_profile_request_sha256",
    }
    if set(authority) != expected_keys or authority["schema"] != (
        "crux.oaf-calibration-candidate-authority/v1"
    ):
        raise SealIntegrityError("calibration candidate authority schema is invalid")
    if (
        authority["calibration_measurement_evidence_sha256"] != measurement_sha256
        or authority["seal_profile_request_sha256"] != profile.sha256
    ):
        raise SealIntegrityError("calibration candidate authority links are unrelated")
    for field in (
        "candidate_manifest_sha256",
        "calibration_measurement_evidence_sha256",
        "seal_profile_request_sha256",
    ):
        _require_sha256_value(authority[field], f"calibration candidate authority {field}")
    try:
        loaded = _load_candidate(
            candidate=directory,
            conversion_audit=directory / CONVERSION_AUDIT_NAME,
            repository_root=repository_root,
        )
    except SealError as error:
        raise SealIntegrityError(f"calibration candidate authority is invalid: {error}") from None
    manifest, manifest_content = _read_canonical_object(
        directory / CANDIDATE_MANIFEST_NAME, "candidate manifest"
    )
    _validate_candidate_manifest(directory / CANDIDATE_MANIFEST_NAME, loaded.backend)
    if (
        sha256_hex(manifest_content) != authority["candidate_manifest_sha256"]
        or manifest["model_artifact_set_sha256"] != checkpoint_artifact_set_sha256
    ):
        raise SealIntegrityError("calibration candidate authority cache identity is unrelated")
    cache = _require_directory(
        model_cache / "sha256" / checkpoint_artifact_set_sha256,
        "calibration candidate model cache",
    )
    components = loaded.backend.payload["checkpoint_components"]
    if not isinstance(components, list):
        raise SealIntegrityError("calibration candidate components are invalid")
    for component in components:
        if not isinstance(component, dict) or set(component) != {"name", "sha256", "size"}:
            raise SealIntegrityError("calibration candidate component is invalid")
        name, digest, size = component["name"], component["sha256"], component["size"]
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not _is_sha256(digest)
        ):
            raise SealIntegrityError("calibration candidate component is invalid")
        content = read_regular_file_no_follow(cache / name)
        if len(content) != size or sha256_hex(content) != digest:
            raise SealIntegrityError("calibration candidate cache bytes do not match the authority")
    return loaded


def _publish_candidate_directory(source: Path, output: Path) -> PublishedArtifact:
    """No-replace publish all validated candidate bytes as one directory identity."""

    _require_absent(output, "calibration candidate")
    parent = _require_directory(output.parent, "calibration candidate parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=parent))
    try:
        entries = sorted(source.iterdir(), key=lambda item: item.name)
        if not entries:
            raise SealIntegrityError("calibration candidate authority is empty")
        for entry in entries:
            if not entry.is_file() or entry.is_symlink():
                raise SealIntegrityError(
                    "calibration candidate authority contains a non-regular entry"
                )
            content = read_regular_file_no_follow(entry)
            target = staging / entry.name
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
        staging_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        os.rename(staging, output)
    except (OSError, SealError):
        for entry in staging.iterdir() if staging.exists() else ():
            entry.unlink()
        if staging.exists():
            staging.rmdir()
        raise SealIntegrityError("calibration candidate publication failed") from None
    manifest = read_regular_file_no_follow(output / CANDIDATE_MANIFEST_NAME)
    return PublishedArtifact(role="seal_candidate", path=output, sha256=sha256_hex(manifest))


def calibrate(
    *,
    request_path: Path | None = None,
    measurement_evidence_path: Path | None = None,
    checkpoint_evidence_path: Path | None = None,
    base_system_evidence_path: Path | None = None,
    image: str,
    host_evidence: Path,
    model_cache: Path,
    output: Path,
    runner: Callable[[int, bool, int], CalibrationProbeResult | Mapping[str, object]] | None = None,
    candidate_authority_path: Path | None = None,
    repository_root: Path | None = None,
) -> PublishedSealCandidate:
    """Rerun a reviewed profile with no hidden headroom or process assumptions."""

    load_native_host_evidence(host_evidence)
    _require_image(image, "calibration")
    _require_directory(Path(model_cache), "model cache")
    if None in (
        request_path,
        measurement_evidence_path,
        checkpoint_evidence_path,
        base_system_evidence_path,
    ):
        raise SealError("calibration requires reviewed request and evidence inputs")
    try:
        profile = load_seal_profile_request(cast(Path, request_path))
        measurement_request_path = cast(Path, request_path).with_name(
            cast(Path, request_path).name.replace(
                "seal-profile-request", "calibration-measurement-request"
            )
        )
        measurement_request = load_calibration_measurement_request(measurement_request_path)
        evidence, rows, measurement_sha = _load_measurement_evidence(
            cast(Path, measurement_evidence_path), measurement_request
        )
        checkpoint_request = load_checkpoint_acquisition_request(_CHECKPOINT_REQUEST_PATH)
        checkpoint = load_checkpoint_acquisition_evidence(
            cast(Path, checkpoint_evidence_path), request=checkpoint_request
        )
        base_request = load_base_system_package_request(_BASE_SYSTEM_REQUEST_PATH)
        base = load_base_system_package_evidence(
            cast(Path, base_system_evidence_path), request=base_request
        )
    except (CheckpointAcquisitionError, SystemPackageError, SealError) as error:
        raise SealIntegrityError(f"calibration authority is invalid: {error}") from None
    if (
        evidence["checkpoint_acquisition_evidence_sha256"] != checkpoint.sha256
        or evidence["base_system_package_evidence_sha256"] != base.sha256
    ):
        raise SealIntegrityError("measurement evidence does not match its authenticated inputs")
    expected = {
        "calibration_measurement_request_sha256": measurement_request.sha256,
        "calibration_measurement_evidence_sha256": measurement_sha,
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": checkpoint.request_sha256,
        "base_system_package_evidence_sha256": base.sha256,
        "base_system_package_request_sha256": base.request_sha256,
    }
    if any(profile.payload[key] != value for key, value in expected.items()):
        raise SealIntegrityError("seal profile request is unrelated to the reviewed evidence")
    if profile.max_input_audio_frames not in measurement_request.frame_counts:
        raise SealIntegrityError("seal profile input bound was not measured")
    host_payload, _ = _read_canonical_object(host_evidence, "native host evidence")
    if (
        evidence["native_host_evidence"] != host_payload
        or evidence["image_manifest_digest"] != image
    ):
        raise SealIntegrityError("calibration evidence authority does not match this host or image")
    in_bound = tuple(row for row in rows if row.input_frame_count <= profile.max_input_audio_frames)
    if not in_bound or any(
        row.exit_code != 0 or row.signal is not None or row.oom_killed for row in in_bound
    ):
        raise SealIntegrityError("in-bound measurement evidence is not healthy")
    _require_headroom(profile, in_bound)
    source: Path | None = None
    if candidate_authority_path is not None:
        try:
            source = _require_directory(
                Path(candidate_authority_path), "calibration candidate authority"
            )
            _load_candidate_authority(
                source,
                profile=profile,
                measurement_sha256=measurement_sha,
                checkpoint_artifact_set_sha256=checkpoint.model_artifact_set_sha256,
                model_cache=Path(model_cache),
                repository_root=Path.cwd() if repository_root is None else repository_root,
            )
        except SealError as error:
            if isinstance(error, SealIntegrityError):
                raise
            raise SealIntegrityError(
                f"calibration candidate authority is invalid: {error}"
            ) from None
    if runner is None:
        raise SealError("calibration runner is unavailable; no output was written")
    probes = (
        (profile.max_input_audio_frames - 1, True, 1),
        (profile.max_input_audio_frames, True, 2),
        (profile.max_input_audio_frames + 1, True, 3),
        (profile.max_input_audio_frames, False, 1),
    )
    persistent_process_ids: list[str] = []
    fresh_process_id: str | None = None
    for frame_count, persistent, ordinal in probes:
        result = _probe_result_from_value(runner(frame_count, persistent, ordinal))
        row = result.row
        if row.input_frame_count != frame_count:
            raise SealIntegrityError(
                "calibration probe row does not match the requested frame count"
            )
        if persistent and frame_count <= profile.max_input_audio_frames:
            persistent_process_ids.append(row.process_instance_id)
        else:
            fresh_process_id = row.process_instance_id
        if frame_count > profile.max_input_audio_frames:
            if (
                not result.rejected_before_inference
                or row.exit_code == 0
                or row.signal is not None
                or row.oom_killed
            ):
                raise SealError("over-bound input was not rejected before inference")
        elif (
            result.rejected_before_inference
            or row.exit_code != 0
            or row.signal is not None
            or row.oom_killed
        ):
            raise SealError("in-bound calibration probe failed")
    if (
        len(persistent_process_ids) != 2
        or not persistent_process_ids[0]
        or len(set(persistent_process_ids)) != 1
        or not fresh_process_id
        or fresh_process_id == persistent_process_ids[0]
    ):
        raise SealIntegrityError(
            "calibration probes do not prove persistent and fresh process identity"
        )
    if source is None:
        raise SealError("calibration candidate authority is unavailable; no output was written")
    artifact = _publish_candidate_directory(source, Path(output))
    return PublishedSealCandidate(artifact)


# All candidate records are validated before the first publication attempt.
# pylint: disable-next=too-many-arguments,too-many-locals
def seal_candidate(
    *,
    candidate: Path,
    backend_lock: Path,
    runtime_lock: Path,
    seal_evidence: Path,
    conversion_audit: Path,
    host_adapter_source_manifest: Path,
    tensor_coverage: Path,
    security_scan: Path,
    oci_layout_manifest: Path,
    smoke_oracle: Path,
    repository_root: Path,
) -> PublishedSeal:
    """Strict-validate a complete candidate and publish dependencies before locks."""

    output_paths = {
        "backend_lock": Path(backend_lock),
        "host_adapter_source_manifest": Path(host_adapter_source_manifest),
        "oci_layout_manifest": Path(oci_layout_manifest),
        "runtime_lock": Path(runtime_lock),
        "seal_evidence": Path(seal_evidence),
        "security_scan": Path(security_scan),
        "smoke_oracle": Path(smoke_oracle),
        "tensor_coverage": Path(tensor_coverage),
    }
    _validate_output_paths(output_paths)
    loaded = _load_candidate(
        candidate=Path(candidate),
        conversion_audit=Path(conversion_audit),
        repository_root=Path(repository_root),
    )
    content_by_role = {
        "backend_lock": canonical_json_bytes(
            _plain_json(loaded.backend.payload),
            trailing_newline=True,
        ),
        "host_adapter_source_manifest": loaded.host_manifest_content,
        "oci_layout_manifest": loaded.contents[OCI_LAYOUT_MANIFEST_NAME],
        "runtime_lock": canonical_json_bytes(
            _plain_json(loaded.runtime.payload),
            trailing_newline=True,
        ),
        "seal_evidence": canonical_json_bytes(
            _plain_json(loaded.seal.payload),
            trailing_newline=True,
        ),
        "security_scan": loaded.contents[SECURITY_SCAN_NAME],
        "smoke_oracle": loaded.contents[SMOKE_ORACLE_NAME],
        "tensor_coverage": loaded.contents[TENSOR_COVERAGE_NAME],
    }
    order = (
        "host_adapter_source_manifest",
        "tensor_coverage",
        "security_scan",
        "oci_layout_manifest",
        "smoke_oracle",
        "seal_evidence",
        "runtime_lock",
        "backend_lock",
    )
    published: list[PublishedArtifact] = []
    try:
        for role in order:
            content = content_by_role[role]
            published.append(
                publish_immutable_bytes(
                    output_paths[role],
                    content,
                    sha256_hex(content),
                    role=role,
                )
            )
    except ArtifactPublicationError:
        raise SealError("seal publication failed") from None
    return PublishedSeal(artifacts=tuple(published), publication_order=order)


def _load_candidate(
    *,
    candidate: Path,
    conversion_audit: Path,
    repository_root: Path,
) -> _Candidate:
    directory = _require_directory(candidate, "candidate directory")
    repository = _require_directory(repository_root, "repository root")
    host = load_native_host_evidence(directory / HOST_EVIDENCE_NAME)
    backend = _load_backend(directory / PROPOSED_BACKEND_LOCK_NAME)
    runtime = _load_runtime(directory / PROPOSED_RUNTIME_LOCK_NAME)
    seal = _load_seal(directory / PROPOSED_SEAL_EVIDENCE_NAME)
    audit = _load_audit(conversion_audit)
    try:
        validate_oaf_lock_set(backend, runtime, seal, audit)
    except BackendLockError as error:
        raise SealError(f"candidate lock set is invalid: {error}") from None

    _validate_candidate_manifest(directory / CANDIDATE_MANIFEST_NAME, backend)
    _validate_host_binding(host, seal)
    host_manifest = build_host_adapter_source_manifest(repository)
    host_manifest_content = canonical_json_bytes(host_manifest, trailing_newline=True)
    expected_host_hash = cast(str, seal.payload["host_adapter_source_manifest_sha256"])
    if sha256_hex(host_manifest_content) != expected_host_hash:
        raise SealError("host adapter source manifest hash does not match the candidate")

    _validate_repository_inputs(repository, seal, runtime)
    contents = _validate_candidate_artifacts(directory, seal)
    return _Candidate(
        backend=backend,
        runtime=runtime,
        seal=seal,
        audit=audit,
        host=host,
        contents=contents,
        host_manifest_content=host_manifest_content,
    )


def _load_backend(path: Path) -> LoadedBackendLock:
    try:
        return load_backend_lock(path)
    except (BackendLockError, OSError) as error:
        raise SealError(f"backend lock candidate is invalid: {error}") from None


def _load_runtime(path: Path) -> LoadedRuntimeLock:
    try:
        return load_runtime_lock(path)
    except (BackendLockError, OSError) as error:
        raise SealError(f"runtime lock candidate is invalid: {error}") from None


def _load_seal(path: Path) -> LoadedSealEvidence:
    try:
        return load_seal_evidence(path)
    except (BackendLockError, OSError) as error:
        raise SealError(f"seal evidence candidate is invalid: {error}") from None


def _load_audit(path: Path) -> LoadedConversionAudit:
    try:
        return load_conversion_audit(path)
    except (BackendLockError, OSError) as error:
        raise SealError(f"conversion audit is invalid: {error}") from None


def _validate_candidate_manifest(path: Path, backend: LoadedBackendLock) -> None:
    payload, _ = _read_canonical_object(path, "candidate manifest")
    if set(payload) != _CANDIDATE_MANIFEST_KEYS or payload["schema"] != CANDIDATE_MANIFEST_SCHEMA:
        raise SealError("candidate manifest fields must match the existing exact schema")
    components = _plain_json(backend.payload["checkpoint_components"])
    required = _plain_json(backend.payload["required_inference_inventory"])
    model_sha256 = sha256_hex(canonical_json_bytes(components))
    required_sha256 = sha256_hex(canonical_json_bytes(required))
    expected_prefix = PurePosixPath("sha256", model_sha256, "model.ckpt-569400").as_posix()
    if (
        payload["checkpoint_components"] != components
        or payload["model_artifact_set_sha256"] != model_sha256
        or payload["required_inference_inventory_sha256"] != required_sha256
        or payload["checkpoint_prefix"] != expected_prefix
    ):
        raise SealError("candidate manifest identity does not match the proposed backend lock")


def _validate_host_binding(
    host: NativeHostEvidence,
    seal: LoadedSealEvidence,
) -> None:
    expected_form = {
        "approved_local": "native_seal_host",
        "github_hosted": "github_hosted_linux_x64",
        "orchestrator_signed": "orchestrator_signed",
    }[host.kind]
    record = seal.payload["native_host_evidence"]
    if (
        not isinstance(record, Mapping)
        or set(record) != {"form", "sha256"}
        or record["form"] != expected_form
        or record["sha256"] != host.sha256
        or host.official_execution_allowed is not True
    ):
        raise SealError("native host evidence does not match the proposed seal")


def _validate_repository_inputs(
    repository: Path,
    seal: LoadedSealEvidence,
    runtime: LoadedRuntimeLock,
) -> None:
    for relative, field in _HASH_FIELDS:
        content = _read_regular(repository / relative, relative)
        expected = seal.payload[field]
        if sha256_hex(content) != expected:
            raise SealError(f"{relative} hash does not match the candidate")
        if field != "instrumentation_patch_sha256" and _contains_final_lock_hash(
            _parse_canonical_json_content(content, relative)
        ):
            raise SealError(f"{relative} contains a final lock hash cycle")
    for field in (
        "upstream_source_manifest_sha256",
        "runner_source_manifest_sha256",
        "distribution_build_manifest_sha256",
    ):
        if runtime.payload.get(field) != seal.payload.get(field):
            raise SealError(f"{field} does not match across runtime evidence")


def _validate_candidate_artifacts(
    directory: Path,
    seal: LoadedSealEvidence,
) -> dict[str, bytes]:
    tensor, tensor_content = _read_canonical_object(
        directory / TENSOR_COVERAGE_NAME,
        "tensor coverage",
    )
    _validate_tensor_coverage(tensor, seal)
    security, security_content = _read_canonical_object(
        directory / SECURITY_SCAN_NAME,
        "security scan",
    )
    advisory_content = _read_regular(
        directory / ADVISORY_SNAPSHOT_NAME,
        "advisory snapshot",
    )
    _validate_security_scan(security, advisory_content, seal)
    oci, oci_content = _read_canonical_object(
        directory / OCI_LAYOUT_MANIFEST_NAME,
        "OCI layout manifest",
    )
    _validate_oci_layout(oci, directory, seal)
    oracle, oracle_content = _read_canonical_object(
        directory / SMOKE_ORACLE_NAME,
        "smoke oracle",
    )
    _validate_smoke(oracle, directory, seal)

    expected_hashes = {
        TENSOR_COVERAGE_NAME: seal.payload["tensor_coverage_sha256"],
        SECURITY_SCAN_NAME: seal.payload["security_scan_sha256"],
        OCI_LAYOUT_MANIFEST_NAME: seal.payload["oci_layout_manifest_sha256"],
        SMOKE_ORACLE_NAME: seal.payload["smoke_oracle_sha256"],
    }
    contents = {
        TENSOR_COVERAGE_NAME: tensor_content,
        SECURITY_SCAN_NAME: security_content,
        OCI_LAYOUT_MANIFEST_NAME: oci_content,
        SMOKE_ORACLE_NAME: oracle_content,
    }
    for name, content in contents.items():
        if sha256_hex(content) != expected_hashes[name]:
            raise SealError(f"{name} hash does not match the proposed seal")
    return contents


def _validate_tensor_coverage(
    payload: dict[str, JsonValue],
    seal: LoadedSealEvidence,
) -> None:
    required_fields = {
        "active_predict_dropout",
        "checkpoint_inventory",
        "non_inference_inventory",
        "note_sequence_byte_parity",
        "required_inference_inventory",
        "schema",
        "uninitialized_required",
    }
    if not required_fields.issubset(payload):
        raise SealError("tensor coverage must enumerate the exact tensor inventories")
    if not isinstance(payload["schema"], str) or not payload["schema"]:
        raise SealError("tensor coverage schema is invalid")
    for field in (
        "checkpoint_inventory",
        "required_inference_inventory",
        "non_inference_inventory",
    ):
        if not isinstance(payload[field], list) or payload[field] != _plain_json(
            seal.payload[field]
        ):
            raise SealError("tensor coverage must enumerate the exact tensor inventories")
    if payload["uninitialized_required"] != []:
        raise SealError("tensor coverage has uninitialized required variables")
    if payload["active_predict_dropout"] is not False:
        raise SealError("tensor coverage has active PREDICT dropout")
    if payload["note_sequence_byte_parity"] is not True:
        raise SealError("tensor coverage does not prove NoteSequence byte parity")
    if _contains_final_lock_hash(payload):
        raise SealError("tensor coverage contains a final lock hash cycle")


def _validate_security_scan(
    payload: dict[str, JsonValue],
    advisory_content: bytes,
    seal: LoadedSealEvidence,
) -> None:
    if (
        not isinstance(payload.get("schema"), str)
        or not payload["schema"]
        or payload.get("advisory_snapshot_sha256") != seal.payload["advisory_snapshot_sha256"]
        or sha256_hex(advisory_content) != seal.payload["advisory_snapshot_sha256"]
    ):
        raise SealError("security scan or advisory snapshot does not match the candidate")
    _parse_canonical_json_content(advisory_content, "advisory snapshot")
    if _contains_final_lock_hash(payload):
        raise SealError("security scan contains a final lock hash cycle")


def _validate_oci_layout(
    payload: dict[str, JsonValue],
    directory: Path,
    seal: LoadedSealEvidence,
) -> None:
    if set(payload) != _OCI_LAYOUT_KEYS:
        raise SealError("OCI layout manifest fields do not match the exact schema")
    archive = payload["archive"]
    if not isinstance(archive, dict) or set(archive) != _ARCHIVE_KEYS:
        raise SealError("OCI layout archive identity is invalid")
    if archive != _plain_json(seal.payload["oci_layout_archive"]):
        raise SealError("OCI layout archive identity does not match the proposed seal")
    if (
        payload["config_digest"] != seal.payload["runtime_image_config_digest"]
        or payload["image_manifest_digest"] != seal.payload["runtime_image_manifest_digest"]
        or payload["layer_digests"] != _plain_json(seal.payload["runtime_image_layer_digests"])
        or not isinstance(payload["schema"], str)
        or not payload["schema"]
    ):
        raise SealError("OCI layout image identities do not match the proposed seal")
    name = archive["name"]
    if not isinstance(name, str) or not name or PurePosixPath(name).name != name or "\\" in name:
        raise SealError("OCI archive filename is invalid")
    content = _read_regular(directory / name, "OCI archive")
    if len(content) != archive["size"] or sha256_hex(content) != archive["sha256"]:
        raise SealError("OCI archive hash or size does not match the manifest")
    if _contains_final_lock_hash(payload):
        raise SealError("OCI layout manifest contains a final lock hash cycle")


def _validate_smoke(
    payload: dict[str, JsonValue],
    directory: Path,
    seal: LoadedSealEvidence,
) -> None:
    if set(payload) != _SMOKE_ORACLE_KEYS or payload["schema"] != SMOKE_ORACLE_SCHEMA:
        raise SealError("smoke oracle fields do not match the exact schema")
    if (
        not isinstance(payload["input_audio_frame_count"], int)
        or isinstance(payload["input_audio_frame_count"], bool)
        or payload["input_audio_frame_count"] <= 0
        or not isinstance(payload["native_events"], list)
        or not payload["native_events"]
    ):
        raise SealError("smoke oracle has no exact nonempty prediction")
    audio = _read_regular(directory / SMOKE_AUDIO_NAME, "smoke audio")
    prediction = _read_regular(directory / SMOKE_PREDICTION_NAME, "smoke prediction")
    if (
        not prediction
        or sha256_hex(audio) != seal.payload["smoke_audio_sha256"]
        or payload["input_audio_sha256"] != seal.payload["smoke_audio_sha256"]
        or sha256_hex(prediction) != seal.payload["smoke_prediction_sha256"]
    ):
        raise SealError("smoke inputs or prediction do not match the proposed seal")
    if _contains_final_lock_hash(payload):
        raise SealError("smoke oracle contains a final lock hash cycle")


def _require_current_native_worker(evidence: NativeHostEvidence) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"amd64", "x86_64"}:
        raise SealError("official execution requires a native Linux x86_64 host")
    if evidence.kind == "github_hosted" and (
        os.environ.get("RUNNER_OS") != "Linux" or os.environ.get("RUNNER_ARCH") != "X64"
    ):
        raise SealError("GitHub-hosted evidence requires RUNNER_OS=Linux and RUNNER_ARCH=X64")


def _read_canonical_object(path: Path, label: str) -> tuple[dict[str, JsonValue], bytes]:
    content = _read_regular(path, label)
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SealError(f"{label} must be canonical JSON with one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise SealError(f"{label} is not strict canonical JSON: {error}") from None
    if not isinstance(value, dict):
        raise SealError(f"{label} must be an object")
    return value, content


def _parse_canonical_json_content(content: bytes, label: str) -> dict[str, JsonValue]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SealError(f"{label} must be canonical JSON with one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise SealError(f"{label} is not strict canonical JSON: {error}") from None
    if not isinstance(value, dict):
        raise SealError(f"{label} must be an object")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    try:
        return read_regular_file_no_follow(path)
    except OSError:
        raise SealError(f"{label} must be a no-follow regular file") from None


def _require_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError:
        raise SealError(f"{label} must be an existing directory") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise SealError(f"{label} must be an existing directory")
    return path


def _require_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise SealError(f"{label} path cannot be inspected") from None
    raise SealError(f"{label} output already exists")


def _validate_output_paths(paths: Mapping[str, Path]) -> None:
    values = [os.path.abspath(path) for path in paths.values()]
    if len(set(values)) != len(values):
        raise SealError("seal output paths must be distinct")


def _contains_final_lock_hash(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(set(value) & _FINAL_LOCK_HASH_KEYS) or any(
            _contains_final_lock_hash(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_final_lock_hash(child) for child in value)
    return False


def _plain_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(child) for child in value]
    if isinstance(value, list):
        return [_plain_json(child) for child in value]
    return cast(JsonValue, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-host")
    validate.add_argument("--evidence", type=Path, required=True)

    packages = commands.add_parser("materialize-system-packages")
    packages.add_argument("--host-evidence", type=Path, required=True)
    packages.add_argument("--bundle", type=Path, required=True)
    packages.add_argument("--build-args-output", type=Path, required=True)

    base_system = commands.add_parser("attest-base-system")
    base_system.add_argument("--request", type=Path, required=True)
    base_system.add_argument("--host-evidence", type=Path, required=True)
    base_system.add_argument("--image", required=True)
    base_system.add_argument("--output", type=Path, required=True)

    measurement = commands.add_parser("measure")
    measurement.add_argument("--request", type=Path, required=True)
    measurement.add_argument("--host-evidence", type=Path, required=True)
    measurement.add_argument("--image", required=True)
    measurement.add_argument("--model-cache", type=Path, required=True)
    measurement.add_argument("--checkpoint-evidence", type=Path, required=True)
    measurement.add_argument("--base-system-evidence", type=Path, required=True)
    measurement.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--request", type=Path, required=True)
    calibration.add_argument("--measurement-evidence", type=Path, required=True)
    calibration.add_argument("--checkpoint-evidence", type=Path, required=True)
    calibration.add_argument("--base-system-evidence", type=Path, required=True)
    calibration.add_argument("--image", required=True)
    calibration.add_argument("--host-evidence", type=Path, required=True)
    calibration.add_argument("--model-cache", type=Path, required=True)
    calibration.add_argument("--candidate-authority", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    seal = commands.add_parser("seal")
    seal.add_argument("--candidate", type=Path, required=True)
    seal.add_argument("--backend-lock", type=Path, required=True)
    seal.add_argument("--runtime-lock", type=Path, required=True)
    seal.add_argument("--seal-evidence", type=Path, required=True)
    seal.add_argument("--conversion-audit", type=Path, required=True)
    seal.add_argument("--host-adapter-source-manifest", type=Path, required=True)
    seal.add_argument("--tensor-coverage", type=Path, required=True)
    seal.add_argument("--security-scan", type=Path, required=True)
    seal.add_argument("--oci-layout-manifest", type=Path, required=True)
    seal.add_argument("--smoke-oracle", type=Path, required=True)
    return parser


# The CLI branches mirror the four exact Task 8 surfaces.
# pylint: disable-next=too-many-branches
def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate-host":
            load_native_host_evidence(arguments.evidence)
            _write_producer_summary(ProducerOutcome("validated", 0, None, None))
            return 0
        if arguments.command == "materialize-system-packages":
            materialize_system_packages(
                host_evidence=arguments.host_evidence,
                bundle=arguments.bundle,
                build_args_output=arguments.build_args_output,
            )
            _write_producer_summary(ProducerOutcome("materialized", 0, None, None))
            return 0
        if arguments.command == "attest-base-system":
            published = attest_base_system(
                request_path=arguments.request,
                host_evidence_path=arguments.host_evidence,
                image=arguments.image,
                output_path=arguments.output,
            )
            _write_producer_summary(
                ProducerOutcome("attested", 0, published.path, published.sha256)
            )
            return 0
        if arguments.command == "measure":
            published = measure(
                request_path=arguments.request,
                host_evidence_path=arguments.host_evidence,
                image=arguments.image,
                model_cache=arguments.model_cache,
                checkpoint_evidence_path=arguments.checkpoint_evidence,
                base_system_evidence_path=arguments.base_system_evidence,
                output_path=arguments.output,
            )
            _write_producer_summary(
                ProducerOutcome("measured", 0, published.path, published.sha256)
            )
            return 0
        if arguments.command == "calibrate":
            published = calibrate(
                request_path=arguments.request,
                measurement_evidence_path=arguments.measurement_evidence,
                checkpoint_evidence_path=arguments.checkpoint_evidence,
                base_system_evidence_path=arguments.base_system_evidence,
                image=arguments.image,
                host_evidence=arguments.host_evidence,
                model_cache=arguments.model_cache,
                output=arguments.output,
                candidate_authority_path=arguments.candidate_authority,
            )
            _write_producer_summary(
                ProducerOutcome("calibrated", 0, published.artifact.path, published.artifact.sha256)
            )
            return 0
        published = seal_candidate(
            candidate=arguments.candidate,
            backend_lock=arguments.backend_lock,
            runtime_lock=arguments.runtime_lock,
            seal_evidence=arguments.seal_evidence,
            conversion_audit=arguments.conversion_audit,
            host_adapter_source_manifest=arguments.host_adapter_source_manifest,
            tensor_coverage=arguments.tensor_coverage,
            security_scan=arguments.security_scan,
            oci_layout_manifest=arguments.oci_layout_manifest,
            smoke_oracle=arguments.smoke_oracle,
            repository_root=Path.cwd(),
        )
        _write_producer_summary(
            ProducerOutcome(
                "sealed", 0, published.artifacts[-1].path, published.artifacts[-1].sha256
            )
        )
        return 0
    except SealError as error:
        exit_code: Literal[1, 2] = 2 if isinstance(error, SealIntegrityError) else 1
        _write_producer_summary(ProducerOutcome("failed", exit_code, None, None))
        print(f"OaF seal failed: {error}", file=sys.stderr)
        return exit_code


def _write_producer_summary(outcome: ProducerOutcome) -> None:
    """Emit the shared machine result only after argparse has accepted the command."""

    summary: JsonValue = {
        "exit_code": outcome.exit_code,
        "report_path": None if outcome.report_path is None else os.fspath(outcome.report_path),
        "report_sha256": outcome.report_sha256,
        "status": outcome.status,
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary, trailing_newline=True))


if __name__ == "__main__":
    raise SystemExit(main())
