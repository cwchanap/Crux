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
    REQUIRED_ENVIRONMENT,
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
    CheckpointAcquisitionEvidence,
    CheckpointAcquisitionRequest,
    load_checkpoint_acquisition_evidence,
    load_checkpoint_acquisition_request,
)
from tools.hpa320.audit_legacy_tf2_conversion import (
    CANDIDATE_MANIFEST_NAME,
    CANDIDATE_MANIFEST_SCHEMA,
)
from tools.hpa320.oaf_host_attestation import (
    HostAttestationError,
    NativeHostAttestationBundle,
    load_native_host_attestation_bundle,
)
from tools.hpa320.oaf_native_artifacts import CANDIDATE_ARTIFACT_PATHS, CANDIDATE_ARTIFACTS
from tools.hpa320.oaf_oci import ImageBuildRecipe, OciArchiveRecipe
from tools.hpa320.oaf_system_packages import (
    BASE_SYSTEM_PACKAGE_EVIDENCE_SCHEMA,
    BaseSystemPackageEvidence,
    BaseSystemPackageRequest,
    ProbeResult,
    SystemPackage,
    SystemPackageError,
    inventory_sha256,
    load_base_system_package_evidence,
    load_base_system_package_request,
)

CANDIDATE_HOST_ATTESTATION_ROOT = Path(
    "docs/superpowers/evidence/hpa-320/native/candidate-host-attestation"
)
SMOKE_AUDIO_NAME = "canonical.wav"
SMOKE_PREDICTION_NAME = "smoke-prediction.jsonl"
_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"

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
        "artifacts",
        "backend_lock_payload_sha256",
        "calibration_bootstrap_evidence_sha256",
        "calibration_measurement_evidence_sha256",
        "checkpoint_components",
        "checkpoint_prefix",
        "model_artifact_set_sha256",
        "native_host_attestation_bundle_sha256",
        "required_inference_inventory_sha256",
        "runtime_lock_payload_sha256",
        "schema",
        "seal_evidence_payload_sha256",
        "seal_profile_request_sha256",
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
        "base_image_config_digest",
        "base_image_layer_diff_ids",
        "base_image_layer_digests",
        "config_digest",
        "image_manifest_digest",
        "index_digest",
        "layer_diff_ids",
        "layer_digests",
        "schema",
    }
)
_ARCHIVE_KEYS = frozenset({"name", "sha256", "size"})
_BOOTSTRAP_EVIDENCE_KEYS = frozenset(
    {
        "base_image_config_digest",
        "base_image_layer_diff_ids",
        "base_image_layer_digests",
        "build_context_manifest_sha256",
        "calibration_bootstrap_request_sha256",
        "image_build",
        "native_host_attestation_bundle_sha256",
        "native_host_evidence",
        "oci_layout_archive",
        "oci_layout_manifest_sha256",
        "runtime_image_config_digest",
        "runtime_image_index_digest",
        "runtime_image_layer_diff_ids",
        "runtime_image_layer_digests",
        "runtime_image_manifest_digest",
        "schema",
    }
)
CALIBRATION_BOOTSTRAP_EVIDENCE_SCHEMA = "crux.oaf-calibration-bootstrap-evidence/v2"
_FINAL_LOCK_HASH_KEYS = frozenset({"backend_lock_sha256", "runtime_lock_sha256"})
CALIBRATION_MEASUREMENT_REQUEST_SCHEMA = "crux.oaf-calibration-measurement-request/v1"
CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA = "crux.oaf-calibration-measurement-evidence/v2"
CALIBRATION_BOOTSTRAP_REQUEST_SCHEMA = "crux.oaf-calibration-bootstrap-request/v1"
SEAL_PROFILE_REQUEST_SCHEMA = "crux.oaf-seal-profile-request/v1"
SEAL_CANDIDATE_SCHEMA = "crux.oaf-seal-candidate/v2"
_MEASUREMENT_REQUEST_KEYS = frozenset(
    {
        "backend_id",
        "calibration_bootstrap_evidence_sha256",
        "calibration_bootstrap_request_sha256",
        "fixture_derivation",
        "fixtures",
        "output_schemas",
        "repetition_count",
        "required_metrics",
        "schema",
    }
)
_MEASUREMENT_EVIDENCE_KEYS = frozenset(
    {
        "base_system_package_evidence_sha256",
        "calibration_bootstrap_evidence_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "measurement_rows",
        "native_host_attestation_bundle_sha256",
        "native_host_evidence",
        "request_sha256",
        "runtime_image_config_digest",
        "runtime_image_manifest_digest",
        "schema",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "base_system_package_evidence_sha256",
        "base_system_package_request_sha256",
        "calibration_bootstrap_evidence_sha256",
        "calibration_bootstrap_request_sha256",
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
        "inference_call_count_after",
        "inference_call_count_before",
        "input_audio_sha256",
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
_FIXTURE_DERIVATION_KEYS = frozenset(
    {
        "algorithm",
        "canonical_header_bytes",
        "channel_count",
        "sample_rate",
        "sample_width_bytes",
        "source_path",
        "source_sha256",
    }
)
_CALIBRATION_FIXTURE_KEYS = frozenset(
    {
        "audio_frame_count",
        "input_audio_sha256",
        "input_view_id",
        "source_audio_id",
        "source_audio_sha256",
        "wav_byte_length",
    }
)
_CHECKPOINT_REQUEST_PATH = Path(
    "config/benchmark/backends/"
    "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
)
_CALIBRATION_BOOTSTRAP_REQUEST_PATH = Path(
    "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json"
)
_CALIBRATION_MEASUREMENT_REQUEST_PATH = Path(
    "config/benchmark/backends/"
    "magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json"
)
_SEAL_PROFILE_REQUEST_PATH = Path(
    "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json"
)
_BASE_SYSTEM_REQUEST_PATH = Path("runtime/oaf_tf1/base-system-package-request.json")
_ACCEPTED_NATIVE_EVIDENCE_ROOT = Path("docs/superpowers/evidence/hpa-320/native")
_OPERATIONAL_IMAGE_IDENTITY_FIELDS = (
    "base_image_config_digest",
    "base_image_layer_diff_ids",
    "base_image_layer_digests",
    "build_context_manifest_sha256",
    "calibration_bootstrap_request_sha256",
    "image_build",
    "oci_layout_archive",
    "oci_layout_manifest_sha256",
    "runtime_image_config_digest",
    "runtime_image_index_digest",
    "runtime_image_layer_diff_ids",
    "runtime_image_layer_digests",
    "runtime_image_manifest_digest",
)
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
_BOOTSTRAP_REQUEST_KEYS = frozenset(
    {
        "backend_id",
        "base_image_manifest_digest",
        "base_system_package_request_sha256",
        "build_context_manifest_sha256",
        "checkpoint_acquisition_request_sha256",
        "container_restrictions",
        "distribution_build_manifest_sha256",
        "environment",
        "image_build",
        "instrumentation_patch_sha256",
        "python_coerce_c_locale",
        "resource_ceiling",
        "runner_source_manifest_sha256",
        "runtime_gid",
        "runtime_uid",
        "schema",
        "upstream_source_manifest_sha256",
    }
)
_IMAGE_BUILD_KEYS = frozenset(
    {
        "annotations",
        "buildkit_image",
        "buildkit_version",
        "buildx_binary_sha256",
        "buildx_binary_size",
        "buildx_binary_url",
        "buildx_version",
        "compression",
        "compression_level",
        "dockerfile_frontend",
        "dockerfile_frontend_version",
        "exporter",
        "exporter_tar",
        "force_compression",
        "inline_cache",
        "multi_platform_deterministic",
        "oci_archive",
        "oci_media_types",
        "platform",
        "provenance",
        "rewrite_timestamp",
        "sbom",
        "source_date_epoch",
    }
)
_OCI_ARCHIVE_RECIPE_KEYS = frozenset(
    {
        "compression",
        "final_zero_blocks",
        "format",
        "gid",
        "gname",
        "member_mode",
        "member_types",
        "mtime",
        "path_order",
        "uid",
        "uname",
    }
)
_BOOTSTRAP_HASH_FIELDS = (
    (
        "runtime/oaf_tf1/base-system-package-request.json",
        "base_system_package_request_sha256",
    ),
    (
        "runtime/oaf_tf1/build-context-manifest.json",
        "build_context_manifest_sha256",
    ),
    (
        (
            "config/benchmark/backends/"
            "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        ),
        "checkpoint_acquisition_request_sha256",
    ),
    *_HASH_FIELDS,
)
_BOOTSTRAP_RESOURCE_CEILING = {
    "cpu_limit_millis": 2_000,
    "memory_limit_bytes": 4_294_967_296,
    "monitor_interval_millis": 10,
    "pid_limit": 256,
    "request_deadline_seconds": 1_800,
    "shm_bytes": 1_073_741_824,
    "startup_deadline_seconds": 300,
    "stderr_max_line_bytes": 65_536,
    "stderr_read_chunk_bytes": 65_536,
    "stderr_ring_buffer_bytes": 1_048_576,
    "stdout_max_line_bytes": 134_217_728,
    "tmp_bytes": 1_073_741_824,
}
_CONTAINER_RESTRICTIONS = {
    "drop_capabilities": ["ALL"],
    "network": "none",
    "no_new_privileges": True,
    "platform": "linux/amd64",
    "read_only_root": True,
}


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
    input_audio_sha256: str
    input_frame_count: int
    repetition: int
    process_instance_id: str
    inference_call_count_before: int
    inference_call_count_after: int
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
class CalibrationBootstrapRequest:
    backend_id: str
    build_context_manifest_sha256: str
    image_build: ImageBuildRecipe
    runtime_uid: int
    runtime_gid: int
    sha256: str
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class CalibrationBootstrapEvidence:
    """Authenticated immutable image identity produced by bootstrap."""

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
    manifest: dict[str, JsonValue] = {
        "covered_roots": list(HOST_ADAPTER_SOURCE_PATHS),
        "files": rows,
        "schema": HOST_ADAPTER_SCHEMA,
    }
    _validate_source_manifest_payload(manifest, HOST_ADAPTER_SCHEMA)
    return manifest


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
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    output_path: Path,
) -> PublishedArtifact:
    """Publish immutable base-system evidence after exact native probes."""

    request = load_base_system_package_request(Path(request_path))
    (
        bootstrap_request,
        bootstrap,
        _bundle,
        host,
        _host_payload,
    ) = _authenticate_bootstrap_for_phase(
        bootstrap_request_path=Path(bootstrap_request_path),
        bootstrap_evidence_path=Path(bootstrap_evidence_path),
        host_attestation_bundle_path=Path(host_attestation_bundle_path),
        host_evidence_path=Path(host_evidence_path),
        phase="bootstrap",
    )
    if request.sha256 != bootstrap_request.payload["base_system_package_request_sha256"]:
        raise SealIntegrityError(
            "base-system request does not match calibration bootstrap authority"
        )
    image = cast(str, bootstrap.payload["runtime_image_config_digest"])
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
        "schema": BASE_SYSTEM_PACKAGE_EVIDENCE_SCHEMA,
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


def load_calibration_bootstrap_request(path: Path) -> CalibrationBootstrapRequest:
    """Strict-load the sole image-build and diagnostic-bootstrap authority."""

    request_path = Path(path)
    payload, content = _read_canonical_object(request_path, "calibration bootstrap request")
    image_build = _validate_calibration_bootstrap_request_payload(payload)
    try:
        repository = request_path.parents[3]
    except IndexError:
        raise SealError("calibration bootstrap request repository path is invalid") from None
    for relative, field in _BOOTSTRAP_HASH_FIELDS:
        expected = cast(str, payload[field])
        actual = sha256_hex(_read_regular(repository / relative, relative))
        if actual != expected:
            raise SealError(f"calibration bootstrap request {field} hash does not match")
    return CalibrationBootstrapRequest(
        backend_id=cast(str, payload["backend_id"]),
        build_context_manifest_sha256=cast(str, payload["build_context_manifest_sha256"]),
        image_build=image_build,
        runtime_uid=cast(int, payload["runtime_uid"]),
        runtime_gid=cast(int, payload["runtime_gid"]),
        sha256=sha256_hex(content),
        payload=payload,
    )


def reissue_calibration_bootstrap_request(
    *,
    request_path: Path,
    repository_root: Path,
) -> str:
    """Reissue only the seven cross-hashes in one canonical bootstrap request."""

    request_path = Path(request_path)
    repository_root = Path(repository_root)
    payload, original = _read_canonical_object(
        request_path,
        "calibration bootstrap request",
    )
    reissued = dict(payload)
    for relative, field in _BOOTSTRAP_HASH_FIELDS:
        content = _read_regular(repository_root / relative, relative)
        reissued[field] = sha256_hex(content)
    _validate_calibration_bootstrap_request_payload(reissued)
    content = canonical_json_bytes(reissued, trailing_newline=True)
    if strict_json_loads(original[:-1], require_canonical=True) != payload:
        raise SealError("existing calibration bootstrap request is not canonical")
    _atomic_replace_regular_file(request_path, content)
    loaded = load_calibration_bootstrap_request(request_path)
    return loaded.sha256


def _validate_calibration_bootstrap_request_payload(
    payload: Mapping[str, JsonValue],
) -> ImageBuildRecipe:
    if (
        set(payload) != _BOOTSTRAP_REQUEST_KEYS
        or payload["schema"] != CALIBRATION_BOOTSTRAP_REQUEST_SCHEMA
    ):
        raise SealError("calibration bootstrap request fields are invalid")
    if (
        payload["backend_id"] != "magenta-egmd-tf1-94529798-8hit-v1"
        or payload["base_image_manifest_digest"]
        != "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
        or payload["runtime_uid"] != 65_532
        or type(payload["runtime_uid"]) is not int
        or payload["runtime_gid"] != 65_532
        or type(payload["runtime_gid"]) is not int
        or payload["python_coerce_c_locale"] != "0"
    ):
        raise SealError("calibration bootstrap request authority is invalid")
    restrictions = payload["container_restrictions"]
    if (
        not isinstance(restrictions, dict)
        or restrictions != _CONTAINER_RESTRICTIONS
        or type(restrictions.get("no_new_privileges")) is not bool
        or type(restrictions.get("read_only_root")) is not bool
    ):
        raise SealError("calibration bootstrap container restrictions are invalid")
    environment = payload["environment"]
    if (
        not isinstance(environment, dict)
        or environment != dict(REQUIRED_ENVIRONMENT)
        or any(type(value) is not str for value in environment.values())
    ):
        raise SealError("calibration bootstrap environment is invalid")
    ceiling = payload["resource_ceiling"]
    if (
        not isinstance(ceiling, dict)
        or ceiling != _BOOTSTRAP_RESOURCE_CEILING
        or any(type(value) is not int for value in ceiling.values())
    ):
        raise SealError("calibration bootstrap resource ceiling is invalid")
    image_build = _load_image_build_recipe(payload["image_build"])
    for _relative, field in _BOOTSTRAP_HASH_FIELDS:
        _require_sha256_value(
            payload[field],
            f"calibration bootstrap request {field}",
        )
    return image_build


def _load_image_build_recipe(value: object) -> ImageBuildRecipe:
    if not isinstance(value, dict) or set(value) != _IMAGE_BUILD_KEYS:
        raise SealError("calibration bootstrap image build fields are invalid")
    archive_value = value["oci_archive"]
    if not isinstance(archive_value, dict) or set(archive_value) != _OCI_ARCHIVE_RECIPE_KEYS:
        raise SealError("calibration bootstrap image build archive fields are invalid")
    archive = OciArchiveRecipe(**cast(dict[str, Any], archive_value))
    expected_archive = OciArchiveRecipe(
        compression="none",
        final_zero_blocks=2,
        format="posix-ustar",
        gid=0,
        gname="",
        member_mode=420,
        member_types="regular-files-only",
        mtime=0,
        path_order="utf8-byte",
        uid=0,
        uname="",
    )
    archive_integer_fields = (
        "final_zero_blocks",
        "gid",
        "member_mode",
        "mtime",
        "uid",
    )
    if archive != expected_archive or any(
        type(archive_value[field]) is not int for field in archive_integer_fields
    ):
        raise SealError("calibration bootstrap image build archive recipe is invalid")
    expected = {
        "annotations": [],
        "buildkit_image": (
            "moby/buildkit@sha256:63db51c9b30208a7c2b1c40392c7ebb9ce2f85ba238a18a85420f8f5ea2d4684"
        ),
        "buildkit_version": "v0.31.2",
        "buildx_binary_sha256": (
            "d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda"
        ),
        "buildx_binary_size": 65_265_826,
        "buildx_binary_url": (
            "https://github.com/docker/buildx/releases/download/v0.35.0/buildx-v0.35.0.linux-amd64"
        ),
        "buildx_version": "v0.35.0",
        "compression": "gzip",
        "compression_level": 6,
        "dockerfile_frontend": (
            "docker/dockerfile-upstream@sha256:"
            "3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516"
        ),
        "dockerfile_frontend_version": "1.25.0",
        "exporter": "oci",
        "exporter_tar": False,
        "force_compression": False,
        "inline_cache": False,
        "multi_platform_deterministic": True,
        "oci_archive": archive_value,
        "oci_media_types": True,
        "platform": "linux/amd64",
        "provenance": False,
        "rewrite_timestamp": True,
        "sbom": False,
        "source_date_epoch": 0,
    }
    boolean_fields = (
        "exporter_tar",
        "force_compression",
        "inline_cache",
        "multi_platform_deterministic",
        "oci_media_types",
        "provenance",
        "rewrite_timestamp",
        "sbom",
    )
    integer_fields = ("buildx_binary_size", "compression_level", "source_date_epoch")
    if (
        value != expected
        or any(type(value[field]) is not bool for field in boolean_fields)
        or any(type(value[field]) is not int for field in integer_fields)
    ):
        raise SealError("calibration bootstrap image build recipe is invalid")
    return ImageBuildRecipe(
        annotations=(),
        buildkit_image=cast(str, value["buildkit_image"]),
        buildkit_version=cast(str, value["buildkit_version"]),
        buildx_binary_sha256=cast(str, value["buildx_binary_sha256"]),
        buildx_binary_size=cast(int, value["buildx_binary_size"]),
        buildx_binary_url=cast(str, value["buildx_binary_url"]),
        buildx_version=cast(str, value["buildx_version"]),
        compression=cast(str, value["compression"]),
        compression_level=cast(int, value["compression_level"]),
        dockerfile_frontend=cast(str, value["dockerfile_frontend"]),
        dockerfile_frontend_version=cast(str, value["dockerfile_frontend_version"]),
        exporter=cast(str, value["exporter"]),
        exporter_tar=cast(bool, value["exporter_tar"]),
        force_compression=cast(bool, value["force_compression"]),
        inline_cache=cast(bool, value["inline_cache"]),
        multi_platform_deterministic=cast(bool, value["multi_platform_deterministic"]),
        oci_archive=archive,
        oci_media_types=cast(bool, value["oci_media_types"]),
        platform=cast(str, value["platform"]),
        provenance=cast(bool, value["provenance"]),
        rewrite_timestamp=cast(bool, value["rewrite_timestamp"]),
        sbom=cast(bool, value["sbom"]),
        source_date_epoch=cast(int, value["source_date_epoch"]),
    )


def load_calibration_bootstrap_evidence(
    request_path: Path,
    evidence_path: Path,
) -> tuple[CalibrationBootstrapRequest, CalibrationBootstrapEvidence]:
    """Authenticate the bootstrap request/evidence image identity chain."""

    request = load_calibration_bootstrap_request(Path(request_path))
    payload, content = _read_canonical_object(Path(evidence_path), "calibration bootstrap evidence")
    if (
        set(payload) != _BOOTSTRAP_EVIDENCE_KEYS
        or payload["schema"] != CALIBRATION_BOOTSTRAP_EVIDENCE_SCHEMA
    ):
        raise SealError("calibration bootstrap evidence fields are invalid")
    if (
        payload["calibration_bootstrap_request_sha256"] != request.sha256
        or payload["build_context_manifest_sha256"] != request.build_context_manifest_sha256
        or payload["image_build"] != request.payload["image_build"]
    ):
        raise SealIntegrityError("calibration bootstrap evidence does not reproduce its request")
    _validate_calibration_bootstrap_evidence_payload(payload)
    return request, CalibrationBootstrapEvidence(
        sha256=sha256_hex(content),
        payload=payload,
    )


def _validate_calibration_bootstrap_evidence_payload(
    payload: Mapping[str, JsonValue],
) -> None:
    if (
        set(payload) != _BOOTSTRAP_EVIDENCE_KEYS
        or payload["schema"] != CALIBRATION_BOOTSTRAP_EVIDENCE_SCHEMA
    ):
        raise SealError("calibration bootstrap evidence fields are invalid")
    for field in (
        "build_context_manifest_sha256",
        "calibration_bootstrap_request_sha256",
        "native_host_attestation_bundle_sha256",
        "oci_layout_manifest_sha256",
    ):
        _require_sha256_value(payload[field], f"calibration bootstrap evidence {field}")
    _load_image_build_recipe(payload["image_build"])
    for field in (
        "base_image_config_digest",
        "runtime_image_config_digest",
        "runtime_image_index_digest",
        "runtime_image_manifest_digest",
    ):
        _require_oci_digest(payload[field], f"calibration bootstrap evidence {field}")
    base_layers = _require_digest_array(
        payload["base_image_layer_digests"],
        "calibration bootstrap base image layers",
    )
    base_diff_ids = _require_digest_array(
        payload["base_image_layer_diff_ids"],
        "calibration bootstrap base image DiffIDs",
    )
    runtime_layers = _require_digest_array(
        payload["runtime_image_layer_digests"],
        "calibration bootstrap runtime image layers",
    )
    runtime_diff_ids = _require_digest_array(
        payload["runtime_image_layer_diff_ids"],
        "calibration bootstrap runtime image DiffIDs",
    )
    if (
        not base_layers
        or len(base_layers) != len(base_diff_ids)
        or len(runtime_layers) != len(runtime_diff_ids)
        or runtime_layers[: len(base_layers)] != base_layers
        or runtime_diff_ids[: len(base_diff_ids)] != base_diff_ids
    ):
        raise SealIntegrityError(
            "calibration bootstrap base image layer prefix or order is invalid"
        )
    _validate_checkpoint_identity(
        payload["oci_layout_archive"],
        "calibration bootstrap OCI archive",
    )
    _native_host_from_record(
        payload["native_host_evidence"],
        "calibration bootstrap native host evidence",
    )


def _authenticate_phase_host(
    *,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    phase: str,
) -> tuple[NativeHostAttestationBundle, NativeHostEvidence, dict[str, JsonValue]]:
    try:
        bundle = load_native_host_attestation_bundle(
            Path(host_attestation_bundle_path),
            expected_phase=phase,
        )
    except HostAttestationError as error:
        raise SealIntegrityError(
            f"{phase} native-host attestation bundle is invalid: {error}"
        ) from None
    host_content = _read_regular(Path(host_evidence_path), f"{phase} native host evidence")
    bundled_host_path = Path(host_attestation_bundle_path).parent / bundle.native_host_evidence.name
    bundled_host_content = _read_regular(
        bundled_host_path,
        f"{phase} bundled native host evidence",
    )
    if (
        host_content != bundled_host_content
        or len(host_content) != bundle.native_host_evidence.size
        or sha256_hex(host_content) != bundle.native_host_evidence.sha256
    ):
        raise SealIntegrityError(
            f"{phase} native host evidence does not match its attestation bundle"
        )
    host_payload = _parse_canonical_json_content(
        host_content,
        f"{phase} native host evidence",
    )
    host = _native_host_from_record(host_payload, f"{phase} native host evidence")
    _require_current_native_worker(host)
    return bundle, host, host_payload


def _authenticate_bootstrap_for_phase(
    *,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    phase: str,
) -> tuple[
    CalibrationBootstrapRequest,
    CalibrationBootstrapEvidence,
    NativeHostAttestationBundle,
    NativeHostEvidence,
    dict[str, JsonValue],
]:
    request, evidence = load_calibration_bootstrap_evidence(
        Path(bootstrap_request_path),
        Path(bootstrap_evidence_path),
    )
    bundle, host, host_payload = _authenticate_phase_host(
        host_attestation_bundle_path=Path(host_attestation_bundle_path),
        host_evidence_path=Path(host_evidence_path),
        phase=phase,
    )
    if phase == "bootstrap" and (
        evidence.payload["native_host_attestation_bundle_sha256"] != bundle.sha256
        or evidence.payload["native_host_evidence"] != host_payload
    ):
        raise SealIntegrityError("bootstrap evidence does not match its owning native-host bundle")
    _require_imported_runtime_image(cast(str, evidence.payload["runtime_image_config_digest"]))
    return request, evidence, bundle, host, host_payload


def _require_imported_runtime_image(config_digest: str) -> None:
    observed = (
        _docker_capture(
            (
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                config_digest,
            ),
            "imported runtime image",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if observed != config_digest:
        raise SealIntegrityError(
            "imported runtime image config digest does not match bootstrap evidence"
        )


def _require_oci_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise SealError(f"{label} is invalid")
    return value


def _require_digest_array(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SealError(f"{label} are invalid")
    return tuple(_require_oci_digest(item, label) for item in value)


def _validate_checkpoint_identity(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != _ARCHIVE_KEYS:
        raise SealError(f"{label} identity is invalid")
    name = value["name"]
    size = value["size"]
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).name != name
        or "\\" in name
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise SealError(f"{label} identity is invalid")
    _require_sha256_value(value["sha256"], f"{label} hash")


def _native_host_from_record(value: object, label: str) -> NativeHostEvidence:
    if not isinstance(value, dict) or set(value) != _HOST_EVIDENCE_KEYS:
        raise SealError(f"{label} fields are invalid")
    try:
        return NativeHostEvidence(
            kind=cast(Any, value["kind"]),
            payload=cast(Mapping[str, JsonValue], value["payload"]),
            sha256=cast(str, value["sha256"]),
            official_execution_allowed=cast(bool, value["official_execution_allowed"]),
        )
    except (StrictJsonError, TypeError, ValueError) as error:
        raise SealError(f"{label} is invalid: {error}") from None


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
    for field in (
        "calibration_bootstrap_evidence_sha256",
        "calibration_bootstrap_request_sha256",
    ):
        _require_sha256_value(payload[field], f"calibration measurement request {field}")
    repetitions = payload["repetition_count"]
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions != 3:
        raise SealError("calibration measurement request bounds are invalid")
    derivation = payload["fixture_derivation"]
    if (
        not isinstance(derivation, dict)
        or set(derivation) != _FIXTURE_DERIVATION_KEYS
        or derivation["algorithm"] != "repeat-pcm-s16le-v1"
        or derivation["canonical_header_bytes"] != 44
        or derivation["channel_count"] != 1
        or derivation["sample_rate"] != 44100
        or derivation["sample_width_bytes"] != 2
        or not isinstance(derivation["source_path"], str)
        or not derivation["source_path"]
    ):
        raise SealError("calibration fixture derivation is invalid")
    _require_sha256_value(derivation["source_sha256"], "calibration fixture source hash")
    fixtures = payload["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise SealError("calibration measurement fixtures are invalid")
    counts: list[int] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict) or set(fixture) != _CALIBRATION_FIXTURE_KEYS:
            raise SealError("calibration measurement fixture fields are invalid")
        for field in ("input_audio_sha256", "source_audio_sha256"):
            _require_sha256_value(fixture[field], "calibration measurement fixture hash")
        frame_count = fixture["audio_frame_count"]
        wav_length = fixture["wav_byte_length"]
        if (
            not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or frame_count <= 0
            or not isinstance(wav_length, int)
            or isinstance(wav_length, bool)
            or wav_length != 44 + frame_count * 2
        ):
            raise SealError("calibration measurement fixture bounds are invalid")
        counts.append(frame_count)
        if not all(
            isinstance(fixture[field], str) and fixture[field]
            for field in ("input_view_id", "source_audio_id")
        ):
            raise SealError("calibration measurement fixture identities are invalid")
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise SealError("calibration measurement fixture frame counts are invalid")
    if not _string_list(payload["required_metrics"]) or payload["output_schemas"] != [
        CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA
    ]:
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
        "calibration_bootstrap_evidence_sha256",
        "calibration_bootstrap_request_sha256",
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
    if payload["runtime_uid"] != 65_532 or payload["runtime_gid"] != 65_532:
        raise SealError("seal profile request runtime identity must be 65532:65532")
    for field, ceiling in _BOOTSTRAP_RESOURCE_CEILING.items():
        if field == "monitor_interval_millis":
            continue
        if cast(int, payload[field]) > ceiling:
            raise SealError(f"seal profile request {field} exceeds bootstrap ceiling")
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
            payload, _ = _read_canonical_object_from_content(content, "schema golden")
            if schema == CALIBRATION_BOOTSTRAP_REQUEST_SCHEMA:
                _validate_calibration_bootstrap_request_payload(payload)
                return
            if schema == CALIBRATION_BOOTSTRAP_EVIDENCE_SCHEMA:
                _validate_calibration_bootstrap_evidence_payload(payload)
                return
            if schema == CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA:
                if set(payload) != _MEASUREMENT_EVIDENCE_KEYS:
                    raise SealError("schema golden measurement evidence fields are invalid")
                for field in (
                    "base_system_package_evidence_sha256",
                    "calibration_bootstrap_evidence_sha256",
                    "checkpoint_acquisition_evidence_sha256",
                    "native_host_attestation_bundle_sha256",
                    "request_sha256",
                ):
                    _require_sha256_value(payload[field], f"schema golden {field}")
                for field in (
                    "runtime_image_config_digest",
                    "runtime_image_manifest_digest",
                ):
                    _require_oci_digest(payload[field], f"schema golden {field}")
                _native_host_from_record(
                    payload["native_host_evidence"],
                    "schema golden native host evidence",
                )
                rows = payload["measurement_rows"]
                if not isinstance(rows, list) or not rows:
                    raise SealError("schema golden measurement rows are invalid")
                for row in rows:
                    _measurement_row_from_value(cast(Mapping[str, object], row))
                return
            if schema == SEAL_CANDIDATE_SCHEMA:
                if set(payload) != _CANDIDATE_MANIFEST_KEYS:
                    raise SealError("schema golden candidate fields are invalid")
                for field in (
                    "backend_lock_payload_sha256",
                    "calibration_bootstrap_evidence_sha256",
                    "calibration_measurement_evidence_sha256",
                    "model_artifact_set_sha256",
                    "native_host_attestation_bundle_sha256",
                    "required_inference_inventory_sha256",
                    "runtime_lock_payload_sha256",
                    "seal_evidence_payload_sha256",
                    "seal_profile_request_sha256",
                ):
                    _require_sha256_value(payload[field], f"schema golden {field}")
                artifacts = payload["artifacts"]
                if not isinstance(artifacts, list):
                    raise SealError("schema golden candidate artifacts are invalid")
                for artifact in artifacts:
                    if not isinstance(artifact, dict) or set(artifact) != {
                        "path",
                        "role",
                        "sha256",
                    }:
                        raise SealError("schema golden candidate artifact is invalid")
                    if not _safe_relative_source_path(artifact["path"]):
                        raise SealError("schema golden candidate artifact path is invalid")
                    if not isinstance(artifact["role"], str) or not artifact["role"]:
                        raise SealError("schema golden candidate artifact role is invalid")
                    _require_sha256_value(
                        artifact["sha256"],
                        "schema golden candidate artifact hash",
                    )
                return
            if schema == "crux.oaf-oci-layout-manifest/v1":
                if set(payload) != _OCI_LAYOUT_KEYS:
                    raise SealError("schema golden OCI fields are invalid")
                archive = payload["archive"]
                if not isinstance(archive, dict) or set(archive) != _ARCHIVE_KEYS:
                    raise SealError("schema golden OCI archive is invalid")
                for field in (
                    "base_image_config_digest",
                    "config_digest",
                    "image_manifest_digest",
                    "index_digest",
                ):
                    _require_oci_digest(payload[field], f"schema golden OCI {field}")
                for field in (
                    "base_image_layer_diff_ids",
                    "base_image_layer_digests",
                    "layer_diff_ids",
                    "layer_digests",
                ):
                    _require_digest_array(payload[field], f"schema golden OCI {field}")
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                    directory_path = Path(directory)
                    archive_path = directory_path / CANDIDATE_ARTIFACT_PATHS["oci_layout_archive"]
                    archive_path.parent.mkdir(parents=True)
                    archive_path.write_bytes(b"schema-golden-oci\n")
                    seal = LoadedSealEvidence(
                        directory_path / "seal.json",
                        {
                            "base_image_config_digest": payload["base_image_config_digest"],
                            "base_image_layer_diff_ids": payload["base_image_layer_diff_ids"],
                            "base_image_layer_digests": payload["base_image_layer_digests"],
                            "oci_layout_archive": archive,
                            "runtime_image_config_digest": payload["config_digest"],
                            "runtime_image_index_digest": payload["index_digest"],
                            "runtime_image_layer_diff_ids": payload["layer_diff_ids"],
                            "runtime_image_layer_digests": payload["layer_digests"],
                            "runtime_image_manifest_digest": payload["image_manifest_digest"],
                        },
                        "0" * 64,
                    )
                    _validate_oci_layout(payload, directory_path, seal)
                return
            if schema == HOST_ADAPTER_SCHEMA:
                _validate_source_manifest_payload(payload, HOST_ADAPTER_SCHEMA)
                fixture_root = Path(__file__).parents[2] / "tests/benchmark/schema_goldens"
                for row in cast(list[dict[str, JsonValue]], payload["files"]):
                    path = fixture_root / cast(str, row["path"])
                    if sha256_hex(_read_regular(path, "schema golden source")) != row["sha256"]:
                        raise SealError("schema golden source hash differs")
                return
            raise SealError("schema golden is unsupported")
        loader = loaders[schema]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "golden.json"
            path.write_bytes(content)
            loaded = loader(path)
            if schema == SEAL_PROFILE_REQUEST_SCHEMA:
                for field in (
                    "base_system_package_evidence_sha256",
                    "base_system_package_request_sha256",
                    "calibration_measurement_evidence_sha256",
                    "calibration_measurement_request_sha256",
                    "checkpoint_acquisition_evidence_sha256",
                    "checkpoint_acquisition_request_sha256",
                ):
                    if loaded.payload[field] != "a" * 64:
                        raise SealError("schema golden profile authority differs")
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


def _validate_source_manifest_payload(payload: Mapping[str, JsonValue], schema: str) -> None:
    """Validate the source-manifest shape before its hashes bind selected source files."""
    if set(payload) != {"covered_roots", "files", "schema"} or payload["schema"] != schema:
        raise SealError("source manifest fields are invalid")
    roots = payload["covered_roots"]
    files = payload["files"]
    if (
        not isinstance(roots, list)
        or not roots
        or roots != sorted(roots)
        or len(set(roots)) != len(roots)
        or any(not _safe_relative_source_path(root) for root in roots)
        or not isinstance(files, list)
    ):
        raise SealError("source manifest roots are invalid")
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise SealError("source manifest file row is invalid")
        path = row["path"]
        if not _safe_relative_source_path(path):
            raise SealError("source manifest file path is invalid")
        if not _source_manifest_path_is_covered(path, roots):
            raise SealError("source manifest file is outside covered roots")
        _require_sha256_value(row["sha256"], "source manifest file hash")
        paths.append(path)
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise SealError("source manifest files are not uniquely sorted")


def _safe_relative_source_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _source_manifest_path_is_covered(path: str, roots: list[JsonValue]) -> bool:
    """Allow covered files, top-level support files, and package ancestors only."""
    if "/" not in path:
        return True
    if any(path == root or path.startswith(f"{root}/") for root in roots):
        return True
    parent = path.rsplit("/", 1)[0]
    return any(isinstance(root, str) and root.startswith(f"{parent}/") for root in roots)


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
        row.inference_call_count_before,
        row.inference_call_count_after,
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
        or row.inference_call_count_after < row.inference_call_count_before
        or not _is_sha256(row.input_audio_sha256)
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


def _row_payload(row: MeasurementRow) -> dict[str, JsonValue]:
    return {
        "exit_code": row.exit_code,
        "inference_call_count_after": row.inference_call_count_after,
        "inference_call_count_before": row.inference_call_count_before,
        "input_audio_sha256": row.input_audio_sha256,
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
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
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
    (
        bootstrap_request,
        bootstrap,
        bundle,
        _host,
        host_payload,
    ) = _authenticate_bootstrap_for_phase(
        bootstrap_request_path=Path(bootstrap_request_path),
        bootstrap_evidence_path=Path(bootstrap_evidence_path),
        host_attestation_bundle_path=Path(host_attestation_bundle_path),
        host_evidence_path=Path(host_evidence_path),
        phase="measurement",
    )
    if (
        request.payload["calibration_bootstrap_request_sha256"] != bootstrap_request.sha256
        or request.payload["calibration_bootstrap_evidence_sha256"] != bootstrap.sha256
    ):
        raise SealIntegrityError(
            "measurement request does not bind the accepted bootstrap authority"
        )
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
    native_runner = None
    if runner is None:
        from tools.hpa320.oaf_native_runner import NativeCalibrationRunner

        native_runner = NativeCalibrationRunner(
            repository_root=Path(bootstrap_request_path).parents[3],
            bootstrap_request=bootstrap_request,
            bootstrap=bootstrap,
            bootstrap_request_path=Path(bootstrap_request_path),
            bootstrap_evidence_path=Path(bootstrap_evidence_path),
            checkpoint_evidence_path=Path(checkpoint_evidence_path),
            base_system_evidence_path=Path(base_system_evidence_path),
            model_cache=Path(model_cache),
            max_input_audio_frames=max(request.frame_counts),
        )
        runner = native_runner.measure
    rows: list[MeasurementRow] = []
    fixtures = {
        cast(int, fixture["audio_frame_count"]): cast(dict[str, JsonValue], fixture)
        for fixture in cast(list[dict[str, JsonValue]], request.payload["fixtures"])
    }
    try:
        for frame_count in request.frame_counts:
            for repetition in range(1, request.repetition_count + 1):
                row = _measurement_row_from_value(runner(request, frame_count, repetition))
                if (
                    row.input_frame_count != frame_count
                    or row.repetition != repetition
                    or row.input_audio_sha256 != fixtures[frame_count]["input_audio_sha256"]
                    or row.exit_code != 0
                    or row.signal is not None
                    or row.oom_killed
                    or row.prediction_sha256 is None
                    or row.inference_call_count_after != row.inference_call_count_before + 1
                ):
                    raise SealError("measurement runner row does not match the requested probe")
                rows.append(row)
    finally:
        if native_runner is not None:
            native_runner.close()
    rows.sort(key=lambda row: (row.input_frame_count, row.process_instance_id, row.repetition))
    if len({row.process_instance_id for row in rows}) != len(rows):
        raise SealIntegrityError("measurement rows did not use one fresh process per repetition")
    payload: JsonValue = {
        "base_system_package_evidence_sha256": base.sha256,
        "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "measurement_rows": [_row_payload(row) for row in rows],
        "native_host_attestation_bundle_sha256": bundle.sha256,
        "native_host_evidence": host_payload,
        "request_sha256": request.sha256,
        "runtime_image_config_digest": bootstrap.payload["runtime_image_config_digest"],
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
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
    for field in (
        "base_system_package_evidence_sha256",
        "calibration_bootstrap_evidence_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "native_host_attestation_bundle_sha256",
    ):
        _require_sha256_value(payload[field], f"calibration measurement evidence {field}")
    for field in (
        "runtime_image_config_digest",
        "runtime_image_manifest_digest",
    ):
        _require_oci_digest(payload[field], f"calibration measurement evidence {field}")
    _native_host_from_record(
        payload["native_host_evidence"],
        "calibration measurement native host evidence",
    )
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
    fixtures = {
        cast(int, fixture["audio_frame_count"]): cast(dict[str, JsonValue], fixture)
        for fixture in cast(list[dict[str, JsonValue]], request.payload["fixtures"])
    }
    if len({row.process_instance_id for row in rows}) != len(rows) or any(
        row.input_audio_sha256 != fixtures[row.input_frame_count]["input_audio_sha256"]
        or row.exit_code != 0
        or row.signal is not None
        or row.oom_killed
        or row.prediction_sha256 is None
        or row.inference_call_count_after != row.inference_call_count_before + 1
        for row in rows
    ):
        raise SealIntegrityError(
            "calibration measurement evidence contains an unhealthy or unrelated row"
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


def _publish_candidate_directory(source: Path, output: Path) -> PublishedArtifact:
    """No-replace publish all validated candidate bytes as one directory identity."""

    _require_absent(output, "calibration candidate")
    parent = _require_directory(output.parent, "calibration candidate parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=parent))
    try:
        entries = sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix())
        files = [entry for entry in entries if entry.is_file() and not entry.is_symlink()]
        if not files or len(files) != sum(1 for entry in entries if not entry.is_dir()):
            raise SealIntegrityError("calibration candidate authority is empty")
        for entry in entries:
            relative = entry.relative_to(source)
            target = staging / relative
            if entry.is_dir() and not entry.is_symlink():
                target.mkdir()
                continue
            if not entry.is_file() or entry.is_symlink():
                raise SealIntegrityError(
                    "calibration candidate authority contains a non-regular entry"
                )
            content = read_regular_file_no_follow(entry)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
        for directory in sorted(
            (entry for entry in entries if entry.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(staging / directory.relative_to(source))
        _fsync_directory(staging)
        os.rename(staging, output)
    except (OSError, SealError):
        _remove_staging_directory(staging)
        raise SealIntegrityError("calibration candidate publication failed") from None
    manifest = read_regular_file_no_follow(output / CANDIDATE_MANIFEST_NAME)
    return PublishedArtifact(role="seal_candidate", path=output, sha256=sha256_hex(manifest))


def _copy_candidate_host_attestation(
    *,
    source_bundle: Path,
    bundle: NativeHostAttestationBundle,
    destination: Path,
) -> None:
    target_root = destination / CANDIDATE_HOST_ATTESTATION_ROOT
    target_root.mkdir(parents=True)
    sources = (
        Path(source_bundle),
        Path(source_bundle).parent / bundle.native_host_evidence.name,
        Path(source_bundle).parent / bundle.native_host_observation.name,
    )
    targets = (
        target_root / "attestation-bundle.json",
        target_root / bundle.native_host_evidence.name,
        target_root / bundle.native_host_observation.name,
    )
    for source, target in zip(sources, targets, strict=True):
        content = _read_regular(source, "candidate native-host attestation file")
        target.write_bytes(content)


def _remove_staging_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        return
    for root, directories, files in os.walk(path, topdown=False, followlinks=False):
        root_path = Path(root)
        for name in files:
            (root_path / name).unlink(missing_ok=True)
        for name in directories:
            child = root_path / name
            if child.is_symlink():
                child.unlink(missing_ok=True)
            else:
                child.rmdir()
    path.rmdir()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def calibrate(
    *,
    request_path: Path,
    measurement_evidence_path: Path,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence: Path,
    model_cache: Path,
    checkpoint_evidence_path: Path,
    base_system_evidence_path: Path,
    output: Path,
    runner: Callable[[int, bool, int], CalibrationProbeResult | Mapping[str, object]] | None = None,
    candidate_builder: Callable[[Path], None] | None = None,
    repository_root: Path | None = None,
) -> PublishedSealCandidate:
    """Rerun a reviewed profile with no hidden headroom or process assumptions."""

    (
        bootstrap_request,
        bootstrap,
        bundle,
        _host,
        host_payload,
    ) = _authenticate_bootstrap_for_phase(
        bootstrap_request_path=Path(bootstrap_request_path),
        bootstrap_evidence_path=Path(bootstrap_evidence_path),
        host_attestation_bundle_path=Path(host_attestation_bundle_path),
        host_evidence_path=Path(host_evidence),
        phase="candidate",
    )
    _require_directory(Path(model_cache), "model cache")
    try:
        profile = load_seal_profile_request(Path(request_path))
        measurement_request_path = Path(request_path).with_name(
            Path(request_path).name.replace(
                "seal-profile-request", "calibration-measurement-request"
            )
        )
        measurement_request = load_calibration_measurement_request(measurement_request_path)
        evidence, rows, measurement_sha = _load_measurement_evidence(
            Path(measurement_evidence_path), measurement_request
        )
        checkpoint_request = load_checkpoint_acquisition_request(_CHECKPOINT_REQUEST_PATH)
        checkpoint = load_checkpoint_acquisition_evidence(
            Path(checkpoint_evidence_path), request=checkpoint_request
        )
        base_request = load_base_system_package_request(_BASE_SYSTEM_REQUEST_PATH)
        base = load_base_system_package_evidence(
            Path(base_system_evidence_path), request=base_request
        )
    except (CheckpointAcquisitionError, SystemPackageError, SealError) as error:
        raise SealIntegrityError(f"calibration authority is invalid: {error}") from None
    if (
        evidence["checkpoint_acquisition_evidence_sha256"] != checkpoint.sha256
        or evidence["base_system_package_evidence_sha256"] != base.sha256
        or evidence["calibration_bootstrap_evidence_sha256"] != bootstrap.sha256
        or evidence["runtime_image_config_digest"]
        != bootstrap.payload["runtime_image_config_digest"]
        or evidence["runtime_image_manifest_digest"]
        != bootstrap.payload["runtime_image_manifest_digest"]
    ):
        raise SealIntegrityError("measurement evidence does not match its authenticated inputs")
    expected = {
        "calibration_bootstrap_request_sha256": bootstrap_request.sha256,
        "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
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
    in_bound = tuple(row for row in rows if row.input_frame_count <= profile.max_input_audio_frames)
    if not in_bound or any(
        row.exit_code != 0 or row.signal is not None or row.oom_killed for row in in_bound
    ):
        raise SealIntegrityError("in-bound measurement evidence is not healthy")
    _require_headroom(profile, in_bound)
    output_path = Path(output)
    _require_absent(output_path, "calibration candidate")
    parent = _require_directory(output_path.parent, "calibration candidate parent")
    repository = (
        Path(bootstrap_request_path).resolve().parents[3]
        if repository_root is None
        else Path(repository_root)
    )
    native_runner = None
    native_evidence_root: Path | None = None
    if runner is None:
        from tools.hpa320.oaf_native_runner import NativeCalibrationRunner

        native_evidence_root = Path(
            tempfile.mkdtemp(prefix=".hpa320-candidate-evidence-", dir=parent)
        )
        native_evidence_root.chmod(0o733)
        native_runner = NativeCalibrationRunner(
            repository_root=repository,
            bootstrap_request=bootstrap_request,
            bootstrap=bootstrap,
            bootstrap_request_path=Path(bootstrap_request_path),
            bootstrap_evidence_path=Path(bootstrap_evidence_path),
            checkpoint_evidence_path=Path(checkpoint_evidence_path),
            base_system_evidence_path=Path(base_system_evidence_path),
            model_cache=Path(model_cache),
            max_input_audio_frames=profile.max_input_audio_frames,
            candidate_evidence_root=native_evidence_root,
        )
        runner = native_runner.probe
    staging: Path | None = None
    try:
        probes = (
            (profile.max_input_audio_frames - 1, True, 1),
            (profile.max_input_audio_frames, True, 2),
            (profile.max_input_audio_frames + 1, True, 3),
            (profile.max_input_audio_frames, False, 1),
        )
        persistent_process_ids: list[str] = []
        fresh_process_id: str | None = None
        boundary_probes: list[JsonValue] = []
        for frame_count, persistent, ordinal in probes:
            result = _probe_result_from_value(runner(frame_count, persistent, ordinal))
            row = result.row
            if row.input_frame_count != frame_count:
                raise SealIntegrityError(
                    "calibration probe row does not match the requested frame count"
                )
            if persistent:
                persistent_process_ids.append(row.process_instance_id)
            else:
                fresh_process_id = row.process_instance_id
            if frame_count > profile.max_input_audio_frames:
                if (
                    not result.rejected_before_inference
                    or row.exit_code != 0
                    or row.signal is not None
                    or row.oom_killed
                    or row.prediction_sha256 is not None
                    or row.inference_call_count_after != row.inference_call_count_before
                ):
                    raise SealError("over-bound input was not rejected before inference")
            elif (
                result.rejected_before_inference
                or row.exit_code != 0
                or row.signal is not None
                or row.oom_killed
                or row.prediction_sha256 is None
                or row.inference_call_count_after != row.inference_call_count_before + 1
            ):
                raise SealError("in-bound calibration probe failed")
            boundary_probes.append(
                {
                    "inference_call_count_after": row.inference_call_count_after,
                    "inference_call_count_before": row.inference_call_count_before,
                    "persistent": persistent,
                    "rejected_before_inference": result.rejected_before_inference,
                    "request_ordinal": ordinal,
                    "row": _row_payload(row),
                }
            )
        if (
            len(persistent_process_ids) != 3
            or not persistent_process_ids[0]
            or len(set(persistent_process_ids)) != 1
            or not fresh_process_id
            or fresh_process_id == persistent_process_ids[0]
        ):
            raise SealIntegrityError(
                "calibration probes do not prove persistent and fresh process identity"
            )
        smoke_native_events: Sequence[Mapping[str, Any]] = ()
        if native_runner is not None:
            smoke_native_events = native_runner.smoke()
            native_runner.close()
        staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.candidate-", dir=parent))
        _copy_candidate_host_attestation(
            source_bundle=Path(host_attestation_bundle_path),
            bundle=bundle,
            destination=staging,
        )
        if candidate_builder is not None:
            candidate_builder(staging)
        elif native_runner is not None and native_evidence_root is not None:
            from tools.hpa320.oaf_candidate_builder import build_native_candidate

            build_native_candidate(
                staging=staging,
                repository_root=repository,
                tensor_coverage_path=native_evidence_root / "tensor-coverage.json",
                native_events=smoke_native_events,
                bootstrap_request=bootstrap_request,
                bootstrap=bootstrap,
                bootstrap_evidence_path=Path(bootstrap_evidence_path),
                measurement_request=measurement_request,
                measurement_sha256=measurement_sha,
                measurement_rows=rows,
                profile=profile,
                checkpoint_request=checkpoint_request,
                checkpoint=checkpoint,
                model_cache=Path(model_cache),
                base_request=base_request,
                base=base,
                bundle=bundle,
                host_payload=host_payload,
                boundary_probes=boundary_probes,
            )
        else:
            raise SealError("calibration candidate generator is unavailable; no output was written")
        loaded = _load_candidate(
            candidate=staging,
            repository_root=repository,
        )
        _validate_generated_candidate_authority(
            loaded=loaded,
            bootstrap_request=bootstrap_request,
            bootstrap=bootstrap,
            measurement_request=measurement_request,
            measurement_sha256=measurement_sha,
            measurement_rows=rows,
            profile=profile,
            checkpoint_request=checkpoint_request,
            checkpoint=checkpoint,
            base_request=base_request,
            base=base,
            bundle=bundle,
            host_payload=host_payload,
            boundary_probes=boundary_probes,
        )
        artifact = _publish_candidate_directory(staging, output_path)
        return PublishedSealCandidate(artifact)
    finally:
        if native_runner is not None:
            native_runner.close()
        if staging is not None:
            _remove_staging_directory(staging)
        if native_evidence_root is not None:
            _remove_staging_directory(native_evidence_root)


# The explicit arguments make every accepted upstream authority visible at this gate.
# pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements
def _validate_generated_candidate_authority(
    *,
    loaded: _Candidate,
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    measurement_request: CalibrationMeasurementRequest,
    measurement_sha256: str,
    measurement_rows: Sequence[MeasurementRow],
    profile: SealProfileRequest,
    checkpoint_request: CheckpointAcquisitionRequest,
    checkpoint: CheckpointAcquisitionEvidence,
    base_request: BaseSystemPackageRequest,
    base: BaseSystemPackageEvidence,
    bundle: NativeHostAttestationBundle,
    host_payload: Mapping[str, JsonValue],
    boundary_probes: Sequence[JsonValue],
) -> None:
    backend = loaded.backend.payload
    runtime = loaded.runtime.payload
    seal = loaded.seal.payload
    checkpoint_components = [
        {
            "name": member.name,
            "sha256": member.sha256,
            "size": member.size,
        }
        for member in checkpoint_request.archive_members
        if member.role == "published_component"
    ]
    checkpoint_archive = {
        "name": checkpoint_request.archive.name,
        "sha256": checkpoint_request.archive.sha256,
        "size": checkpoint_request.archive.size,
    }
    package_inventory = [
        {
            "architecture": package.architecture,
            "name": package.name,
            "version": package.version,
        }
        for package in base.package_inventory
    ]
    expected_backend = {
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": checkpoint.request_sha256,
        "checkpoint_archive": checkpoint_archive,
        "checkpoint_components": checkpoint_components,
        "max_input_audio_frames": profile.max_input_audio_frames,
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
    }
    expected_runtime = {
        "additional_system_packages": [],
        "base_image": base_request.base_image,
        "base_image_archive_keyring_sha256": (base_request.base_image_archive_keyring_sha256),
        "base_image_config_digest": bootstrap.payload["base_image_config_digest"],
        "base_image_layer_diff_ids": bootstrap.payload["base_image_layer_diff_ids"],
        "base_image_layer_digests": bootstrap.payload["base_image_layer_digests"],
        "base_image_manifest_digest": base_request.base_image_manifest_digest,
        "base_system_package_evidence_sha256": base.sha256,
        "base_system_package_inventory": package_inventory,
        "base_system_package_inventory_sha256": base.package_inventory_sha256,
        "base_system_package_request_sha256": base.request_sha256,
        "build_context_manifest_sha256": bootstrap_request.build_context_manifest_sha256,
        "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
        "calibration_bootstrap_request_sha256": bootstrap_request.sha256,
        "image_build": bootstrap.payload["image_build"],
        "platform": base_request.platform,
        "runtime_image_config_digest": bootstrap.payload["runtime_image_config_digest"],
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
    }
    for field in (
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
    ):
        expected_runtime[field] = profile.payload[field]
    expected_seal = {
        **expected_backend,
        **{
            field: expected_runtime[field]
            for field in (
                "additional_system_packages",
                "base_image_archive_keyring_sha256",
                "base_image_config_digest",
                "base_image_layer_diff_ids",
                "base_image_layer_digests",
                "base_image_manifest_digest",
                "base_system_package_evidence_sha256",
                "base_system_package_inventory",
                "base_system_package_inventory_sha256",
                "base_system_package_request_sha256",
                "build_context_manifest_sha256",
                "calibration_bootstrap_evidence_sha256",
                "calibration_bootstrap_request_sha256",
                "runtime_image_config_digest",
                "runtime_image_manifest_digest",
            )
        },
        "boundary_probes": boundary_probes,
        "calibration_measurement_evidence_sha256": measurement_sha256,
        "calibration_measurement_request_sha256": measurement_request.sha256,
        "measurements": [_row_payload(row) for row in measurement_rows],
        "native_host_attestation_bundle_sha256": bundle.sha256,
        "native_host_evidence": host_payload,
        "oci_layout_archive": bootstrap.payload["oci_layout_archive"],
        "oci_layout_manifest_sha256": bootstrap.payload["oci_layout_manifest_sha256"],
        "runtime_gid": bootstrap_request.runtime_gid,
        "runtime_image_index_digest": bootstrap.payload["runtime_image_index_digest"],
        "runtime_image_layer_diff_ids": bootstrap.payload["runtime_image_layer_diff_ids"],
        "runtime_image_layer_digests": bootstrap.payload["runtime_image_layer_digests"],
        "runtime_uid": bootstrap_request.runtime_uid,
        "seal_profile_request_sha256": profile.sha256,
    }
    for field in (
        "cpu_limit_millis",
        "memory_limit_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "shm_bytes",
        "startup_deadline_seconds",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
        "tmp_bytes",
    ):
        expected_seal[field] = profile.payload[field]
    for payload, expected in (
        (backend, expected_backend),
        (runtime, expected_runtime),
        (seal, expected_seal),
    ):
        if any(
            _plain_json(payload.get(field)) != _plain_json(value)
            for field, value in expected.items()
        ):
            raise SealIntegrityError(
                "generated calibration candidate is unrelated to accepted authority"
            )
    if (
        sha256_hex(canonical_json_bytes(_plain_json(backend["checkpoint_components"])))
        != checkpoint.model_artifact_set_sha256
    ):
        raise SealIntegrityError("generated calibration candidate checkpoint identity is unrelated")


def seal_candidate(
    *,
    candidate: Path,
    repository_root: Path,
) -> PublishedSeal:
    """Publish only the candidate's exact authenticated repository-relative allowlist."""

    repository = _require_directory(Path(repository_root), "repository root")
    loaded = _load_candidate(
        candidate=Path(candidate),
        repository_root=repository,
    )
    order = tuple(role for role, _path in CANDIDATE_ARTIFACTS)
    published: list[PublishedArtifact] = []
    try:
        for role, relative in CANDIDATE_ARTIFACTS:
            content = loaded.contents[role]
            published.append(
                publish_immutable_bytes(
                    repository / relative,
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
    repository_root: Path,
) -> _Candidate:
    directory = _require_directory(candidate, "candidate directory")
    repository = _require_directory(repository_root, "repository root")
    manifest, contents = _load_candidate_manifest(directory)
    host = load_native_host_evidence(directory / CANDIDATE_ARTIFACT_PATHS["native_host_evidence"])
    backend = _load_backend(directory / CANDIDATE_ARTIFACT_PATHS["backend_lock"])
    runtime = _load_runtime(directory / CANDIDATE_ARTIFACT_PATHS["runtime_lock"])
    seal = _load_seal(directory / CANDIDATE_ARTIFACT_PATHS["seal_evidence"])
    audit = _load_audit(directory / CANDIDATE_ARTIFACT_PATHS["conversion_audit"])
    try:
        validate_oaf_lock_set(backend, runtime, seal, audit)
    except BackendLockError as error:
        raise SealError(f"candidate lock set is invalid: {error}") from None

    bundle_path = directory / CANDIDATE_ARTIFACT_PATHS["native_host_attestation_bundle"]
    try:
        bundle = load_native_host_attestation_bundle(
            bundle_path,
            expected_phase="candidate",
        )
    except HostAttestationError as error:
        raise SealError(f"candidate native-host bundle is invalid: {error}") from None
    _validate_candidate_manifest(
        manifest,
        backend=backend,
        runtime=runtime,
        seal=seal,
        bundle=bundle,
    )
    _validate_host_binding(host, seal)
    host_manifest = build_host_adapter_source_manifest(repository)
    host_manifest_content = canonical_json_bytes(host_manifest, trailing_newline=True)
    expected_host_hash = cast(str, seal.payload["host_adapter_source_manifest_sha256"])
    if sha256_hex(host_manifest_content) != expected_host_hash:
        raise SealError("host adapter source manifest hash does not match the candidate")
    if contents["host_adapter_source_manifest"] != host_manifest_content:
        raise SealError("candidate host adapter source manifest bytes differ")

    _validate_repository_inputs(repository, seal, runtime)
    _validate_candidate_artifacts(directory, seal)
    return _Candidate(
        backend=backend,
        runtime=runtime,
        seal=seal,
        audit=audit,
        host=host,
        contents=contents,
        host_manifest_content=host_manifest_content,
    )


def _load_candidate_manifest(
    directory: Path,
) -> tuple[dict[str, JsonValue], dict[str, bytes]]:
    payload, _ = _read_canonical_object(
        directory / CANDIDATE_MANIFEST_NAME,
        "candidate manifest",
    )
    if set(payload) != _CANDIDATE_MANIFEST_KEYS or payload["schema"] != CANDIDATE_MANIFEST_SCHEMA:
        raise SealError("candidate manifest fields must match the exact schema")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(CANDIDATE_ARTIFACTS):
        raise SealError("candidate artifact allowlist is incomplete")
    contents: dict[str, bytes] = {}
    for row, (expected_role, expected_path) in zip(
        artifacts,
        CANDIDATE_ARTIFACTS,
        strict=True,
    ):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "role", "sha256"}
            or row["role"] != expected_role
            or row["path"] != expected_path
        ):
            raise SealError("candidate artifact allowlist order or path is invalid")
        _require_sha256_value(row["sha256"], "candidate artifact hash")
        content = _read_regular(directory / expected_path, f"candidate {expected_role}")
        if sha256_hex(content) != row["sha256"]:
            label = "OCI archive" if expected_role == "oci_layout_archive" else expected_role
            raise SealError(f"candidate {label} hash does not match its allowlist")
        contents[expected_role] = content
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = {CANDIDATE_MANIFEST_NAME}
    expected_files.update(path for _role, path in CANDIDATE_ARTIFACTS)
    if actual_files != expected_files:
        raise SealError("candidate directory contains files outside its exact allowlist")
    return payload, contents


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


def _validate_candidate_manifest(
    payload: Mapping[str, JsonValue],
    *,
    backend: LoadedBackendLock,
    runtime: LoadedRuntimeLock,
    seal: LoadedSealEvidence,
    bundle: NativeHostAttestationBundle,
) -> None:
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
        or payload["backend_lock_payload_sha256"] != backend.sha256
        or payload["runtime_lock_payload_sha256"] != runtime.sha256
        or payload["seal_evidence_payload_sha256"] != seal.sha256
        or payload["calibration_bootstrap_evidence_sha256"]
        != seal.payload["calibration_bootstrap_evidence_sha256"]
        or payload["calibration_measurement_evidence_sha256"]
        != seal.payload["calibration_measurement_evidence_sha256"]
        or payload["seal_profile_request_sha256"] != seal.payload["seal_profile_request_sha256"]
        or payload["native_host_attestation_bundle_sha256"] != bundle.sha256
        or payload["native_host_attestation_bundle_sha256"]
        != seal.payload["native_host_attestation_bundle_sha256"]
    ):
        raise SealError("candidate manifest identity does not match the proposed backend lock")
    for field in (
        "backend_lock_payload_sha256",
        "calibration_bootstrap_evidence_sha256",
        "calibration_measurement_evidence_sha256",
        "model_artifact_set_sha256",
        "native_host_attestation_bundle_sha256",
        "required_inference_inventory_sha256",
        "runtime_lock_payload_sha256",
        "seal_evidence_payload_sha256",
        "seal_profile_request_sha256",
    ):
        _require_sha256_value(payload[field], f"candidate manifest {field}")


def _validate_host_binding(
    host: NativeHostEvidence,
    seal: LoadedSealEvidence,
) -> None:
    record = seal.payload["native_host_evidence"]
    if (
        not isinstance(record, Mapping)
        or set(record) != _HOST_EVIDENCE_KEYS
        or record["kind"] != host.kind
        or record["official_execution_allowed"] != host.official_execution_allowed
        or _plain_json(record["payload"]) != _plain_json(host.payload)
        or record["sha256"] != host.sha256
        or host.official_execution_allowed is not True
    ):
        raise SealError("native host evidence does not match the proposed seal")
    if (
        seal.payload["reference_host_numeric_fingerprint"]
        != host.host_numeric_fingerprint.as_json()
    ):
        raise SealError("reference host numeric fingerprint does not match native host evidence")


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
) -> None:
    tensor, tensor_content = _read_canonical_object(
        directory / CANDIDATE_ARTIFACT_PATHS["tensor_coverage"],
        "tensor coverage",
    )
    _validate_tensor_coverage(tensor, seal)
    security, security_content = _read_canonical_object(
        directory / CANDIDATE_ARTIFACT_PATHS["security_scan"],
        "security scan",
    )
    advisory_content = _read_regular(
        directory / CANDIDATE_ARTIFACT_PATHS["advisory_snapshot"],
        "advisory snapshot",
    )
    _validate_security_scan(security, advisory_content, seal)
    oci, oci_content = _read_canonical_object(
        directory / CANDIDATE_ARTIFACT_PATHS["oci_layout_manifest"],
        "OCI layout manifest",
    )
    _validate_oci_layout(oci, directory, seal)
    oracle, oracle_content = _read_canonical_object(
        directory / CANDIDATE_ARTIFACT_PATHS["smoke_oracle"],
        "smoke oracle",
    )
    _validate_smoke(oracle, directory, seal)

    expected_hashes = {
        "tensor_coverage": seal.payload["tensor_coverage_sha256"],
        "security_scan": seal.payload["security_scan_sha256"],
        "oci_layout_manifest": seal.payload["oci_layout_manifest_sha256"],
        "smoke_oracle": seal.payload["smoke_oracle_sha256"],
    }
    contents = {
        "tensor_coverage": tensor_content,
        "security_scan": security_content,
        "oci_layout_manifest": oci_content,
        "smoke_oracle": oracle_content,
    }
    for name, content in contents.items():
        if sha256_hex(content) != expected_hashes[name]:
            raise SealError(f"{name} hash does not match the proposed seal")


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
        or payload["index_digest"] != seal.payload["runtime_image_index_digest"]
        or payload["image_manifest_digest"] != seal.payload["runtime_image_manifest_digest"]
        or payload["layer_digests"] != _plain_json(seal.payload["runtime_image_layer_digests"])
        or payload["layer_diff_ids"] != _plain_json(seal.payload["runtime_image_layer_diff_ids"])
        or payload["base_image_config_digest"] != seal.payload["base_image_config_digest"]
        or payload["base_image_layer_digests"]
        != _plain_json(seal.payload["base_image_layer_digests"])
        or payload["base_image_layer_diff_ids"]
        != _plain_json(seal.payload["base_image_layer_diff_ids"])
        or not isinstance(payload["schema"], str)
        or not payload["schema"]
    ):
        raise SealError("OCI layout image identities do not match the proposed seal")
    base_layers = cast(list[JsonValue], payload["base_image_layer_digests"])
    base_diff_ids = cast(list[JsonValue], payload["base_image_layer_diff_ids"])
    layers = cast(list[JsonValue], payload["layer_digests"])
    diff_ids = cast(list[JsonValue], payload["layer_diff_ids"])
    if (
        not base_layers
        or len(base_layers) != len(base_diff_ids)
        or len(layers) != len(diff_ids)
        or layers[: len(base_layers)] != base_layers
        or diff_ids[: len(base_diff_ids)] != base_diff_ids
    ):
        raise SealError("OCI layout does not preserve the exact base layer order")
    name = archive["name"]
    if not isinstance(name, str) or not name or PurePosixPath(name).name != name or "\\" in name:
        raise SealError("OCI archive filename is invalid")
    expected_archive_path = CANDIDATE_ARTIFACT_PATHS["oci_layout_archive"]
    if PurePosixPath(expected_archive_path).name != name:
        raise SealError("OCI archive filename differs from the candidate allowlist")
    content = _read_regular(directory / expected_archive_path, "OCI archive")
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
    audio = _read_regular(
        directory / CANDIDATE_ARTIFACT_PATHS["smoke_audio"],
        "smoke audio",
    )
    prediction = _read_regular(
        directory / CANDIDATE_ARTIFACT_PATHS["smoke_prediction"],
        "smoke prediction",
    )
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


def _atomic_replace_regular_file(path: Path, content: bytes) -> None:
    target = Path(path)
    parent = _require_directory(target.parent, "replacement parent")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.reissue-",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(parent)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise SealError("calibration bootstrap request replacement failed") from None


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


def bootstrap_image(
    *,
    request_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    output: Path,
    repository_root: Path,
) -> PublishedArtifact:
    """Run the native-only deterministic image producer without a module import cycle."""

    from tools.hpa320.oaf_native_calibration import (
        bootstrap_image as native_bootstrap_image,
    )

    return native_bootstrap_image(
        request_path=request_path,
        host_attestation_bundle_path=host_attestation_bundle_path,
        host_evidence_path=host_evidence_path,
        output=output,
        repository_root=repository_root,
    )


def validate_native_work_phase(
    *,
    phase: str,
    payload_root: Path,
    repository_root: Path,
) -> None:
    """Apply the selected phase's existing request/evidence/content validation."""

    try:
        validator = _PHASE_VALIDATORS[phase]
    except KeyError as error:
        raise SealError("native work phase is invalid") from error
    try:
        validator(payload_root=Path(payload_root), repository_root=Path(repository_root))
    except SealError:
        raise
    except (CheckpointAcquisitionError, HostAttestationError, OSError, SystemPackageError) as error:
        raise SealError(f"{phase} native work payload is invalid: {error}") from None


def _validate_bootstrap_work_payload(*, payload_root: Path, repository_root: Path) -> None:
    _validate_bootstrap_authority(
        payload_root=payload_root,
        repository_root=repository_root,
        phase="bootstrap",
        checkpoint_evidence_name="checkpoint-acquisition-evidence.json",
        image_directory_name="calibration-image",
        host_directory_name="bootstrap-host-attestation",
    )


def _validate_measurement_work_payload(*, payload_root: Path, repository_root: Path) -> None:
    root = _require_directory(payload_root, "native work payload root")
    repository = _require_directory(repository_root, "repository root")
    accepted = _load_accepted_native_authorities(repository)
    operational = _load_operational_authority(
        payload_root=root,
        repository_root=repository,
        phase="measurement",
        host_directory_name="measurement-host-attestation",
        accepted_bootstrap=accepted.bootstrap,
        checkpoint_request=accepted.checkpoint_request,
    )
    if operational.checkpoint.sha256 != accepted.checkpoint.sha256:
        raise SealError(
            "operational checkpoint does not reproduce the accepted checkpoint authority"
        )
    _validate_measurement_authority(
        payload_root=root,
        repository_root=repository,
        accepted=accepted,
        operational=operational,
    )


def _validate_candidate_work_payload(*, payload_root: Path, repository_root: Path) -> None:
    root = _require_directory(payload_root, "native work payload root")
    repository = _require_directory(repository_root, "repository root")
    accepted = _load_accepted_native_authorities(repository)
    operational = _load_operational_authority(
        payload_root=root,
        repository_root=repository,
        phase="candidate",
        host_directory_name="candidate-host-attestation",
        accepted_bootstrap=accepted.bootstrap,
        checkpoint_request=accepted.checkpoint_request,
    )
    if operational.checkpoint.sha256 != accepted.checkpoint.sha256:
        raise SealError(
            "operational checkpoint does not reproduce the accepted checkpoint authority"
        )
    measurement_request = load_calibration_measurement_request(
        repository / _CALIBRATION_MEASUREMENT_REQUEST_PATH
    )
    _validate_measurement_request_binding(measurement_request, accepted)
    measurement, rows, measurement_sha256 = _load_measurement_evidence(
        repository / _ACCEPTED_NATIVE_EVIDENCE_ROOT / "calibration-measurement-evidence.json",
        measurement_request,
    )
    _validate_measurement_evidence_authority(
        measurement=measurement,
        accepted=accepted,
        bundle=accepted.measurement_bundle,
        host=accepted.measurement_host,
        host_payload=accepted.measurement_host_payload,
        operational_bootstrap=accepted.bootstrap,
    )
    profile = load_seal_profile_request(repository / _SEAL_PROFILE_REQUEST_PATH)
    _validate_seal_profile_authority(
        profile=profile,
        accepted=accepted,
        measurement_request=measurement_request,
        measurement_sha256=measurement_sha256,
    )
    candidate = _load_candidate(candidate=root / "seal-candidate", repository_root=repository)
    _require_matching_candidate_host_files(root)
    _validate_candidate_authority(
        candidate=candidate,
        accepted=accepted,
        operational=operational,
        measurement_request=measurement_request,
        measurement_sha256=measurement_sha256,
        measurement_rows=rows,
        profile=profile,
    )


@dataclass(frozen=True)
class _AcceptedNativeAuthorities:
    base: BaseSystemPackageEvidence
    base_request: BaseSystemPackageRequest
    bootstrap: CalibrationBootstrapEvidence
    bootstrap_request: CalibrationBootstrapRequest
    checkpoint: CheckpointAcquisitionEvidence
    checkpoint_request: CheckpointAcquisitionRequest
    measurement_bundle: NativeHostAttestationBundle
    measurement_host: NativeHostEvidence
    measurement_host_payload: dict[str, JsonValue]


@dataclass(frozen=True)
class _OperationalNativeAuthority:
    bootstrap: CalibrationBootstrapEvidence
    bundle: NativeHostAttestationBundle
    checkpoint: CheckpointAcquisitionEvidence
    host: NativeHostEvidence
    host_payload: dict[str, JsonValue]


def _load_accepted_native_authorities(repository: Path) -> _AcceptedNativeAuthorities:
    evidence_root = repository / _ACCEPTED_NATIVE_EVIDENCE_ROOT
    bootstrap_request, bootstrap = load_calibration_bootstrap_evidence(
        repository / _CALIBRATION_BOOTSTRAP_REQUEST_PATH,
        evidence_root / "calibration-bootstrap-evidence.json",
    )
    checkpoint_request = load_checkpoint_acquisition_request(repository / _CHECKPOINT_REQUEST_PATH)
    checkpoint = load_checkpoint_acquisition_evidence(
        evidence_root / "checkpoint-acquisition-evidence.json",
        request=checkpoint_request,
    )
    base_request = load_base_system_package_request(repository / _BASE_SYSTEM_REQUEST_PATH)
    base = load_base_system_package_evidence(
        evidence_root / "base-system-package-evidence.json",
        request=base_request,
    )
    if (
        bootstrap_request.payload["checkpoint_acquisition_request_sha256"]
        != checkpoint_request.sha256
        or bootstrap_request.payload["base_system_package_request_sha256"] != base_request.sha256
    ):
        raise SealError("bootstrap request does not bind its accepted operational authorities")
    measurement_bundle, measurement_host, measurement_host_payload = (
        _load_accepted_measurement_host_authority(evidence_root)
    )
    return _AcceptedNativeAuthorities(
        base=base,
        base_request=base_request,
        bootstrap=bootstrap,
        bootstrap_request=bootstrap_request,
        checkpoint=checkpoint,
        checkpoint_request=checkpoint_request,
        measurement_bundle=measurement_bundle,
        measurement_host=measurement_host,
        measurement_host_payload=measurement_host_payload,
    )


def _load_accepted_measurement_host_authority(
    evidence_root: Path,
) -> tuple[NativeHostAttestationBundle, NativeHostEvidence, dict[str, JsonValue]]:
    host_root = evidence_root / "measurement-host-attestation"
    bundle_path = host_root / "attestation-bundle.json"
    host_path = host_root / "native-host-evidence.json"
    try:
        bundle = load_native_host_attestation_bundle(bundle_path, expected_phase="measurement")
    except HostAttestationError as error:
        raise SealError(f"accepted measurement native-host bundle is invalid: {error}") from None
    host_content = _read_regular(host_path, "accepted measurement native host evidence")
    bundled_host_content = _read_regular(
        host_root / bundle.native_host_evidence.name,
        "accepted measurement bundled native host evidence",
    )
    if (
        host_content != bundled_host_content
        or len(host_content) != bundle.native_host_evidence.size
        or sha256_hex(host_content) != bundle.native_host_evidence.sha256
    ):
        raise SealError("accepted measurement native host evidence does not match its bundle")
    host_payload = _parse_canonical_json_content(
        host_content,
        "accepted measurement native host evidence",
    )
    host = _native_host_from_record(host_payload, "accepted measurement native host evidence")
    return bundle, host, host_payload


def _load_operational_authority(
    *,
    payload_root: Path,
    repository_root: Path,
    phase: str,
    host_directory_name: str,
    accepted_bootstrap: CalibrationBootstrapEvidence,
    checkpoint_request: CheckpointAcquisitionRequest,
) -> _OperationalNativeAuthority:
    bootstrap_request, bootstrap = load_calibration_bootstrap_evidence(
        repository_root / _CALIBRATION_BOOTSTRAP_REQUEST_PATH,
        payload_root / "operational-image/calibration-bootstrap-evidence.json",
    )
    bundle, host, host_payload = _authenticate_phase_host(
        host_attestation_bundle_path=payload_root / host_directory_name / "attestation-bundle.json",
        host_evidence_path=payload_root / host_directory_name / "native-host-evidence.json",
        phase=phase,
    )
    if (
        bootstrap.payload["native_host_attestation_bundle_sha256"] != bundle.sha256
        or bootstrap.payload["native_host_evidence"] != host_payload
    ):
        raise SealError("operational bootstrap image does not match its phase host authority")
    _require_imported_runtime_image(cast(str, bootstrap.payload["runtime_image_config_digest"]))
    _validate_bootstrap_image_files(
        image_directory=payload_root / "operational-image",
        bootstrap=bootstrap,
    )
    _validate_operational_image_identity(
        accepted=accepted_bootstrap,
        operational=bootstrap,
    )
    checkpoint = load_checkpoint_acquisition_evidence(
        payload_root / "operational-checkpoint-acquisition-evidence.json",
        request=checkpoint_request,
    )
    if (
        bootstrap_request.sha256
        != accepted_bootstrap.payload["calibration_bootstrap_request_sha256"]
    ):
        raise SealError("operational bootstrap image does not reproduce the accepted request")
    return _OperationalNativeAuthority(
        bootstrap=bootstrap,
        bundle=bundle,
        checkpoint=checkpoint,
        host=host,
        host_payload=host_payload,
    )


def _validate_operational_image_identity(
    *,
    accepted: CalibrationBootstrapEvidence,
    operational: CalibrationBootstrapEvidence,
) -> None:
    if any(
        _plain_json(accepted.payload[field]) != _plain_json(operational.payload[field])
        for field in _OPERATIONAL_IMAGE_IDENTITY_FIELDS
    ):
        raise SealError("operational bootstrap image differs from the accepted image identity")


def _validate_measurement_request_binding(
    request: CalibrationMeasurementRequest,
    accepted: _AcceptedNativeAuthorities,
) -> None:
    if (
        request.payload["calibration_bootstrap_request_sha256"] != accepted.bootstrap_request.sha256
        or request.payload["calibration_bootstrap_evidence_sha256"] != accepted.bootstrap.sha256
    ):
        raise SealError("measurement request does not bind the accepted bootstrap authority")


def _validate_bootstrap_authority(
    *,
    payload_root: Path,
    repository_root: Path,
    phase: str,
    checkpoint_evidence_name: str,
    image_directory_name: str,
    host_directory_name: str,
) -> tuple[
    CalibrationBootstrapEvidence,
    CheckpointAcquisitionEvidence,
    BaseSystemPackageEvidence,
    NativeHostAttestationBundle,
    NativeHostEvidence,
    dict[str, JsonValue],
]:
    root = _require_directory(payload_root, "native work payload root")
    repository = _require_directory(repository_root, "repository root")
    bootstrap_request, bootstrap, bundle, host, host_payload = _authenticate_bootstrap_for_phase(
        bootstrap_request_path=repository / _CALIBRATION_BOOTSTRAP_REQUEST_PATH,
        bootstrap_evidence_path=root / image_directory_name / "calibration-bootstrap-evidence.json",
        host_attestation_bundle_path=(root / host_directory_name / "attestation-bundle.json"),
        host_evidence_path=root / host_directory_name / "native-host-evidence.json",
        phase=phase,
    )
    _validate_bootstrap_image_files(
        image_directory=root / image_directory_name,
        bootstrap=bootstrap,
    )
    checkpoint_request = load_checkpoint_acquisition_request(repository / _CHECKPOINT_REQUEST_PATH)
    checkpoint = load_checkpoint_acquisition_evidence(
        root / checkpoint_evidence_name,
        request=checkpoint_request,
    )
    base_request = load_base_system_package_request(repository / _BASE_SYSTEM_REQUEST_PATH)
    base_path = root / "base-system-package-evidence.json"
    if phase != "bootstrap":
        base_path = repository / "base-system-package-evidence.json"
    base = load_base_system_package_evidence(base_path, request=base_request)
    if (
        bootstrap_request.payload["checkpoint_acquisition_request_sha256"]
        != checkpoint_request.sha256
        or bootstrap_request.payload["base_system_package_request_sha256"] != base_request.sha256
    ):
        raise SealError("bootstrap request does not bind its operational authorities")
    bootstrap_host = _native_host_from_record(
        bootstrap.payload["native_host_evidence"],
        "bootstrap native host evidence",
    )
    if not _same_native_host(base.native_host_evidence, bootstrap_host):
        raise SealError("base-system evidence does not match bootstrap host authority")
    return bootstrap, checkpoint, base, bundle, host, host_payload


def _validate_measurement_authority(
    *,
    payload_root: Path,
    repository_root: Path,
    accepted: _AcceptedNativeAuthorities,
    operational: _OperationalNativeAuthority,
) -> None:
    repository = _require_directory(repository_root, "repository root")
    request = load_calibration_measurement_request(
        repository / _CALIBRATION_MEASUREMENT_REQUEST_PATH
    )
    _validate_measurement_request_binding(request, accepted)
    payload, _rows, _sha256 = _load_measurement_evidence(
        payload_root / "calibration-measurement-evidence.json",
        request,
    )
    _validate_measurement_evidence_authority(
        measurement=payload,
        accepted=accepted,
        bundle=operational.bundle,
        host=operational.host,
        host_payload=operational.host_payload,
        operational_bootstrap=operational.bootstrap,
    )


def _validate_measurement_evidence_authority(
    *,
    measurement: Mapping[str, JsonValue],
    accepted: _AcceptedNativeAuthorities,
    bundle: NativeHostAttestationBundle | None = None,
    host: NativeHostEvidence | None = None,
    host_payload: Mapping[str, JsonValue] | None = None,
    operational_bootstrap: CalibrationBootstrapEvidence | None = None,
) -> None:
    if (
        measurement["calibration_bootstrap_evidence_sha256"] != accepted.bootstrap.sha256
        or measurement["checkpoint_acquisition_evidence_sha256"] != accepted.checkpoint.sha256
        or measurement["base_system_package_evidence_sha256"] != accepted.base.sha256
    ):
        raise SealError("measurement evidence does not bind its accepted authorities")
    if operational_bootstrap is not None and (
        measurement["runtime_image_config_digest"]
        != operational_bootstrap.payload["runtime_image_config_digest"]
        or measurement["runtime_image_manifest_digest"]
        != operational_bootstrap.payload["runtime_image_manifest_digest"]
    ):
        raise SealError("measurement evidence does not bind its operational image identity")
    if bundle is not None and measurement["native_host_attestation_bundle_sha256"] != bundle.sha256:
        raise SealError("measurement evidence does not bind its phase host bundle")
    if host_payload is not None and measurement["native_host_evidence"] != host_payload:
        raise SealError("measurement evidence does not bind its phase host evidence")
    if host is not None and not _same_native_host(
        _native_host_from_record(measurement["native_host_evidence"], "measurement native host"),
        host,
    ):
        raise SealError("measurement evidence does not bind its phase host identity")


def _validate_seal_profile_authority(
    *,
    profile: SealProfileRequest,
    accepted: _AcceptedNativeAuthorities,
    measurement_request: CalibrationMeasurementRequest,
    measurement_sha256: str,
) -> None:
    expected = {
        "calibration_bootstrap_request_sha256": accepted.bootstrap_request.sha256,
        "calibration_bootstrap_evidence_sha256": accepted.bootstrap.sha256,
        "calibration_measurement_request_sha256": measurement_request.sha256,
        "calibration_measurement_evidence_sha256": measurement_sha256,
        "checkpoint_acquisition_evidence_sha256": accepted.checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": accepted.checkpoint.request_sha256,
        "base_system_package_evidence_sha256": accepted.base.sha256,
        "base_system_package_request_sha256": accepted.base.request_sha256,
    }
    if any(profile.payload[field] != value for field, value in expected.items()):
        raise SealError("seal profile request is unrelated to the accepted authority")


def _validate_candidate_authority(
    *,
    candidate: _Candidate,
    accepted: _AcceptedNativeAuthorities,
    operational: _OperationalNativeAuthority,
    measurement_request: CalibrationMeasurementRequest,
    measurement_sha256: str,
    measurement_rows: Sequence[MeasurementRow],
    profile: SealProfileRequest,
) -> None:
    checkpoint_components = [
        {
            "name": member.name,
            "sha256": member.sha256,
            "size": member.size,
        }
        for member in accepted.checkpoint_request.archive_members
        if member.role == "published_component"
    ]
    checkpoint_archive = {
        "name": accepted.checkpoint_request.archive.name,
        "sha256": accepted.checkpoint_request.archive.sha256,
        "size": accepted.checkpoint_request.archive.size,
    }
    package_inventory = [
        {
            "architecture": package.architecture,
            "name": package.name,
            "version": package.version,
        }
        for package in accepted.base.package_inventory
    ]
    expected_backend = {
        "checkpoint_acquisition_evidence_sha256": operational.checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": operational.checkpoint.request_sha256,
        "checkpoint_archive": checkpoint_archive,
        "checkpoint_components": checkpoint_components,
        "max_input_audio_frames": profile.max_input_audio_frames,
        "runtime_image_manifest_digest": operational.bootstrap.payload[
            "runtime_image_manifest_digest"
        ],
    }
    expected_runtime = {
        "additional_system_packages": [],
        "base_image": accepted.base_request.base_image,
        "base_image_archive_keyring_sha256": (
            accepted.base_request.base_image_archive_keyring_sha256
        ),
        "base_image_config_digest": operational.bootstrap.payload["base_image_config_digest"],
        "base_image_layer_diff_ids": operational.bootstrap.payload["base_image_layer_diff_ids"],
        "base_image_layer_digests": operational.bootstrap.payload["base_image_layer_digests"],
        "base_image_manifest_digest": accepted.base_request.base_image_manifest_digest,
        "base_system_package_evidence_sha256": accepted.base.sha256,
        "base_system_package_inventory": package_inventory,
        "base_system_package_inventory_sha256": accepted.base.package_inventory_sha256,
        "base_system_package_request_sha256": accepted.base.request_sha256,
        "build_context_manifest_sha256": accepted.bootstrap_request.build_context_manifest_sha256,
        "calibration_bootstrap_evidence_sha256": accepted.bootstrap.sha256,
        "calibration_bootstrap_request_sha256": accepted.bootstrap_request.sha256,
        "image_build": accepted.bootstrap.payload["image_build"],
        "platform": accepted.base_request.platform,
        "runtime_image_config_digest": operational.bootstrap.payload["runtime_image_config_digest"],
        "runtime_image_manifest_digest": operational.bootstrap.payload[
            "runtime_image_manifest_digest"
        ],
    }
    for field in (
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
    ):
        expected_runtime[field] = profile.payload[field]
    expected_seal = {
        **expected_backend,
        **{
            field: expected_runtime[field]
            for field in (
                "additional_system_packages",
                "base_image_archive_keyring_sha256",
                "base_image_config_digest",
                "base_image_layer_diff_ids",
                "base_image_layer_digests",
                "base_image_manifest_digest",
                "base_system_package_evidence_sha256",
                "base_system_package_inventory",
                "base_system_package_inventory_sha256",
                "base_system_package_request_sha256",
                "build_context_manifest_sha256",
                "calibration_bootstrap_evidence_sha256",
                "calibration_bootstrap_request_sha256",
                "runtime_image_config_digest",
                "runtime_image_manifest_digest",
            )
        },
        "calibration_measurement_evidence_sha256": measurement_sha256,
        "calibration_measurement_request_sha256": measurement_request.sha256,
        "measurements": [_row_payload(row) for row in measurement_rows],
        "native_host_attestation_bundle_sha256": operational.bundle.sha256,
        "native_host_evidence": operational.host_payload,
        "oci_layout_archive": operational.bootstrap.payload["oci_layout_archive"],
        "oci_layout_manifest_sha256": operational.bootstrap.payload["oci_layout_manifest_sha256"],
        "runtime_gid": accepted.bootstrap_request.runtime_gid,
        "runtime_image_index_digest": operational.bootstrap.payload["runtime_image_index_digest"],
        "runtime_image_layer_diff_ids": operational.bootstrap.payload[
            "runtime_image_layer_diff_ids"
        ],
        "runtime_image_layer_digests": operational.bootstrap.payload["runtime_image_layer_digests"],
        "runtime_uid": accepted.bootstrap_request.runtime_uid,
        "seal_profile_request_sha256": profile.sha256,
    }
    for field in (
        "cpu_limit_millis",
        "memory_limit_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "shm_bytes",
        "startup_deadline_seconds",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "stdout_max_line_bytes",
        "tmp_bytes",
    ):
        expected_seal[field] = profile.payload[field]
    _require_authority_fields(candidate.backend.payload, expected_backend, "candidate backend")
    _require_authority_fields(candidate.runtime.payload, expected_runtime, "candidate runtime")
    _require_authority_fields(candidate.seal.payload, expected_seal, "candidate seal")


def _require_authority_fields(
    payload: Mapping[str, JsonValue],
    expected: Mapping[str, JsonValue],
    label: str,
) -> None:
    if any(
        _plain_json(payload.get(field)) != _plain_json(value) for field, value in expected.items()
    ):
        raise SealError(f"{label} is unrelated to the accepted authority")


def _validate_bootstrap_image_files(
    *,
    image_directory: Path,
    bootstrap: CalibrationBootstrapEvidence,
) -> None:
    archive = _read_regular(image_directory / "runtime.oci.tar", "bootstrap OCI archive")
    expected_archive = cast(dict[str, JsonValue], bootstrap.payload["oci_layout_archive"])
    if (
        expected_archive.get("name") != "runtime.oci.tar"
        or expected_archive.get("size") != len(archive)
        or expected_archive.get("sha256") != sha256_hex(archive)
    ):
        raise SealError("bootstrap OCI archive does not match bootstrap evidence")
    layout = _read_regular(
        image_directory / "oci-layout-manifest.json",
        "bootstrap OCI layout manifest",
    )
    if bootstrap.payload["oci_layout_manifest_sha256"] != sha256_hex(layout):
        raise SealError("bootstrap OCI layout manifest does not match bootstrap evidence")


def _same_native_host(left: NativeHostEvidence, right: NativeHostEvidence) -> bool:
    return (
        left.kind == right.kind
        and left.official_execution_allowed == right.official_execution_allowed
        and _plain_json(left.payload) == _plain_json(right.payload)
        and left.sha256 == right.sha256
    )


def _require_matching_candidate_host_files(payload_root: Path) -> None:
    outer = payload_root / "candidate-host-attestation"
    inner = (
        payload_root
        / "seal-candidate/docs/superpowers/evidence/hpa-320/native/candidate-host-attestation"
    )
    for name in (
        "attestation-bundle.json",
        "native-host-evidence.json",
        "native-host-observation.json",
    ):
        if _read_regular(outer / name, f"candidate phase {name}") != _read_regular(
            inner / name,
            f"candidate repository-shaped {name}",
        ):
            raise SealError("candidate phase host files do not match repository-shaped copies")


_PHASE_VALIDATORS: Mapping[str, Callable[..., None]] = {
    "bootstrap": _validate_bootstrap_work_payload,
    "measurement": _validate_measurement_work_payload,
    "candidate": _validate_candidate_work_payload,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-host")
    validate.add_argument("--evidence", type=Path, required=True)

    bootstrap = commands.add_parser("bootstrap-image")
    bootstrap.add_argument("--request", type=Path, required=True)
    bootstrap.add_argument("--host-attestation-bundle", type=Path, required=True)
    bootstrap.add_argument("--host-evidence", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--repository-root", type=Path, required=True)

    packages = commands.add_parser("materialize-system-packages")
    packages.add_argument("--host-evidence", type=Path, required=True)
    packages.add_argument("--bundle", type=Path, required=True)
    packages.add_argument("--build-args-output", type=Path, required=True)

    base_system = commands.add_parser("attest-base-system")
    base_system.add_argument("--request", type=Path, required=True)
    base_system.add_argument("--bootstrap-request", type=Path, required=True)
    base_system.add_argument("--bootstrap-evidence", type=Path, required=True)
    base_system.add_argument("--host-attestation-bundle", type=Path, required=True)
    base_system.add_argument("--host-evidence", type=Path, required=True)
    base_system.add_argument("--output", type=Path, required=True)

    measurement = commands.add_parser("measure")
    measurement.add_argument("--request", type=Path, required=True)
    measurement.add_argument("--bootstrap-request", type=Path, required=True)
    measurement.add_argument("--bootstrap-evidence", type=Path, required=True)
    measurement.add_argument("--host-attestation-bundle", type=Path, required=True)
    measurement.add_argument("--host-evidence", type=Path, required=True)
    measurement.add_argument("--model-cache", type=Path, required=True)
    measurement.add_argument("--checkpoint-evidence", type=Path, required=True)
    measurement.add_argument("--base-system-evidence", type=Path, required=True)
    measurement.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--request", type=Path, required=True)
    calibration.add_argument("--measurement-evidence", type=Path, required=True)
    calibration.add_argument("--bootstrap-request", type=Path, required=True)
    calibration.add_argument("--bootstrap-evidence", type=Path, required=True)
    calibration.add_argument("--host-attestation-bundle", type=Path, required=True)
    calibration.add_argument("--checkpoint-evidence", type=Path, required=True)
    calibration.add_argument("--base-system-evidence", type=Path, required=True)
    calibration.add_argument("--host-evidence", type=Path, required=True)
    calibration.add_argument("--model-cache", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    reissue = commands.add_parser("reissue-bootstrap-request")
    reissue.add_argument("--repository-root", type=Path, required=True)
    reissue.add_argument("--request", type=Path, required=True)

    seal = commands.add_parser("seal")
    seal.add_argument("--candidate", type=Path, required=True)
    seal.add_argument("--repository-root", type=Path, required=True)
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
        if arguments.command == "reissue-bootstrap-request":
            digest = reissue_calibration_bootstrap_request(
                request_path=arguments.request,
                repository_root=arguments.repository_root,
            )
            _write_producer_summary(ProducerOutcome("reissued", 0, arguments.request, digest))
            return 0
        if arguments.command == "bootstrap-image":
            published = bootstrap_image(
                request_path=arguments.request,
                host_attestation_bundle_path=arguments.host_attestation_bundle,
                host_evidence_path=arguments.host_evidence,
                output=arguments.output,
                repository_root=arguments.repository_root,
            )
            _write_producer_summary(
                ProducerOutcome("bootstrapped", 0, published.path, published.sha256)
            )
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
                bootstrap_request_path=arguments.bootstrap_request,
                bootstrap_evidence_path=arguments.bootstrap_evidence,
                host_attestation_bundle_path=arguments.host_attestation_bundle,
                host_evidence_path=arguments.host_evidence,
                output_path=arguments.output,
            )
            _write_producer_summary(
                ProducerOutcome("attested", 0, published.path, published.sha256)
            )
            return 0
        if arguments.command == "measure":
            published = measure(
                request_path=arguments.request,
                bootstrap_request_path=arguments.bootstrap_request,
                bootstrap_evidence_path=arguments.bootstrap_evidence,
                host_attestation_bundle_path=arguments.host_attestation_bundle,
                host_evidence_path=arguments.host_evidence,
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
                bootstrap_request_path=arguments.bootstrap_request,
                bootstrap_evidence_path=arguments.bootstrap_evidence,
                host_attestation_bundle_path=arguments.host_attestation_bundle,
                checkpoint_evidence_path=arguments.checkpoint_evidence,
                base_system_evidence_path=arguments.base_system_evidence,
                host_evidence=arguments.host_evidence,
                model_cache=arguments.model_cache,
                output=arguments.output,
            )
            _write_producer_summary(
                ProducerOutcome("calibrated", 0, published.artifact.path, published.artifact.sha256)
            )
            return 0
        published = seal_candidate(
            candidate=arguments.candidate,
            repository_root=arguments.repository_root,
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
