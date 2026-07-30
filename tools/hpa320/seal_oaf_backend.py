#!/usr/bin/env python3
"""Validate and immutably publish the frozen OaF backend seal."""

# The seal gate deliberately keeps every cross-record check visible in one module.
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import os
import platform
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

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
from tools.hpa320.audit_legacy_tf2_conversion import (
    CANDIDATE_MANIFEST_NAME,
    CANDIDATE_MANIFEST_SCHEMA,
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


@dataclass(frozen=True)
class PublishedSeal:
    """The immutable artifacts published by one successful seal operation."""

    artifacts: tuple[PublishedArtifact, ...]
    publication_order: tuple[str, ...]


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


def calibrate(
    *,
    image: str,
    host_evidence: Path,
    model_cache: Path,
    output: Path,
) -> None:
    """Enforce calibration prerequisites without accepting an invented native report."""

    load_native_host_evidence(host_evidence)
    if (
        not isinstance(image, str)
        or not image
        or image.strip().lower() in {"auto", "none", "sentinel", "unlimited", "unset"}
    ):
        raise SealError("calibration image identity is invalid")
    _require_directory(Path(model_cache), "model cache")
    _require_absent(Path(output), "calibration candidate")
    raise SealError("native calibration output contract is not specified; no output was written")


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

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--image", required=True)
    calibration.add_argument("--host-evidence", type=Path, required=True)
    calibration.add_argument("--model-cache", type=Path, required=True)
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
            evidence = load_native_host_evidence(arguments.evidence)
            summary = {
                "kind": evidence.kind,
                "official_execution_allowed": evidence.official_execution_allowed,
                "sha256": evidence.sha256,
            }
            sys.stdout.buffer.write(canonical_json_bytes(summary, trailing_newline=True))
            return 0
        if arguments.command == "materialize-system-packages":
            materialize_system_packages(
                host_evidence=arguments.host_evidence,
                bundle=arguments.bundle,
                build_args_output=arguments.build_args_output,
            )
            return 0
        if arguments.command == "calibrate":
            calibrate(
                image=arguments.image,
                host_evidence=arguments.host_evidence,
                model_cache=arguments.model_cache,
                output=arguments.output,
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
        summary = {
            "backend_lock_sha256": published.artifacts[-1].sha256,
            "publication_order": list(published.publication_order),
            "status": "sealed",
        }
        sys.stdout.buffer.write(canonical_json_bytes(summary, trailing_newline=True))
        return 0
    except SealError as error:
        print(f"OaF seal failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
