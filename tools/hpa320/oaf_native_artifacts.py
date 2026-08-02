"""Strict, immutable manifests for same-job OaF native work artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, cast
from uuid import uuid4

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    DirectoryAnchor,
    _rename_no_replace_syscall,
    open_directory_anchor,
    publish_immutable_bytes,
    read_regular_file_no_follow,
)
from src.benchmark.checkpoint_acquisition import CheckpointIdentity
from tools.hpa320.oaf_host_attestation import (
    HostAttestationError,
    NativeHostAttestationBundle,
    load_native_host_attestation_bundle,
)

MANIFEST_SCHEMA = "crux.oaf-native-work-artifact-manifest/v1"
MANIFEST_NAME = "artifact-manifest.json"
ROLES = frozenset(
    {
        "advisory_snapshot",
        "backend_lock",
        "base_system_package_evidence",
        "calibration_bootstrap_evidence",
        "calibration_measurement_evidence",
        "checkpoint_acquisition_evidence",
        "conversion_audit",
        "diagnostic",
        "host_adapter_source_manifest",
        "native_host_attestation_bundle",
        "native_host_evidence",
        "native_host_observation",
        "oci_layout_archive",
        "oci_layout_manifest",
        "runtime_lock",
        "seal_candidate_manifest",
        "seal_evidence",
        "security_scan",
        "smoke_audio",
        "smoke_oracle",
        "smoke_prediction",
        "tensor_coverage",
    }
)

BOOTSTRAP_FILES = MappingProxyType(
    {
        "bootstrap-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
        "bootstrap-host-attestation/native-host-evidence.json": "native_host_evidence",
        "bootstrap-host-attestation/native-host-observation.json": "native_host_observation",
        "checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
        "calibration-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
        "calibration-image/oci-layout-manifest.json": "oci_layout_manifest",
        "calibration-image/runtime.oci.tar": "oci_layout_archive",
        "base-system-package-evidence.json": "base_system_package_evidence",
    }
)

MEASUREMENT_FILES = MappingProxyType(
    {
        "measurement-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
        "measurement-host-attestation/native-host-evidence.json": "native_host_evidence",
        "measurement-host-attestation/native-host-observation.json": "native_host_observation",
        "operational-checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
        "operational-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
        "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
        "operational-image/runtime.oci.tar": "oci_layout_archive",
        "calibration-measurement-evidence.json": "calibration_measurement_evidence",
    }
)

CANDIDATE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "conversion_audit",
        "docs/superpowers/evidence/hpa-320/legacy-conversion-audit.json",
    ),
    (
        "native_host_attestation_bundle",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/attestation-bundle.json",
    ),
    (
        "native_host_evidence",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-evidence.json",
    ),
    (
        "native_host_observation",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-observation.json",
    ),
    (
        "host_adapter_source_manifest",
        "runtime/oaf_tf1/host-adapter-source-manifest.json",
    ),
    (
        "tensor_coverage",
        "docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json",
    ),
    (
        "advisory_snapshot",
        "docs/superpowers/evidence/hpa-320/oaf-advisory-snapshot.json",
    ),
    (
        "security_scan",
        "docs/superpowers/evidence/hpa-320/oaf-security-scan.json",
    ),
    (
        "oci_layout_archive",
        "artifacts/benchmark/backends/oaf-tf1/runtime.oci.tar",
    ),
    (
        "oci_layout_manifest",
        "docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json",
    ),
    ("smoke_audio", "tests/fixtures/oaf_tf1_smoke/canonical.wav"),
    (
        "smoke_prediction",
        "docs/superpowers/evidence/hpa-320/oaf-smoke-prediction.jsonl",
    ),
    ("smoke_oracle", "tests/fixtures/oaf_tf1_smoke/smoke-oracle.json"),
    (
        "seal_evidence",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json",
    ),
    (
        "runtime_lock",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json",
    ),
    (
        "backend_lock",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json",
    ),
)
CANDIDATE_ARTIFACT_PATHS: Mapping[str, str] = MappingProxyType(dict(CANDIDATE_ARTIFACTS))
CANDIDATE_FILES = MappingProxyType(
    {
        "candidate-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
        "candidate-host-attestation/native-host-evidence.json": "native_host_evidence",
        "candidate-host-attestation/native-host-observation.json": "native_host_observation",
        "operational-checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
        "operational-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
        "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
        "operational-image/runtime.oci.tar": "oci_layout_archive",
        "seal-candidate/candidate-manifest.json": "seal_candidate_manifest",
        **{f"seal-candidate/{path}": role for role, path in CANDIDATE_ARTIFACTS},
    }
)
if len(CANDIDATE_FILES) != 24:
    raise ValueError("CANDIDATE_FILES must contain exactly 24 entries")

PHASE_FILES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "bootstrap": BOOTSTRAP_FILES,
        "measurement": MEASUREMENT_FILES,
        "candidate": CANDIDATE_FILES,
        "diagnostic": MappingProxyType({}),
    }
)
PHASE_HOST_BUNDLE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "bootstrap": "bootstrap-host-attestation/attestation-bundle.json",
        "measurement": "measurement-host-attestation/attestation-bundle.json",
        "candidate": "candidate-host-attestation/attestation-bundle.json",
    }
)

_MANIFEST_KEYS = frozenset(
    {
        "files",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_workflow_ref",
        "github_workflow_sha",
        "native_host_attestation_bundle_sha256",
        "phase",
        "schema",
        "status",
        "workflow_commit",
    }
)
_REFERENCE_KEYS = frozenset({"path", "role", "sha256", "size"})
_CHUNK_SIZE = 1024 * 1024
_MANIFEST_MAX_BYTES = 1024 * 1024
_SCAN_CHUNK_HOOK: Callable[[Path], None] | None = None
_ARCHIVE_CHUNK_HOOK: Callable[[Path], None] | None = None
_GH_VERSION = re.compile(r"gh version ([0-9]+)\.([0-9]+)\.([0-9]+)(?:\s.*)?\Z")
_WORKFLOW_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class NativeArtifactError(ValueError):
    """The native work-artifact payload or manifest is unsafe or inconsistent."""


@dataclass(frozen=True)
class VerifiedGitHubAttestation:
    """The one statement jointly verified for an immutable work-artifact pair."""

    gh_version: str
    statement: Mapping[str, JsonValue]
    statement_sha256: str
    subjects: tuple[CheckpointIdentity, ...]


@dataclass(frozen=True)
class _SnapshotBinding:
    """One file-system identity that must survive the native acceptance gate."""

    path: Path
    metadata: os.stat_result
    sha256: str | None


@dataclass(frozen=True)
class _AttestationSnapshot:
    """Private immutable copies consumed by every acceptance gate."""

    archive_path: Path
    manifest_path: Path
    payload_root: Path
    bundle_path: Path
    bundle_identity: CheckpointIdentity
    trusted_root_path: Path
    trusted_root_identity: CheckpointIdentity
    bindings: tuple[_SnapshotBinding, ...]


def _run_checked_command(command: tuple[str, ...]) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise NativeArtifactError(
            f"GitHub attestation command timed out after {error.timeout} seconds"
        ) from error
    if completed.returncode != 0:
        raise NativeArtifactError("GitHub attestation command failed")
    if not isinstance(completed.stdout, bytes):
        raise NativeArtifactError("GitHub attestation command did not produce bytes")
    return completed.stdout


def verify_github_attestations(
    *,
    phase: str,
    workflow_commit: str,
    manifest_path: Path,
    archive_path: Path,
    sigstore_bundle_path: Path,
    trusted_root_path: Path,
    command_runner: Callable[[tuple[str, ...]], bytes] = _run_checked_command,
) -> VerifiedGitHubAttestation:
    """Verify both signed subjects under the frozen GitHub policy."""

    if phase not in PHASE_HOST_BUNDLE_PATHS:
        raise NativeArtifactError("native work attestation phase is invalid")
    if _WORKFLOW_COMMIT.fullmatch(workflow_commit) is None:
        raise NativeArtifactError("native work attestation workflow commit is invalid")
    try:
        gh_version = _supported_gh_version(command_runner(("gh", "version")))
        subjects = _attestation_subjects(
            phase=phase,
            workflow_commit=workflow_commit,
            manifest_path=Path(manifest_path),
            archive_path=Path(archive_path),
        )
        policy = (
            "--repo",
            "cwchanap/Crux",
            "--signer-workflow",
            f"cwchanap/Crux/.github/workflows/hpa320-native-{phase}.yml",
            "--source-digest",
            workflow_commit,
            "--signer-digest",
            workflow_commit,
            "--deny-self-hosted-runners",
            "--digest-alg",
            "sha256",
            "--predicate-type",
            "https://slsa.dev/provenance/v1",
            "--cert-oidc-issuer",
            "https://token.actions.githubusercontent.com",
            "--bundle",
            str(sigstore_bundle_path),
            "--custom-trusted-root",
            str(trusted_root_path),
            "--format",
            "json",
        )
        statements = tuple(
            _verification_statement(
                command_runner(("gh", "attestation", "verify", str(path), *policy))
            )
            for path in (Path(manifest_path), Path(archive_path))
        )
    except NativeArtifactError:
        raise
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as error:
        raise NativeArtifactError("GitHub attestation verification failed") from error
    first, second = statements
    first_content = canonical_json_bytes(first)
    if first_content != canonical_json_bytes(second):
        raise NativeArtifactError("GitHub attestation statements disagree")
    _validate_attestation_subjects(first, subjects)
    return VerifiedGitHubAttestation(
        gh_version=gh_version,
        statement=first,
        statement_sha256=hashlib.sha256(first_content).hexdigest(),
        subjects=subjects,
    )


def _supported_gh_version(output: bytes) -> str:
    try:
        first_line = output.decode("utf-8", errors="strict").splitlines()[0]
    except (IndexError, UnicodeDecodeError) as error:
        raise NativeArtifactError("GitHub CLI version output is invalid") from error
    match = _GH_VERSION.fullmatch(first_line)
    if match is None:
        raise NativeArtifactError("GitHub CLI version output is invalid")
    version = tuple(int(part) for part in match.groups())
    if version < (2, 68, 0) or version >= (3, 0, 0):
        raise NativeArtifactError("GitHub CLI must satisfy 2.68.0 <= gh < 3.0.0")
    return ".".join(match.groups())


def _attestation_subjects(
    *,
    phase: str,
    workflow_commit: str,
    manifest_path: Path,
    archive_path: Path,
) -> tuple[CheckpointIdentity, ...]:
    manifest_name, archive_name = _attestation_subject_names(
        phase=phase,
        workflow_commit=workflow_commit,
        manifest_path=manifest_path,
        archive_path=archive_path,
    )
    paths_and_names = (
        (manifest_path, manifest_name),
        (archive_path, archive_name),
    )
    subjects: list[CheckpointIdentity] = []
    for path, name in paths_and_names:
        try:
            identity = _stream_regular_file_identity(path)
        except OSError as error:
            raise NativeArtifactError("GitHub attestation subject is missing or unsafe") from error
        subjects.append(
            CheckpointIdentity(
                name=name,
                sha256=identity.sha256,
                size=identity.size,
            )
        )
    return tuple(subjects)


def _attestation_subject_names(
    *,
    phase: str,
    workflow_commit: str,
    manifest_path: Path,
    archive_path: Path,
) -> tuple[str, str]:
    archive_name = f"hpa320-native-{phase}-{workflow_commit}.tar"
    if manifest_path.name != MANIFEST_NAME or archive_path.name != archive_name:
        raise NativeArtifactError("GitHub attestation subject basename is invalid")
    return MANIFEST_NAME, archive_name


def _verification_statement(content: bytes) -> dict[str, JsonValue]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeArtifactError(
            "GitHub attestation verification result is invalid JSON"
        ) from error
    if not isinstance(value, list) or len(value) != 1:
        raise NativeArtifactError("GitHub attestation verification result must have one entry")
    result = value[0]
    if not isinstance(result, dict) or "verificationResult" not in result:
        raise NativeArtifactError("GitHub attestation verification result fields are invalid")
    verification = result["verificationResult"]
    if not isinstance(verification, dict) or "statement" not in verification:
        raise NativeArtifactError("GitHub attestation statement is missing")
    statement = verification["statement"]
    if not isinstance(statement, dict):
        raise NativeArtifactError("GitHub attestation statement is invalid")
    predicate = statement.get("predicate")
    if isinstance(predicate, dict) and predicate.get("status") == "failed":
        raise NativeArtifactError("GitHub attestation statement is a failed result")
    return cast(dict[str, JsonValue], statement)


@contextmanager
def _attestation_snapshot(
    *,
    phase: str,
    payload_root: Path,
    archive_path: Path,
    sigstore_bundle_path: Path,
    trusted_root_path: Path,
) -> Iterator[_AttestationSnapshot]:
    """Copy only the exact phase allowlist before any external verifier runs."""

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    snapshot_root: Path | None = None
    try:
        source_root = _absolute_path(payload_root, "payload root")
        source_archive = _absolute_path(archive_path, "archive")
        source_bundle = _absolute_path(sigstore_bundle_path, "sigstore bundle")
        source_trusted_root = _absolute_path(trusted_root_path, "trusted root")
        _scan_phase_payload(phase=phase, payload_root=source_root, include_manifest=False)
        temporary_directory = tempfile.TemporaryDirectory(prefix=".hpa320-attestation-")
        temporary_root = Path(os.path.realpath(temporary_directory.name))
        snapshot_root = temporary_root / "payload"
        snapshot_root.mkdir(mode=0o700)
        for relative in (MANIFEST_NAME, *_phase_files(phase)):
            _copy_snapshot_file(
                source=source_root / relative,
                destination=snapshot_root / relative,
            )
        snapshot_archive = temporary_root / source_archive.name
        _copy_snapshot_file(source=source_archive, destination=snapshot_archive)
        snapshot_bundle = temporary_root / "sigstore.bundle.json"
        snapshot_trusted_root = temporary_root / "trusted-root.json"
        _copy_snapshot_file(source=source_bundle, destination=snapshot_bundle)
        _copy_snapshot_file(source=source_trusted_root, destination=snapshot_trusted_root)
        _scan_phase_payload(phase=phase, payload_root=source_root, include_manifest=False)
        _seal_snapshot(
            phase=phase,
            payload_root=snapshot_root,
            archive_path=snapshot_archive,
        )
        os.chmod(snapshot_bundle, 0o400)
        os.chmod(snapshot_trusted_root, 0o400)
        bindings: list[_SnapshotBinding] = list(
            _capture_snapshot_bindings(
                phase=phase,
                payload_root=snapshot_root,
                archive_path=snapshot_archive,
            )
        )
        bindings.append(_capture_regular_file_binding(snapshot_bundle))
        bindings.append(_capture_regular_file_binding(snapshot_trusted_root))
        yield _AttestationSnapshot(
            archive_path=snapshot_archive,
            manifest_path=snapshot_root / MANIFEST_NAME,
            payload_root=snapshot_root,
            bundle_path=snapshot_bundle,
            bundle_identity=_stream_regular_file_identity(snapshot_bundle),
            trusted_root_path=snapshot_trusted_root,
            trusted_root_identity=_stream_regular_file_identity(snapshot_trusted_root),
            bindings=tuple(bindings),
        )
    except NativeArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise NativeArtifactError("native work attestation snapshot failed") from error
    finally:
        if temporary_directory is not None:
            try:
                if snapshot_root is not None:
                    _make_snapshot_cleanup_writable(snapshot_root)
                temporary_directory.cleanup()
            except OSError as error:
                raise NativeArtifactError(
                    "native work attestation snapshot cleanup failed"
                ) from error


def _copy_snapshot_file(*, source: Path, destination: Path) -> None:
    descriptor: int | None = None
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open_directory_anchor(destination.parent) as parent:
        try:
            parent.verify()
            _require_absent_at(parent, destination.name, "snapshot destination")
            descriptor = os.open(
                destination.name,
                _new_regular_file_flags(),
                0o600,
                dir_fd=parent.descriptor,
            )
            initial = _require_owned_regular_descriptor(descriptor)
            source_identity = _copy_regular_file(
                source_path=source,
                destination_descriptor=descriptor,
            )
            os.fsync(descriptor)
            copied = _require_owned_regular_descriptor(descriptor)
            if not _same_inode(initial, copied):
                raise NativeArtifactError("snapshot destination inode changed while copying")
            _verify_regular_file_binding_at(parent, destination.name, copied)
            copied_identity = _stream_descriptor_identity(
                descriptor,
                name=destination.name,
                expected=copied,
            )
            if (copied_identity.sha256, copied_identity.size) != (
                source_identity.sha256,
                source_identity.size,
            ):
                raise NativeArtifactError("snapshot copy differs from its source")
            parent.verify()
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _snapshot_directory_paths(*, phase: str, payload_root: Path) -> tuple[Path, ...]:
    return (
        payload_root,
        *(
            payload_root / relative
            for relative in sorted(_expected_directories(_phase_files(phase)))
        ),
    )


def _snapshot_file_paths(*, phase: str, payload_root: Path, archive_path: Path) -> tuple[Path, ...]:
    return (
        payload_root / MANIFEST_NAME,
        *(payload_root / relative for relative in sorted(_phase_files(phase))),
        archive_path,
    )


def _seal_snapshot(*, phase: str, payload_root: Path, archive_path: Path) -> None:
    for path in _snapshot_file_paths(
        phase=phase,
        payload_root=payload_root,
        archive_path=archive_path,
    ):
        os.chmod(path, 0o400)
    for path in sorted(
        _snapshot_directory_paths(phase=phase, payload_root=payload_root),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        os.chmod(path, 0o500)


def _capture_regular_file_binding(path: Path) -> _SnapshotBinding:
    before = os.stat(path, follow_symlinks=False)
    identity = _stream_regular_file_identity(path)
    after = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(after.st_mode) or not _same_file_snapshot(before, after):
        raise NativeArtifactError("native work attestation snapshot file is unsafe")
    return _SnapshotBinding(path=path, metadata=after, sha256=identity.sha256)


def _capture_snapshot_bindings(
    *,
    phase: str,
    payload_root: Path,
    archive_path: Path,
) -> tuple[_SnapshotBinding, ...]:
    bindings: list[_SnapshotBinding] = []
    for path in _snapshot_directory_paths(phase=phase, payload_root=payload_root):
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise NativeArtifactError("native work attestation snapshot directory is unsafe")
        bindings.append(_SnapshotBinding(path=path, metadata=metadata, sha256=None))
    for path in _snapshot_file_paths(
        phase=phase,
        payload_root=payload_root,
        archive_path=archive_path,
    ):
        bindings.append(_capture_regular_file_binding(path))
    return tuple(bindings)


def _require_live_snapshot_bindings(snapshot: _AttestationSnapshot) -> None:
    for binding in snapshot.bindings:
        metadata = os.stat(binding.path, follow_symlinks=False)
        if not _same_file_snapshot(binding.metadata, metadata):
            raise NativeArtifactError("native work attestation snapshot identity changed")
        if binding.sha256 is None:
            if not stat.S_ISDIR(metadata.st_mode):
                raise NativeArtifactError("native work attestation snapshot directory changed")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise NativeArtifactError("native work attestation snapshot file changed")
        identity = _stream_regular_file_identity(binding.path)
        after = os.stat(binding.path, follow_symlinks=False)
        if (
            not _same_file_snapshot(binding.metadata, after)
            or identity.sha256 != binding.sha256
            or identity.size != binding.metadata.st_size
        ):
            raise NativeArtifactError("native work attestation snapshot content changed")


def _make_snapshot_cleanup_writable(snapshot_root: Path) -> None:
    for path in sorted(snapshot_root.rglob("*"), key=lambda candidate: len(candidate.parts)):
        metadata = os.stat(path, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(path, 0o700)
        elif stat.S_ISREG(metadata.st_mode):
            os.chmod(path, 0o600)
    metadata = os.stat(snapshot_root, follow_symlinks=False)
    if stat.S_ISDIR(metadata.st_mode):
        os.chmod(snapshot_root, 0o700)


def _validate_attestation_subjects(
    statement: Mapping[str, JsonValue],
    expected: tuple[CheckpointIdentity, ...],
) -> None:
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or len(subjects) != len(expected):
        raise NativeArtifactError("GitHub attestation subjects are invalid")
    actual: list[tuple[str, str]] = []
    for subject in subjects:
        if not isinstance(subject, dict) or set(subject) != {"name", "digest"}:
            raise NativeArtifactError("GitHub attestation subject fields are invalid")
        name = subject["name"]
        digest = subject["digest"]
        if (
            not isinstance(name, str)
            or not isinstance(digest, dict)
            or set(digest) != {"sha256"}
            or not isinstance(digest["sha256"], str)
        ):
            raise NativeArtifactError("GitHub attestation subject identity is invalid")
        actual.append((name, digest["sha256"]))
    expected_identities = [(subject.name, subject.sha256) for subject in expected]
    if sorted(actual) != sorted(expected_identities):
        raise NativeArtifactError("GitHub attestation subjects do not match local artifacts")


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    role: str
    sha256: str
    size: int


@dataclass(frozen=True)
class NativeWorkArtifactManifest:  # pylint: disable=too-many-instance-attributes
    path: Path
    phase: str
    files: tuple[ArtifactReference, ...]
    github_job: str
    github_repository: str
    github_run_attempt: int
    github_run_id: int
    github_workflow_ref: str
    github_workflow_sha: str
    native_host_attestation_bundle_sha256: str
    status: str
    workflow_commit: str
    sha256: str


@dataclass(frozen=True)
class BootstrapPayload:
    root: Path
    host_bundle: Path


@dataclass(frozen=True)
class PackedBootstrap:
    payload_root: Path
    manifest: Path
    archive: Path


@dataclass(frozen=True)
class _ArchiveMember:
    name: str
    mode: int
    size: int
    source_relative: str | None
    source_identity: CheckpointIdentity | None
    typeflag: bytes


def publish_native_work_manifest(
    *,
    phase: str,
    payload_root: Path,
    host_bundle_path: Path,
) -> NativeWorkArtifactManifest:
    """Scan the exact phase allowlist and atomically publish artifact-manifest.json."""

    try:
        root = _absolute_path(payload_root, "payload root")
        host_path = _absolute_path(host_bundle_path, "host bundle")
        expected_host_path = root / _phase_host_bundle_path(phase)
        if host_path != expected_host_path:
            raise NativeArtifactError("native-host bundle is not at the phase-owned path")
        with open_directory_anchor(root) as anchor:
            _verify_root_binding(root, anchor)
            _require_manifest_absent(anchor)
            host = load_native_host_attestation_bundle(host_path, expected_phase=phase)
            files = _scan_phase_payload(
                phase=phase,
                payload_root=root,
                include_manifest=False,
                anchor=anchor,
            )
            if load_native_host_attestation_bundle(host_path, expected_phase=phase) != host:
                raise NativeArtifactError("native-host bundle changed while payload was scanned")
            _validate_host_bundle_reference(phase=phase, files=files, host=host)
            content = canonical_json_bytes(
                _manifest_payload(phase=phase, files=files, host=host),
                trailing_newline=True,
            )
            digest = hashlib.sha256(content).hexdigest()
            _verify_root_binding(root, anchor)
            published = publish_immutable_bytes(
                root / MANIFEST_NAME,
                content,
                digest,
                role="native_work_manifest",
                anchor=anchor,
            )
            _verify_root_binding(root, anchor)
        return load_native_work_manifest(published.path, expected_phase=phase)
    except NativeArtifactError:
        raise
    except (
        ArtifactPublicationError,
        HostAttestationError,
        OSError,
        StrictJsonError,
        TypeError,
        ValueError,
    ) as error:
        raise NativeArtifactError("native work manifest publication failed") from error


def load_native_work_manifest(
    path: Path,
    *,
    expected_phase: str,
) -> NativeWorkArtifactManifest:
    """Strict-load one canonical native work manifest."""

    try:
        if expected_phase not in PHASE_FILES or expected_phase not in PHASE_HOST_BUNDLE_PATHS:
            raise NativeArtifactError("expected native work phase is invalid")
        manifest_path = _absolute_path(path, "manifest")
        if manifest_path.name != MANIFEST_NAME:
            raise NativeArtifactError("native work manifest has the wrong filename")
        root = manifest_path.parent
        with open_directory_anchor(root) as anchor:
            _verify_root_binding(root, anchor)
            content = read_regular_file_no_follow(
                manifest_path,
                anchor=anchor,
                max_bytes=_MANIFEST_MAX_BYTES,
            )
            payload = _canonical_object(content)
            if (
                set(payload) != _MANIFEST_KEYS
                or payload["schema"] != MANIFEST_SCHEMA
                or payload["phase"] != expected_phase
                or payload["status"] != "success"
            ):
                raise NativeArtifactError("native work manifest schema or phase is invalid")
            host = load_native_host_attestation_bundle(
                root / _phase_host_bundle_path(expected_phase),
                expected_phase=expected_phase,
            )
            _verify_root_binding(root, anchor)
            files = _load_references(payload["files"], expected_phase)
            _validate_host_bundle_reference(phase=expected_phase, files=files, host=host)
            _validate_host_bound_metadata(payload, host)
            return _manifest_from_payload(
                path=manifest_path,
                phase=expected_phase,
                files=files,
                payload=payload,
                sha256=hashlib.sha256(content).hexdigest(),
            )
    except NativeArtifactError:
        raise
    except (HostAttestationError, OSError, StrictJsonError, TypeError, ValueError) as error:
        raise NativeArtifactError("native work manifest is invalid or unsafe") from error


def verify_native_work_payload(
    *,
    payload_root: Path,
    manifest: NativeWorkArtifactManifest,
) -> None:
    """Reproduce every manifest row through no-follow descriptors."""

    try:
        if not isinstance(manifest, NativeWorkArtifactManifest):
            raise NativeArtifactError("native work manifest type is invalid")
        root = _absolute_path(payload_root, "payload root")
        if manifest.path != root / MANIFEST_NAME:
            raise NativeArtifactError("native work manifest does not belong to payload root")
        with open_directory_anchor(root) as anchor:
            _verify_root_binding(root, anchor)
            manifest_content = read_regular_file_no_follow(
                manifest.path,
                anchor=anchor,
                max_bytes=_MANIFEST_MAX_BYTES,
            )
            if hashlib.sha256(manifest_content).hexdigest() != manifest.sha256:
                raise NativeArtifactError("native work manifest bytes changed")
            expected = tuple((row.path, row.role, row.sha256, row.size) for row in manifest.files)
            actual = tuple(
                (row.path, row.role, row.sha256, row.size)
                for row in _scan_phase_payload(
                    phase=manifest.phase,
                    payload_root=root,
                    include_manifest=False,
                    anchor=anchor,
                )
            )
            _verify_root_binding(root, anchor)
            if actual != expected:
                raise NativeArtifactError("native work payload does not reproduce its manifest")
    except NativeArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise NativeArtifactError("native work payload is invalid or unsafe") from error


def _manifest_payload(
    *,
    phase: str,
    files: tuple[ArtifactReference, ...],
    host: NativeHostAttestationBundle,
) -> dict[str, JsonValue]:
    evidence = host.evidence.payload
    return {
        "files": [
            {"path": row.path, "role": row.role, "sha256": row.sha256, "size": row.size}
            for row in files
        ],
        "github_job": cast(str, evidence["github_job"]),
        "github_repository": cast(str, evidence["github_repository"]),
        "github_run_attempt": cast(int, evidence["github_run_attempt"]),
        "github_run_id": cast(int, evidence["github_run_id"]),
        "github_workflow_ref": cast(str, evidence["github_workflow_ref"]),
        "github_workflow_sha": cast(str, evidence["github_workflow_sha"]),
        "native_host_attestation_bundle_sha256": host.sha256,
        "phase": phase,
        "schema": MANIFEST_SCHEMA,
        "status": "success",
        "workflow_commit": cast(str, evidence["workflow_commit"]),
    }


def _manifest_from_payload(
    *,
    path: Path,
    phase: str,
    files: tuple[ArtifactReference, ...],
    payload: Mapping[str, JsonValue],
    sha256: str,
) -> NativeWorkArtifactManifest:
    return NativeWorkArtifactManifest(
        path=path,
        phase=phase,
        files=files,
        github_job=cast(str, payload["github_job"]),
        github_repository=cast(str, payload["github_repository"]),
        github_run_attempt=cast(int, payload["github_run_attempt"]),
        github_run_id=cast(int, payload["github_run_id"]),
        github_workflow_ref=cast(str, payload["github_workflow_ref"]),
        github_workflow_sha=cast(str, payload["github_workflow_sha"]),
        native_host_attestation_bundle_sha256=cast(
            str,
            payload["native_host_attestation_bundle_sha256"],
        ),
        status=cast(str, payload["status"]),
        workflow_commit=cast(str, payload["workflow_commit"]),
        sha256=sha256,
    )


def _load_references(value: object, phase: str) -> tuple[ArtifactReference, ...]:
    if not isinstance(value, list) or not value:
        raise NativeArtifactError("native work manifest files must be a nonempty array")
    expected_mapping = _phase_files(phase)
    expected_rows = tuple(
        sorted(
            ((role, path) for path, role in expected_mapping.items()),
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
        )
    )
    references: list[ArtifactReference] = []
    seen_paths: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != _REFERENCE_KEYS:
            raise NativeArtifactError("native work manifest file reference fields are invalid")
        path = raw["path"]
        role = raw["role"]
        digest = raw["sha256"]
        size = raw["size"]
        if not isinstance(path, str) or not isinstance(role, str):
            raise NativeArtifactError("native work manifest file reference is invalid")
        if not _safe_relative_path(path) or _is_self_reference(path) or role not in ROLES:
            raise NativeArtifactError("native work manifest file reference is invalid")
        if not _is_sha256(digest):
            raise NativeArtifactError("native work manifest file reference is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise NativeArtifactError("native work manifest file reference is invalid")
        if path in seen_paths or expected_mapping.get(path) != role:
            raise NativeArtifactError("native work manifest file reference is invalid")
        seen_paths.add(path)
        references.append(
            ArtifactReference(
                path=path,
                role=role,
                sha256=cast(str, digest),
                size=size,
            )
        )
    actual_rows = tuple((row.role, row.path) for row in references)
    if actual_rows != expected_rows:
        raise NativeArtifactError("native work manifest rows do not match the phase allowlist")
    return tuple(references)


def _validate_host_bound_metadata(
    payload: Mapping[str, JsonValue],
    host: NativeHostAttestationBundle,
) -> None:
    evidence = host.evidence.payload
    expected = {
        "github_job": evidence["github_job"],
        "github_repository": evidence["github_repository"],
        "github_run_attempt": evidence["github_run_attempt"],
        "github_run_id": evidence["github_run_id"],
        "github_workflow_ref": evidence["github_workflow_ref"],
        "github_workflow_sha": evidence["github_workflow_sha"],
        "native_host_attestation_bundle_sha256": host.sha256,
        "workflow_commit": evidence["workflow_commit"],
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise NativeArtifactError("native work manifest does not match its host attestation bundle")


def _validate_host_bundle_reference(
    *,
    phase: str,
    files: tuple[ArtifactReference, ...],
    host: NativeHostAttestationBundle,
) -> None:
    expected_path = _phase_host_bundle_path(phase)
    matching = [row for row in files if row.path == expected_path]
    if (
        len(matching) != 1
        or matching[0].role != "native_host_attestation_bundle"
        or matching[0].sha256 != host.sha256
    ):
        raise NativeArtifactError("native-host bundle row does not match its attestation identity")


def _scan_phase_payload(
    *,
    phase: str,
    payload_root: Path,
    include_manifest: bool,
    anchor: DirectoryAnchor | None = None,
) -> tuple[ArtifactReference, ...]:
    root = _absolute_path(payload_root, "payload root")
    if anchor is None:
        with open_directory_anchor(root) as owned_anchor:
            _verify_root_binding(root, owned_anchor)
            rows = _scan_with_anchor(
                phase=phase,
                root=root,
                include_manifest=include_manifest,
                anchor=owned_anchor,
            )
            _verify_root_binding(root, owned_anchor)
            return rows
    _verify_root_binding(root, anchor)
    return _scan_with_anchor(
        phase=phase,
        root=root,
        include_manifest=include_manifest,
        anchor=anchor,
    )


def _scan_with_anchor(
    *,
    phase: str,
    root: Path,
    include_manifest: bool,
    anchor: DirectoryAnchor,
) -> tuple[ArtifactReference, ...]:
    mapping = _phase_files(phase)
    expected_directories = _expected_directories(mapping)
    rows = _scan_directory(
        descriptor=anchor.descriptor,
        relative_parts=(),
        root=root,
        mapping=mapping,
        expected_directories=expected_directories,
        include_manifest=include_manifest,
        anchor=anchor,
    )
    _verify_root_binding(root, anchor)
    actual = {row.path: row.role for row in rows}
    if actual != dict(mapping) or len(rows) != len(mapping):
        raise NativeArtifactError("native work payload differs from the exact phase allowlist")
    return tuple(sorted(rows, key=lambda row: (row.role.encode("utf-8"), row.path.encode("utf-8"))))


# The descriptor, rooted path, and immutable mapping are each a security boundary.
# pylint: disable-next=too-many-arguments
def _scan_directory(
    *,
    descriptor: int,
    relative_parts: tuple[str, ...],
    root: Path,
    mapping: Mapping[str, str],
    expected_directories: set[str],
    include_manifest: bool,
    anchor: DirectoryAnchor,
) -> list[ArtifactReference]:
    rows: list[ArtifactReference] = []
    anchor.verify()
    names = sorted(os.listdir(descriptor), key=lambda name: name.encode("utf-8"))
    for name in names:
        if name in {"", ".", ".."}:
            raise NativeArtifactError("native work payload has an unsafe directory entry")
        relative = "/".join((*relative_parts, name))
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            if relative not in expected_directories:
                raise NativeArtifactError("native work payload has an unallowlisted directory")
            child_descriptor: int | None = None
            try:
                child_descriptor = os.open(name, _directory_flags(), dir_fd=descriptor)
                opened = os.fstat(child_descriptor)
                if not _same_inode(metadata, opened) or not stat.S_ISDIR(opened.st_mode):
                    raise NativeArtifactError("native work payload directory identity changed")
                rows.extend(
                    _scan_directory(
                        descriptor=child_descriptor,
                        relative_parts=(*relative_parts, name),
                        root=root,
                        mapping=mapping,
                        expected_directories=expected_directories,
                        include_manifest=include_manifest,
                        anchor=anchor,
                    )
                )
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not _same_inode(metadata, current):
                    raise NativeArtifactError("native work payload directory binding changed")
            finally:
                if child_descriptor is not None:
                    os.close(child_descriptor)
        elif stat.S_ISREG(metadata.st_mode):
            if relative == MANIFEST_NAME and not include_manifest:
                continue
            if relative not in mapping:
                raise NativeArtifactError("native work payload has an unallowlisted file")
            rows.append(
                _hash_regular_file(
                    descriptor=descriptor,
                    name=name,
                    path=root / relative,
                    relative=relative,
                    expected_metadata=metadata,
                    role=mapping[relative],
                    anchor=anchor,
                )
            )
        else:
            raise NativeArtifactError("native work payload has a non-regular entry")
    anchor.verify()
    return rows


# The descriptor and its path binding are intentionally carried independently.
# pylint: disable-next=too-many-arguments
def _hash_regular_file(
    *,
    descriptor: int,
    name: str,
    path: Path,
    relative: str,
    expected_metadata: os.stat_result,
    role: str,
    anchor: DirectoryAnchor,
) -> ArtifactReference:
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(name, _regular_file_flags(), dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or not _same_inode(expected_metadata, before)
        ):
            raise NativeArtifactError("native work payload file identity changed")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if _SCAN_CHUNK_HOOK is not None:
                _SCAN_CHUNK_HOOK(path)
        after = os.fstat(file_descriptor)
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        anchor.verify()
        if (
            not _same_file_snapshot(before, after)
            or size != before.st_size
            or not _same_inode(expected_metadata, current)
        ):
            raise NativeArtifactError("native work payload file changed while being read")
        return ArtifactReference(path=relative, role=role, sha256=digest.hexdigest(), size=size)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _canonical_object(content: bytes) -> dict[str, JsonValue]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise NativeArtifactError("native work manifest must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, dict):
        raise NativeArtifactError("native work manifest must be an object")
    return value


def _absolute_path(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise NativeArtifactError(f"{label} path must be a Path")
    return Path(os.path.abspath(os.fspath(path)))


def _phase_files(phase: str) -> Mapping[str, str]:
    if phase not in PHASE_FILES or not PHASE_FILES[phase]:
        raise NativeArtifactError("native work phase does not have a successful allowlist")
    return PHASE_FILES[phase]


def _phase_host_bundle_path(phase: str) -> str:
    try:
        return PHASE_HOST_BUNDLE_PATHS[phase]
    except KeyError:
        raise NativeArtifactError("native work phase has no host attestation bundle") from None


def _expected_directories(mapping: Mapping[str, str]) -> set[str]:
    directories: set[str] = set()
    for path in mapping:
        parts = path.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))
    return directories


def _safe_relative_path(path: str) -> bool:
    return (
        bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


def _is_self_reference(path: str) -> bool:
    return path == MANIFEST_NAME or (
        path.startswith("hpa320-native-")
        and (path.endswith(".tar") or path.endswith(".sigstore.json"))
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_root_binding(root: Path, anchor: DirectoryAnchor) -> None:
    anchor.verify()
    metadata = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or not _same_inode(metadata, anchor.metadata):
        raise NativeArtifactError("native work payload root binding changed")


def _require_manifest_absent(anchor: DirectoryAnchor) -> None:
    try:
        os.stat(MANIFEST_NAME, dir_fd=anchor.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise NativeArtifactError("native work manifest destination already exists")


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("no-follow directory descriptors are unavailable")
    return os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)


def _regular_file_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("no-follow regular-file descriptors are unavailable")
    return os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _same_file_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        _same_inode(first, second)
        and first.st_mode == second.st_mode
        and first.st_nlink == second.st_nlink
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_ctime_ns == second.st_ctime_ns
    )


def pack_native_work_archive(
    *,
    phase: str,
    payload_root: Path,
    manifest_path: Path,
    archive_path: Path,
) -> CheckpointIdentity:
    """Write, self-verify, and no-replace publish the canonical ustar archive."""

    temporary_name: str | None = None
    temporary_stat: os.stat_result | None = None
    temporary_descriptor: int | None = None
    published_stat: os.stat_result | None = None
    published = False
    try:
        archive = _absolute_path(archive_path, "archive")
        _require_leaf_path(archive, "archive")
        root, manifest, members = _archive_recipe(
            phase=phase,
            payload_root=payload_root,
            manifest_path=manifest_path,
        )
        with open_directory_anchor(archive.parent) as parent:
            parent.verify()
            _require_absent_at(parent, archive.name, "archive destination")
            temporary_name = f".{archive.name}.{uuid4().hex}.tmp"
            temporary_descriptor = os.open(
                temporary_name,
                _new_regular_file_flags(),
                0o600,
                dir_fd=parent.descriptor,
            )
            temporary_stat = _require_owned_regular_descriptor(temporary_descriptor)
            _write_canonical_archive(
                descriptor=temporary_descriptor,
                payload_root=root,
                members=members,
            )
            os.fsync(temporary_descriptor)
            temporary_stat = _require_owned_regular_descriptor(temporary_descriptor)
            parent.verify()

            verified_identity = verify_native_work_archive(
                phase=phase,
                payload_root=root,
                manifest_path=manifest.path,
                archive_path=archive.parent / temporary_name,
            )
            temporary_identity = _stream_descriptor_identity(
                temporary_descriptor,
                name=temporary_name,
                expected=temporary_stat,
            )
            if (temporary_identity.sha256, temporary_identity.size) != (
                verified_identity.sha256,
                verified_identity.size,
            ):
                raise NativeArtifactError("native work archive changed after strict verification")
            _verify_regular_file_binding_at(parent, temporary_name, temporary_stat)
            _rename_no_replace_syscall(
                source=temporary_name,
                destination=archive.name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
            )
            published = True
            published_stat = os.stat(
                archive.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            temporary_name = None
            parent.verify()
            os.fsync(parent.descriptor)
            parent.verify()
            _verify_regular_file_binding_at(parent, archive.name, temporary_stat)
            published_identity = _stream_descriptor_identity(
                temporary_descriptor,
                name=archive.name,
            )
            if (published_identity.sha256, published_identity.size) != (
                verified_identity.sha256,
                verified_identity.size,
            ):
                raise NativeArtifactError("published native work archive identity changed")
            published = False
            os.close(temporary_descriptor)
            temporary_descriptor = None
            return CheckpointIdentity(
                name=archive.name,
                sha256=published_identity.sha256,
                size=published_identity.size,
            )
    except NativeArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise NativeArtifactError("native work archive publication failed") from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            _remove_temporary_archive(archive_path, temporary_name)
        if published and published_stat is not None:
            _remove_linked_archive_if_owned(archive_path, published_stat)


def verify_native_work_archive(
    *,
    phase: str,
    payload_root: Path,
    manifest_path: Path,
    archive_path: Path,
) -> CheckpointIdentity:
    """Require the archive to be the byte-exact canonical packing."""

    try:
        archive = _absolute_path(archive_path, "archive")
        _require_leaf_path(archive, "archive")
        root, _manifest, members = _archive_recipe(
            phase=phase,
            payload_root=payload_root,
            manifest_path=manifest_path,
        )
        actual_identity = _verify_archive_structure(
            archive_path=archive,
            payload_root=root,
            members=members,
        )
        expected_identity = _repack_for_comparison(
            payload_root=root,
            members=members,
            parent=archive.parent,
        )
        if (actual_identity.sha256, actual_identity.size) != (
            expected_identity.sha256,
            expected_identity.size,
        ):
            raise NativeArtifactError("native work archive is not the canonical packing")
        return CheckpointIdentity(
            name=archive.name,
            sha256=actual_identity.sha256,
            size=actual_identity.size,
        )
    except NativeArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise NativeArtifactError("native work archive is invalid or unsafe") from error


def copy_attestation_bundle(*, source: Path, destination: Path) -> CheckpointIdentity:
    """Copy one no-follow regular Sigstore bundle to a previously absent path."""

    temporary_name: str | None = None
    temporary_stat: os.stat_result | None = None
    temporary_descriptor: int | None = None
    published_stat: os.stat_result | None = None
    published = False
    try:
        source_path = _absolute_path(source, "attestation source")
        destination_path = _absolute_path(destination, "attestation destination")
        _require_leaf_path(source_path, "attestation source")
        _require_leaf_path(destination_path, "attestation destination")
        with open_directory_anchor(destination_path.parent) as destination_parent:
            destination_parent.verify()
            _require_absent_at(destination_parent, destination_path.name, "attestation destination")
            temporary_name = f".{destination_path.name}.{uuid4().hex}.tmp"
            temporary_descriptor = os.open(
                temporary_name,
                _new_regular_file_flags(),
                0o600,
                dir_fd=destination_parent.descriptor,
            )
            temporary_stat = _require_owned_regular_descriptor(temporary_descriptor)
            source_identity = _copy_regular_file(
                source_path=source_path,
                destination_descriptor=temporary_descriptor,
            )
            os.fsync(temporary_descriptor)
            temporary_stat = _require_owned_regular_descriptor(temporary_descriptor)
            destination_parent.verify()
            temporary_identity = _stream_descriptor_identity(
                temporary_descriptor,
                name=temporary_name,
                expected=temporary_stat,
            )
            if (temporary_identity.sha256, temporary_identity.size) != (
                source_identity.sha256,
                source_identity.size,
            ):
                raise NativeArtifactError(
                    "attestation bundle temporary copy differs from its source"
                )
            _verify_regular_file_binding_at(
                destination_parent,
                temporary_name,
                temporary_stat,
            )
            _rename_no_replace_syscall(
                source=temporary_name,
                destination=destination_path.name,
                src_dir_fd=destination_parent.descriptor,
                dst_dir_fd=destination_parent.descriptor,
            )
            published = True
            published_stat = os.stat(
                destination_path.name,
                dir_fd=destination_parent.descriptor,
                follow_symlinks=False,
            )
            temporary_name = None
            destination_parent.verify()
            os.fsync(destination_parent.descriptor)
            destination_parent.verify()
            _verify_regular_file_binding_at(
                destination_parent,
                destination_path.name,
                temporary_stat,
            )
            copied_identity = _stream_descriptor_identity(
                temporary_descriptor,
                name=destination_path.name,
            )
            if (copied_identity.sha256, copied_identity.size) != (
                source_identity.sha256,
                source_identity.size,
            ):
                raise NativeArtifactError("attestation bundle destination identity changed")
            published = False
            os.close(temporary_descriptor)
            temporary_descriptor = None
            return CheckpointIdentity(
                name=destination_path.name,
                sha256=copied_identity.sha256,
                size=copied_identity.size,
            )
    except NativeArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise NativeArtifactError("attestation bundle copy failed") from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            _remove_temporary_archive(destination, temporary_name)
        if published and published_stat is not None:
            _remove_linked_archive_if_owned(destination, published_stat)


def _archive_recipe(
    *,
    phase: str,
    payload_root: Path,
    manifest_path: Path,
) -> tuple[Path, NativeWorkArtifactManifest, tuple[_ArchiveMember, ...]]:
    root = _absolute_path(payload_root, "payload root")
    manifest = load_native_work_manifest(manifest_path, expected_phase=phase)
    if manifest.path != root / MANIFEST_NAME:
        raise NativeArtifactError("native work manifest does not belong to payload root")
    verify_native_work_payload(payload_root=root, manifest=manifest)
    manifest_content = read_regular_file_no_follow(manifest.path, max_bytes=_MANIFEST_MAX_BYTES)
    manifest_identity = CheckpointIdentity(
        name=MANIFEST_NAME,
        sha256=hashlib.sha256(manifest_content).hexdigest(),
        size=len(manifest_content),
    )
    if manifest_identity.sha256 != manifest.sha256:
        raise NativeArtifactError("native work manifest bytes changed")
    root_member = _archive_root_name(phase=phase, workflow_commit=manifest.workflow_commit)
    directories = sorted(
        _expected_directories({row.path: row.role for row in manifest.files}),
        key=lambda path: path.encode("utf-8"),
    )
    members: list[_ArchiveMember] = [
        _ArchiveMember(
            name=root_member,
            mode=0o755,
            size=0,
            source_relative=None,
            source_identity=None,
            typeflag=b"5",
        )
    ]
    members.extend(
        _ArchiveMember(
            name=f"{root_member}{directory}/",
            mode=0o755,
            size=0,
            source_relative=None,
            source_identity=None,
            typeflag=b"5",
        )
        for directory in directories
    )
    source_members = [
        _ArchiveMember(
            name=f"{root_member}{MANIFEST_NAME}",
            mode=0o644,
            size=manifest_identity.size,
            source_relative=MANIFEST_NAME,
            source_identity=manifest_identity,
            typeflag=b"0",
        )
    ]
    source_members.extend(
        _ArchiveMember(
            name=f"{root_member}{row.path}",
            mode=0o644,
            size=row.size,
            source_relative=row.path,
            source_identity=CheckpointIdentity(
                name=row.path,
                sha256=row.sha256,
                size=row.size,
            ),
            typeflag=b"0",
        )
        for row in manifest.files
    )
    members.extend(sorted(source_members, key=lambda member: member.name.encode("utf-8")))
    return root, manifest, tuple(members)


def _archive_root_name(*, phase: str, workflow_commit: str) -> str:
    if (
        phase not in PHASE_FILES
        or len(workflow_commit) != 40
        or any(character not in "0123456789abcdef" for character in workflow_commit)
    ):
        raise NativeArtifactError("native work archive root identity is invalid")
    return f"hpa320-native-{phase}-{workflow_commit}/"


def _write_canonical_archive(
    *,
    descriptor: int,
    payload_root: Path,
    members: tuple[_ArchiveMember, ...],
) -> None:
    with open_directory_anchor(payload_root) as payload_anchor:
        for member in members:
            _write_all(
                descriptor,
                _ustar_header(
                    name=_utf8_member_name(member.name),
                    mode=member.mode,
                    size=member.size,
                    typeflag=member.typeflag,
                ),
            )
            if member.source_relative is None:
                continue
            if member.source_identity is None:
                raise NativeArtifactError("native work archive member source is missing")
            _write_payload_member(
                archive_descriptor=descriptor,
                payload_root=payload_root,
                payload_anchor=payload_anchor,
                relative=member.source_relative,
                expected=member.source_identity,
            )
            padding = (-member.size) % 512
            if padding:
                _write_all(descriptor, b"\0" * padding)
        _write_all(descriptor, b"\0" * 1024)


def _write_payload_member(
    *,
    archive_descriptor: int,
    payload_root: Path,
    payload_anchor: DirectoryAnchor,
    relative: str,
    expected: CheckpointIdentity,
) -> None:
    digest = hashlib.sha256()
    size = 0
    with _open_payload_regular_file(
        payload_root=payload_root,
        payload_anchor=payload_anchor,
        relative=relative,
    ) as descriptor:
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            _write_all(archive_descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
            if _ARCHIVE_CHUNK_HOOK is not None:
                _ARCHIVE_CHUNK_HOOK(payload_root / relative)
    if (digest.hexdigest(), size) != (expected.sha256, expected.size):
        raise NativeArtifactError("native work archive member differs from its manifest identity")


def _verify_archive_structure(
    *,
    archive_path: Path,
    payload_root: Path,
    members: tuple[_ArchiveMember, ...],
) -> CheckpointIdentity:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(archive_path) as archive_descriptor:
        with open_directory_anchor(payload_root) as payload_anchor:
            for member in members:
                header = _read_exact(archive_descriptor, 512)
                _parse_ustar_header(header)
                expected_header = _ustar_header(
                    name=_utf8_member_name(member.name),
                    mode=member.mode,
                    size=member.size,
                    typeflag=member.typeflag,
                )
                if header != expected_header:
                    raise NativeArtifactError("native work archive member header is not canonical")
                digest.update(header)
                size += len(header)
                if member.source_relative is None:
                    continue
                if member.source_identity is None:
                    raise NativeArtifactError("native work archive member source is missing")
                member_digest, member_size = _compare_archive_member_to_payload(
                    archive_descriptor=archive_descriptor,
                    payload_root=payload_root,
                    payload_anchor=payload_anchor,
                    relative=member.source_relative,
                    expected=member.source_identity,
                    archive_digest=digest,
                )
                if (member_digest, member_size) != (
                    member.source_identity.sha256,
                    member.source_identity.size,
                ):
                    raise NativeArtifactError("native work archive member differs from its payload")
                size += member_size
                padding = (-member.size) % 512
                if padding:
                    padding_bytes = _read_exact(archive_descriptor, padding)
                    if padding_bytes != b"\0" * padding:
                        raise NativeArtifactError("native work archive member padding is invalid")
                    digest.update(padding_bytes)
                    size += len(padding_bytes)
            terminal = _read_exact(archive_descriptor, 1024)
            if terminal != b"\0" * 1024:
                raise NativeArtifactError("native work archive terminal blocks are invalid")
            if os.read(archive_descriptor, 1):
                raise NativeArtifactError("native work archive has trailing bytes")
            digest.update(terminal)
            size += len(terminal)
    return CheckpointIdentity(name=archive_path.name, sha256=digest.hexdigest(), size=size)


def _compare_archive_member_to_payload(
    *,
    archive_descriptor: int,
    payload_root: Path,
    payload_anchor: DirectoryAnchor,
    relative: str,
    expected: CheckpointIdentity,
    archive_digest: hashlib._Hash,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    remaining = expected.size
    with _open_payload_regular_file(
        payload_root=payload_root,
        payload_anchor=payload_anchor,
        relative=relative,
    ) as payload_descriptor:
        while remaining:
            count = min(_CHUNK_SIZE, remaining)
            archive_chunk = _read_exact(archive_descriptor, count)
            payload_chunk = _read_exact(payload_descriptor, count)
            if archive_chunk != payload_chunk:
                raise NativeArtifactError("native work archive body differs from payload")
            digest.update(payload_chunk)
            archive_digest.update(archive_chunk)
            remaining -= count
        if os.read(payload_descriptor, 1):
            raise NativeArtifactError("native work payload member size changed")
    return digest.hexdigest(), expected.size


def _repack_for_comparison(
    *,
    payload_root: Path,
    members: tuple[_ArchiveMember, ...],
    parent: Path,
) -> CheckpointIdentity:
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        with open_directory_anchor(parent) as parent_anchor:
            temporary_name = f".native-work-repack.{uuid4().hex}.tmp"
            descriptor = os.open(
                temporary_name,
                _new_regular_file_flags(),
                0o600,
                dir_fd=parent_anchor.descriptor,
            )
            _require_owned_regular_descriptor(descriptor)
            _write_canonical_archive(
                descriptor=descriptor,
                payload_root=payload_root,
                members=members,
            )
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            identity = _stream_regular_file_identity(parent / temporary_name)
            parent_anchor.verify()
            return identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            _remove_temporary_archive(parent / "unused", temporary_name)


def _stream_regular_file_identity(path: Path) -> CheckpointIdentity:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(path) as descriptor:
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return CheckpointIdentity(name=path.name, sha256=digest.hexdigest(), size=size)


def _stream_descriptor_identity(
    descriptor: int,
    *,
    name: str,
    expected: os.stat_result | None = None,
) -> CheckpointIdentity:
    before = _require_owned_regular_descriptor(descriptor)
    if expected is not None and not _same_file_snapshot(before, expected):
        raise NativeArtifactError("native work archive temporary inode changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, _CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = _require_owned_regular_descriptor(descriptor)
    if not _same_file_snapshot(before, after):
        raise NativeArtifactError("native work archive descriptor changed while it was read")
    return CheckpointIdentity(name=name, sha256=digest.hexdigest(), size=size)


def _copy_regular_file(
    *,
    source_path: Path,
    destination_descriptor: int,
) -> CheckpointIdentity:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(source_path) as source_descriptor:
        while True:
            chunk = os.read(source_descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            _write_all(destination_descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
    return CheckpointIdentity(name=source_path.name, sha256=digest.hexdigest(), size=size)


@contextmanager
def _open_regular_file(path: Path) -> Iterator[int]:
    absolute = _absolute_path(path, "regular file")
    _require_leaf_path(absolute, "regular file")
    descriptor: int | None = None
    with open_directory_anchor(absolute.parent) as parent:
        try:
            parent.verify()
            expected = os.stat(absolute.name, dir_fd=parent.descriptor, follow_symlinks=False)
            descriptor = os.open(
                absolute.name,
                _regular_file_flags(),
                dir_fd=parent.descriptor,
            )
            before = _require_owned_regular_descriptor(descriptor)
            if not _same_inode(expected, before):
                raise NativeArtifactError("regular file identity changed before it was opened")
            yield descriptor
            after = os.fstat(descriptor)
            current = os.stat(absolute.name, dir_fd=parent.descriptor, follow_symlinks=False)
            parent.verify()
            if (
                not _same_file_snapshot(before, after)
                or not _same_inode(expected, current)
                or not stat.S_ISREG(current.st_mode)
            ):
                raise NativeArtifactError("regular file changed while it was read")
        finally:
            if descriptor is not None:
                os.close(descriptor)


@contextmanager
def _open_payload_regular_file(
    *,
    payload_root: Path,
    payload_anchor: DirectoryAnchor,
    relative: str,
) -> Iterator[int]:
    if not _safe_relative_path(relative):
        raise NativeArtifactError("native work archive member path is unsafe")
    directory_descriptor = os.dup(payload_anchor.descriptor)
    owned_descriptors: list[int] = [directory_descriptor]
    descriptor: int | None = None
    bindings: list[tuple[int, str, os.stat_result]] = []
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            expected_directory = os.stat(
                part,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(expected_directory.st_mode):
                raise NativeArtifactError("native work archive directory is unsafe")
            child = os.open(part, _directory_flags(), dir_fd=directory_descriptor)
            opened_directory = os.fstat(child)
            if not stat.S_ISDIR(opened_directory.st_mode) or not _same_inode(
                expected_directory, opened_directory
            ):
                os.close(child)
                raise NativeArtifactError("native work archive directory identity changed")
            bindings.append((directory_descriptor, part, expected_directory))
            owned_descriptors.append(child)
            directory_descriptor = child
        name = parts[-1]
        expected = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        descriptor = os.open(name, _regular_file_flags(), dir_fd=directory_descriptor)
        before = _require_owned_regular_descriptor(descriptor)
        if not _same_inode(expected, before):
            raise NativeArtifactError("native work archive member identity changed before reading")
        yield descriptor
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        payload_anchor.verify()
        _verify_directory_bindings(bindings)
        if (
            not _same_file_snapshot(before, after)
            or not _same_inode(expected, current)
            or not stat.S_ISREG(current.st_mode)
        ):
            raise NativeArtifactError("native work archive member changed while it was read")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for owned_descriptor in reversed(owned_descriptors):
            os.close(owned_descriptor)


def _verify_directory_bindings(bindings: list[tuple[int, str, os.stat_result]]) -> None:
    for parent_descriptor, name, expected in bindings:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or not _same_inode(expected, current):
            raise NativeArtifactError("native work archive directory binding changed")


def _ustar_header(*, name: bytes, mode: int, size: int, typeflag: bytes) -> bytes:
    prefix, leaf = _split_ustar_path(name)
    if typeflag not in {b"0", b"5"}:
        raise NativeArtifactError("native work archive member type is invalid")
    block = bytearray(512)
    _write_bytes(block, 0, 100, leaf)
    _write_octal(block, 100, 8, mode)
    _write_octal(block, 108, 8, 0)
    _write_octal(block, 116, 8, 0)
    _write_octal(block, 124, 12, size)
    _write_octal(block, 136, 12, 0)
    block[148:156] = b"        "
    block[156:157] = typeflag
    block[257:263] = b"ustar\0"
    block[263:265] = b"00"
    _write_bytes(block, 345, 155, prefix)
    checksum = sum(block)
    block[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    return bytes(block)


def _split_ustar_path(name: bytes) -> tuple[bytes, bytes]:
    if not name or b"\0" in name:
        raise NativeArtifactError("native work archive member name is invalid")
    try:
        decoded = name.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NativeArtifactError("native work archive member name is not UTF-8") from error
    if decoded.encode("utf-8") != name:
        raise NativeArtifactError("native work archive member name is not stable UTF-8")
    if len(name) <= 100:
        return b"", name
    separators = [index for index, value in enumerate(name) if value == ord("/")]
    for separator in reversed(separators):
        prefix = name[:separator]
        leaf = name[separator + 1 :]
        if prefix and leaf and len(prefix) <= 155 and len(leaf) <= 100:
            return prefix, leaf
    raise NativeArtifactError("native work archive member name does not fit POSIX ustar")


def _utf8_member_name(name: str) -> bytes:
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NativeArtifactError("native work archive member name is not UTF-8") from error
    if b"\0" in encoded:
        raise NativeArtifactError("native work archive member name is invalid")
    return encoded


def _write_bytes(block: bytearray, offset: int, width: int, value: bytes) -> None:
    if len(value) > width or b"\0" in value:
        raise NativeArtifactError("native work archive field is invalid")
    block[offset : offset + width] = value.ljust(width, b"\0")


def _write_octal(block: bytearray, offset: int, width: int, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NativeArtifactError("native work archive numeric field is invalid")
    maximum = (8 ** (width - 1)) - 1
    if value > maximum:
        raise NativeArtifactError("native work archive numeric field exceeds POSIX ustar")
    block[offset : offset + width] = f"{value:0{width - 1}o}\0".encode("ascii")


def _parse_ustar_header(block: bytes) -> None:
    if len(block) != 512 or block == b"\0" * 512:
        raise NativeArtifactError("native work archive header is incomplete")
    checksum_field = block[148:156]
    if (
        len(checksum_field) != 8
        or checksum_field[6:] != b"\0 "
        or any(value not in b"01234567" for value in checksum_field[:6])
    ):
        raise NativeArtifactError("native work archive checksum field is invalid")
    expected_checksum = sum(block[:148] + b" " * 8 + block[156:])
    if int(checksum_field[:6], 8) != expected_checksum:
        raise NativeArtifactError("native work archive checksum is invalid")
    name = _read_ustar_string(block[0:100], "member name")
    prefix = _read_ustar_string(block[345:500], "member prefix")
    if not name:
        raise NativeArtifactError("native work archive member name is empty")
    try:
        (prefix + (b"/" if prefix else b"") + name).decode("utf-8")
    except UnicodeDecodeError as error:
        raise NativeArtifactError("native work archive member name is not UTF-8") from error
    _read_ustar_octal(block[100:108], "mode")
    _read_ustar_octal(block[108:116], "uid")
    _read_ustar_octal(block[116:124], "gid")
    _read_ustar_octal(block[124:136], "size")
    _read_ustar_octal(block[136:148], "mtime")
    if block[156:157] not in {b"0", b"5"}:
        raise NativeArtifactError("native work archive member type is invalid")
    if block[157:257] != b"\0" * 100:
        raise NativeArtifactError("native work archive member link target is invalid")
    if block[257:263] != b"ustar\0" or block[263:265] != b"00":
        raise NativeArtifactError("native work archive format is not POSIX ustar")
    if block[265:345] != b"\0" * 80:
        raise NativeArtifactError("native work archive owner metadata is invalid")
    if block[329:345] != b"\0" * 16 or block[500:] != b"\0" * 12:
        raise NativeArtifactError("native work archive device metadata is invalid")


def _read_ustar_string(value: bytes, label: str) -> bytes:
    try:
        end = value.index(b"\0")
    except ValueError:
        return value
    if value[end + 1 :] != b"\0" * (len(value) - end - 1):
        raise NativeArtifactError(f"native work archive {label} has trailing data")
    return value[:end]


def _read_ustar_octal(value: bytes, label: str) -> int:
    if len(value) < 2 or value[-1:] != b"\0" or any(byte not in b"01234567" for byte in value[:-1]):
        raise NativeArtifactError(f"native work archive {label} field is invalid")
    return int(value[:-1], 8)


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise NativeArtifactError("native work archive is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("native work archive write was incomplete")
        offset += written


def _require_owned_regular_descriptor(descriptor: int) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise NativeArtifactError("native work archive file is not a private regular file")
    return metadata


def _require_leaf_path(path: Path, label: str) -> None:
    if path.name in {"", ".", ".."} or path.parent == path:
        raise NativeArtifactError(f"{label} must name a regular file")


def _require_absent_at(anchor: DirectoryAnchor, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=anchor.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise NativeArtifactError(f"{label} already exists")


def _new_regular_file_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("no-follow file creation is unavailable")
    return os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0)


def _verify_regular_file_binding_at(
    anchor: DirectoryAnchor,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=anchor.descriptor, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or not _same_inode(current, expected):
        raise NativeArtifactError("native work archive path binding changed")


def _unlink_at(descriptor: int, name: str) -> None:
    os.unlink(name, dir_fd=descriptor)


def _remove_temporary_archive(path: Path, name: str) -> None:
    try:
        absolute = _absolute_path(path, "archive")
        with open_directory_anchor(absolute.parent) as parent:
            os.unlink(name, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
    except (OSError, NativeArtifactError):
        pass


def _remove_directory_tree_at(
    *,
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or not _same_inode(expected, opened)
            or not _same_inode(expected, current)
        ):
            raise NativeArtifactError("native work archive cleanup directory binding changed")
        for child_name in os.listdir(descriptor):
            if child_name in {"", ".", ".."}:
                raise NativeArtifactError(
                    "native work archive cleanup found an unsafe directory entry"
                )
            child = os.stat(child_name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(child.st_mode):
                _remove_directory_tree_at(
                    parent_descriptor=descriptor,
                    name=child_name,
                    expected=child,
                )
            else:
                current_child = os.stat(
                    child_name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(child, current_child):
                    raise NativeArtifactError("native work archive cleanup entry binding changed")
                os.unlink(child_name, dir_fd=descriptor)
        os.fsync(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or not _same_inode(expected, current):
            raise NativeArtifactError("native work archive cleanup directory binding changed")
        os.rmdir(name, dir_fd=parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_linked_archive_if_owned(path: Path, expected: os.stat_result) -> None:
    try:
        absolute = _absolute_path(path, "archive")
        with open_directory_anchor(absolute.parent) as parent:
            current = os.stat(absolute.name, dir_fd=parent.descriptor, follow_symlinks=False)
            if _same_inode(current, expected):
                if stat.S_ISDIR(current.st_mode):
                    _remove_directory_tree_at(
                        parent_descriptor=parent.descriptor,
                        name=absolute.name,
                        expected=current,
                    )
                else:
                    os.unlink(absolute.name, dir_fd=parent.descriptor)
                os.fsync(parent.descriptor)
    except (FileNotFoundError, OSError, NativeArtifactError):
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--phase", required=True, choices=tuple(PHASE_FILES))
    publish_parser.add_argument("--payload-root", required=True, type=Path)
    publish_parser.add_argument("--host-bundle", required=True, type=Path)
    publish_parser.add_argument("--archive", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--phase", required=True, choices=tuple(PHASE_FILES))
    verify_parser.add_argument("--payload-root", required=True, type=Path)
    verify_parser.add_argument("--archive", required=True, type=Path)
    verify_parser.add_argument("--bundle", type=Path)
    verify_parser.add_argument("--expected-bundle-sha256")
    verify_parser.add_argument("--expected-bundle-size", type=int)
    verify_parser.add_argument("--expected-bundle-identity", type=Path)
    attestation_parser = subparsers.add_parser("verify-attestation")
    attestation_parser.add_argument(
        "--phase", required=True, choices=tuple(PHASE_HOST_BUNDLE_PATHS)
    )
    attestation_parser.add_argument("--workflow-commit", required=True)
    attestation_parser.add_argument("--payload-root", required=True, type=Path)
    attestation_parser.add_argument("--archive", required=True, type=Path)
    attestation_parser.add_argument("--bundle", required=True, type=Path)
    attestation_parser.add_argument("--trusted-root", required=True, type=Path)
    attestation_parser.add_argument("--repository-root", required=True, type=Path)
    copy_parser = subparsers.add_parser("copy-bundle")
    copy_parser.add_argument("--source", required=True, type=Path)
    copy_parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "publish":
            manifest = publish_native_work_manifest(
                phase=arguments.phase,
                payload_root=arguments.payload_root,
                host_bundle_path=arguments.host_bundle,
            )
            pack_native_work_archive(
                phase=arguments.phase,
                payload_root=arguments.payload_root,
                manifest_path=manifest.path,
                archive_path=arguments.archive,
            )
            verify_native_work_archive(
                phase=arguments.phase,
                payload_root=arguments.payload_root,
                manifest_path=manifest.path,
                archive_path=arguments.archive,
            )
        elif arguments.command == "verify":
            manifest = load_native_work_manifest(
                arguments.payload_root / MANIFEST_NAME,
                expected_phase=arguments.phase,
            )
            verify_native_work_payload(payload_root=arguments.payload_root, manifest=manifest)
            verify_native_work_archive(
                phase=arguments.phase,
                payload_root=arguments.payload_root,
                manifest_path=manifest.path,
                archive_path=arguments.archive,
            )
            if arguments.bundle is not None:
                expected_sha256, expected_size = _resolve_expected_bundle_identity(arguments)
                _verify_optional_attestation_bundle(
                    arguments.bundle,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                )
        elif arguments.command == "verify-attestation":
            report: bytes
            with _attestation_snapshot(
                phase=arguments.phase,
                payload_root=arguments.payload_root,
                archive_path=arguments.archive,
                sigstore_bundle_path=arguments.bundle,
                trusted_root_path=arguments.trusted_root,
            ) as snapshot:
                verified = verify_github_attestations(
                    phase=arguments.phase,
                    workflow_commit=arguments.workflow_commit,
                    manifest_path=snapshot.manifest_path,
                    archive_path=snapshot.archive_path,
                    sigstore_bundle_path=snapshot.bundle_path,
                    trusted_root_path=snapshot.trusted_root_path,
                )
                _require_snapshot_subjects(
                    phase=arguments.phase,
                    workflow_commit=arguments.workflow_commit,
                    snapshot=snapshot,
                    verified=verified,
                )
                load_native_work_manifest(snapshot.manifest_path, expected_phase=arguments.phase)
                archive = verify_native_work_archive(
                    phase=arguments.phase,
                    payload_root=snapshot.payload_root,
                    manifest_path=snapshot.manifest_path,
                    archive_path=snapshot.archive_path,
                )
                _require_verified_archive_identity(archive=archive, verified=verified)
                _validate_native_work_phase(
                    phase=arguments.phase,
                    payload_root=snapshot.payload_root,
                    repository_root=arguments.repository_root,
                )
                _require_live_snapshot_bindings(snapshot)
                manifest = load_native_work_manifest(
                    snapshot.manifest_path,
                    expected_phase=arguments.phase,
                )
                verify_native_work_payload(
                    payload_root=snapshot.payload_root,
                    manifest=manifest,
                )
                archive = verify_native_work_archive(
                    phase=arguments.phase,
                    payload_root=snapshot.payload_root,
                    manifest_path=snapshot.manifest_path,
                    archive_path=snapshot.archive_path,
                )
                _require_verified_archive_identity(archive=archive, verified=verified)
                _require_snapshot_subjects(
                    phase=arguments.phase,
                    workflow_commit=arguments.workflow_commit,
                    snapshot=snapshot,
                    verified=verified,
                )
                report = _attestation_review_report(arguments, verified, snapshot)
            sys.stdout.buffer.write(report)
        elif arguments.command == "copy-bundle":
            identity = copy_attestation_bundle(
                source=arguments.source, destination=arguments.destination
            )
            sys.stdout.buffer.write(
                canonical_json_bytes(
                    {
                        "destination": os.fspath(arguments.destination),
                        "sha256": identity.sha256,
                        "size": identity.size,
                        "status": "copied",
                    },
                    trailing_newline=True,
                )
            )
        else:
            raise NativeArtifactError("native work archive command is invalid")
    except NativeArtifactError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


def _validate_native_work_phase(
    *,
    phase: str,
    payload_root: Path,
    repository_root: Path,
) -> None:
    try:
        from tools.hpa320.seal_oaf_backend import validate_native_work_phase

        validate_native_work_phase(
            phase=phase,
            payload_root=payload_root,
            repository_root=repository_root,
        )
    except (ValueError, ImportError) as error:
        raise NativeArtifactError("native work phase acceptance failed") from error


def _require_snapshot_subjects(
    *,
    phase: str,
    workflow_commit: str,
    snapshot: _AttestationSnapshot,
    verified: VerifiedGitHubAttestation,
) -> None:
    expected_names = _attestation_subject_names(
        phase=phase,
        workflow_commit=workflow_commit,
        manifest_path=snapshot.manifest_path,
        archive_path=snapshot.archive_path,
    )
    _require_live_snapshot_bindings(snapshot)
    subjects = _snapshot_subjects(
        phase=phase,
        workflow_commit=workflow_commit,
        snapshot=snapshot,
    )
    _require_live_snapshot_bindings(snapshot)
    if (
        tuple(subject.name for subject in subjects) != expected_names
        or subjects != verified.subjects
    ):
        raise NativeArtifactError("native work attestation snapshot differs from verified subjects")


def _snapshot_subjects(
    *,
    phase: str,
    workflow_commit: str,
    snapshot: _AttestationSnapshot,
) -> tuple[CheckpointIdentity, ...]:
    manifest_name, archive_name = _attestation_subject_names(
        phase=phase,
        workflow_commit=workflow_commit,
        manifest_path=snapshot.manifest_path,
        archive_path=snapshot.archive_path,
    )
    manifest_identity = _stream_regular_file_identity(snapshot.manifest_path)
    archive_identity = _stream_regular_file_identity(snapshot.archive_path)
    return (
        CheckpointIdentity(
            name=manifest_name,
            sha256=manifest_identity.sha256,
            size=manifest_identity.size,
        ),
        CheckpointIdentity(
            name=archive_name,
            sha256=archive_identity.sha256,
            size=archive_identity.size,
        ),
    )


def _require_verified_archive_identity(
    *,
    archive: CheckpointIdentity,
    verified: VerifiedGitHubAttestation,
) -> None:
    expected = verified.subjects[1]
    if (archive.sha256, archive.size) != (expected.sha256, expected.size):
        raise NativeArtifactError("native work archive differs from the verified subject")


def _attestation_review_report(
    arguments: argparse.Namespace,
    verified: VerifiedGitHubAttestation,
    snapshot: _AttestationSnapshot,
) -> bytes:
    report: JsonValue = {
        "arguments": {
            "archive": str(arguments.archive),
            "bundle": str(arguments.bundle),
            "payload_root": str(arguments.payload_root),
            "phase": arguments.phase,
            "repository_root": str(arguments.repository_root),
            "trusted_root": str(arguments.trusted_root),
            "workflow_commit": arguments.workflow_commit,
        },
        "gh_version": verified.gh_version,
        "sigstore_bundle": {
            "sha256": snapshot.bundle_identity.sha256,
            "size": snapshot.bundle_identity.size,
        },
        "statement_sha256": verified.statement_sha256,
        "subjects": [
            {"name": subject.name, "sha256": subject.sha256, "size": subject.size}
            for subject in verified.subjects
        ],
        "trusted_root": {
            "sha256": snapshot.trusted_root_identity.sha256,
            "size": snapshot.trusted_root_identity.size,
        },
    }
    return canonical_json_bytes(report, trailing_newline=True)


def _verify_optional_attestation_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    try:
        identity = _stream_regular_file_identity(path)
    except OSError as error:
        raise NativeArtifactError("native work attestation bundle is missing or unsafe") from error
    if identity.sha256 != expected_sha256 or identity.size != expected_size:
        raise NativeArtifactError("native work attestation bundle identity does not match")


def _resolve_expected_bundle_identity(arguments: argparse.Namespace) -> tuple[str, int]:
    """Select the expected sigstore bundle identity from exactly one source.

    The identity may arrive either as an explicit sha256/size pair or as a JSON
    file emitted by ``copy-bundle``. Requiring exactly one source keeps the
    trust flow unambiguous and avoids ``$GITHUB_OUTPUT`` step outputs.
    """

    has_explicit = (
        arguments.expected_bundle_sha256 is not None and arguments.expected_bundle_size is not None
    )
    has_identity_file = arguments.expected_bundle_identity is not None
    if has_explicit and has_identity_file:
        raise NativeArtifactError("native work attestation bundle identity is ambiguous")
    if has_identity_file:
        return _load_expected_bundle_identity(arguments.expected_bundle_identity)
    if has_explicit:
        return arguments.expected_bundle_sha256, arguments.expected_bundle_size
    raise NativeArtifactError("native work attestation bundle identity is required")


def _load_expected_bundle_identity(path: Path) -> tuple[str, int]:
    try:
        content = read_regular_file_no_follow(path)
    except OSError as error:
        raise NativeArtifactError(
            "native work attestation bundle identity is missing or unsafe"
        ) from error
    try:
        value = strict_json_loads(content)
    except StrictJsonError as error:
        raise NativeArtifactError("native work attestation bundle identity is invalid") from error
    if not isinstance(value, Mapping):
        raise NativeArtifactError("native work attestation bundle identity is invalid")
    sha256 = value.get("sha256")
    size = value.get("size")
    if not isinstance(sha256, str) or not isinstance(size, int) or isinstance(size, bool):
        raise NativeArtifactError("native work attestation bundle identity is invalid")
    return sha256, size


if __name__ == "__main__":
    raise SystemExit(main())
