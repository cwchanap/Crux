"""Strict, immutable manifests for same-job OaF native work artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    DirectoryAnchor,
    open_directory_anchor,
    publish_immutable_bytes,
    read_regular_file_no_follow,
)
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
assert len(CANDIDATE_FILES) == 24

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


class NativeArtifactError(ValueError):
    """The native work-artifact payload or manifest is unsafe or inconsistent."""


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
