from __future__ import annotations

# The public records/signature are fixed by the v1 contract. Exact ``type`` checks
# intentionally reject booleans where JSON integers are required.
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-branches,unidiomatic-typecheck
import os
import platform
import stat
import subprocess
import unicodedata
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal, cast
from uuid import UUID, uuid4

from src.benchmark.backend_identity import (
    JsonValue,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_publication import publish_immutable_bytes
from src.benchmark.backends import PublishedArtifact

_ATTESTATION_SCHEMA = "crux.backend-execution-attestation/v1"
_HOST_NUMERIC_FINGERPRINT_KEYS = frozenset(
    {"architecture", "cpu_family", "cpu_model", "cpu_stepping", "cpu_vendor_id"}
)
_LINUX_CPUINFO_REQUIRED_FIELDS = frozenset(
    {"processor", "vendor_id", "cpu family", "model", "stepping"}
)
_SOURCE_MANIFEST_KEYS = frozenset({"schema", "covered_roots", "files"})
_SOURCE_FILE_KEYS = frozenset({"path", "sha256", "license"})
_CHANGED_FILE_KEYS = frozenset({"path", "sha256", "status"})
_HEURISTIC_BACKEND_ID = "heuristic-onset-v1"
_ATTESTATION_KEYS = frozenset(
    {
        "backend_id",
        "changed_files_manifest",
        "checkout_dirty",
        "cpu_limit",
        "descriptor_sha256",
        "git_commit",
        "host_numeric_fingerprint",
        "memory_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "schema",
        "shm_bytes",
        "startup_deadline_seconds",
        "strict_mode",
        "tmp_bytes",
    }
)


class AttestationError(ValueError):
    pass


@dataclass(frozen=True)
class HostNumericFingerprint:
    architecture: str
    cpu_vendor_id: str
    cpu_family: str
    cpu_model: str
    cpu_stepping: str

    def __post_init__(self) -> None:
        for field in _HOST_NUMERIC_FINGERPRINT_KEYS:
            _require_nonempty_string(getattr(self, field), f"host numeric fingerprint {field}")

    def as_json(self) -> dict[str, str]:
        return {
            "architecture": self.architecture,
            "cpu_vendor_id": self.cpu_vendor_id,
            "cpu_family": self.cpu_family,
            "cpu_model": self.cpu_model,
            "cpu_stepping": self.cpu_stepping,
        }


def parse_host_numeric_fingerprint(value: object) -> HostNumericFingerprint:
    if not isinstance(value, dict) or set(value) != _HOST_NUMERIC_FINGERPRINT_KEYS:
        raise AttestationError("host numeric fingerprint must contain the exact key set")
    if any(not isinstance(field_value, str) for field_value in value.values()):
        raise AttestationError("host numeric fingerprint fields must be strings")
    return HostNumericFingerprint(**cast(dict[str, str], value))


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: Literal["modified", "deleted", "untracked"]
    sha256: str | None

    def __post_init__(self) -> None:
        _validate_repository_path(self.path)
        if self.status not in {"modified", "deleted", "untracked"}:
            raise AttestationError("changed-file status is invalid")
        if self.status == "deleted":
            if self.sha256 is not None:
                raise AttestationError("deleted changed-file sha256 must be null")
        elif self.sha256 is None:
            raise AttestationError("present changed-file sha256 is required")
        else:
            _require_hash(self.sha256, "changed-file sha256")


@dataclass(frozen=True)
class ExecutionConditions:
    cpu_limit: str | None
    memory_bytes: int | None
    pid_limit: int | None
    tmp_bytes: int | None
    shm_bytes: int | None
    startup_deadline_seconds: int
    request_deadline_seconds: int

    def __post_init__(self) -> None:
        if self.cpu_limit is not None:
            if not isinstance(self.cpu_limit, str) or not self.cpu_limit:
                raise AttestationError("cpu_limit must be a positive string or null")
            try:
                positive_cpu = Decimal(self.cpu_limit)
            except InvalidOperation:
                raise AttestationError("cpu_limit must be positive") from None
            if not positive_cpu.is_finite() or positive_cpu <= 0:
                raise AttestationError("cpu_limit must be positive")
        for field in ("memory_bytes", "pid_limit", "tmp_bytes", "shm_bytes"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise AttestationError(f"{field} must be positive or null")
        for field in ("startup_deadline_seconds", "request_deadline_seconds"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise AttestationError(f"{field} must be positive")


@dataclass(frozen=True)
class ExecutionAttestation:
    schema: Literal["crux.backend-execution-attestation/v1"]
    backend_id: str
    descriptor_sha256: str
    git_commit: str
    checkout_dirty: bool
    strict_mode: bool
    changed_files_manifest: PublishedArtifact | None
    host_numeric_fingerprint: HostNumericFingerprint | None
    conditions: ExecutionConditions


def validate_execution_attestation(
    content: bytes,
    *,
    expected_backend_id: str,
    expected_descriptor_sha256: str,
) -> ExecutionAttestation:
    try:
        _require_nonempty_string(expected_backend_id, "expected backend_id")
        _require_hash(expected_descriptor_sha256, "expected descriptor_sha256")
        if not isinstance(content, bytes) or not content.endswith(b"\n"):
            raise AttestationError("execution attestation must end with newline")
        value = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(value, dict) or set(value) != _ATTESTATION_KEYS:
            raise AttestationError("execution attestation must contain the exact key set")
        if value["schema"] != _ATTESTATION_SCHEMA:
            raise AttestationError(f"schema must be {_ATTESTATION_SCHEMA}")
        if value["backend_id"] != expected_backend_id:
            raise AttestationError("execution attestation backend_id mismatch")
        if value["descriptor_sha256"] != expected_descriptor_sha256:
            raise AttestationError("execution attestation descriptor_sha256 mismatch")
        _require_hash(value["descriptor_sha256"], "descriptor_sha256")  # type: ignore[arg-type]
        git_commit = value["git_commit"]
        if (
            not isinstance(git_commit, str)
            or len(git_commit) != 40
            or any(character not in "0123456789abcdef" for character in git_commit)
        ):
            raise AttestationError("git_commit must be lowercase Git identity")
        checkout_dirty = value["checkout_dirty"]
        strict_mode = value["strict_mode"]
        if type(checkout_dirty) is not bool or type(strict_mode) is not bool:
            raise AttestationError("attestation boolean fields must be boolean")
        fingerprint_value = value["host_numeric_fingerprint"]
        host_numeric_fingerprint = (
            None if fingerprint_value is None else parse_host_numeric_fingerprint(fingerprint_value)
        )
        changed = value["changed_files_manifest"]
        changed_artifact: PublishedArtifact | None = None
        if changed is not None:
            if not isinstance(changed, dict) or set(changed) != {"path", "sha256"}:
                raise AttestationError("changed_files_manifest must be an artifact reference")
            changed_path = changed["path"]
            changed_sha256 = changed["sha256"]
            if not isinstance(changed_path, str) or not isinstance(changed_sha256, str):
                raise AttestationError("changed_files_manifest fields must be strings")
            _validate_repository_path(changed_path)
            _require_hash(
                changed_sha256,
                "changed_files_manifest sha256",
            )
            if not checkout_dirty or strict_mode:
                raise AttestationError(
                    "changed_files_manifest requires a dirty non-strict checkout"
                )
            changed_artifact = PublishedArtifact(
                role="changed_files_manifest",
                path=Path(changed_path),
                sha256=changed_sha256,
            )
        conditions = ExecutionConditions(
            cpu_limit=value["cpu_limit"],  # type: ignore[arg-type]
            memory_bytes=value["memory_bytes"],  # type: ignore[arg-type]
            pid_limit=value["pid_limit"],  # type: ignore[arg-type]
            tmp_bytes=value["tmp_bytes"],  # type: ignore[arg-type]
            shm_bytes=value["shm_bytes"],  # type: ignore[arg-type]
            startup_deadline_seconds=value["startup_deadline_seconds"],  # type: ignore[arg-type]
            request_deadline_seconds=value["request_deadline_seconds"],  # type: ignore[arg-type]
        )
        _validate_backend_conditions(expected_backend_id, conditions)
        return ExecutionAttestation(
            schema=_ATTESTATION_SCHEMA,
            backend_id=expected_backend_id,
            descriptor_sha256=expected_descriptor_sha256,
            git_commit=git_commit,
            checkout_dirty=checkout_dirty,
            strict_mode=strict_mode,
            changed_files_manifest=changed_artifact,
            host_numeric_fingerprint=host_numeric_fingerprint,
            conditions=conditions,
        )
    except AttestationError:
        raise
    except (KeyError, TypeError, ValueError):
        raise AttestationError("execution attestation is invalid") from None


def validate_changed_file_manifest(content: bytes) -> tuple[ChangedFile, ...]:
    try:
        if not isinstance(content, bytes) or not content.endswith(b"\n"):
            raise AttestationError("changed-file manifest must end with newline")
        value = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(value, list) or not value:
            raise AttestationError("changed-file manifest must be a nonempty array")

        changed_files: list[ChangedFile] = []
        for row in value:
            if not isinstance(row, dict) or set(row) != _CHANGED_FILE_KEYS:
                raise AttestationError("changed-file row must contain the exact key set")
            path = row["path"]
            status = row["status"]
            digest = row["sha256"]
            if (
                not isinstance(path, str)
                or not isinstance(status, str)
                or (digest is not None and not isinstance(digest, str))
            ):
                raise AttestationError("changed-file row fields have invalid types")
            changed_files.append(
                ChangedFile(
                    path=path,
                    status=cast(
                        Literal["modified", "deleted", "untracked"],
                        status,
                    ),
                    sha256=digest,
                )
            )

        paths = [changed.path for changed in changed_files]
        if len(set(paths)) != len(paths):
            raise AttestationError("changed-file manifest contains duplicate paths")
        if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
            raise AttestationError("changed-file manifest paths must be bytewise sorted")
        return tuple(changed_files)
    except AttestationError:
        raise
    except (KeyError, TypeError, ValueError):
        raise AttestationError("changed-file manifest is invalid") from None


def build_changed_file_manifest(
    repository_root: Path,
    source_manifests: Collection[Path],
) -> tuple[ChangedFile, ...]:
    try:
        with _repository_descriptor(repository_root) as repository_descriptor:
            enumerated, covered_roots = _load_source_scope(
                repository_root,
                repository_descriptor,
                source_manifests,
            )
            status_output = _git_status(repository_root)
            return _changed_files(
                repository_descriptor,
                status_output,
                enumerated,
                covered_roots,
            )
    except AttestationError:
        raise
    except Exception:
        raise AttestationError("changed-file manifest could not be built") from None


def publish_execution_attestation(
    repository_root: Path,
    backend_root: Path,
    *,
    backend_id: str,
    descriptor_sha256: str,
    source_manifests: Collection[Path],
    strict_mode: bool,
    conditions: ExecutionConditions,
    expected_host_numeric_fingerprint: HostNumericFingerprint | None = None,
    now: datetime | None = None,
    run_id: UUID | None = None,
) -> PublishedArtifact:
    try:
        _require_nonempty_string(backend_id, "backend_id")
        _require_hash(descriptor_sha256, "descriptor_sha256")
        if type(strict_mode) is not bool:
            raise AttestationError("strict_mode must be boolean")
        _validate_backend_conditions(backend_id, conditions)
        host_numeric_fingerprint = _host_numeric_fingerprint()
        if (
            expected_host_numeric_fingerprint is not None
            and host_numeric_fingerprint != expected_host_numeric_fingerprint
        ):
            raise AttestationError("measured host numeric fingerprint disagrees with evidence")
        effective_now = datetime.now(UTC) if now is None else now
        effective_run_id = uuid4() if run_id is None else run_id
        timestamp = _filename_timestamp(effective_now)
        run_id_text = _uuid4_text(effective_run_id)
        with _repository_descriptor(repository_root) as repository_descriptor:
            enumerated, covered_roots = _load_source_scope(
                repository_root,
                repository_descriptor,
                source_manifests,
            )
            git_commit = _git_commit(repository_root)
            status_output = _git_status(repository_root)
            changed_files = _changed_files(
                repository_descriptor,
                status_output,
                enumerated,
                covered_roots,
            )
        if strict_mode and changed_files:
            raise AttestationError("strict mode rejects inference-relevant changes")

        changed_artifact = _publish_changed_files(backend_root, changed_files)
        attestation = ExecutionAttestation(
            schema=_ATTESTATION_SCHEMA,
            backend_id=backend_id,
            descriptor_sha256=descriptor_sha256,
            git_commit=git_commit,
            checkout_dirty=bool(status_output),
            strict_mode=strict_mode,
            changed_files_manifest=changed_artifact,
            host_numeric_fingerprint=host_numeric_fingerprint,
            conditions=conditions,
        )
        payload = _attestation_payload(attestation, repository_root)
        content = canonical_json_bytes(payload, trailing_newline=True)
        path = backend_root / "attestations" / f"{timestamp}-{run_id_text}.json"
        return publish_immutable_bytes(
            path,
            content,
            sha256_hex(content),
            role="execution_attestation",
        )
    except AttestationError:
        raise
    except Exception:
        raise AttestationError("execution attestation could not be published") from None


@contextmanager
def _repository_descriptor(repository_root: Path) -> Iterator[int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise OSError("anchored repository reads are unavailable")
    descriptor = os.open(repository_root, os.O_RDONLY | directory_flag | no_follow)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("repository root is not a directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _repository_relative_manifest(repository_root: Path, manifest_path: Path) -> str:
    root = Path(os.path.abspath(repository_root))
    candidate = (
        Path(os.path.abspath(manifest_path))
        if manifest_path.is_absolute()
        else root / manifest_path
    )
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        raise AttestationError("source manifest must be inside repository root") from None
    _validate_repository_path(relative)
    return relative


def _read_repository_regular_file(
    repository_descriptor: int,
    relative_path: str,
    *,
    label: str,
) -> bytes:
    _validate_repository_path(relative_path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or directory_flag is None or non_block is None:
        raise OSError(f"{label} anchored reads are unavailable")
    parts = PurePosixPath(relative_path).parts
    directory_descriptor = os.dup(repository_descriptor)
    file_descriptor: int | None = None
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=directory_descriptor,
            )
            try:
                is_directory = stat.S_ISDIR(os.fstat(next_descriptor).st_mode)
            except OSError:
                os.close(next_descriptor)
                raise
            if not is_directory:
                os.close(next_descriptor)
                raise OSError(f"{label} parent is not a regular directory")
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | no_follow | non_block,
            dir_fd=directory_descriptor,
        )
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise OSError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(file_descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _load_source_scope(
    repository_root: Path,
    repository_descriptor: int,
    source_manifests: Collection[Path],
) -> tuple[set[str], set[str]]:
    if not source_manifests:
        raise AttestationError("at least one source manifest is required")
    enumerated: set[str] = set()
    covered_roots: set[str] = set()
    for manifest_path in source_manifests:
        try:
            relative_manifest = _repository_relative_manifest(repository_root, manifest_path)
            content = _read_repository_regular_file(
                repository_descriptor,
                relative_manifest,
                label="source manifest",
            )
            payload = strict_json_loads(content)
        except Exception:
            raise AttestationError("source manifest is invalid") from None
        if not isinstance(payload, dict) or set(payload) != _SOURCE_MANIFEST_KEYS:
            raise AttestationError("source manifest must contain the exact key set")
        if not isinstance(payload["schema"], str) or not payload["schema"]:
            raise AttestationError("source manifest schema is invalid")
        roots = payload["covered_roots"]
        files = payload["files"]
        if not isinstance(roots, list) or not isinstance(files, list):
            raise AttestationError("source manifest paths must be arrays")
        local_paths: set[str] = set()
        for root in roots:
            if not isinstance(root, str):
                raise AttestationError("covered root path must be a string")
            _validate_repository_path(root)
            covered_roots.add(root)
        for row in files:
            if not isinstance(row, dict):
                raise AttestationError("source manifest file row must be an object")
            if not {"path", "sha256"}.issubset(row) or not set(row).issubset(_SOURCE_FILE_KEYS):
                raise AttestationError("source manifest file row has invalid keys")
            path = row["path"]
            digest = row["sha256"]
            if not isinstance(path, str):
                raise AttestationError("source manifest path must be a string")
            if not isinstance(digest, str):
                raise AttestationError("source manifest sha256 must be a string")
            _validate_repository_path(path)
            _require_hash(digest, "source manifest sha256")
            if path in local_paths:
                raise AttestationError("source manifest has a duplicate path")
            local_paths.add(path)
            enumerated.add(path)
    return enumerated, covered_roots


def _git_commit(repository_root: Path) -> str:
    output = _run_git(repository_root, ["rev-parse", "HEAD"])
    try:
        commit = output.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        raise AttestationError("Git commit must be ASCII") from None
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise AttestationError("Git commit must be lowercase hexadecimal")
    return commit


def _git_status(repository_root: Path) -> bytes:
    return _run_git(
        repository_root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        ],
    )


def _run_git(repository_root: Path, arguments: list[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        raise AttestationError("Git state is unavailable") from None
    return completed.stdout


def _changed_files(
    repository_descriptor: int,
    status_output: bytes,
    enumerated: set[str],
    covered_roots: set[str],
) -> tuple[ChangedFile, ...]:
    changes: dict[str, ChangedFile] = {}
    for token in status_output.split(b"\0"):
        if not token:
            continue
        if len(token) < 4 or token[2:3] != b" ":
            raise AttestationError("Git status record is invalid")
        status = token[:2].decode("ascii", errors="strict")
        try:
            path = token[3:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise AttestationError("Git status path must be valid UTF-8") from None
        _validate_repository_path(path)
        if status == "??" and (path in enumerated or _is_below_covered_root(path, covered_roots)):
            change = ChangedFile(
                path,
                "untracked",
                _hash_repository_file(repository_descriptor, path),
            )
        elif "D" in status and path in enumerated:
            change = ChangedFile(path, "deleted", None)
        elif path in enumerated and _status_has_present_change(status):
            change = ChangedFile(
                path,
                "modified",
                _hash_repository_file(repository_descriptor, path),
            )
        elif _status_is_new(status) and _is_below_covered_root(path, covered_roots):
            change = ChangedFile(
                path,
                "untracked",
                _hash_repository_file(repository_descriptor, path),
            )
        else:
            continue
        if path in changes:
            raise AttestationError("Git status contains a duplicate relevant path")
        changes[path] = change
    return tuple(changes[path] for path in sorted(changes, key=lambda value: value.encode("utf-8")))


def _status_has_present_change(status: str) -> bool:
    return any(code in status for code in ("M", "A", "T", "U", "?"))


def _status_is_new(status: str) -> bool:
    return status == "??" or "A" in status


def _is_below_covered_root(path: str, covered_roots: set[str]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in covered_roots)


def _hash_repository_file(repository_descriptor: int, relative_path: str) -> str:
    _validate_repository_path(relative_path)
    try:
        content = _read_repository_regular_file(
            repository_descriptor,
            relative_path,
            label="changed-file",
        )
        return sha256_hex(content)
    except OSError:
        raise AttestationError("changed-file path must remain a regular file") from None


def _publish_changed_files(
    backend_root: Path,
    changed_files: tuple[ChangedFile, ...],
) -> PublishedArtifact | None:
    if not changed_files:
        return None
    payload: list[JsonValue] = [
        {
            "path": changed.path,
            "sha256": changed.sha256,
            "status": changed.status,
        }
        for changed in changed_files
    ]
    content = canonical_json_bytes(payload, trailing_newline=True)
    digest = sha256_hex(content)
    path = backend_root / "attestations" / "changed-files" / "sha256" / f"{digest}.json"
    return publish_immutable_bytes(
        path,
        content,
        digest,
        role="changed_files_manifest",
    )


def _attestation_payload(
    attestation: ExecutionAttestation,
    repository_root: Path,
) -> dict[str, JsonValue]:
    changed_reference: dict[str, JsonValue] | None = None
    if attestation.changed_files_manifest is not None:
        changed_reference = _artifact_reference(
            attestation.changed_files_manifest,
            repository_root,
        )
    return {
        "backend_id": attestation.backend_id,
        "changed_files_manifest": changed_reference,
        "checkout_dirty": attestation.checkout_dirty,
        "cpu_limit": attestation.conditions.cpu_limit,
        "descriptor_sha256": attestation.descriptor_sha256,
        "git_commit": attestation.git_commit,
        "host_numeric_fingerprint": (
            None
            if attestation.host_numeric_fingerprint is None
            else attestation.host_numeric_fingerprint.as_json()
        ),
        "memory_bytes": attestation.conditions.memory_bytes,
        "pid_limit": attestation.conditions.pid_limit,
        "request_deadline_seconds": attestation.conditions.request_deadline_seconds,
        "schema": attestation.schema,
        "shm_bytes": attestation.conditions.shm_bytes,
        "startup_deadline_seconds": attestation.conditions.startup_deadline_seconds,
        "strict_mode": attestation.strict_mode,
        "tmp_bytes": attestation.conditions.tmp_bytes,
    }


def _artifact_reference(
    artifact: PublishedArtifact,
    repository_root: Path,
) -> dict[str, JsonValue]:
    try:
        relative_path = artifact.path.relative_to(repository_root).as_posix()
    except ValueError:
        raise AttestationError("attestation artifacts must be repository-relative") from None
    _validate_repository_path(relative_path)
    return {"path": relative_path, "sha256": artifact.sha256}


def _validate_backend_conditions(
    backend_id: str,
    conditions: ExecutionConditions,
) -> None:
    if backend_id == _HEURISTIC_BACKEND_ID:
        return
    for field in ("cpu_limit", "memory_bytes", "pid_limit", "tmp_bytes", "shm_bytes"):
        if getattr(conditions, field) is None:
            raise AttestationError(f"{field} is required for container execution")


def _host_numeric_fingerprint() -> HostNumericFingerprint:
    fingerprints = _collect_host_numeric_fingerprints()
    if not fingerprints:
        raise AttestationError("host numeric fingerprint requires visible logical CPUs")
    normalized = tuple(parse_host_numeric_fingerprint(value) for value in fingerprints)
    if any(value != normalized[0] for value in normalized[1:]):
        raise AttestationError("logical CPU fingerprints are inconsistent")
    return normalized[0]


def _collect_host_numeric_fingerprints() -> tuple[dict[str, str], ...]:
    system = platform.system()
    if system == "Linux":
        return _linux_host_numeric_fingerprints()
    if system == "Darwin":
        return _darwin_host_numeric_fingerprints()
    raise AttestationError(f"host numeric fingerprint is unsupported on {system or 'unknown'}")


def _linux_host_numeric_fingerprints() -> tuple[dict[str, str], ...]:
    try:
        content = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="strict")
    except OSError:
        raise AttestationError("host CPU facts are unreadable") from None
    architecture = platform.machine()
    _require_nonempty_string(architecture, "host architecture")
    records = _linux_cpuinfo_records(content)
    visible_cpu_ids = _visible_linux_cpu_ids()
    fingerprints: list[dict[str, str]] = []
    for record in records:
        processor = record.get("processor")
        if processor is None or (visible_cpu_ids is not None and processor not in visible_cpu_ids):
            continue
        try:
            fingerprints.append(
                {
                    "architecture": architecture,
                    "cpu_family": record["cpu family"],
                    "cpu_model": record["model"],
                    "cpu_stepping": record["stepping"],
                    "cpu_vendor_id": record["vendor_id"],
                }
            )
        except KeyError:
            raise AttestationError("host CPU facts are incomplete") from None
    if not fingerprints:
        raise AttestationError("host CPU facts contain no visible logical CPUs")
    return tuple(fingerprints)


def _linux_cpuinfo_records(content: str) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    record: dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            if record:
                records.append(record)
                record = {}
            continue
        if ":" not in line:
            raise AttestationError("host CPU facts are malformed")
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key or key in record or (not value and key in _LINUX_CPUINFO_REQUIRED_FIELDS):
            raise AttestationError("host CPU facts are malformed")
        record[key] = value
    if record:
        records.append(record)
    return tuple(records)


def _visible_linux_cpu_ids() -> frozenset[str] | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if not callable(get_affinity):
        return None
    try:
        # Pylint cannot retain the preceding callable narrowing through getattr.
        # pylint: disable-next=not-callable
        cpu_ids = get_affinity(0)
        return frozenset(str(cpu_id) for cpu_id in cpu_ids)
    except OSError:
        return None


def _darwin_host_numeric_fingerprints() -> tuple[dict[str, str], ...]:
    architecture = _sysctl_text("hw.machine")
    try:
        fingerprint = {
            "architecture": architecture,
            "cpu_family": _sysctl_text("machdep.cpu.family"),
            "cpu_model": _sysctl_text("machdep.cpu.model"),
            "cpu_stepping": _sysctl_text("machdep.cpu.stepping"),
            "cpu_vendor_id": _sysctl_text("machdep.cpu.vendor"),
        }
    except AttestationError:
        if architecture not in {"arm64", "arm64e"}:
            raise
        fingerprint = {
            "architecture": architecture,
            "cpu_family": _sysctl_text("hw.cpufamily"),
            "cpu_model": _sysctl_text("hw.model"),
            "cpu_stepping": _sysctl_text("hw.cpusubtype"),
            "cpu_vendor_id": "Apple",
        }
    logical_cpu_count = _sysctl_text("hw.ncpu")
    try:
        count = int(logical_cpu_count)
    except ValueError:
        raise AttestationError("host logical CPU count is invalid") from None
    if count <= 0:
        raise AttestationError("host logical CPU count is invalid")
    return tuple(dict(fingerprint) for _ in range(count))


def _sysctl_text(name: str) -> str:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        value = completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        raise AttestationError(f"host CPU fact {name} is unavailable") from None
    _require_nonempty_string(value, f"host CPU fact {name}")
    return value


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != _ATTESTATION_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise AttestationError("schema golden must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, dict):
        raise AttestationError("execution attestation schema golden must be an object")
    backend_id = value.get("backend_id")
    descriptor_sha256 = value.get("descriptor_sha256")
    if not isinstance(backend_id, str) or not isinstance(descriptor_sha256, str):
        raise AttestationError("execution attestation schema golden lacks identity")
    validate_execution_attestation(
        content,
        expected_backend_id=backend_id,
        expected_descriptor_sha256=descriptor_sha256,
    )


def _validate_repository_path(path: str) -> None:
    if not isinstance(path, str):
        raise AttestationError("repository path must be a string")
    try:
        encoded = path.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise AttestationError("repository path must be valid UTF-8") from None
    if not encoded or unicodedata.normalize("NFC", path) != path:
        raise AttestationError("repository path must be nonempty NFC")
    if path.startswith("/") or "\\" in path or path.endswith("/"):
        raise AttestationError("repository path must be relative POSIX")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise AttestationError("repository path contains an invalid component")
    if PurePosixPath(path).is_absolute():
        raise AttestationError("repository path must be relative")


def _require_hash(value: str, field: str) -> None:
    try:
        require_sha256(value, field)
    except ValueError:
        raise AttestationError(f"{field} must be lowercase SHA-256") from None


def _require_nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise AttestationError(f"{field} must be nonempty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise AttestationError(f"{field} must be valid UTF-8") from None


def _filename_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise AttestationError("attestation time must be UTC")
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _uuid4_text(value: UUID) -> str:
    if not isinstance(value, UUID) or value.version != 4:
        raise AttestationError("attestation run_id must be UUIDv4")
    rendered = str(value)
    if rendered != rendered.lower():
        raise AttestationError("attestation run_id must be lowercase UUIDv4")
    return rendered
