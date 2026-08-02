#!/usr/bin/env python3
"""Generate, validate, and materialize the exact OaF image build context."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_publication import (
    _rename_no_replace_syscall,
    read_regular_file_no_follow,
)

BUILD_CONTEXT_SCHEMA = "crux.oaf-build-context-manifest/v1"
BUILD_CONTEXT_MANIFEST_PATH = "runtime/oaf_tf1/build-context-manifest.json"
DIRECTORY_MODE = 0o755
FILE_MODE = 0o644
WHEELHOUSE_PREFIX = "runtime/oaf_tf1/wheelhouse/"

# These are reviewed Dockerfile inputs. Task-specific entrypoints are added here when
# their COPY instructions land; the repository root is never supplied as the context.
REVIEWED_REPOSITORY_PATHS = (
    ".github/workflows/hpa320-native-bootstrap.yml",
    ".github/workflows/hpa320-native-candidate.yml",
    "tools/hpa320/_fsync.py",
    "tools/hpa320/audit_legacy_tf2_conversion.py",
    "tools/hpa320/generate_runner_source_manifest.py",
    "tools/hpa320/github_host_evidence.py",
    "tools/hpa320/oaf_build_context.py",
    "tools/hpa320/oaf_candidate_builder.py",
    "tools/hpa320/oaf_host_attestation.py",
    "tools/hpa320/oaf_native_calibration.py",
    "tools/hpa320/oaf_native_artifacts.py",
    "tools/hpa320/oaf_native_runner.py",
    "tools/hpa320/oaf_oci.py",
    "tools/hpa320/oaf_system_packages.py",
    "tools/hpa320/seal_oaf_backend.py",
)
REVIEWED_REPOSITORY_ROOTS = ("runtime/oaf_tf1",)
REVIEWED_WHEELHOUSE_ROOTS = ("runtime", "test")

_MANIFEST_KEYS = frozenset({"directory_mode", "file_mode", "files", "manifest_path", "schema"})
_SOURCE_FILE_KEYS = frozenset({"byte_length", "path", "sha256"})


class BuildContextError(ValueError):
    """The reviewed build context is unsafe, incomplete, or byte-inconsistent."""


@dataclass(frozen=True)
class SourceFile:
    byte_length: int
    path: str
    sha256: str


@dataclass(frozen=True)
class BuildContextManifest:
    directory_mode: int
    file_mode: int
    files: tuple[SourceFile, ...]
    manifest_path: str
    sha256: str


def load_build_context_manifest(path: Path) -> BuildContextManifest:
    """Strict-load one canonical self-excluding build-context manifest."""

    content = _read_regular(Path(path), "build-context manifest")
    return _validate_manifest_content(content)


def _validate_manifest_content(content: bytes) -> BuildContextManifest:
    """Strict-validate canonical build-context manifest bytes.

    Factored from :func:`load_build_context_manifest` so the publication path
    can schema-validate the exact bytes it wrote through a held descriptor,
    without re-reading the output by path (which a concurrent substitution
    could swap for a different, structurally valid authority).
    """

    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise BuildContextError("build-context manifest must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise BuildContextError(f"build-context manifest JSON is invalid: {error}") from None
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise BuildContextError("build-context manifest fields are invalid")
    if (
        value["schema"] != BUILD_CONTEXT_SCHEMA
        or value["directory_mode"] != DIRECTORY_MODE
        or type(value["directory_mode"]) is not int
        or value["file_mode"] != FILE_MODE
        or type(value["file_mode"]) is not int
        or value["manifest_path"] != BUILD_CONTEXT_MANIFEST_PATH
    ):
        raise BuildContextError("build-context manifest authority is invalid")
    rows = value["files"]
    if not isinstance(rows, list) or not rows:
        raise BuildContextError("build-context manifest files are invalid")
    parsed: list[SourceFile] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _SOURCE_FILE_KEYS:
            raise BuildContextError("build-context source file fields are invalid")
        relative = row["path"]
        size = row["byte_length"]
        digest = row["sha256"]
        if (
            not _safe_relative_path(relative)
            or relative == BUILD_CONTEXT_MANIFEST_PATH
            or _is_ignore_file(relative)
            or _is_generated_cache_path(relative)
        ):
            raise BuildContextError("build-context source file path is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BuildContextError("build-context source file size is invalid")
        if not _is_sha256(digest):
            raise BuildContextError("build-context source file hash is invalid")
        parsed.append(SourceFile(size, relative, digest))
    paths = [row.path for row in parsed]
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")) or len(paths) != len(
        set(paths)
    ):
        raise BuildContextError("build-context source files are not uniquely sorted")
    return BuildContextManifest(
        directory_mode=DIRECTORY_MODE,
        file_mode=FILE_MODE,
        files=tuple(parsed),
        manifest_path=BUILD_CONTEXT_MANIFEST_PATH,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def generate_build_context_manifest(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
) -> bytes:
    """Render canonical bytes for only the source-reviewed Dockerfile inputs."""

    repository = _require_root(Path(repository_root), "repository root")
    wheelhouse = _require_root(Path(wheelhouse_root), "wheelhouse root")
    sources: dict[str, bytes] = {}
    for relative in REVIEWED_REPOSITORY_PATHS:
        _add_source(sources, relative, repository, relative)
    for relative_root in REVIEWED_REPOSITORY_ROOTS:
        for relative in _enumerate_regular_files(repository, relative_root):
            if relative == BUILD_CONTEXT_MANIFEST_PATH or relative.startswith(WHEELHOUSE_PREFIX):
                continue
            _add_source(sources, relative, repository, relative)
    for relative_root in REVIEWED_WHEELHOUSE_ROOTS:
        for wheel_relative in _enumerate_regular_files(wheelhouse, relative_root):
            logical = WHEELHOUSE_PREFIX + wheel_relative
            _add_source(sources, logical, wheelhouse, wheel_relative)
    if not sources:
        raise BuildContextError("reviewed build context is empty")
    files: list[JsonValue] = []
    for relative in sorted(sources, key=lambda item: item.encode("utf-8")):
        content = sources[relative]
        files.append(
            {
                "byte_length": len(content),
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    payload: JsonValue = {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "files": files,
        "manifest_path": BUILD_CONTEXT_MANIFEST_PATH,
        "schema": BUILD_CONTEXT_SCHEMA,
    }
    return canonical_json_bytes(payload, trailing_newline=True)


def materialize_build_context(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
    manifest_path: Path,
    destination: Path,
) -> Path:
    """Create a fresh normalized directory containing exactly the authenticated files."""

    repository = _require_root(Path(repository_root), "repository root")
    wheelhouse = _require_root(Path(wheelhouse_root), "wheelhouse root")
    manifest_source = Path(manifest_path)
    manifest = load_build_context_manifest(manifest_source)
    manifest_content = _read_source_path(manifest_source, "build-context manifest")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise BuildContextError("build-context destination must be absent")
    try:
        target.mkdir(mode=DIRECTORY_MODE, parents=False)
    except OSError as error:
        raise BuildContextError("build-context destination could not be created") from error
    expected_files = {row.path for row in manifest.files} | {manifest.manifest_path}
    try:
        for row in manifest.files:
            if row.path.startswith(WHEELHOUSE_PREFIX):
                source_root = wheelhouse
                source_relative = row.path.removeprefix(WHEELHOUSE_PREFIX)
            else:
                source_root = repository
                source_relative = row.path
            content = _read_source(source_root, source_relative)
            if len(content) != row.byte_length:
                raise BuildContextError(f"build-context source size drift: {row.path}")
            if hashlib.sha256(content).hexdigest() != row.sha256:
                raise BuildContextError(f"build-context source hash drift: {row.path}")
            _write_normalized_file(target, row.path, content)
        _write_normalized_file(target, manifest.manifest_path, manifest_content)
        actual_files = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise BuildContextError("materialized build context has missing or extra entries")
        for directory in sorted(
            (path for path in target.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _normalize_directory(directory)
        _normalize_directory(target)
        return target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate one isolated build-context manifest through the production loader."""

    if schema != BUILD_CONTEXT_SCHEMA:
        raise ValueError("build-context schema golden is unsupported")
    import tempfile

    try:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "build-context-manifest.json"
            path.write_bytes(content)
            load_build_context_manifest(path)
    except (BuildContextError, OSError) as error:
        raise ValueError(str(error)) from None


def _add_source(
    sources: dict[str, bytes],
    logical_path: str,
    source_root: Path,
    source_relative: str,
) -> None:
    if (
        not _safe_relative_path(logical_path)
        or logical_path == BUILD_CONTEXT_MANIFEST_PATH
        or _is_ignore_file(logical_path)
        or logical_path in sources
    ):
        raise BuildContextError("reviewed build-context path is invalid or duplicated")
    sources[logical_path] = _read_source(source_root, source_relative)


def _enumerate_regular_files(root: Path, relative_root: str) -> tuple[str, ...]:
    if not _safe_relative_path(relative_root):
        raise BuildContextError("reviewed build-context root is invalid")
    start = root / Path(*PurePosixPath(relative_root).parts)
    metadata = _lstat(start, "reviewed build-context root")
    if not stat.S_ISDIR(metadata.st_mode):
        raise BuildContextError("reviewed build-context root is not a directory")
    paths: list[str] = []
    for path in start.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_generated_cache_path(relative):
            continue
        metadata = _lstat(path, "reviewed build-context input")
        if stat.S_ISLNK(metadata.st_mode):
            raise BuildContextError("reviewed build-context input contains a link")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise BuildContextError("reviewed build-context input is not regular")
        paths.append(relative)
    return tuple(sorted(paths, key=lambda item: item.encode("utf-8")))


def _read_source(root: Path, relative: str) -> bytes:
    if not _safe_relative_path(relative):
        raise BuildContextError("build-context source path is invalid")
    current = root
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        current /= part
        metadata = _lstat(current, "build-context source ancestor")
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise BuildContextError("build-context source ancestor is not a stable directory")
    path = root.joinpath(*parts)
    metadata = _lstat(path, "build-context source")
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BuildContextError("build-context source is not a regular file")
    if metadata.st_nlink != 1:
        raise BuildContextError("build-context source is multiply linked")
    return _read_source_path(path, "build-context source")


def _read_source_path(path: Path, label: str) -> bytes:
    metadata = _lstat(path, label)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BuildContextError(f"{label} is not a stable single-link regular file")
    try:
        return read_regular_file_no_follow(path)
    except OSError as error:
        raise BuildContextError(f"{label} is not a stable regular file") from error


def _write_normalized_file(root: Path, relative: str, content: bytes) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            FILE_MODE,
        )
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fchmod(descriptor, FILE_MODE)
            os.utime(descriptor, ns=(0, 0))
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BuildContextError("materialized build-context file could not be written") from error


def _normalize_directory(path: Path) -> None:
    try:
        path.chmod(DIRECTORY_MODE, follow_symlinks=False)
        os.utime(path, ns=(0, 0), follow_symlinks=False)
    except OSError as error:
        raise BuildContextError("materialized build-context directory is unstable") from error


def _read_regular(path: Path, label: str) -> bytes:
    return _read_source_path(path, label)


def _require_root(path: Path, label: str) -> Path:
    metadata = _lstat(path, label)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BuildContextError(f"{label} must be a stable directory")
    return path


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise BuildContextError(f"{label} is missing or unreadable") from error


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _is_ignore_file(relative: str) -> bool:
    return PurePosixPath(relative).name == ".dockerignore" or relative.endswith(".dockerignore")


def _is_generated_cache_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
            for part in path.parts
        )
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _open_parent_directory(parent: Path) -> int:
    """Open ``parent`` as a no-follow directory descriptor and return it.

    The descriptor is retained across the manifest publication so that the
    identity check, unlink, and directory fsync performed during rollback all
    resolve relative to the same directory inode. This closes the window where
    a path-based ``stat`` authenticates a file in the original parent while a
    subsequent path-based ``unlink`` removes a same-named file from a parent
    directory that was renamed and replaced between the two calls.
    """
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise BuildContextError("build-context manifest parent durability support is unavailable")
    descriptor = os.open(parent, os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0))
    try:
        current = os.fstat(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        os.close(descriptor)
        raise BuildContextError("build-context manifest parent must be a stable directory")
    return descriptor


def _unlink_if_owned(name: str, metadata: os.stat_result, parent_fd: int) -> None:
    """Remove ``name`` only if it is still the inode we created.

    Uses an atomic rename-to-quarantine protocol rather than stat-then-unlink
    so a concurrent writer that replaces ``name`` between the ownership check
    and the unlink cannot cause rollback to delete the replacement. The entry
    is atomically renamed to a unique quarantine name within the held parent
    directory, re-verified by inode, and only then unlinked. Both the
    quarantine move and any mismatch restoration use no-replace rename
    semantics, so a pre-existing quarantine destination or a newer occupant at
    ``name`` is never silently overwritten. If the quarantine destination is
    already occupied, the owned file is left in place rather than risk
    deleting a replacement; if ``name`` is reoccupied before restoration, the
    quarantined entry is retained for recovery rather than clobbering the new
    occupant.
    """

    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        return
    quarantine = f".{name}.rollback-quarantine.{metadata.st_ino}.{os.getpid()}"
    try:
        _rename_no_replace_syscall(
            source=name, destination=quarantine, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
        )
    except OSError:
        # The quarantine destination is already occupied (or no-replace rename
        # is unsupported): do not overwrite it. Leave the owned file in place
        # rather than risk deleting a replacement we do not own.
        return
    try:
        quarantined = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if (quarantined.st_dev, quarantined.st_ino) != (metadata.st_dev, metadata.st_ino):
        # A concurrent writer replaced ``name`` between our stat and the
        # atomic rename. The replacement is now at ``quarantine``; restore it
        # to ``name`` and leave it untouched rather than deleting a file we
        # do not own. Use no-replace so a newer occupant at ``name`` is
        # preserved and the quarantined entry is retained for recovery.
        try:
            _rename_no_replace_syscall(
                source=quarantine, destination=name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
            )
        except OSError:
            # ``name`` is occupied by a newer file (or rename is unsupported):
            # preserve the quarantined entry for recovery, never overwrite the
            # new occupant.
            pass
        return
    try:
        os.unlink(quarantine, dir_fd=parent_fd)
    except OSError:
        return
    try:
        os.fsync(parent_fd)
    except OSError:
        pass


def _verify_published_descriptor(
    name: str,
    parent_fd: int,
    created_metadata: os.stat_result,
    content: bytes,
) -> None:
    """Validate the published manifest through the held parent descriptor.

    Reopens ``name`` relative to ``parent_fd`` (not by path) so a concurrent
    substitution at the output path cannot make publication succeed for a
    different authority. Confirms the reopened file is the same inode we
    created, that its exact bytes equal ``content``, and that the bytes
    actually read from disk—not the in-memory ``content`` variable—
    schema-validate as a canonical build-context manifest. The verification
    read is bounded to ``len(content) + 1`` bytes (with an up-front size
    check) so a concurrent writer that appends an unbounded suffix to the same
    inode cannot turn validation into an OOM or nonterminating operation.
    """

    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
            created_metadata.st_dev,
            created_metadata.st_ino,
        ):
            raise BuildContextError(
                "build-context manifest output was substituted before validation"
            )
        # Bound the verification read so an appended suffix cannot consume
        # unbounded memory/time. The expected length is known, so reject any
        # size mismatch up front and read at most one byte past it: an empty
        # extra read confirms EOF at exactly len(content), a nonempty extra
        # read means bytes were appended after the size check.
        expected = len(content)
        if current.st_size != expected:
            raise BuildContextError("build-context manifest output size drifted before validation")
        bounded = bytearray()
        to_read = expected + 1
        while to_read > 0:
            chunk = os.read(descriptor, min(to_read, 65536))
            if not chunk:
                break
            bounded += chunk
            to_read -= len(chunk)
        if len(bounded) != expected or bytes(bounded) != content:
            raise BuildContextError("build-context manifest output bytes drifted before validation")
        _validate_manifest_content(bytes(bounded))
    finally:
        os.close(descriptor)


def _verify_parent_ancestry(parent: Path, parent_metadata: os.stat_result) -> None:
    """Confirm the path-facing parent still resolves to the held directory inode.

    The held ``parent_fd`` always refers to the directory the manifest was
    created in, but the path ``parent`` may have been renamed and replaced by a
    different directory. If so, the published path no longer refers to the
    directory we wrote into; treat the ancestry substitution as failure rather
    than reporting success for an output the caller's path can no longer reach.
    """

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise BuildContextError("build-context manifest parent ancestry support is unavailable")
    try:
        descriptor = os.open(
            parent,
            os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise BuildContextError(
            "build-context manifest parent path no longer resolves to a stable directory"
        ) from error
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ):
            raise BuildContextError(
                "build-context manifest parent directory was substituted before completion"
            )
    finally:
        os.close(descriptor)


def _write_manifest_content(path: Path, content: bytes, *, replace: bool) -> None:
    output = Path(path)
    parent = _require_root(output.parent, "build-context manifest parent")
    if not replace:
        if output.exists() or output.is_symlink():
            raise BuildContextError("build-context manifest output must be absent")
        _create_manifest_content(output, parent, content)
    else:
        _replace_manifest_content(output, parent, content)


def _create_manifest_content(  # pylint: disable=too-many-statements
    output: Path, parent: Path, content: bytes
) -> None:
    published_name: str | None = None
    created_metadata: os.stat_result | None = None
    parent_fd = _open_parent_directory(parent)
    parent_metadata = os.fstat(parent_fd)
    try:
        descriptor = os.open(
            output.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            FILE_MODE,
            dir_fd=parent_fd,
        )
        created_metadata = os.fstat(descriptor)
        published_name = output.name
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fchmod(descriptor, FILE_MODE)
            os.utime(descriptor, ns=(0, 0))
            os.fsync(descriptor)
            # Validate the manifest through the held parent descriptor
            # rather than by path: a concurrent substitution at the output
            # path must not make publication succeed for a different, even
            # structurally valid, authority. Reopening relative to
            # ``parent_fd`` and checking the inode proves the file we
            # created is the one being validated, and byte-equality with
            # ``content`` proves it is the authority we generated.
            _verify_published_descriptor(output.name, parent_fd, created_metadata, content)
            # Sync the directory entry through the held parent descriptor,
            # not a path re-open, so a parent-directory swap cannot sync the
            # replacement directory.
            os.fsync(parent_fd)
            # Confirm the path-facing parent still resolves to the held
            # directory inode; an ancestry substitution means the published
            # path no longer refers to the directory we wrote into.
            _verify_parent_ancestry(parent, parent_metadata)
            # Re-verify the output through the held parent descriptor after
            # the directory fsync: a same-directory substitution between the
            # first verification and here would pass the ancestry check
            # (parent inode unchanged) but leave a replacement at the output
            # path. Reopening and re-checking inode + exact bytes closes
            # that window before publication is declared successful.
            _verify_published_descriptor(output.name, parent_fd, created_metadata, content)
        finally:
            os.close(descriptor)
        published_name = None
    except (BuildContextError, OSError):
        if published_name is not None and created_metadata is not None:
            _unlink_if_owned(published_name, created_metadata, parent_fd)
        raise
    finally:
        os.close(parent_fd)


def _replace_manifest_content(  # pylint: disable=too-many-statements
    output: Path, parent: Path, content: bytes
) -> None:
    # Mirror the create path's descriptor-based substitution and durability
    # guarantees: hold the parent directory descriptor throughout, validate
    # the temporary through that descriptor (not by path), atomically replace
    # into the output, and fsync the held parent descriptor. Track whether
    # the replace completed so a post-replace fsync failure rolls back the
    # owned output rather than unlinking the already-renamed temporary.
    published_name: str | None = None
    created_metadata: os.stat_result | None = None
    parent_fd = _open_parent_directory(parent)
    parent_metadata = os.fstat(parent_fd)
    try:
        temp_name = f".{output.name}.replace.{os.getpid()}.{os.urandom(8).hex()}"
        descriptor = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            FILE_MODE,
            dir_fd=parent_fd,
        )
        created_metadata = os.fstat(descriptor)
        published_name = temp_name
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fchmod(descriptor, FILE_MODE)
            os.utime(descriptor, ns=(0, 0))
            os.fsync(descriptor)
            # Validate the temporary through the held parent descriptor rather
            # than by path, so a substitution at the temporary path cannot
            # make publication succeed for a different authority.
            _verify_published_descriptor(temp_name, parent_fd, created_metadata, content)
        finally:
            os.close(descriptor)
        # Atomically replace the output with the validated temporary. Both
        # paths are relative to the held parent descriptor so a parent
        # directory swap cannot redirect the rename.
        os.replace(temp_name, output.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        # The replace completed: the owned inode now lives at the output path.
        # Track it so a post-replace failure rolls back the output, not the
        # already-renamed temporary (which no longer exists).
        published_name = output.name
        # Sync the directory entry through the held parent descriptor, not a
        # path re-open, so a parent-directory swap cannot sync the replacement
        # directory.
        os.fsync(parent_fd)
        # Confirm the path-facing parent still resolves to the held directory
        # inode; an ancestry substitution means the published path no longer
        # refers to the directory we wrote into.
        _verify_parent_ancestry(parent, parent_metadata)
        # Re-verify the output through the held parent descriptor after the
        # directory fsync: a same-directory substitution between the rename
        # and here would pass the ancestry check (parent inode unchanged) but
        # leave a replacement at the output path. Reopening and re-checking
        # inode + exact bytes closes that window before publication is
        # declared successful.
        _verify_published_descriptor(output.name, parent_fd, created_metadata, content)
        published_name = None
    except (BuildContextError, OSError):
        if published_name is not None and created_metadata is not None:
            _unlink_if_owned(published_name, created_metadata, parent_fd)
        raise
    finally:
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--repository-root", type=Path, required=True)
    generate.add_argument("--wheelhouse-root", type=Path, required=True)
    generate.add_argument("--output", type=Path)
    mode = generate.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        content = generate_build_context_manifest(
            repository_root=args.repository_root,
            wheelhouse_root=args.wheelhouse_root,
        )
        if args.output is None:
            if args.check or args.replace:
                raise BuildContextError("build-context manifest mode requires --output")
            sys.stdout.buffer.write(content)
        else:
            output = Path(args.output)
            if args.check:
                checked = _read_regular(output, "build-context manifest output")
                if checked != content:
                    raise BuildContextError(
                        "build-context manifest does not match generated canonical bytes"
                    )
            else:
                _write_manifest_content(output, content, replace=args.replace)
    except (BuildContextError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
