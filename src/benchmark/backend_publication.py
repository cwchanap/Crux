from __future__ import annotations

# Publication converts every low-level failure into one stable error and must also
# catch cleanup failures raised by injected or platform filesystem implementations.
# pylint: disable=broad-exception-caught
import os
import stat
from pathlib import Path
from uuid import uuid4

from src.benchmark.backend_identity import require_sha256, sha256_hex
from src.benchmark.backends import PublishedArtifact
from src.benchmark.durability import ensure_durable_directory, fsync_directory

_PUBLICATION_ERROR = "artifact_publication_failed"


class ArtifactPublicationError(OSError):
    pass


def publish_immutable_bytes(
    path: Path,
    content: bytes,
    expected_sha256: str,
    *,
    role: str,
) -> PublishedArtifact:
    temporary_path: Path | None = None
    temporary_exists = False
    failed = False
    cleanup_failed = False
    try:
        _validate_publication(path, content, expected_sha256, role)
        ensure_durable_directory(path.parent)
        temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary_stat = _require_regular_descriptor(temporary.fileno())
            _verify_path_binding(temporary_path, temporary_stat)
            written = temporary.write(content)
            if written != len(content):
                raise OSError("artifact write was incomplete")
            temporary.flush()
            os.fsync(temporary.fileno())
            _verify_path_binding(temporary_path, temporary_stat)
            try:
                os.link(temporary_path, path)
                linked = True
            except FileExistsError:
                linked = False
        temporary_path.unlink()
        temporary_exists = False
        _verify_existing_file(
            path,
            content,
            expected_stat=temporary_stat if linked else None,
        )
    except Exception:
        failed = True
    finally:
        if temporary_exists and temporary_path is not None:
            try:
                temporary_path.unlink()
            except Exception:
                cleanup_failed = True
    if failed or cleanup_failed:
        raise ArtifactPublicationError(_PUBLICATION_ERROR) from None
    return PublishedArtifact(role=role, path=path, sha256=expected_sha256)


def atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    temporary_exists = False
    failed = False
    cleanup_failed = False
    try:
        if not isinstance(path, Path) or not isinstance(content, bytes):
            raise TypeError("invalid artifact replacement")
        ensure_durable_directory(path.parent)
        _require_absent_or_regular(path)
        temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary_stat = _require_regular_descriptor(temporary.fileno())
            _verify_path_binding(temporary_path, temporary_stat)
            written = temporary.write(content)
            if written != len(content):
                raise OSError("artifact write was incomplete")
            temporary.flush()
            os.fsync(temporary.fileno())
            _verify_path_binding(temporary_path, temporary_stat)
            _require_absent_or_regular(path)
            os.replace(temporary_path, path)
            temporary_exists = False
            _verify_path_binding(path, temporary_stat)
            os.fsync(temporary.fileno())
            _verify_path_binding(path, temporary_stat)
        fsync_directory(path.parent)
    except Exception:
        failed = True
    finally:
        if temporary_exists and temporary_path is not None:
            try:
                temporary_path.unlink()
            except Exception:
                cleanup_failed = True
    if failed or cleanup_failed:
        raise ArtifactPublicationError(_PUBLICATION_ERROR) from None


def read_regular_file_no_follow(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _regular_file_open_flags())
        descriptor_stat = _require_regular_descriptor(descriptor)
        content = _read_descriptor(descriptor)
        _verify_path_binding(path, descriptor_stat)
        return content
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_publication(
    path: Path,
    content: bytes,
    expected_sha256: str,
    role: str,
) -> None:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a Path")
    if not isinstance(content, bytes):
        raise TypeError("artifact content must be bytes")
    if not isinstance(role, str) or not role:
        raise ValueError("artifact role must be nonempty")
    require_sha256(expected_sha256, "expected_sha256")
    if sha256_hex(content) != expected_sha256:
        raise ValueError("artifact content hash mismatch")


def _verify_existing_file(
    path: Path,
    expected_content: bytes,
    *,
    expected_stat: os.stat_result | None,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _regular_file_open_flags())
        descriptor_stat = _require_regular_descriptor(descriptor)
        if expected_stat is not None and not _same_inode(descriptor_stat, expected_stat):
            raise OSError("artifact destination inode changed")
        actual_content = _read_descriptor(descriptor)
        _verify_path_binding(path, descriptor_stat)
        if actual_content != expected_content:
            raise OSError("artifact destination content differs")
        os.fsync(descriptor)
        _verify_path_binding(path, descriptor_stat)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fsync_directory(path.parent)


def _require_absent_or_regular(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("artifact destination is not a regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _regular_file_open_flags())
        descriptor_stat = _require_regular_descriptor(descriptor)
        _verify_path_binding(path, descriptor_stat)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _regular_file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_block is None:
        raise OSError("no-follow regular-file descriptors are unavailable")
    return os.O_RDONLY | no_follow | non_block


def _require_regular_descriptor(descriptor: int) -> os.stat_result:
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise OSError("artifact is not a regular file")
    return descriptor_stat


def _verify_path_binding(path: Path, descriptor_stat: os.stat_result) -> None:
    path_stat = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(path_stat.st_mode) or not _same_inode(path_stat, descriptor_stat):
        raise OSError("artifact path binding changed")


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)
