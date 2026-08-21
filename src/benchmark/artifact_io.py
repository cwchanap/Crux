"""Small neutral helpers for immutable benchmark files."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path


class ArtifactPublicationError(OSError):
    """An immutable artifact could not be created or matched."""


@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str


def read_regular_file_no_follow(path: Path) -> bytes:
    """Read one regular file while rejecting a symlink at the leaf path."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow | close_on_exec)
    try:
        return _read_regular_file_descriptor(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact:
    """Create one file without replacement, or reuse matching immutable bytes."""
    if not isinstance(path, Path) or not isinstance(content, bytes):
        raise TypeError("path must be a Path and content must be bytes")
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ArtifactPublicationError("artifact publication failed") from error
    try:
        digest = _publish_immutable_file_into(directory_fd, path.name, content)
    finally:
        os.close(directory_fd)
    return PublishedArtifact(path=path, sha256=digest)


def publish_immutable_file_at(directory_fd: int, name: str, content: bytes) -> str:
    """Publish one immutable file relative to a held directory descriptor.

    ``name`` must be a single path component.  Every read, temporary file, and
    link is performed relative to ``directory_fd``, so the publication cannot
    be redirected by renaming or swapping the directory after the descriptor
    was validated.  Returns the SHA-256 digest of the published bytes.
    """
    if (
        not isinstance(directory_fd, int)
        or not isinstance(name, str)
        or not isinstance(content, bytes)
    ):
        raise TypeError("directory_fd must be an int, name a str, and content bytes")
    return _publish_immutable_file_into(directory_fd, name, content)


def _read_regular_file_descriptor(descriptor: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("artifact is not a regular file")
    chunks: list[bytes] = []
    while True:
        content = os.read(descriptor, 1024 * 1024)
        if not content:
            return b"".join(chunks)
        chunks.append(content)


def _read_regular_file_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        return _read_regular_file_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable_file_into(directory_fd: int, name: str, content: bytes) -> str:
    if not name or "/" in name or name in {".", ".."}:
        raise ArtifactPublicationError("artifact publication failed")
    digest = hashlib.sha256(content).hexdigest()

    try:
        existing = _read_regular_file_at(directory_fd, name)
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise ArtifactPublicationError("artifact publication failed") from error
    if existing is not None:
        if existing != content:
            raise ArtifactPublicationError("artifact already exists with different bytes")
        return digest

    temporary_name = _create_temporary_file_at(directory_fd, name, content)
    try:
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            try:
                existing = _read_regular_file_at(directory_fd, name)
            except OSError as error:
                raise ArtifactPublicationError("artifact publication failed") from error
            if existing != content:
                raise ArtifactPublicationError("artifact already exists with different bytes")
        else:
            os.fsync(directory_fd)
        return digest
    except ArtifactPublicationError:
        raise
    except (OSError, NotImplementedError) as error:
        # NotImplementedError is not an OSError (platforms without os.link
        # follow_symlinks/dir_fd support raise it), yet it must still surface as
        # a publication failure, not escape unwrapped.
        raise ArtifactPublicationError("artifact publication failed") from error
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except OSError:
            # The publication itself has already either succeeded or failed;
            # do not hide its stable result behind cleanup noise.
            pass


def _create_temporary_file_at(directory_fd: int, name: str, content: bytes) -> str:
    """Create one exclusive temporary file beside ``name`` and fill it."""
    prefix = f".{name}."
    for _ in range(100):
        temporary_name = f"{prefix}{secrets.token_hex(6)}"
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as open_error:
            # ENOSPC, EACCES, EMFILE, etc. are not collision retries.  This
            # helper is called before the caller's unlink finally and OSError-
            # to-ArtifactPublicationError wrapper are in scope, so a raw OSError
            # here would escape unwrapped, breaking the helper's contract (OaF
            # maps ArtifactPublicationError to prediction_publish_failed but a
            # raw OSError to prediction_artifact_invalid).  Wrap it here so the
            # distinction stays observable regardless of where the failure lands.
            raise ArtifactPublicationError("artifact publication failed") from open_error
        try:
            try:
                _write_all(descriptor, content)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as write_error:
            # The caller's unlink finally only runs once this function returns
            # the temporary name, so a write/fsync failure must clean up the
            # partial file here.  Wrap the failure so callers see the helper's
            # established ArtifactPublicationError contract rather than a raw
            # OSError leaking through the pre-try call site.
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
            raise ArtifactPublicationError("artifact publication failed") from write_error
        return temporary_name
    raise ArtifactPublicationError("artifact publication failed")


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]
