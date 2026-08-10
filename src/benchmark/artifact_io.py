"""Small neutral helpers for immutable benchmark files."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
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
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("artifact is not a regular file")
        chunks: list[bytes] = []
        while True:
            content = os.read(descriptor, 1024 * 1024)
            if not content:
                return b"".join(chunks)
            chunks.append(content)
    finally:
        os.close(descriptor)


def publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact:
    """Create one file without replacement, or reuse matching immutable bytes."""
    if not isinstance(path, Path) or not isinstance(content, bytes):
        raise TypeError("path must be a Path and content must be bytes")
    digest = hashlib.sha256(content).hexdigest()
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactPublicationError("artifact publication failed") from error

    try:
        existing = read_regular_file_no_follow(path)
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise ArtifactPublicationError("artifact publication failed") from error
    if existing is not None:
        if existing != content:
            raise ArtifactPublicationError("artifact already exists with different bytes")
        return PublishedArtifact(path=path, sha256=digest)

    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
        temporary_path = Path(temporary_name)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            try:
                existing = read_regular_file_no_follow(path)
            except OSError as error:
                raise ArtifactPublicationError("artifact publication failed") from error
            if existing != content:
                raise ArtifactPublicationError("artifact already exists with different bytes")
        else:
            _fsync_directory(parent)
        return PublishedArtifact(path=path, sha256=digest)
    except ArtifactPublicationError:
        raise
    except OSError as error:
        raise ArtifactPublicationError("artifact publication failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # The publication itself has already either succeeded or failed;
                # do not hide its stable result behind cleanup noise.
                pass


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
