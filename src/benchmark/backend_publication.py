from __future__ import annotations

# Publication converts every low-level failure into one stable error and must also
# catch cleanup failures raised by injected or platform filesystem implementations.
# pylint: disable=broad-exception-caught
import ctypes
import errno
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from src.benchmark.backend_identity import require_sha256, sha256_hex
from src.benchmark.backends import PublishedArtifact

_PUBLICATION_ERROR = "artifact_publication_failed"


class ArtifactPublicationError(OSError):
    pass


@dataclass(frozen=True)
class PublishedDirectory:
    path: Path
    metadata: os.stat_result


class DirectoryPublicationError(ArtifactPublicationError):
    def __init__(self, publication: PublishedDirectory) -> None:
        super().__init__("atomic directory publication failed after rename")
        self.publication = publication


class PrivateSnapshotIntegrityError(OSError):
    pass


class _AncestorBindingError(OSError):
    pass


def rename_directory_no_replace(source: Path, destination: Path) -> PublishedDirectory:
    """Atomically publish a directory only when destination is absent."""

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.parent != destination_path.parent:
        raise ArtifactPublicationError("publication rename must stay within one parent directory")
    publication: PublishedDirectory | None = None
    source_descriptor: int | None = None
    try:
        with _anchored_parent(source_path, create=False) as parent:
            parent.verify()
            source_descriptor = os.open(
                parent.name,
                _directory_open_flags(),
                dir_fd=parent.parent_descriptor,
            )
            source_metadata = os.fstat(source_descriptor)
            _verify_directory_binding_at(parent, source_metadata)
            _rename_no_replace_syscall(
                parent.name,
                destination_path.name,
                src_dir_fd=parent.parent_descriptor,
                dst_dir_fd=parent.parent_descriptor,
            )
            publication = PublishedDirectory(destination_path, source_metadata)
            published_metadata = os.stat(
                destination_path.name,
                dir_fd=parent.parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(published_metadata.st_mode):
                raise OSError("publication destination is not a directory")
            if not _same_inode(source_metadata, published_metadata):
                raise OSError("publication destination binding changed")
            parent.verify()
            os.fsync(parent.parent_descriptor)
            parent.verify()
            return publication
    except FileExistsError as error:
        raise ArtifactPublicationError("publication destination already exists") from error
    except OSError as error:
        if publication is not None:
            raise DirectoryPublicationError(publication) from error
        raise ArtifactPublicationError("atomic no-replace directory publication failed") from error
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)


def rollback_published_directory(publication: PublishedDirectory) -> None:
    """Remove one inode-bound directory and synchronize its anchored parent."""

    if not isinstance(publication, PublishedDirectory):
        raise TypeError("publication must be a PublishedDirectory")
    directory_descriptor: int | None = None
    try:
        with _anchored_parent(publication.path, create=False) as parent:
            parent.verify()
            directory_descriptor = os.open(
                parent.name,
                _directory_open_flags(),
                dir_fd=parent.parent_descriptor,
            )
            directory_metadata = os.fstat(directory_descriptor)
            _verify_directory_binding_at(parent, directory_metadata)
            if not _same_inode(directory_metadata, publication.metadata):
                raise OSError("published directory binding changed")
            for entry in os.listdir(directory_descriptor):
                entry_metadata = os.stat(
                    entry,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(entry_metadata.st_mode):
                    raise OSError("published directory contains an unsafe entry")
                os.unlink(entry, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
            parent.verify()
            _verify_directory_binding_at(parent, directory_metadata)
            os.rmdir(parent.name, dir_fd=parent.parent_descriptor)
            os.fsync(parent.parent_descriptor)
            parent.verify()
    except OSError as error:
        raise ArtifactPublicationError("published directory rollback failed") from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


@dataclass(frozen=True)
class DirectoryAnchor:
    path: Path
    descriptor: int
    metadata: os.stat_result

    def verify(self) -> None:
        current = os.fstat(self.descriptor)
        if not stat.S_ISDIR(current.st_mode) or not _same_inode(current, self.metadata):
            raise _AncestorBindingError("artifact root descriptor changed")

    def relative_path(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else self.path / path
        try:
            relative = candidate.relative_to(self.path)
        except ValueError:
            raise OSError("artifact path escapes anchored root") from None
        if not relative.parts:
            raise OSError("artifact path must not equal anchored root")
        return relative


@dataclass(frozen=True)
class RegularFileAnchor:
    path: Path
    descriptor: int
    metadata: os.stat_result
    content: bytes

    def verify(self) -> None:
        current = os.fstat(self.descriptor)
        if not stat.S_ISREG(current.st_mode) or not _same_inode(current, self.metadata):
            raise _AncestorBindingError("artifact file descriptor changed")


@dataclass(frozen=True)
class PrivateFileSnapshot:  # pylint: disable=too-many-instance-attributes
    path: Path
    parent_descriptor: int
    parent_metadata: os.stat_result
    directory_descriptor: int
    directory_metadata: os.stat_result
    directory_name: str
    descriptor: int
    metadata: os.stat_result
    expected_sha256: str
    byte_length: int

    def verify(self) -> None:
        parent = os.fstat(self.parent_descriptor)
        if not stat.S_ISDIR(parent.st_mode) or not _same_inode(parent, self.parent_metadata):
            raise PrivateSnapshotIntegrityError("private snapshot parent changed")
        parent_path = os.stat(self.path.parent.parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_path.st_mode) or not _same_inode(
            parent_path,
            self.parent_metadata,
        ):
            raise PrivateSnapshotIntegrityError("private snapshot parent binding changed")

        directory = os.fstat(self.directory_descriptor)
        if (
            not stat.S_ISDIR(directory.st_mode)
            or not _same_inode(directory, self.directory_metadata)
            or stat.S_IMODE(directory.st_mode) != 0o500
        ):
            raise PrivateSnapshotIntegrityError("private snapshot directory changed")
        directory_path = os.stat(
            self.directory_name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(directory_path.st_mode) or not _same_inode(
            directory_path,
            self.directory_metadata,
        ):
            raise PrivateSnapshotIntegrityError("private snapshot directory binding changed")

        current = os.fstat(self.descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or not _same_inode(current, self.metadata)
            or stat.S_IMODE(current.st_mode) != 0o400
            or current.st_size != self.byte_length
        ):
            raise PrivateSnapshotIntegrityError("private snapshot file changed")
        path_metadata = os.stat(
            self.path.name,
            dir_fd=self.directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_metadata.st_mode) or not _same_inode(
            path_metadata,
            self.metadata,
        ):
            raise PrivateSnapshotIntegrityError("private snapshot file binding changed")
        if sha256_hex(_read_descriptor(self.descriptor)) != self.expected_sha256:
            raise PrivateSnapshotIntegrityError("private snapshot content changed")


@dataclass
class _AnchoredParent:
    descriptors: list[int]
    bindings: list[tuple[int, str, os.stat_result]]
    parent_descriptor: int
    name: str

    def verify(self) -> None:
        for parent_descriptor, component, expected in self.bindings:
            current = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(current.st_mode) or not _same_inode(current, expected):
                raise _AncestorBindingError("artifact ancestor binding changed")


# The immutable publish transaction keeps cleanup and durability state explicit.
# pylint: disable-next=too-many-branches,too-many-statements
def publish_immutable_bytes(
    path: Path,
    content: bytes,
    expected_sha256: str,
    *,
    role: str,
    anchor: DirectoryAnchor | None = None,
) -> PublishedArtifact:
    temporary_name: str | None = None
    temporary_descriptor: int | None = None
    temporary_stat: os.stat_result | None = None
    parent: _AnchoredParent | None = None
    linked = False
    ancestor_failed = False
    failed = False
    cleanup_failed = False
    try:
        _validate_publication(path, content, expected_sha256, role)
        with _anchored_parent(path, create=True, anchor=anchor) as anchored:
            parent = anchored
            try:
                parent.verify()
                temporary_name = f".{path.name}.{uuid4().hex}.tmp"
                temporary_descriptor = _create_temporary(anchored, temporary_name)
                temporary_stat = _require_regular_descriptor(temporary_descriptor)
                _write_all(temporary_descriptor, content)
                os.fsync(temporary_descriptor)
                anchored.verify()
                try:
                    os.link(
                        temporary_name,
                        anchored.name,
                        src_dir_fd=anchored.parent_descriptor,
                        dst_dir_fd=anchored.parent_descriptor,
                        follow_symlinks=False,
                    )
                    linked = True
                except FileExistsError:
                    linked = False
                anchored.verify()
                _unlink_at(anchored.parent_descriptor, temporary_name)
                temporary_name = None
                _verify_existing_file(
                    anchored,
                    content,
                    expected_stat=temporary_stat if linked else None,
                )
            except _AncestorBindingError:
                ancestor_failed = True
                failed = True
            except Exception:
                failed = True
            finally:
                if temporary_descriptor is not None:
                    try:
                        os.close(temporary_descriptor)
                    except Exception:
                        cleanup_failed = True
                if temporary_name is not None and temporary_descriptor is not None:
                    try:
                        _unlink_at(parent.parent_descriptor, temporary_name)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        cleanup_failed = True
                if ancestor_failed and linked and temporary_stat is not None:
                    try:
                        _unlink_if_same_inode(
                            parent.parent_descriptor,
                            parent.name,
                            temporary_stat,
                        )
                    except Exception:
                        cleanup_failed = True
    except Exception:
        failed = True
    if failed or cleanup_failed:
        raise ArtifactPublicationError(_PUBLICATION_ERROR) from None
    return PublishedArtifact(role=role, path=path, sha256=expected_sha256)


def atomic_replace_bytes(
    path: Path,
    content: bytes,
    *,
    anchor: DirectoryAnchor | None = None,
) -> None:
    temporary_name: str | None = None
    temporary_descriptor: int | None = None
    parent: _AnchoredParent | None = None
    failed = False
    cleanup_failed = False
    try:
        if not isinstance(path, Path) or not isinstance(content, bytes):
            raise TypeError("invalid artifact replacement")
        with _anchored_parent(path, create=True, anchor=anchor) as anchored:
            parent = anchored
            try:
                anchored.verify()
                _require_absent_or_regular(anchored)
                temporary_name = f".{path.name}.{uuid4().hex}.tmp"
                temporary_descriptor = _create_temporary(anchored, temporary_name)
                temporary_stat = _require_regular_descriptor(temporary_descriptor)
                _write_all(temporary_descriptor, content)
                os.fsync(temporary_descriptor)
                anchored.verify()
                _require_absent_or_regular(anchored)
                os.replace(
                    temporary_name,
                    anchored.name,
                    src_dir_fd=anchored.parent_descriptor,
                    dst_dir_fd=anchored.parent_descriptor,
                )
                temporary_name = None
                anchored.verify()
                _verify_file_binding_at(anchored, temporary_stat)
                os.fsync(temporary_descriptor)
                anchored.verify()
                os.fsync(anchored.parent_descriptor)
                anchored.verify()
            except Exception:
                failed = True
            finally:
                if temporary_descriptor is not None:
                    try:
                        os.close(temporary_descriptor)
                    except Exception:
                        cleanup_failed = True
                if temporary_name is not None and temporary_descriptor is not None:
                    try:
                        _unlink_at(parent.parent_descriptor, temporary_name)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        cleanup_failed = True
    except Exception:
        failed = True
    if failed or cleanup_failed:
        raise ArtifactPublicationError(_PUBLICATION_ERROR) from None


def read_regular_file_no_follow(
    path: Path,
    *,
    anchor: DirectoryAnchor | None = None,
    max_bytes: int | None = None,
) -> bytes:
    if max_bytes is not None and (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)):
        raise TypeError("max_bytes must be an integer or None")
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be nonnegative")
    descriptor: int | None = None
    with _anchored_parent(path, create=False, anchor=anchor) as parent:
        try:
            parent.verify()
            descriptor = os.open(
                parent.name,
                _regular_file_open_flags(),
                dir_fd=parent.parent_descriptor,
            )
            descriptor_stat = _require_regular_descriptor(descriptor)
            if max_bytes is not None and descriptor_stat.st_size > max_bytes:
                raise OSError("artifact exceeds bounded read size")
            content = _read_descriptor(descriptor, max_bytes=max_bytes)
            parent.verify()
            _verify_file_binding_at(parent, descriptor_stat)
            return content
        finally:
            if descriptor is not None:
                os.close(descriptor)


@contextmanager
def open_lock_file_no_follow(
    path: Path,
    *,
    anchor: DirectoryAnchor | None = None,
) -> Iterator[int]:
    descriptor: int | None = None
    with _anchored_parent(path, create=True, anchor=anchor) as parent:
        try:
            no_follow = getattr(os, "O_NOFOLLOW", None)
            if no_follow is None:
                raise OSError("no-follow lock files are unavailable")
            descriptor = os.open(
                parent.name,
                os.O_RDWR | os.O_CREAT | no_follow | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent.parent_descriptor,
            )
            descriptor_stat = _require_regular_descriptor(descriptor)
            parent.verify()
            _verify_file_binding_at(parent, descriptor_stat)
            yield descriptor
            parent.verify()
            _verify_file_binding_at(parent, descriptor_stat)
        finally:
            if descriptor is not None:
                os.close(descriptor)


def unlink_regular_file_no_follow(
    path: Path,
    *,
    anchor: DirectoryAnchor | None = None,
) -> None:
    descriptor: int | None = None
    with _anchored_parent(path, create=False, anchor=anchor) as parent:
        try:
            descriptor = os.open(
                parent.name,
                _regular_file_open_flags(),
                dir_fd=parent.parent_descriptor,
            )
            descriptor_stat = _require_regular_descriptor(descriptor)
            parent.verify()
            _verify_file_binding_at(parent, descriptor_stat)
            _unlink_at(parent.parent_descriptor, parent.name)
            os.fsync(parent.parent_descriptor)
            parent.verify()
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


@contextmanager
def open_directory_anchor(path: Path) -> Iterator[DirectoryAnchor]:
    descriptor: int | None = None
    try:
        if not isinstance(path, Path) or not path.is_absolute():
            raise OSError("directory anchor path must be absolute")
        descriptor = os.open(path, _directory_open_flags())
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("directory anchor must be a directory")
        path_metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(path_metadata.st_mode) or not _same_inode(metadata, path_metadata):
            raise OSError("directory anchor path binding changed")
        anchor = DirectoryAnchor(path=path, descriptor=descriptor, metadata=metadata)
        anchor.verify()
        yield anchor
        anchor.verify()
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def open_regular_file_anchor(
    path: Path,
    *,
    anchor: DirectoryAnchor | None = None,
) -> Iterator[RegularFileAnchor]:
    descriptor: int | None = None
    try:
        with _anchored_parent(path, create=False, anchor=anchor) as parent:
            parent.verify()
            descriptor = os.open(
                parent.name,
                _regular_file_open_flags(),
                dir_fd=parent.parent_descriptor,
            )
            metadata = _require_regular_descriptor(descriptor)
            content = _read_descriptor(descriptor)
            parent.verify()
            _verify_file_binding_at(parent, metadata)
            file_anchor = RegularFileAnchor(
                path=path,
                descriptor=descriptor,
                metadata=metadata,
                content=content,
            )
            file_anchor.verify()
            yield file_anchor
            file_anchor.verify()
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
# Snapshot creation and cleanup keep every descriptor and path mutation explicit.
# pylint: disable-next=too-many-branches,too-many-locals,too-many-statements
def open_private_file_snapshot(
    content: bytes,
    expected_sha256: str,
    *,
    root: Path,
) -> Iterator[PrivateFileSnapshot]:
    parent_path: Path | None = None
    parent_descriptor: int | None = None
    parent_metadata: os.stat_result | None = None
    directory_path: Path | None = None
    directory_name: str | None = None
    directory_descriptor: int | None = None
    directory_metadata: os.stat_result | None = None
    descriptor: int | None = None
    integrity_failed = False
    cleanup_failed = False
    try:
        if not isinstance(content, bytes):
            raise TypeError("private snapshot content must be bytes")
        require_sha256(expected_sha256, "expected_sha256")
        if sha256_hex(content) != expected_sha256:
            raise ValueError("private snapshot content hash mismatch")
        if not isinstance(root, Path) or not root.is_absolute():
            raise OSError("private snapshot root must be an absolute Path")

        parent_path = root
        parent_descriptor = os.open(parent_path, _directory_open_flags())
        parent_metadata = os.fstat(parent_descriptor)
        parent_path_metadata = os.stat(parent_path, follow_symlinks=False)
        if not stat.S_ISDIR(parent_path_metadata.st_mode) or not _same_inode(
            parent_path_metadata,
            parent_metadata,
        ):
            raise PrivateSnapshotIntegrityError("private snapshot parent binding changed")

        directory_name = f"crux-backend-input-{uuid4().hex}"
        os.mkdir(directory_name, mode=0o700, dir_fd=parent_descriptor)
        directory_path = parent_path / directory_name
        directory_metadata = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise PrivateSnapshotIntegrityError("private snapshot directory is invalid")
        directory_descriptor = os.open(
            directory_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        if not _same_inode(os.fstat(directory_descriptor), directory_metadata):
            raise PrivateSnapshotIntegrityError("private snapshot directory binding changed")

        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise OSError("no-follow private snapshots are unavailable")
        descriptor = os.open(
            "input.wav",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        _require_regular_descriptor(descriptor)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.fchmod(directory_descriptor, 0o500)
        os.fsync(directory_descriptor)

        directory_metadata = os.fstat(directory_descriptor)
        metadata = os.fstat(descriptor)
        snapshot = PrivateFileSnapshot(
            path=directory_path / "input.wav",
            parent_descriptor=parent_descriptor,
            parent_metadata=parent_metadata,
            directory_descriptor=directory_descriptor,
            directory_metadata=directory_metadata,
            directory_name=directory_name,
            descriptor=descriptor,
            metadata=metadata,
            expected_sha256=expected_sha256,
            byte_length=len(content),
        )
        snapshot.verify()
        yield snapshot
    finally:
        if directory_descriptor is not None:
            try:
                os.fchmod(directory_descriptor, 0o700)
            except OSError:
                cleanup_failed = True
            try:
                os.unlink("input.wav", dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_failed = True
        if (
            parent_descriptor is not None
            and directory_name is not None
            and directory_metadata is not None
        ):
            try:
                directory_cleanup_failed, directory_integrity_failed = (
                    _cleanup_private_snapshot_directories(
                        parent_descriptor,
                        directory_name,
                        directory_metadata,
                    )
                )
                cleanup_failed = cleanup_failed or directory_cleanup_failed
                integrity_failed = integrity_failed or directory_integrity_failed
            except Exception:
                cleanup_failed = True
                integrity_failed = True
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                cleanup_failed = True
        if parent_descriptor is not None:
            try:
                parent_current = os.fstat(parent_descriptor)
                if parent_metadata is None or not _same_inode(
                    parent_current,
                    parent_metadata,
                ):
                    integrity_failed = True
            except OSError:
                cleanup_failed = True
            try:
                os.close(parent_descriptor)
            except OSError:
                cleanup_failed = True
        if directory_path is not None and parent_path is not None:
            try:
                parent_path_current = os.stat(parent_path, follow_symlinks=False)
                if parent_metadata is None or not _same_inode(
                    parent_path_current,
                    parent_metadata,
                ):
                    integrity_failed = True
            except OSError:
                integrity_failed = True
        if cleanup_failed or integrity_failed:
            raise PrivateSnapshotIntegrityError("private snapshot integrity failed") from None


def resolve_private_snapshot_root() -> Path:
    try:
        configured_root = Path(tempfile.gettempdir())
        resolved_root = configured_root.resolve(strict=True)
        metadata = resolved_root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("private snapshot root is not a directory")
        return resolved_root
    except (OSError, RuntimeError, TypeError, ValueError):
        raise OSError("private snapshot root is unavailable") from None


def _cleanup_private_snapshot_directories(
    parent_descriptor: int,
    directory_name: str,
    directory_metadata: os.stat_result,
) -> tuple[bool, bool]:
    cleanup_failed = False
    integrity_failed = False
    trusted_name: str | None = None
    current = _optional_stat_at(parent_descriptor, directory_name)
    if (
        current is not None
        and stat.S_ISDIR(current.st_mode)
        and _same_inode(
            current,
            directory_metadata,
        )
    ):
        trusted_name = directory_name
    else:
        integrity_failed = True
        trusted_name = _find_directory_inode_name(parent_descriptor, directory_metadata)

    if trusted_name is not None:
        try:
            os.rmdir(trusted_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            integrity_failed = True
        except OSError:
            cleanup_failed = True
    else:
        cleanup_failed = True

    replacement = _optional_stat_at(parent_descriptor, directory_name)
    if (
        integrity_failed
        and replacement is not None
        and stat.S_ISDIR(replacement.st_mode)
        and not _same_inode(replacement, directory_metadata)
    ):
        try:
            os.rmdir(directory_name, dir_fd=parent_descriptor)
        except OSError:
            # A nonempty or otherwise untrusted replacement is not ours to delete.
            pass
    return cleanup_failed, integrity_failed


def _find_directory_inode_name(
    parent_descriptor: int,
    directory_metadata: os.stat_result,
) -> str | None:
    matches: list[str] = []
    for name in os.listdir(parent_descriptor):
        current = _optional_stat_at(parent_descriptor, name)
        if (
            current is not None
            and stat.S_ISDIR(current.st_mode)
            and _same_inode(current, directory_metadata)
        ):
            matches.append(name)
    if len(matches) != 1:
        return None
    return matches[0]


def _optional_stat_at(
    parent_descriptor: int,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


@contextmanager
# Every parent transition is checked before and after directory creation.
# pylint: disable-next=too-many-branches,too-many-statements
def _anchored_parent(
    path: Path,
    *,
    create: bool,
    anchor: DirectoryAnchor | None = None,
) -> Iterator[_AnchoredParent]:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a Path")
    root_path: str | None = None
    if anchor is not None:
        if not isinstance(anchor, DirectoryAnchor):
            raise TypeError("artifact anchor must be a DirectoryAnchor")
        anchor.verify()
        components = anchor.relative_path(path).parts
    else:
        parts = path.parts
        if path.is_absolute():
            root_path = path.anchor
            components = parts[1:]
        else:
            root_path = "."
            components = parts
    if (
        not components
        or components[-1] in {"", ".", ".."}
        or any(component in {"", ".", ".."} for component in components[:-1])
    ):
        raise OSError("artifact path is invalid")

    root_descriptor = _open_root_descriptor(anchor, root_path)
    descriptors: list[int] = []
    bindings: list[tuple[int, str, os.stat_result]] = []
    try:
        current = root_descriptor
        descriptors.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise OSError("artifact root is not a directory")
        for component in components[:-1]:
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current,
                )
            except FileNotFoundError:
                if not create:
                    raise
                _verify_bindings(bindings)
                try:
                    os.mkdir(component, mode=0o777, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current,
                )
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise OSError("artifact parent is not a directory")
            descriptors.append(child)
            bindings.append((current, component, child_stat))
            current = child
        anchored = _AnchoredParent(
            descriptors=descriptors,
            bindings=bindings,
            parent_descriptor=current,
            name=components[-1],
        )
        anchored.verify()
        yield anchored
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_root_descriptor(
    anchor: DirectoryAnchor | None,
    root_path: str | None,
) -> int:
    if anchor is not None:
        return os.dup(anchor.descriptor)
    if root_path is None:
        raise OSError("artifact root is unavailable")
    return os.open(root_path, _directory_open_flags())


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("anchored directory descriptors are unavailable")
    return os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)


def _create_temporary(parent: _AnchoredParent, name: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("no-follow temporary creation is unavailable")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0)
    return os.open(name, flags, 0o666, dir_fd=parent.parent_descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("artifact write was incomplete")
        offset += written


def _verify_existing_file(
    parent: _AnchoredParent,
    expected_content: bytes,
    *,
    expected_stat: os.stat_result | None,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            parent.name,
            _regular_file_open_flags(),
            dir_fd=parent.parent_descriptor,
        )
        descriptor_stat = _require_regular_descriptor(descriptor)
        if expected_stat is not None and not _same_inode(descriptor_stat, expected_stat):
            raise OSError("artifact destination inode changed")
        actual_content = _read_descriptor(descriptor)
        parent.verify()
        _verify_file_binding_at(parent, descriptor_stat)
        if actual_content != expected_content:
            raise OSError("artifact destination content differs")
        os.fsync(descriptor)
        parent.verify()
        _verify_file_binding_at(parent, descriptor_stat)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    os.fsync(parent.parent_descriptor)
    parent.verify()


def _require_absent_or_regular(parent: _AnchoredParent) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            parent.name,
            _regular_file_open_flags(),
            dir_fd=parent.parent_descriptor,
        )
    except FileNotFoundError:
        return
    try:
        descriptor_stat = _require_regular_descriptor(descriptor)
        parent.verify()
        _verify_file_binding_at(parent, descriptor_stat)
    finally:
        os.close(descriptor)


def _regular_file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_block is None:
        raise OSError("no-follow regular-file descriptors are unavailable")
    return os.O_RDONLY | no_follow | non_block | getattr(os, "O_CLOEXEC", 0)


def _require_regular_descriptor(descriptor: int) -> os.stat_result:
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise OSError("artifact is not a regular file")
    return descriptor_stat


def _verify_file_binding_at(
    parent: _AnchoredParent,
    descriptor_stat: os.stat_result,
) -> None:
    path_stat = os.stat(
        parent.name,
        dir_fd=parent.parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(path_stat.st_mode) or not _same_inode(path_stat, descriptor_stat):
        raise OSError("artifact path binding changed")


def _verify_directory_binding_at(
    parent: _AnchoredParent,
    descriptor_stat: os.stat_result,
) -> None:
    path_stat = os.stat(
        parent.name,
        dir_fd=parent.parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(path_stat.st_mode) or not _same_inode(path_stat, descriptor_stat):
        raise OSError("directory path binding changed")


def _verify_bindings(bindings: list[tuple[int, str, os.stat_result]]) -> None:
    for parent_descriptor, component, expected in bindings:
        current = os.stat(
            component,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(current.st_mode) or not _same_inode(current, expected):
            raise _AncestorBindingError("artifact ancestor binding changed")


def _unlink_at(parent_descriptor: int, name: str) -> None:
    os.unlink(name, dir_fd=parent_descriptor)


def _unlink_if_same_inode(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISREG(current.st_mode) and _same_inode(current, expected):
        _unlink_at(parent_descriptor, name)


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _rename_no_replace_syscall(
    source: str,
    destination: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
) -> None:
    """Rename sibling paths with the platform no-replace syscall."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError:
            raise OSError(errno.ENOTSUP, "renameatx_np is unavailable") from None
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            src_dir_fd,
            encoded_source,
            dst_dir_fd,
            encoded_destination,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable") from None
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            src_dir_fd,
            encoded_source,
            dst_dir_fd,
            encoded_destination,
            0x00000001,  # RENAME_NOREPLACE
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _read_descriptor(descriptor: int, *, max_bytes: int | None = None) -> bytes:
    chunks: list[bytes] = []
    total = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        read_size = 1024 * 1024
        if max_bytes is not None:
            read_size = min(read_size, max_bytes - total + 1)
        chunk = os.read(descriptor, read_size)
        if not chunk:
            break
        total += len(chunk)
        if max_bytes is not None and total > max_bytes:
            raise OSError("artifact exceeds bounded read size")
        chunks.append(chunk)
    return b"".join(chunks)
