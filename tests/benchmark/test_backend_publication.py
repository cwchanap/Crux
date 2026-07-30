from __future__ import annotations

# Exact schema/durability fixtures intentionally repeat production-facing field
# shapes, and the failing writer deliberately owns a manually controlled file.
# pylint: disable=duplicate-code,consider-using-with
import os
import stat
from pathlib import Path
from uuid import UUID

import pytest

from src.benchmark import backend_publication
from src.benchmark.backend_identity import sha256_hex
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    atomic_replace_bytes,
    publish_immutable_bytes,
)


def test_immutable_publication_uses_expected_bytes_hash_and_role(tmp_path: Path) -> None:
    content = b"immutable backend artifact\n"
    destination = tmp_path / "nested" / "artifact.json"

    published = publish_immutable_bytes(
        destination,
        content,
        sha256_hex(content),
        role="prediction",
    )

    assert published.role == "prediction"
    assert published.path == destination
    assert published.sha256 == sha256_hex(content)
    assert destination.read_bytes() == content
    assert stat.S_ISREG(destination.lstat().st_mode)


def test_immutable_publication_rejects_wrong_expected_hash_before_visibility(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(destination, b"content", "0" * 64, role="prediction")

    assert not destination.exists()


def test_exact_existing_immutable_content_is_idempotent_and_keeps_inode(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    content = b"same bytes\n"
    destination.write_bytes(content)
    before = destination.stat()

    published = publish_immutable_bytes(
        destination,
        content,
        sha256_hex(content),
        role="report",
    )

    after = destination.stat()
    assert published.sha256 == sha256_hex(content)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_existing_immutable_hash_collision_rejects_different_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"prior valid bytes\n")

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"new bytes\n",
            sha256_hex(b"new bytes\n"),
            role="report",
        )

    assert destination.read_bytes() == b"prior valid bytes\n"


@pytest.mark.parametrize("broken", [False, True])
def test_immutable_publication_rejects_symlink_destination(
    tmp_path: Path,
    broken: bool,
) -> None:
    target = tmp_path / "target.json"
    if not broken:
        target.write_bytes(b"target")
    destination = tmp_path / "artifact.json"
    destination.symlink_to(target if not broken else tmp_path / "missing.json")

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert destination.is_symlink()
    if not broken:
        assert target.read_bytes() == b"target"


def test_immutable_publication_rejects_non_regular_destination_without_blocking(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.fifo"
    os.mkfifo(destination)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert stat.S_ISFIFO(destination.lstat().st_mode)


def test_immutable_temporary_creation_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = UUID("12345678-1234-4678-9234-567812345678")
    temporary = tmp_path / ".artifact.json.12345678123446789234567812345678.tmp"
    temporary.write_bytes(b"attacker bytes")
    monkeypatch.setattr(backend_publication, "uuid4", lambda: fixed)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            tmp_path / "artifact.json",
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert temporary.read_bytes() == b"attacker bytes"
    assert not (tmp_path / "artifact.json").exists()


def test_fresh_immutable_publication_fsyncs_file_then_directory_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    events: list[str] = []
    real_link = os.link
    real_fsync = os.fsync
    real_fsync_directory = backend_publication.fsync_directory

    def tracked_link(source: Path, target: Path) -> None:
        real_link(source, target)
        events.append("linked")

    def tracked_fsync(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if destination.exists():
            destination_stat = destination.stat(follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                events.append("file_fsync")
        real_fsync(descriptor)

    def tracked_fsync_directory(path: Path) -> None:
        if path == tmp_path:
            events.append("directory_fsync")
        real_fsync_directory(path)

    monkeypatch.setattr(backend_publication.os, "link", tracked_link)
    monkeypatch.setattr(backend_publication.os, "fsync", tracked_fsync)
    monkeypatch.setattr(backend_publication, "fsync_directory", tracked_fsync_directory)

    publish_immutable_bytes(
        destination,
        b"content",
        sha256_hex(b"content"),
        role="report",
    )

    linked = events.index("linked")
    winner_fsync = events.index("file_fsync", linked)
    directory_fsync = events.index("directory_fsync", winner_fsync)
    assert linked < winner_fsync < directory_fsync


def test_immutable_publication_rejects_destination_inode_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    real_link = os.link

    def substituted_link(source: Path, target: Path) -> None:
        real_link(source, target)
        destination.unlink()
        destination.write_bytes(b"substituted")

    monkeypatch.setattr(backend_publication.os, "link", substituted_link)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"expected",
            sha256_hex(b"expected"),
            role="report",
        )

    assert destination.read_bytes() == b"substituted"


def test_immutable_publication_propagates_temporary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    real_unlink = Path.unlink

    def failed_temporary_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path.name.startswith(".artifact.json."):
            raise OSError("cleanup failed")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failed_temporary_unlink)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"expected",
            sha256_hex(b"expected"),
            role="report",
        )

    assert destination.read_bytes() == b"expected"


def test_atomic_replace_publishes_exact_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    atomic_replace_bytes(destination, b"new\n")

    assert destination.read_bytes() == b"new\n"
    assert stat.S_ISREG(destination.lstat().st_mode)


@pytest.mark.parametrize("broken", [False, True])
def test_atomic_replace_rejects_symlink_destination(
    tmp_path: Path,
    broken: bool,
) -> None:
    target = tmp_path / "target.json"
    if not broken:
        target.write_bytes(b"target")
    destination = tmp_path / "latest.json"
    destination.symlink_to(target if not broken else tmp_path / "missing.json")

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.is_symlink()
    if not broken:
        assert target.read_bytes() == b"target"


def test_atomic_replace_write_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")
    real_open = Path.open

    class FailingWriter:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def __enter__(self) -> "FailingWriter":
            self.wrapped.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self.wrapped.__exit__(*args)  # type: ignore[attr-defined]

        def write(self, _content: bytes) -> int:
            raise OSError("write failed")

        def fileno(self) -> int:
            return self.wrapped.fileno()  # type: ignore[attr-defined,no-any-return]

        def flush(self) -> None:
            self.wrapped.flush()  # type: ignore[attr-defined]

    def failing_open(path: Path, *args: object, **kwargs: object) -> object:
        mode = args[0] if args else kwargs.pop("mode", "r")
        remaining = args[1:] if args else ()
        opened = real_open(path, mode, *remaining, **kwargs)
        if mode == "xb" and path.name.startswith(".latest.json."):
            return FailingWriter(opened)
        return opened

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"


def test_atomic_replace_fsync_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    def failed_fsync(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(backend_publication.os, "fsync", failed_fsync)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"


def test_atomic_replace_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    def failed_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(backend_publication.os, "replace", failed_replace)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"
